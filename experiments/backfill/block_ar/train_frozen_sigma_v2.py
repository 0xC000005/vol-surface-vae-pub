"""
Phase-2 frozen-encoder sigma head training — v2.

Key change from v1: train on vol_scale (deterministic, computed from history)
instead of empirical future sigma (noisy, unpredictable per-sample).

Multiple training modes:
1. "vol_scale": MSE against hand-coded vol_scale (deterministic target, easiest)
2. "h1_change": MSE against |h=1 incremental change| (per-sample but condition-dependent)
3. "contrastive": Ranking loss — sigma should be higher for higher vol_of_vol
4. "batch_calibration": Match within-batch Q5/Q1 ratio to GT cross-sample ratio
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


class SigmaHead(nn.Module):
    """MLP sigma head: encoder features → positive scalar sigma."""

    def __init__(self, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Returns positive sigma. (B, 1)"""
        return F.softplus(self.net(condition)) + 0.01


def compute_vol_scale(history, global_mean_vol=0.0187):
    """Compute hand-coded vol_scale from history."""
    past_abs = denormalize_iv(history)
    mean_iv = past_abs.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vol = daily_chg.std(dim=1, keepdim=True)
    return (vol / global_mean_vol).clamp(0.5, 2.0)  # (B, 1)


def compute_vol_of_vol(history):
    """Compute vol_of_vol from history."""
    past_abs = denormalize_iv(history)
    mean_iv = past_abs.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    return daily_chg.std(dim=1)  # (B,)


def train_vol_scale(model, sigma_head, dataloader, optimizer, device, gmv=0.0187):
    """Train sigma head to predict vol_scale from frozen encoder features."""
    sigma_head.train()
    total_loss = 0.0
    total_corr = 0.0
    n = 0

    for batch in tqdm(dataloader, desc="Training (vol_scale)", leave=False):
        history = batch["history"].to(device)
        B = history.shape[0]

        with torch.no_grad():
            cond = model.encoder(history, mask=None)
            if model.config.forward_only:
                cond = cond + model.encoder.null_embedding.expand(B, -1)

        target = compute_vol_scale(history, gmv).to(device)  # (B, 1)
        pred = sigma_head(cond)  # (B, 1)

        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            c = torch.corrcoef(torch.stack([pred.squeeze(), target.squeeze()]))[0, 1].item()
        total_loss += loss.item()
        total_corr += c if not np.isnan(c) else 0
        n += 1

    return {"loss": total_loss / n, "correlation": total_corr / n}


def train_contrastive(model, sigma_head, dataloader, optimizer, device, margin=0.1):
    """Contrastive ranking: sigma should be higher for higher vol_of_vol."""
    sigma_head.train()
    total_loss = 0.0
    total_correct = 0.0
    total_pairs = 0
    n = 0

    for batch in tqdm(dataloader, desc="Training (contrastive)", leave=False):
        history = batch["history"].to(device)
        B = history.shape[0]

        with torch.no_grad():
            cond = model.encoder(history, mask=None)
            if model.config.forward_only:
                cond = cond + model.encoder.null_embedding.expand(B, -1)

        vov = compute_vol_of_vol(history).to(device)  # (B,)
        pred = sigma_head(cond).squeeze()  # (B,)

        # Random pairs: (i, j) where vov_i > vov_j → sigma_i should > sigma_j
        idx = torch.randperm(B, device=device)
        i1, i2 = idx[:B//2], idx[B//2:B//2*2]
        vov1, vov2 = vov[i1], vov[i2]
        pred1, pred2 = pred[i1], pred[i2]

        # Ensure vov1 > vov2 (swap if needed)
        swap = vov1 < vov2
        vov_high = torch.where(swap, vov2, vov1)
        vov_low = torch.where(swap, vov1, vov2)
        pred_high = torch.where(swap, pred2, pred1)
        pred_low = torch.where(swap, pred1, pred2)

        # Margin ranking loss: penalize if pred_high < pred_low + margin
        loss = F.margin_ranking_loss(
            pred_high, pred_low,
            torch.ones_like(pred_high),
            margin=margin,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct = (pred_high > pred_low).float().mean().item()

        total_loss += loss.item()
        total_correct += correct
        n += 1

    return {"loss": total_loss / n, "ranking_accuracy": total_correct / n}


def train_batch_calibration(model, sigma_head, dataloader, optimizer, device, gmv=0.0187):
    """Match within-batch Q5/Q1 sigma ratio to GT cross-sample spread ratio."""
    sigma_head.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(dataloader, desc="Training (batch_cal)", leave=False):
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        B = history.shape[0]
        if B < 16:
            continue

        with torch.no_grad():
            cond = model.encoder(history, mask=None)
            if model.config.forward_only:
                cond = cond + model.encoder.null_embedding.expand(B, -1)

        vov = compute_vol_of_vol(history)  # (B,)

        # Split by median vol_of_vol
        median_vov = vov.median()
        high_mask = vov >= median_vov
        low_mask = ~high_mask

        if high_mask.sum() < 4 or low_mask.sum() < 4:
            continue

        # Compute GT cross-sample std of h=1 changes per group
        baseline = denormalize_iv(history[:, -1]).clamp(min=0.01)
        fut_h1 = denormalize_iv(future[:, 0])
        h1_change = (fut_h1 - baseline).abs().mean(dim=(-1, -2))  # (B,)

        gt_high_std = h1_change[high_mask].std()
        gt_low_std = h1_change[low_mask].std()
        gt_ratio = (gt_high_std / gt_low_std.clamp(min=1e-6)).clamp(0.5, 4.0)

        # Predict sigma
        pred = sigma_head(cond.to(device)).squeeze()  # (B,)

        # Match ratio of predicted sigmas to GT ratio
        pred_high = pred[high_mask.to(device)].mean()
        pred_low = pred[low_mask.to(device)].mean()
        pred_ratio = pred_high / pred_low.clamp(min=1e-6)

        loss = (pred_ratio - gt_ratio).pow(2)

        # Also add a regularization: predicted sigma should be close to vol_scale
        vol_scale = compute_vol_scale(history, gmv).to(device).squeeze()
        reg_loss = F.mse_loss(pred, vol_scale) * 0.1

        total_loss_val = loss + reg_loss

        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        total_loss += total_loss_val.item()
        n += 1

    return {"loss": total_loss / max(n, 1)}


def validate(model, sigma_head, dataloader, device, gmv=0.0187):
    """Validate: compute correlations and Q5/Q1 of predicted sigma."""
    sigma_head.eval()
    model.eval()

    all_pred = []
    all_vov = []
    all_vs = []
    all_h1_chg = []

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            B = history.shape[0]

            cond = model.encoder(history, mask=None)
            if model.config.forward_only:
                cond = cond + model.encoder.null_embedding.expand(B, -1)

            pred = sigma_head(cond).squeeze().cpu()
            all_pred.append(pred)

            vov = compute_vol_of_vol(history).cpu()
            all_vov.append(vov)

            vs = compute_vol_scale(history, gmv).squeeze().cpu()
            all_vs.append(vs)

            baseline = denormalize_iv(history[:, -1]).clamp(min=0.01)
            fut_h1 = denormalize_iv(future[:, 0])
            h1_chg = (fut_h1 - baseline).abs().mean(dim=(-1, -2)).cpu()
            all_h1_chg.append(h1_chg)

    pred = torch.cat(all_pred).numpy()
    vov = torch.cat(all_vov).numpy()
    vs = torch.cat(all_vs).numpy()
    h1 = torch.cat(all_h1_chg).numpy()

    # Correlations
    corr_pred_vov = stats.spearmanr(pred, vov)[0]
    corr_pred_vs = stats.spearmanr(pred, vs)[0]
    corr_pred_h1 = stats.spearmanr(pred, h1)[0]

    # Q5/Q1 of predicted sigma by vol_of_vol quintile
    q = np.quantile(vov, [0.2, 0.8])
    q1_mask = vov <= q[0]
    q5_mask = vov >= q[1]
    q5q1_pred = pred[q5_mask].mean() / pred[q1_mask].mean() if pred[q1_mask].mean() > 0 else float('nan')
    q5q1_vs = vs[q5_mask].mean() / vs[q1_mask].mean()

    cov_pred = pred.std() / pred.mean() if pred.mean() > 0 else 0

    return {
        "corr_pred_vov": corr_pred_vov,
        "corr_pred_vs": corr_pred_vs,
        "corr_pred_h1": corr_pred_h1,
        "q5q1_pred": q5q1_pred,
        "q5q1_vs": q5q1_vs,
        "cov_pred": cov_pred,
        "pred_mean": pred.mean(),
        "pred_std": pred.std(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="vol_scale",
                        choices=["vol_scale", "contrastive", "batch_calibration"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Load model
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cd = cp["config"]
    if dataclasses.is_dataclass(cd):
        cd = dataclasses.asdict(cd)
    mc = BlockARConfig(**{k: v for k, v in cd.items()
                          if k in {f.name for f in dataclasses.fields(BlockARConfig)}})
    model = ConditionalBlockARDDPM(mc)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Frozen model: {sum(p.numel() for p in model.parameters()):,} params")

    # Sigma head
    sh = SigmaHead(mc.bottleneck_dim, args.hidden_dim).to(args.device)
    print(f"Sigma head: {sum(p.numel() for p in sh.parameters()):,} params")

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    train_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    val_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4040, end_idx=4540)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    opt = torch.optim.AdamW(sh.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr/10)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_q5q1 = 0.0

    print(f"\nTraining sigma head (mode={args.mode}, {args.epochs} epochs)")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        if args.mode == "vol_scale":
            tm = train_vol_scale(model, sh, train_dl, opt, args.device, mc.global_mean_vol)
        elif args.mode == "contrastive":
            tm = train_contrastive(model, sh, train_dl, opt, args.device)
        elif args.mode == "batch_calibration":
            tm = train_batch_calibration(model, sh, train_dl, opt, args.device, mc.global_mean_vol)
        sched.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            vm = validate(model, sh, val_dl, args.device, mc.global_mean_vol)
            extra = ""
            if "ranking_accuracy" in tm:
                extra = f" | Rank acc: {tm['ranking_accuracy']:.3f}"
            if "correlation" in tm:
                extra = f" | Train corr: {tm['correlation']:.3f}"
            print(
                f"Epoch {epoch:3d}/{args.epochs} | Loss: {tm['loss']:.6f}{extra} | "
                f"Val corr(vov): {vm['corr_pred_vov']:.3f} | "
                f"Val corr(vs): {vm['corr_pred_vs']:.3f} | "
                f"Val corr(h1): {vm['corr_pred_h1']:.3f} | "
                f"Q5/Q1 pred: {vm['q5q1_pred']:.3f} (vs: {vm['q5q1_vs']:.3f}) | "
                f"CoV: {vm['cov_pred']:.3f}"
            )

            if vm["q5q1_pred"] > best_q5q1:
                best_q5q1 = vm["q5q1_pred"]
                torch.save({
                    "sigma_head_state_dict": sh.state_dict(),
                    "epoch": epoch,
                    "val_metrics": vm,
                    "config": {"hidden_dim": args.hidden_dim, "mode": args.mode,
                               "base_checkpoint": args.checkpoint, "cond_dim": mc.bottleneck_dim},
                }, f"{args.output_dir}/best_sigma_head.pt")
        else:
            extra = ""
            if "ranking_accuracy" in tm:
                extra = f" | Rank acc: {tm['ranking_accuracy']:.3f}"
            if "correlation" in tm:
                extra = f" | Train corr: {tm['correlation']:.3f}"
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {tm['loss']:.6f}{extra}")

    print(f"\nBest val Q5/Q1: {best_q5q1:.3f}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
