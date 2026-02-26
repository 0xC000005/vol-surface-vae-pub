"""
Separate-pathway sigma head: learns vol_scale directly from raw history.

Three architectures:
1. "raw_mlp": Flatten last 10 days of history → MLP → sigma
   (Direct access to daily changes, bypasses encoder bottleneck)
2. "mini_gru": Small GRU on mean-IV time series → sigma
   (Dedicated temporal model for volatility pattern)
3. "formula": Differentiable vol_scale formula (baseline — tests integration only)

The sigma head is trained SEPARATELY from the main model.
At inference, sigma_head(history) replaces hand-coded vol_scale.
"""

import argparse
import dataclasses
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


def compute_vol_scale(history, gmv=0.0187):
    past_abs = denormalize_iv(history)
    mean_iv = past_abs.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vol = daily_chg.std(dim=1, keepdim=True)
    return (vol / gmv).clamp(0.5, 2.0)


def compute_vol_of_vol(history):
    past_abs = denormalize_iv(history)
    mean_iv = past_abs.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    return daily_chg.std(dim=1)


class RawMLPSigma(nn.Module):
    """Sigma from flattened recent history frames.

    Takes last K frames of mean-IV, computes daily changes,
    and feeds to MLP. Direct access to the information vol_scale uses.
    """

    def __init__(self, lookback: int = 10, hidden_dim: int = 32):
        super().__init__()
        # Input: daily changes of mean IV over last lookback frames
        self.lookback = lookback
        self.net = nn.Sequential(
            nn.Linear(lookback - 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, T, 5, 5) normalized IV history

        Returns:
            sigma: (B, 1) positive
        """
        past_abs = denormalize_iv(history)
        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T)
        # Use last lookback frames
        recent = mean_iv[:, -self.lookback:]  # (B, lookback)
        daily_chg = recent[:, 1:] - recent[:, :-1]  # (B, lookback-1)
        raw = self.net(daily_chg)  # (B, 1)
        return F.softplus(raw) + 0.01  # (B, 1) positive


class MiniGRUSigma(nn.Module):
    """Small GRU dedicated to volatility pattern → sigma.

    Separate from the main encoder. Processes the mean-IV time series
    (not the full 5x5 surface) to learn vol_of_vol patterns.
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,  # scalar mean-IV per timestep
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        past_abs = denormalize_iv(history)
        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T)
        # Daily changes as input (captures volatility pattern)
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]  # (B, T-1)
        x = daily_chg.unsqueeze(-1)  # (B, T-1, 1)
        _, h_n = self.gru(x)  # h_n: (1, B, hidden)
        h = h_n.squeeze(0)  # (B, hidden)
        raw = self.head(h)  # (B, 1)
        return F.softplus(raw) + 0.01


class FormulaSigma(nn.Module):
    """Differentiable vol_scale formula (baseline).

    Tests that integration works by using the exact hand-coded formula.
    Has NO learnable parameters — should match vol_scale exactly.
    """

    def __init__(self, gmv: float = 0.0187):
        super().__init__()
        self.gmv = gmv
        # Add a dummy learnable parameter to make optimizer happy
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return compute_vol_scale(history, self.gmv)  # (B, 1)


def train_epoch(sigma_head, dataloader, optimizer, device, gmv=0.0187):
    sigma_head.train()
    total_loss = 0.0
    total_corr = 0.0
    n = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        history = batch["history"].to(device)
        target = compute_vol_scale(history, gmv).to(device)  # (B, 1)

        pred = sigma_head(history)  # (B, 1)
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


def validate(sigma_head, dataloader, device, gmv=0.0187):
    sigma_head.eval()
    all_pred = []
    all_vs = []
    all_vov = []
    all_h1 = []

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            pred = sigma_head(history).squeeze().cpu()
            all_pred.append(pred)

            vs = compute_vol_scale(history, gmv).squeeze().cpu()
            all_vs.append(vs)

            vov = compute_vol_of_vol(history).cpu()
            all_vov.append(vov)

            baseline = denormalize_iv(history[:, -1]).clamp(min=0.01)
            fut_h1 = denormalize_iv(future[:, 0])
            h1 = (fut_h1 - baseline).abs().mean(dim=(-1, -2)).cpu()
            all_h1.append(h1)

    pred = torch.cat(all_pred).numpy()
    vs = torch.cat(all_vs).numpy()
    vov = torch.cat(all_vov).numpy()
    h1 = torch.cat(all_h1).numpy()

    corr_vs = stats.spearmanr(pred, vs)[0]
    corr_vov = stats.spearmanr(pred, vov)[0]
    corr_h1 = stats.spearmanr(pred, h1)[0]

    q = np.quantile(vov, [0.2, 0.8])
    q1m = vov <= q[0]
    q5m = vov >= q[1]
    q5q1_pred = pred[q5m].mean() / pred[q1m].mean() if pred[q1m].mean() > 0 else 0
    q5q1_vs = vs[q5m].mean() / vs[q1m].mean()

    mse = np.mean((pred - vs) ** 2)

    return {
        "corr_vs": corr_vs,
        "corr_vov": corr_vov,
        "corr_h1": corr_h1,
        "q5q1_pred": q5q1_pred,
        "q5q1_vs": q5q1_vs,
        "mse_vs_target": mse,
        "pred_mean": pred.mean(),
        "pred_std": pred.std(),
        "cov": pred.std() / pred.mean() if pred.mean() > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="raw_mlp",
                        choices=["raw_mlp", "mini_gru", "formula"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lookback", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Create sigma head
    if args.arch == "raw_mlp":
        sh = RawMLPSigma(lookback=args.lookback, hidden_dim=args.hidden_dim).to(args.device)
    elif args.arch == "mini_gru":
        sh = MiniGRUSigma(hidden_dim=args.hidden_dim).to(args.device)
    elif args.arch == "formula":
        sh = FormulaSigma().to(args.device)

    n_params = sum(p.numel() for p in sh.parameters() if p.requires_grad)
    print(f"Architecture: {args.arch}, Params: {n_params:,}")

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    train_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    val_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4040, end_idx=4540)
    test_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    opt = torch.optim.AdamW(sh.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr/10)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_corr = -1.0

    print(f"\nTraining {args.arch} sigma head ({args.epochs} epochs)")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        tm = train_epoch(sh, train_dl, opt, args.device)
        sched.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            vm = validate(sh, val_dl, args.device)
            print(
                f"Epoch {epoch:3d}/{args.epochs} | Loss: {tm['loss']:.6f} | "
                f"Train corr: {tm['correlation']:.3f} | "
                f"Val corr(vs): {vm['corr_vs']:.3f} | "
                f"Val corr(vov): {vm['corr_vov']:.3f} | "
                f"Val corr(h1): {vm['corr_h1']:.3f} | "
                f"Q5/Q1: {vm['q5q1_pred']:.3f} (vs: {vm['q5q1_vs']:.3f}) | "
                f"MSE: {vm['mse_vs_target']:.4f}"
            )
            if vm["corr_vs"] > best_corr:
                best_corr = vm["corr_vs"]
                torch.save({
                    "state_dict": sh.state_dict(),
                    "epoch": epoch,
                    "val_metrics": vm,
                    "config": {"arch": args.arch, "hidden_dim": args.hidden_dim,
                               "lookback": args.lookback},
                }, f"{args.output_dir}/best_sigma_head.pt")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {tm['loss']:.6f} | Train corr: {tm['correlation']:.3f}")

    # Final test
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    tm_test = validate(sh, test_dl, args.device)
    for k, v in tm_test.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
