"""
Warm-start fine-tuning: vol_scaled → e2e_nll learned sigma.

Three-phase approach:
Phase 1: Load pre-trained vol_scaled model (Ratio V1)
Phase 2: Freeze everything, train sigma head to predict vol_scale (supervised)
Phase 3: Unfreeze, switch to e2e_nll mode, fine-tune with low LR

This avoids sigma collapse by initializing near the hand-coded solution.
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
    normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def compute_vol_scale(history, gmv=0.0187):
    """Hand-coded vol_scale from history."""
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


def phase2_train_sigma(model, dataloader, device, epochs=10, lr=1e-3):
    """Phase 2: Train sigma head to predict log(vol_scale) from frozen encoder."""
    # Freeze everything except sigma head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.log_std_head.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.log_std_head.parameters(), lr=lr)
    gmv = model.config.global_mean_vol

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_corr = 0
        n = 0
        for batch in tqdm(dataloader, desc=f"Phase 2 epoch {epoch}", leave=False):
            history = batch["history"].to(device)
            B = history.shape[0]

            with torch.no_grad():
                cond = model.encoder(history, mask=None)
                if model.config.forward_only:
                    cond = cond + model.encoder.null_embedding.expand(B, -1)

            # Target: log(vol_scale)
            vs = compute_vol_scale(history, gmv)  # (B, 1)
            target_log_vs = torch.log(vs).to(device)

            # Predict
            log_sigma = model.log_std_head(cond)  # (B, 1)
            loss = F.mse_loss(log_sigma, target_log_vs)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred_sigma = torch.exp(log_sigma.clamp(-2, 2)).squeeze()
                target_vs = vs.squeeze()
                c = stats.spearmanr(pred_sigma.cpu().numpy(), target_vs.cpu().numpy())[0]
            total_loss += loss.item()
            total_corr += c if not np.isnan(c) else 0
            n += 1

        print(f"  Phase 2 epoch {epoch}: loss={total_loss/n:.6f}, corr(sigma,vs)={total_corr/n:.3f}")

    # Check sigma statistics
    model.eval()
    all_sigma = []
    all_vs = []
    all_vov = []
    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            B = history.shape[0]
            cond = model.encoder(history, mask=None)
            if model.config.forward_only:
                cond = cond + model.encoder.null_embedding.expand(B, -1)
            log_sigma = model.log_std_head(cond)
            sigma = torch.exp(log_sigma.clamp(-2, 2)).squeeze().cpu()
            all_sigma.append(sigma)
            all_vs.append(compute_vol_scale(history, gmv).squeeze().cpu())
            all_vov.append(compute_vol_of_vol(history).cpu())

    sigma = torch.cat(all_sigma).numpy()
    vs = torch.cat(all_vs).numpy()
    vov = torch.cat(all_vov).numpy()
    print(f"  Phase 2 result: sigma mean={sigma.mean():.4f} std={sigma.std():.4f}")
    print(f"  Spearman(sigma, vol_scale)={stats.spearmanr(sigma, vs)[0]:.3f}")
    print(f"  Spearman(sigma, vol_of_vol)={stats.spearmanr(sigma, vov)[0]:.3f}")

    # Unfreeze everything for Phase 3
    for p in model.parameters():
        p.requires_grad = True


def phase3_finetune(model, train_dl, val_dl, device, epochs=5, lr=5e-5, output_dir=""):
    """Phase 3: Fine-tune with e2e_nll mode (non-detached sigma)."""
    # Switch mode to e2e_nll
    model.config.ratio_target_mode = "e2e_nll"

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/10)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n = 0
        for batch in tqdm(train_dl, desc=f"Phase 3 epoch {epoch}", leave=False):
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            result = model(history, future)
            loss = result["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n += 1

        sched.step()

        # Validate
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_dl:
                history = batch["history"].to(device)
                future = batch["future"].to(device)
                result = model(history, future)
                val_loss += result["loss"].item()
                vn += 1
        val_loss /= vn

        # Check sigma
        all_sigma = []
        all_vov = []
        with torch.no_grad():
            for batch in val_dl:
                history = batch["history"].to(device)
                B = history.shape[0]
                cond = model.encoder(history, mask=None)
                if model.config.forward_only:
                    cond = cond + model.encoder.null_embedding.expand(B, -1)
                log_sigma = model.log_std_head(cond)
                sigma = torch.exp(log_sigma.clamp(-2, 2)).squeeze().cpu()
                all_sigma.append(sigma)
                all_vov.append(compute_vol_of_vol(batch["history"]).cpu())

        sigma = torch.cat(all_sigma).numpy()
        vov = torch.cat(all_vov).numpy()
        rho = stats.spearmanr(sigma, vov)[0]

        print(f"  Phase 3 epoch {epoch}: train_loss={total_loss/n:.6f} val_loss={val_loss:.6f} "
              f"sigma mean={sigma.mean():.4f} std={sigma.std():.4f} rho(vov)={rho:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": dataclasses.asdict(model.config),
                "epoch": epoch,
                "phase": 3,
                "val_loss": val_loss,
                "sigma_stats": {"mean": float(sigma.mean()), "std": float(sigma.std()), "rho_vov": float(rho)},
            }, f"{output_dir}/best_model.pt")
            print(f"    Saved best model (val_loss={val_loss:.6f})")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(model.config),
        "epoch": epochs,
        "phase": 3,
        "sigma_stats": {"mean": float(sigma.mean()), "std": float(sigma.std()), "rho_vov": float(rho)},
    }, f"{output_dir}/final_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained vol_scaled checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase2_lr", type=float, default=1e-3)
    parser.add_argument("--phase3_epochs", type=int, default=10)
    parser.add_argument("--phase3_lr", type=float, default=5e-5)
    parser.add_argument("--nsdiff_sigma_lambda", type=float, default=0.1)
    parser.add_argument("--e2e_sigma_reg", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Phase 1: Load pre-trained model
    print(f"Loading checkpoint: {args.checkpoint}")
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cd = cp["config"]
    if dataclasses.is_dataclass(cd):
        cd = dataclasses.asdict(cd)

    # Create model with e2e_nll mode (adds log_std_head)
    cd_filtered = {k: v for k, v in cd.items()
                   if k in {f.name for f in dataclasses.fields(BlockARConfig)}}
    cd_filtered["ratio_target_mode"] = "e2e_nll"
    cd_filtered["nsdiff_sigma_lambda"] = args.nsdiff_sigma_lambda
    cd_filtered["e2e_sigma_reg"] = args.e2e_sigma_reg
    config = BlockARConfig(**cd_filtered)

    model = ConditionalBlockARDDPM(config)

    # Load weights from vol_scaled model (sigma head will be randomly initialized)
    missing, unexpected = model.load_state_dict(cp["model_state_dict"], strict=False)
    print(f"Loaded model. Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model = model.to(args.device)

    n_base = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in model.log_std_head.parameters())
    n_head = sum(p.numel() for p in model.log_std_head.parameters())
    print(f"Base model: {n_base:,} params, Sigma head: {n_head:,} params")

    # Data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    train_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    val_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4040, end_idx=4540)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Phase 2: Train sigma head to predict vol_scale
    print(f"\n{'='*60}")
    print("PHASE 2: Train sigma head on frozen encoder (predicting vol_scale)")
    print(f"{'='*60}")
    phase2_train_sigma(model, train_dl, args.device, epochs=args.phase2_epochs, lr=args.phase2_lr)

    # Phase 3: Fine-tune with e2e_nll
    print(f"\n{'='*60}")
    print("PHASE 3: Fine-tune with e2e_nll (non-detached sigma)")
    print(f"{'='*60}")
    phase3_finetune(model, train_dl, val_dl, args.device,
                    epochs=args.phase3_epochs, lr=args.phase3_lr,
                    output_dir=args.output_dir)

    print(f"\nDone. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
