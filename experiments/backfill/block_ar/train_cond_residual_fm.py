#!/usr/bin/env python
"""
154c: Conditional Residual FM (build on 154b)

Same as 154b but the residual FM is CONDITIONAL — receives encoder output so it
can generate regime-aware perturbations (wider in turbulent, narrower in calm).

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_cond_residual_fm.py \
        --base_model models/backfill/flow_153a/final_model.pt \
        --epochs 200 --batch_size 64 --lr 5e-4 \
        --output_dir models/backfill/flow_154c --device cuda
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp, kurtosis

import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv,
    make_serializable
)
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples


def generate_base_predictions(base_model, encoder, surfaces, start_idx, end_idx,
                               train_mean, train_std, device, n_steps=8):
    """Generate 153a ODE predictions and encoder conditions for a range of windows."""
    H, T, DIM = 30, 30, 750
    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)
    dt = 1.0 / n_steps

    predictions = []; gt_futures = []; conditions = []

    with torch.no_grad():
        for i in range(start_idx, end_idx - H - T + 1):
            hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
            cond = encoder(normalize_iv(hist))
            conditions.append(cond.cpu().numpy())

            # Average of 5 ODE runs for stable mean
            preds = []
            for _ in range(5):
                x = torch.randn(1, DIM, device=device)
                for step in range(n_steps):
                    t = torch.full((1,), step * dt, device=device)
                    x = x + base_model(x, t, cond=cond) * dt
                preds.append((x * std_t + mean_t).clamp(0, 1).cpu().numpy().flatten())
            predictions.append(np.mean(preds, axis=0))
            gt_futures.append(surfaces[i+H:i+H+T].reshape(-1))

            if len(predictions) % 100 == 0:
                print(f"  Generated {len(predictions)} predictions...")

    return (np.array(predictions, dtype=np.float32),
            np.array(gt_futures, dtype=np.float32),
            np.concatenate(conditions, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load base model + encoder (frozen)
    ckpt = torch.load(args.base_model, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    base_model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.to(device).eval()
    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in base_model.parameters(): p.requires_grad = False
    for p in encoder.parameters(): p.requires_grad = False

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, DIM = 30, 30, 750

    # Check if base predictions already cached
    cache_path = Path("models/backfill/flow_154b/base_predictions.npz")
    if cache_path.exists():
        print("Loading cached base predictions from 154b...")
        cached = np.load(cache_path)
        val_preds = cached["val_preds"]; val_gts = cached["val_gts"]
        train_preds = cached["train_preds"]; train_gts = cached["train_gts"]
        # Still need conditions
        print("Computing encoder conditions...")
        val_conds = []
        with torch.no_grad():
            for i in range(4040, 4540 - H - T + 1):
                hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
                val_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
        val_conds = np.concatenate(val_conds)
        train_conds = []
        with torch.no_grad():
            for i in range(3540, 4040 - H - T + 1):
                hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
                train_conds.append(encoder(normalize_iv(hist)).cpu().numpy())
        train_conds = np.concatenate(train_conds)
    else:
        print("Generating base predictions...")
        val_preds, val_gts, val_conds = generate_base_predictions(
            base_model, encoder, surfaces, 4040, 4540,
            ckpt["train_mean"], ckpt["train_std"], device)
        train_preds, train_gts, train_conds = generate_base_predictions(
            base_model, encoder, surfaces, 3540, 4040,
            ckpt["train_mean"], ckpt["train_std"], device)

    val_residuals = val_gts - val_preds
    train_residuals = train_gts - train_preds
    res_mean = val_residuals.mean(axis=0, keepdims=True)
    res_std = val_residuals.std(axis=0, keepdims=True) + 1e-6
    val_res_norm = (val_residuals - res_mean) / res_std
    train_res_norm = (train_residuals - res_mean) / res_std

    print("=" * 60)
    print("154c: Conditional Residual FM")
    print("=" * 60)
    print(f"  Val residuals: {val_res_norm.shape}, conditions: {val_conds.shape}")

    # Conditional residual FM (same architecture as 153a but smaller)
    res_model = ConditionalFactoredVelocityTransformer(
        n_frames=T, n_cells=25,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        cond_dim=cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in res_model.parameters())
    print(f"  Residual FM params: {n_params:,}")

    train_ds = TensorDataset(
        torch.from_numpy(val_res_norm),
        torch.from_numpy(val_conds),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(res_model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        res_model.train()
        epoch_loss = 0; nb = 0
        for x1, cond in train_loader:
            x1 = x1.to(device); cond = cond.to(device)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            v = res_model(x_t, t, cond=cond)
            loss = F.mse_loss(v, x1 - x0)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        train_loss = epoch_loss / nb
        elapsed = time.time() - t0

        # Val
        res_model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for i in range(0, len(train_res_norm), args.batch_size):
                x1 = torch.from_numpy(train_res_norm[i:i+args.batch_size]).to(device)
                c = torch.from_numpy(train_conds[i:i+args.batch_size]).to(device)
                B = x1.shape[0]
                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=device)
                x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
                v = res_model(x_t, t, cond=c)
                vl += F.mse_loss(v, x1 - x0).item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": res_model.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
                "res_mean": res_mean, "res_std": res_std,
                "base_model_path": args.base_model,
                "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "n_steps": args.n_steps,
                           "n_frames": T, "n_cells": 25, "dim": DIM,
                           "cond_dim": cond_dim, "type": "conditional_residual_fm"},
            }, f"{args.output_dir}/best_model.pt")

        if epoch % 40 == 0 or epoch == 1:
            with torch.no_grad():
                all_combined = []
                n_eval = min(512, len(val_preds))
                dt = 1.0 / args.n_steps
                for si in range(0, n_eval, 64):
                    eb = min(64, n_eval - si)
                    x = torch.randn(eb, DIM, device=device)
                    c = torch.from_numpy(val_conds[si:si+eb]).to(device)
                    for step in range(args.n_steps):
                        tt = torch.full((eb,), step * dt, device=device)
                        x = x + res_model(x, tt, cond=c) * dt
                    res_raw = x.cpu().numpy() * res_std + res_mean
                    combined = np.clip(val_preds[si:si+eb] + res_raw, 0, 1)
                    all_combined.append(combined)
                samples = np.concatenate(all_combined)
            m = evaluate_samples(samples, val_gts)
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)  eff_rank={m['eff_rank']:.2f}(GT={m['gt_eff_rank']:.2f})  "
                  f"PC1={m['pc1']:.3f}  PC2={m['pc2']:.3f}  frob={m['frob']:.1f}  "
                  f"KS={m['ks_pass']}/25  kurt={m['kurt_ratio']:.3f}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, **m})
        else:
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)")

    torch.save({
        "model_state_dict": res_model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss,
        "res_mean": res_mean, "res_std": res_std,
        "base_model_path": args.base_model,
        "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "n_steps": args.n_steps,
                   "n_frames": T, "n_cells": 25, "dim": DIM,
                   "cond_dim": cond_dim, "type": "conditional_residual_fm"},
    }, f"{args.output_dir}/final_model.pt")
    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
