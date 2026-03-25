#!/usr/bin/env python
"""
154b: Residual Flow Matching (ArchesWeatherGen Design)

RC16-H2-S2: Train a second FM model on normalized residuals from 153a's ODE
predictions on held-out data. At inference: sample = base_pred + residual_sample.

Key insight from ArchesWeatherGen: residuals on HELD-OUT data are wider than
training data residuals (base has overfit to training). Must use held-out residuals.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_residual_fm.py \
        --base_model models/backfill/flow_153a/final_model.pt \
        --epochs 200 --batch_size 64 --lr 5e-4 \
        --output_dir models/backfill/flow_154b --device cuda
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
from experiments.backfill.block_ar.train_oneshot_flow import (
    FactoredVelocityTransformer, evaluate_samples
)


def generate_base_predictions(base_model, encoder, surfaces, start_idx, end_idx,
                               train_mean, train_std, device, n_steps=8):
    """Generate 153a ODE predictions for a range of windows."""
    H, T, DIM = 30, 30, 750
    mean_t = torch.from_numpy(train_mean).float().to(device)
    std_t = torch.from_numpy(train_std).float().to(device)

    predictions = []
    gt_futures = []
    dt = 1.0 / n_steps

    with torch.no_grad():
        for i in range(start_idx, end_idx - H - T + 1):
            hist = torch.from_numpy(surfaces[i:i+H][None].astype(np.float32)).to(device)
            cond = encoder(normalize_iv(hist))

            # Average over 5 ODE samples for a stable mean prediction
            preds = []
            for _ in range(5):
                x = torch.randn(1, DIM, device=device)
                for step in range(n_steps):
                    t = torch.full((1,), step * dt, device=device)
                    x = x + base_model(x, t, cond=cond) * dt
                pred = (x * std_t + mean_t).clamp(0, 1).cpu().numpy().flatten()
                preds.append(pred)
            mean_pred = np.mean(preds, axis=0)
            predictions.append(mean_pred)

            gt = surfaces[i+H:i+H+T].reshape(-1)
            gt_futures.append(gt)

            if len(predictions) % 100 == 0:
                print(f"  Generated {len(predictions)} predictions...")

    return np.array(predictions, dtype=np.float32), np.array(gt_futures, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str,
                        default="models/backfill/flow_153a/final_model.pt")
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=64)  # Smaller than base (128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)  # Fewer layers
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load base model (frozen)
    ckpt = torch.load(args.base_model, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    base_model = ConditionalFactoredVelocityTransformer(
        n_frames=cfg["n_frames"], n_cells=cfg["n_cells"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        cond_dim=cfg["cond_dim"],
    )
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.to(device).eval()
    encoder, _ = load_encoder(args.encoder_path, device)
    for p in base_model.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = False

    base_train_mean = ckpt["train_mean"]
    base_train_std = ckpt["train_std"]

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, T, DIM = 30, 30, 750

    # Step 1: Generate base predictions on VALIDATION set (held-out)
    # Use validation (4040-4540) for training the residual FM
    # Use a subset of training (3540-4040) for the residual FM's validation
    print("=" * 60)
    print("154b: Residual FM — Generating Base Predictions")
    print("=" * 60)

    print("\nGenerating predictions for VALIDATION windows (4040-4540)...")
    val_preds, val_gts = generate_base_predictions(
        base_model, encoder, surfaces, 4040, 4540,
        base_train_mean, base_train_std, device)

    print("Generating predictions for TRAIN-tail windows (3540-4040)...")
    train_preds, train_gts = generate_base_predictions(
        base_model, encoder, surfaces, 3540, 4040,
        base_train_mean, base_train_std, device)

    # Compute residuals
    val_residuals = val_gts - val_preds    # Held-out: wider residuals
    train_residuals = train_gts - train_preds  # For residual FM validation

    print(f"\nResidual statistics:")
    print(f"  Val residuals: mean={val_residuals.mean():.5f}, std={val_residuals.std():.5f}")
    print(f"  Train residuals: mean={train_residuals.mean():.5f}, std={train_residuals.std():.5f}")
    print(f"  Val/Train std ratio: {val_residuals.std() / train_residuals.std():.2f}")

    # Standardize residuals using VALIDATION statistics
    res_mean = val_residuals.mean(axis=0, keepdims=True)
    res_std = val_residuals.std(axis=0, keepdims=True) + 1e-6
    val_res_norm = (val_residuals - res_mean) / res_std
    train_res_norm = (train_residuals - res_mean) / res_std

    print(f"  Normalized val residuals: shape={val_res_norm.shape}")
    print(f"  Normalized train residuals: shape={train_res_norm.shape}")

    # Step 2: Train residual FM
    print("\n" + "=" * 60)
    print("154b: Training Residual FM")
    print("=" * 60)

    # Smaller architecture than base (residuals are simpler than absolute surfaces)
    residual_model = FactoredVelocityTransformer(
        n_frames=T, n_cells=25,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in residual_model.parameters())
    print(f"  Residual FM parameters: {n_params:,} (vs base {sum(p.numel() for p in base_model.parameters()):,})")

    train_ds = TensorDataset(torch.from_numpy(val_res_norm))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    optimizer = torch.optim.AdamW(residual_model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        residual_model.train()
        epoch_loss = 0; nb = 0
        for (x1,) in train_loader:
            x1 = x1.to(device)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            v = residual_model(x_t, t)
            loss = F.mse_loss(v, x1 - x0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        train_loss = epoch_loss / nb
        elapsed = time.time() - t0

        # Val loss on train-tail residuals
        residual_model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for i in range(0, len(train_res_norm), args.batch_size):
                x1 = torch.from_numpy(train_res_norm[i:i+args.batch_size]).to(device)
                B = x1.shape[0]
                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=device)
                x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
                v = residual_model(x_t, t)
                vl += F.mse_loss(v, x1 - x0).item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": residual_model.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
                "res_mean": res_mean, "res_std": res_std,
                "base_model_path": args.base_model,
                "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "n_steps": args.n_steps,
                           "n_frames": T, "n_cells": 25, "dim": DIM,
                           "type": "residual_fm"},
            }, f"{args.output_dir}/best_model.pt")

        # Evaluate every 40 epochs
        if epoch % 40 == 0 or epoch == 1:
            with torch.no_grad():
                # Generate residual samples and add to base predictions
                all_combined = []
                eval_batch = 64
                n_eval = min(512, len(val_preds))
                dt = 1.0 / args.n_steps
                for si in range(0, n_eval, eval_batch):
                    eb = min(eval_batch, n_eval - si)
                    # Sample residuals
                    x = torch.randn(eb, DIM, device=device)
                    for step in range(args.n_steps):
                        tt = torch.full((eb,), step * dt, device=device)
                        x = x + residual_model(x, tt) * dt
                    # Denormalize residuals
                    res_raw = x.cpu().numpy() * res_std + res_mean
                    # Add to base predictions
                    base_pred_batch = val_preds[si:si+eb]
                    combined = base_pred_batch + res_raw
                    combined = np.clip(combined, 0, 1)
                    all_combined.append(combined)
                samples = np.concatenate(all_combined)

            m = evaluate_samples(samples, val_gts)
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)  eff_rank={m['eff_rank']:.2f}(GT={m['gt_eff_rank']:.2f})  "
                  f"PC1={m['pc1']:.3f}  PC2={m['pc2']:.3f}  frob={m['frob']:.1f}  "
                  f"KS={m['ks_pass']}/25  kurt={m['kurt_ratio']:.3f}  "
                  f"spread h1={m['spread_h1']:.4f} h30={m['spread_h30']:.4f}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, **m})
        else:
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)")

    # Save final
    torch.save({
        "model_state_dict": residual_model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss,
        "res_mean": res_mean, "res_std": res_std,
        "base_model_path": args.base_model,
        "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "n_steps": args.n_steps,
                   "n_frames": T, "n_cells": 25, "dim": DIM,
                   "type": "residual_fm"},
    }, f"{args.output_dir}/final_model.pt")

    # Save base predictions for later use
    np.savez(f"{args.output_dir}/base_predictions.npz",
             val_preds=val_preds, val_gts=val_gts,
             train_preds=train_preds, train_gts=train_gts,
             res_mean=res_mean, res_std=res_std)

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
