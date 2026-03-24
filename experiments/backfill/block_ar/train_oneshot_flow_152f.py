#!/usr/bin/env python
"""
152f: One-Shot Flow Matching with Data-Dependent Source (Persistence)

RC15-H2: Instead of x0 ~ N(0,I), use persistence forecast (last history frame
repeated 30 times) as the ODE source distribution. Based on Lim et al. (2410.03229)
which proves data-dependent paths produce lower-variance velocity fields.

Same architecture as 152e (FactoredVelocityTransformer). Only the source changes.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_oneshot_flow_152f.py \
        --epochs 100 --batch_size 64 --lr 5e-4 \
        --output_dir models/backfill/flow_152f --device cuda
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

# Import the same architecture from 152e
import sys; sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_oneshot_flow import (
    FactoredVelocityTransformer, evaluate_samples
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Data — same as 152e but also build persistence sources
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, F_LEN, DIM = 30, 30, 750

    train_end = 4040
    futures = []
    persist_sources = []
    for i in range(train_end - H - F_LEN + 1):
        future = surfaces[i+H:i+H+F_LEN].reshape(-1)  # (750,)
        # Persistence = last history frame repeated F_LEN times
        last_frame = surfaces[i+H-1]  # (5, 5)
        persistence = np.tile(last_frame.reshape(1, -1), (F_LEN, 1)).reshape(-1)  # (750,)
        futures.append(future)
        persist_sources.append(persistence)

    train_data = np.array(futures, dtype=np.float32)
    train_persist = np.array(persist_sources, dtype=np.float32)

    val_futures = []
    val_persist = []
    for i in range(train_end - H - F_LEN + 1, 4540 - H - F_LEN + 1):
        future = surfaces[i+H:i+H+F_LEN].reshape(-1)
        last_frame = surfaces[i+H-1]
        persistence = np.tile(last_frame.reshape(1, -1), (F_LEN, 1)).reshape(-1)
        val_futures.append(future)
        val_persist.append(persistence)

    val_data = np.array(val_futures, dtype=np.float32)
    val_persist_data = np.array(val_persist, dtype=np.float32)

    # Standardize using the SAME statistics as futures (important!)
    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True) + 1e-6
    train_norm = (train_data - train_mean) / train_std
    val_norm = (val_data - train_mean) / train_std

    # Standardize persistence sources with SAME stats
    train_persist_norm = (train_persist - train_mean) / train_std
    val_persist_norm = (val_persist_data - train_mean) / train_std

    print("=" * 60)
    print("152f: One-Shot Flow Matching — Data-Dependent Source (Persistence)")
    print("=" * 60)
    print(f"  Train: {train_norm.shape}, Val: {val_norm.shape}")

    # Verify persistence is close to but different from future
    dist = np.linalg.norm(train_norm - train_persist_norm, axis=1)
    print(f"  Persistence-to-future distance: mean={dist.mean():.2f}, std={dist.std():.2f}")
    print(f"  Noise-to-future distance: mean={np.linalg.norm(train_norm, axis=1).mean():.2f}")

    # Model — same architecture as 152e
    model = FactoredVelocityTransformer(
        n_frames=F_LEN, n_cells=25,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Dataset includes both future (x1) and persistence source (x0)
    train_ds = TensorDataset(
        torch.from_numpy(train_norm),
        torch.from_numpy(train_persist_norm)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    # For velocity variance tracking (diagnostic 3)
    vel_variances_152f = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0; nb = 0
        for x1, x0_persist in train_loader:
            x1 = x1.to(device)
            x0 = x0_persist.to(device)  # Data-dependent source!
            B = x1.shape[0]

            t = torch.rand(B, device=device)
            # OT-CFM interpolation: x_t = (1-t)*x0 + t*x1
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
            v = model(x_t, t)
            # CFM target velocity: x1 - x0 (displacement from source to target)
            loss = F.mse_loss(v, x1 - x0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); nb += 1
        scheduler.step()
        train_loss = epoch_loss / nb
        elapsed = time.time() - t0

        # Val loss
        model.eval()
        vl = 0; nv = 0
        with torch.no_grad():
            for i in range(0, len(val_norm), args.batch_size):
                x1 = torch.from_numpy(val_norm[i:i+args.batch_size]).to(device)
                x0 = torch.from_numpy(val_persist_norm[i:i+args.batch_size]).to(device)
                B = x1.shape[0]
                t = torch.rand(B, device=device)
                x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
                v = model(x_t, t)
                vl += F.mse_loss(v, x1 - x0).item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "train_mean": train_mean, "train_std": train_std,
                "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "n_steps": args.n_steps,
                           "n_frames": F_LEN, "n_cells": 25, "dim": DIM,
                           "source": "persistence"},
            }, f"{args.output_dir}/best_model.pt")

        # Evaluate every 20 epochs (more frequent since only 100 total)
        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                # Velocity variance at t=0.5 (diagnostic 3)
                sample_idx = np.random.choice(len(train_norm), 256, replace=False)
                x1_samp = torch.from_numpy(train_norm[sample_idx]).to(device)
                x0_samp = torch.from_numpy(train_persist_norm[sample_idx]).to(device)
                t_half = torch.full((256,), 0.5, device=device)
                x_mid = 0.5 * x0_samp + 0.5 * x1_samp
                v_mid = model(x_mid, t_half)
                vel_var = v_mid.var(dim=0).mean().item()
                vel_variances_152f.append({"epoch": epoch, "vel_var_t05": vel_var})

                # Generate samples — ODE starts from a RANDOM persistence source
                # At inference, we don't have the "correct" persistence for a specific window.
                # Use random training persistence sources as starting points.
                all_samp = []
                eval_batch = 64
                n_eval = 512
                dt = 1.0 / args.n_steps
                for si in range(0, n_eval, eval_batch):
                    eb = min(eval_batch, n_eval - si)
                    # Sample random persistence sources
                    idx = np.random.choice(len(train_persist_norm), eb, replace=True)
                    x = torch.from_numpy(train_persist_norm[idx]).to(device)
                    for step in range(args.n_steps):
                        tt = torch.full((eb,), step * dt, device=device)
                        x = x + model(x, tt) * dt
                    all_samp.append(x.cpu().numpy())
                samples = np.concatenate(all_samp)
                samples = samples * train_std + train_mean
                samples = np.clip(samples, 0, 1)
            m = evaluate_samples(samples, train_data)
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)  eff_rank={m['eff_rank']:.2f}(GT={m['gt_eff_rank']:.2f})  "
                  f"PC1={m['pc1']:.3f}  PC2={m['pc2']:.3f}  frob={m['frob']:.1f}  "
                  f"KS={m['ks_pass']}/25  kurt={m['kurt_ratio']:.3f}  "
                  f"spread h1={m['spread_h1']:.4f} h30={m['spread_h30']:.4f} "
                  f"vel_var_t05={vel_var:.4f}")
            history.append({"epoch": epoch, "train_loss": train_loss,
                           "val_loss": val_loss, "vel_var_t05": vel_var, **m})
        else:
            print(f"Ep {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({elapsed:.1f}s)")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss, "train_mean": train_mean, "train_std": train_std,
        "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "n_steps": args.n_steps,
                   "n_frames": F_LEN, "n_cells": 25, "dim": DIM,
                   "source": "persistence"},
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{args.output_dir}/velocity_variance.json", "w") as f:
        json.dump(vel_variances_152f, f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Velocity variances saved to {args.output_dir}/velocity_variance.json")


if __name__ == "__main__":
    main()
