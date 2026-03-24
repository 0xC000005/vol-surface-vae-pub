#!/usr/bin/env python
"""
153b: Conditional One-Shot FM with Residual Prediction

RC15-H1-S2: Predict delta = future - persistence instead of absolute future.
FMAP design: residual prediction reduces ODE transport distance.
Build on 153a (conditional factored transformer + frozen encoder).

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_cond_residual_flow.py \
        --epochs 200 --batch_size 64 --lr 5e-4 \
        --encoder_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --output_dir models/backfill/flow_153b --device cuda
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
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_cond_oneshot_flow import (
    ConditionalFactoredVelocityTransformer, load_encoder, normalize_iv,
    make_serializable
)
from experiments.backfill.block_ar.train_oneshot_flow import evaluate_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--encoder_path", type=str,
                        default="models/backfill/block_ar_vol_scaled_30ep/best_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load encoder (FROZEN)
    encoder, cond_dim = load_encoder(args.encoder_path, device)
    for p in encoder.parameters():
        p.requires_grad = False

    # Data — need history, future, AND persistence
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    H, F_LEN, DIM = 30, 30, 750
    train_end = 4040

    train_histories = []
    train_futures = []
    train_persist = []
    for i in range(train_end - H - F_LEN + 1):
        history = surfaces[i:i+H]
        future = surfaces[i+H:i+H+F_LEN].reshape(-1)
        last_frame = surfaces[i+H-1]
        persistence = np.tile(last_frame.reshape(1, -1), (F_LEN, 1)).reshape(-1)
        train_histories.append(history)
        train_futures.append(future)
        train_persist.append(persistence)

    train_hist = np.array(train_histories, dtype=np.float32)
    train_data = np.array(train_futures, dtype=np.float32)
    train_persist_data = np.array(train_persist, dtype=np.float32)

    # Compute residuals: delta = future - persistence
    train_residuals = train_data - train_persist_data

    val_histories = []
    val_futures = []
    val_persist = []
    for i in range(train_end - H - F_LEN + 1, 4540 - H - F_LEN + 1):
        val_histories.append(surfaces[i:i+H])
        val_futures.append(surfaces[i+H:i+H+F_LEN].reshape(-1))
        last_frame = surfaces[i+H-1]
        val_persist.append(np.tile(last_frame.reshape(1, -1), (F_LEN, 1)).reshape(-1))

    val_hist = np.array(val_histories, dtype=np.float32)
    val_data = np.array(val_futures, dtype=np.float32)
    val_persist_data = np.array(val_persist, dtype=np.float32)
    val_residuals = val_data - val_persist_data

    # Standardize RESIDUALS (not absolute futures!)
    res_mean = train_residuals.mean(axis=0, keepdims=True)
    res_std = train_residuals.std(axis=0, keepdims=True) + 1e-6
    train_res_norm = (train_residuals - res_mean) / res_std
    val_res_norm = (val_residuals - res_mean) / res_std

    # Also keep absolute future stats for evaluation
    train_mean_abs = train_data.mean(axis=0, keepdims=True)
    train_std_abs = train_data.std(axis=0, keepdims=True) + 1e-6

    print("=" * 60)
    print("153b: Conditional One-Shot FM — Residual Prediction")
    print("=" * 60)

    # Residual statistics
    res_norm = np.linalg.norm(train_residuals, axis=1)
    abs_norm = np.linalg.norm(train_data, axis=1)
    print(f"  Residual L2 norm: mean={res_norm.mean():.2f}, std={res_norm.std():.2f}")
    print(f"  Absolute L2 norm: mean={abs_norm.mean():.2f}, std={abs_norm.std():.2f}")
    print(f"  Transport reduction: {1 - res_norm.mean()/abs_norm.mean():.1%}")
    print(f"  Train residuals: {train_res_norm.shape}, Val: {val_res_norm.shape}")

    # Pre-compute encoder conditions
    print("Pre-computing encoder conditions...")
    train_conds = []
    with torch.no_grad():
        for i in range(0, len(train_hist), 256):
            bh = torch.from_numpy(train_hist[i:i+256]).to(device)
            c = encoder(normalize_iv(bh))
            train_conds.append(c.cpu().numpy())
    train_conds = np.concatenate(train_conds)

    val_conds = []
    with torch.no_grad():
        for i in range(0, len(val_hist), 256):
            bh = torch.from_numpy(val_hist[i:i+256]).to(device)
            c = encoder(normalize_iv(bh))
            val_conds.append(c.cpu().numpy())
    val_conds = np.concatenate(val_conds)

    # Model — same architecture as 153a
    model = ConditionalFactoredVelocityTransformer(
        n_frames=F_LEN, n_cells=25,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        cond_dim=cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Dataset: residuals + conditions + persistence (for reconstruction)
    train_ds = TensorDataset(
        torch.from_numpy(train_res_norm),
        torch.from_numpy(train_conds),
        torch.from_numpy(train_persist_data),  # for eval reconstruction
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0; nb = 0
        for x1_res, cond, _ in train_loader:
            x1_res = x1_res.to(device)  # standardized residual
            cond = cond.to(device)
            B = x1_res.shape[0]
            x0 = torch.randn_like(x1_res)  # N(0,I) source
            t = torch.rand(B, device=device)
            x_t = (1 - t[:, None]) * x0 + t[:, None] * x1_res
            v = model(x_t, t, cond=cond)
            loss = F.mse_loss(v, x1_res - x0)
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
            for i in range(0, len(val_res_norm), args.batch_size):
                x1_res = torch.from_numpy(val_res_norm[i:i+args.batch_size]).to(device)
                cond = torch.from_numpy(val_conds[i:i+args.batch_size]).to(device)
                B = x1_res.shape[0]
                x0 = torch.randn_like(x1_res)
                t = torch.rand(B, device=device)
                x_t = (1 - t[:, None]) * x0 + t[:, None] * x1_res
                v = model(x_t, t, cond=cond)
                vl += F.mse_loss(v, x1_res - x0).item() * B; nv += B
        val_loss = vl / nv

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss,
                "res_mean": res_mean, "res_std": res_std,
                "train_mean_abs": train_mean_abs, "train_std_abs": train_std_abs,
                "encoder_path": args.encoder_path,
                "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                           "n_layers": args.n_layers, "n_steps": args.n_steps,
                           "n_frames": F_LEN, "n_cells": 25, "dim": DIM,
                           "cond_dim": cond_dim, "source": "gaussian",
                           "conditioning": "concatenation",
                           "prediction": "residual"},
            }, f"{args.output_dir}/best_model.pt")

        # Evaluate every 40 epochs
        if epoch % 40 == 0 or epoch == 1:
            with torch.no_grad():
                all_samp = []
                eval_batch = 64
                n_eval = 512
                dt_val = 1.0 / args.n_steps
                for si in range(0, n_eval, eval_batch):
                    eb = min(eval_batch, n_eval - si)
                    x = torch.randn(eb, DIM, device=device)
                    idx = np.random.choice(len(train_conds), eb, replace=True)
                    c = torch.from_numpy(train_conds[idx]).to(device)
                    persist_batch = train_persist_data[idx]
                    for step in range(args.n_steps):
                        tt = torch.full((eb,), step * dt_val, device=device)
                        x = x + model(x, tt, cond=c) * dt_val
                    # Denormalize residual and add persistence back
                    res_raw = x.cpu().numpy() * res_std + res_mean
                    samples_batch = persist_batch + res_raw
                    samples_batch = np.clip(samples_batch, 0, 1)
                    all_samp.append(samples_batch)
                samples = np.concatenate(all_samp)
            m = evaluate_samples(samples, train_data)
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
        "model_state_dict": model.state_dict(), "epoch": args.epochs,
        "val_loss": val_loss,
        "res_mean": res_mean, "res_std": res_std,
        "train_mean_abs": train_mean_abs, "train_std_abs": train_std_abs,
        "encoder_path": args.encoder_path,
        "config": {"d_model": args.d_model, "n_heads": args.n_heads,
                   "n_layers": args.n_layers, "n_steps": args.n_steps,
                   "n_frames": F_LEN, "n_cells": 25, "dim": DIM,
                   "cond_dim": cond_dim, "source": "gaussian",
                   "conditioning": "concatenation",
                   "prediction": "residual"},
    }, f"{args.output_dir}/final_model.pt")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(make_serializable(history), f, indent=2)
    print(f"\nBest val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
