#!/usr/bin/env python
"""
232c: twCRPS pathwise-jump auxiliary. Attacks max-jump KS gate.

Adds a threshold-weighted CRPS loss on the per-window functional
  M = max_{t, cell} |ΔIV_{t,cell}|    (per path; 25 cells · 30 steps)

twCRPS form (Gneiting & Ranjan 2011):
  twCRPS(F, y) = int w(z) (F(z) - 1{y <= z})^2 dz

Chaining weight w(z) = (z - threshold)_+ is the "upper-tail" weight that
focuses score on large z. We approximate twCRPS via the energy-score form
  twCRPS(y_sample, y) ~ E|t(X) - t(y)| - 0.5 E|t(X) - t(X')|
where t(z) = max(z, threshold) (Allen 2023 chaining-transform variant).

Threshold: q90 of train-split GT max-jump distribution (pre-computed).

Warm-start from 229a@ep30. Inference uses scale_anchor at 0.50.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    make_serializable,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    variogram_score,
)
from experiments.backfill.block_ar.train_227a_factor_ar import (
    FactorARModel, load_model as load_227a_model,
    compute_trajectory_loss, afcrps_per_step,
)


# ---------------------------------------------------------------------------
# Pathwise max-jump functional + twCRPS
# ---------------------------------------------------------------------------

def pathwise_max_jump(
    trajectory: torch.Tensor,
    prev0: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample max |ΔIV_t| over t=1..N-1 and all cells.

    trajectory: (B, K, N, 5, 5)
    prev0: (B, 5, 5) = history last frame
    Returns: (B, K) per-sample max-jump magnitude.
    """
    B, K, N = trajectory.shape[:3]
    # Concatenate prev0 as t=-1 so ΔIV at t=0 = traj[0] - prev0
    prev_bkf = prev0.unsqueeze(1).expand(B, K, 5, 5).unsqueeze(2)  # (B, K, 1, 5, 5)
    full = torch.cat([prev_bkf, trajectory], dim=2)  # (B, K, N+1, 5, 5)
    deltas = (full[:, :, 1:] - full[:, :, :-1]).abs()  # (B, K, N, 5, 5)
    max_per_sample = deltas.reshape(B, K, -1).max(dim=-1).values  # (B, K)
    return max_per_sample


def gt_max_jump(future_01: torch.Tensor, hist_last: torch.Tensor) -> torch.Tensor:
    """Return per-window GT max |ΔIV_t|. future_01: (B, N, 5, 5), hist_last: (B, 5, 5)."""
    B, N = future_01.shape[:2]
    full = torch.cat([hist_last.unsqueeze(1), future_01], dim=1)  # (B, N+1, 5, 5)
    deltas = (full[:, 1:] - full[:, :-1]).abs()
    return deltas.reshape(B, -1).max(dim=-1).values  # (B,)


def twcrps_chained(
    samples: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Threshold-weighted CRPS via chaining transform v(z) = max(z, threshold).

    samples: (B, K), target: (B,). Returns scalar loss.

    Energy-score form: E|v(X) - v(y)| - 0.5 E|v(X) - v(X')|
    """
    vX = samples.clamp_min(threshold)  # (B, K)
    vy = target.clamp_min(threshold).unsqueeze(1)  # (B, 1)
    # |v(X) - v(y)| averaged over K
    mae = (vX - vy).abs().mean(dim=1)  # (B,)
    # Spread via random permutation of K
    K = samples.shape[1]
    idx = torch.randperm(K, device=samples.device)
    spread = (vX - vX[:, idx]).abs().mean(dim=1)  # (B,)
    return (mae - 0.5 * spread).mean()


# ---------------------------------------------------------------------------
# Model (reuses 227a FactorARModel; no architectural change)
# ---------------------------------------------------------------------------

def compute_232c_loss(
    trajectory: torch.Tensor,
    future_01: torch.Tensor,
    history_last: torch.Tensor,
    lambda_vs: float,
    lambda_jump: float,
    jump_threshold: float,
    loss_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-step ES + VS + pathwise twCRPS on max-jump."""
    B, K, N = trajectory.shape[:3]
    total_main = torch.tensor(0.0, device=trajectory.device)
    total_vs = torch.tensor(0.0, device=trajectory.device)
    for t in range(N):
        samples_t = trajectory[:, :, t].reshape(B, K, -1)
        gt_t = future_01[:, t].reshape(B, -1)
        if loss_type == "afcrps":
            total_main = total_main + afcrps_per_step(samples_t, gt_t)
        else:
            total_main = total_main + energy_score(samples_t, gt_t)
        total_vs = total_vs + variogram_score(samples_t, gt_t, p=0.5)

    # twCRPS on max-jump
    sample_maxes = pathwise_max_jump(trajectory, history_last)  # (B, K)
    target_maxes = gt_max_jump(future_01, history_last)  # (B,)
    jump_loss = twcrps_chained(sample_maxes, target_maxes, jump_threshold)

    loss = total_main + lambda_vs * total_vs + lambda_jump * jump_loss
    metrics = {
        "main": float((total_main / N).detach().item()),
        "vs": float((total_vs / N).detach().item()),
        "jump": float(jump_loss.detach().item()),
        "sample_max_mean": float(sample_maxes.mean().detach().item()),
        "target_max_mean": float(target_maxes.mean().detach().item()),
        "loss": float((loss / N).detach().item()),
    }
    return loss, metrics


def compute_jump_threshold(train_future: torch.Tensor, train_hist: torch.Tensor, q: float = 0.90) -> float:
    """Compute q-quantile of GT max-jump distribution from training set."""
    hist_last = train_hist[:, -1]  # (N, 5, 5)
    targets = gt_max_jump(train_future, hist_last)  # (N,)
    return float(torch.quantile(targets, q).item())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str, device: torch.device,
) -> tuple[FactorARModel, dict[str, Any]]:
    # 232c's model is 227a's FactorARModel (no arch change); reuse 227a loader
    return load_227a_model(checkpoint_path, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="232c: twCRPS pathwise-jump aux")
    p.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    p.add_argument("--pca_init", type=str, default="models/backfill/226a_pca_init.npz")
    p.add_argument("--warmstart_checkpoint", type=str, required=True)
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--val_size", type=int, default=441)
    p.add_argument("--max_train_windows", type=int, default=4010)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_members", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_vs", type=float, default=0.05)
    p.add_argument("--lambda_jump", type=float, default=0.5)
    p.add_argument("--jump_threshold_q", type=float, default=0.90)
    p.add_argument("--loss_type", type=str, default="es", choices=["es", "afcrps"])
    p.add_argument("--factor_rank", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--decoder_hidden", type=int, default=256)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--pos_embed_dim", type=int, default=16)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--ewma_alpha", type=float, default=0.20)
    p.add_argument("--scale_floor", type=float, default=1e-4)
    p.add_argument("--noise_skip", action="store_true")
    p.add_argument("--d_scale", type=float, default=3.0)
    p.add_argument("--no_tanh", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=3)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FactorARModel(
        n_cells=25, factor_rank=args.factor_rank,
        hidden_dim=args.hidden_dim, gru_layers=2, gru_dropout=0.1,
        decoder_hidden=args.decoder_hidden,
        rho=args.rho, ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor, include_scale_feature=True,
        pos_embed_dim=args.pos_embed_dim,
        noise_skip=args.noise_skip, d_scale=args.d_scale,
        cell_spread=False, decoder_layers=args.decoder_layers,
        no_tanh=args.no_tanh,
        use_scale_anchor=False, scale_anchor_alpha=0.50,
    ).to(device)

    pca = np.load(args.pca_init)
    model.init_from_pca(pca["lambda_init"], pca["d_init"])
    ws = torch.load(args.warmstart_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ws["model_state_dict"], strict=False)
    print(f"Warm-started from {args.warmstart_checkpoint}")

    # Data
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = args.test_start - args.history_len - args.n_steps
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, args.n_steps
    )
    train_future = train_future.view(train_hist.shape[0], args.n_steps, 5, 5)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.n_steps
    )
    val_future = val_future.view(val_hist.shape[0], args.n_steps, 5, 5)

    # Compute jump threshold from TRAINING data
    jump_threshold = compute_jump_threshold(
        train_future, train_hist, q=args.jump_threshold_q
    )
    print(f"Jump threshold (q{int(args.jump_threshold_q*100)} of train GT max-jump): {jump_threshold:.4f}")

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = {
        "type": "factor_ar_232c",
        "n_cells": 25, "factor_rank": args.factor_rank,
        "hidden_dim": args.hidden_dim, "gru_layers": 2, "gru_dropout": 0.1,
        "decoder_hidden": args.decoder_hidden,
        "rho": args.rho, "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor, "include_scale_feature": True,
        "pos_embed_dim": args.pos_embed_dim, "lambda_vs": args.lambda_vs,
        "lambda_jump": args.lambda_jump, "jump_threshold": jump_threshold,
        "jump_threshold_q": args.jump_threshold_q,
        "n_steps": args.n_steps, "n_members": args.n_members,
        "noise_skip": args.noise_skip, "d_scale": args.d_scale,
        "cell_spread": False, "decoder_layers": args.decoder_layers,
        "loss_type": args.loss_type, "no_tanh": args.no_tanh,
        "use_scale_anchor": False, "scale_anchor_alpha": 0.50,
        "warmstart_checkpoint": args.warmstart_checkpoint,
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n232c twCRPS Jump-Aux Factor AR")
    print(f"  lambda_jump={args.lambda_jump} threshold(q={args.jump_threshold_q})={jump_threshold:.4f}")
    print(f"  decoder_hidden={args.decoder_hidden}")
    print(f"  n_members={args.n_members} batch={args.batch_size} lr={args.lr}")
    print(f"  train={train_hist.shape[0]} val={val_hist.shape[0]}")
    print(f"  params: {n_params:,}")

    history_log: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "main": 0.0, "vs": 0.0, "jump": 0.0,
                   "sample_max_mean": 0.0, "target_max_mean": 0.0}
        count = 0
        for history_01, future_01 in train_loader:
            trajectory = model(history_01, n_members=args.n_members, n_steps=args.n_steps)
            history_last = history_01[:, -1]
            loss, metrics = compute_232c_loss(
                trajectory, future_01, history_last,
                lambda_vs=args.lambda_vs, lambda_jump=args.lambda_jump,
                jump_threshold=jump_threshold, loss_type=args.loss_type,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            b = history_01.shape[0]
            for k in running:
                running[k] += metrics[k] * b
            count += b
        train_metrics = {k: v / count for k, v in running.items()}

        model.eval()
        val_sum = {"main": 0.0, "vs": 0.0, "jump": 0.0,
                   "sample_max_mean": 0.0, "target_max_mean": 0.0}
        val_count = 0
        with torch.no_grad():
            for v_start in range(0, val_hist.shape[0], args.batch_size):
                v_end = min(v_start + args.batch_size, val_hist.shape[0])
                vh = val_hist[v_start:v_end]
                vf = val_future[v_start:v_end]
                vt = model(vh, n_members=args.n_members, n_steps=args.n_steps)
                hist_last = vh[:, -1]
                _, vm = compute_232c_loss(
                    vt, vf, hist_last,
                    args.lambda_vs, args.lambda_jump, jump_threshold, args.loss_type,
                )
                vb = vh.shape[0]
                for k in val_sum:
                    val_sum[k] += vm[k] * vb
                val_count += vb
        val_main = val_sum["main"] / val_count
        val_vs = val_sum["vs"] / val_count
        val_jump = val_sum["jump"] / val_count
        val_loss = val_main + args.lambda_vs * val_vs + args.lambda_jump * val_jump

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "elapsed": elapsed,
            "train_loss": train_metrics["loss"],
            "train_main": train_metrics["main"],
            "train_vs": train_metrics["vs"],
            "train_jump": train_metrics["jump"],
            "val_main": val_main, "val_vs": val_vs,
            "val_jump": val_jump, "val_loss": val_loss,
            "train_sample_max": train_metrics["sample_max_mean"],
            "train_target_max": train_metrics["target_max_mean"],
            "val_sample_max": val_sum["sample_max_mean"] / val_count,
            "val_target_max": val_sum["target_max_mean"] / val_count,
        }
        history_log.append(make_serializable(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
            }, out_dir / "best_model.pt")
        if epoch % args.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        print(
            f"[{epoch:3d}/{args.epochs}] loss={train_metrics['loss']:.4f} "
            f"main={train_metrics['main']:.4f} jump={train_metrics['jump']:.4f} "
            f"val={val_loss:.4f}  sample_max={train_metrics['sample_max_mean']:.3f} "
            f"gt_max={train_metrics['target_max_mean']:.3f}  ({elapsed:.1f}s)"
        )

    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "epoch": args.epochs, "val_loss": val_loss,
    }, out_dir / "final_model.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nSaved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
