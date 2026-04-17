#!/usr/bin/env python
"""
232d: Per-cell adaptive loss weighting. Attacks worst-cell coverage gate.

229a fails worst-cell coverage at h=30 (0.344 vs 0.70 gate). The failure is
typically concentrated in a few corner cells (deep OTM/short-maturity or deep
ITM/long-maturity) with sparse training data. Uniform loss weighting treats
all cells equally, so the optimizer has no incentive to specialize attention
to these.

This experiment replaces the uniform per-cell weight in energy_score/vs with
data-derived weights. The weight for cell c is determined by coverage deficit
observed during training: cells where the model is currently poorly
calibrated get higher weight.

Weight derivation (online):
  - Every `rebalance_every` epochs, evaluate per-cell coverage on a fixed
    validation subset
  - Cells with 90% coverage below `cov_threshold` (default 0.60) get weight
    `max_weight`; cells at or above get weight 1.0
  - Smooth transition between via linear interpolation

Bitter-Lesson check: weights are DERIVED from data (observed coverage gap),
not from domain knowledge about which cells matter. This is compliant.

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
    FactorARModel, load_model as load_227a_model, afcrps_per_step,
)


# ---------------------------------------------------------------------------
# Per-cell weighted energy score + variogram score
# ---------------------------------------------------------------------------

def weighted_energy_score(
    samples: torch.Tensor, target: torch.Tensor, cell_weights: torch.Tensor
) -> torch.Tensor:
    """Energy score with per-cell weights (D=25).

    samples: (B, K, D), target: (B, D), cell_weights: (D,)
    Returns: scalar.
    """
    B, K, D = samples.shape
    w = cell_weights.view(1, 1, D)  # broadcast over B, K
    # MAE weighted per-cell
    mae_per_cell = (samples - target.unsqueeze(1)).abs()  # (B, K, D)
    mae = (mae_per_cell * w).mean(dim=(1, 2))  # (B,)  — mean over K, weighted mean over D
    # Spread weighted per-cell (random permutation)
    idx = torch.randperm(K, device=samples.device)
    spread_per_cell = (samples - samples[:, idx]).abs()  # (B, K, D)
    spread = (spread_per_cell * w).mean(dim=(1, 2))  # (B,)
    # ES = MAE - 0.5 * spread
    es = mae - 0.5 * spread
    return es.mean()


def weighted_variogram_score(
    samples: torch.Tensor, target: torch.Tensor, cell_weights: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Variogram score with per-cell pair weight sqrt(w_i * w_j).

    Uses eps-clamped |diff|^p to avoid inf-gradient at diagonal (|x-x|=0)
    under p<1. Mirrors the stock variogram_score in 212s.
    """
    B, K, D = samples.shape
    eps = 1e-8
    # GT variogram: |gt[i] - gt[j]|^p
    gt_diff = (target.unsqueeze(2) - target.unsqueeze(1)).abs().clamp_min(eps).pow(p)  # (B, D, D)
    # Sample variogram (mean over K): E[|x[i] - x[j]|^p]
    samp_diff = (samples.unsqueeze(3) - samples.unsqueeze(2)).abs().clamp_min(eps).pow(p)
    samp_mean = samp_diff.mean(dim=1)  # (B, D, D)
    # squared error between sample and GT variograms, weighted by sqrt(w_i * w_j)
    pair_w = torch.sqrt(cell_weights.unsqueeze(1) * cell_weights.unsqueeze(0))  # (D, D)
    err = (samp_mean - gt_diff) ** 2
    # Use upper-triangular mask (i<j) — each pair counted once, no diagonal
    mask = torch.triu(torch.ones(D, D, device=samples.device, dtype=torch.bool), diagonal=1)
    # weighted[B, D, D] -> [B, n_pairs]
    weighted = err * pair_w.unsqueeze(0)
    return weighted[:, mask].mean()


# ---------------------------------------------------------------------------
# Coverage-based weight computation
# ---------------------------------------------------------------------------

def compute_per_cell_coverage(
    samples: torch.Tensor, target: torch.Tensor, alpha: float = 0.90
) -> torch.Tensor:
    """Compute per-cell 90% CI coverage. samples: (B, K, N, 5, 5), target: (B, N, 5, 5).

    Returns: (D=25,) per-cell coverage averaged over (B, N).
    """
    B, K, N = samples.shape[:3]
    s = samples.reshape(B, K, N, 25)
    t = target.reshape(B, N, 25)
    q_low = torch.quantile(s, (1.0 - alpha) / 2.0, dim=1)  # (B, N, 25)
    q_high = torch.quantile(s, 1.0 - (1.0 - alpha) / 2.0, dim=1)  # (B, N, 25)
    covered = ((t >= q_low) & (t <= q_high)).float()  # (B, N, 25)
    return covered.mean(dim=(0, 1))  # (25,)


def update_cell_weights(
    coverage: torch.Tensor,
    cov_threshold: float = 0.60,
    max_weight: float = 5.0,
    min_weight: float = 1.0,
) -> torch.Tensor:
    """Derive per-cell loss weights from observed coverage.

    cov >= 0.90 → weight 1.0
    cov <= cov_threshold → weight max_weight
    Linear interp between.
    """
    # normalize
    x = (0.90 - coverage).clamp_min(0.0)  # deficit from target
    x_max = 0.90 - cov_threshold  # max deficit
    frac = (x / max(x_max, 1e-3)).clamp(0.0, 1.0)
    weights = min_weight + (max_weight - min_weight) * frac
    return weights


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------

def compute_232d_loss(
    trajectory: torch.Tensor,
    future_01: torch.Tensor,
    cell_weights: torch.Tensor,
    lambda_vs: float,
    loss_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, K, N = trajectory.shape[:3]
    total_main = torch.tensor(0.0, device=trajectory.device)
    total_vs = torch.tensor(0.0, device=trajectory.device)
    # Normalize weights so the average weight is 1.0 (preserves loss scale)
    w = cell_weights / cell_weights.mean().clamp_min(1e-6)
    for t in range(N):
        samples_t = trajectory[:, :, t].reshape(B, K, -1)
        gt_t = future_01[:, t].reshape(B, -1)
        total_main = total_main + weighted_energy_score(samples_t, gt_t, w)
        total_vs = total_vs + weighted_variogram_score(samples_t, gt_t, w, p=0.5)
    loss = total_main + lambda_vs * total_vs
    metrics = {
        "main": float((total_main / N).detach().item()),
        "vs": float((total_vs / N).detach().item()),
        "max_weight": float(w.max().item()),
        "loss": float((loss / N).detach().item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str, device: torch.device,
) -> tuple[FactorARModel, dict[str, Any]]:
    return load_227a_model(checkpoint_path, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="232d: per-cell adaptive weighting")
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
    # weighting
    p.add_argument("--rebalance_every", type=int, default=3)
    p.add_argument("--cov_threshold", type=float, default=0.60)
    p.add_argument("--max_weight", type=float, default=5.0)
    p.add_argument("--coverage_eval_samples", type=int, default=32)
    p.add_argument("--coverage_eval_windows", type=int, default=96)
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

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Initial cell weights: uniform. Will be updated after first rebalance.
    cell_weights = torch.ones(25, device=device)

    config = {
        "type": "cellw_factor_ar_232d",
        "n_cells": 25, "factor_rank": args.factor_rank,
        "hidden_dim": args.hidden_dim, "gru_layers": 2, "gru_dropout": 0.1,
        "decoder_hidden": args.decoder_hidden,
        "rho": args.rho, "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor, "include_scale_feature": True,
        "pos_embed_dim": args.pos_embed_dim, "lambda_vs": args.lambda_vs,
        "n_steps": args.n_steps, "n_members": args.n_members,
        "noise_skip": args.noise_skip, "d_scale": args.d_scale,
        "cell_spread": False, "decoder_layers": args.decoder_layers,
        "loss_type": args.loss_type, "no_tanh": args.no_tanh,
        "use_scale_anchor": False, "scale_anchor_alpha": 0.50,
        "rebalance_every": args.rebalance_every,
        "cov_threshold": args.cov_threshold, "max_weight": args.max_weight,
        "warmstart_checkpoint": args.warmstart_checkpoint,
    }

    print(f"\n232d Per-Cell Adaptive Weighting")
    print(f"  rebalance_every={args.rebalance_every} cov_threshold={args.cov_threshold} max_weight={args.max_weight}")
    print(f"  decoder_hidden={args.decoder_hidden} n_members={args.n_members} lr={args.lr}")
    print(f"  train={train_hist.shape[0]} val={val_hist.shape[0]}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    history_log: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    # Fixed coverage-eval subset (from training windows, first N) for reproducibility
    cov_eval_hist = train_hist[:args.coverage_eval_windows]
    cov_eval_future = train_future[:args.coverage_eval_windows]

    for epoch in range(1, args.epochs + 1):
        # Rebalance cell weights every N epochs
        if (epoch - 1) % args.rebalance_every == 0:
            model.eval()
            with torch.no_grad():
                cov_samples_list = []
                for s in range(0, cov_eval_hist.shape[0], args.batch_size):
                    e = min(s + args.batch_size, cov_eval_hist.shape[0])
                    chunk = model(cov_eval_hist[s:e], n_members=args.coverage_eval_samples, n_steps=args.n_steps)
                    cov_samples_list.append(chunk)
                cov_samples = torch.cat(cov_samples_list, dim=0)
                coverage = compute_per_cell_coverage(cov_samples, cov_eval_future, alpha=0.90)
                new_weights = update_cell_weights(
                    coverage, cov_threshold=args.cov_threshold, max_weight=args.max_weight
                )
                cell_weights = new_weights
            print(f"[rebalance @ ep {epoch}] coverage range=[{coverage.min():.3f}, {coverage.max():.3f}]  "
                  f"weights range=[{cell_weights.min():.2f}, {cell_weights.max():.2f}]")

        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "main": 0.0, "vs": 0.0, "max_weight": 0.0}
        count = 0
        for history_01, future_01 in train_loader:
            trajectory = model(history_01, n_members=args.n_members, n_steps=args.n_steps)
            loss, metrics = compute_232d_loss(
                trajectory, future_01, cell_weights, args.lambda_vs, args.loss_type,
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
        val_sum = {"main": 0.0, "vs": 0.0}
        val_count = 0
        with torch.no_grad():
            for v_start in range(0, val_hist.shape[0], args.batch_size):
                v_end = min(v_start + args.batch_size, val_hist.shape[0])
                vh = val_hist[v_start:v_end]
                vf = val_future[v_start:v_end]
                vt = model(vh, n_members=args.n_members, n_steps=args.n_steps)
                _, vm = compute_232d_loss(
                    vt, vf, cell_weights, args.lambda_vs, args.loss_type,
                )
                vb = vh.shape[0]
                val_sum["main"] += vm["main"] * vb
                val_sum["vs"] += vm["vs"] * vb
                val_count += vb
        val_main = val_sum["main"] / val_count
        val_vs = val_sum["vs"] / val_count
        val_loss = val_main + args.lambda_vs * val_vs

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "elapsed": elapsed,
            "train_loss": train_metrics["loss"],
            "train_main": train_metrics["main"],
            "train_vs": train_metrics["vs"],
            "max_weight_epoch": train_metrics["max_weight"],
            "val_main": val_main, "val_vs": val_vs, "val_loss": val_loss,
            "cell_weights": cell_weights.cpu().tolist(),
        }
        history_log.append(make_serializable(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
                "cell_weights": cell_weights.cpu().tolist(),
            }, out_dir / "best_model.pt")
        if epoch % args.checkpoint_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "val_loss": val_loss,
                "cell_weights": cell_weights.cpu().tolist(),
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        print(
            f"[{epoch:3d}/{args.epochs}] loss={train_metrics['loss']:.4f} "
            f"main={train_metrics['main']:.4f} val={val_loss:.4f} "
            f"max_w={train_metrics['max_weight']:.2f}  ({elapsed:.1f}s)"
        )

    torch.save({
        "model_state_dict": model.state_dict(), "config": config,
        "epoch": args.epochs, "val_loss": val_loss,
        "cell_weights": cell_weights.cpu().tolist(),
    }, out_dir / "final_model.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nSaved to {out_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
