#!/usr/bin/env python
"""
224a: Self-Forcing NLL — Conditional NLL through differentiable rollout.

Core idea: roll out K steps with gradient, then compute NLL of the real
next day given the augmented history containing K generated days. The
gradient flows through the entire rollout chain, teaching the model to
produce outputs that lead to correct subsequent conditionals.

This synthesizes:
- 222's differentiable rollout (gradient reaches model)
- 223's conditional NLL (enforces conditionality, prevents bootstrap)
- GraphCast/Self-Forcing curriculum (manages supervision mismatch)
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
    evaluate_h1,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
)
from experiments.backfill.block_ar.train_212af_h1_conditional_flow_local_scale_asinh_nll import (
    ConditionalFlowLocalScaleAsinhNLLModel,
)
from experiments.backfill.block_ar.train_222a_distributional_invariant_finetuning import (
    energy_distance_1d_percell,
)


def parse_curriculum(schedule_str: str) -> list[tuple[int, int]]:
    """Parse 'epoch:K,epoch:K,...' into sorted list of (epoch, K) pairs."""
    pairs: list[tuple[int, int]] = []
    for part in schedule_str.split(","):
        epoch_s, k_s = part.strip().split(":")
        pairs.append((int(epoch_s), int(k_s)))
    return sorted(pairs)


def get_curriculum_k(epoch: int, schedule: list[tuple[int, int]]) -> int:
    """Get current rollout steps K for this epoch."""
    k = schedule[0][1]
    for ep, kk in schedule:
        if epoch >= ep:
            k = kk
    return k


def self_forcing_rollout_nll(
    model: ConditionalFlowLocalScaleAsinhNLLModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    K: int,
    n_paths: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Roll out K steps with gradient, compute NLL on real target at step K+1.

    Args:
        model: 212ai-class model (sample_next_iv is differentiable)
        history_01: (B, 30, 5, 5) real history in [0,1]
        future_01: (B, future_len, 5, 5) real future days
        K: number of rollout steps before NLL evaluation
        n_paths: independent rollout paths per window (variance reduction)

    Returns:
        nll: scalar loss
        metrics: dict with diagnostic values
    """
    B = history_01.shape[0]
    n_cells = model.n_cells

    # Expand to n_paths independent rollouts
    hist = (
        history_01.unsqueeze(1)
        .expand(B, n_paths, -1, 5, 5)
        .reshape(B * n_paths, history_01.shape[1], 5, 5)
        .clone()
    )

    # Roll out K steps with full gradient (no BPTT truncation for K<=5)
    rollout_deltas = []
    levels_list = [hist[:, -1].reshape(B * n_paths, n_cells)]  # h=0: last real day
    for t in range(K):
        next_iv = model.sample_next_iv(hist, n_samples=1).squeeze(1)  # (B*n_paths, 25)
        prev_step = hist[:, -1].reshape(B * n_paths, n_cells)
        rollout_deltas.append((next_iv - prev_step).abs().mean().item())
        levels_list.append(next_iv)
        next_frame = next_iv.view(B * n_paths, 1, 5, 5)
        hist = torch.cat([hist[:, 1:], next_frame], dim=1)

    # Target: the real day at position K in future (0-indexed)
    # future_01[:, 0] = day 31 (first future day)
    # future_01[:, K] = day 31+K (target after K rollout steps)
    target = future_01[:, K].reshape(B, n_cells)  # (B, 25)
    target_exp = (
        target.unsqueeze(1)
        .expand(B, n_paths, n_cells)
        .reshape(B * n_paths, n_cells)
    )

    # Compute NLL of target given augmented history
    prev = hist[:, -1].reshape(B * n_paths, n_cells)
    target_delta = target_exp - prev
    _, local_scale = model.encode_with_scale(hist)
    local_scale = local_scale.clamp_min(model.scale_floor)
    target_v = torch.asinh(target_delta / local_scale)
    log_prob = model.log_prob_transformed_innovation(hist, target_v)
    nll = -log_prob.mean() / n_cells

    # Stack levels for optional change ED: (B*n_paths, K+1, 25)
    rollout_levels = torch.stack(levels_list, dim=1)

    metrics = {
        "sf_nll": float(nll.detach().item()),
        "sf_mean_delta": float(np.mean(rollout_deltas)) if rollout_deltas else 0.0,
        "sf_target_delta_mean": float(target_delta.abs().mean().item()),
        "sf_local_scale_mean": float(local_scale.mean().item()),
    }
    return nll, metrics, rollout_levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="224a Self-Forcing NLL: conditional NLL through differentiable rollout"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--nll_weight", type=float, default=0.05)
    parser.add_argument("--sf_weight", type=float, default=1.0)
    parser.add_argument("--sf_ramp_epochs", type=int, default=2)
    parser.add_argument("--n_paths", type=int, default=4)
    parser.add_argument("--curriculum", type=str, default="1:1",
                        help="Comma-separated epoch:K pairs, e.g. '1:1,4:2,8:3'")
    parser.add_argument("--future_len", type=int, default=2,
                        help="Number of future days to load (must be >= max K + 1)")
    parser.add_argument("--lambda_change", type=float, default=0.0,
                        help="GT-anchored change ED weight (0=disabled, -1=auto-balance)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    init_payload = torch.load(
        args.init_checkpoint, map_location=device, weights_only=False
    )
    cfg = init_payload["config"]
    model = ConditionalFlowLocalScaleAsinhNLLModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        flow_hidden=cfg["flow_hidden"],
        n_coupling_layers=cfg["n_coupling_layers"],
        ewma_alpha=cfg["ewma_alpha"],
        scale_floor=cfg["scale_floor"],
        include_scale_feature=cfg["include_scale_feature"],
    ).to(device)
    model.load_state_dict(init_payload["model_state_dict"])

    # Load data
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    # Build windows
    max_train_idx = args.test_start - args.history_len - args.future_len
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices_ms = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    val_indices_ms = val_indices_ms[: args.max_val_windows]

    # Training: multi-step windows (history + future_len days)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, args.future_len
    )

    # Validation: one-step windows for h1 eval
    val_max_idx = args.test_start - args.history_len - 1
    val_indices_1s = np.arange(val_max_idx - args.val_size, val_max_idx)
    val_indices_1s = val_indices_1s[: args.max_val_windows]
    val_hist, val_target = build_one_step_windows(
        val_indices_1s, surf_tensor, args.history_len
    )

    train_delta_np = np.diff(
        surfaces[: args.test_start].reshape(-1, 25), axis=0
    )
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # GT change pool for optional change ED
    gt_change_pool = None
    lambda_change = args.lambda_change
    if lambda_change != 0.0:
        gt_change_pool = torch.from_numpy(
            np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0).astype(np.float32)
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = parse_curriculum(args.curriculum)

    print("224a Self-Forcing NLL")
    print(f"  init={args.init_checkpoint}")
    print(f"  curriculum={args.curriculum}")
    print(f"  sf_weight={args.sf_weight} sf_ramp_epochs={args.sf_ramp_epochs}")
    print(f"  n_paths={args.n_paths} future_len={args.future_len}")
    print(f"  lambda_change={lambda_change}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    payload: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_K = get_curriculum_k(epoch, schedule)
        sf_ramp = min(1.0, epoch / max(args.sf_ramp_epochs, 1))
        effective_sf_weight = args.sf_weight * sf_ramp

        model.train()
        running = {
            "loss": 0.0,
            "clean_loss": 0.0,
            "sf_nll": 0.0,
            "sf_mean_delta": 0.0,
            "change_ed": 0.0,
        }
        count = 0

        for history_01, future_01 in train_loader:
            B = history_01.shape[0]

            # Clean branch: standard 212ai loss on real data
            clean_target = future_01[:, 0].reshape(B, 5, 5).reshape(B, model.n_cells)
            clean_loss, _ = model.training_loss(
                history_01,
                clean_target,
                n_samples=args.train_samples,
                nll_weight=args.nll_weight,
            )

            # Self-forcing branch: NLL through differentiable rollout
            sf_nll, sf_metrics, rollout_levels = self_forcing_rollout_nll(
                model=model,
                history_01=history_01,
                future_01=future_01,
                K=curr_K,
                n_paths=args.n_paths,
            )

            loss = clean_loss + effective_sf_weight * sf_nll

            # Optional change ED on shared rollout trajectory
            change_ed_val = 0.0
            if gt_change_pool is not None and lambda_change != 0.0:
                # Changes from rollout: (B*n_paths, K, 25)
                changes = rollout_levels[:, 1:] - rollout_levels[:, :-1]
                # Pool all K changes together for more samples
                pooled_changes = changes.reshape(-1, model.n_cells)  # (B*n_paths*K, 25)
                # Subsample GT to match
                gt_idx = torch.randint(0, gt_change_pool.shape[0],
                                       (pooled_changes.shape[0],), device=device)
                gt_sub = gt_change_pool[gt_idx]
                change_ed = energy_distance_1d_percell(pooled_changes, gt_sub)
                change_ed_val = float(change_ed.detach().item())
                # Auto-balance on first batch of first epoch
                if lambda_change < 0 and count == 0:
                    ratio = float(sf_nll.detach().item()) / max(float(change_ed.detach().item()), 1e-8)
                    lambda_change = ratio
                    print(f"  Auto-balanced lambda_change = {lambda_change:.1f}")
                loss = loss + lambda_change * change_ed

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = B
            running["loss"] += float(loss.item()) * batch
            running["clean_loss"] += float(clean_loss.item()) * batch
            running["sf_nll"] += sf_metrics["sf_nll"] * batch
            running["sf_mean_delta"] += sf_metrics["sf_mean_delta"] * batch
            running["change_ed"] += change_ed_val * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}

        model.eval()
        val_metrics = evaluate_h1(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            eval_samples=args.eval_samples,
        )

        score = (
            max(0.0, 0.85 - val_metrics["val_coverage_90"])
            + max(0.0, 0.58 - val_metrics["val_realized_q99_coverage_90"])
            + max(0.0, 0.85 - val_metrics["val_h1_quiet_ratio"])
            + max(0.0, val_metrics["val_h1_shoulder_ratio"] - 1.10)
            + max(0.0, 0.65 - val_metrics["val_h1_kurtosis_ratio"])
        )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "curriculum_K": curr_K,
            "sf_ramp": sf_ramp,
            "effective_sf_weight": effective_sf_weight,
            **train_metrics,
            **val_metrics,
            "selection_score": float(score),
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_conditional_flow_223_generated_history",
                "n_cells": cfg["n_cells"],
                "history_feat_dim": cfg["history_feat_dim"],
                "hidden_dim": cfg["hidden_dim"],
                "gru_layers": cfg["gru_layers"],
                "gru_dropout": cfg["gru_dropout"],
                "flow_hidden": cfg["flow_hidden"],
                "n_coupling_layers": cfg["n_coupling_layers"],
                "history_len": cfg["history_len"],
                "ewma_alpha": cfg["ewma_alpha"],
                "scale_floor": cfg["scale_floor"],
                "include_scale_feature": cfg["include_scale_feature"],
                "nll_weight": args.nll_weight,
                "sf_weight": args.sf_weight,
                "sf_ramp_epochs": args.sf_ramp_epochs,
                "n_paths": args.n_paths,
                "curriculum": args.curriculum,
                "future_len": args.future_len,
                "init_checkpoint": args.init_checkpoint,
            },
            "metrics": history[-1],
            "thresholds": {"q95": q95_threshold, "q99": q99_threshold},
        }
        torch.save(payload, out_dir / "last_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        chg_str = f" chgED={train_metrics['train_change_ed']:.5f}" if lambda_change != 0.0 else ""
        print(
            f"[{epoch:02d}/{args.epochs}] K={curr_K} sf_w={effective_sf_weight:.2f} "
            f"loss={train_metrics['train_loss']:.4f} "
            f"clean={train_metrics['train_clean_loss']:.4f} "
            f"sf_nll={train_metrics['train_sf_nll']:.4f} "
            f"sf_delta={train_metrics['train_sf_mean_delta']:.5f}{chg_str} "
            f"cov90={val_metrics['val_coverage_90']:.3f} "
            f"kurt={val_metrics['val_h1_kurtosis_ratio']:.3f} "
            f"score={score:.3f}"
        )

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

    torch.save(payload, out_dir / "final_model.pt")
    print(f"Done. Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
