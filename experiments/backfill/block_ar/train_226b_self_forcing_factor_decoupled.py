#!/usr/bin/env python
"""
226b: Self-Forcing on factor-decoupled flow (226a).

Roll out K steps with gradient through 226a's sample_next_iv. At the endpoint,
compute ES + VS against the real target. Gradient flows through the entire
rollout chain, teaching the model to produce outputs that lead to correct
subsequent conditionals.

Loss = h1_weight * (ES + VS)(at h=1) + sf_weight * (ES + VS)(at rollout endpoint)

The h=1 anchor preserves one-step marginal quality. The self-forcing term
teaches multi-step dynamics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
    evaluate_h1,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    variogram_score,
)
from experiments.backfill.block_ar.train_226a_factor_decoupled_flow import (
    FactorDecoupledFlowModel,
    load_model as load_226a_model,
)


def parse_curriculum(schedule_str: str) -> list[tuple[int, int]]:
    """Parse 'epoch:K,epoch:K,...' into sorted list of (epoch, K) pairs."""
    pairs = []
    for part in schedule_str.split(","):
        epoch_s, k_s = part.strip().split(":")
        pairs.append((int(epoch_s), int(k_s)))
    return sorted(pairs)


def get_curriculum_k(epoch: int, schedule: list[tuple[int, int]]) -> int:
    k = schedule[0][1]
    for ep, kk in schedule:
        if epoch >= ep:
            k = kk
    return k


def self_forcing_es_vs(
    model: FactorDecoupledFlowModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    K: int,
    n_paths: int = 4,
    n_endpoint_samples: int = 32,
    lambda_vs: float = 0.03,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Roll out K steps with gradient, compute ES+VS at endpoint.

    Args:
        model: 226a factor-decoupled flow model
        history_01: (B, 30, 5, 5) real history
        future_01: (B, future_len, 5, 5) real future days
        K: rollout steps before evaluation
        n_paths: independent rollout paths per window
        n_endpoint_samples: samples at endpoint for ES+VS
        lambda_vs: variogram score weight
    """
    B = history_01.shape[0]
    n_cells = model.n_cells

    # Expand for independent rollout paths
    hist = (
        history_01.unsqueeze(1)
        .expand(B, n_paths, -1, 5, 5)
        .reshape(B * n_paths, history_01.shape[1], 5, 5)
        .clone()
    )

    # Roll out K steps with full gradient
    rollout_deltas = []
    for t in range(K):
        next_iv = model.sample_next_iv(hist, n_samples=1).squeeze(1)  # (B*n_paths, 25)
        prev_step = hist[:, -1].reshape(B * n_paths, n_cells)
        rollout_deltas.append((next_iv - prev_step).abs().mean().item())
        next_frame = next_iv.view(B * n_paths, 1, 5, 5)
        hist = torch.cat([hist[:, 1:], next_frame], dim=1)

    # Target: real day at position K in future
    target = future_01[:, K].reshape(B, n_cells)  # (B, 25)
    target_exp = (
        target.unsqueeze(1)
        .expand(B, n_paths, n_cells)
        .reshape(B * n_paths, n_cells)
    )

    # Sample from model at endpoint
    v_samples, local_scale, prev = model.sample_transformed_innovation(
        hist, n_samples=n_endpoint_samples
    )
    target_delta = target_exp - prev
    target_v = torch.asinh(target_delta / local_scale.clamp_min(model.scale_floor))

    # ES + VS at endpoint
    es = energy_score(v_samples, target_v)
    vs = variogram_score(v_samples, target_v, p=0.5)
    loss = es + lambda_vs * vs

    metrics = {
        "sf_es": float(es.detach().item()),
        "sf_vs": float(vs.detach().item()),
        "sf_loss": float(loss.detach().item()),
        "sf_mean_delta": float(np.mean(rollout_deltas)) if rollout_deltas else 0.0,
    }
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="226b: Self-Forcing on factor-decoupled flow"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True,
                        help="226a checkpoint to fine-tune")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=64,
                        help="Samples for h=1 ES+VS anchor")
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--h1_weight", type=float, default=1.0,
                        help="Weight for h=1 ES+VS anchor loss")
    parser.add_argument("--sf_weight", type=float, default=1.0,
                        help="Weight for self-forcing loss at rollout endpoint")
    parser.add_argument("--lambda_vs", type=float, default=0.03)
    parser.add_argument("--n_paths", type=int, default=4)
    parser.add_argument("--n_endpoint_samples", type=int, default=32)
    parser.add_argument("--curriculum", type=str, default="1:1,5:2,10:3",
                        help="Comma-separated epoch:K pairs")
    parser.add_argument("--future_len", type=int, default=5,
                        help="Future days to load (must be >= max K + 1)")
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

    schedule = parse_curriculum(args.curriculum)
    max_K = max(k for _, k in schedule)
    assert args.future_len >= max_K + 1, (
        f"future_len={args.future_len} must be >= max_K+1={max_K+1}"
    )

    # Load 226a model
    model, init_payload = load_226a_model(args.init_checkpoint, device)
    model.train()
    print(f"Loaded 226a from {args.init_checkpoint}")
    print(f"  epoch={init_payload.get('epoch', '?')}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Data: multi-step windows for self-forcing
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[:args.max_train_windows]
    val_indices = val_indices[:args.max_val_windows]

    # Multi-step windows: (B, history_len, 5, 5) + (B, future_len, 5, 5)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, args.future_len
    )
    train_future = train_future.view(train_hist.shape[0], args.future_len, 5, 5)

    # H=1 windows for anchor loss
    h1_train_hist, h1_train_target = build_one_step_windows(
        train_indices, surf_tensor, args.history_len
    )
    val_hist, val_target = build_one_step_windows(
        val_indices, surf_tensor, args.history_len
    )

    train_delta_np = np.diff(surfaces[:args.test_start].reshape(-1, 25), axis=0)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    # Combine h=1 and multi-step into one dataset
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future, h1_train_target),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_target),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    config = {
        **init_payload.get("config", {}),
        "type": "factor_decoupled_flow_226b_self_forcing",
        "init_checkpoint": args.init_checkpoint,
        "curriculum": args.curriculum,
        "h1_weight": args.h1_weight,
        "sf_weight": args.sf_weight,
        "lambda_vs": args.lambda_vs,
        "n_paths": args.n_paths,
        "lr": args.lr,
    }

    print(f"\n226b Self-Forcing on Factor-Decoupled Flow")
    print(f"  curriculum: {args.curriculum}")
    print(f"  h1_weight={args.h1_weight}, sf_weight={args.sf_weight}, lambda_vs={args.lambda_vs}")
    print(f"  n_paths={args.n_paths}, endpoint_samples={args.n_endpoint_samples}")
    print(f"  lr={args.lr}, batch_size={args.batch_size}")

    history_log: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        K = get_curriculum_k(epoch, schedule)
        model.train()

        running = {
            "loss": 0.0, "h1_loss": 0.0, "sf_loss": 0.0,
            "h1_es": 0.0, "h1_vs": 0.0, "sf_es": 0.0, "sf_vs": 0.0,
        }
        count = 0

        for history_01, future_01, h1_target_01 in train_loader:
            # H=1 anchor: ES + VS on one-step prediction
            h1_loss, h1_metrics = model.training_loss(
                history_01, h1_target_01,
                n_samples=args.train_samples,
                lambda_vs=args.lambda_vs,
            )

            # Self-forcing: roll out K steps, ES+VS at endpoint
            sf_loss, sf_metrics = self_forcing_es_vs(
                model, history_01, future_01,
                K=K,
                n_paths=args.n_paths,
                n_endpoint_samples=args.n_endpoint_samples,
                lambda_vs=args.lambda_vs,
            )

            loss = args.h1_weight * h1_loss + args.sf_weight * sf_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b = history_01.shape[0]
            running["loss"] += loss.item() * b
            running["h1_loss"] += h1_loss.item() * b
            running["sf_loss"] += sf_loss.item() * b
            running["h1_es"] += h1_metrics["energy"].item() * b
            running["h1_vs"] += h1_metrics["variogram"].item() * b
            running["sf_es"] += sf_metrics["sf_es"] * b
            running["sf_vs"] += sf_metrics["sf_vs"] * b
            count += b

        train_metrics = {k: v / count for k, v in running.items()}

        # Validation (h=1 only)
        with torch.no_grad():
            val_metrics = evaluate_h1(
                model, val_loader, q95_threshold, q99_threshold, args.eval_samples
            )

        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "K": K, "elapsed": elapsed,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics,
        }
        history_log.append(make_serializable(record))

        # Model selection
        score = (
            max(0.0, 0.85 - val_metrics.get("val_coverage_90", 0.0))
            + max(0.0, 0.58 - val_metrics.get("val_realized_q99_coverage_90", 0.0))
            + max(0.0, 0.85 - val_metrics.get("val_h1_quiet_ratio", 0.0))
            + max(0.0, val_metrics.get("val_h1_shoulder_ratio", 1.0) - 1.10)
            + max(0.0, 0.65 - val_metrics.get("val_h1_kurtosis_ratio", 0.0))
        )

        if score < best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "score": score,
                "val_metrics": make_serializable(val_metrics),
            }, out_dir / "best_model.pt")

        # Save every 3 epochs for checkpoint sweep
        if epoch % 3 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_metrics": make_serializable(val_metrics),
            }, out_dir / f"checkpoint_ep{epoch}.pt")

        cov = val_metrics.get("val_coverage_90", 0.0)
        print(
            f"[{epoch:3d}/{args.epochs}] K={K} "
            f"loss={train_metrics['loss']:.4f} "
            f"h1={train_metrics['h1_loss']:.4f} sf={train_metrics['sf_loss']:.4f} "
            f"cov90={cov:.3f} score={score:.4f} "
            f"({elapsed:.1f}s)"
        )

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epoch": args.epochs,
        "val_metrics": make_serializable(val_metrics),
    }, out_dir / "final_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
