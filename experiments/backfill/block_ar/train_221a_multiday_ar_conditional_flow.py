#!/usr/bin/env python
"""
221a: Multi-day AR conditional flow trained FROM SCRATCH on rollout.

Key difference from 220 series: the 220 experiments either froze a pre-trained
212ai or fine-tuned it with small rollout losses. This trains the same architecture
(GRU encoder + affine coupling flow + EWMA local-scale + asinh innovations) from
scratch with multi-horizon energy score as the PRIMARY objective.

Architecture: identical to 212ai/220d (RecurrentFlowTransitionModel)
Training: multi-horizon ES at h=1,5,15,30 + scheduled sampling + horizon curriculum
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
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    make_serializable,
    normalize_iv,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)
from experiments.backfill.block_ar.train_220d_recurrent_flow_transition import (
    RecurrentFlowTransitionModel,
)


def parse_curriculum(s: str) -> list[tuple[int, int]]:
    """Parse curriculum schedule string like '0:5,10:15,20:30' into [(epoch, horizon)]."""
    pairs = []
    for part in s.split(","):
        epoch_str, horizon_str = part.split(":")
        pairs.append((int(epoch_str), int(horizon_str)))
    return sorted(pairs, key=lambda x: x[0])


def get_curriculum_horizon(epoch: int, schedule: list[tuple[int, int]]) -> int:
    h = schedule[0][1]
    for ep, horizon in schedule:
        if epoch >= ep:
            h = horizon
    return h


def get_scheduled_sampling_prob(
    epoch: int,
    ss_start_epoch: int,
    ss_ramp_epochs: int,
) -> float:
    """Return probability of using model's own output instead of GT.

    Ramps from 0.0 (full teacher forcing) to 1.0 (full free-run).
    """
    if ss_ramp_epochs <= 0:
        return 0.0
    progress = max(0.0, (epoch - ss_start_epoch) / ss_ramp_epochs)
    return min(1.0, progress)


def multistep_ar_flow_loss(
    model: RecurrentFlowTransitionModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    train_samples: int,
    future_len: int,
    scoring_horizons: list[int],
    horizon_weights: list[float],
    free_run_prob: float,
    bptt_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = history_01.shape[0]
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(batch_size, future_01.shape[1], -1)

    state_stack, local_scale, prev = model.encode_history_with_state(history_01)

    total_loss = torch.tensor(0.0, device=history_01.device)
    n_scored = 0
    running_es = 0.0
    running_mae = 0.0
    running_v_std = 0.0
    running_floor = 0.0
    count_steps = 0

    for step in range(future_len):
        target_t = future_flat[:, step, :]
        target_v = torch.asinh(
            (target_t - prev) / local_scale.clamp_min(model.scale_floor)
        )

        v_samples, next_samples = model.sample_next_from_state(
            state_stack=state_stack,
            prev_01=prev,
            local_scale=local_scale,
            n_samples=train_samples,
        )

        h = step + 1
        if h in scoring_horizons:
            idx = scoring_horizons.index(h)
            w = horizon_weights[idx] if idx < len(horizon_weights) else 1.0
            es_t = energy_score(v_samples, target_v)
            total_loss = total_loss + w * es_t
            n_scored += 1
            running_es += float(es_t.detach())

        mean_next = next_samples.mean(dim=1)
        running_mae += float((mean_next - target_t).abs().mean().detach())
        running_v_std += float(v_samples.std(dim=1).mean().detach())
        running_floor += float((next_samples < 0.02).float().mean().detach())
        count_steps += 1

        # Scheduled sampling: use own output or GT for next conditioning
        use_own = (
            free_run_prob > 0.0
            and step < future_len - 1
            and torch.rand(()).item() < free_run_prob
        )
        next_frame = mean_next.detach() if use_own else target_t

        step_feat, next_scale = model._step_features(prev, next_frame, local_scale)
        state_stack = model.recurrent_step(step_feat, state_stack)
        local_scale = next_scale
        prev = next_frame

        # BPTT truncation
        if bptt_steps > 0 and (step + 1) % bptt_steps == 0:
            state_stack = state_stack.detach()
            local_scale = local_scale.detach()
            prev = prev.detach()

    if n_scored > 0:
        total_loss = total_loss / n_scored

    metrics = {
        "total_loss": total_loss.detach(),
        "mean_es": running_es / max(n_scored, 1),
        "mean_mae": running_mae / max(count_steps, 1),
        "mean_v_std": running_v_std / max(count_steps, 1),
        "mean_floor_rate": running_floor / max(count_steps, 1),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_rollout(
    model: RecurrentFlowTransitionModel,
    val_loader: DataLoader,
    future_len: int,
    eval_samples: int,
    scoring_horizons: list[int],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0

    for history_01, future_01 in val_loader:
        batch_size = history_01.shape[0]
        future_flat = future_01.reshape(batch_size, future_01.shape[1], -1)
        state_stack, local_scale, prev = model.encode_history_with_state(history_01)

        for step in range(future_len):
            target_t = future_flat[:, step, :]
            target_v = torch.asinh(
                (target_t - prev) / local_scale.clamp_min(model.scale_floor)
            )

            v_samples, next_samples = model.sample_next_from_state(
                state_stack=state_stack,
                prev_01=prev,
                local_scale=local_scale,
                n_samples=eval_samples,
            )

            h = step + 1
            if h in scoring_horizons:
                es = float(energy_score(v_samples, target_v).item())
                key = f"es_h{h}"
                totals[key] = totals.get(key, 0.0) + es * batch_size

                mean_next = next_samples.mean(dim=1)
                mae = float((mean_next - target_t).abs().mean().item())
                totals[f"mae_h{h}"] = totals.get(f"mae_h{h}", 0.0) + mae * batch_size

                # Coverage at 90%
                lower = next_samples.quantile(0.05, dim=1)
                upper = next_samples.quantile(0.95, dim=1)
                covered = ((target_t >= lower) & (target_t <= upper)).float().mean()
                totals[f"cov90_h{h}"] = totals.get(f"cov90_h{h}", 0.0) + float(covered) * batch_size

            # Teacher-forced: always use GT for next step in validation
            step_feat, next_scale = model._step_features(prev, target_t, local_scale)
            state_stack = model.recurrent_step(step_feat, state_stack)
            local_scale = next_scale
            prev = target_t

        total_count += batch_size

    return {f"val_{k}": v / max(total_count, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_free_rollout(
    model: RecurrentFlowTransitionModel,
    val_loader: DataLoader,
    future_len: int,
    eval_samples: int,
) -> dict[str, float]:
    """Free-run evaluation: model conditions on its own outputs (no GT)."""
    model.eval()
    totals: dict[str, float] = {}
    total_count = 0

    for history_01, future_01 in val_loader:
        batch_size = history_01.shape[0]
        future_flat = future_01.reshape(batch_size, future_01.shape[1], -1)
        state_stack, local_scale, prev = model.encode_history_with_state(history_01)

        for step in range(future_len):
            target_t = future_flat[:, step, :]

            _v, next_samples = model.sample_next_from_state(
                state_stack=state_stack,
                prev_01=prev,
                local_scale=local_scale,
                n_samples=eval_samples,
            )

            mean_next = next_samples.mean(dim=1)

            for h_check in [1, 5, 15, 30]:
                if step + 1 == h_check:
                    mae = float((mean_next - target_t).abs().mean().item())
                    totals[f"freerun_mae_h{h_check}"] = totals.get(f"freerun_mae_h{h_check}", 0.0) + mae * batch_size

                    lower = next_samples.quantile(0.05, dim=1)
                    upper = next_samples.quantile(0.95, dim=1)
                    covered = ((target_t >= lower) & (target_t <= upper)).float().mean()
                    totals[f"freerun_cov90_h{h_check}"] = (
                        totals.get(f"freerun_cov90_h{h_check}", 0.0) + float(covered) * batch_size
                    )

                    floor_rate = float((next_samples < 0.02).float().mean().item())
                    totals[f"freerun_floor_h{h_check}"] = (
                        totals.get(f"freerun_floor_h{h_check}", 0.0) + floor_rate * batch_size
                    )

            # Free-run: use model's own ensemble mean as next input
            step_feat, next_scale = model._step_features(prev, mean_next, local_scale)
            state_stack = model.recurrent_step(step_feat, state_stack)
            local_scale = next_scale
            prev = mean_next

        total_count += batch_size

    return {f"val_{k}": v / max(total_count, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="221a multi-day AR conditional flow from scratch")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=8)
    parser.add_argument("--eval_samples", type=int, default=32)
    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--flow_hidden", type=int, default=256)
    parser.add_argument("--n_coupling_layers", type=int, default=6)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--include_scale_feature", action="store_true", default=True)
    parser.add_argument("--no_scale_feature", action="store_false", dest="include_scale_feature")
    # Training
    parser.add_argument("--bptt_steps", type=int, default=5)
    parser.add_argument("--curriculum_schedule", type=str, default="0:5,10:15,20:30")
    parser.add_argument("--ss_start_epoch", type=int, default=5, help="Epoch to start scheduled sampling ramp")
    parser.add_argument("--ss_ramp_epochs", type=int, default=20, help="Epochs over which to ramp free-run prob 0→1")
    parser.add_argument("--scoring_horizons", type=str, default="1,5,15,30")
    parser.add_argument("--horizon_weights", type=str, default="0.4,0.2,0.2,0.2")
    # Support
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    # Eval
    parser.add_argument("--freerun_eval_interval", type=int, default=5,
                        help="Run free-run rollout evaluation every N epochs")
    parser.add_argument("--freerun_eval_limit", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curriculum = parse_curriculum(args.curriculum_schedule)
    scoring_horizons = [int(x) for x in args.scoring_horizons.split(",")]
    horizon_weights = [float(x) for x in args.horizon_weights.split(",")]

    # --- Data ---
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - args.future_len
    train_indices = np.arange(0, max_train_idx - args.val_size)[: args.max_train_windows]
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)[: args.max_val_windows]

    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, args.history_len, args.future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.future_len
    )

    train_loader = DataLoader(
        TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False
    )

    # Subset for free-run eval
    freerun_hist = val_hist[: args.freerun_eval_limit]
    freerun_future = val_future[: args.freerun_eval_limit]
    freerun_loader = DataLoader(
        TensorDataset(freerun_hist, freerun_future), batch_size=args.batch_size, shuffle=False
    )

    # --- Model: train from scratch ---
    history_feat_dim = 25 + 25 + (25 if args.include_scale_feature else 0)
    model = RecurrentFlowTransitionModel(
        n_cells=25,
        history_feat_dim=history_feat_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        flow_hidden=args.flow_hidden,
        n_coupling_layers=args.n_coupling_layers,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        include_scale_feature=args.include_scale_feature,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
    ).to(device)

    # Initialize GRUCells from GRU weights (they share the same parameter structure)
    model.init_recurrent_cells_from_gru()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"221a multi-day AR conditional flow — training from scratch")
    print(f"  params={n_params:,d}")
    print(f"  curriculum={curriculum}")
    print(f"  scoring_horizons={scoring_horizons}  weights={horizon_weights}")
    print(f"  ss_start={args.ss_start_epoch}  ss_ramp={args.ss_ramp_epochs}")
    print(f"  bptt_steps={args.bptt_steps}")
    print(f"  train_windows={train_hist.shape[0]}  val_windows={val_hist.shape[0]}")

    history_log: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0

    config = {
        "type": "multiday_ar_conditional_flow_221a",
        "n_cells": 25,
        "history_feat_dim": history_feat_dim,
        "hidden_dim": args.hidden_dim,
        "gru_layers": args.gru_layers,
        "gru_dropout": args.gru_dropout,
        "flow_hidden": args.flow_hidden,
        "n_coupling_layers": args.n_coupling_layers,
        "ewma_alpha": args.ewma_alpha,
        "scale_floor": args.scale_floor,
        "include_scale_feature": args.include_scale_feature,
        "support_lo": args.support_lo,
        "support_hi": args.support_hi,
        "history_len": args.history_len,
        "future_len": args.future_len,
        "curriculum_schedule": args.curriculum_schedule,
        "scoring_horizons": scoring_horizons,
        "horizon_weights": horizon_weights,
        "bptt_steps": args.bptt_steps,
        "ss_start_epoch": args.ss_start_epoch,
        "ss_ramp_epochs": args.ss_ramp_epochs,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_horizon = get_curriculum_horizon(epoch, curriculum)
        free_run_prob = get_scheduled_sampling_prob(epoch, args.ss_start_epoch, args.ss_ramp_epochs)
        active_horizons = [h for h in scoring_horizons if h <= curr_horizon]
        active_weights = horizon_weights[: len(active_horizons)]

        model.train()
        running: dict[str, float] = {}
        count = 0

        for history_01, future_01 in train_loader:
            loss, metrics = multistep_ar_flow_loss(
                model=model,
                history_01=history_01,
                future_01=future_01,
                train_samples=args.train_samples,
                future_len=curr_horizon,
                scoring_horizons=active_horizons,
                horizon_weights=active_weights,
                free_run_prob=free_run_prob,
                bptt_steps=args.bptt_steps,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = history_01.shape[0]
            count += bs
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v) * bs

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        scheduler.step()

        # Teacher-forced validation
        val_metrics = evaluate_rollout(
            model, val_loader,
            future_len=min(curr_horizon, args.future_len),
            eval_samples=args.eval_samples,
            scoring_horizons=active_horizons,
        )

        # Free-run validation (periodically)
        freerun_metrics: dict[str, float] = {}
        if epoch % args.freerun_eval_interval == 0 or epoch == args.epochs:
            freerun_metrics = evaluate_free_rollout(
                model, freerun_loader,
                future_len=min(curr_horizon, args.future_len),
                eval_samples=args.eval_samples,
            )

        # Selection score: balanced across horizons and free-run quality
        score_parts = []
        for h in active_horizons:
            es_key = f"val_es_h{h}"
            cov_key = f"val_cov90_h{h}"
            if es_key in val_metrics:
                score_parts.append(val_metrics[es_key])
            if cov_key in val_metrics:
                score_parts.append(max(0.0, 0.80 - val_metrics[cov_key]))
        # Add free-run coverage penalty if available
        for h in [1, 15, 30]:
            fr_key = f"val_freerun_cov90_h{h}"
            if fr_key in freerun_metrics:
                score_parts.append(max(0.0, 0.70 - freerun_metrics[fr_key]))
            fr_floor_key = f"val_freerun_floor_h{h}"
            if fr_floor_key in freerun_metrics:
                score_parts.append(freerun_metrics[fr_floor_key])
        score = sum(score_parts) / max(len(score_parts), 1)

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            "curriculum_horizon": curr_horizon,
            "free_run_prob": free_run_prob,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_metrics,
            **val_metrics,
            **freerun_metrics,
            "selection_score": float(score),
        }
        history_log.append(make_serializable(record))

        # Print summary
        es_str = " ".join(
            f"es_h{h}={val_metrics.get(f'val_es_h{h}', 0):.4f}"
            for h in active_horizons
        )
        cov_str = " ".join(
            f"cov_h{h}={val_metrics.get(f'val_cov90_h{h}', 0):.1%}"
            for h in active_horizons
        )
        fr_str = ""
        if freerun_metrics:
            fr_str = " | FR: " + " ".join(
                f"h{h}={freerun_metrics.get(f'val_freerun_cov90_h{h}', 0):.1%}"
                for h in [1, 15, 30] if f"val_freerun_cov90_h{h}" in freerun_metrics
            )
        print(
            f"  [{epoch:3d}/{args.epochs}] T={curr_horizon:2d} p_free={free_run_prob:.2f} "
            f"| {es_str} | {cov_str}{fr_str} | score={score:.4f} "
            f"({time.time() - t0:.1f}s)"
        )

        # Save checkpoints
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": history_log[-1],
        }
        torch.save(payload, out_dir / "last_model.pt")

        if score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(payload, out_dir / "best_model.pt")
            print(f"    *** new best at epoch {epoch} (score={score:.4f})")

        if epoch == args.epochs:
            torch.save(payload, out_dir / "final_model.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\nDone. Best epoch={best_epoch} score={best_score:.4f}")
    print(f"Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
