#!/usr/bin/env python
"""634a: rollout proper-score fine-tune for the native joint increment model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_conditioned_increment_flow_matching import (  # noqa: E402
    GenericStateConditionedIncrementFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.train_596a_final_path_joint_objective import (  # noqa: E402
    full_path_energy_score,
    horizon_path_weights,
    sliced_projection_wasserstein_score,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (
    _spec_to_dict,
)  # noqa: E402
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (
    build_blocks,
)  # noqa: E402
from experiments.backfill.block_ar.train_629a_state_conditioned_increment_flow import (  # noqa: E402
    sample_smoke,
    select_state_increment_scope,
)


def rollout_level_increment_scores_with_grad(
    model: GenericStateConditionedIncrementFlowMatching,
    history_level_values: torch.Tensor,
    history_increment_values: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a differentiable rollout in level-score and increment-score coordinates."""
    if n_steps < 1 or n_steps > model.cfg.future_len:
        raise ValueError(
            f"expected n_steps in [1,{model.cfg.future_len}], got {n_steps}"
        )
    k = int(n_samples)
    if k < 1:
        raise ValueError("n_samples must be positive")
    n_flow = max(1, int(flow_steps))
    dt = 1.0 / float(n_flow)
    temp = float(temperature)

    level_scores = model.level_values_to_scores(history_level_values)
    increment_scores = model.increment_values_to_scores(history_increment_values)
    bsz = int(level_scores.shape[0])
    prefix_level_scores = (
        level_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    prefix_increment_scores = (
        increment_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    prefix_level_values = (
        history_level_values.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )

    level_frames: list[torch.Tensor] = []
    increment_frames: list[torch.Tensor] = []
    for _step in range(int(n_steps)):
        memory_state = model._encode_prefix(
            prefix_level_scores, prefix_increment_scores
        )[:, -1]
        current_level_score = prefix_level_scores[:, -1]
        x = temp * torch.randn_like(current_level_score)
        source_scale = model._conditional_source_scale(memory_state)
        if source_scale is not None:
            x = x * source_scale
        for flow_step in range(n_flow):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=level_scores.device,
                dtype=level_scores.dtype,
            )
            x = x + dt * model.velocity(x, current_level_score, memory_state, t)
        next_increment_score = x
        next_increment_value = model.increment_scores_to_values(next_increment_score)
        next_level_value = prefix_level_values[:, -1] + next_increment_value
        next_level_score = model.level_values_to_scores(next_level_value)

        level_frames.append(next_level_score.view(bsz, k, model.cfg.n_cells))
        increment_frames.append(next_increment_score.view(bsz, k, model.cfg.n_cells))
        prefix_level_values = torch.cat(
            [prefix_level_values, next_level_value[:, None, :]], dim=1
        )
        prefix_level_scores = torch.cat(
            [prefix_level_scores, next_level_score[:, None, :]], dim=1
        )
        prefix_increment_scores = torch.cat(
            [prefix_increment_scores, next_increment_score[:, None, :]], dim=1
        )

    return torch.stack(level_frames, dim=2), torch.stack(increment_frames, dim=2)


def rollout_proper_score_loss(
    model: GenericStateConditionedIncrementFlowMatching,
    history_level_values: torch.Tensor,
    history_increment_values: torch.Tensor,
    future_level_values: torch.Tensor,
    future_increment_values: torch.Tensor,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    rollout_weight: float,
    sw_weight: float,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    n_projections: int,
    n_quantiles: int,
    energy_eps: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """FM-anchored proper score over generated level and increment paths."""
    fm_loss, fm_metrics = model.training_loss(
        history_level_values,
        history_increment_values,
        future_level_values,
        future_increment_values,
    )
    target_level_scores = model.level_values_to_scores(future_level_values)
    target_increment_scores = model.increment_values_to_scores(future_increment_values)
    sampled_level_scores, sampled_increment_scores = (
        rollout_level_increment_scores_with_grad(
            model,
            history_level_values,
            history_increment_values,
            n_samples=int(train_sample_count),
            n_steps=int(future_level_values.shape[1]),
            flow_steps=int(rollout_flow_steps),
            temperature=float(temperature),
        )
    )
    sampled_path = torch.cat([sampled_level_scores, sampled_increment_scores], dim=-1)
    target_path = torch.cat([target_level_scores, target_increment_scores], dim=-1)
    weights = horizon_path_weights(
        target_path.shape[1],
        end_weight=float(horizon_end_weight),
        device=target_path.device,
        dtype=target_path.dtype,
    )
    energy, target_dist, pair_dist = full_path_energy_score(
        sampled_path,
        target_path,
        eps=float(energy_eps),
        horizon_weights=weights,
    )
    sw = sliced_projection_wasserstein_score(
        sampled_path,
        target_path,
        n_projections=int(n_projections),
        n_quantiles=int(n_quantiles),
        horizon_weights=weights,
    )
    joint = energy + float(sw_weight) * sw
    total = float(fm_anchor_weight) * fm_loss + float(rollout_weight) * joint
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": energy.detach(),
        "sw": sw.detach(),
        "joint": joint.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "target_path_std": target_path.std(unbiased=False).detach(),
        "sample_path_std": sampled_path.std(unbiased=False).detach(),
        "target_level_h1_std": target_level_scores[:, 0].std(unbiased=False).detach(),
        "sample_level_h1_std": sampled_level_scores[:, :, 0]
        .std(unbiased=False)
        .detach(),
        "target_level_h30_std": target_level_scores[:, -1].std(unbiased=False).detach(),
        "sample_level_h30_std": sampled_level_scores[:, :, -1]
        .std(unbiased=False)
        .detach(),
        "target_increment_std": target_increment_scores.std(unbiased=False).detach(),
        "sample_increment_std": sampled_increment_scores.std(unbiased=False).detach(),
        "memory_abs": fm_metrics["memory_abs"].detach(),
    }
    return total, metrics


def run_epoch(
    model: GenericStateConditionedIncrementFlowMatching,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    train_sample_count: int,
    rollout_flow_steps: int,
    rollout_weight: float,
    sw_weight: float,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    n_projections: int,
    n_quantiles: int,
    energy_eps: float,
    temperature: float,
    clip_grad: float,
    max_batches: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    n_batches = 0
    for history_level, history_increment, future_level, future_increment in loader:
        if int(max_batches) > 0 and n_batches >= int(max_batches):
            break
        with torch.set_grad_enabled(train_mode):
            loss, metrics = rollout_proper_score_loss(
                model,
                history_level.to(device),
                history_increment.to(device),
                future_level.to(device),
                future_increment.to(device),
                train_sample_count=int(train_sample_count),
                rollout_flow_steps=int(rollout_flow_steps),
                rollout_weight=float(rollout_weight),
                sw_weight=float(sw_weight),
                fm_anchor_weight=float(fm_anchor_weight),
                horizon_end_weight=float(horizon_end_weight),
                n_projections=int(n_projections),
                n_quantiles=int(n_quantiles),
                energy_eps=float(energy_eps),
                temperature=float(temperature),
            )
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(clip_grad) > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                optimizer.step()
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/631a_joint38_statecond_increment_scale_e8_w2048_s631/best_model.pt",
    )
    parser.add_argument(
        "--state_scope", choices=["iv_only", "joint38"], default="joint38"
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--rollout_weight", type=float, default=0.03)
    parser.add_argument("--sw_weight", type=float, default=0.5)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--horizon_end_weight", type=float, default=1.5)
    parser.add_argument("--n_projections", type=int, default=16)
    parser.add_argument("--n_quantiles", type=int, default=16)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=32)
    parser.add_argument("--smoke_samples", type=int, default=4)
    parser.add_argument("--smoke_steps", type=int, default=8)
    parser.add_argument("--smoke_chunk_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=634)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8"
    )

    model, payload = load_model(args.checkpoint, device)
    cfg = payload["config"]
    if int(cfg["history_len"]) != int(args.history_len) or int(
        cfg["future_len"]
    ) != int(args.future_len):
        raise ValueError(
            "checkpoint horizon configuration does not match requested data"
        )
    if payload.get("state_scope", args.state_scope) != args.state_scope:
        raise ValueError("checkpoint state scope does not match requested state scope")

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    (
        train_level,
        train_increment,
        train_future_level,
        train_future_increment,
        _train_raw,
        train_specs,
    ) = select_state_increment_scope(train_block, args.state_scope, int(args.iv_count))
    (
        val_level,
        val_increment,
        val_future_level,
        val_future_increment,
        val_raw,
        val_specs,
    ) = select_state_increment_scope(
        val_block,
        args.state_scope,
        int(args.iv_count),
    )
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    actual = [spec.name for spec in train_specs]
    if expected and expected != actual:
        raise RuntimeError("checkpoint state specs do not match rebuilt training specs")
    if actual != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_level),
            torch.from_numpy(train_increment),
            torch.from_numpy(train_future_level),
            torch.from_numpy(train_future_increment),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_level),
            torch.from_numpy(val_increment),
            torch.from_numpy(val_future_level),
            torch.from_numpy(val_future_increment),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.epochs))
    )

    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": payload.get(
            "model_coordinate", "state_conditioned_encoded_increment"
        ),
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "objective": {
            "loss": "fm_anchor_plus_level_increment_path_energy_sliced_wasserstein",
            "rollout_weight": float(args.rollout_weight),
            "sw_weight": float(args.sw_weight),
            "fm_anchor_weight": float(args.fm_anchor_weight),
            "horizon_end_weight": float(args.horizon_end_weight),
            "train_sample_count": int(args.train_sample_count),
            "rollout_flow_steps": int(args.rollout_flow_steps),
            "n_projections": int(args.n_projections),
            "n_quantiles": int(args.n_quantiles),
        },
    }

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')} best_val: {payload.get('best_val')}")
    print(f"Train/val shapes: {train_level.shape} / {val_level.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    records: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_avg = run_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            rollout_weight=float(args.rollout_weight),
            sw_weight=float(args.sw_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
            horizon_end_weight=float(args.horizon_end_weight),
            n_projections=int(args.n_projections),
            n_quantiles=int(args.n_quantiles),
            energy_eps=float(args.energy_eps),
            temperature=float(args.sample_temperature),
            clip_grad=float(args.clip_grad),
            max_batches=int(args.max_train_batches),
        )
        val_avg = run_epoch(
            model,
            val_loader,
            None,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            rollout_weight=float(args.rollout_weight),
            sw_weight=float(args.sw_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
            horizon_end_weight=float(args.horizon_end_weight),
            n_projections=int(args.n_projections),
            n_quantiles=int(args.n_quantiles),
            energy_eps=float(args.energy_eps),
            temperature=float(args.sample_temperature),
            clip_grad=float(args.clip_grad),
            max_batches=int(args.max_val_batches),
        )
        scheduler.step()
        rec = {
            "epoch": int(epoch),
            **{f"train_{key}": value for key, value in train_avg.items()},
            **{f"val_{key}": value for key, value in val_avg.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - t0),
        }
        records.append(rec)
        print(json.dumps(make_serializable(rec), sort_keys=True), flush=True)
        if rec["val_total"] < best_val:
            best_val = float(rec["val_total"])
            best_epoch = int(epoch)
            save_checkpoint(
                str(out_dir / "best_model.pt"),
                model,
                model.cfg,
                epoch,
                best_val,
                extra=extra,
            )

    save_checkpoint(
        str(out_dir / "final_model.pt"),
        model,
        model.cfg,
        int(args.epochs),
        best_val,
        extra=extra,
    )
    smoke = sample_smoke(
        model,
        val_level[: min(8, int(val_level.shape[0]))],
        val_increment[: min(8, int(val_increment.shape[0]))],
        val_raw[: min(8, int(val_raw.shape[0]))],
        train_specs,
        samples=int(args.smoke_samples),
        steps=min(int(args.smoke_steps), int(args.future_len)),
        chunk_size=int(args.smoke_chunk_size),
        device=device,
        iv_count=min(int(args.iv_count), int(train_level.shape[-1])),
    )
    summary: dict[str, Any] = {
        "args": vars(args),
        "config": asdict(model.cfg),
        "state_scope": args.state_scope,
        "model_coordinate": payload.get(
            "model_coordinate", "state_conditioned_encoded_increment"
        ),
        "n_state_vars": int(train_level.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_level.shape),
        "val_shape": list(val_level.shape),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "objective": extra["objective"],
        "sample_smoke": smoke,
        "panel_metadata": panel_metadata,
        "output_dir": str(out_dir),
    }
    (out_dir / "training_history.json").write_text(
        json.dumps(make_serializable(records), indent=2),
        encoding="utf-8",
    )
    (out_dir / "train_summary.json").write_text(
        json.dumps(make_serializable(summary), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
