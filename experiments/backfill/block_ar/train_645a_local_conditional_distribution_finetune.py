#!/usr/bin/env python
"""645a: local conditional distribution-alignment finetune for 641a."""

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

from diffusion.block_ar.generic_state_conditioned_mixed_coordinate_flow_matching import (  # noqa: E402
    GenericStateConditionedMixedCoordinateFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    _spec_to_dict,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (
    build_blocks,
)  # noqa: E402
from experiments.backfill.block_ar.train_629a_state_conditioned_increment_flow import (  # noqa: E402
    sample_smoke,
    select_state_increment_scope,
)


def rollout_level_scores_with_grad(
    model: GenericStateConditionedMixedCoordinateFlowMatching,
    history_level_values: torch.Tensor,
    history_increment_values: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable 641a rollout returning generated future level scores."""
    if n_steps < 1 or n_steps > model.cfg.future_len:
        raise ValueError(
            f"expected n_steps in [1,{model.cfg.future_len}], got {n_steps}"
        )
    level_scores = model.level_values_to_scores(history_level_values)
    increment_scores = model.increment_values_to_scores(history_increment_values)
    bsz = int(level_scores.shape[0])
    k = int(n_samples)
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
    mask = model.level_score_mask.to(device=level_scores.device)[None, :]
    dt = 1.0 / float(flow_steps)
    frames: list[torch.Tensor] = []
    for _step in range(n_steps):
        memory_state = model._encode_prefix(
            prefix_level_scores, prefix_increment_scores
        )[:, -1]
        current_level_score = prefix_level_scores[:, -1]
        x = float(temperature) * torch.randn(
            bsz * k,
            model.cfg.n_cells,
            device=level_scores.device,
            dtype=level_scores.dtype,
        )
        source_scale = model._conditional_source_scale(memory_state)
        if source_scale is not None:
            x = x * source_scale
        for flow_step in range(int(flow_steps)):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=level_scores.device,
                dtype=level_scores.dtype,
            )
            x = x + dt * model.velocity(x, current_level_score, memory_state, t)

        level_next_score = current_level_score + x
        level_next_value = model._level_scores_to_values(level_next_score)
        increment_score_next = x
        increment_next_value = model.increment_scores_to_values(increment_score_next)
        increment_next_level = prefix_level_values[:, -1] + increment_next_value
        increment_next_level_score = model.level_values_to_scores(increment_next_level)

        next_level_value = torch.where(mask, level_next_value, increment_next_level)
        next_level_score = torch.where(
            mask, level_next_score, increment_next_level_score
        )
        next_increment_value = next_level_value - prefix_level_values[:, -1]
        next_increment_score = model.increment_values_to_scores(next_increment_value)

        frames.append(next_level_score.view(bsz, k, model.cfg.n_cells))
        prefix_level_values = torch.cat(
            [prefix_level_values, next_level_value[:, None, :]], dim=1
        )
        prefix_level_scores = torch.cat(
            [prefix_level_scores, next_level_score[:, None, :]], dim=1
        )
        prefix_increment_scores = torch.cat(
            [prefix_increment_scores, next_increment_score[:, None, :]],
            dim=1,
        )
    return torch.stack(frames, dim=2)


def horizon_weights(
    horizon: int,
    *,
    end_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if horizon == 1:
        return torch.ones(1, device=device, dtype=dtype)
    weights = torch.linspace(
        1.0, float(end_weight), horizon, device=device, dtype=dtype
    )
    return weights / weights.mean().clamp_min(1e-12)


def local_neighbor_indices(
    model: GenericStateConditionedMixedCoordinateFlowMatching,
    history_level_values: torch.Tensor,
    history_increment_values: torch.Tensor,
    *,
    k_neighbors: int,
    recent_steps: int,
) -> torch.Tensor:
    with torch.no_grad():
        level_scores = model.level_values_to_scores(history_level_values).detach()
        increment_scores = model.increment_values_to_scores(
            history_increment_values
        ).detach()
        recent = increment_scores[
            :, -min(int(recent_steps), increment_scores.shape[1]) :
        ]
        features = torch.cat(
            [
                level_scores[:, -1],
                recent.abs().mean(dim=1),
                recent.square().mean(dim=1),
            ],
            dim=-1,
        )
        features = (features - features.mean(dim=0, keepdim=True)) / features.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        dist = torch.cdist(features, features)
        k = max(1, min(int(k_neighbors), int(features.shape[0])))
        return torch.topk(dist, k=k, dim=1, largest=False).indices


def local_sliced_wasserstein_loss(
    generated_level_scores: torch.Tensor,
    target_level_scores: torch.Tensor,
    neighbor_idx: torch.Tensor,
    *,
    n_projections: int,
    n_quantiles: int,
    horizon_end_weight: float,
) -> torch.Tensor:
    if generated_level_scores.ndim != 4 or target_level_scores.ndim != 3:
        raise ValueError("expected generated [B,K,T,C] and target [B,T,C]")
    bsz, _samples, horizon, n_cells = generated_level_scores.shape
    weights = horizon_weights(
        horizon,
        end_weight=float(horizon_end_weight),
        device=generated_level_scores.device,
        dtype=generated_level_scores.dtype,
    )
    shape = (1, 1, horizon, 1)
    gen_flat = (generated_level_scores * weights.view(shape)).reshape(
        bsz, generated_level_scores.shape[1], -1
    )
    target_sets = target_level_scores[neighbor_idx]
    tgt_flat = (target_sets * weights.view(1, 1, horizon, 1)).reshape(
        bsz, target_sets.shape[1], -1
    )
    qs = torch.linspace(
        0.0,
        1.0,
        int(n_quantiles),
        device=generated_level_scores.device,
        dtype=generated_level_scores.dtype,
    )
    losses: list[torch.Tensor] = []
    for batch_idx in range(bsz):
        projections = torch.randn(
            int(n_projections),
            gen_flat.shape[-1],
            device=generated_level_scores.device,
            dtype=generated_level_scores.dtype,
        )
        projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        gen_proj = gen_flat[batch_idx] @ projections.T
        tgt_proj = tgt_flat[batch_idx] @ projections.T
        gen_q = torch.quantile(gen_proj, qs, dim=0)
        tgt_q = torch.quantile(tgt_proj, qs, dim=0)
        losses.append((gen_q - tgt_q).square().mean())
    return torch.sqrt(torch.stack(losses).mean() + 1e-12)


def local_distribution_loss(
    model: GenericStateConditionedMixedCoordinateFlowMatching,
    history_level: torch.Tensor,
    history_increment: torch.Tensor,
    future_level: torch.Tensor,
    future_increment: torch.Tensor,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    rollout_steps: int,
    sample_temperature: float,
    k_neighbors: int,
    recent_steps: int,
    local_weight: float,
    fm_anchor_weight: float,
    n_projections: int,
    n_quantiles: int,
    horizon_end_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(
        history_level,
        history_increment,
        future_level,
        future_increment,
    )
    target_level_scores = model.level_values_to_scores(
        future_level[:, : int(rollout_steps)]
    )
    generated_level_scores = rollout_level_scores_with_grad(
        model,
        history_level,
        history_increment,
        n_samples=int(train_sample_count),
        n_steps=int(rollout_steps),
        flow_steps=int(rollout_flow_steps),
        temperature=float(sample_temperature),
    )
    neighbor_idx = local_neighbor_indices(
        model,
        history_level,
        history_increment,
        k_neighbors=int(k_neighbors),
        recent_steps=int(recent_steps),
    )
    local_sw = local_sliced_wasserstein_loss(
        generated_level_scores,
        target_level_scores,
        neighbor_idx,
        n_projections=int(n_projections),
        n_quantiles=int(n_quantiles),
        horizon_end_weight=float(horizon_end_weight),
    )
    total = float(fm_anchor_weight) * fm_loss + float(local_weight) * local_sw
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "local_sw": local_sw.detach(),
        "sample_score_std": generated_level_scores.std(unbiased=False).detach(),
        "target_score_std": target_level_scores.std(unbiased=False).detach(),
        "neighbor_count": torch.tensor(
            float(neighbor_idx.shape[1]), device=history_level.device
        ),
    }
    if "memory_abs" in fm_metrics:
        metrics["memory_abs"] = fm_metrics["memory_abs"].detach()
    return total, metrics


def run_loss_epoch(
    model: GenericStateConditionedMixedCoordinateFlowMatching,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    count = 0
    for batch_idx, (
        history_level,
        history_increment,
        future_level,
        future_increment,
    ) in enumerate(loader):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        history_level = history_level.to(device)
        history_increment = history_increment.to(device)
        future_level = future_level.to(device)
        future_increment = future_increment.to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            loss, metrics = local_distribution_loss(
                model,
                history_level,
                history_increment,
                future_level,
                future_increment,
                train_sample_count=int(args.train_sample_count),
                rollout_flow_steps=int(args.rollout_flow_steps),
                rollout_steps=min(int(args.rollout_steps), int(model.cfg.future_len)),
                sample_temperature=float(args.train_sample_temperature),
                k_neighbors=int(args.k_neighbors),
                recent_steps=int(args.recent_steps),
                local_weight=float(args.local_weight),
                fm_anchor_weight=float(args.fm_anchor_weight),
                n_projections=int(args.n_projections),
                n_quantiles=int(args.n_quantiles),
                horizon_end_weight=float(args.horizon_end_weight),
            )
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
            optimizer.step()
        batch_n = int(history_level.shape[0])
        count += batch_n
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item()) * batch_n
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument(
        "--clean_nonpositive_log_levels", action="store_true", default=True
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=30)
    parser.add_argument("--train_sample_temperature", type=float, default=1.0)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--recent_steps", type=int, default=10)
    parser.add_argument("--local_weight", type=float, default=0.05)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--n_projections", type=int, default=16)
    parser.add_argument("--n_quantiles", type=int, default=8)
    parser.add_argument("--horizon_end_weight", type=float, default=2.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=16)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=645)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_model(args.checkpoint, device)
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
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")
    if int(train_level.shape[-1]) != int(model.cfg.n_cells):
        raise RuntimeError("checkpoint dimension does not match rebuilt state scope")

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
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    history_records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    extra = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": int(payload.get("epoch", -1)),
        "source_best_val": float(payload.get("best_val", float("nan"))),
        "state_scope": args.state_scope,
        "model_coordinate": "state_conditioned_mixed_coordinate",
        "finetune_objective": "local_conditional_distribution_alignment",
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
        "generated_coordinate_policy": payload.get("generated_coordinate_policy", {}),
    }
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_loss_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=int(args.max_train_batches),
            args=args,
        )
        val_metrics = run_loss_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            max_batches=int(args.max_val_batches),
            args=args,
        )
        val_total = float(val_metrics.get("total", float("inf")))
        record = {
            "epoch": int(epoch),
            "elapsed_s": float(time.time() - t0),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        history_records.append(record)
        is_best = val_total < best_val
        if is_best:
            best_val = val_total
            best_epoch = int(epoch)
            save_checkpoint(
                str(best_path), model, model.cfg, epoch, best_val, extra=extra
            )
        print(
            f"epoch {epoch:03d} train_total={train_metrics.get('total', float('nan')):.6f} "
            f"val_total={val_total:.6f}{' best' if is_best else ''}",
            flush=True,
        )

    final_path = output_dir / "final_model.pt"
    save_checkpoint(
        str(final_path), model, model.cfg, int(args.epochs), best_val, extra=extra
    )
    smoke = sample_smoke(
        model,
        val_level[: int(args.sample_windows)],
        val_increment[: int(args.sample_windows)],
        val_raw[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(model.cfg.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
    )
    summary = {
        "args": vars(args),
        "config": asdict(model.cfg),
        "state_scope": args.state_scope,
        "model_coordinate": "state_conditioned_mixed_coordinate",
        "finetune_objective": "local_conditional_distribution_alignment",
        "n_state_vars": int(model.cfg.n_cells),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_level.shape),
        "val_shape": list(val_level.shape),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "sample_smoke": smoke,
        "panel_metadata": panel_metadata,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(make_serializable(history_records), indent=2),
        encoding="utf-8",
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(make_serializable(summary), indent=2),
        encoding="utf-8",
    )
    (output_dir / "args.json").write_text(
        json.dumps(make_serializable(vars(args)), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
