#!/usr/bin/env python
"""666a: sampled-rollout energy fine-tune for normalized-innovation AR flows."""

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

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    GenericStateAwareNormalizedInnovationFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.train_596a_final_path_joint_objective import (  # noqa: E402
    full_path_energy_score,
    horizon_path_weights,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)
from experiments.backfill.block_ar.train_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    sample_smoke,
    select_normalized_innovation_scope,
)


def differentiable_rollout_paths(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable free-running sampler returning normalized innovations and level paths."""
    if n_steps < 1 or n_steps > model.cfg.future_len:
        raise ValueError(f"expected n_steps in [1,{model.cfg.future_len}], got {n_steps}")
    level_scores = model.level_values_to_scores(history_level_values)
    bsz = int(level_scores.shape[0])
    k = int(n_samples)
    prefix_level_values = (
        history_level_values.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    prefix_level_scores = (
        level_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    prefix_norm = (
        history_normalized_innovation.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    center_rep = center.unsqueeze(1).expand(bsz, k, model.cfg.n_cells).reshape(bsz * k, model.cfg.n_cells)
    scale_rep = scale.unsqueeze(1).expand(bsz, k, model.cfg.n_cells).reshape(bsz * k, model.cfg.n_cells)
    dt = 1.0 / float(max(1, int(flow_steps)))
    norm_frames: list[torch.Tensor] = []
    level_frames: list[torch.Tensor] = []
    for _step in range(int(n_steps)):
        memory_state = model._encode_prefix(prefix_level_scores, prefix_norm, center_rep, scale_rep)[:, -1]
        current_level_score = prefix_level_scores[:, -1]
        x = float(temperature) * torch.randn(
            bsz * k,
            model.cfg.n_cells,
            device=history_level_values.device,
            dtype=history_level_values.dtype,
        )
        for flow_step in range(max(1, int(flow_steps))):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=history_level_values.device,
                dtype=history_level_values.dtype,
            )
            x = x + dt * model.velocity(x, current_level_score, memory_state, t)
        next_norm = x
        next_increment = next_norm * scale_rep + center_rep
        next_level_value = prefix_level_values[:, -1] + next_increment
        next_level_score = model.level_values_to_scores(next_level_value)
        norm_frames.append(next_norm.view(bsz, k, model.cfg.n_cells))
        level_frames.append(next_level_value.view(bsz, k, model.cfg.n_cells))
        prefix_level_values = torch.cat([prefix_level_values, next_level_value[:, None, :]], dim=1)
        prefix_level_scores = torch.cat([prefix_level_scores, next_level_score[:, None, :]], dim=1)
        prefix_norm = torch.cat([prefix_norm, next_norm[:, None, :]], dim=1)
    return torch.stack(norm_frames, dim=2), torch.stack(level_frames, dim=2)


def differentiable_normalized_rollout_samples(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable free-running sampler returning normalized innovations."""
    sampled_norm, _sampled_level = differentiable_rollout_paths(
        model,
        history_level_values,
        history_normalized_innovation,
        center,
        scale,
        n_samples=int(n_samples),
        n_steps=int(n_steps),
        flow_steps=int(flow_steps),
        temperature=float(temperature),
    )
    return sampled_norm


def normalized_rollout_energy_loss(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    history_level_values: torch.Tensor,
    history_normalized_innovation: torch.Tensor,
    future_level_values: torch.Tensor,
    future_normalized_innovation: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_weight: float,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    energy_eps: float,
    temperature: float,
    level_energy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(
        history_level_values,
        history_normalized_innovation,
        future_level_values,
        future_normalized_innovation,
        center,
        scale,
    )
    sampled_norm, sampled_level = differentiable_rollout_paths(
        model,
        history_level_values,
        history_normalized_innovation,
        center,
        scale,
        n_samples=int(train_sample_count),
        n_steps=int(future_normalized_innovation.shape[1]),
        flow_steps=int(rollout_flow_steps),
        temperature=float(temperature),
    )
    weights = horizon_path_weights(
        int(future_normalized_innovation.shape[1]),
        end_weight=float(horizon_end_weight),
        device=future_normalized_innovation.device,
        dtype=future_normalized_innovation.dtype,
    )
    energy, target_dist, pair_dist = full_path_energy_score(
        sampled_norm,
        future_normalized_innovation,
        eps=float(energy_eps),
        horizon_weights=weights,
    )
    if float(level_energy_weight) > 0.0:
        level_energy, level_target_dist, level_pair_dist = full_path_energy_score(
            sampled_level,
            future_level_values,
            eps=float(energy_eps),
            horizon_weights=weights,
        )
    else:
        level_energy = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
        level_target_dist = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
        level_pair_dist = torch.zeros((), device=fm_loss.device, dtype=fm_loss.dtype)
    total = (
        float(fm_anchor_weight) * fm_loss
        + float(energy_weight) * energy
        + float(level_energy_weight) * level_energy
    )
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": energy.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "level_energy": level_energy.detach(),
        "level_energy_target_dist": level_target_dist.detach(),
        "level_energy_pair_dist": level_pair_dist.detach(),
        "target_norm_std": future_normalized_innovation.std(unbiased=False).detach(),
        "sample_norm_std": sampled_norm.std(unbiased=False).detach(),
        "target_level_std": future_level_values.std(unbiased=False).detach(),
        "sample_level_std": sampled_level.std(unbiased=False).detach(),
        "sample_h1_std": sampled_norm[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_norm[:, :, -1].std(unbiased=False).detach(),
        "memory_abs": fm_metrics["memory_abs"].detach(),
    }
    return total, metrics


def run_epoch(
    model: GenericStateAwareNormalizedInnovationFlowMatching,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_weight: float,
    level_energy_weight: float,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    energy_eps: float,
    temperature: float,
    clip_grad: float,
    max_batches: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    n_batches = 0
    for history_level, history_norm, future_level, future_norm, center, scale in loader:
        if int(max_batches) > 0 and n_batches >= int(max_batches):
            break
        with torch.set_grad_enabled(train_mode):
            loss, metrics = normalized_rollout_energy_loss(
                model,
                history_level.to(device),
                history_norm.to(device),
                future_level.to(device),
                future_norm.to(device),
                center.to(device),
                scale.to(device),
                train_sample_count=int(train_sample_count),
                rollout_flow_steps=int(rollout_flow_steps),
                energy_weight=float(energy_weight),
                level_energy_weight=float(level_energy_weight),
                fm_anchor_weight=float(fm_anchor_weight),
                horizon_end_weight=float(horizon_end_weight),
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state_scope", choices=["iv_only", "anchor_only", "joint38"], default="joint38")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--energy_weight", type=float, default=0.2)
    parser.add_argument("--level_energy_weight", type=float, default=0.0)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--horizon_end_weight", type=float, default=1.2)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(args.checkpoint, device)
    args.history_len = int(model.cfg.history_len)
    args.future_len = int(model.cfg.future_len)
    args.state_scope = payload.get("state_scope", args.state_scope)
    norm_cfg = payload.get("normalization", {})
    args.iv_transform = norm_cfg.get("iv_transform", payload.get("iv_transform", args.iv_transform))
    args.iv_lower_bound = float(norm_cfg.get("iv_lower_bound", payload.get("iv_lower_bound", args.iv_lower_bound)))
    args.iv_upper_bound = float(norm_cfg.get("iv_upper_bound", payload.get("iv_upper_bound", args.iv_upper_bound)))
    args.scale_floor = float(norm_cfg.get("scale_floor", args.scale_floor))
    half_life = norm_cfg.get("scale_half_life", args.scale_half_life)
    scale_half_life = None if half_life is None or float(half_life) <= 0.0 else float(half_life)

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_level, train_norm, train_future_level, train_future_norm, train_center, train_scale, _train_raw, train_specs = (
        select_normalized_innovation_scope(
            train_block,
            args.state_scope,
            int(args.iv_count),
            scale_half_life=scale_half_life,
            scale_floor=float(args.scale_floor),
        )
    )
    val_level, val_norm, val_future_level, val_future_norm, val_center, val_scale, val_raw, val_specs = (
        select_normalized_innovation_scope(
            val_block,
            args.state_scope,
            int(args.iv_count),
            scale_half_life=scale_half_life,
            scale_floor=float(args.scale_floor),
        )
    )
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    if expected and expected != [spec.name for spec in train_specs]:
        raise RuntimeError("checkpoint state specs do not match rebuilt specs")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_level),
            torch.from_numpy(train_norm),
            torch.from_numpy(train_future_level),
            torch.from_numpy(train_future_norm),
            torch.from_numpy(train_center),
            torch.from_numpy(train_scale),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(val_level),
            torch.from_numpy(val_norm),
            torch.from_numpy(val_future_level),
            torch.from_numpy(val_future_norm),
            torch.from_numpy(val_center),
            torch.from_numpy(val_scale),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, Any]] = []
    best_path = output_dir / "best_model.pt"
    objective = {
        "base": "flow_matching_mse",
        "rollout_energy_coordinate": "normalized_innovation",
        "train_sample_count": int(args.train_sample_count),
        "rollout_flow_steps": int(args.rollout_flow_steps),
        "energy_weight": float(args.energy_weight),
        "level_energy_weight": float(args.level_energy_weight),
        "fm_anchor_weight": float(args.fm_anchor_weight),
        "horizon_end_weight": float(args.horizon_end_weight),
    }
    extra = {
        "state_scope": args.state_scope,
        "model_coordinate": payload.get("model_coordinate", "state_aware_normalized_innovation"),
        "normalization": norm_cfg,
        "finetune_objective": objective,
        "iv_transform": args.iv_transform,
        "iv_lower_bound": float(args.iv_lower_bound),
        "iv_upper_bound": float(args.iv_upper_bound),
        "iv_count": int(args.iv_count),
        "state_specs": payload.get("state_specs", []),
        "panel_metadata": panel_metadata,
        "source_checkpoint": args.checkpoint,
    }
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            energy_weight=float(args.energy_weight),
            level_energy_weight=float(args.level_energy_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
            horizon_end_weight=float(args.horizon_end_weight),
            energy_eps=float(args.energy_eps),
            temperature=float(args.temperature),
            clip_grad=float(args.clip_grad),
            max_batches=int(args.max_train_batches),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device=device,
                train_sample_count=int(args.train_sample_count),
                rollout_flow_steps=int(args.rollout_flow_steps),
                energy_weight=float(args.energy_weight),
                level_energy_weight=float(args.level_energy_weight),
                fm_anchor_weight=float(args.fm_anchor_weight),
                horizon_end_weight=float(args.horizon_end_weight),
                energy_eps=float(args.energy_eps),
                temperature=float(args.temperature),
                clip_grad=0.0,
                max_batches=int(args.max_val_batches),
            )
        record = {
            "epoch": int(epoch),
            "elapsed_s": float(time.time() - t0),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        records.append(record)
        val_total = float(val_metrics.get("total", float("inf")))
        if val_total < best_val:
            best_val = val_total
            best_epoch = int(epoch)
            save_checkpoint(str(best_path), model, model.cfg, epoch, best_val, extra=extra)
        print(
            f"epoch {epoch:03d} train_total={train_metrics['total']:.6f} "
            f"val_total={val_total:.6f}{' best' if best_epoch == epoch else ''}",
            flush=True,
        )

    final_path = output_dir / "final_model.pt"
    save_checkpoint(str(final_path), model, model.cfg, int(args.epochs), best_val, extra=extra)
    smoke = sample_smoke(
        model,
        val_level[: int(args.sample_windows)],
        val_norm[: int(args.sample_windows)],
        val_center[: int(args.sample_windows)],
        val_scale[: int(args.sample_windows)],
        val_raw[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
    )
    summary = {
        "args": vars(args),
        "config": asdict(model.cfg),
        "state_scope": args.state_scope,
        "normalization": norm_cfg,
        "finetune_objective": objective,
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "sample_smoke": smoke,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_history.json").write_text(json.dumps(make_serializable(records), indent=2), encoding="utf-8")
    (output_dir / "train_summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")
    (output_dir / "args.json").write_text(json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
