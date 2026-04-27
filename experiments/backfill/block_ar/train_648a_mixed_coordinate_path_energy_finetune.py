#!/usr/bin/env python
"""648a: full-path energy-score fine-tune for the 647a mixed-coordinate model."""

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

from diffusion.block_ar.generic_mixed_coordinate_path_flow_matching import (  # noqa: E402
    GenericMixedCoordinatePathFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import (
    make_serializable,
)  # noqa: E402
from experiments.backfill.block_ar.train_596a_final_path_joint_objective import (  # noqa: E402
    full_path_energy_score,
    horizon_path_weights,
)
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    _spec_to_dict,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)
from experiments.backfill.block_ar.train_629a_state_conditioned_increment_flow import (  # noqa: E402
    sample_smoke,
    select_state_increment_scope,
)


def differentiable_mixed_path_samples(
    model: GenericMixedCoordinatePathFlowMatching,
    history_level_values: torch.Tensor,
    history_increment_values: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable path sampler returning generated mixed coordinates."""
    if n_steps < 1 or n_steps > model.cfg.future_len:
        raise ValueError(
            f"expected n_steps in [1,{model.cfg.future_len}], got {n_steps}"
        )
    context, _current_level_score = model.encode_history(
        history_level_values,
        history_increment_values,
    )
    bsz = int(history_level_values.shape[0])
    k = int(n_samples)
    ctx = (
        context.unsqueeze(1)
        .expand(bsz, k, model.cfg.memory_dim)
        .reshape(bsz * k, model.cfg.memory_dim)
    )
    x = float(temperature) * torch.randn(
        bsz * k,
        int(n_steps),
        model.cfg.n_cells,
        device=history_level_values.device,
        dtype=history_level_values.dtype,
    )
    dt = 1.0 / float(flow_steps)
    for flow_step in range(int(flow_steps)):
        t = torch.full(
            (bsz * k,),
            (flow_step + 0.5) * dt,
            device=history_level_values.device,
            dtype=history_level_values.dtype,
        )
        x = x + dt * model.predict_velocity(x, ctx, t)
    return x.view(bsz, k, int(n_steps), model.cfg.n_cells)


def mixed_path_energy_loss(
    model: GenericMixedCoordinatePathFlowMatching,
    history_level_values: torch.Tensor,
    history_increment_values: torch.Tensor,
    future_level_values: torch.Tensor,
    future_increment_values: torch.Tensor,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_weight: float,
    fm_anchor_weight: float,
    horizon_end_weight: float,
    energy_eps: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Composite proper-score loss in the generated mixed-coordinate path."""
    fm_loss, fm_metrics = model.training_loss(
        history_level_values,
        history_increment_values,
        future_level_values,
        future_increment_values,
    )
    target_mixed = model.target_mixed_coordinates(
        history_level_values,
        future_level_values,
        future_increment_values,
    )
    sampled_mixed = differentiable_mixed_path_samples(
        model,
        history_level_values,
        history_increment_values,
        n_samples=int(train_sample_count),
        n_steps=int(target_mixed.shape[1]),
        flow_steps=int(rollout_flow_steps),
        temperature=float(temperature),
    )
    weights = horizon_path_weights(
        int(target_mixed.shape[1]),
        end_weight=float(horizon_end_weight),
        device=target_mixed.device,
        dtype=target_mixed.dtype,
    )
    energy, target_dist, pair_dist = full_path_energy_score(
        sampled_mixed,
        target_mixed,
        eps=float(energy_eps),
        horizon_weights=weights,
    )
    total = float(fm_anchor_weight) * fm_loss + float(energy_weight) * energy
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": energy.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "target_mixed_std": target_mixed.std(unbiased=False).detach(),
        "sample_mixed_std": sampled_mixed.std(unbiased=False).detach(),
        "target_h1_std": target_mixed[:, 0].std(unbiased=False).detach(),
        "sample_h1_std": sampled_mixed[:, :, 0].std(unbiased=False).detach(),
        "target_h30_std": target_mixed[:, -1].std(unbiased=False).detach(),
        "sample_h30_std": sampled_mixed[:, :, -1].std(unbiased=False).detach(),
        "context_abs": fm_metrics["context_abs"].detach(),
    }
    return total, metrics


def run_epoch(
    model: GenericMixedCoordinatePathFlowMatching,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_weight: float,
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
    for history_level, history_increment, future_level, future_increment in loader:
        if int(max_batches) > 0 and n_batches >= int(max_batches):
            break
        with torch.set_grad_enabled(train_mode):
            loss, metrics = mixed_path_energy_loss(
                model,
                history_level.to(device),
                history_increment.to(device),
                future_level.to(device),
                future_increment.to(device),
                train_sample_count=int(train_sample_count),
                rollout_flow_steps=int(rollout_flow_steps),
                energy_weight=float(energy_weight),
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
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/647a_joint38_mixed_path_flow_e8_w2048_s647/best_model.pt",
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--energy_weight", type=float, default=0.10)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--horizon_end_weight", type=float, default=1.5)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=648)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_model(args.checkpoint, device)
    args.history_len = int(model.cfg.history_len)
    args.future_len = int(model.cfg.future_len)
    scope = payload.get("state_scope", args.state_scope)
    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    (
        train_level,
        train_increment,
        train_future_level,
        train_future_increment,
        _train_raw,
        train_specs,
    ) = select_state_increment_scope(train_block, scope, int(args.iv_count))
    (
        val_level,
        val_increment,
        val_future_level,
        val_future_increment,
        val_raw,
        val_specs,
    ) = select_state_increment_scope(val_block, scope, int(args.iv_count))
    expected = [spec["name"] for spec in payload.get("state_specs", [])]
    actual = [spec.name for spec in train_specs]
    if expected and expected != actual:
        raise RuntimeError("checkpoint state specs do not match rebuilt train specs")
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
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
        drop_last=False,
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
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    history_records: list[dict[str, Any]] = []
    extra = {
        "state_scope": scope,
        "model_coordinate": payload.get("model_coordinate", "mixed_coordinate_path"),
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
        "base_checkpoint": args.checkpoint,
        "finetune_loss": "full_path_energy_in_mixed_coordinates",
        "generated_coordinate_policy": payload.get("generated_coordinate_policy", {}),
    }
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            opt,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            energy_weight=float(args.energy_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
            horizon_end_weight=float(args.horizon_end_weight),
            energy_eps=float(args.energy_eps),
            temperature=float(args.temperature),
            clip_grad=float(args.clip_grad),
            max_batches=int(args.max_train_batches),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            energy_weight=float(args.energy_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
            horizon_end_weight=float(args.horizon_end_weight),
            energy_eps=float(args.energy_eps),
            temperature=float(args.temperature),
            clip_grad=float(args.clip_grad),
            max_batches=int(args.max_val_batches),
        )
        val_total = float(val_metrics["total"])
        record = {
            "epoch": int(epoch),
            "elapsed_s": float(time.time() - t0),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
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
            f"epoch {epoch:03d} train={train_metrics['total']:.6f} "
            f"val={val_total:.6f} energy={val_metrics['energy']:.6f}"
            f"{' best' if is_best else ''}",
            flush=True,
        )

    final_path = output_dir / "final_model.pt"
    save_checkpoint(
        str(final_path),
        model,
        model.cfg,
        int(args.epochs),
        best_val,
        extra=extra,
    )
    smoke = sample_smoke(
        model,
        val_level[: int(args.sample_windows)],
        val_increment[: int(args.sample_windows)],
        val_raw[: int(args.sample_windows)],
        train_specs,
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=min(int(args.iv_count), int(val_level.shape[-1])),
    )
    summary = {
        "args": vars(args),
        "config": asdict(model.cfg),
        "state_scope": scope,
        "model_coordinate": extra["model_coordinate"],
        "base_checkpoint": args.checkpoint,
        "n_state_vars": int(val_level.shape[-1]),
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
