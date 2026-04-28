#!/usr/bin/env python
"""661a: generalized 510a-style joint AR patch-energy trainer.

This keeps the generic empirical-score AR transition model from 609a and adds the
510a patch-energy/free-rollout objective over the full selected state panel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from diffusion.block_ar.generic_empirical_score_transition_flow_matching import (  # noqa: E402
    GenericEmpiricalScoreTransitionFMConfig,
    GenericEmpiricalScoreTransitionFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    _spec_to_dict,
    build_blocks,
    fit_empirical_quantiles,
    sample_smoke,
    select_scope,
)


def patch_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_len: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Energy score averaged over overlapping future patches.

    `samples` are generated empirical-score paths with shape `[B,K,T,C]`.
    `target` is the realized empirical-score path with shape `[B,T,C]`.
    """
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected samples [B,K,T,C] and target [B,T,C]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("sample and target dimensions do not match")
    bsz, _n_samples, horizon, n_cells = samples.shape
    length = int(patch_len)
    if length < 1 or length > horizon:
        raise ValueError(f"patch_len must be in [1,{horizon}], got {patch_len}")
    scale = math.sqrt(float(length * n_cells))
    scores: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    pair_terms: list[torch.Tensor] = []
    for start in range(0, horizon - length + 1):
        sample_patch = samples[:, :, start : start + length].reshape(
            bsz, samples.shape[1], length * n_cells
        )
        target_patch = target[:, start : start + length].reshape(bsz, length * n_cells)
        target_dist = (
            torch.sqrt(
                (sample_patch - target_patch[:, None, :]).pow(2).sum(dim=-1)
                + float(eps)
            ).mean(dim=1)
            / scale
        )
        pair_dist = torch.cdist(sample_patch, sample_patch, p=2).mean(dim=(1, 2)) / scale
        scores.append(target_dist - 0.5 * pair_dist)
        target_terms.append(target_dist)
        pair_terms.append(pair_dist)
    return (
        torch.stack(scores).mean(),
        torch.stack(target_terms).mean(),
        torch.stack(pair_terms).mean(),
    )


def sample_rollout_scores_with_grad(
    model: GenericEmpiricalScoreTransitionFlowMatching,
    history_values: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    """Differentiable version of `sample_batched` that returns score paths."""
    if n_steps < 1 or n_steps > model.cfg.future_len:
        raise ValueError(f"expected n_steps in [1,{model.cfg.future_len}], got {n_steps}")
    history_scores = model.values_to_scores(history_values)
    bsz, _hist_len, n_cells = history_scores.shape
    k = int(n_samples)
    prefix = (
        history_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, n_cells)
        .reshape(bsz * k, model.cfg.history_len, n_cells)
        .clone()
    )
    rho = float(max(0.0, min(0.999, model.cfg.path_source_corr)))
    ar_rho = float(max(0.0, min(0.999, model.cfg.path_source_ar)))
    path_source = None
    if rho > 0.0:
        path_source = torch.randn(
            bsz * k,
            n_cells,
            device=history_values.device,
            dtype=history_values.dtype,
        )
    temporal_source = model._ar1_source_noise(
        torch.Size((bsz * k, int(n_steps), n_cells)),
        ar_rho,
        history_values.device,
        history_values.dtype,
    )
    dt = 1.0 / float(flow_steps)
    frames: list[torch.Tensor] = []
    for step in range(int(n_steps)):
        memory_state = model._encode_prefix_scores(prefix)[:, -1]
        current_score = prefix[:, -1]
        if path_source is None:
            x = float(temperature) * temporal_source[:, step]
        else:
            x = float(temperature) * (
                math.sqrt(rho) * path_source
                + math.sqrt(1.0 - rho) * temporal_source[:, step]
            )
        source_scale = model._conditional_source_scale(memory_state)
        if source_scale is not None:
            x = x * source_scale
        for flow_step in range(int(flow_steps)):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=history_values.device,
                dtype=history_values.dtype,
            )
            x = x + dt * model.velocity(x, current_score, memory_state, t)
        next_score = current_score + x
        frames.append(next_score.view(bsz, k, n_cells))
        prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
    return torch.stack(frames, dim=2)


def combined_loss(
    model: GenericEmpiricalScoreTransitionFlowMatching,
    history_values: torch.Tensor,
    future_values: torch.Tensor,
    *,
    train_sample_count: int,
    rollout_flow_steps: int,
    patch_len: int,
    patch_energy_weight: float,
    fm_anchor_weight: float,
    energy_eps: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_values, future_values)
    target_scores = model.values_to_scores(future_values)
    sampled_scores = sample_rollout_scores_with_grad(
        model,
        history_values,
        n_samples=int(train_sample_count),
        n_steps=int(target_scores.shape[1]),
        flow_steps=int(rollout_flow_steps),
        temperature=float(temperature),
    )
    patch_loss, target_dist, pair_dist = patch_energy_score(
        sampled_scores,
        target_scores,
        patch_len=int(patch_len),
        eps=float(energy_eps),
    )
    total = float(fm_anchor_weight) * fm_loss + float(patch_energy_weight) * patch_loss
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "patch_energy": patch_loss.detach(),
        "patch_target_dist": target_dist.detach(),
        "patch_pair_dist": pair_dist.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "target_score_std": target_scores.std(unbiased=False).detach(),
        "sample_score_std": sampled_scores.std(unbiased=False).detach(),
        "target_h1_std": target_scores[:, 0].std(unbiased=False).detach(),
        "sample_h1_std": sampled_scores[:, :, 0].std(unbiased=False).detach(),
        "target_h30_std": target_scores[:, -1].std(unbiased=False).detach(),
        "sample_h30_std": sampled_scores[:, :, -1].std(unbiased=False).detach(),
        "memory_abs": fm_metrics["memory_abs"].detach(),
    }
    return total, metrics


def run_epoch(
    model: GenericEmpiricalScoreTransitionFlowMatching,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    train_sample_count: int,
    rollout_flow_steps: int,
    patch_len: int,
    patch_energy_weight: float,
    fm_anchor_weight: float,
    energy_eps: float,
    temperature: float,
    clip_grad: float,
    max_batches: int,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    n_batches = 0
    for history, future in loader:
        if int(max_batches) > 0 and n_batches >= int(max_batches):
            break
        history = history.to(device)
        future = future.to(device)
        with torch.set_grad_enabled(train_mode):
            loss, metrics = combined_loss(
                model,
                history,
                future,
                train_sample_count=int(train_sample_count),
                rollout_flow_steps=int(rollout_flow_steps),
                patch_len=int(patch_len),
                patch_energy_weight=float(patch_energy_weight),
                fm_anchor_weight=float(fm_anchor_weight),
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


def build_model(
    args: argparse.Namespace,
    train_history: np.ndarray,
    train_future: np.ndarray,
    device: torch.device,
) -> tuple[GenericEmpiricalScoreTransitionFlowMatching, GenericEmpiricalScoreTransitionFMConfig]:
    if args.checkpoint:
        model, payload = load_model(args.checkpoint, device)
        cfg = model.cfg
        if cfg.history_len != int(args.history_len) or cfg.future_len != int(args.future_len):
            raise ValueError("checkpoint horizon configuration does not match data")
        if cfg.n_cells != int(train_history.shape[-1]):
            raise ValueError("checkpoint n_cells does not match selected state panel")
        print(f"Loaded source checkpoint {args.checkpoint}")
        print(f"Source epoch={payload.get('epoch')} best_val={payload.get('best_val')}")
        return model, cfg

    value_quantiles, quantile_levels = fit_empirical_quantiles(
        train_history,
        train_future,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    cfg = GenericEmpiricalScoreTransitionFMConfig(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        n_cells=int(train_history.shape[-1]),
        memory_dim=int(args.memory_dim),
        memory_layers=int(args.memory_layers),
        memory_heads=int(args.memory_heads),
        memory_ff=int(args.memory_ff),
        token_dim=int(args.token_dim),
        token_layers=int(args.token_layers),
        token_heads=int(args.token_heads),
        token_ff=int(args.token_ff),
        time_dim=int(args.time_dim),
        model_dropout=float(args.model_dropout),
        flow_steps=int(args.flow_steps),
        sample_temperature=float(args.sample_temperature),
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
        prefix_feature_mode=args.prefix_feature_mode,
        path_source_corr=float(args.path_source_corr),
        path_source_ar=float(args.path_source_ar),
        conditioning_mode="prefix",
    )
    model = GenericEmpiricalScoreTransitionFlowMatching(cfg).to(device)
    model.set_empirical_quantiles(
        torch.from_numpy(value_quantiles).to(device),
        torch.from_numpy(quantile_levels).to(device),
    )
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--state_scope", choices=["iv_only", "joint38"], default="joint38")
    parser.add_argument("--value_coordinate", choices=["raw", "encoded"], default="encoded")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--token_layers", type=int, default=3)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--token_ff", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--model_dropout", type=float, default=0.05)
    parser.add_argument("--flow_steps", type=int, default=16)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--path_source_corr", type=float, default=0.0)
    parser.add_argument("--path_source_ar", type=float, default=0.0)
    parser.add_argument("--prefix_feature_mode", choices=["basic", "scale"], default="scale")
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--patch_len", type=int, default=5)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--patch_energy_weight", type=float, default=0.05)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=661)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_history, train_future, train_specs = select_scope(
        train_block,
        args.state_scope,
        int(args.iv_count),
        args.value_coordinate,
    )
    val_history, val_future, val_specs = select_scope(
        val_block,
        args.state_scope,
        int(args.iv_count),
        args.value_coordinate,
    )
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")

    model, cfg = build_model(args, train_history, train_future, device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_history), torch.from_numpy(train_future)),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_history), torch.from_numpy(val_future)),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs))

    extra = {
        "state_scope": args.state_scope,
        "value_coordinate": args.value_coordinate,
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
        "model_family": "661a_generalized_510a_joint_patch_energy",
    }
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    records: list[dict[str, Any]] = []
    t0 = time.time()
    print(f"Training 661a on {train_history.shape} -> {train_future.shape}")
    print(f"Validation {val_history.shape} -> {val_future.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, int(args.epochs) + 1):
        train_avg = run_epoch(
            model,
            train_loader,
            opt,
            device=device,
            train_sample_count=int(args.train_sample_count),
            rollout_flow_steps=int(args.rollout_flow_steps),
            patch_len=int(args.patch_len),
            patch_energy_weight=float(args.patch_energy_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
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
            patch_len=int(args.patch_len),
            patch_energy_weight=float(args.patch_energy_weight),
            fm_anchor_weight=float(args.fm_anchor_weight),
            energy_eps=float(args.energy_eps),
            temperature=float(args.sample_temperature),
            clip_grad=0.0,
            max_batches=int(args.max_val_batches),
        )
        scheduler.step()
        record = {
            "epoch": int(epoch),
            "train_total": float(train_avg["total"]),
            "train_fm_loss": float(train_avg["fm_loss"]),
            "train_patch_energy": float(train_avg["patch_energy"]),
            "val_total": float(val_avg["total"]),
            "val_fm_loss": float(val_avg["fm_loss"]),
            "val_patch_energy": float(val_avg["patch_energy"]),
            "val_patch_target_dist": float(val_avg["patch_target_dist"]),
            "val_patch_pair_dist": float(val_avg["patch_pair_dist"]),
            "val_sample_score_std": float(val_avg["sample_score_std"]),
            "val_target_score_std": float(val_avg["target_score_std"]),
            "val_sample_h1_std": float(val_avg["sample_h1_std"]),
            "val_target_h1_std": float(val_avg["target_h1_std"]),
            "val_sample_h30_std": float(val_avg["sample_h30_std"]),
            "val_target_h30_std": float(val_avg["target_h30_std"]),
            "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": float(time.time() - t0),
        }
        records.append(record)
        is_best = record["val_total"] < best_val
        if is_best:
            best_val = float(record["val_total"])
            best_epoch = int(epoch)
            save_checkpoint(str(best_path), model, cfg, epoch, best_val, extra=extra)
        print(
            f"epoch {epoch:03d} train={record['train_total']:.6f} "
            f"val={record['val_total']:.6f} patch={record['val_patch_energy']:.6f}"
            f"{' best' if is_best else ''}",
            flush=True,
        )

    final_path = output_dir / "final_model.pt"
    save_checkpoint(str(final_path), model, cfg, int(args.epochs), best_val, extra=extra)
    smoke = sample_smoke(
        model,
        val_history[: int(args.sample_windows)],
        samples=int(args.sample_count),
        steps=min(int(args.sample_steps), int(args.future_len)),
        chunk_size=int(args.chunk_size),
        device=device,
        iv_count=int(args.iv_count),
        specs=train_specs,
        value_coordinate=args.value_coordinate,
    )
    summary = {
        "args": vars(args),
        "config": asdict(cfg),
        "state_scope": args.state_scope,
        "value_coordinate": args.value_coordinate,
        "n_state_vars": int(train_history.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_history.shape),
        "val_shape": list(val_history.shape),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(records[-1]["val_total"] if records else float("nan")),
        "sample_smoke": smoke,
        "panel_metadata": panel_metadata,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(make_serializable(records), indent=2),
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
