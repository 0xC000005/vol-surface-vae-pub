#!/usr/bin/env python
"""614a: train a generic AR Gaussian transition law by conditional likelihood."""

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

from diffusion.block_ar.generic_gaussian_transition_law import (  # noqa: E402
    GenericGaussianTransitionConfig,
    GenericGaussianTransitionLaw,
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


def eval_loss(
    model: GenericGaussianTransitionLaw,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for history, future in loader:
            history = history.to(device)
            future = future.to(device)
            loss, _metrics = model.training_loss(history, future)
            total += float(loss.item()) * int(history.shape[0])
            count += int(history.shape[0])
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_scope", choices=["iv_only", "joint38"], default="joint38")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=1024)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--memory_dim", type=int, default=128)
    parser.add_argument("--memory_layers", type=int, default=3)
    parser.add_argument("--memory_heads", type=int, default=4)
    parser.add_argument("--memory_ff", type=int, default=256)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--model_dropout", type=float, default=0.05)
    parser.add_argument("--diag_floor", type=float, default=0.03)
    parser.add_argument("--diag_max", type=float, default=3.0)
    parser.add_argument("--offdiag_scale", type=float, default=0.25)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--distribution_family", choices=["gaussian", "student_t"], default="gaussian")
    parser.add_argument("--student_t_df", type=float, default=5.0)
    parser.add_argument("--mean_loss_weight", type=float, default=0.0)
    parser.add_argument("--prefix_feature_mode", choices=["basic", "scale"], default="basic")
    parser.add_argument("--sample_windows", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _columns, panel_metadata, train_block, val_block = build_blocks(args)
    train_history, train_future, train_specs = select_scope(train_block, args.state_scope, int(args.iv_count))
    val_history, val_future, val_specs = select_scope(val_block, args.state_scope, int(args.iv_count))
    if [spec.name for spec in train_specs] != [spec.name for spec in val_specs]:
        raise RuntimeError("train/val state specs differ")

    value_quantiles, quantile_levels = fit_empirical_quantiles(
        train_history,
        train_future,
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
    )
    cfg = GenericGaussianTransitionConfig(
        history_len=int(args.history_len),
        future_len=int(args.future_len),
        n_cells=int(train_history.shape[-1]),
        n_quantiles=int(args.n_quantiles),
        cdf_eps=float(args.cdf_eps),
        prefix_feature_mode=args.prefix_feature_mode,
        memory_dim=int(args.memory_dim),
        memory_layers=int(args.memory_layers),
        memory_heads=int(args.memory_heads),
        memory_ff=int(args.memory_ff),
        head_hidden=int(args.head_hidden),
        model_dropout=float(args.model_dropout),
        diag_floor=float(args.diag_floor),
        diag_max=float(args.diag_max),
        offdiag_scale=float(args.offdiag_scale),
        sample_temperature=float(args.sample_temperature),
        distribution_family=args.distribution_family,
        student_t_df=float(args.student_t_df),
        mean_loss_weight=float(args.mean_loss_weight),
    )
    model = GenericGaussianTransitionLaw(cfg).to(device)
    model.set_empirical_quantiles(
        torch.from_numpy(value_quantiles).to(device),
        torch.from_numpy(quantile_levels).to(device),
    )
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

    history_records: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_model.pt"
    t0 = time.time()
    extra = {
        "state_scope": args.state_scope,
        "iv_count": int(args.iv_count),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "panel_metadata": panel_metadata,
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        metric_sums: dict[str, float] = {}
        for history, future in train_loader:
            history = history.to(device)
            future = future.to(device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = model.training_loss(history, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_n = int(history.shape[0])
            total += float(loss.item()) * batch_n
            count += batch_n
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * batch_n
        train_loss = total / max(count, 1)
        val_loss = eval_loss(model, val_loader, device)
        record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "elapsed_s": float(time.time() - t0),
        }
        for key, value in metric_sums.items():
            record[f"train_{key}"] = float(value / max(count, 1))
        history_records.append(record)
        is_best = val_loss < best_val
        if is_best:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            save_checkpoint(str(best_path), model, cfg, epoch, best_val, extra=extra)
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}"
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
    )
    summary = {
        "args": vars(args),
        "config": asdict(cfg),
        "state_scope": args.state_scope,
        "n_state_vars": int(train_history.shape[-1]),
        "state_specs": [_spec_to_dict(spec) for spec in train_specs],
        "train_shape": list(train_history.shape),
        "val_shape": list(val_history.shape),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "final_val_loss": float(history_records[-1]["val_loss"] if history_records else float("nan")),
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
    (output_dir / "args.json").write_text(json.dumps(make_serializable(vars(args)), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
