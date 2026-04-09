#!/usr/bin/env python
"""
Fixed H=1 smoke evaluation for 210h.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_window_metadata,
    load_model as load_teacher_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import (
    collect_teacher_forced_records,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_210h_h1_teacher_guided_token_transformer import (
    assign_tokens_from_prototypes,
    evaluate_h1,
    load_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="210h H=1 smoke eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    max_train_idx = args.test_start - args.history_len - 30
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)
    model, payload = load_model(args.checkpoint, device)
    config = payload["config"]

    teacher_model, _ = load_teacher_model(config["teacher_checkpoint"], device)
    val_future = target_01.unsqueeze(1)
    val_meta = build_window_metadata(
        history_01.detach().cpu().numpy(),
        val_future.detach().cpu().numpy(),
    )
    val_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=history_01,
        future_flat=val_future,
        window_meta=val_meta,
        split_name="val",
        q95=q95,
        q99=q99,
        batch_size=args.batch_size,
        device=device,
    )
    event_prototypes = np.asarray(config["event_prototypes"], dtype=np.float32)
    val_tokens = torch.from_numpy(assign_tokens_from_prototypes(val_records, event_prototypes)).to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(history_01, target_01, val_tokens),
        batch_size=args.batch_size,
        shuffle=False,
    )
    metrics = evaluate_h1(model, loader, q95_threshold=q95, q99_threshold=q99, eval_samples=args.eval_samples)

    smoke_gate_checks = {
        "h1_cov90_ok": 0.85 <= metrics["val_coverage_90"] <= 0.93,
        "h1_q99_cov_ok": metrics["val_realized_q99_coverage_90"] >= 0.58,
        "h1_quiet_ok": metrics["val_h1_quiet_ratio"] >= 0.85,
        "h1_shoulder_ok": metrics["val_h1_shoulder_ratio"] <= 1.10,
        "h1_kurtosis_ok": metrics["val_h1_kurtosis_ratio"] >= 0.65,
    }
    token_checks = {
        "token_prior_top1_ok": metrics["val_prior_top1_mean"] >= 0.10,
        "token_mu_disp_ok": metrics["val_mu_dispersion"] >= 0.01,
        "token_event_top3_ok": metrics["val_event_token_top3"] >= 0.50 if metrics["val_event_window_count"] > 0 else False,
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "n_val_windows": int(history_01.shape[0]),
        "thresholds": {"q95": q95, "q99": q99},
        "h1_metrics": {
            "coverage_90": float(metrics["val_coverage_90"]),
            "mae": float(metrics["val_mae"]),
            "width_90": float(metrics["val_width_90"]),
            "realized_q95_coverage_90": float(metrics["val_realized_q95_coverage_90"]),
            "realized_q99_coverage_90": float(metrics["val_realized_q99_coverage_90"]),
            "q95_cell_count": int(metrics["val_q95_cell_count"]),
            "q99_cell_count": int(metrics["val_q99_cell_count"]),
        },
        "shape_metrics": {
            "quiet_ratio": float(metrics["val_h1_quiet_ratio"]),
            "shoulder_ratio": float(metrics["val_h1_shoulder_ratio"]),
            "extreme_ratio": float(metrics["val_h1_extreme_ratio"]),
            "kurtosis_ratio": float(metrics["val_h1_kurtosis_ratio"]),
        },
        "token_metrics": {
            "token_top1": float(metrics["val_token_top1"]),
            "event_token_top1": float(metrics["val_event_token_top1"]),
            "event_token_top3": float(metrics["val_event_token_top3"]),
            "event_window_count": int(metrics["val_event_window_count"]),
            "prior_top1_mean": float(metrics["val_prior_top1_mean"]),
            "prior_entropy_mean": float(metrics["val_prior_entropy_mean"]),
            "prior_active_tokens": float(metrics["val_prior_active_tokens"]),
            "mu_dispersion": float(metrics["val_mu_dispersion"]),
            "scale_dispersion": float(metrics["val_scale_dispersion"]),
        },
        "smoke_gate_checks": smoke_gate_checks,
        "smoke_gate_pass": bool(all(smoke_gate_checks.values())),
        "token_checks": token_checks,
        "token_branch_pass": bool(all(smoke_gate_checks.values()) and all(token_checks.values())),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(
        json.dumps(
            {
                "smoke_gate_checks": smoke_gate_checks,
                "smoke_gate_pass": bool(all(smoke_gate_checks.values())),
                "token_checks": token_checks,
                "token_branch_pass": bool(all(smoke_gate_checks.values()) and all(token_checks.values())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
