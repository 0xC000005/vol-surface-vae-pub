#!/usr/bin/env python
"""
Fixed H=1 smoke evaluation for 211a.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_211a_h1_decoder_only_token_transformer import (
    build_tokenized_sequences,
    evaluate_h1,
    load_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="211a H=1 smoke eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
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

    surf_tensor = torch.from_numpy(surfaces)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)
    model, payload = load_model(args.checkpoint, device)

    tok = build_tokenized_sequences(
        history_01=history_01,
        target_01=target_01,
        quantizer=model.quantizer,
        support_lo=model.support_lo,
        support_hi=model.support_hi,
        support_eps=model.support_eps,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            tok["input_ids"],
            tok["labels"],
            tok["prefix_ids"],
            history_01,
            target_01,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    metrics = evaluate_h1(
        model=model,
        loader=loader,
        q95_threshold=q95,
        q99_threshold=q99,
        eval_samples=args.eval_samples,
        sample_batch_size=args.sample_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    checks = {
        "h1_cov90_ok": 0.85 <= metrics["val_coverage_90"] <= 0.93,
        "h1_q99_cov_ok": metrics["val_realized_q99_coverage_90"] >= 0.58,
        "h1_quiet_ok": metrics["val_h1_quiet_ratio"] >= 0.85,
        "h1_shoulder_ok": metrics["val_h1_shoulder_ratio"] <= 1.10,
        "h1_kurtosis_ok": metrics["val_h1_kurtosis_ratio"] >= 0.65,
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "n_val_windows": int(history_01.shape[0]),
        "thresholds": {"q95": q95, "q99": q99},
        "backend": "transformers.GPT2LMHeadModel",
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
        "move_size_metrics": {
            "very_small_lte_0p005": {
                "gt_share": float(metrics["val_h1_very_small_0p005_gt_share"]),
                "sample_share": float(metrics["val_h1_very_small_0p005_sample_share"]),
                "ratio": float(metrics["val_h1_very_small_0p005_ratio"]),
            },
            "small_lte_0p010": {
                "gt_share": float(metrics["val_h1_small_0p010_gt_share"]),
                "sample_share": float(metrics["val_h1_small_0p010_sample_share"]),
                "ratio": float(metrics["val_h1_small_0p010_ratio"]),
            },
            "moderate_lte_0p020": {
                "gt_share": float(metrics["val_h1_moderate_0p020_gt_share"]),
                "sample_share": float(metrics["val_h1_moderate_0p020_sample_share"]),
                "ratio": float(metrics["val_h1_moderate_0p020_ratio"]),
            },
            "large_lte_0p050": {
                "gt_share": float(metrics["val_h1_large_0p050_gt_share"]),
                "sample_share": float(metrics["val_h1_large_0p050_sample_share"]),
                "ratio": float(metrics["val_h1_large_0p050_ratio"]),
            },
        },
        "decoder_metrics": {
            "token_ce": float(metrics["val_token_ce"]),
            "token_top1": float(metrics["val_token_top1"]),
            "token_entropy": float(metrics["val_token_entropy"]),
            "unique_sequence_ratio": float(metrics["val_unique_sequence_ratio"]),
        },
        "smoke_gate_checks": checks,
        "smoke_gate_pass": bool(all(checks.values())),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"smoke_gate_checks": checks, "smoke_gate_pass": bool(all(checks.values()))}, indent=2))


if __name__ == "__main__":
    main()
