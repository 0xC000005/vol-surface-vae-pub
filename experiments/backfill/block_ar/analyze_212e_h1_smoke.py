#!/usr/bin/env python
"""
Fixed H=1 smoke evaluation for 212e.
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
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import compute_h1_shape_stats
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="212e H=1 smoke eval")
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

    samples = []
    for start in range(0, history_01.shape[0], args.batch_size):
        end = min(start + args.batch_size, history_01.shape[0])
        batch_hist = history_01[start:end]
        samp = model.sample_next_iv(batch_hist, n_samples=args.eval_samples)
        samples.append(samp.detach().cpu().numpy())
    samples = np.concatenate(samples, axis=0)

    target = target_01.detach().cpu().numpy()
    prev = history_01[:, -1].reshape(history_01.shape[0], -1).detach().cpu().numpy()
    q05 = np.quantile(samples, 0.05, axis=1)
    q95s = np.quantile(samples, 0.95, axis=1)
    mean_pred = samples.mean(axis=1)

    realized_abs = np.abs(target - prev)
    q95_mask = realized_abs >= q95
    q99_mask = realized_abs >= q99

    gt_delta = target - prev
    sample_delta = samples - prev[:, None, :]
    shape_stats = compute_h1_shape_stats(gt_delta, sample_delta)

    checks = {
        "h1_cov90_ok": 0.85 <= float(((target >= q05) & (target <= q95s)).mean()) <= 0.93,
        "h1_q99_cov_ok": float(((target[q99_mask] >= q05[q99_mask]) & (target[q99_mask] <= q95s[q99_mask])).mean()) >= 0.58 if q99_mask.any() else False,
        "h1_quiet_ok": shape_stats["quiet_ratio"] >= 0.85,
        "h1_shoulder_ok": shape_stats["shoulder_ratio"] <= 1.10,
        "h1_kurtosis_ok": shape_stats["kurtosis_ratio"] >= 0.65,
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "n_val_windows": int(history_01.shape[0]),
        "thresholds": {"q95": q95, "q99": q99},
        "h1_metrics": {
            "coverage_90": float(((target >= q05) & (target <= q95s)).mean()),
            "mae": float(np.abs(mean_pred - target).mean()),
            "width_90": float(np.mean(q95s - q05)),
            "realized_q95_coverage_90": float(((target[q95_mask] >= q05[q95_mask]) & (target[q95_mask] <= q95s[q95_mask])).mean()) if q95_mask.any() else float("nan"),
            "realized_q99_coverage_90": float(((target[q99_mask] >= q05[q99_mask]) & (target[q99_mask] <= q95s[q99_mask])).mean()) if q99_mask.any() else float("nan"),
            "q95_cell_count": int(q95_mask.sum()),
            "q99_cell_count": int(q99_mask.sum()),
        },
        "shape_metrics": shape_stats,
        "sample_stats": {
            "sample_delta_std_mean": float(sample_delta.std(axis=1).mean()),
        },
        "smoke_gate_checks": checks,
        "smoke_gate_pass": bool(all(checks.values())),
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"smoke_gate_checks": checks, "smoke_gate_pass": bool(all(checks.values()))}, indent=2))


if __name__ == "__main__":
    main()
