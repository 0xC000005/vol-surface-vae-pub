#!/usr/bin/env python
"""
Check whether 194c's failure is mainly a train/val/test split generalization issue.

We compare train-like, val-like, and test-like windows on the same rollout and
alignment diagnostics. If the same failure pattern is present on train-like data,
the issue is not primarily out-of-sample generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_194c_two_state_mechanism import (
    analyze_194a_teacher_forced,
    build_test_subset,
    hard_late_alignment_summary,
    load_194a_model,
    make_serializable,
    regime_masks_from_history,
    rollout_coverage_arrays,
)


def safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def split_summary(
    model,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
    rollout_samples: int,
) -> dict[str, Any]:
    _vov, q20, q80, calm_mask, turb_mask = regime_masks_from_history(history_norm)
    tf_diag = analyze_194a_teacher_forced(
        model, history_norm, future_norm, device=device, batch_size=batch_size
    )
    ro_diag = rollout_coverage_arrays(
        model,
        history_norm,
        future_norm,
        device=device,
        n_samples=rollout_samples,
        batch_size=batch_size,
    )
    hard_summary = hard_late_alignment_summary(
        tf_diag["gamma"],
        tf_diag["state_var_cells"],
        tf_diag["expected_var_cells"],
        ro_diag["inside90"],
        calm_mask=calm_mask,
        turb_mask=turb_mask,
    )
    window_width = ro_diag["width"].mean(axis=(1, 2))
    turb_calm = float(window_width[turb_mask].mean() / window_width[calm_mask].mean()) if calm_mask.any() and turb_mask.any() else float("nan")

    return {
        "windows": int(history_norm.shape[0]),
        "q20_vov": q20,
        "q80_vov": q80,
        "calm_windows": int(calm_mask.sum()),
        "turb_windows": int(turb_mask.sum()),
        "cov90": float(ro_diag["inside90"].mean()),
        "width90": float(ro_diag["width"].mean()),
        "turb_calm_ratio": turb_calm,
        "overall_state_occupancy": tf_diag["gamma"].mean(axis=(0, 1)).tolist(),
        "turb_h30_state_occupancy": tf_diag["gamma"][turb_mask, 29, :].mean(axis=0).tolist() if turb_mask.any() else [],
        "calm_h30_state_occupancy": tf_diag["gamma"][calm_mask, 29, :].mean(axis=0).tolist() if calm_mask.any() else [],
        "hard_vs_high_state_prob_corr": hard_summary["hard_vs_high_state_prob_corr"],
        "mean_high_state_prob_hard_vs_clean": [
            hard_summary["mean_high_state_prob_hard"],
            hard_summary["mean_high_state_prob_clean"],
        ],
        "mean_expected_var_hard_vs_clean": [
            hard_summary["mean_expected_var_hard"],
            hard_summary["mean_expected_var_clean"],
        ],
        "expected_var_share_on_hard": hard_summary["expected_var_share_on_hard"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split generalization check for 194c")
    parser.add_argument("--ckpt", type=str, default="models/backfill/regime_switching_ar_latent_factor_194c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--probe_windows", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/194c_split_generalization",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_len = args.history_len
    future_len = args.future_len
    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_end = max_train_idx - val_size
    train_probe_start = max(train_end - args.probe_windows, 0)
    val_probe_start = train_end
    test_probe_start = 4540

    model, _payload = load_194a_model(Path(args.ckpt), device)

    splits = {
        "train_like": build_test_subset(args.data_path, hist_len, future_len, train_probe_start, args.probe_windows),
        "val_like": build_test_subset(args.data_path, hist_len, future_len, val_probe_start, min(args.probe_windows, val_size)),
        "test_like": build_test_subset(args.data_path, hist_len, future_len, test_probe_start, args.probe_windows),
    }

    summary: dict[str, Any] = {
        "config": {
            "train_probe_start": train_probe_start,
            "val_probe_start": val_probe_start,
            "test_probe_start": test_probe_start,
            "probe_windows": args.probe_windows,
            "batch_size": args.batch_size,
            "rollout_samples": args.rollout_samples,
        }
    }

    for name, (history_norm, future_norm) in splits.items():
        summary[name] = split_summary(
            model,
            history_norm,
            future_norm,
            device=device,
            batch_size=args.batch_size,
            rollout_samples=args.rollout_samples,
        )

    summary["interpretation"] = {
        "generalization_read": (
            "If train_like already shows weak hard-state alignment and broad overcoverage similar to val_like/test_like, "
            "the failure is primarily model/optimization, not split generalization."
        )
    }

    out_path = output_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(make_serializable(summary), f, indent=2)

    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved split summary to {out_path}")


if __name__ == "__main__":
    main()
