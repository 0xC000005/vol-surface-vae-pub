#!/usr/bin/env python
"""
Preparation audit for the next AR rare-event phase.

Summarizes:
  - raw tail sparsity in the dataset
  - train-window sparsity for hard late-horizon windows
  - current strict-status of the strongest AR and one-shot anchors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, ".")


SUITE_KEYS = [
    "surface",
    "coverage",
    "conditionality",
    "time_series",
    "block_ar",
    "cointegration",
    "regime_coverage",
    "distributional",
    "cross_cell_correlation",
    "mean_reversion",
    "pathwise_jump_realism",
]


def load_summary(path: str) -> dict:
    return json.load(open(path))


def summarize_model(path: str) -> dict:
    d = load_summary(path)
    return {
        "path": path,
        "pass_count": int(sum(bool(d[k].get("overall_pass", False)) for k in SUITE_KEYS)),
        "passes": {k: bool(d[k].get("overall_pass", False)) for k in SUITE_KEYS},
        "h1_cov90": float(d["coverage"]["per_horizon"]["1"]["0.9"]),
        "h30_cov90": float(d["coverage"]["per_horizon"]["30"]["0.9"]),
        "h1_worst_cell_cov90": float(d["coverage"]["worst_cell_per_horizon"]["1"]),
        "h30_worst_cell_cov90": float(d["coverage"]["worst_cell_per_horizon"]["30"]),
        "turb_calm_ratio": float(d["conditionality"]["turb_calm_ratio"]),
        "kurtosis_ratio": float(d["time_series"]["kurtosis"]["kurtosis_ratio"]),
        "quiet_mass_ratio": float(d["time_series"]["exceedance_spectrum"]["quiet_mass"]["ratio"]),
        "shoulder_mass_ratio": float(d["time_series"]["exceedance_spectrum"]["shoulder_mass"]["ratio"]),
        "extreme_mass_ratio": float(d["time_series"]["exceedance_spectrum"]["extreme_mass"]["ratio"]),
        "regime_layer2_passes": int(d["regime_coverage"]["layer2_n_passing"]),
        "regime_layer2_total": int(d["regime_coverage"]["layer2_n_total"]),
        "regime_catastrophic_rate": float(d["regime_coverage"]["layer3_catastrophic_rate"]),
        "bad_window_rate": float(d["distributional"]["window_floor"]["pct_bad"]),
        "mean_reversion_pass": bool(d["mean_reversion"]["overall_pass"]),
        "jump_pass": bool(d["pathwise_jump_realism"]["overall_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AR sparse-case readiness audit")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/200_design/200a_ar_sparse_case_audit",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    surf = data["surface"].astype(np.float32)
    changes = np.diff(surf, axis=0).reshape(-1)
    abs_changes = np.abs(changes)

    q95 = float(np.quantile(abs_changes, 0.95))
    q99 = float(np.quantile(abs_changes, 0.99))
    q995 = float(np.quantile(abs_changes, 0.995))

    hist_len = args.history_len
    fut_len = args.future_len
    max_train_idx = args.test_start - hist_len - fut_len
    starts = np.arange(max_train_idx)
    daily_mean = surf.mean(axis=(1, 2))

    vov = []
    h30_energy = []
    future_q99_counts = []
    for s in starts:
        hist = daily_mean[s : s + hist_len]
        fut = surf[s + hist_len : s + hist_len + fut_len]
        prev = surf[s + hist_len - 1]
        vov.append(np.diff(hist).std())
        h30_energy.append(np.abs(fut[-1] - prev).sum())
        future_deltas = np.abs(np.diff(np.concatenate([prev[None], fut], axis=0), axis=0)).reshape(-1)
        future_q99_counts.append(int((future_deltas >= q99).sum()))

    vov = np.asarray(vov)
    h30_energy = np.asarray(h30_energy)
    future_q99_counts = np.asarray(future_q99_counts)

    q80_vov = float(np.quantile(vov, 0.8))
    turb = vov >= q80_vov
    q80_h30_turb = float(np.quantile(h30_energy[turb], 0.8))
    hard_late = turb & (h30_energy >= q80_h30_turb)

    summary = {
        "config": {
            "data_path": args.data_path,
            "history_len": hist_len,
            "future_len": fut_len,
            "test_start": args.test_start,
            "train_windows": int(len(starts)),
        },
        "raw_tail_sparsity": {
            "surface_shape": list(surf.shape),
            "n_cell_day_changes": int(abs_changes.size),
            "abs_q95_threshold": q95,
            "abs_q99_threshold": q99,
            "abs_q995_threshold": q995,
            "q95_fraction": float((abs_changes >= q95).mean()),
            "q99_fraction": float((abs_changes >= q99).mean()),
            "q995_fraction": float((abs_changes >= q995).mean()),
        },
        "window_level_sparsity": {
            "turbulent_windows": int(turb.sum()),
            "turbulent_fraction": float(turb.mean()),
            "hard_late_windows": int(hard_late.sum()),
            "hard_late_fraction": float(hard_late.mean()),
            "mean_future_q99_count_all": float(future_q99_counts.mean()),
            "mean_future_q99_count_turb": float(future_q99_counts[turb].mean()),
            "mean_future_q99_count_hard_late": float(future_q99_counts[hard_late].mean()),
        },
        "current_anchors": {
            "169c_best": summarize_model("results/block_ar/169c_best_v2_s3mrjspec_full_30d/summary.json"),
            "183c_best": summarize_model("results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json"),
        },
        "interpretation": {
            "read": (
                "Sparse hard windows are not vanishingly rare, but they are a small minority. "
                "This is enough to matter for risk, but small enough that uniform training losses "
                "will be dominated by common windows."
            )
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
