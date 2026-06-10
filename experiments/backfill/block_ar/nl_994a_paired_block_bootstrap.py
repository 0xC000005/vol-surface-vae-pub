#!/usr/bin/env python
"""994a paired moving-block-bootstrap summarizer for scenario-level eval reports.

Given two ``scenario_level_eval_report.json`` files that share query windows
(possibly the same file with two different method keys), compute paired
per-window metric deltas ``delta = b - a`` and a moving-block bootstrap CI for
the mean delta. The block resampling respects serial dependence induced by
overlapping 30-day futures: with the default block length of 30 windows, every
window inside a block keeps its neighbors, so dependence up to 30 windows is
preserved within blocks.

Orientation: for ensemble_crps_z / energy_score_z, NEGATIVE mean delta means
method/report B is better than A; for coverage_80 closer to 0.80 is better and
the delta sign alone is not a quality verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_METRICS = ("ensemble_crps_z", "energy_score_z", "coverage_80")


def load_method_scores(
    report_path: str | Path,
    *,
    method: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[int, dict[str, float]]:
    """Return {window_index: {metric: value}} for one method in a report."""

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = report.get("window_scores", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{report_path}: report has no window_scores")
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        methods = row.get("methods", {})
        if method not in methods:
            raise ValueError(
                f"{report_path}: window {row.get('window_index')} lacks "
                f"method {method!r}"
            )
        entry: dict[str, float] = {}
        for metric in metrics:
            value = methods[method].get(metric)
            if value is not None and np.isfinite(float(value)):
                entry[metric] = float(value)
        out[int(row["window_index"])] = entry
    return out


def paired_deltas(
    scores_a: dict[int, dict[str, float]],
    scores_b: dict[int, dict[str, float]],
    *,
    metric: str,
) -> tuple[np.ndarray, list[int]]:
    """Paired per-window deltas (b - a), ordered by window index."""

    shared = sorted(
        idx
        for idx in set(scores_a) & set(scores_b)
        if metric in scores_a[idx] and metric in scores_b[idx]
    )
    if not shared:
        raise ValueError(f"no shared windows with finite metric {metric!r}")
    deltas = np.asarray(
        [scores_b[idx][metric] - scores_a[idx][metric] for idx in shared],
        dtype=np.float64,
    )
    return deltas, shared


def moving_block_bootstrap_mean(
    series: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap distribution of the series mean.

    Overlapping blocks of length L are drawn with replacement, concatenated to
    cover n observations (trimmed), and the mean of each resampled series is
    returned. L is clamped to the series length.
    """

    values = np.asarray(series, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n == 0:
        raise ValueError("series must be non-empty")
    if int(n_boot) <= 0:
        raise ValueError("n_boot must be positive")
    length = max(1, min(int(block_length), n))
    n_starts = n - length + 1
    n_blocks = int(np.ceil(n / length))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, n_starts, size=(int(n_boot), n_blocks))
    offsets = np.arange(length)
    # (n_boot, n_blocks, length) -> trim to n columns
    samples = values[(starts[:, :, None] + offsets[None, None, :])].reshape(
        int(n_boot), -1
    )[:, :n]
    return samples.mean(axis=1)


def summarize_paired_metric(
    deltas: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    seed: int,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    boot_means = moving_block_bootstrap_mean(
        deltas, block_length=block_length, n_boot=n_boot, seed=seed
    )
    alpha = (1.0 - float(ci_level)) / 2.0
    length_used = max(1, min(int(block_length), int(deltas.size)))
    return {
        "n_windows": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci_level": float(ci_level),
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha)),
        "block_length_used": length_used,
        "n_effective_blocks": float(deltas.size / length_used),
        "n_boot": int(n_boot),
        "frac_boot_means_below_zero": float(np.mean(boot_means < 0.0)),
        "ci_excludes_zero": bool(
            float(np.quantile(boot_means, alpha)) > 0.0
            or float(np.quantile(boot_means, 1.0 - alpha)) < 0.0
        ),
    }


def run_paired_block_bootstrap(
    *,
    report_a: str | Path,
    report_b: str | Path,
    method_a: str,
    method_b: str,
    metrics: tuple[str, ...],
    block_length: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    scores_a = load_method_scores(report_a, method=method_a, metrics=metrics)
    scores_b = load_method_scores(report_b, method=method_b, metrics=metrics)
    results: dict[str, Any] = {}
    shared_windows: list[int] = []
    for metric in metrics:
        deltas, shared = paired_deltas(scores_a, scores_b, metric=metric)
        shared_windows = shared
        results[metric] = summarize_paired_metric(
            deltas,
            block_length=block_length,
            n_boot=n_boot,
            seed=seed,
        )
    return {
        "schema_version": "nl_994a_paired_block_bootstrap_v1",
        "report_a": str(report_a),
        "report_b": str(report_b),
        "method_a": str(method_a),
        "method_b": str(method_b),
        "delta_orientation": "delta = method_b - method_a (per shared window)",
        "shared_window_indices": [int(idx) for idx in shared_windows],
        "block_length": int(block_length),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "metrics": results,
        "note": (
            "moving-block bootstrap with overlapping blocks; block length in "
            "QUERY WINDOWS (default 30 to respect 30-day future overlap; with "
            "stride-5 queries this is conservative since dependence spans ~6 "
            "queries)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-a", required=True)
    parser.add_argument("--report-b", required=True)
    parser.add_argument("--method-a", default="narrative_generator_topk")
    parser.add_argument("--method-b", default="narrative_generator_topk")
    parser.add_argument(
        "--metrics", default="ensemble_crps_z,energy_score_z,coverage_80"
    )
    parser.add_argument("--block-length", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=994)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    metrics = tuple(
        name.strip() for name in str(args.metrics).split(",") if name.strip()
    )
    payload = run_paired_block_bootstrap(
        report_a=args.report_a,
        report_b=args.report_b,
        method_a=args.method_a,
        method_b=args.method_b,
        metrics=metrics,
        block_length=int(args.block_length),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"written: {output_path}")
    print(
        json.dumps(
            {metric: payload["metrics"][metric] for metric in metrics},
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
