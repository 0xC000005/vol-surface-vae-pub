"""Attribute residual text-memory scenario gains against an incumbent.

This diagnostic is intentionally post-experiment only. It does not tune the
bridge. It summarizes where a candidate residual method beats or loses to the
current narrative support-mixture incumbent on held-out scenario metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

LOWER_IS_BETTER = {
    "ensemble_crps_z",
    "energy_score_z",
    "mean_path_mae_z",
    "mean_path_rmse_z",
    "terminal_mae_z",
}
HIGHER_IS_BETTER = {"coverage_80"}
DEFAULT_METRICS = (
    "ensemble_crps_z",
    "energy_score_z",
    "coverage_80",
    "mean_path_mae_z",
    "terminal_mae_z",
)


def _round(value: float | None, digits: int = 12) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _metric_delta(candidate: float, incumbent: float, metric: str) -> float:
    if metric in HIGHER_IS_BETTER:
        return candidate - incumbent
    return incumbent - candidate


def _pearson(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return _round(float(np.corrcoef(x, y)[0, 1]))


def _method_metric(row: dict, method: str, metric: str) -> float | None:
    value = row.get("methods", {}).get(method, {}).get(metric)
    if value is None:
        return None
    return float(value)


def _window_row(
    row: dict,
    *,
    incumbent_method: str,
    candidate_method: str,
    metrics: Iterable[str],
) -> dict:
    metric_values = {}
    for metric in metrics:
        incumbent = _method_metric(row, incumbent_method, metric)
        candidate = _method_metric(row, candidate_method, metric)
        if incumbent is None or candidate is None:
            continue
        delta = _metric_delta(candidate, incumbent, metric)
        metric_values[metric] = {
            "incumbent": _round(incumbent),
            "candidate": _round(candidate),
            "delta_positive_is_better": _round(delta),
            "candidate_better": bool(delta > 0.0),
        }

    top_cosines = [float(x) for x in row.get("top_train_cosines", [])]
    return {
        "window_id": row.get("window_id"),
        "window_index": row.get("window_index"),
        "block_window_index": row.get("block_window_index"),
        "top1_cosine": _round(top_cosines[0]) if top_cosines else None,
        "topk_cosine_mean": (
            _round(float(np.mean(top_cosines))) if top_cosines else None
        ),
        "topk_cosine_gap": (
            _round(top_cosines[0] - top_cosines[1]) if len(top_cosines) > 1 else None
        ),
        "top_train_indices": row.get("top_train_indices", []),
        "top_train_window_ids": row.get("top_train_window_ids", []),
        "metrics": metric_values,
    }


def _summarize_metric(rows: list[dict], metric: str) -> dict:
    deltas = [
        float(row["metrics"][metric]["delta_positive_is_better"])
        for row in rows
        if metric in row["metrics"]
    ]
    if not deltas:
        return {
            "count": 0,
            "mean_delta_positive_is_better": None,
            "median_delta_positive_is_better": None,
            "win_rate": None,
        }
    return {
        "count": len(deltas),
        "mean_delta_positive_is_better": _round(float(np.mean(deltas))),
        "median_delta_positive_is_better": _round(float(np.median(deltas))),
        "win_rate": _round(float(np.mean(np.asarray(deltas) > 0.0))),
        "positive_window_count": int(np.sum(np.asarray(deltas) > 0.0)),
        "negative_window_count": int(np.sum(np.asarray(deltas) < 0.0)),
    }


def _rank_by_metric(rows: list[dict], metric: str, *, reverse: bool) -> list[dict]:
    eligible = [row for row in rows if metric in row["metrics"]]
    eligible.sort(
        key=lambda row: float(row["metrics"][metric]["delta_positive_is_better"]),
        reverse=reverse,
    )
    ranked = []
    for row in eligible[:5]:
        ranked.append(
            {
                "window_id": row["window_id"],
                "window_index": row["window_index"],
                "block_window_index": row["block_window_index"],
                "top1_cosine": row["top1_cosine"],
                "topk_cosine_gap": row["topk_cosine_gap"],
                "delta_positive_is_better": row["metrics"][metric][
                    "delta_positive_is_better"
                ],
                "candidate": row["metrics"][metric]["candidate"],
                "incumbent": row["metrics"][metric]["incumbent"],
                "top_train_window_ids": row["top_train_window_ids"],
            }
        )
    return ranked


def analyze_report(
    report: dict,
    *,
    incumbent_method: str = "narrative_generator_topk",
    candidate_method: str = "narrative_residual_topk_a025",
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> dict:
    metric_tuple = tuple(metrics)
    rows = [
        _window_row(
            row,
            incumbent_method=incumbent_method,
            candidate_method=candidate_method,
            metrics=metric_tuple,
        )
        for row in report.get("window_scores", [])
    ]

    crps_gains = [
        float(row["metrics"]["ensemble_crps_z"]["delta_positive_is_better"])
        for row in rows
        if "ensemble_crps_z" in row["metrics"]
    ]
    top1 = [
        float(row["top1_cosine"])
        for row in rows
        if row["top1_cosine"] is not None and "ensemble_crps_z" in row["metrics"]
    ]
    gaps = [
        float(row["topk_cosine_gap"])
        for row in rows
        if row["topk_cosine_gap"] is not None and "ensemble_crps_z" in row["metrics"]
    ]

    result = {
        "status": "ok",
        "incumbent_method": incumbent_method,
        "candidate_method": candidate_method,
        "window_count": len(rows),
        "metric_summary": {
            metric: _summarize_metric(rows, metric) for metric in metric_tuple
        },
        "support_diagnostics": {
            "top1_cosine_mean": _round(float(np.mean(top1))) if top1 else None,
            "topk_cosine_gap_mean": _round(float(np.mean(gaps))) if gaps else None,
            "top1_cosine_gain_correlation": _pearson(top1, crps_gains),
            "topk_gap_gain_correlation": _pearson(gaps, crps_gains),
        },
        "best_windows_by_crps": _rank_by_metric(rows, "ensemble_crps_z", reverse=True),
        "worst_windows_by_crps": _rank_by_metric(
            rows, "ensemble_crps_z", reverse=False
        ),
        "window_attribution": rows,
    }
    return result


def _write_markdown(path: Path, analysis: dict) -> None:
    metrics = analysis["metric_summary"]
    lines = [
        "# Residual Scenario Attribution",
        "",
        f"- Incumbent: `{analysis['incumbent_method']}`",
        f"- Candidate: `{analysis['candidate_method']}`",
        f"- Windows: `{analysis['window_count']}`",
        "",
        "## Metric Summary",
        "",
        "| Metric | Mean Delta | Median Delta | Win Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, row in metrics.items():
        lines.append(
            "| {metric} | {mean} | {median} | {win} |".format(
                metric=metric,
                mean=row["mean_delta_positive_is_better"],
                median=row["median_delta_positive_is_better"],
                win=row["win_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Support Diagnostics",
            "",
            f"- Top-1 cosine mean: `{analysis['support_diagnostics']['top1_cosine_mean']}`",
            f"- Top-k cosine gap mean: `{analysis['support_diagnostics']['topk_cosine_gap_mean']}`",
            f"- Top-1 cosine vs CRPS-gain correlation: `{analysis['support_diagnostics']['top1_cosine_gain_correlation']}`",
            f"- Top-k gap vs CRPS-gain correlation: `{analysis['support_diagnostics']['topk_gap_gain_correlation']}`",
            "",
            "## Best CRPS Windows",
            "",
        ]
    )
    for row in analysis["best_windows_by_crps"]:
        lines.append(
            f"- `{row['window_id']}`: delta `{row['delta_positive_is_better']}`, "
            f"top1 `{row['top1_cosine']}`, gap `{row['topk_cosine_gap']}`"
        )
    lines.extend(["", "## Worst CRPS Windows", ""])
    for row in analysis["worst_windows_by_crps"]:
        lines.append(
            f"- `{row['window_id']}`: delta `{row['delta_positive_is_better']}`, "
            f"top1 `{row['top1_cosine']}`, gap `{row['topk_cosine_gap']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--incumbent-method", default="narrative_generator_topk")
    parser.add_argument("--candidate-method", default="narrative_residual_topk_a025")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    analysis = analyze_report(
        report,
        incumbent_method=args.incumbent_method,
        candidate_method=args.candidate_method,
        metrics=metrics,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "residual_scenario_attribution.json"
    md_path = output_dir / "residual_scenario_attribution.md"
    json_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(md_path, analysis)
    print(
        json.dumps({"json": str(json_path), "markdown": str(md_path)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
