#!/usr/bin/env python
"""Summarize seed stability for text/start memory diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = [
    "heldout_mean_target_cosine",
    "heldout_hard_negative_mean_gap",
    "heldout_hard_negative_mean_margin",
    "heldout_recall_at_1_test_pool",
    "heldout_recall_at_3_test_pool",
    "heldout_mean_top_train_cosine",
]


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def parse_report_arg(raw: str) -> tuple[int, str]:
    seed, path = str(raw).split(":", 1)
    return int(seed), path


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": _round(float(arr.mean())),
        "std": _round(float(arr.std(ddof=0))),
        "min": _round(float(arr.min())),
        "max": _round(float(arr.max())),
        "values": [_round(value) for value in values],
    }


def extract_method_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name, payload in report.get("results", {}).items():
        summary = payload.get("summary", {})
        rows[str(name)] = {
            metric: float(summary[metric])
            for metric in METRICS
            if metric in summary and summary[metric] is not None
        }
    return rows


def summarize_text_start_memory_stability(
    reports: list[tuple[int, dict[str, Any]]],
    *,
    baseline: str,
    candidate: str,
    target_cosine_floor_delta: float = -0.01,
    hard_negative_floor_delta: float = -0.05,
) -> dict[str, Any]:
    per_method: dict[str, dict[str, list[float]]] = {}
    paired_deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    seed_rows: list[dict[str, Any]] = []
    pass_count = 0
    for seed, report in reports:
        method_metrics = extract_method_metrics(report)
        if baseline not in method_metrics or candidate not in method_metrics:
            raise ValueError(f"report for seed {seed} missing baseline or candidate")
        for method, metrics in method_metrics.items():
            bucket = per_method.setdefault(method, {metric: [] for metric in METRICS})
            for metric in METRICS:
                if metric in metrics:
                    bucket[metric].append(float(metrics[metric]))
        deltas = {
            metric: float(method_metrics[candidate][metric])
            - float(method_metrics[baseline][metric])
            for metric in METRICS
            if metric in method_metrics[candidate]
            and metric in method_metrics[baseline]
        }
        for metric, value in deltas.items():
            paired_deltas[metric].append(float(value))
        passed = (
            deltas.get("heldout_mean_target_cosine", -999.0)
            >= float(target_cosine_floor_delta)
            and deltas.get("heldout_hard_negative_mean_gap", -999.0)
            >= float(hard_negative_floor_delta)
            and deltas.get("heldout_hard_negative_mean_margin", -999.0)
            >= float(hard_negative_floor_delta)
        )
        pass_count += int(passed)
        seed_rows.append(
            {
                "seed": int(seed),
                "passes_preservation_gate": bool(passed),
                "deltas": {metric: _round(value) for metric, value in deltas.items()},
            }
        )
    method_summary = {
        method: {
            metric: _metric_summary(values)
            for metric, values in metrics.items()
            if values
        }
        for method, metrics in per_method.items()
    }
    delta_summary = {
        metric: _metric_summary(values)
        for metric, values in paired_deltas.items()
        if values
    }
    status = "pass" if pass_count == len(reports) else "diagnostic_only"
    return {
        "status": status,
        "baseline": baseline,
        "candidate": candidate,
        "seed_count": len(reports),
        "preservation_pass_count": pass_count,
        "target_cosine_floor_delta": float(target_cosine_floor_delta),
        "hard_negative_floor_delta": float(hard_negative_floor_delta),
        "seed_rows": seed_rows,
        "method_summary": method_summary,
        "candidate_minus_baseline": delta_summary,
        "decision": (
            "candidate is stable enough for downstream scenario evaluation"
            if status == "pass"
            else "candidate is diagnostic only pending more stable preservation"
        ),
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Text/Start Memory Stability",
        "",
        f"Status: `{report['status']}`.",
        f"Baseline: `{report['baseline']}`.",
        f"Candidate: `{report['candidate']}`.",
        f"Preservation pass count: `{report['preservation_pass_count']}/{report['seed_count']}`.",
        "",
        "## Candidate Minus Baseline",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, stats in report["candidate_minus_baseline"].items():
        lines.append(
            f"| `{metric}` | {stats['mean']} | {stats['std']} | "
            f"{stats['min']} | {stats['max']} |"
        )
    lines.extend(
        [
            "",
            "## Seed Rows",
            "",
            "| Seed | Pass | Target Delta | Gap Delta | Margin Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["seed_rows"]:
        deltas = row["deltas"]
        lines.append(
            f"| {row['seed']} | {row['passes_preservation_gate']} | "
            f"{deltas.get('heldout_mean_target_cosine')} | "
            f"{deltas.get('heldout_hard_negative_mean_gap')} | "
            f"{deltas.get('heldout_hard_negative_mean_margin')} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="SEED:PATH")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-cosine-floor-delta", type=float, default=-0.01)
    parser.add_argument("--hard-negative-floor-delta", type=float, default=-0.05)
    args = parser.parse_args()
    reports = [
        (seed, _load_json(path))
        for seed, path in (parse_report_arg(item) for item in args.report)
    ]
    summary = summarize_text_start_memory_stability(
        reports,
        baseline=str(args.baseline),
        candidate=str(args.candidate),
        target_cosine_floor_delta=float(args.target_cosine_floor_delta),
        hard_negative_floor_delta=float(args.hard_negative_floor_delta),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["artifact_paths"] = {
        "report": str(output_dir / "text_start_memory_stability.json"),
        "markdown": str(output_dir / "text_start_memory_stability.md"),
    }
    _write_json(output_dir / "text_start_memory_stability.json", summary)
    write_markdown_report(output_dir / "text_start_memory_stability.md", summary)
    print(
        json.dumps(
            {
                "report": summary["artifact_paths"]["report"],
                "status": summary["status"],
                "preservation_pass_count": summary["preservation_pass_count"],
                "seed_count": summary["seed_count"],
                "candidate_minus_baseline": summary["candidate_minus_baseline"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
