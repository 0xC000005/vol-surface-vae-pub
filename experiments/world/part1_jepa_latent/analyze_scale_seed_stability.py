from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_RUNS = [
    (
        680,
        Path("results/world/masked_multiview_barlow_scale_head127.json"),
        "HEAD127",
    ),
    (
        681,
        Path("results/world/masked_multiview_barlow_scale_seed681_head131.json"),
        "HEAD131_seed681",
    ),
    (
        682,
        Path("results/world/masked_multiview_barlow_scale_seed682_head131.json"),
        "HEAD131_seed682",
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(seed: int, path: Path, label: str) -> dict[str, Any]:
    result = _load_json(path)
    alignment = result["val_metrics"]["view_alignment"]
    retrieval = alignment["retrieval"]
    raw_retrieval = result["raw_val_baseline"]["retrieval"]
    health = alignment["view_a_health"]
    barlow = alignment["barlow"]
    return {
        "seed": seed,
        "label": label,
        "path": str(path),
        "top1": float(retrieval["top1"]),
        "top5": float(retrieval["top5"]),
        "top10": float(retrieval["top10"]),
        "mrr": float(retrieval["mrr"]),
        "raw_top10": float(raw_retrieval["top10"]),
        "top10_minus_raw": float(retrieval["top10"] - raw_retrieval["top10"]),
        "effective_rank": float(health["effective_rank"]),
        "variance_min": float(health["variance_min"]),
        "offdiag_abs_mean": float(barlow["offdiag_abs_mean"]),
        "final_loss": float(result["history"][-1]["loss"]),
    }


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": mean(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def analyze_seed_stability() -> dict[str, Any]:
    rows = [_metric_row(seed, path, label) for seed, path, label in DEFAULT_RUNS]
    summary = {
        key: _summarize([float(row[key]) for row in rows])
        for key in (
            "top1",
            "top10",
            "mrr",
            "top10_minus_raw",
            "effective_rank",
            "variance_min",
            "offdiag_abs_mean",
            "final_loss",
        )
    }
    representation_stable = (
        summary["top10"]["min"] >= 0.83
        and summary["effective_rank"]["min"] >= 20.0
        and summary["variance_min"]["min"] >= 0.01
        and summary["offdiag_abs_mean"]["max"] <= 0.20
    )
    return {
        "assessment": "world_model_scaled_seed_stability_smoke",
        "date": "2026-05-10",
        "objective_family": "masked_multiview_invariance",
        "runs": rows,
        "summary": summary,
        "representation_stability_smoke_passed": representation_stable,
        "promotion_decision": "DO_NOT_PROMOTE",
        "interpretation": (
            "The scaled flat Barlow representation-health metrics are stable over "
            "three smoke-scale seeds. This upgrades scale/stability evidence for "
            "representation health, but it does not clear Part 1 because baseline "
            "superiority, regime probes, exact-state retention, and full-data "
            "stability remain unresolved."
        ),
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(result: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "Date: 2026-05-10",
        "",
        "## Iteration Type",
        "",
        "`experiment`",
        "",
        "## Objective Family",
        "",
        "`masked_multiview_invariance` scale/stability diagnostic.",
        "",
        "## Hypothesis",
        "",
        "If the HEAD127 scale improvement is real, same-config seeds should keep",
        "retrieval, rank, variance, and redundancy in the same healthy range.",
        "",
        "## Falsifier",
        "",
        "A new seed collapses rank or variance, loses most same-state retrieval,",
        "or shows materially worse redundancy than the HEAD127 seed.",
        "",
        "## Seed Results",
        "",
        "| seed | label | top1 | top10 | raw top10 | top10-raw | mrr | rank | var min | offdiag | final loss |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["runs"]:
        lines.append(
            "| {seed} | {label} | {top1} | {top10} | {raw_top10} | {top10_raw} | {mrr} | {rank} | {var_min} | {offdiag} | {loss} |".format(
                seed=row["seed"],
                label=row["label"],
                top1=_fmt(row["top1"]),
                top10=_fmt(row["top10"]),
                raw_top10=_fmt(row["raw_top10"]),
                top10_raw=_fmt(row["top10_minus_raw"]),
                mrr=_fmt(row["mrr"]),
                rank=_fmt(row["effective_rank"]),
                var_min=_fmt(row["variance_min"]),
                offdiag=_fmt(row["offdiag_abs_mean"]),
                loss=_fmt(row["final_loss"]),
            )
        )
    lines.extend(
        [
            "",
            "## Summary Ranges",
            "",
            "| metric | min | mean | max | std |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric, stats in result["summary"].items():
        lines.append(
            f"| {metric} | {_fmt(stats['min'])} | {_fmt(stats['mean'])} | {_fmt(stats['max'])} | {_fmt(stats['std'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Representation stability smoke passed: `{result['representation_stability_smoke_passed']}`.",
            f"- Promotion decision: `{result['promotion_decision']}`.",
            "",
            result["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze scaled Part 1 representation-health seed stability"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/world/scale_seed_stability_head131.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/world/reports/world_model_head131_scale_seed_stability.md"
        ),
    )
    parser.add_argument(
        "--report-title",
        default="World Model HEAD131: Scale Seed Stability",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = analyze_seed_stability()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(
        render_markdown(result, title=args.report_title),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "representation_stability_smoke_passed": result[
                    "representation_stability_smoke_passed"
                ],
                "promotion_decision": result["promotion_decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
