#!/usr/bin/env python
"""Build paper/demo evidence for fixed-start narrative conditionality."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BAKEOFF_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_matrix_858b/start_conditioned_bakeoff.json"
)
DEFAULT_CONTRAST_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_contrast_858b/"
    "fixed_start_narrative_contrast.json"
)
DEFAULT_GATE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_conditioning_gate_858b/"
    "fixed_start_conditioning_gate.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_conditioning_analysis_859a"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _top_markets_from_contrasts(rows: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in _as_list(row.get("largest_abs_market_gaps")):
            item = _as_dict(item)
            market = str(item.get("market", ""))
            if market:
                counts[market] += 1
    return [market for market, _ in counts.most_common(5)]


def _quality_by_start(bakeoff_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _as_list(bakeoff_report.get("rows")):
        row = _as_dict(row)
        grouped[str(row.get("start_name", ""))].append(row)
    output: dict[str, dict[str, Any]] = {}
    for start_name, rows in grouped.items():
        energy = []
        crps = []
        status_counts: Counter[str] = Counter()
        direction_counts: Counter[str] = Counter()
        support_match = []
        for row in rows:
            metrics = _as_dict(row.get("scenario_metrics"))
            energy.append(
                _as_float(metrics.get("energy_score_z_improvement_vs_persistence"))
            )
            crps.append(
                _as_float(metrics.get("ensemble_crps_z_improvement_vs_persistence"))
            )
            status_counts[str(row.get("validation_operational", ""))] += 1
            direction_counts[str(row.get("memory_prior_direction_status", ""))] += 1
            support_match.append(
                _as_float(row.get("memory_prior_support_weighted_match_rate"))
            )
        output[start_name] = {
            "run_count": len(rows),
            "mean_energy_improvement_vs_persistence": _mean(energy),
            "mean_crps_improvement_vs_persistence": _mean(crps),
            "mean_support_match_rate": _mean(support_match),
            "operational_status_counts": dict(status_counts),
            "direction_status_counts": dict(direction_counts),
        }
    return output


def _contrasts_by_start(
    contrast_report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _as_list(contrast_report.get("pairwise_contrasts")):
        row = _as_dict(row)
        grouped[str(row.get("start_name", ""))].append(row)
    return grouped


def build_conditioning_analysis(
    *,
    bakeoff_report: dict[str, Any],
    contrast_report: dict[str, Any],
    gate_report: dict[str, Any],
    top_n: int = 8,
) -> dict[str, Any]:
    """Combine gate, quality, and contrast metrics into a presentation packet."""
    quality = _quality_by_start(bakeoff_report)
    contrasts = _contrasts_by_start(contrast_report)
    start_summaries = []
    for block in _as_list(gate_report.get("start_block_assessments")):
        block = _as_dict(block)
        start_name = str(block.get("start_name", ""))
        quality_row = quality.get(start_name, {})
        contrast_rows = contrasts.get(start_name, [])
        start_summaries.append(
            {
                "start_name": start_name,
                "status": str(block.get("status", "")),
                "case_count": int(block.get("case_count", 0) or 0),
                "max_standardized_l2_gap": _as_float(
                    block.get("max_standardized_l2_gap")
                ),
                "median_standardized_l2_gap": _as_float(
                    block.get("median_standardized_l2_gap")
                ),
                "min_standardized_l2_gap": _as_float(
                    block.get("min_standardized_l2_gap")
                ),
                "top_separation_markets": _as_list(block.get("top_separation_markets"))
                or _top_markets_from_contrasts(contrast_rows),
                "warnings": _as_list(block.get("warnings")),
                "failures": _as_list(block.get("failures")),
                **quality_row,
            }
        )
    start_summaries = sorted(
        start_summaries,
        key=lambda row: _as_float(row.get("max_standardized_l2_gap")),
        reverse=True,
    )
    top_pairwise = sorted(
        [
            _as_dict(row)
            for rows in contrasts.values()
            for row in rows
            if isinstance(row, dict)
        ],
        key=lambda row: _as_float(row.get("standardized_l2_gap")),
        reverse=True,
    )[:top_n]
    damped = [
        row
        for row in start_summaries
        if "start_dampens_narrative_influence" in row.get("warnings", [])
    ]
    interpretation = [
        (
            "Conditionality is measured by holding the same starting level fixed "
            "within each start block and changing only the narrative condition."
        ),
        (
            "The support/direction audit passes separately from operational "
            "validation, so direction consistency should not be read as every "
            "rollout validation row passing."
        ),
    ]
    if damped:
        interpretation.append(
            "Some starts dampen narrative influence: "
            + ", ".join(row["start_name"] for row in damped)
            + ". These are audit warnings rather than hard failures."
        )
    return {
        "overall_status": str(gate_report.get("overall_status", "")),
        "scope_note": (
            "Paper/demo packet for fixed-start narrative conditionality. It shows "
            "that the same starting level can produce different scenario "
            "distributions when the risk-manager narrative changes."
        ),
        "headline": {
            "start_count": len(start_summaries),
            "case_count": int(contrast_report.get("case_count", 0) or 0),
            "damped_start_count": len(damped),
            "hard_fail_count": int(gate_report.get("hard_fail_count", 0) or 0),
            "check_warning_count": int(gate_report.get("warning_count", 0) or 0),
            "start_block_warning_count": int(
                gate_report.get("start_block_warning_count", 0) or 0
            ),
            "total_warning_count": int(gate_report.get("total_warning_count", 0) or 0),
        },
        "gate_checks": _as_list(gate_report.get("checks")),
        "start_summaries": start_summaries,
        "top_pairwise_contrasts": top_pairwise,
        "interpretation": interpretation,
    }


def _format(value: Any) -> str:
    return f"{_as_float(value):.3f}"


def render_markdown(analysis: dict[str, Any]) -> str:
    headline = _as_dict(analysis.get("headline"))
    lines = [
        "# Fixed-Start Narrative Conditionality Analysis",
        "",
        str(analysis.get("scope_note", "")),
        "",
        "## Headline",
        "",
        f"- Overall status: `{analysis.get('overall_status')}`",
        f"- Starts: `{headline.get('start_count')}`",
        f"- Cases: `{headline.get('case_count')}`",
        f"- Damped starts: `{headline.get('damped_start_count')}`",
        f"- Hard failures: `{headline.get('hard_fail_count')}`",
        f"- Check warnings: `{headline.get('check_warning_count')}`",
        f"- Start-block warnings: `{headline.get('start_block_warning_count')}`",
        "",
        "## Interpretation",
        "",
    ]
    for item in _as_list(analysis.get("interpretation")):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Start Summary",
            "",
            "| Start | Status | Max Gap | Median Gap | Energy Imp | CRPS Imp | Operational | Top Markets | Warnings |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in _as_list(analysis.get("start_summaries")):
        row = _as_dict(row)
        lines.append(
            f"| `{row.get('start_name')}` | `{row.get('status')}` | "
            f"`{_format(row.get('max_standardized_l2_gap'))}` | "
            f"`{_format(row.get('median_standardized_l2_gap'))}` | "
            f"`{_format(row.get('mean_energy_improvement_vs_persistence'))}` | "
            f"`{_format(row.get('mean_crps_improvement_vs_persistence'))}` | "
            f"`{row.get('operational_status_counts', {})}` | "
            f"{', '.join(str(x) for x in _as_list(row.get('top_separation_markets')))} | "
            f"{', '.join(str(x) for x in _as_list(row.get('warnings')))} |"
        )
    lines.extend(
        [
            "",
            "## Top Narrative Contrasts",
            "",
            "| Start | Left | Right | Standardized Gap | Largest Markets |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in _as_list(analysis.get("top_pairwise_contrasts")):
        row = _as_dict(row)
        markets = []
        for item in _as_list(row.get("largest_abs_market_gaps"))[:5]:
            item = _as_dict(item)
            markets.append(
                f"{item.get('market')}={_as_float(item.get('standardized_mean_gap')):.2f}"
            )
        lines.append(
            f"| `{row.get('start_name')}` | `{row.get('left_case')}` | "
            f"`{row.get('right_case')}` | "
            f"`{_format(row.get('standardized_l2_gap'))}` | "
            f"{'; '.join(markets)} |"
        )
    return "\n".join(lines) + "\n"


def write_plots(analysis: dict[str, Any], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starts = [row["start_name"] for row in analysis["start_summaries"]]
    max_gaps = [row["max_standardized_l2_gap"] for row in analysis["start_summaries"]]
    med_gaps = [
        row["median_standardized_l2_gap"] for row in analysis["start_summaries"]
    ]
    energy = [
        row.get("mean_energy_improvement_vs_persistence", 0.0)
        for row in analysis["start_summaries"]
    ]
    crps = [
        row.get("mean_crps_improvement_vs_persistence", 0.0)
        for row in analysis["start_summaries"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_path = output_dir / "fixed_start_narrative_gap_by_start.png"
    quality_path = output_dir / "fixed_start_quality_by_start.png"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(starts))
    ax.bar([i - 0.18 for i in x], max_gaps, width=0.36, label="max gap")
    ax.bar([i + 0.18 for i in x], med_gaps, width=0.36, label="median gap")
    ax.set_xticks(list(x), starts, rotation=30, ha="right")
    ax.set_ylabel("Standardized terminal-distribution gap")
    ax.set_title("Narrative separation with starting level held fixed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(gap_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.18 for i in x], energy, width=0.36, label="energy")
    ax.bar([i + 0.18 for i in x], crps, width=0.36, label="CRPS")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), starts, rotation=30, ha="right")
    ax.set_ylabel("Improvement vs persistence")
    ax.set_title("Distributional quality by fixed start")
    ax.legend()
    fig.tight_layout()
    fig.savefig(quality_path, dpi=160)
    plt.close(fig)
    return {"gap_plot": str(gap_path), "quality_plot": str(quality_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff-report", type=Path, default=DEFAULT_BAKEOFF_REPORT)
    parser.add_argument("--contrast-report", type=Path, default=DEFAULT_CONTRAST_REPORT)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    analysis = build_conditioning_analysis(
        bakeoff_report=_load_json(args.bakeoff_report),
        contrast_report=_load_json(args.contrast_report),
        gate_report=_load_json(args.gate_report),
        top_n=int(args.top_n),
    )
    analysis["inputs"] = {
        "bakeoff_report": str(args.bakeoff_report),
        "contrast_report": str(args.contrast_report),
        "gate_report": str(args.gate_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        analysis["plots"] = write_plots(analysis, args.output_dir)
    json_path = args.output_dir / "fixed_start_conditioning_analysis.json"
    markdown_path = args.output_dir / "fixed_start_conditioning_analysis.md"
    _write_json(json_path, analysis)
    markdown_path.write_text(render_markdown(analysis), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": analysis["overall_status"],
                "report": str(json_path),
                "markdown": str(markdown_path),
                "plots": analysis.get("plots", {}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
