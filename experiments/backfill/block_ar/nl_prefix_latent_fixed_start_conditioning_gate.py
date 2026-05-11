#!/usr/bin/env python
"""Gate fixed-start narrative conditionality for the prefix-latent product path."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_CONTRAST_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_contrast_857b/"
    "fixed_start_narrative_contrast.json"
)
DEFAULT_BAKEOFF_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_matrix_857b/"
    "start_conditioned_bakeoff.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_conditioning_gate_858a"
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


def _status(has_fail: bool, has_warning: bool) -> str:
    if has_fail:
        return "fail"
    return "warning" if has_warning else "pass"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _primary_variant_summary(bakeoff_report: dict[str, Any]) -> dict[str, Any]:
    summaries = [
        item
        for item in _as_list(bakeoff_report.get("variant_summary"))
        if isinstance(item, dict)
    ]
    if summaries:
        return summaries[0]
    rows = [
        row for row in _as_list(bakeoff_report.get("rows")) if isinstance(row, dict)
    ]
    energy = []
    crps = []
    for row in rows:
        metrics = row.get("scenario_metrics", {})
        if isinstance(metrics, dict):
            energy.append(
                _as_float(metrics.get("energy_score_z_improvement_vs_persistence"))
            )
            crps.append(
                _as_float(metrics.get("ensemble_crps_z_improvement_vs_persistence"))
            )
    return {
        "variant_name": "rows",
        "run_count": len(rows),
        "mean_energy_improvement_vs_persistence": (
            sum(energy) / len(energy) if energy else 0.0
        ),
        "mean_crps_improvement_vs_persistence": sum(crps) / len(crps) if crps else 0.0,
        "mean_support_weighted_match_rate": 0.0,
        "total_final_mixture_mismatches": 0,
        "operational_status_counts": {},
        "direction_status_counts": {},
    }


def _fixed_start_check(
    contrast_report: dict[str, Any], *, max_start_diff: float
) -> dict[str, Any]:
    start_max = _as_float(contrast_report.get("start_max_abs_diff"))
    bad_blocks = [
        block
        for block in _as_list(contrast_report.get("start_blocks"))
        if isinstance(block, dict)
        and _as_float(block.get("start_max_abs_diff")) > max_start_diff
    ]
    fail = (
        str(contrast_report.get("status")) == "fail"
        or start_max > max_start_diff
        or bool(bad_blocks)
    )
    return {
        "name": "fixed_start_equality",
        "status": "fail" if fail else "pass",
        "detail": (
            f"max_start_abs_diff={start_max:.6g}, "
            f"bad_blocks={len(bad_blocks)}, threshold={max_start_diff:.6g}"
        ),
    }


def _support_direction_check(
    contrast_report: dict[str, Any], *, min_support_match: float
) -> dict[str, Any]:
    bad_direction = 0
    warning_direction = 0
    low_support = 0
    mixture_mismatches = 0
    for case in _as_list(contrast_report.get("case_summaries")):
        if not isinstance(case, dict):
            continue
        direction = str(case.get("direction_status", "")).lower()
        if direction in {"fail", "reject"}:
            bad_direction += 1
        elif direction == "warning":
            warning_direction += 1
        if _as_float(case.get("support_match_rate")) < min_support_match:
            low_support += 1
        mixture_mismatches += int(case.get("final_mixture_mismatch_count", 0) or 0)
    fail = bad_direction > 0 or low_support > 0 or mixture_mismatches > 0
    warn = warning_direction > 0
    return {
        "name": "support_direction_consistency",
        "status": _status(fail, warn),
        "detail": (
            f"bad_direction={bad_direction}, warning_direction={warning_direction}, "
            f"low_support={low_support}, mixture_mismatches={mixture_mismatches}"
        ),
    }


def _scenario_quality_check(
    bakeoff_report: dict[str, Any],
    *,
    min_energy_improvement: float,
    min_crps_improvement: float,
) -> dict[str, Any]:
    summary = _primary_variant_summary(bakeoff_report)
    energy = _as_float(summary.get("mean_energy_improvement_vs_persistence"))
    crps = _as_float(summary.get("mean_crps_improvement_vs_persistence"))
    fail = energy < min_energy_improvement or crps < min_crps_improvement
    return {
        "name": "scenario_quality_vs_persistence",
        "status": "fail" if fail else "pass",
        "detail": (
            f"energy_improvement={energy:.4f}, crps_improvement={crps:.4f}, "
            f"thresholds=({min_energy_improvement:.4f},{min_crps_improvement:.4f})"
        ),
        "metrics": {
            "variant_name": str(summary.get("variant_name", "")),
            "run_count": int(summary.get("run_count", 0) or 0),
            "mean_energy_improvement_vs_persistence": energy,
            "mean_crps_improvement_vs_persistence": crps,
        },
    }


def _count_status(counts: dict[str, Any]) -> tuple[str, int, int]:
    fail_count = 0
    warning_count = 0
    for key, value in counts.items():
        count = int(value or 0)
        status = str(key).lower()
        if status == "warning":
            warning_count += count
        elif status != "pass":
            fail_count += count
    return _status(fail_count > 0, warning_count > 0), fail_count, warning_count


def _operational_validation_observation(
    bakeoff_report: dict[str, Any],
) -> dict[str, Any]:
    summary = _primary_variant_summary(bakeoff_report)
    operational_counts = summary.get("operational_status_counts", {})
    direction_counts = summary.get("direction_status_counts", {})
    operational_counts = (
        operational_counts if isinstance(operational_counts, dict) else {}
    )
    direction_counts = direction_counts if isinstance(direction_counts, dict) else {}
    op_status, op_fail, op_warning = _count_status(operational_counts)
    direction_status, direction_fail, direction_warning = _count_status(
        direction_counts
    )
    fail = op_status == "fail" or direction_status == "fail"
    warn = op_status == "warning" or direction_status == "warning"
    return {
        "name": "operational_validation_observation",
        "status": _status(fail, warn),
        "detail": (
            f"operational_counts={dict(operational_counts)}, "
            f"direction_counts={dict(direction_counts)}, "
            f"operational_fail={op_fail}, operational_warning={op_warning}, "
            f"direction_fail={direction_fail}, direction_warning={direction_warning}"
        ),
    }


def _start_block_assessments(
    contrast_report: dict[str, Any],
    *,
    min_block_max_gap: float,
    min_block_median_gap: float,
    damping_ratio: float,
) -> list[dict[str, Any]]:
    by_start: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _as_list(contrast_report.get("pairwise_contrasts")):
        if isinstance(row, dict):
            by_start[str(row.get("start_name", ""))].append(row)
    global_max = max(
        (
            _as_float(row.get("standardized_l2_gap"))
            for rows in by_start.values()
            for row in rows
        ),
        default=0.0,
    )
    assessments: list[dict[str, Any]] = []
    for block in _as_list(contrast_report.get("start_blocks")):
        if not isinstance(block, dict):
            continue
        start_name = str(block.get("start_name", ""))
        rows = by_start.get(start_name, [])
        gaps = sorted(
            [_as_float(row.get("standardized_l2_gap")) for row in rows],
            reverse=True,
        )
        max_gap = gaps[0] if gaps else 0.0
        median_gap = float(median(gaps)) if gaps else 0.0
        min_gap = gaps[-1] if gaps else 0.0
        warnings: list[str] = []
        failures: list[str] = []
        if not gaps:
            failures.append("no_pairwise_narrative_contrasts")
        if max_gap < min_block_max_gap:
            failures.append("narrative_separation_too_low")
        if median_gap < min_block_median_gap:
            warnings.append("median_narrative_separation_low")
        if global_max > 0.0 and max_gap < global_max * damping_ratio:
            warnings.append("start_dampens_narrative_influence")
        top_markets: list[str] = []
        for item in _as_list(rows[0].get("largest_abs_market_gaps") if rows else []):
            if isinstance(item, dict) and item.get("market") is not None:
                top_markets.append(str(item["market"]))
        assessments.append(
            {
                "start_name": start_name,
                "case_count": int(block.get("case_count", 0) or 0),
                "pair_count": len(gaps),
                "max_standardized_l2_gap": max_gap,
                "median_standardized_l2_gap": median_gap,
                "min_standardized_l2_gap": min_gap,
                "top_separation_markets": top_markets[:5],
                "warnings": warnings,
                "failures": failures,
                "status": _status(bool(failures), bool(warnings)),
            }
        )
    return assessments


def _narrative_separation_check(
    assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    fail_count = sum(1 for row in assessments if row["status"] == "fail")
    warning_count = sum(1 for row in assessments if row["status"] == "warning")
    return {
        "name": "fixed_start_narrative_separation",
        "status": _status(fail_count > 0, warning_count > 0),
        "detail": f"start_blocks={len(assessments)}, fail={fail_count}, warning={warning_count}",
    }


def evaluate_fixed_start_conditioning_gate(
    *,
    contrast_report: dict[str, Any],
    bakeoff_report: dict[str, Any],
    max_start_diff: float = 1e-6,
    min_support_match: float = 0.95,
    min_energy_improvement: float = 0.0,
    min_crps_improvement: float = 0.0,
    min_block_max_gap: float = 0.5,
    min_block_median_gap: float = 0.5,
    damping_ratio: float = 0.75,
) -> dict[str, Any]:
    """Evaluate whether narrative conditionality is visible at fixed starts."""
    block_assessments = _start_block_assessments(
        contrast_report,
        min_block_max_gap=min_block_max_gap,
        min_block_median_gap=min_block_median_gap,
        damping_ratio=damping_ratio,
    )
    checks = [
        _fixed_start_check(contrast_report, max_start_diff=max_start_diff),
        _support_direction_check(contrast_report, min_support_match=min_support_match),
        _scenario_quality_check(
            bakeoff_report,
            min_energy_improvement=min_energy_improvement,
            min_crps_improvement=min_crps_improvement,
        ),
        _operational_validation_observation(bakeoff_report),
        _narrative_separation_check(block_assessments),
    ]
    hard_fail_count = sum(1 for row in checks if row["status"] == "fail")
    check_warning_count = sum(1 for row in checks if row["status"] == "warning")
    start_block_warning_count = sum(
        1 for row in block_assessments if row["status"] == "warning"
    )
    total_warning_count = check_warning_count + start_block_warning_count
    return {
        "overall_status": _status(hard_fail_count > 0, total_warning_count > 0),
        "scope_note": (
            "Product/evaluation gate for fixed-start narrative conditionality. "
            "It verifies that the starting state is held fixed, support direction "
            "checks pass, distributional quality is not worse than persistence, "
            "operational validation counts are surfaced separately, and narrative "
            "changes create measurable scenario-distribution gaps."
        ),
        "hard_fail_count": hard_fail_count,
        "warning_count": check_warning_count,
        "start_block_warning_count": start_block_warning_count,
        "total_warning_count": total_warning_count,
        "thresholds": {
            "max_start_diff": max_start_diff,
            "min_support_match": min_support_match,
            "min_energy_improvement": min_energy_improvement,
            "min_crps_improvement": min_crps_improvement,
            "min_block_max_gap": min_block_max_gap,
            "min_block_median_gap": min_block_median_gap,
            "damping_ratio": damping_ratio,
        },
        "checks": checks,
        "start_block_assessments": block_assessments,
    }


def render_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Narrative Conditioning Gate",
        "",
        gate["scope_note"],
        "",
        f"- Overall status: `{gate.get('overall_status')}`",
        f"- Hard failures: `{gate.get('hard_fail_count')}`",
        f"- Check warnings: `{gate.get('warning_count')}`",
        f"- Start-block warnings: `{gate.get('start_block_warning_count')}`",
        f"- Total warnings: `{gate.get('total_warning_count')}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in _as_list(gate.get("checks")):
        lines.append(
            f"| `{row.get('name')}` | `{row.get('status')}` | {row.get('detail', '')} |"
        )
    lines.extend(
        [
            "",
            "## Start Blocks",
            "",
            "| Start | Status | Cases | Max Gap | Median Gap | Min Gap | Top Markets | Warnings | Failures |",
            "|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in _as_list(gate.get("start_block_assessments")):
        markets = ", ".join(_as_list(row.get("top_separation_markets")))
        warnings = ", ".join(_as_list(row.get("warnings")))
        failures = ", ".join(_as_list(row.get("failures")))
        lines.append(
            f"| `{row.get('start_name')}` | `{row.get('status')}` | "
            f"`{row.get('case_count')}` | "
            f"`{_as_float(row.get('max_standardized_l2_gap')):.3f}` | "
            f"`{_as_float(row.get('median_standardized_l2_gap')):.3f}` | "
            f"`{_as_float(row.get('min_standardized_l2_gap')):.3f}` | "
            f"{markets} | {warnings} | {failures} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast-report", type=Path, default=DEFAULT_CONTRAST_REPORT)
    parser.add_argument("--bakeoff-report", type=Path, default=DEFAULT_BAKEOFF_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-start-diff", type=float, default=1e-6)
    parser.add_argument("--min-support-match", type=float, default=0.95)
    parser.add_argument("--min-energy-improvement", type=float, default=0.0)
    parser.add_argument("--min-crps-improvement", type=float, default=0.0)
    parser.add_argument("--min-block-max-gap", type=float, default=0.5)
    parser.add_argument("--min-block-median-gap", type=float, default=0.5)
    parser.add_argument("--damping-ratio", type=float, default=0.75)
    args = parser.parse_args()

    gate = evaluate_fixed_start_conditioning_gate(
        contrast_report=_load_json(args.contrast_report),
        bakeoff_report=_load_json(args.bakeoff_report),
        max_start_diff=float(args.max_start_diff),
        min_support_match=float(args.min_support_match),
        min_energy_improvement=float(args.min_energy_improvement),
        min_crps_improvement=float(args.min_crps_improvement),
        min_block_max_gap=float(args.min_block_max_gap),
        min_block_median_gap=float(args.min_block_median_gap),
        damping_ratio=float(args.damping_ratio),
    )
    gate["inputs"] = {
        "contrast_report": str(args.contrast_report),
        "bakeoff_report": str(args.bakeoff_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "fixed_start_conditioning_gate.json"
    markdown_path = args.output_dir / "fixed_start_conditioning_gate.md"
    _write_json(json_path, gate)
    markdown_path.write_text(render_markdown(gate), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": gate["overall_status"],
                "report": str(json_path),
                "markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
