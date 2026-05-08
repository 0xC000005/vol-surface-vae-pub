#!/usr/bin/env python
"""Start-conditioned acceptance matrix for the narrative prefix-latent path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_product_acceptance_smoke import (  # noqa: E402
    run_product_acceptance_smoke,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    build_prefix_latent_run_args,
    export_historical_start_json_for_app,
)


FRAGILE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_report_809a_fragile/condition_only_report.json"
)
DEFENSIVE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_report_809b_defensive/condition_only_report.json"
)
RATES_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_report_809c_rates/condition_only_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_start_conditioned_acceptance_815b"
)

DEFAULT_MATRIX = [
    {
        "case_name": "fragile_risk_on",
        "start_name": "recommended_start_18",
        "condition_report": FRAGILE_REPORT,
        "candidate_index": 18,
        "expected_operational_status": "pass",
    },
    {
        "case_name": "fragile_risk_on",
        "start_name": "alternate_start_0",
        "condition_report": FRAGILE_REPORT,
        "candidate_index": 0,
        "expected_operational_status": "warning",
    },
    {
        "case_name": "defensive_risk_off",
        "start_name": "recommended_start_22",
        "condition_report": DEFENSIVE_REPORT,
        "candidate_index": 22,
        "expected_operational_status": "warning",
    },
    {
        "case_name": "rates_selloff",
        "start_name": "recommended_start_18",
        "condition_report": RATES_REPORT,
        "candidate_index": 18,
        "expected_operational_status": "pass",
    },
    {
        "case_name": "fragile_risk_on",
        "start_name": "extreme_user_start",
        "condition_report": FRAGILE_REPORT,
        "base_candidate_index": 18,
        "start_modifier": "extreme_out_of_support",
        "expected_operational_status": "fail",
    },
]


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


def apply_start_modifier(payload: dict[str, Any], modifier: str) -> dict[str, Any]:
    """Return a modified raw joint39 start JSON payload."""

    if modifier != "extreme_out_of_support":
        raise ValueError(f"unknown start modifier: {modifier!r}")
    values = payload.get("values_by_name")
    if not isinstance(values, dict):
        raise ValueError("start modifier requires values_by_name payload")
    modified = dict(payload)
    new_values = {str(key): float(value) for key, value in values.items()}
    for key, value in list(new_values.items()):
        lower = key.lower()
        if lower.startswith("iv:") or "iv_" in lower:
            new_values[key] = float(value) * 2.5
    overrides = {
        "factor:aaa_oas": 8.0,
        "factor:bbb_oas": 14.0,
        "factor:crude_oil": 180.0,
        "factor:gold": 3500.0,
        "factor:spx": float(new_values.get("factor:spx", 5000.0)) * 3.0,
        "factor:us2y": 8.0,
        "factor:us10y": 9.0,
        "factor:vix": 80.0,
    }
    for key, value in overrides.items():
        if key in new_values:
            new_values[key] = float(value)
    modified["label"] = "extreme_user_start_out_of_support"
    modified["values_by_name"] = new_values
    return modified


def expectation_met(actual: str, expected: str) -> bool:
    return str(actual) == str(expected)


def summarize_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": int(len(rows)),
        "expectation_fail_count": int(
            sum(not bool(row.get("expectation_met")) for row in rows)
        ),
        "overall_status": (
            "pass"
            if rows and all(bool(row.get("expectation_met")) for row in rows)
            else "fail"
        ),
        "cases": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Start-Conditioned Prefix-Latent Acceptance",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Case count: `{summary.get('case_count')}`",
        f"- Expectation fail count: `{summary.get('expectation_fail_count')}`",
        "",
        "| Case | Start | Expected | Actual | Met | Start Type | Start z | Weighted Start z | Run Report |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in summary.get("cases", []):
        lines.append(
            f"| `{row.get('case_name')}` | `{row.get('start_name')}` | "
            f"`{row.get('expected_operational_status')}` | "
            f"`{row.get('validation_operational')}` | "
            f"`{bool(row.get('expectation_met'))}` | "
            f"`{row.get('start_type')}` | "
            f"`{float(row.get('start_distance_z', 0.0)):.3f}` | "
            f"`{float(row.get('memory_prior_weighted_start_distance_z', 0.0)):.3f}` | "
            f"`{row.get('run_report')}` |"
        )
    return "\n".join(lines)


def _operational_row(report: dict[str, Any]) -> dict[str, Any]:
    for row in report.get("variant_rows", []):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return row
    return {}


def _row_from_run(
    *,
    case: dict[str, Any],
    start_type: str,
    run_report: str,
    report: dict[str, Any],
    product_report: str | None = None,
) -> dict[str, Any]:
    gate = report.get("validation_gate", {})
    operational = _operational_row(report)
    expected = str(case.get("expected_operational_status", ""))
    actual = str(gate.get("operational_status", ""))
    return {
        "case_name": str(case.get("case_name", "")),
        "start_name": str(case.get("start_name", "")),
        "start_type": str(start_type),
        "candidate_index": case.get(
            "candidate_index", case.get("base_candidate_index")
        ),
        "expected_operational_status": expected,
        "validation_overall": str(gate.get("overall_status", "")),
        "validation_operational": actual,
        "expectation_met": expectation_met(actual, expected),
        "start_distance_z": float(operational.get("start_distance_z", 0.0) or 0.0),
        "memory_prior_weighted_start_distance_z": float(
            operational.get("memory_prior_weighted_start_distance_z", 0.0) or 0.0
        ),
        "memory_prior_analogue_count": int(
            operational.get("memory_prior_analogue_count", 0) or 0
        ),
        "run_report": str(run_report),
        "product_report": str(product_report or ""),
    }


def run_candidate_start_case(
    *,
    case: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = output_dir / str(case["case_name"]) / str(case["start_name"])
    smoke_args = argparse.Namespace(
        condition_report=str(case["condition_report"]),
        output_dir=str(case_dir),
        candidate_index=int(case["candidate_index"]),
        samples=int(args.samples),
        steps=int(args.steps),
        chunk_size=int(args.chunk_size),
        device=str(args.device),
    )
    product = run_product_acceptance_smoke(smoke_args)
    run_report = str(product["run_report"])
    report = _load_json(run_report)
    return _row_from_run(
        case=case,
        start_type="historical_candidate",
        run_report=run_report,
        report=report,
        product_report=str(product["artifact_paths"]["report"]),
    )


def run_modified_start_case(
    *,
    case: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = output_dir / str(case["case_name"]) / str(case["start_name"])
    export_status, start_json, _preview = export_historical_start_json_for_app(
        str(case["base_candidate_index"]),
        output_dir=case_dir,
    )
    if not start_json:
        raise RuntimeError(export_status)
    modified = apply_start_modifier(_load_json(start_json), str(case["start_modifier"]))
    modified_start_json = case_dir / f"{case['start_name']}.json"
    _write_json(modified_start_json, modified)
    run_args = build_prefix_latent_run_args(
        start_mode="user_start_state",
        samples=int(args.samples),
        condition_report=str(case["condition_report"]),
        start_state_json=str(modified_start_json),
        output_dir=str(case_dir / "user_start_run"),
    )
    run_args.steps = int(args.steps)
    run_args.chunk_size = int(args.chunk_size)
    run_args.device = str(args.device)
    report = run_prefix_latent_story_smoke(run_args)
    run_report = str(report.get("artifact_paths", {}).get("report", ""))
    return _row_from_run(
        case=case,
        start_type=str(case["start_modifier"]),
        run_report=run_report,
        report=report,
    )


def run_start_conditioned_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = (
        DEFAULT_MATRIX[: int(args.case_count)] if args.case_count else DEFAULT_MATRIX
    )
    rows = []
    for case in selected_cases:
        if "candidate_index" in case:
            rows.append(
                run_candidate_start_case(case=case, output_dir=output_dir, args=args)
            )
        else:
            rows.append(
                run_modified_start_case(case=case, output_dir=output_dir, args=args)
            )
    summary = summarize_matrix(rows)
    output = {
        "status": summary["overall_status"],
        "scope_note": (
            "No OpenAI calls. Exercises the fixed-start narrative mixture "
            "contract across compatible, warning, and out-of-support starts."
        ),
        **summary,
        "artifact_paths": {
            "report": str(output_dir / "start_conditioned_acceptance.json"),
            "markdown": str(output_dir / "start_conditioned_acceptance.md"),
        },
    }
    _write_json(output["artifact_paths"]["report"], output)
    Path(output["artifact_paths"]["markdown"]).write_text(
        render_markdown(output).rstrip() + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-count", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_start_conditioned_acceptance(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "expectation_fail_count": summary["expectation_fail_count"],
                "report": summary["artifact_paths"]["report"],
                "markdown": summary["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
