#!/usr/bin/env python
"""Product-level acceptance smoke for the narrative prefix-latent demo.

This harness makes no OpenAI calls. It exercises the production-facing contract:

1. export a valid raw joint39 start JSON from a support candidate;
2. preview that JSON;
3. run the condition-only narrative memory with the exported user start;
4. verify support diagnostics, validation gate, and fan-chart artifacts exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    build_prefix_latent_run_args,
    export_historical_start_json_for_app,
)


DEFAULT_CONDITION_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "risk_manager_story_gradio_demo/prefix_latent_live_smoke/"
    "condition_only_live/condition_only_report/condition_only_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_product_acceptance_smoke_812a"
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def acceptance_checks(
    *,
    report: dict[str, Any],
    preview_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gate = _as_dict(report.get("validation_gate"))
    query = _as_dict(report.get("cached_query"))
    memory_prior = _as_dict(query.get("memory_prior"))
    memory_prior_contract = str(query.get("memory_prior_contract", ""))
    generation = _as_dict(report.get("generation"))
    path_quantiles = _as_list(generation.get("path_quantiles"))
    variant_rows = _as_list(report.get("variant_rows"))
    user_start_rows = [
        row
        for row in variant_rows
        if isinstance(row, dict) and str(row.get("variant")) == "user_start_state"
    ]
    checks = [
        {
            "name": "validation_gate_not_fail",
            "passed": str(gate.get("overall_status")) != "fail"
            and str(gate.get("operational_status")) != "fail",
            "detail": (
                f"overall={gate.get('overall_status')}, "
                f"operational={gate.get('operational_status')}"
            ),
        },
        {
            "name": "user_start_variant_present",
            "passed": bool(user_start_rows),
            "detail": (
                user_start_rows[0].get("start_window_id", "") if user_start_rows else ""
            ),
        },
        {
            "name": "support_candidates_present",
            "passed": bool(_as_list(memory_prior.get("candidate_details"))),
            "detail": str(len(_as_list(memory_prior.get("candidate_details")))),
        },
        {
            "name": "fixed_start_memory_prior_contract",
            "passed": memory_prior_contract == "per_variant_narrative_and_fixed_start"
            and str(memory_prior.get("query_start_source")) == "provided_start_state",
            "detail": (
                f"contract={memory_prior_contract}, "
                f"query_start_source={memory_prior.get('query_start_source')}"
            ),
        },
        {
            "name": "user_start_mixture_diagnostics_present",
            "passed": bool(user_start_rows)
            and int(user_start_rows[0].get("memory_prior_analogue_count", 0) or 0) > 0
            and str(user_start_rows[0].get("memory_prior_query_start_source"))
            == "provided_start_state",
            "detail": (
                f"analogue_count="
                f"{user_start_rows[0].get('memory_prior_analogue_count', '') if user_start_rows else ''}"
            ),
        },
        {
            "name": "start_preview_present",
            "passed": bool(preview_rows)
            and any(row.get("Field") == "Field count" for row in preview_rows),
            "detail": str(len(preview_rows)),
        },
        {
            "name": "spx_fan_available",
            "passed": any(
                isinstance(row, dict)
                and str(row.get("market")) == "SPX"
                and str(row.get("analogue_key", "ALL")) == "ALL"
                for row in path_quantiles
            ),
            "detail": str(len(path_quantiles)),
        },
        {
            "name": "selected_iv_cell_fan_available",
            "passed": any(
                isinstance(row, dict)
                and str(row.get("market")) in {"IV_ATM_3M", "IV_ATM_1Y"}
                for row in path_quantiles
            ),
            "detail": str(len(path_quantiles)),
        },
    ]
    return checks


def acceptance_status(checks: list[dict[str, Any]]) -> str:
    return "pass" if all(bool(row.get("passed")) for row in checks) else "fail"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Prefix-Latent Product Acceptance Smoke",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Candidate index: `{summary.get('candidate_index')}`",
        f"- Exported start JSON: `{summary.get('exported_start_json')}`",
        f"- Run report: `{summary.get('run_report')}`",
        "",
        "## Checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for row in _as_list(summary.get("checks")):
        lines.append(
            f"| `{row.get('name')}` | `{bool(row.get('passed'))}` | "
            f"{row.get('detail', '')} |"
        )
    return "\n".join(lines)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_product_acceptance_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_status, start_json, preview = export_historical_start_json_for_app(
        str(args.candidate_index),
        output_dir=output_dir,
    )
    if not start_json:
        raise RuntimeError(export_status)
    run_args = build_prefix_latent_run_args(
        start_mode="user_start_state",
        samples=int(args.samples),
        condition_report=str(args.condition_report),
        start_state_json=str(start_json),
        output_dir=str(output_dir / "user_start_run"),
    )
    run_args.steps = int(args.steps)
    run_args.chunk_size = int(args.chunk_size)
    run_args.device = str(args.device)
    report = run_prefix_latent_story_smoke(run_args)
    preview_rows = preview.to_dict("records")
    checks = acceptance_checks(report=report, preview_rows=preview_rows)
    summary = {
        "status": acceptance_status(checks),
        "scope_note": (
            "No OpenAI calls. Uses an existing condition-only report, exports a "
            "candidate start JSON, previews it, and runs user_start_state rollout."
        ),
        "candidate_index": int(args.candidate_index),
        "condition_report": str(args.condition_report),
        "export_status": export_status,
        "exported_start_json": str(start_json),
        "preview_rows": preview_rows,
        "run_report": str(_as_dict(report.get("artifact_paths")).get("report", "")),
        "run_arrays": str(_as_dict(report.get("artifact_paths")).get("arrays", "")),
        "checks": checks,
        "artifact_paths": {
            "report": str(output_dir / "product_acceptance_smoke.json"),
            "markdown": str(output_dir / "product_acceptance_smoke.md"),
        },
    }
    write_json(output_dir / "product_acceptance_smoke.json", summary)
    (output_dir / "product_acceptance_smoke.md").write_text(
        render_markdown(summary).rstrip() + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-report", default=DEFAULT_CONDITION_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-index", type=int, default=18)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_product_acceptance_smoke(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "report": summary["artifact_paths"]["report"],
                "markdown": summary["artifact_paths"]["markdown"],
                "run_report": summary["run_report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
