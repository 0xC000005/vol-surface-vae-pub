#!/usr/bin/env python
"""Casebook acceptance suite for the narrative prefix-latent product path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_product_acceptance_smoke import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_SMOKE_OUTPUT_DIR,
    run_product_acceptance_smoke,
)


DEFAULT_CASES = [
    {
        "name": "fragile_risk_on",
        "candidate_index": 18,
        "condition_report": (
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_condition_only_report_809a_fragile/"
            "condition_only_report.json"
        ),
    },
    {
        "name": "defensive_risk_off",
        "candidate_index": 22,
        "condition_report": (
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_condition_only_report_809b_defensive/"
            "condition_only_report.json"
        ),
    },
    {
        "name": "rates_selloff",
        "candidate_index": 18,
        "condition_report": (
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_condition_only_report_809c_rates/"
            "condition_only_report.json"
        ),
    },
]
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_product_acceptance_casebook_813a"
)


def summarize_casebook(case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in case_summaries:
        checks = item.get("checks", [])
        failed_checks = [
            str(row.get("name"))
            for row in checks
            if isinstance(row, dict) and not bool(row.get("passed"))
        ]
        rows.append(
            {
                "name": str(item.get("name", "")),
                "status": str(item.get("status", "")),
                "candidate_index": int(item.get("candidate_index", -1)),
                "failed_checks": failed_checks,
                "run_report": str(item.get("run_report", "")),
                "exported_start_json": str(item.get("exported_start_json", "")),
            }
        )
    return {
        "case_count": int(len(rows)),
        "pass_count": int(sum(row["status"] == "pass" for row in rows)),
        "fail_count": int(sum(row["status"] != "pass" for row in rows)),
        "overall_status": "pass"
        if rows and all(row["status"] == "pass" for row in rows)
        else "fail",
        "cases": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Prefix-Latent Product Acceptance Casebook",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Case count: `{summary.get('case_count')}`",
        f"- Pass count: `{summary.get('pass_count')}`",
        f"- Fail count: `{summary.get('fail_count')}`",
        "",
        "| Case | Status | Candidate | Failed Checks |",
        "|---|---:|---:|---|",
    ]
    for row in summary.get("cases", []):
        failed = ", ".join(row.get("failed_checks", []))
        lines.append(
            f"| `{row.get('name')}` | `{row.get('status')}` | "
            f"`{row.get('candidate_index')}` | {failed} |"
        )
    return "\n".join(lines)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_acceptance_casebook(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = DEFAULT_CASES[: int(args.case_count)] if args.case_count else DEFAULT_CASES
    case_summaries = []
    for case in selected_cases:
        case_output_dir = output_dir / str(case["name"])
        smoke_args = SimpleNamespace(
            condition_report=str(case["condition_report"]),
            output_dir=str(case_output_dir),
            candidate_index=int(case["candidate_index"]),
            samples=int(args.samples),
            steps=int(args.steps),
            chunk_size=int(args.chunk_size),
            device=str(args.device),
        )
        result = run_product_acceptance_smoke(smoke_args)
        case_summaries.append({**result, "name": str(case["name"])})
    summary = summarize_casebook(case_summaries)
    output = {
        "status": summary["overall_status"],
        "scope_note": (
            "No OpenAI calls. Runs product acceptance smoke across saved "
            "condition-only narrative reports and selected start templates."
        ),
        **summary,
        "artifact_paths": {
            "report": str(output_dir / "product_acceptance_casebook.json"),
            "markdown": str(output_dir / "product_acceptance_casebook.md"),
        },
    }
    write_json(output_dir / "product_acceptance_casebook.json", output)
    (output_dir / "product_acceptance_casebook.md").write_text(
        render_markdown(output).rstrip() + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = run_acceptance_casebook(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "pass_count": summary["pass_count"],
                "report": summary["artifact_paths"]["report"],
                "markdown": summary["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
