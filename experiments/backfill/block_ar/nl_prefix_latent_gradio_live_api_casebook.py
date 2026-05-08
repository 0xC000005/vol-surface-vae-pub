#!/usr/bin/env python
"""Run a bounded live OpenAI Gradio API casebook for prefix-latent demos."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_gradio_api_smoke import (  # noqa: E402
    DEFAULT_URL,
    run_gradio_api_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    cached_prefix_casebook_update,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_api_casebook_828a"
)
DEFAULT_CASES = [
    "commodity_inflation_pressure:18",
    "dollar_liquidity_squeeze:22",
    "safe_haven_gold_bid:18",
]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Live Gradio API Casebook Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Pass count: `{summary['pass_count']}`",
        f"- Total OpenAI tokens: `{summary['total_openai_tokens']}`",
        f"- Min support candidates: `{summary['min_support_candidate_count']}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary["cases"]:
        lines.extend(
            [
                f"### {case['case_name']}",
                "",
                f"- Status: `{case['status']}`",
                f"- Start index: `{case['expected_start_index']}`",
                f"- Condition validation: `{case['condition_only_validation_status']}`",
                f"- Selected start: `{case['selected_start_status']}`",
                f"- Overall: `{case['overall_status']}`",
                f"- Forward warnings: `{case['condition_only_forward_warning_count']}`",
                f"- Support candidates: `{case['support_candidate_count']}`",
                f"- Support mode: `{case['support_prior_mode']}`",
                f"- Summary: `{case['summary_path']}`",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_from_choice(choice: str) -> dict[str, Any]:
    (
        story,
        use_explicit_start,
        explicit_start_index,
        _condition_only_story,
        _live_story,
        _condition_report,
        _status,
    ) = cached_prefix_casebook_update(choice)
    if not use_explicit_start:
        raise ValueError(f"casebook choice does not specify a start: {choice}")
    return {
        "choice": choice,
        "case_name": _slug(choice),
        "story": story,
        "expected_start_index": int(explicit_start_index),
    }


def run_live_api_casebook(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [_case_from_choice(choice) for choice in args.casebook_choices]
    case_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for case in cases:
        case_output_dir = output_dir / case["case_name"]
        try:
            result = run_gradio_api_smoke(
                SimpleNamespace(
                    url=str(args.url),
                    output_dir=str(case_output_dir),
                    mode="live_condition_only",
                    casebook_choice=case["choice"],
                    story=case["story"],
                    expected_start_index=int(case["expected_start_index"]),
                    samples=int(args.samples),
                    fan_market=str(args.fan_market),
                    redraw_market=str(args.redraw_market),
                )
            )
        except Exception as error:
            errors.append(f"{case['case_name']}: {type(error).__name__}: {error}")
            if not bool(args.continue_on_error):
                break
            result = {
                "status": "fail",
                "errors": [str(error)],
                "artifact_paths": {
                    "summary": str(case_output_dir / "gradio_api_smoke_summary.json")
                },
            }
        result["case_name"] = case["case_name"]
        result["casebook_choice"] = case["choice"]
        result["expected_start_index"] = int(case["expected_start_index"])
        result["summary_path"] = str(
            Path(result.get("artifact_paths", {}).get("summary", ""))
        )
        case_summaries.append(result)

    status_counts = Counter(str(case.get("status", "")) for case in case_summaries)
    total_openai_tokens = 0
    for case in case_summaries:
        usage = case.get("openai_usage", {})
        if isinstance(usage, dict):
            total_openai_tokens += int(usage.get("total_tokens", 0) or 0)
    support_counts = [
        int(case.get("support_candidate_count", 0) or 0) for case in case_summaries
    ]
    pass_count = status_counts.get("ok", 0)
    if pass_count != len(cases):
        errors.append("not_all_cases_passed")
    for case in case_summaries:
        if str(case.get("condition_only_validation_status", "")) != "pass":
            errors.append(f"{case['case_name']}: condition validation not pass")
        if int(case.get("condition_only_forward_warning_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: forward warning missing")
        if int(case.get("support_candidate_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: support candidates missing")
        if int(case.get("redraw_trace_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: redraw traces missing")

    summary = {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "url": str(args.url),
        "case_count": len(cases),
        "pass_count": pass_count,
        "status_counts": dict(status_counts),
        "total_openai_tokens": total_openai_tokens,
        "min_support_candidate_count": min(support_counts) if support_counts else 0,
        "casebook_choices": [case["choice"] for case in cases],
        "samples": int(args.samples),
        "fan_market": str(args.fan_market),
        "redraw_market": str(args.redraw_market),
        "cases": case_summaries,
        "artifact_paths": {
            "summary": str(output_dir / "gradio_live_api_casebook_summary.json"),
            "markdown": str(output_dir / "gradio_live_api_casebook_summary.md"),
        },
    }
    _write_json(output_dir / "gradio_live_api_casebook_summary.json", summary)
    _write_markdown(output_dir / "gradio_live_api_casebook_summary.md", summary)
    if errors:
        raise RuntimeError(f"Live Gradio API casebook failed: {errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    parser.add_argument(
        "--casebook-choice",
        dest="casebook_choices",
        action="append",
        default=None,
        help="Casebook choice to replay as a live condition-only story.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.casebook_choices is None:
        args.casebook_choices = list(DEFAULT_CASES)
    summary = run_live_api_casebook(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "status": summary["status"],
                "case_count": summary["case_count"],
                "pass_count": summary["pass_count"],
                "total_openai_tokens": summary["total_openai_tokens"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
