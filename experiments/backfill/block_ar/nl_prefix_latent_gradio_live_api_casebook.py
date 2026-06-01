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
from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (  # noqa: E402
    UnqualifiedNarrativeError,
    assert_professional_story,
    default_casebook_stories,
    select_casebook_stories,
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


def _load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    input_path = Path(path)
    if not input_path.exists():
        return None
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _write_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Live Gradio API Casebook Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Pass count: `{summary['pass_count']}`",
        f"- Total OpenAI tokens: `{summary['total_openai_tokens']}`",
        f"- Min support candidates: `{summary['min_support_candidate_count']}`",
        f"- Calibration applied: `{summary['calibration_applied_count']}/{summary['case_count']}`",
        f"- Min calibration support gate: `{summary['min_calibration_support_gate']}`",
        f"- Allow unqualified narratives: `{summary['allow_unqualified_narratives']}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary["cases"]:
        lines.extend(
            [
                f"### {case.get('case_name', '')}",
                "",
                f"- Status: `{case.get('status', '')}`",
                f"- Start index: `{case.get('expected_start_index', '')}`",
                f"- Condition validation: `{case.get('condition_only_validation_status', '')}`",
                f"- Selected start: `{case.get('selected_start_status', '')}`",
                f"- Overall: `{case.get('overall_status', '')}`",
                f"- Forward warnings: `{case.get('condition_only_forward_warning_count', '')}`",
                f"- Support candidates: `{case.get('support_candidate_count', '')}`",
                f"- Support mode: `{case.get('support_prior_mode', '')}`",
                f"- Calibration applied: `{case.get('narrative_calibration_applied', '')}`",
                f"- Calibration beta: `{case.get('narrative_calibration_effective_beta', '')}`",
                f"- Calibration support gate: `{case.get('narrative_calibration_support_gate', '')}`",
                f"- Summary: `{case.get('summary_path', '')}`",
                "",
            ]
        )
        if case.get("errors"):
            lines.extend(["Errors:", ""])
            for error in case.get("errors", []):
                lines.append(f"- `{error}`")
            lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_from_choice(
    choice: str,
    *,
    allow_unqualified_narratives: bool = False,
) -> dict[str, Any]:
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
    assert_professional_story(
        story,
        context=f"casebook_choice:{choice}",
        allow_unqualified=allow_unqualified_narratives,
    )
    return {
        "choice": choice,
        "case_name": _slug(choice),
        "story": story,
        "expected_start_index": int(explicit_start_index),
    }


def _case_from_default_story(
    story_case: dict[str, str],
    start_index: int,
    *,
    allow_unqualified_narratives: bool = False,
) -> dict[str, Any]:
    name = str(story_case["name"])
    story = str(story_case["story"])
    assert_professional_story(
        story,
        context=f"default_story_case:{name}",
        allow_unqualified=allow_unqualified_narratives,
    )
    return {
        "choice": f"{name}:{int(start_index)}",
        "case_name": f"{_slug(name)}_{int(start_index)}",
        "story": story,
        "expected_start_index": int(start_index),
    }


def _selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    allow_unqualified = bool(getattr(args, "allow_unqualified_narratives", False))
    if bool(getattr(args, "use_default_story_deck", False)):
        story_names = getattr(args, "default_story_cases", None)
        fixed_start = int(getattr(args, "fixed_start_index"))
        return [
            _case_from_default_story(
                story_case,
                fixed_start,
                allow_unqualified_narratives=allow_unqualified,
            )
            for story_case in select_casebook_stories(case_names=story_names)
        ]
    return [
        _case_from_choice(
            choice,
            allow_unqualified_narratives=allow_unqualified,
        )
        for choice in args.casebook_choices
    ]


def run_live_api_casebook(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _selected_cases(args)
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
                    allow_start_warning=bool(
                        getattr(args, "allow_start_warning", False)
                    ),
                    allow_condition_warning=bool(
                        getattr(args, "allow_condition_warning", False)
                    ),
                )
            )
        except Exception as error:
            errors.append(f"{case['case_name']}: {type(error).__name__}: {error}")
            if not bool(args.continue_on_error):
                break
            summary_path = case_output_dir / "gradio_api_smoke_summary.json"
            result = _load_json_if_exists(summary_path) or {
                "artifact_paths": {"summary": str(summary_path)},
            }
            result["status"] = "fail"
            result.setdefault("errors", [str(error)])
            result.setdefault(
                "artifact_paths",
                {"summary": str(summary_path)},
            )
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
    calibration_gates = [
        float(case.get("narrative_calibration_support_gate", 0.0) or 0.0)
        for case in case_summaries
    ]
    calibration_applied_count = sum(
        1 for case in case_summaries if bool(case.get("narrative_calibration_applied"))
    )
    pass_count = status_counts.get("ok", 0)
    if pass_count != len(cases):
        errors.append("not_all_cases_passed")
    for case in case_summaries:
        condition_status = str(case.get("condition_only_validation_status", ""))
        if condition_status != "pass" and not (
            bool(getattr(args, "allow_condition_warning", False))
            and condition_status == "warning"
        ):
            errors.append(f"{case['case_name']}: condition validation not pass")
        selected_status = str(case.get("selected_start_status", ""))
        if selected_status != "pass" and not (
            bool(getattr(args, "allow_start_warning", False))
            and selected_status == "warning"
        ):
            errors.append(f"{case['case_name']}: selected start not pass")
        if int(case.get("condition_only_forward_warning_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: forward warning missing")
        if int(case.get("support_candidate_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: support candidates missing")
        if int(case.get("redraw_trace_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: redraw traces missing")
        if not bool(case.get("narrative_calibration_applied")):
            errors.append(f"{case['case_name']}: narrative calibration not applied")
        if float(case.get("narrative_calibration_support_gate", 0.0) or 0.0) <= 0.0:
            errors.append(f"{case['case_name']}: calibration support gate not positive")
        if int(case.get("narrative_calibration_active_direction_count", 0) or 0) < 1:
            errors.append(f"{case['case_name']}: calibration active direction missing")

    summary = {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "url": str(args.url),
        "case_count": len(cases),
        "pass_count": pass_count,
        "status_counts": dict(status_counts),
        "total_openai_tokens": total_openai_tokens,
        "min_support_candidate_count": min(support_counts) if support_counts else 0,
        "calibration_applied_count": calibration_applied_count,
        "min_calibration_support_gate": (
            min(calibration_gates) if calibration_gates else 0.0
        ),
        "allow_start_warning": bool(getattr(args, "allow_start_warning", False)),
        "allow_condition_warning": bool(
            getattr(args, "allow_condition_warning", False)
        ),
        "selected_start_warning_count": sum(
            1
            for case in case_summaries
            if str(case.get("selected_start_status", "")) == "warning"
        ),
        "condition_validation_warning_count": sum(
            1
            for case in case_summaries
            if str(case.get("condition_only_validation_status", "")) == "warning"
        ),
        "casebook_choices": [case["choice"] for case in cases],
        "use_default_story_deck": bool(getattr(args, "use_default_story_deck", False)),
        "allow_unqualified_narratives": bool(
            getattr(args, "allow_unqualified_narratives", False)
        ),
        "fixed_start_index": (
            int(args.fixed_start_index)
            if bool(getattr(args, "use_default_story_deck", False))
            else None
        ),
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
    parser.add_argument(
        "--use-default-story-deck",
        action="store_true",
        help=(
            "Replay the default professional narrative deck with one fixed "
            "historical start instead of cached casebook choices."
        ),
    )
    parser.add_argument(
        "--default-story-case",
        dest="default_story_cases",
        action="append",
        default=None,
        choices=[item["name"] for item in default_casebook_stories()],
        help="Default story-deck case name to include; repeat to select a subset.",
    )
    parser.add_argument("--fixed-start-index", type=int, default=22)
    parser.add_argument("--allow-start-warning", action="store_true")
    parser.add_argument("--allow-condition-warning", action="store_true")
    parser.add_argument(
        "--allow-unqualified-narratives",
        action="store_true",
        help=(
            "Explicit opt-out for legacy/smoke/ablation runs. By default, "
            "casebook narratives must satisfy the two specialist-doc "
            "risk-manager narrative standard."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.casebook_choices is None and not bool(args.use_default_story_deck):
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
