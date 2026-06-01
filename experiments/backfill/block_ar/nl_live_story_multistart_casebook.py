#!/usr/bin/env python
"""Run and summarize a multi-start live story-deck validation sweep."""

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

from experiments.backfill.block_ar.nl_live_story_deck_analysis import (  # noqa: E402
    build_live_story_deck_analysis,
    plot_terminal_delta_panel,
)
from experiments.backfill.block_ar.nl_prefix_latent_gradio_api_smoke import (  # noqa: E402
    DEFAULT_URL,
)
from experiments.backfill.block_ar.nl_prefix_latent_gradio_live_api_casebook import (  # noqa: E402
    run_live_api_casebook,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_story_multistart_950a"
)
DEFAULT_START_INDICES = [0, 18, 22, 40, 77]


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
        "# Multi-Start Live Story-Deck Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Starts requested: `{summary['start_count']}`",
        f"- Starts completed: `{summary['completed_start_count']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Pass count: `{summary['pass_count']}`",
        f"- Calibration applied: `{summary['calibration_applied_count']}/{summary['case_count']}`",
        f"- Total OpenAI tokens: `{summary['total_openai_tokens']}`",
        f"- Min calibration support gate: `{summary['min_calibration_support_gate']}`",
        f"- Max pairwise support Jaccard: `{summary['max_pairwise_support_jaccard']}`",
        "",
        "## Starts",
        "",
    ]
    for row in summary["starts"]:
        lines.extend(
            [
                f"### Start {row['start_index']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Cases: `{row.get('case_count', 0)}`",
                f"- Pass count: `{row.get('pass_count', 0)}`",
                f"- Calibration applied: `{row.get('calibration_applied_count', 0)}`",
                f"- Max support Jaccard: `{row.get('max_pairwise_support_jaccard', '')}`",
                f"- Summary: `{row.get('casebook_summary_path', '')}`",
                f"- Analysis: `{row.get('analysis_path', '')}`",
                "",
            ]
        )
    if summary["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- `{error}`" for error in summary["errors"])
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _start_indices(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "start_indices", None)
    if values:
        return [int(value) for value in values]
    return list(DEFAULT_START_INDICES)


def _analyze_start_casebook(
    *,
    start: int,
    start_dir: Path,
    casebook: dict[str, Any],
    no_plot: bool,
) -> dict[str, Any]:
    casebook_summary_path = str(
        casebook.get("artifact_paths", {}).get(
            "summary",
            str(start_dir / "gradio_live_api_casebook_summary.json"),
        )
    )
    analysis = build_live_story_deck_analysis(casebook_summary_path)
    analysis_path = start_dir / "fixed_start_live_story_deck_analysis.json"
    _write_json(analysis_path, analysis)
    plot_path = ""
    if not no_plot:
        plot_path = plot_terminal_delta_panel(
            analysis,
            start_dir / "fixed_start_terminal_mean_deltas.png",
        )
    return {
        "status": str(casebook.get("status", "")),
        "start_index": int(start),
        "case_count": int(casebook.get("case_count", 0) or 0),
        "pass_count": int(casebook.get("pass_count", 0) or 0),
        "calibration_applied_count": int(
            casebook.get("calibration_applied_count", 0) or 0
        ),
        "selected_start_warning_count": int(
            casebook.get("selected_start_warning_count", 0) or 0
        ),
        "condition_validation_warning_count": int(
            casebook.get("condition_validation_warning_count", 0) or 0
        ),
        "total_openai_tokens": int(casebook.get("total_openai_tokens", 0) or 0),
        "min_calibration_support_gate": float(
            casebook.get("min_calibration_support_gate", 0.0) or 0.0
        ),
        "mean_pairwise_support_jaccard": analysis.get(
            "mean_pairwise_support_jaccard"
        ),
        "max_pairwise_support_jaccard": analysis.get("max_pairwise_support_jaccard"),
        "casebook_summary_path": casebook_summary_path,
        "analysis_path": str(analysis_path),
        "plot_path": str(plot_path),
    }


def run_multistart_live_story_deck(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    starts = _start_indices(args)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for start in starts:
        start_dir = output_dir / f"start_{int(start)}"
        try:
            casebook = run_live_api_casebook(
                SimpleNamespace(
                    url=str(args.url),
                    output_dir=str(start_dir),
                    samples=int(args.samples),
                    fan_market=str(args.fan_market),
                    redraw_market=str(args.redraw_market),
                    casebook_choices=None,
                    use_default_story_deck=True,
                    default_story_cases=getattr(args, "default_story_cases", None),
                    fixed_start_index=int(start),
                    allow_start_warning=True,
                    allow_condition_warning=True,
                    continue_on_error=bool(getattr(args, "continue_on_error", False)),
                )
            )
            rows.append(
                _analyze_start_casebook(
                    start=int(start),
                    start_dir=start_dir,
                    casebook=casebook,
                    no_plot=bool(getattr(args, "no_plot", False)),
                )
            )
        except Exception as error:
            errors.append(f"start {int(start)}: {type(error).__name__}: {error}")
            failed_summary = _load_json_if_exists(
                start_dir / "gradio_live_api_casebook_summary.json"
            )
            if failed_summary is not None:
                try:
                    rows.append(
                        _analyze_start_casebook(
                            start=int(start),
                            start_dir=start_dir,
                            casebook=failed_summary,
                            no_plot=bool(getattr(args, "no_plot", False)),
                        )
                    )
                except Exception as analysis_error:
                    errors.append(
                        f"start {int(start)} analysis: "
                        f"{type(analysis_error).__name__}: {analysis_error}"
                    )
                    rows.append(
                        {
                            "status": "fail",
                            "start_index": int(start),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            else:
                rows.append(
                    {
                        "status": "fail",
                        "start_index": int(start),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            if not bool(getattr(args, "continue_on_error", False)):
                break

    completed = [row for row in rows if row.get("status") == "ok"]
    analyzed = [row for row in rows if row.get("analysis_path")]
    case_count = sum(int(row.get("case_count", 0) or 0) for row in analyzed)
    pass_count = sum(int(row.get("pass_count", 0) or 0) for row in analyzed)
    calibration_count = sum(
        int(row.get("calibration_applied_count", 0) or 0) for row in analyzed
    )
    selected_start_warning_count = sum(
        int(row.get("selected_start_warning_count", 0) or 0) for row in analyzed
    )
    condition_validation_warning_count = sum(
        int(row.get("condition_validation_warning_count", 0) or 0)
        for row in analyzed
    )
    gates = [
        float(row.get("min_calibration_support_gate", 0.0) or 0.0)
        for row in analyzed
    ]
    support_jaccards = [
        float(row.get("max_pairwise_support_jaccard"))
        for row in analyzed
        if row.get("max_pairwise_support_jaccard") is not None
    ]
    if len(completed) != len(starts):
        errors.append("not_all_starts_completed")
    if case_count and pass_count != case_count:
        errors.append("not_all_cases_passed")
    if case_count and calibration_count != case_count:
        errors.append("not_all_cases_calibrated")
    summary = {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "url": str(args.url),
        "start_indices": starts,
        "start_count": len(starts),
        "completed_start_count": len(completed),
        "analyzed_start_count": len(analyzed),
        "case_count": case_count,
        "pass_count": pass_count,
        "calibration_applied_count": calibration_count,
        "selected_start_warning_count": selected_start_warning_count,
        "condition_validation_warning_count": condition_validation_warning_count,
        "total_openai_tokens": sum(
            int(row.get("total_openai_tokens", 0) or 0) for row in analyzed
        ),
        "min_calibration_support_gate": min(gates) if gates else 0.0,
        "max_pairwise_support_jaccard": (
            max(support_jaccards) if support_jaccards else None
        ),
        "mean_of_start_max_support_jaccard": (
            sum(support_jaccards) / len(support_jaccards)
            if support_jaccards
            else None
        ),
        "samples": int(args.samples),
        "fan_market": str(args.fan_market),
        "redraw_market": str(args.redraw_market),
        "default_story_cases": getattr(args, "default_story_cases", None),
        "starts": rows,
        "artifact_paths": {
            "summary": str(output_dir / "multi_start_live_story_deck_summary.json"),
            "markdown": str(output_dir / "multi_start_live_story_deck_summary.md"),
        },
    }
    _write_json(output_dir / "multi_start_live_story_deck_summary.json", summary)
    _write_markdown(output_dir / "multi_start_live_story_deck_summary.md", summary)
    if errors and not bool(getattr(args, "continue_on_error", False)):
        raise RuntimeError(f"multi-start live story deck failed: {errors}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", dest="start_indices", action="append", type=int)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    parser.add_argument("--default-story-case", dest="default_story_cases", action="append")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    summary = run_multistart_live_story_deck(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "status": summary["status"],
                "start_count": summary["start_count"],
                "case_count": summary["case_count"],
                "pass_count": summary["pass_count"],
                "total_openai_tokens": summary["total_openai_tokens"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
