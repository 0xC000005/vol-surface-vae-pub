#!/usr/bin/env python
"""Small live narrative casebook for prefix-latent TestFlight diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (
    load_specialist_standards,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_live_casebook_792a"
)
DEFAULT_SCRIPT = "experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py"
PROFESSIONAL_STORY_SECTIONS = (
    "Scenario title:",
    "Mechanical summary:",
    "Dominant mechanism:",
    "Trigger and transmission:",
    "Cross-asset reaction:",
    "Portfolio/risk implication:",
    "Evidence and ambiguity:",
    "No-forecast caveat:",
    "Warning-only forward risk:",
)


class UnqualifiedNarrativeError(ValueError):
    """Raised when a narrative fails the professional risk-manager story gate."""


def professional_story_standard_metadata() -> dict[str, Any]:
    """Return the specialist-document standard used by default casebook stories."""

    standards = load_specialist_standards()
    return {
        "rubric_text": standards.rubric_text,
        "source_documents": [
            {"path": str(doc.path), "sha256": str(doc.sha256)}
            for doc in standards.documents
        ],
        "required_sections": list(PROFESSIONAL_STORY_SECTIONS),
    }


def validate_professional_story(text: str) -> list[str]:
    """Return missing professional sections for a casebook story."""

    story = str(text or "")
    missing = [
        section for section in PROFESSIONAL_STORY_SECTIONS if section not in story
    ]
    if "not a forecast" not in story.lower():
        missing.append("phrase:not a forecast")
    if "current/recent" not in story.lower():
        missing.append("phrase:current/recent")
    return missing


def assert_professional_story(
    text: str,
    *,
    context: str = "narrative",
    allow_unqualified: bool = False,
) -> list[str]:
    """Validate a story against the specialist-document professional standard."""

    missing = validate_professional_story(text)
    if missing and not allow_unqualified:
        raise UnqualifiedNarrativeError(
            f"{context} is not risk-manager qualified; missing: {', '.join(missing)}. "
            "Use an explicit allow_unqualified_narratives option only for labeled "
            "ablation, smoke, or legacy tests."
        )
    return missing


def assert_professional_story_deck(
    stories: list[dict[str, str]],
    *,
    allow_unqualified: bool = False,
) -> list[dict[str, str]]:
    """Validate all stories in a deck and return the deck unchanged."""

    for item in stories:
        assert_professional_story(
            str(item.get("story", "")),
            context=f"default_casebook_story:{item.get('name', 'unknown')}",
            allow_unqualified=allow_unqualified,
        )
    return stories


def _slug(text: str) -> str:
    keep = []
    for char in str(text).lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_") or "case"


def default_casebook_stories() -> list[dict[str, str]]:
    """Return document-qualified professional live TestFlight narratives.

    The two specialist documents under `research/narrative_specialist/` define
    the required professional style: start from the mechanical state, name the
    dominant mechanism, describe transmission and cross-asset reaction, state
    portfolio relevance, preserve ambiguity, and keep forward-looking language
    warning-only. These stories are fixed-start counterfactual inputs; they
    describe the current/recent condition rather than prescribing the future.
    """

    stories = [
        {
            "name": "fragile_risk_on_rebound",
            "story": (
                "Scenario title: Fragile risk-on rebound. Mechanical summary: "
                "The current/recent tape shows equities recovering, volatility "
                "compressing, credit spreads stabilizing or tightening, and "
                "investors moving back toward carry and higher-beta exposure. "
                "Dominant mechanism: risk appetite is returning after a prior "
                "stress episode, but the recovery looks liquidity-led rather "
                "than clearly fundamentals-led. Trigger and transmission: lower "
                "realized volatility and calmer credit conditions reduce the "
                "need for defensive hedges, which supports equity duration and "
                "carry-sensitive exposures. Cross-asset reaction: SPX is firmer, "
                "VIX is lower, BBB OAS is stable to tighter, the dollar is not "
                "the dominant driver, and commodities are mixed to firmer. "
                "Portfolio/risk implication: portfolios with short-volatility, "
                "long-equity, and carry exposure benefit, while downside remains "
                "concentrated in a renewed volatility shock. Evidence and "
                "ambiguity: the evidence is market-price behavior, not a "
                "confirmed macro catalyst; the ambiguity is whether the rebound "
                "is durable or just positioning relief. No-forecast caveat: this "
                "is not a forecast; it is a current/recent conditioning story "
                "for scenario generation. Warning-only forward risk: a volatility "
                "reversal could unwind the rebound, but that statement is a risk "
                "flag rather than a requested future path."
            ),
        },
        {
            "name": "defensive_risk_off_shock",
            "story": (
                "Scenario title: Defensive risk-off shock. Mechanical summary: "
                "The current/recent market state shows equities selling off, "
                "volatility rising, credit spreads widening, and investors "
                "moving toward dollar liquidity and safe-haven assets. Dominant "
                "mechanism: risk tolerance is being withdrawn across assets as "
                "liquidity preference rises. Trigger and transmission: a shock "
                "to confidence or financing conditions forces de-risking, which "
                "raises hedging demand, weakens equity beta, and pushes credit "
                "risk premia wider. Cross-asset reaction: SPX is lower, VIX is "
                "higher, BBB OAS is wider, DXY is firmer, rates are biased lower "
                "if duration is treated as a safe asset, and gold can be "
                "supported. Portfolio/risk implication: long-equity and credit "
                "spread exposures carry the main downside, while duration and "
                "safe-haven allocations may provide partial ballast. Evidence "
                "and ambiguity: the story is inferred from cross-asset stress; "
                "the precise catalyst is unspecified. No-forecast caveat: this "
                "is not a forecast; it describes the current/recent risk-off "
                "conditioning state. Warning-only forward risk: funding stress "
                "could broaden into forced de-risking, but that is not used as a "
                "desired future outcome."
            ),
        },
        {
            "name": "rates_selloff_tightening_fear",
            "story": (
                "Scenario title: Rates-led tightening scare. Mechanical summary: "
                "The current/recent state has Treasury yields moving higher, "
                "equities struggling with duration pressure, the dollar firming, "
                "and volatility rising in a controlled but persistent way. "
                "Dominant mechanism: higher discount rates are tightening "
                "financial conditions and challenging long-duration risk assets. "
                "Trigger and transmission: resilient inflation or policy-rate "
                "repricing lifts yields, which compresses equity multiples, "
                "supports the dollar, and pressures rate-sensitive exposures. "
                "Cross-asset reaction: US10Y is higher, SPX is softer, DXY is "
                "firmer, VIX is higher, and credit is vulnerable when the rates "
                "move damages risk appetite. Portfolio/risk implication: "
                "duration-heavy equity, long-bond, and EM-FX exposures are "
                "vulnerable; financials and value may be less exposed depending "
                "on curve shape. Evidence and ambiguity: the narrative is built "
                "from rates, equity, FX, and volatility moves; it does not assume "
                "a specific central-bank headline. No-forecast caveat: this is "
                "not a forecast; it is a current/recent conditioning story. "
                "Warning-only forward risk: another rates leg could pressure "
                "risk assets, but the generator must produce the future "
                "distribution."
            ),
        },
        {
            "name": "commodity_inflation_pressure",
            "story": (
                "Scenario title: Commodity inflation pressure. Mechanical "
                "summary: The current/recent market state shows crude oil and "
                "other inflation-sensitive assets firming, Treasury yields moving "
                "higher, equities softening or trading choppily, and volatility "
                "remaining supported. Dominant mechanism: a supply-side or "
                "commodity-led inflation impulse is tightening real financial "
                "conditions. Trigger and transmission: higher energy prices raise "
                "inflation concern, lift nominal yields, pressure margins, and "
                "make equity duration less attractive. Cross-asset reaction: "
                "crude is higher, US10Y is higher, SPX is softer, VIX is firmer, "
                "gold is supported as an inflation hedge, and the dollar is "
                "firmer when global liquidity is tight. Portfolio/risk implication: "
                "energy and inflation-hedge exposures can offset pressure on "
                "long-duration equities and rate-sensitive assets. Evidence and "
                "ambiguity: the story is based on cross-asset inflation pricing; "
                "it does not assert a specific geopolitical event. No-forecast "
                "caveat: this is not a forecast; it is a current/recent "
                "conditioning narrative. Warning-only forward risk: persistent "
                "inflation concern could keep financial conditions tight, but "
                "that is a warning rather than a prescribed scenario path."
            ),
        },
        {
            "name": "dollar_liquidity_squeeze",
            "story": (
                "Scenario title: Dollar liquidity squeeze. Mechanical summary: "
                "The current/recent tape has the dollar strengthening, USDJPY "
                "moving higher, equities under pressure, credit spreads widening, "
                "and volatility elevated. Dominant mechanism: global dollar "
                "liquidity is becoming scarcer, so investors reduce risk and seek "
                "cash-like dollar exposure. Trigger and transmission: tighter "
                "funding conditions or dollar scarcity supports DXY and USDJPY, "
                "raises hedging demand, weakens equity beta, and makes credit "
                "risk more expensive. Cross-asset reaction: DXY is higher, "
                "USDJPY is higher, SPX is lower, VIX is higher, BBB OAS is "
                "wider, and commodities can weaken if global demand concern "
                "dominates. Portfolio/risk implication: unhedged foreign assets, "
                "EM risk, credit beta, and leveraged carry trades are the main "
                "vulnerabilities. Evidence and ambiguity: the current evidence is "
                "a dollar-plus-risk-off configuration; the exact funding catalyst "
                "is unspecified. No-forecast caveat: this is not a forecast; it "
                "describes the current/recent conditioning state. Warning-only "
                "forward risk: funding stress could spill into broader "
                "de-risking, but the model is not being told to force that future."
            ),
        },
        {
            "name": "safe_haven_gold_bid",
            "story": (
                "Scenario title: Safe-haven gold bid. Mechanical summary: The "
                "current/recent market state shows gold supported, Treasury "
                "yields lower, equities choppy to weaker, and volatility elevated "
                "while the dollar is not the only defensive channel. Dominant "
                "mechanism: investors are paying for safety and convexity rather "
                "than adding broad cyclical risk. Trigger and transmission: "
                "growth uncertainty, policy credibility concern, or geopolitical "
                "risk can raise demand for stores of value and duration, while "
                "limiting equity risk appetite. Cross-asset reaction: gold is "
                "higher, US10Y is lower, VIX is higher, SPX is mixed to weaker, "
                "and DXY is mixed when safe-haven demand is split between gold "
                "and dollars. Portfolio/risk implication: gold and duration can "
                "hedge part of the portfolio, while equity beta and short-vol "
                "exposure remain vulnerable to renewed stress. Evidence and "
                "ambiguity: the evidence is a safe-haven configuration, not a "
                "named event; equity direction is less clean than in a pure "
                "risk-off shock. No-forecast caveat: this is not a forecast; it "
                "is a current/recent conditioning narrative. Warning-only forward "
                "risk: safe-haven demand could broaden into risk-off behavior, "
                "but that language is warning-only."
            ),
        },
    ]
    return assert_professional_story_deck(stories)


def select_casebook_stories(
    *,
    case_names: list[str] | None = None,
    case_count: int | None = None,
    allow_unqualified: bool = False,
) -> list[dict[str, str]]:
    """Select casebook stories by exact name or by default ordering."""

    stories = default_casebook_stories()
    if case_names:
        by_name = {str(item["name"]): item for item in stories}
        missing = [name for name in case_names if name not in by_name]
        if missing:
            available = ", ".join(sorted(by_name))
            raise ValueError(
                f"unknown case_name(s): {', '.join(missing)}; available: {available}"
            )
        return assert_professional_story_deck(
            [by_name[name] for name in case_names],
            allow_unqualified=allow_unqualified,
        )
    if case_count:
        return assert_professional_story_deck(
            stories[: int(case_count)],
            allow_unqualified=allow_unqualified,
        )
    return assert_professional_story_deck(stories, allow_unqualified=allow_unqualified)


def build_case_command(
    *,
    story: str,
    output_dir: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build a subprocess command for one live-story prefix-latent case."""

    return [
        sys.executable,
        str(args.script),
        "--live-story",
        "--story",
        str(story),
        "--output-dir",
        str(output_dir),
        "--steps",
        str(int(args.steps)),
        "--samples",
        str(int(args.samples)),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--start-mode",
        str(args.start_mode),
        "--device",
        str(args.device),
        "--grounding-model",
        str(args.grounding_model),
        "--embedding-model",
        str(args.embedding_model),
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


def summarize_case_report(
    *,
    case_name: str,
    report_path: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Extract a compact diagnostic row from one prefix-latent story report."""

    query = report.get("cached_query", {})
    gate = report.get("validation_gate", {})
    generation = report.get("generation", {})
    grounding = query.get("grounding") if isinstance(query, dict) else {}
    if not isinstance(grounding, dict):
        grounding = {}
    cases = gate.get("cases", []) if isinstance(gate, dict) else []
    operational_cases = [
        item
        for item in cases
        if isinstance(item, dict) and bool(item.get("is_operational"))
    ]
    selected_case = operational_cases[0] if operational_cases else {}
    memory_cosines = [
        float(item["input_memory_cosine"])
        for item in cases
        if isinstance(item, dict) and item.get("input_memory_cosine") is not None
    ]
    warning_counts = gate.get("warning_counts", {}) if isinstance(gate, dict) else {}
    fail_counts = gate.get("fail_counts", {}) if isinstance(gate, dict) else {}
    return {
        "case_name": str(case_name),
        "report_path": str(report_path),
        "condition_source": str(query.get("condition_source", "")),
        "overall_status": str(gate.get("overall_status", "")),
        "operational_status": str(gate.get("operational_status", "")),
        "selected_start_status": str(
            gate.get("selected_start_status", gate.get("operational_status", ""))
        ),
        "diagnostic_baseline_status": str(gate.get("diagnostic_baseline_status", "")),
        "stress_status": str(gate.get("stress_status", "")),
        "warning_counts": warning_counts if isinstance(warning_counts, dict) else {},
        "fail_counts": fail_counts if isinstance(fail_counts, dict) else {},
        "selected_start_memory_cosine": selected_case.get("input_memory_cosine"),
        "selected_start_terminal_shift_z": selected_case.get(
            "terminal_mean_abs_delta_z"
        ),
        "selected_start_warnings": selected_case.get("warnings", []),
        "min_memory_cosine": min(memory_cosines) if memory_cosines else None,
        "mean_memory_cosine": (
            sum(memory_cosines) / len(memory_cosines) if memory_cosines else None
        ),
        "grounding_frame": str(grounding.get("narrative_frame", "")),
        "grounding_warning_count": len(grounding.get("grounding_warnings", [])),
        "generated_state_shape": generation.get("generated_state_shape"),
    }


def run_casebook(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stories = select_casebook_stories(
        case_names=getattr(args, "case_name", None),
        case_count=int(args.case_count),
    )
    case_rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for case_no, item in enumerate(stories, start=1):
        case_name = str(item["name"])
        case_dir = output_dir / f"{case_no:02d}_{_slug(case_name)}"
        command = build_case_command(
            story=str(item["story"]),
            output_dir=str(case_dir),
            args=args,
        )
        commands.append(command)
        subprocess.run(command, cwd=Path.cwd(), check=True)
        report_path = case_dir / "prefix_latent_story_smoke_report.json"
        report = _load_json(report_path)
        case_rows.append(
            summarize_case_report(
                case_name=case_name,
                report_path=str(report_path),
                report=report,
            )
        )
    status_counts: dict[str, int] = {}
    for row in case_rows:
        status = str(row.get("overall_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "status": "ok",
        "scope_note": (
            "Small live OpenAI prefix-latent casebook. Each case calls OpenAI "
            "for grounding and embedding, then runs the decoded-prefix frozen "
            "joint39 rollout."
        ),
        "case_count": len(case_rows),
        "status_counts": status_counts,
        "cases": case_rows,
        "commands": commands,
        "artifact_paths": {
            "summary": str(output_dir / "live_prefix_casebook_summary.json"),
        },
    }
    _write_json(output_dir / "live_prefix_casebook_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--script", default=DEFAULT_SCRIPT)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument(
        "--case-name",
        action="append",
        help="Exact default-casebook story name to run. May be repeated.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--start-mode", default="balanced_memory_start")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grounding-model", default="gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()
    summary = run_casebook(args)
    print(
        json.dumps(
            {
                "summary": summary["artifact_paths"]["summary"],
                "case_count": summary["case_count"],
                "status_counts": summary["status_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
