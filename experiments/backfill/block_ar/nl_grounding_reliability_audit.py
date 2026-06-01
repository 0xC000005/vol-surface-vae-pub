#!/usr/bin/env python
"""Audit condition-only grounding reliability for narrative scenario generation.

This audit measures the LLM grounding sidecar as an interpretation layer. It
does not score generated scenario quality; that remains the CRPS/energy
backtest surface. Here we check whether extracted current/recent market claims
are faithful to the narrative, whether future-looking language stays warning
only, whether grounded directions agree with known historical-prefix labels
when supplied, and whether selected support regimes pass direction checks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_market_alignment import (  # noqa: E402
    expected_delta_sign,
)
from experiments.backfill.block_ar.nl_prefix_latent_temporal_grounding_testflight import (  # noqa: E402
    FORWARD_LANGUAGE_MARKERS,
    ConditionOnlyGroundingResult,
    ground_condition_only_story_with_openai,
    validate_condition_only_grounding_result,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_grounding_reliability_audit_965a_testflight"
)

DEFAULT_LIVE_STORY_DECK = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/"
    "fixed_start22_calibrated_story_deck_conditionality_summary.json"
)

DEFAULT_HISTORICAL_PIPELINE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_openai_full_906b_all_windows/narrative_pipeline_report.json"
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _norm_market(raw: Any) -> str:
    text = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "CREDIT": "BBB_OAS",
        "CREDIT_SPREADS": "BBB_OAS",
        "CRUDE": "CRUDE_OIL",
        "OIL": "CRUDE_OIL",
        "SPREADS": "BBB_OAS",
        "TREASURY_YIELDS": "US10Y",
        "US_10Y": "US10Y",
        "VOL": "VIX",
        "VOLATILITY": "VIX",
    }
    return aliases.get(text, text)


def _content_tokens(text: Any) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[A-Za-z0-9_:-]+", str(text).lower()):
        token = raw.strip("_:-")
        if len(token) >= 4:
            tokens.add(token)
    return tokens


def _text_has_forward_language(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in FORWARD_LANGUAGE_MARKERS)


def _all_implications(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    direct = _as_list(grounding.get("market_implications"))
    if direct:
        return [_as_dict(row) for row in direct if isinstance(row, dict)]
    rows = []
    rows.extend(_as_list(grounding.get("current_market_state_implications")))
    rows.extend(_as_list(grounding.get("recent_regime_implications")))
    nested = _as_dict(grounding.get("condition_only_grounding"))
    rows.extend(_as_list(nested.get("current_market_state_implications")))
    rows.extend(_as_list(nested.get("recent_regime_implications")))
    return [_as_dict(row) for row in rows if isinstance(row, dict)]


def _forward_warnings(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _as_list(grounding.get("non_conditioning_forward_language"))
    nested = _as_dict(grounding.get("condition_only_grounding"))
    rows += _as_list(nested.get("non_conditioning_forward_language"))
    return [_as_dict(row) for row in rows if isinstance(row, dict)]


def _evidence_supported(item: dict[str, Any], story: str) -> bool:
    evidence_rows = _as_list(item.get("evidence"))
    if not evidence_rows:
        return False
    story_lower = str(story or "").lower()
    story_tokens = _content_tokens(story)
    for evidence in evidence_rows:
        evidence_text = str(evidence or "").strip()
        if not evidence_text:
            continue
        if evidence_text.lower() in story_lower:
            return True
        evidence_tokens = _content_tokens(evidence_text)
        if evidence_tokens and evidence_tokens.issubset(story_tokens):
            return True
    return False


def _claim_faithfulness(
    *,
    story: str,
    implications: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for item in implications:
        supported = _evidence_supported(item, story)
        rows.append(
            {
                "market": _norm_market(item.get("market")),
                "direction": str(item.get("direction", "")),
                "confidence": str(item.get("confidence", "")),
                "evidence": item.get("evidence", []),
                "supported_by_story": supported,
            }
        )
    unsupported = [row for row in rows if not bool(row["supported_by_story"])]
    return {
        "claim_count": len(rows),
        "supported_claim_count": len(rows) - len(unsupported),
        "unsupported_claim_count": len(unsupported),
        "faithfulness_rate": (
            (len(rows) - len(unsupported)) / len(rows) if rows else None
        ),
        "unsupported_claims": unsupported,
        "claims": rows,
    }


def _future_language(
    *,
    story: str,
    grounding: dict[str, Any],
    validation: dict[str, Any],
    expected_forward_language: bool | None,
) -> dict[str, Any]:
    warnings = _forward_warnings(grounding)
    expected = (
        bool(expected_forward_language)
        if expected_forward_language is not None
        else _text_has_forward_language(story)
    )
    leakage_count = int(validation.get("forward_warning_leakage_count", 0) or 0)
    return {
        "expected_forward_language": expected,
        "warning_count": len(warnings),
        "detected": bool(warnings) if expected else None,
        "leakage_count": leakage_count,
        "warnings": warnings,
    }


def _direction_agreement(
    *,
    implications: list[dict[str, Any]],
    expected_implications: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_by_market = {
        _norm_market(item.get("market")): _as_dict(item)
        for item in expected_implications
        if isinstance(item, dict) and _norm_market(item.get("market"))
    }
    checked = []
    for item in implications:
        market = _norm_market(item.get("market"))
        expected = expected_by_market.get(market)
        if not expected:
            continue
        observed_sign = expected_delta_sign(item.get("direction"))
        expected_sign = expected_delta_sign(expected.get("direction"))
        if observed_sign is None or expected_sign is None:
            continue
        checked.append(
            {
                "market": market,
                "direction": str(item.get("direction", "")),
                "expected_direction": str(expected.get("direction", "")),
                "confidence": str(item.get("confidence", "")),
                "expected_sign": expected_sign,
                "observed_sign": observed_sign,
                "aligned": observed_sign == expected_sign,
            }
        )
    matches = [row for row in checked if bool(row["aligned"])]
    high_conf = [
        row for row in checked if str(row.get("confidence", "")).lower() == "high"
    ]
    high_matches = [row for row in high_conf if bool(row["aligned"])]
    expected_markets = {
        market
        for market, item in expected_by_market.items()
        if expected_delta_sign(item.get("direction")) is not None
    }
    return {
        "expected_count": len(expected_markets),
        "checked_count": len(checked),
        "match_count": len(matches),
        "mismatch_count": len(checked) - len(matches),
        "agreement_rate": len(matches) / len(checked) if checked else None,
        "coverage_rate": len(checked) / len(expected_markets)
        if expected_markets
        else None,
        "high_confidence_checked_count": len(high_conf),
        "high_confidence_match_count": len(high_matches),
        "high_confidence_agreement_rate": (
            len(high_matches) / len(high_conf) if high_conf else None
        ),
        "checked": checked,
    }


def _support_direction(direction_check: dict[str, Any]) -> dict[str, Any]:
    if not direction_check:
        return {
            "available": False,
            "status": "missing",
            "support_weighted_match_rate": None,
            "final_mixture_checked_count": 0,
            "final_mixture_mismatch_count": 0,
        }
    status = str(direction_check.get("status", "")).lower() or "unknown"
    return {
        "available": True,
        "status": status,
        "passed": status == "pass",
        "support_weighted_match_rate": _float_or_none(
            direction_check.get("support_weighted_match_rate")
        ),
        "support_weighted_mismatch_rate": _float_or_none(
            direction_check.get("support_weighted_mismatch_rate")
        ),
        "final_mixture_checked_count": int(
            direction_check.get("final_mixture_checked_count", 0) or 0
        ),
        "final_mixture_mismatch_count": int(
            direction_check.get("final_mixture_mismatch_count", 0) or 0
        ),
        "reason": str(direction_check.get("reason", "")),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _usage_total_tokens(metadata: dict[str, Any]) -> int:
    usage = _as_dict(metadata.get("usage"))
    for key in ("total_tokens", "total_token_count"):
        if usage.get(key) is not None:
            return int(usage.get(key) or 0)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return input_tokens + output_tokens


def build_grounding_reliability_audit(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for raw_case in cases:
        case = _as_dict(raw_case)
        story = str(case.get("story", ""))
        grounding = _as_dict(case.get("grounding"))
        validation = _as_dict(case.get("validation"))
        implications = _all_implications(grounding)
        faithfulness = _claim_faithfulness(story=story, implications=implications)
        future = _future_language(
            story=story,
            grounding=grounding,
            validation=validation,
            expected_forward_language=case.get("expected_forward_language"),
        )
        historical = _direction_agreement(
            implications=implications,
            expected_implications=[
                _as_dict(row)
                for row in _as_list(case.get("expected_implications"))
                if isinstance(row, dict)
            ],
        )
        support = _support_direction(_as_dict(case.get("support_direction_check")))
        grounding_metadata = _as_dict(case.get("grounding_metadata"))
        rows.append(
            {
                "case_name": str(case.get("case_name", "")),
                "source_path": str(case.get("source_path", "")),
                "claim_faithfulness": faithfulness,
                "future_language": future,
                "historical_direction": historical,
                "support_direction": support,
                "grounding_metadata": grounding_metadata,
            }
        )

    claim_count = sum(row["claim_faithfulness"]["claim_count"] for row in rows)
    supported_count = sum(
        row["claim_faithfulness"]["supported_claim_count"] for row in rows
    )
    expected_forward = [
        row
        for row in rows
        if bool(row["future_language"]["expected_forward_language"])
    ]
    future_detected = [
        row for row in expected_forward if bool(row["future_language"]["detected"])
    ]
    leakage_cases = [
        row for row in rows if int(row["future_language"]["leakage_count"]) > 0
    ]
    hist_checked = sum(row["historical_direction"]["checked_count"] for row in rows)
    hist_match = sum(row["historical_direction"]["match_count"] for row in rows)
    hist_expected = sum(row["historical_direction"]["expected_count"] for row in rows)
    support_available = [
        row for row in rows if bool(row["support_direction"]["available"])
    ]
    support_pass = [
        row for row in support_available if bool(row["support_direction"].get("passed"))
    ]
    summary = {
        "case_count": len(rows),
        "api_grounding_case_count": sum(
            1 for row in rows if row.get("grounding_metadata")
        ),
        "api_grounding_total_tokens": sum(
            _usage_total_tokens(_as_dict(row.get("grounding_metadata")))
            for row in rows
        ),
        "claim_count": claim_count,
        "supported_claim_count": supported_count,
        "unsupported_claim_count": claim_count - supported_count,
        "claim_faithfulness_rate": supported_count / claim_count
        if claim_count
        else None,
        "expected_forward_case_count": len(expected_forward),
        "future_language_detection_rate": (
            len(future_detected) / len(expected_forward) if expected_forward else None
        ),
        "future_language_leakage_case_count": len(leakage_cases),
        "future_language_leakage_rate": len(leakage_cases) / len(rows)
        if rows
        else None,
        "historical_direction_expected_count": hist_expected,
        "historical_direction_checked_count": hist_checked,
        "historical_direction_match_count": hist_match,
        "historical_direction_agreement_rate": hist_match / hist_checked
        if hist_checked
        else None,
        "historical_direction_coverage_rate": hist_checked / hist_expected
        if hist_expected
        else None,
        "support_direction_case_count": len(support_available),
        "support_direction_pass_count": len(support_pass),
        "support_direction_pass_rate": len(support_pass) / len(support_available)
        if support_available
        else None,
        "mean_support_weighted_match_rate": _mean_present(
            [
                row["support_direction"].get("support_weighted_match_rate")
                for row in support_available
            ]
        ),
    }
    status = "pass"
    if summary["future_language_leakage_case_count"] or (
        summary["claim_faithfulness_rate"] is not None
        and summary["claim_faithfulness_rate"] < 0.90
    ):
        status = "warning"
    return {
        "status": status,
        "scope_note": (
            "Grounding reliability audit for the narrative-conditioned scenario "
            "generator. This validates the visible story-to-claims sidecar, not "
            "the generated scenario distribution."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "summary": summary,
        "cases": rows,
    }


def case_from_prefix_report(path: str | Path) -> dict[str, Any]:
    report = _load_json(path)
    cached = _as_dict(report.get("cached_query"))
    grounding = _as_dict(cached.get("grounding"))
    condition_case = _as_dict(report.get("condition_only_case"))
    story = str(condition_case.get("story") or grounding.get("cleaned_conditioning_text") or "")
    validation = _as_dict(
        condition_case.get("condition_only_validation")
        or grounding.get("condition_only_validation")
        or _as_dict(grounding.get("condition_only_grounding")).get(
            "condition_only_validation"
        )
    )
    memory_prior = _as_dict(cached.get("memory_prior"))
    return {
        "case_name": str(report.get("case_name") or Path(path).parent.name),
        "source_path": str(path),
        "story": story,
        "grounding": grounding,
        "validation": validation,
        "expected_forward_language": bool(
            _as_list(
                _as_dict(condition_case.get("story_split")).get(
                    "non_conditioning_forward_sentences"
                )
            )
            or _forward_warnings(grounding)
        ),
        "support_direction_check": _as_dict(memory_prior.get("direction_check")),
    }


def build_grounding_reliability_audit_from_prefix_reports(
    paths: list[str | Path],
) -> dict[str, Any]:
    return build_grounding_reliability_audit([case_from_prefix_report(path) for path in paths])


def cases_from_live_story_deck(path: str | Path) -> list[dict[str, Any]]:
    deck = _load_json(path)
    cases = []
    for item in _as_list(deck.get("cases")):
        if not isinstance(item, dict):
            continue
        report_path = item.get("prefix_report_snapshot_path") or item.get(
            "report_snapshot"
        )
        if report_path and Path(report_path).exists():
            case = case_from_prefix_report(report_path)
            case["case_name"] = str(item.get("case_name") or case["case_name"])
            cases.append(case)
    return cases


def cases_from_condition_summary(path: str | Path) -> list[dict[str, Any]]:
    summary = _load_json(path)
    cases = []
    for item in _as_list(summary.get("cases")):
        if not isinstance(item, dict):
            continue
        grounding = _as_dict(item.get("condition_only_grounding"))
        validation = _as_dict(item.get("condition_only_validation"))
        cases.append(
            {
                "case_name": str(item.get("case_name", "")),
                "source_path": str(path),
                "story": str(item.get("story", "")),
                "grounding": grounding,
                "validation": validation,
                "expected_forward_language": bool(
                    _as_list(
                        _as_dict(item.get("story_split")).get(
                            "non_conditioning_forward_sentences"
                        )
                    )
                    or _forward_warnings(grounding)
                ),
            }
        )
    return cases


def _primary_story_from_bundle(bundle: dict[str, Any]) -> str:
    for narrative in _as_list(bundle.get("narratives")):
        if isinstance(narrative, dict) and narrative.get("text"):
            return str(narrative.get("text"))
    descriptions = _as_list(_as_dict(bundle.get("source_description_bundle")).get("descriptions"))
    for item in descriptions:
        if isinstance(item, dict) and item.get("text"):
            return str(item.get("text"))
    return ""


def cases_from_historical_pipeline_report(
    path: str | Path,
    *,
    case_count: int,
    model: str,
    dotenv: str | Path,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    report = _load_json(path)
    cases = []
    for bundle in _as_list(report.get("narrative_bundles"))[: int(case_count)]:
        if not isinstance(bundle, dict):
            continue
        story = _primary_story_from_bundle(bundle)
        if not story:
            continue
        grounding, metadata = ground_condition_only_story_with_openai(
            story,
            model=model,
            dotenv_path=dotenv,
            max_output_tokens=max_output_tokens,
        )
        validation = validate_condition_only_grounding_result(grounding)
        cases.append(
            {
                "case_name": str(bundle.get("window_id", "")),
                "source_path": str(path),
                "story": story,
                "grounding": grounding.model_dump(),
                "grounding_metadata": metadata,
                "validation": validation,
                "expected_implications": _as_list(bundle.get("market_implications"))
                or _as_list(bundle.get("observed_market_facts")),
                "expected_forward_language": _text_has_forward_language(story),
            }
        )
    return cases


def _write_markdown(path: str | Path, audit: dict[str, Any]) -> None:
    summary = _as_dict(audit.get("summary"))
    lines = [
        "# Grounding Reliability Audit",
        "",
        "This audit scores the story-to-claims grounding sidecar, not generated scenario quality.",
        "",
        "## Summary",
        "",
        f"- Cases: `{summary.get('case_count')}`",
        f"- API-grounded cases: `{summary.get('api_grounding_case_count')}`",
        f"- API grounding tokens: `{summary.get('api_grounding_total_tokens')}`",
        f"- Claims: `{summary.get('claim_count')}`",
        f"- Claim faithfulness: `{_fmt_rate(summary.get('claim_faithfulness_rate'))}`",
        f"- Future-language detection: `{_fmt_rate(summary.get('future_language_detection_rate'))}`",
        f"- Future-language leakage: `{_fmt_rate(summary.get('future_language_leakage_rate'))}`",
        f"- Historical direction agreement: `{_fmt_rate(summary.get('historical_direction_agreement_rate'))}`",
        f"- Historical direction coverage: `{_fmt_rate(summary.get('historical_direction_coverage_rate'))}`",
        f"- Support-direction pass rate: `{_fmt_rate(summary.get('support_direction_pass_rate'))}`",
        f"- Mean support weighted match: `{_fmt_rate(summary.get('mean_support_weighted_match_rate'))}`",
        "",
        "## Cases",
        "",
    ]
    for case in _as_list(audit.get("cases")):
        if not isinstance(case, dict):
            continue
        faith = _as_dict(case.get("claim_faithfulness"))
        future = _as_dict(case.get("future_language"))
        hist = _as_dict(case.get("historical_direction"))
        support = _as_dict(case.get("support_direction"))
        lines.extend(
            [
                f"### {case.get('case_name', '')}",
                "",
                f"- Claim faithfulness: `{_fmt_rate(faith.get('faithfulness_rate'))}` "
                f"({faith.get('supported_claim_count')}/{faith.get('claim_count')})",
                f"- Forward warning expected/detected: `{future.get('expected_forward_language')}` / `{future.get('detected')}`",
                f"- Forward leakage count: `{future.get('leakage_count')}`",
                f"- Historical direction agreement: `{_fmt_rate(hist.get('agreement_rate'))}` "
                f"({hist.get('match_count')}/{hist.get('checked_count')})",
                f"- Support direction: `{support.get('status')}`",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_latex_table(path: str | Path, audit: dict[str, Any]) -> None:
    summary = _as_dict(audit.get("summary"))
    rows = [
        ("Cases", str(summary.get("case_count"))),
        ("Extracted claims", str(summary.get("claim_count"))),
        ("Claim faithfulness", _fmt_rate(summary.get("claim_faithfulness_rate"))),
        (
            "Future-language detection",
            _fmt_rate(summary.get("future_language_detection_rate")),
        ),
        (
            "Future-language leakage",
            _fmt_rate(summary.get("future_language_leakage_rate")),
        ),
        (
            "Historical direction agreement",
            _fmt_rate(summary.get("historical_direction_agreement_rate")),
        ),
        (
            "Historical direction coverage",
            _fmt_rate(summary.get("historical_direction_coverage_rate")),
        ),
        (
            "Support-direction pass rate",
            _fmt_rate(summary.get("support_direction_pass_rate")),
        ),
    ]
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
    ]
    for metric, value in rows:
        safe_metric = str(metric).replace("&", "\\&")
        safe_value = str(value).replace("%", "\\%")
        lines.append(f"{safe_metric} & {safe_value} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live-story-deck", action="append", default=[])
    parser.add_argument("--prefix-report", action="append", default=[])
    parser.add_argument("--condition-summary", action="append", default=[])
    parser.add_argument("--historical-pipeline-report")
    parser.add_argument("--historical-case-count", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    for path in args.live_story_deck:
        cases.extend(cases_from_live_story_deck(path))
    for path in args.prefix_report:
        cases.append(case_from_prefix_report(path))
    for path in args.condition_summary:
        cases.extend(cases_from_condition_summary(path))
    if args.historical_pipeline_report and int(args.historical_case_count) > 0:
        cases.extend(
            cases_from_historical_pipeline_report(
                args.historical_pipeline_report,
                case_count=int(args.historical_case_count),
                model=str(args.model),
                dotenv=args.dotenv,
                max_output_tokens=int(args.max_output_tokens),
            )
        )
    if not cases:
        raise SystemExit("no audit cases supplied")

    audit = build_grounding_reliability_audit(cases)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "grounding_reliability_audit.json"
    markdown_path = output_dir / "grounding_reliability_audit.md"
    latex_path = output_dir / "grounding_reliability_table.tex"
    _write_json(json_path, audit)
    _write_markdown(markdown_path, audit)
    _write_latex_table(latex_path, audit)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "latex": str(latex_path),
                "markdown": str(markdown_path),
                "case_count": audit["summary"]["case_count"],
                "status": audit["status"],
                "summary": audit["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
