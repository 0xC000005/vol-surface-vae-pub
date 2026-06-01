#!/usr/bin/env python
"""Explain narrative-support direction gate failures from saved live artifacts.

This is an artifact-only diagnostic: it reads saved live casebook summaries and
prefix reports, then identifies which grounded implication directions are or are
not supported by the selected historical prefix mixture. It does not call
OpenAI, rerun the generator, or change the product gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SUMMARY = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "qualified_narrative_multistart_audit_956a/start_0_22_77_samples8/"
    "multi_start_live_story_deck_summary.json"
)
DEFAULT_OUTPUT = DEFAULT_SUMMARY.parent / "support_direction_failure_diagnostic.json"
DEFAULT_MARKDOWN = DEFAULT_SUMMARY.parent / "support_direction_failure_diagnostic.md"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _norm_market(raw: Any) -> str:
    text = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "CRUDE": "CRUDE_OIL",
        "OIL": "CRUDE_OIL",
        "CREDIT": "BBB_OAS",
        "CREDIT_SPREADS": "BBB_OAS",
        "SPREADS": "BBB_OAS",
        "DOLLAR": "DXY",
        "USD": "DXY",
        "VOL": "VIX",
        "VOLATILITY": "VIX",
        "RATES": "US10Y",
        "US_10Y": "US10Y",
    }
    return aliases.get(text, text)


def _implication_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    grounding = _as_dict(_as_dict(report.get("cached_query")).get("grounding"))
    rows = _as_list(grounding.get("market_implications"))
    if rows:
        return [_clean_implication(row) for row in rows if isinstance(row, dict)]
    nested = _as_dict(grounding.get("condition_only_grounding"))
    nested_rows = _as_list(nested.get("current_market_state_implications"))
    nested_rows += _as_list(nested.get("recent_regime_implications"))
    return [_clean_implication(row) for row in nested_rows if isinstance(row, dict)]


def _clean_implication(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market": _norm_market(row.get("market")),
        "direction": str(row.get("direction", "")),
        "confidence": str(row.get("confidence", "")),
        "inferred": bool(row.get("inferred", False)),
        "target_use": str(row.get("target_use", "")),
        "horizon": str(row.get("horizon", "")),
    }


def _direction_check(report: dict[str, Any]) -> dict[str, Any]:
    cached = _as_dict(report.get("cached_query"))
    prior = _as_dict(cached.get("memory_prior"))
    check = _as_dict(prior.get("direction_check"))
    if check:
        return check
    for row in _as_list(report.get("variant_rows")):
        variant = _as_dict(row)
        if bool(variant.get("is_operational", False)):
            return {
                "status": str(variant.get("memory_prior_direction_status", "")),
                "reason": str(variant.get("memory_prior_direction_reason", "")),
            }
    return {}


def _alignment_checked_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    alignment = _as_dict(candidate.get("recent_prefix_alignment"))
    return [_as_dict(row) for row in _as_list(alignment.get("checked"))]


def _candidate_match_rate(candidate: dict[str, Any]) -> float | None:
    checked = int(candidate.get("recent_prefix_checked", 0) or 0)
    matches = int(candidate.get("recent_prefix_match_count", 0) or 0)
    if checked <= 0:
        return None
    return float(matches / checked)


def _candidate_summary(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    checked = int(candidate.get("recent_prefix_checked", 0) or 0)
    mismatches = int(candidate.get("recent_prefix_mismatches", 0) or 0)
    return {
        "rank": int(candidate.get("rank", rank)),
        "window_index": int(candidate.get("window_index", -1)),
        "window_id": str(candidate.get("window_id") or candidate.get("window_index", "")),
        "weight": float(candidate.get("weight", 0.0) or 0.0),
        "memory_support_cosine": _optional_float(candidate.get("memory_support_cosine")),
        "start_distance_z": _optional_float(candidate.get("start_distance_z")),
        "direction_checked_count": checked,
        "direction_match_count": int(candidate.get("recent_prefix_match_count", 0) or 0),
        "direction_mismatch_count": mismatches,
        "direction_match_rate": _candidate_match_rate(candidate),
        "direction_mismatch_markets": [
            _norm_market(row.get("market"))
            for row in _alignment_checked_rows(candidate)
            if not bool(row.get("aligned", False))
        ],
    }


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _per_market_support(
    *,
    candidates: list[dict[str, Any]],
    final_alignment: dict[str, Any],
    implications: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    implication_by_market = {
        str(row["market"]): row for row in implications if str(row.get("market", ""))
    }
    market_rows: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        weight = float(candidate.get("weight", 0.0) or 0.0)
        for row in _alignment_checked_rows(candidate):
            market = _norm_market(row.get("market"))
            stats = market_rows.setdefault(
                market,
                {
                    "support_checked_weight": 0.0,
                    "support_aligned_weight": 0.0,
                    "support_mismatched_weight": 0.0,
                    "candidate_count": 0,
                },
            )
            stats["support_checked_weight"] += weight
            stats["candidate_count"] += 1
            if bool(row.get("aligned", False)):
                stats["support_aligned_weight"] += weight
            else:
                stats["support_mismatched_weight"] += weight
    for row in _as_list(final_alignment.get("checked")):
        checked = _as_dict(row)
        market = _norm_market(checked.get("market"))
        stats = market_rows.setdefault(
            market,
            {
                "support_checked_weight": 0.0,
                "support_aligned_weight": 0.0,
                "support_mismatched_weight": 0.0,
                "candidate_count": 0,
            },
        )
        stats["final_aligned"] = bool(checked.get("aligned", False))
        stats["final_terminal_delta"] = _optional_float(
            checked.get("mean_terminal_delta")
        )
        stats["final_direction"] = str(checked.get("direction", ""))
    for market, stats in market_rows.items():
        checked_weight = float(stats.get("support_checked_weight", 0.0) or 0.0)
        stats["support_match_rate"] = (
            float(stats.get("support_aligned_weight", 0.0) or 0.0) / checked_weight
            if checked_weight > 0.0
            else None
        )
        implication = implication_by_market.get(market)
        if implication is not None:
            stats["implication"] = implication
    return dict(sorted(market_rows.items()))


def _failure_reasons(
    *,
    direction_check: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    status = str(direction_check.get("status", "")).strip().lower()
    if status not in {"reject", "rejected", "fail", "failed"}:
        return reasons
    reason = str(direction_check.get("reason", "")).strip()
    if reason:
        reasons.append(reason)
    match_rate = _optional_float(direction_check.get("support_weighted_match_rate"))
    min_rate = _optional_float(direction_check.get("min_support_match_rate"))
    if match_rate is not None and min_rate is not None and match_rate < min_rate:
        reasons.append("support_weighted_match_rate_below_min")
    mismatched_weight = sum(
        float(candidate.get("weight", 0.0) or 0.0)
        for candidate in candidates
        if int(candidate.get("recent_prefix_mismatches", 0) or 0) > 0
    )
    if mismatched_weight > 0.5:
        reasons.append("high_weight_on_direction_mismatched_support")
    if not reasons:
        reasons.append("direction_support_rejected")
    return list(dict.fromkeys(reasons))


def summarize_case_report(
    case_name: str,
    report: dict[str, Any],
    *,
    smoke_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one live prefix report's direction-support evidence."""

    smoke = smoke_case or {}
    cached = _as_dict(report.get("cached_query"))
    prior = _as_dict(cached.get("memory_prior"))
    direction_check = _direction_check(report)
    final_alignment = _as_dict(direction_check.get("final_mixture_alignment"))
    candidates = [_as_dict(row) for row in _as_list(prior.get("candidate_details"))]
    candidate_summaries = [
        _candidate_summary(candidate, rank=rank)
        for rank, candidate in enumerate(candidates, start=1)
    ]
    status = str(direction_check.get("status", "")).strip().lower()
    support_gate = smoke.get("narrative_calibration_support_gate")
    if support_gate is None:
        support_gate = 0.0 if status in {"reject", "rejected", "fail", "failed"} else 1.0
    implications = _implication_rows(report)
    per_market = _per_market_support(
        candidates=candidates,
        final_alignment=final_alignment,
        implications=implications,
    )
    final_mismatch_markets = list(
        dict.fromkeys(
            _norm_market(row.get("market"))
        for row in _as_list(final_alignment.get("checked"))
        if not bool(_as_dict(row).get("aligned", False))
        )
    )
    return {
        "case_name": str(case_name),
        "status": str(smoke.get("status", "")),
        "start_index": smoke.get("start_index"),
        "support_gate": float(support_gate or 0.0),
        "prior_mode": str(prior.get("mode", "")),
        "direction_status": status,
        "direction_reason": str(direction_check.get("reason", "")),
        "support_weighted_match_rate": _optional_float(
            direction_check.get("support_weighted_match_rate")
        ),
        "support_weighted_mismatch_rate": _optional_float(
            direction_check.get("support_weighted_mismatch_rate")
        ),
        "min_support_match_rate": _optional_float(
            direction_check.get("min_support_match_rate")
        ),
        "final_mixture_checked_count": int(
            direction_check.get("final_mixture_checked_count", 0) or 0
        ),
        "final_mixture_mismatch_count": int(
            direction_check.get("final_mixture_mismatch_count", 0) or 0
        ),
        "final_mismatch_markets": final_mismatch_markets,
        "failure_reasons": _failure_reasons(
            direction_check=direction_check,
            candidates=candidates,
        ),
        "active_implications": implications,
        "candidate_count": int(len(candidate_summaries)),
        "candidates": candidate_summaries,
        "per_market_support": per_market,
    }


def _casebook_cases(summary: dict[str, Any], summary_path: Path) -> list[dict[str, Any]]:
    if isinstance(summary.get("cases"), list):
        return [_as_dict(case) for case in summary.get("cases", [])]
    cases: list[dict[str, Any]] = []
    for row in _as_list(summary.get("starts")):
        casebook_path = Path(str(_as_dict(row).get("casebook_summary_path", "")))
        if not casebook_path.exists():
            candidate = summary_path.parent / casebook_path
            casebook_path = candidate if candidate.exists() else casebook_path
        if not casebook_path.exists():
            continue
        casebook = _load_json(casebook_path)
        cases.extend(_casebook_cases(casebook, casebook_path))
    return cases


def build_support_direction_failure_diagnostic(
    summary_path: str | Path,
) -> dict[str, Any]:
    """Build a per-case and aggregate direction-support failure diagnostic."""

    summary_file = Path(summary_path)
    summary = _load_json(summary_file)
    cases: list[dict[str, Any]] = []
    missing_reports: list[str] = []
    for smoke_case in _casebook_cases(summary, summary_file):
        report_path = Path(str(smoke_case.get("prefix_report_snapshot_path", "")))
        if not report_path.exists():
            missing_reports.append(str(report_path))
            continue
        report = _load_json(report_path)
        cases.append(
            summarize_case_report(
                str(smoke_case.get("case_name", report_path.parent.name)),
                report,
                smoke_case=smoke_case,
            )
        )
    direction_status_counts: dict[str, int] = {}
    market_failure_counts: dict[str, int] = {}
    for case in cases:
        status = str(case.get("direction_status", ""))
        direction_status_counts[status] = direction_status_counts.get(status, 0) + 1
        if status in {"reject", "rejected", "fail", "failed"}:
            for market in case.get("final_mismatch_markets", []):
                market_failure_counts[str(market)] = (
                    market_failure_counts.get(str(market), 0) + 1
                )
    rejected = sum(
        1
        for case in cases
        if str(case.get("direction_status", "")).lower()
        in {"reject", "rejected", "fail", "failed"}
    )
    return {
        "status": "warning" if rejected else "ok",
        "summary_path": str(summary_file),
        "case_count": int(len(cases)),
        "rejected_case_count": int(rejected),
        "missing_report_count": int(len(missing_reports)),
        "missing_reports": missing_reports,
        "direction_status_counts": dict(sorted(direction_status_counts.items())),
        "market_failure_counts": dict(sorted(market_failure_counts.items())),
        "cases": cases,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Support Direction Failure Diagnostic",
        "",
        f"- Status: `{report['status']}`",
        f"- Cases analyzed: `{report['case_count']}`",
        f"- Rejected cases: `{report['rejected_case_count']}`",
        f"- Direction statuses: `{report['direction_status_counts']}`",
        f"- Final-mismatch markets: `{report['market_failure_counts']}`",
        "",
    ]
    for case in report.get("cases", []):
        lines.extend(
            [
                f"## {case['case_name']}",
                "",
                f"- Direction status: `{case['direction_status']}`",
                f"- Direction reason: `{case['direction_reason']}`",
                f"- Support gate: `{case['support_gate']}`",
                f"- Support weighted match rate: `{case['support_weighted_match_rate']}`",
                f"- Final mismatch markets: `{case['final_mismatch_markets']}`",
                f"- Failure reasons: `{case['failure_reasons']}`",
                "",
                "| Market | Support match rate | Final aligned | Final terminal delta |",
                "| --- | ---: | --- | ---: |",
            ]
        )
        for market, stats in case.get("per_market_support", {}).items():
            lines.append(
                f"| {market} | {stats.get('support_match_rate')} | "
                f"{stats.get('final_aligned', '')} | {stats.get('final_terminal_delta', '')} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_support_direction_failure_diagnostic(args.summary)
    _write_json(args.output, report)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "case_count": report["case_count"],
                "rejected_case_count": report["rejected_case_count"],
                "output": str(args.output),
                "markdown": str(args.markdown),
            },
            sort_keys=True,
        )
    )
    return 0 if report["missing_report_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
