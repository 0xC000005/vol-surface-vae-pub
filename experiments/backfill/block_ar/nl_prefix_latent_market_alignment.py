"""Market-implication alignment helpers for narrative-conditioned scenarios."""

from __future__ import annotations

from typing import Any


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def expected_delta_sign(direction: Any) -> int | None:
    text = str(direction).strip().lower()
    positive = {
        "up",
        "higher",
        "rise",
        "rising",
        "wider",
        "widening",
        "weaker",
    }
    negative = {
        "down",
        "lower",
        "fall",
        "falling",
        "tighter",
        "tightening",
        "stronger",
    }
    if text in positive:
        return 1
    if text in negative:
        return -1
    return None


def _evidence_text(item: dict[str, Any]) -> str:
    evidence = item.get("evidence", "")
    if isinstance(evidence, list):
        return " ".join(str(part) for part in evidence).lower()
    return str(evidence).lower()


def is_terminal_direction_checkable(item: dict[str, Any]) -> bool:
    """Return whether an implication should be checked as a prefix delta.

    Grounding can describe either a recent move ("VIX is rising") or a current
    level state ("VIX is elevated"). The former can be checked against a
    30-day terminal delta; the latter needs a level/support check and should not
    be treated as a directional path claim.
    """

    horizon = str(item.get("horizon", "")).lower()
    evidence = _evidence_text(item)
    if not evidence:
        return True
    motion_terms = {
        "bid",
        "compressing",
        "drifting",
        "dropping",
        "falling",
        "firmer",
        "firming",
        "going",
        "higher",
        "lower",
        "moving",
        "rallying",
        "rising",
        "selling",
        "selloff",
        "sliding",
        "softening",
        "strengthening",
        "tightening",
        "weakening",
        "widening",
    }
    static_terms = {
        "choppy",
        "elevated",
        "flat",
        "not providing",
        "remains elevated",
        "under pressure",
    }
    has_motion = any(term in evidence for term in motion_terms)
    has_static = any(term in evidence for term in static_terms)
    if horizon == "current_state" and has_static and not has_motion:
        return False
    return True


def market_implication_alignment(
    *,
    grounding: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare direct grounded implications with generated terminal directions."""

    scenario_by_market = {
        str(row.get("Market", "")).upper(): row
        for row in scenario_rows
        if isinstance(row, dict)
    }
    checked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    implications = grounding.get("market_implications", [])
    if not isinstance(implications, list):
        implications = []
    for item in implications:
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", "")).upper()
        direction = str(item.get("direction", ""))
        if not is_terminal_direction_checkable(item):
            skipped.append(
                {
                    "market": market,
                    "direction": direction,
                    "reason": "static_current_state_not_terminal_direction",
                    "evidence": item.get("evidence"),
                }
            )
            continue
        expected = expected_delta_sign(direction)
        scenario = scenario_by_market.get(market)
        if expected is None or scenario is None:
            skipped.append(
                {
                    "market": market,
                    "direction": direction,
                    "reason": "unsupported_direction_or_missing_scenario",
                }
            )
            continue
        observed = parse_float(scenario.get("Mean Terminal Delta"))
        if observed is None:
            skipped.append(
                {
                    "market": market,
                    "direction": direction,
                    "reason": "non_numeric_terminal_delta",
                }
            )
            continue
        observed_sign = 0
        if observed > tolerance:
            observed_sign = 1
        elif observed < -tolerance:
            observed_sign = -1
        checked.append(
            {
                "market": market,
                "direction": direction,
                "expected_sign": expected,
                "mean_terminal_delta": observed,
                "observed_sign": observed_sign,
                "aligned": observed_sign == expected,
                "confidence": item.get("confidence"),
                "inferred": bool(item.get("inferred", False)),
            }
        )
    mismatches = [row for row in checked if not bool(row.get("aligned"))]
    return {
        "status": "pass" if not mismatches else "warning",
        "checked_count": len(checked),
        "match_count": len(checked) - len(mismatches),
        "mismatch_count": len(mismatches),
        "skipped_count": len(skipped),
        "checked": checked,
        "mismatches": mismatches,
        "skipped": skipped,
    }
