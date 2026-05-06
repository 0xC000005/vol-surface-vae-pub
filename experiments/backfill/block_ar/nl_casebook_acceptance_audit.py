#!/usr/bin/env python
"""Audit risk-manager casebooks against production-facing acceptance rules.

This is an offline gate over `nl_risk_manager_casebook.py` output. It turns
casebook rows into explicit pass/warning/fail decisions and bottleneck tags so
we can decide what blocks production readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STATUS_ORDER = {"pass": 0, "warning": 1, "fail": 2}


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


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != raw or raw in {float("inf"), float("-inf")}:
        return None
    return raw


def _rule(
    *,
    name: str,
    category: str,
    status: str,
    code: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in STATUS_ORDER:
        raise ValueError(f"unknown rule status: {status}")
    return {
        "name": name,
        "category": category,
        "status": status,
        "code": code,
        "reason": reason,
        "evidence": evidence or {},
    }


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "fail"
    return max(statuses, key=lambda value: STATUS_ORDER[value])


def _narrative_grounding_rule(case: dict[str, Any]) -> dict[str, Any]:
    narrative = str(case.get("input_narrative", "")).strip()
    implications = _as_list(case.get("market_implications"))
    grounding = _as_dict(case.get("grounding"))
    validation_errors = _as_list(grounding.get("validation_errors"))
    validation_warnings = _as_list(grounding.get("validation_warnings"))
    external_news_used = bool(grounding.get("external_news_used", False))
    if not narrative or not implications:
        return _rule(
            name="narrative_grounding",
            category="label_quality",
            status="fail",
            code="missing_narrative_or_implications",
            reason="The case lacks either a narrative or extracted market implications.",
            evidence={
                "has_narrative": bool(narrative),
                "market_implication_count": len(implications),
            },
        )
    if validation_errors or external_news_used:
        return _rule(
            name="narrative_grounding",
            category="label_quality",
            status="fail",
            code="label_validation_error",
            reason="The label contains validation errors or used external news in an offline market-state case.",
            evidence={
                "validation_errors": validation_errors,
                "external_news_used": external_news_used,
            },
        )
    if validation_warnings:
        return _rule(
            name="narrative_grounding",
            category="label_quality",
            status="warning",
            code="label_validation_warning",
            reason="The label is usable but has grounding warnings that should be reviewed.",
            evidence={"validation_warnings": validation_warnings},
        )
    return _rule(
        name="narrative_grounding",
        category="label_quality",
        status="pass",
        code="grounded_narrative",
        reason="Narrative and market implications are present with no label validation errors.",
        evidence={"market_implication_count": len(implications)},
    )


def _bridge_alignment_rule(case: dict[str, Any]) -> dict[str, Any]:
    bridge = _as_dict(case.get("bridge"))
    target_cosine = _float(bridge.get("target_cosine"))
    hard_negative_gap = _float(bridge.get("hard_negative_gap"))
    true_rank_test_pool = bridge.get("true_rank_test_pool")
    if target_cosine is None:
        return _rule(
            name="bridge_alignment",
            category="bridge_alignment",
            status="fail",
            code="missing_bridge_target_cosine",
            reason="Bridge target cosine is missing.",
            evidence=bridge,
        )
    if target_cosine < 0.75:
        return _rule(
            name="bridge_alignment",
            category="bridge_alignment",
            status="fail",
            code="weak_bridge_alignment",
            reason="The language condition is too far from the target scenario memory.",
            evidence={
                "target_cosine": target_cosine,
                "hard_negative_gap": hard_negative_gap,
                "true_rank_test_pool": true_rank_test_pool,
            },
        )
    rank_warning = (
        isinstance(true_rank_test_pool, int) and int(true_rank_test_pool) > 15
    )
    gap_warning = hard_negative_gap is not None and hard_negative_gap < 0.5
    if target_cosine < 0.85 or rank_warning or gap_warning:
        return _rule(
            name="bridge_alignment",
            category="bridge_alignment",
            status="warning",
            code="borderline_bridge_alignment",
            reason="Bridge alignment is usable but should be reviewed for analogue quality.",
            evidence={
                "target_cosine": target_cosine,
                "hard_negative_gap": hard_negative_gap,
                "true_rank_test_pool": true_rank_test_pool,
            },
        )
    return _rule(
        name="bridge_alignment",
        category="bridge_alignment",
        status="pass",
        code="bridge_alignment_ok",
        reason="Bridge target cosine and hard-negative separation are acceptable.",
        evidence={
            "target_cosine": target_cosine,
            "hard_negative_gap": hard_negative_gap,
            "true_rank_test_pool": true_rank_test_pool,
        },
    )


def _analogue_quality_rule(case: dict[str, Any]) -> dict[str, Any]:
    analogues = [
        row
        for row in _as_list(case.get("historical_analogues"))
        if isinstance(row, dict)
    ]
    if not analogues:
        return _rule(
            name="historical_analogues",
            category="analogue_quality",
            status="fail",
            code="missing_historical_analogues",
            reason="No historical analogues are available for the case.",
        )
    top_similarity = _float(analogues[0].get("similarity"))
    missing_narratives = [
        row.get("window_id")
        for row in analogues
        if not str(row.get("primary_narrative", "")).strip()
    ]
    if top_similarity is None or top_similarity < 0.70:
        return _rule(
            name="historical_analogues",
            category="analogue_quality",
            status="fail",
            code="weak_historical_analogue_similarity",
            reason="The closest analogue is too weak to be product-facing.",
            evidence={
                "top_similarity": top_similarity,
                "analogue_count": len(analogues),
            },
        )
    if len(analogues) < 3 or top_similarity < 0.80 or missing_narratives:
        return _rule(
            name="historical_analogues",
            category="analogue_quality",
            status="warning",
            code="borderline_historical_analogues",
            reason="Analogues exist but are incomplete or only moderately similar.",
            evidence={
                "top_similarity": top_similarity,
                "analogue_count": len(analogues),
                "missing_narrative_window_ids": missing_narratives,
            },
        )
    return _rule(
        name="historical_analogues",
        category="analogue_quality",
        status="pass",
        code="historical_analogues_ok",
        reason="The case has multiple high-similarity analogues with narratives.",
        evidence={"top_similarity": top_similarity, "analogue_count": len(analogues)},
    )


def _scenario_distribution_rule(case: dict[str, Any]) -> dict[str, Any]:
    narrative = _as_dict(_as_dict(case.get("scores")).get("narrative_generator_topk"))
    energy = _float(narrative.get("energy_score_improvement_vs_persistence"))
    crps = _float(narrative.get("ensemble_crps_improvement_vs_persistence"))
    coverage = _float(narrative.get("coverage_80"))
    if energy is None or crps is None:
        return _rule(
            name="scenario_distribution",
            category="generator_distribution",
            status="fail",
            code="missing_distribution_metrics",
            reason="Narrative-generator distribution metrics are missing.",
            evidence=narrative,
        )
    if energy <= 0.0 and crps <= 0.0:
        return _rule(
            name="scenario_distribution",
            category="generator_distribution",
            status="fail",
            code="distribution_lags_persistence",
            reason="The generated scenario distribution is worse than persistence on both core distribution metrics.",
            evidence={
                "energy_improvement": energy,
                "crps_improvement": crps,
                "coverage_80": coverage,
            },
        )
    if energy <= 0.0 or crps <= 0.0 or (coverage is not None and coverage < 0.40):
        return _rule(
            name="scenario_distribution",
            category="generator_distribution",
            status="warning",
            code="borderline_distribution_metrics",
            reason="The generated distribution has mixed distributional evidence.",
            evidence={
                "energy_improvement": energy,
                "crps_improvement": crps,
                "coverage_80": coverage,
            },
        )
    return _rule(
        name="scenario_distribution",
        category="generator_distribution",
        status="pass",
        code="distribution_metrics_ok",
        reason="The generated distribution improves over persistence on energy score and CRPS.",
        evidence={
            "energy_improvement": energy,
            "crps_improvement": crps,
            "coverage_80": coverage,
        },
    )


def _point_path_framing_rule(case: dict[str, Any]) -> dict[str, Any]:
    narrative = _as_dict(_as_dict(case.get("scores")).get("narrative_generator_topk"))
    mean_path = _float(narrative.get("mean_path_mae_improvement_vs_persistence"))
    if mean_path is None:
        return _rule(
            name="point_path_framing",
            category="product_framing",
            status="warning",
            code="missing_point_path_metric",
            reason="Mean-path point metric is missing; product framing cannot be checked.",
        )
    if mean_path < 0.0:
        return _rule(
            name="point_path_framing",
            category="product_framing",
            status="warning",
            code="point_path_lags_persistence",
            reason="The case should be presented as scenario distribution, not point forecast.",
            evidence={"mean_path_mae_improvement_vs_persistence": mean_path},
        )
    return _rule(
        name="point_path_framing",
        category="product_framing",
        status="pass",
        code="point_path_not_a_blocker",
        reason="Mean-path metric does not contradict the scenario framing.",
        evidence={"mean_path_mae_improvement_vs_persistence": mean_path},
    )


def _unsupported_causality_rule(case: dict[str, Any]) -> dict[str, Any]:
    grounding = _as_dict(case.get("grounding"))
    non_observed = _as_list(grounding.get("non_observed_catalysts"))
    if len(non_observed) > 6:
        return _rule(
            name="unsupported_causality_surface",
            category="label_quality",
            status="warning",
            code="too_many_non_observed_catalysts",
            reason="The case carries many hypothetical catalysts or analogies; risk-manager review should check narrative focus.",
            evidence={"non_observed_catalyst_count": len(non_observed)},
        )
    return _rule(
        name="unsupported_causality_surface",
        category="label_quality",
        status="pass",
        code="causality_surface_controlled",
        reason="Unsupported catalysts are absent or limited and explicitly grounded as analogies.",
        evidence={"non_observed_catalyst_count": len(non_observed)},
    )


def _product_framing_by_window(
    product_report_package: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not product_report_package:
        return {}
    case_framing = product_report_package.get("case_framing")
    if isinstance(case_framing, dict):
        return {
            str(window_id): framing
            for window_id, framing in case_framing.items()
            if isinstance(framing, dict)
        }
    rows: dict[str, dict[str, Any]] = {}
    for report in _as_list(product_report_package.get("case_reports")):
        if not isinstance(report, dict) or not report.get("window_id"):
            continue
        rows[str(report["window_id"])] = {
            "framing_ready": bool(report.get("framing_ready")),
            "framing_checks": _as_dict(report.get("framing_checks")),
            "report_contract": report.get("report_contract"),
        }
    return rows


def _product_framing_satisfies_point_warning(framing: dict[str, Any] | None) -> bool:
    if not framing:
        return False
    checks = _as_dict(framing.get("framing_checks"))
    return (
        bool(framing.get("framing_ready"))
        and str(framing.get("report_contract", ""))
        == "scenario_distribution_not_point_forecast"
        and bool(checks.get("not_point_forecast_statement"))
        and bool(checks.get("distribution_metrics_present"))
        and bool(checks.get("historical_analogues_present"))
        and bool(checks.get("grounding_section_present"))
    )


def _apply_product_report_framing(
    rules: list[dict[str, Any]],
    *,
    product_framing: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not _product_framing_satisfies_point_warning(product_framing):
        return rules
    updated: list[dict[str, Any]] = []
    for rule in rules:
        if (
            rule.get("name") == "point_path_framing"
            and rule.get("code") == "point_path_lags_persistence"
            and rule.get("status") == "warning"
        ):
            updated.append(
                {
                    **rule,
                    "status": "pass",
                    "code": "distributional_product_framing_satisfied",
                    "reason": (
                        "The product report explicitly frames this as a "
                        "scenario distribution and historical-analogue tool, "
                        "not a point forecast."
                    ),
                    "evidence": {
                        **_as_dict(rule.get("evidence")),
                        "product_framing": product_framing,
                    },
                }
            )
            continue
        updated.append(rule)
    return updated


def audit_case(
    case: dict[str, Any],
    *,
    product_framing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply production-facing acceptance rules to one casebook case."""

    rules = [
        _narrative_grounding_rule(case),
        _bridge_alignment_rule(case),
        _analogue_quality_rule(case),
        _scenario_distribution_rule(case),
        _point_path_framing_rule(case),
        _unsupported_causality_rule(case),
    ]
    rules = _apply_product_report_framing(rules, product_framing=product_framing)
    status = _worst_status([rule["status"] for rule in rules])
    rule_counts = Counter(rule["status"] for rule in rules)
    bottlenecks = sorted(
        {
            rule["category"]
            for rule in rules
            if rule["status"] != "pass"
            and not (
                rule["name"] == "point_path_framing"
                and rule["code"] == "point_path_lags_persistence"
                and status == "fail"
            )
        }
    )
    if not bottlenecks and status != "pass":
        bottlenecks = ["unknown"]
    return {
        "window_id": str(case.get("window_id", "")),
        "status": status,
        "regime_tags": (
            list(case.get("regime_tags", []))
            if isinstance(case.get("regime_tags"), list)
            else []
        ),
        "rule_counts": {
            key: int(rule_counts.get(key, 0)) for key in ("fail", "pass", "warning")
        },
        "failure_codes": [rule["code"] for rule in rules if rule["status"] == "fail"],
        "warning_codes": [
            rule["code"] for rule in rules if rule["status"] == "warning"
        ],
        "bottleneck_tags": bottlenecks,
        "rules": rules,
    }


def _acceptance_gate(status_counts: dict[str, int], case_count: int) -> dict[str, Any]:
    fail_count = int(status_counts.get("fail", 0))
    warning_count = int(status_counts.get("warning", 0))
    pass_count = int(status_counts.get("pass", 0))
    fail_rate = fail_count / max(case_count, 1)
    warning_or_fail_rate = (fail_count + warning_count) / max(case_count, 1)
    if fail_count == 0 and warning_count <= max(2, int(0.25 * max(case_count, 1))):
        status = "ready_for_scaled_pilot"
        reason = "No failing cases and warning load is low enough for a larger pilot."
    elif fail_rate <= 0.10:
        status = "needs_targeted_hardening"
        reason = "Few or no failing cases, but warning/failure load still needs targeted hardening before production."
    else:
        status = "not_production_ready"
        reason = (
            "Failure rate is too high for a production-facing risk-manager workflow."
        )
    return {
        "status": status,
        "reason": reason,
        "pass_rate": pass_count / max(case_count, 1),
        "fail_rate": fail_rate,
        "warning_or_fail_rate": warning_or_fail_rate,
    }


def build_acceptance_audit(
    casebook: dict[str, Any],
    *,
    title: str = "Risk Manager Casebook Acceptance Audit",
    product_report_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an aggregate acceptance audit from a casebook."""

    product_framing = _product_framing_by_window(product_report_package)
    case_results = [
        audit_case(
            case,
            product_framing=product_framing.get(str(case.get("window_id", ""))),
        )
        for case in _as_list(casebook.get("cases"))
        if isinstance(case, dict)
    ]
    status_counts = Counter(row["status"] for row in case_results)
    bottleneck_counts = Counter(
        tag for row in case_results for tag in row.get("bottleneck_tags", [])
    )
    regime_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in case_results:
        regimes = row.get("regime_tags") or ["unclassified"]
        for regime in regimes:
            regime_status_counts[str(regime)][str(row["status"])] += 1
    case_count = len(case_results)
    status_counts_dict = {
        key: int(status_counts[key])
        for key in ("fail", "pass", "warning")
        if int(status_counts[key]) > 0
    }
    return {
        "title": title,
        "status": "casebook_acceptance_audit",
        "source_casebook_title": casebook.get("title"),
        "source_product_report_title": (
            (product_report_package or {}).get("title")
            if isinstance(product_report_package, dict)
            else None
        ),
        "summary": {
            "case_count": case_count,
            "status_counts": status_counts_dict,
            "bottleneck_counts": dict(sorted(bottleneck_counts.items())),
            "regime_status_counts": {
                regime: {
                    key: int(counter[key])
                    for key in ("fail", "pass", "warning")
                    if int(counter[key]) > 0
                }
                for regime, counter in sorted(regime_status_counts.items())
            },
            "production_gate": _acceptance_gate(status_counts_dict, case_count),
        },
        "case_results": case_results,
    }


def _fmt_pct(value: Any) -> str:
    raw = _float(value)
    return "n/a" if raw is None else f"{raw * 100.0:.1f}%"


def _case_issue_text(row: dict[str, Any]) -> str:
    codes = list(row.get("failure_codes", [])) + list(row.get("warning_codes", []))
    return ", ".join(codes[:4]) if codes else "none"


def render_acceptance_markdown(audit: dict[str, Any]) -> str:
    """Render the acceptance audit as Markdown."""

    title = str(audit.get("title") or "Risk Manager Casebook Acceptance Audit")
    summary = _as_dict(audit.get("summary"))
    gate = _as_dict(summary.get("production_gate"))
    lines = [
        f"# {title}",
        "",
        "## Production Gate",
        "",
        f"- Gate status: {gate.get('status')}",
        f"- Reason: {gate.get('reason')}",
        f"- Pass rate: {_fmt_pct(gate.get('pass_rate'))}",
        f"- Fail rate: {_fmt_pct(gate.get('fail_rate'))}",
        f"- Warning-or-fail rate: {_fmt_pct(gate.get('warning_or_fail_rate'))}",
        "",
        "## Summary",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Status counts: {json.dumps(summary.get('status_counts', {}), sort_keys=True)}",
        f"- Bottlenecks: {json.dumps(summary.get('bottleneck_counts', {}), sort_keys=True)}",
        f"- Regime status counts: {json.dumps(summary.get('regime_status_counts', {}), sort_keys=True)}",
        "",
        "## Case Results",
        "",
        "| Window | Status | Regimes | Bottlenecks | Key Issues |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in _as_list(audit.get("case_results")):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("window_id", "")),
                    str(row.get("status", "")),
                    ", ".join(row.get("regime_tags", [])) or "unclassified",
                    ", ".join(row.get("bottleneck_tags", [])) or "none",
                    _case_issue_text(row),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Rule Details", ""])
    for row in _as_list(audit.get("case_results")):
        if not isinstance(row, dict):
            continue
        lines.extend([f"### {row.get('window_id')} ({row.get('status')})", ""])
        for rule in _as_list(row.get("rules")):
            if not isinstance(rule, dict):
                continue
            lines.append(
                f"- `{rule.get('status')}` `{rule.get('code')}` "
                f"({rule.get('category')}): {rule.get('reason')}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", required=True)
    parser.add_argument("--product-report-summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Risk Manager Casebook Acceptance Audit")
    args = parser.parse_args()

    product_report_package = (
        _load_json(args.product_report_summary) if args.product_report_summary else None
    )
    audit = build_acceptance_audit(
        _load_json(args.casebook),
        title=args.title,
        product_report_package=product_report_package,
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "acceptance_audit.json"
    markdown_path = output_dir / "acceptance_audit.md"
    _write_json(json_path, audit)
    _write_text(markdown_path, render_acceptance_markdown(audit))
    print(
        json.dumps(
            {
                "audit_json": str(json_path),
                "audit_markdown": str(markdown_path),
                "case_count": audit["summary"]["case_count"],
                "production_gate": audit["summary"]["production_gate"],
                "status_counts": audit["summary"]["status_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
