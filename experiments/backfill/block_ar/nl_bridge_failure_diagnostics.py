#!/usr/bin/env python
"""Diagnose bridge-alignment failures from saved narrative scenario artifacts.

This script is offline. It reads the product-framed acceptance audit, casebook,
bridge report, and pipeline report, then focuses only on cases where
`bridge_alignment` remains a bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WEAK_TARGET_COSINE = 0.75
BORDERLINE_TARGET_COSINE = 0.85
STRICT_RANK_THRESHOLD = 15
GOOD_NEIGHBOR_COSINE = 0.85
LOW_HARD_NEGATIVE_GAP = 0.50
GOOD_HARD_NEGATIVE_GAP = 0.75


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


def _fmt_float(value: Any, digits: int = 3) -> str:
    raw = _float(value)
    return "n/a" if raw is None else f"{raw:.{digits}f}"


def _index_by_window_id(rows: list[Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("window_id"):
            indexed[str(row["window_id"])] = row
    return indexed


def _bridge_anchor_rows_by_window(
    bridge_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _as_list(
        _as_dict(bridge_report.get("evaluation")).get("heldout_examples")
    ):
        if not isinstance(row, dict) or not row.get("window_id"):
            continue
        window_id = str(row["window_id"])
        if str(row.get("role", "")) == "anchor":
            rows[window_id] = row
        elif window_id not in rows:
            rows[window_id] = row
    return rows


def _hard_negative_rows_by_window(
    bridge_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    hard_negative = _as_dict(
        _as_dict(bridge_report.get("evaluation")).get("hard_negative_separation")
    )
    return _index_by_window_id(_as_list(hard_negative.get("windows")))


def _pipeline_bundles_by_window(
    pipeline_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return _index_by_window_id(_as_list(pipeline_report.get("narrative_bundles")))


def _casebook_cases_by_window(casebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _index_by_window_id(_as_list(casebook.get("cases")))


def _bridge_problem_rows(acceptance_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _as_list(acceptance_audit.get("case_results")):
        if not isinstance(row, dict):
            continue
        if "bridge_alignment" in _as_list(row.get("bottleneck_tags")):
            rows.append(row)
    return rows


def _market_direction(case: dict[str, Any], market: str) -> str:
    for row in _as_list(case.get("market_implications")):
        if isinstance(row, dict) and str(row.get("market", "")).upper() == market:
            return str(row.get("direction", "")).lower()
    return ""


def _has_mixed_regime(case: dict[str, Any]) -> bool:
    spx = _market_direction(case, "SPX")
    vix = _market_direction(case, "VIX")
    bbb = _market_direction(case, "BBB_OAS")
    aaa = _market_direction(case, "AAA_OAS")
    if spx == "up" and vix == "up":
        return True
    if spx == "down" and vix == "down":
        return True
    if spx == "up" and (bbb == "wider" or aaa == "wider"):
        return True
    if spx == "down" and (bbb == "tighter" or aaa == "tighter"):
        return True
    return False


def _top_similarity(row: dict[str, Any], key: str) -> float | None:
    pool = _as_list(row.get(key))
    if not pool or not isinstance(pool[0], dict):
        return None
    return _float(pool[0].get("cosine"))


def _top_analogue_similarity(case: dict[str, Any]) -> float | None:
    analogues = _as_list(case.get("historical_analogues"))
    if not analogues or not isinstance(analogues[0], dict):
        return None
    return _float(analogues[0].get("similarity"))


def _label_has_review_issue(case: dict[str, Any], audit_row: dict[str, Any]) -> bool:
    grounding = _as_dict(case.get("grounding"))
    return (
        bool(_as_list(grounding.get("validation_errors")))
        or bool(_as_list(grounding.get("validation_warnings")))
        or "label_quality" in _as_list(audit_row.get("bottleneck_tags"))
    )


def _mode(
    code: str, severity: str, explanation: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "explanation": explanation,
        "evidence": evidence,
    }


def classify_bridge_failure_modes(
    *,
    case: dict[str, Any],
    audit_row: dict[str, Any],
    bridge_row: dict[str, Any],
    hard_negative_row: dict[str, Any],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Classify likely bridge failure modes with conservative heuristics."""

    target_cosine = _float(
        bridge_row.get(
            "target_cosine", _as_dict(case.get("bridge")).get("target_cosine")
        )
    )
    true_rank_test_pool = bridge_row.get(
        "true_rank_test_pool", _as_dict(case.get("bridge")).get("true_rank_test_pool")
    )
    hard_negative_gap = _float(
        hard_negative_row.get(
            "negative_gap", _as_dict(case.get("bridge")).get("hard_negative_gap")
        )
    )
    top_train_similarity = _top_similarity(bridge_row, "top_train_pool")
    if top_train_similarity is None:
        top_train_similarity = _top_analogue_similarity(case)
    modes: list[dict[str, Any]] = []
    if target_cosine is None:
        modes.append(
            _mode(
                "missing_bridge_metrics",
                "fail",
                "Target cosine is missing, so bridge alignment cannot be audited.",
                {},
            )
        )
    elif target_cosine < WEAK_TARGET_COSINE:
        modes.append(
            _mode(
                "weak_target_alignment",
                "fail",
                "Predicted condition memory is too far from the target memory.",
                {"target_cosine": target_cosine},
            )
        )
    elif target_cosine < BORDERLINE_TARGET_COSINE:
        modes.append(
            _mode(
                "borderline_target_alignment",
                "warning",
                "Bridge cosine is usable but below the product threshold.",
                {"target_cosine": target_cosine},
            )
        )
    if hard_negative_gap is not None and hard_negative_gap < LOW_HARD_NEGATIVE_GAP:
        modes.append(
            _mode(
                "weak_hard_negative_separation",
                "warning",
                "Contrastive hard negatives remain too close to positives.",
                {"hard_negative_gap": hard_negative_gap},
            )
        )
    if _has_mixed_regime(case):
        modes.append(
            _mode(
                "mixed_regime_semantic_ambiguity",
                "warning",
                "Observed facts combine risk-on and hedging/stress signals, which can be semantically hard for text embeddings.",
                {
                    "spx": _market_direction(case, "SPX"),
                    "vix": _market_direction(case, "VIX"),
                    "bbb_oas": _market_direction(case, "BBB_OAS"),
                    "aaa_oas": _market_direction(case, "AAA_OAS"),
                },
            )
        )
    if _label_has_review_issue(case, audit_row):
        modes.append(
            _mode(
                "label_quality_review",
                "warning",
                "The same case also has label-quality warnings/errors, so label repair should precede model changes.",
                {
                    "validation_errors": _as_list(
                        _as_dict(case.get("grounding")).get("validation_errors")
                    ),
                    "validation_warnings": _as_list(
                        _as_dict(case.get("grounding")).get("validation_warnings")
                    ),
                },
            )
        )
    rank_is_strict = (
        isinstance(true_rank_test_pool, int)
        and int(true_rank_test_pool) > STRICT_RANK_THRESHOLD
    )
    if (
        rank_is_strict
        and top_train_similarity is not None
        and top_train_similarity >= GOOD_NEIGHBOR_COSINE
        and (hard_negative_gap is None or hard_negative_gap >= GOOD_HARD_NEGATIVE_GAP)
    ):
        modes.append(
            _mode(
                "dense_neighbor_rank_metric_strictness",
                "warning",
                "The bridge retrieves highly similar neighbours while the exact target rank is poor; rank may be too strict for dense regimes.",
                {
                    "true_rank_test_pool": true_rank_test_pool,
                    "top_train_similarity": top_train_similarity,
                    "hard_negative_gap": hard_negative_gap,
                },
            )
        )
    if (
        target_cosine is not None
        and target_cosine < WEAK_TARGET_COSINE
        and hard_negative_gap is not None
        and hard_negative_gap >= GOOD_HARD_NEGATIVE_GAP
        and not _label_has_review_issue(case, audit_row)
    ):
        modes.append(
            _mode(
                "bridge_model_hard_case",
                "fail",
                "Labels and hard-negative separation look acceptable, so this should enter the bridge hard-case set.",
                {
                    "target_cosine": target_cosine,
                    "hard_negative_gap": hard_negative_gap,
                    "contrastive_narrative_count": len(
                        _as_list(bundle.get("contrastive_narratives"))
                    ),
                },
            )
        )
    if not modes:
        modes.append(
            _mode(
                "bridge_review_no_clear_mode",
                "warning",
                "Bridge was flagged by the acceptance audit, but simple heuristics did not isolate a clear cause.",
                {},
            )
        )
    return modes


def _actions_for_modes(modes: list[dict[str, Any]]) -> list[str]:
    mapping = {
        "weak_target_alignment": "targeted_bridge_retraining_or_bakeoff_candidate",
        "borderline_target_alignment": "add_to_bridge_hard_case_validation_set",
        "weak_hard_negative_separation": "strengthen_hard_negative_training",
        "mixed_regime_semantic_ambiguity": "add_mixed_regime_contrastive_bridge_examples",
        "label_quality_review": "repair_or_regenerate_grounded_labels",
        "dense_neighbor_rank_metric_strictness": "review_rank_metric_against_analogue_acceptance",
        "bridge_model_hard_case": "targeted_bridge_retraining_or_bakeoff_candidate",
        "missing_bridge_metrics": "repair_bridge_evaluation_artifacts",
        "bridge_review_no_clear_mode": "manual_bridge_case_review",
    }
    actions: list[str] = []
    for mode in modes:
        action = mapping.get(str(mode.get("code", "")))
        if action and action not in actions:
            actions.append(action)
    return actions


def _diagnostic_case(
    *,
    case: dict[str, Any],
    audit_row: dict[str, Any],
    bridge_row: dict[str, Any],
    hard_negative_row: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    modes = classify_bridge_failure_modes(
        case=case,
        audit_row=audit_row,
        bridge_row=bridge_row,
        hard_negative_row=hard_negative_row,
        bundle=bundle,
    )
    return {
        "window_id": str(case.get("window_id", "")),
        "acceptance_status": audit_row.get("status"),
        "acceptance_codes": list(audit_row.get("failure_codes", []))
        + list(audit_row.get("warning_codes", [])),
        "regime_tags": (
            list(case.get("regime_tags", []))
            if isinstance(case.get("regime_tags"), list)
            else []
        ),
        "input_narrative": str(case.get("input_narrative", "")),
        "observed_fact_tokens": (
            _as_list(bundle.get("narratives"))[0].get("observed_fact_tokens", "")
            if _as_list(bundle.get("narratives"))
            and isinstance(_as_list(bundle.get("narratives"))[0], dict)
            else ""
        ),
        "market_implications": _as_list(case.get("market_implications")),
        "bridge": {
            "query_kind": bridge_row.get(
                "kind", _as_dict(case.get("bridge")).get("query_kind")
            ),
            "target_cosine": _float(
                bridge_row.get(
                    "target_cosine", _as_dict(case.get("bridge")).get("target_cosine")
                )
            ),
            "target_mse": _float(bridge_row.get("target_mse")),
            "true_rank_test_pool": bridge_row.get(
                "true_rank_test_pool",
                _as_dict(case.get("bridge")).get("true_rank_test_pool"),
            ),
            "true_rank_full_pool": bridge_row.get("true_rank_full_pool"),
            "hard_negative_gap": _float(
                hard_negative_row.get(
                    "negative_gap",
                    _as_dict(case.get("bridge")).get("hard_negative_gap"),
                )
            ),
            "hard_negative_positive_mean_cosine": _float(
                hard_negative_row.get("positive_mean_cosine")
            ),
            "hard_negative_negative_mean_cosine": _float(
                hard_negative_row.get("negative_mean_cosine")
            ),
        },
        "retrieval": {
            "top_train_pool": _as_list(bridge_row.get("top_train_pool"))[:5],
            "top_test_pool": _as_list(bridge_row.get("top_test_pool"))[:5],
            "historical_analogues": _as_list(case.get("historical_analogues"))[:3],
        },
        "label_quality": {
            "validation_errors": _as_list(
                _as_dict(case.get("grounding")).get("validation_errors")
            ),
            "validation_warnings": _as_list(
                _as_dict(case.get("grounding")).get("validation_warnings")
            ),
            "contrastive_narrative_count": len(
                _as_list(bundle.get("contrastive_narratives"))
            ),
        },
        "failure_modes": modes,
        "recommended_actions": _actions_for_modes(modes),
    }


def _recommended_next_step(action_counts: Counter[str]) -> str:
    if not action_counts:
        return "No bridge-alignment cases were found."
    top_action, top_count = action_counts.most_common(1)[0]
    if top_action == "review_rank_metric_against_analogue_acceptance":
        return (
            "First review bridge acceptance thresholds against analogue quality: "
            f"{top_count} cases retrieve close neighbours but miss exact target rank."
        )
    if top_action == "targeted_bridge_retraining_or_bakeoff_candidate":
        return (
            "Build a bridge hard-case validation set from these cases, then rerun "
            "bridge training and architecture bake-off only on that targeted set."
        )
    if top_action == "repair_or_regenerate_grounded_labels":
        return "Repair or regenerate labels for bridge cases with label-quality warnings before changing the bridge."
    if top_action == "add_mixed_regime_contrastive_bridge_examples":
        return "Add explicit mixed-regime contrastive examples, especially risk-on plus hedging-demand cases."
    return f"Prioritize `{top_action}` across {top_count} bridge-problem cases."


def build_bridge_failure_diagnostics(
    casebook: dict[str, Any],
    acceptance_audit: dict[str, Any],
    bridge_report: dict[str, Any],
    pipeline_report: dict[str, Any],
    *,
    title: str = "Bridge Failure Diagnostics",
) -> dict[str, Any]:
    """Build diagnostics for cases where bridge alignment remains a bottleneck."""

    cases_by_id = _casebook_cases_by_window(casebook)
    bridge_by_id = _bridge_anchor_rows_by_window(bridge_report)
    hard_negative_by_id = _hard_negative_rows_by_window(bridge_report)
    bundles_by_id = _pipeline_bundles_by_window(pipeline_report)
    problem_rows = _bridge_problem_rows(acceptance_audit)
    diagnostic_cases: list[dict[str, Any]] = []
    for audit_row in problem_rows:
        window_id = str(audit_row.get("window_id", ""))
        case = cases_by_id.get(window_id)
        if not case:
            continue
        diagnostic_cases.append(
            _diagnostic_case(
                case=case,
                audit_row=audit_row,
                bridge_row=bridge_by_id.get(window_id, _as_dict(case.get("bridge"))),
                hard_negative_row=hard_negative_by_id.get(window_id, {}),
                bundle=bundles_by_id.get(window_id, {}),
            )
        )
    mode_counts = Counter(
        str(mode["code"]) for case in diagnostic_cases for mode in case["failure_modes"]
    )
    action_counts = Counter(
        action for case in diagnostic_cases for action in case["recommended_actions"]
    )
    bridge_failure_count = sum(
        1 for case in diagnostic_cases if case.get("acceptance_status") == "fail"
    )
    bridge_warning_count = sum(
        1 for case in diagnostic_cases if case.get("acceptance_status") == "warning"
    )
    return {
        "title": title,
        "status": "bridge_failure_diagnostics",
        "summary": {
            "bridge_problem_case_count": len(diagnostic_cases),
            "bridge_failure_count": bridge_failure_count,
            "bridge_warning_count": bridge_warning_count,
            "failure_mode_counts": dict(sorted(mode_counts.items())),
            "recommended_action_counts": dict(sorted(action_counts.items())),
            "recommended_next_step": _recommended_next_step(action_counts),
        },
        "cases": diagnostic_cases,
    }


def _mode_line(mode: dict[str, Any]) -> str:
    return f"`{mode.get('severity')}` `{mode.get('code')}`: {mode.get('explanation')}"


def _action_line(actions: list[Any]) -> str:
    return ", ".join(str(action) for action in actions) if actions else "manual_review"


def render_bridge_diagnostics_markdown(diagnostics: dict[str, Any]) -> str:
    """Render bridge diagnostics as Markdown."""

    summary = _as_dict(diagnostics.get("summary"))
    lines = [
        f"# {diagnostics.get('title') or 'Bridge Failure Diagnostics'}",
        "",
        "## Recommended Next Step",
        "",
        str(summary.get("recommended_next_step", "")),
        "",
        "## Summary",
        "",
        f"- Bridge-problem cases: {summary.get('bridge_problem_case_count', 0)}",
        f"- Bridge failures: {summary.get('bridge_failure_count', 0)}",
        f"- Bridge warnings: {summary.get('bridge_warning_count', 0)}",
        f"- Failure modes: {json.dumps(summary.get('failure_mode_counts', {}), sort_keys=True)}",
        f"- Recommended actions: {json.dumps(summary.get('recommended_action_counts', {}), sort_keys=True)}",
        "",
        "## Cases",
        "",
        "| Window | Status | Target Cosine | Test Rank | Hard-Neg Gap | Modes | Actions |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for case in _as_list(diagnostics.get("cases")):
        if not isinstance(case, dict):
            continue
        bridge = _as_dict(case.get("bridge"))
        modes = ", ".join(
            str(mode.get("code")) for mode in _as_list(case.get("failure_modes"))
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("window_id", "")),
                    str(case.get("acceptance_status", "")),
                    _fmt_float(bridge.get("target_cosine")),
                    str(bridge.get("true_rank_test_pool", "n/a")),
                    _fmt_float(bridge.get("hard_negative_gap")),
                    modes,
                    _action_line(_as_list(case.get("recommended_actions"))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Case Details", ""])
    for case in _as_list(diagnostics.get("cases")):
        if not isinstance(case, dict):
            continue
        lines.extend(
            [
                f"### {case.get('window_id')}",
                "",
                f"- Status: {case.get('acceptance_status')}",
                f"- Regimes: {', '.join(case.get('regime_tags', [])) or 'unclassified'}",
                f"- Narrative: {case.get('input_narrative', '')}",
                f"- Observed facts: {case.get('observed_fact_tokens', '')}",
                f"- Recommended actions: {_action_line(_as_list(case.get('recommended_actions')))}",
                "",
                "Failure modes:",
            ]
        )
        for mode in _as_list(case.get("failure_modes")):
            if isinstance(mode, dict):
                lines.append(f"- {_mode_line(mode)}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casebook", required=True)
    parser.add_argument("--acceptance-audit", required=True)
    parser.add_argument("--bridge-report", required=True)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Bridge Failure Diagnostics")
    args = parser.parse_args()

    diagnostics = build_bridge_failure_diagnostics(
        _load_json(args.casebook),
        _load_json(args.acceptance_audit),
        _load_json(args.bridge_report),
        _load_json(args.pipeline_report),
        title=args.title,
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "bridge_failure_diagnostics.json"
    markdown_path = output_dir / "bridge_failure_diagnostics.md"
    _write_json(json_path, diagnostics)
    _write_text(markdown_path, render_bridge_diagnostics_markdown(diagnostics))
    print(
        json.dumps(
            {
                "diagnostics_json": str(json_path),
                "diagnostics_markdown": str(markdown_path),
                "summary": diagnostics["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
