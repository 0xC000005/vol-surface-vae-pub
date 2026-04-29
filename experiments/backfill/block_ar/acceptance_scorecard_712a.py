#!/usr/bin/env python
"""712a: general conditional scenario-generator acceptance scorecard.

This is an acceptance layer over existing artifacts. It does not train,
generate, or recalibrate scenarios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SCOPES = ("iv_only", "anchor_only", "joint")
LOCKED_FRAMEWORK_FIELDS = (
    "generated_coordinate",
    "normalization_family",
    "temporal_factorization",
    "backend",
    "stochastic_source",
    "shared_core",
    "scalar_loss_terms",
    "loss_weights",
    "sampler",
    "training_protocol",
)
ALLOWED_SCOPE_DIFFERENCES = {
    "decoder_head",
    "group_balancing",
    "input_dim",
    "input_head",
    "output_dim",
    "support_transform",
}
IV_MONITORING_ONLY_SUITES = {"cointegration"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _ratio(value: float, base: float) -> float:
    return value / base if base > 1e-12 else 1.0


def _check(pass_flag: bool, value: Any, target: Any) -> dict[str, Any]:
    return {"pass": bool(pass_flag), "value": value, "target": target}


def score_iv_suite(result: dict[str, Any]) -> dict[str, Any]:
    """Score IV output for general acceptance.

    The old path-prediction conditionality gate is replaceable by the
    population risk-state allocation diagnostic. The old IV-EWMA cointegration
    suite is retained as a monitor only until a formal new dependence or
    cointegration gate is defined.
    """
    summary = result.get("summary", {})
    failed = list(summary.get("failed_suites", []))
    n_pass = int(summary.get("n_pass", 0))
    n_total = int(summary.get("n_total", 11))
    risk_state_pass = _as_bool(_get(result, "risk_state_allocation", "overall_pass", default=False))
    conditionality_pass = _as_bool(_get(result, "conditionality", "overall_pass", default=False))

    effective_failed = []
    for suite in failed:
        if suite == "conditionality" and (risk_state_pass or conditionality_pass):
            continue
        if suite in IV_MONITORING_ONLY_SUITES:
            continue
        effective_failed.append(str(suite))
    pass_flag = not effective_failed and n_total >= 11
    return {
        "pass": bool(pass_flag),
        "original_n_pass": n_pass,
        "n_total": n_total,
        "original_failed_suites": failed,
        "effective_failed_suites": effective_failed,
        "monitoring_only_suites": [
            suite for suite in failed if suite in IV_MONITORING_ONLY_SUITES
        ],
        "risk_state_allocation_pass": bool(risk_state_pass),
        "conditionality_pass": bool(conditionality_pass),
        "conditionality_policy": "risk_state_allocation_may_replace_old_conditionality_gate",
        "cointegration_policy": "old_iv_ewma_cointegration_monitoring_only_until_new_gate_defined",
    }


def score_anchor_panel(panel_payload: dict[str, Any]) -> dict[str, Any]:
    """Score anchor-only panel audit summary."""
    summary = _summary(panel_payload)
    n_factors = int(summary.get("n_factors", 0))
    finite_rate = _finite_float(summary.get("finite_rate"))
    ks_mean = _finite_float(summary.get("factor_delta_ks_mean"), default=1.0)
    ks_pass_count = int(summary.get("factor_delta_ks_pass_020", 0))
    tail_median = _finite_float(summary.get("factor_tail_q99_ratio_median"), default=0.0)
    tail_pass_count = int(summary.get("factor_tail_q99_pass_05_20", 0))
    factor_corr = summary.get("factor_factor_corr", {})
    factor_corr_value = _finite_float(factor_corr.get("upper_corr"), default=0.0)
    factor_abs_ratio = _ratio(
        _finite_float(factor_corr.get("gen_mean_abs")),
        _finite_float(factor_corr.get("gt_mean_abs")),
    )
    conditional = summary.get("conditional_panel", {})
    median_reduction = _finite_float(
        conditional.get("median_mae_reduction_vs_rolled_pct"),
        default=-100.0,
    )
    history_width_rho = _finite_float(
        conditional.get("history_activity_width_spearman"),
        default=-1.0,
    )

    checks = {
        "finite_rate": _check(finite_rate >= 0.999, finite_rate, ">=0.999"),
        "factor_delta_ks": _check(
            n_factors > 0 and ks_mean <= 0.20 and ks_pass_count == n_factors,
            {"mean": ks_mean, "pass_count": ks_pass_count, "n_factors": n_factors},
            "mean<=0.20 and all factors pass",
        ),
        "factor_tail_scale": _check(
            n_factors > 0
            and 0.5 <= tail_median <= 2.0
            and tail_pass_count == n_factors,
            {"median": tail_median, "pass_count": tail_pass_count, "n_factors": n_factors},
            "median in [0.5,2.0] and all factors pass",
        ),
        "factor_factor_corr": _check(
            factor_corr_value >= 0.50 and factor_abs_ratio >= 0.35,
            {"upper_corr": factor_corr_value, "abs_corr_ratio": factor_abs_ratio},
            "upper_corr>=0.50 and generated/GT mean abs corr>=0.35",
        ),
        "conditional_panel": _check(
            median_reduction >= 0.0 and history_width_rho >= 0.10,
            {
                "median_mae_reduction_vs_rolled_pct": median_reduction,
                "history_activity_width_spearman": history_width_rho,
            },
            "median_mae_reduction>=0 and history_activity_width_spearman>=0.10",
        ),
    }
    return {
        "pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
    }


def score_joint_panel(panel_payload: dict[str, Any]) -> dict[str, Any]:
    """Score native IV-plus-anchor joint-panel audit summary."""
    summary = _summary(panel_payload)
    anchor_score = score_anchor_panel(summary)
    iv_factor = summary.get("iv_factor_corr", {})
    matrix_corr = _finite_float(iv_factor.get("matrix_corr"), default=0.0)
    iv_factor_abs_ratio = _ratio(
        _finite_float(iv_factor.get("gen_mean_abs")),
        _finite_float(iv_factor.get("gt_mean_abs")),
    )
    checks = dict(anchor_score["checks"])
    checks["iv_factor_corr"] = _check(
        matrix_corr >= 0.50 and iv_factor_abs_ratio >= 0.35,
        {"matrix_corr": matrix_corr, "abs_corr_ratio": iv_factor_abs_ratio},
        "matrix_corr>=0.50 and generated/GT mean abs corr>=0.35",
    )
    return {
        "pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
    }


def _values_equal(values: list[Any]) -> bool:
    return all(value == values[0] for value in values[1:])


def score_framework_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Score whether a result is one frozen framework across IV/anchor/joint scopes."""
    scope_values = manifest.get("scope_values", {})
    missing_scopes = [scope for scope in REQUIRED_SCOPES if scope not in scope_values]
    fields = sorted(
        {
            key
            for values in scope_values.values()
            if isinstance(values, dict)
            for key in values
        }
    )
    differing_fields = []
    missing_locked_fields = []
    for field in fields:
        values = [scope_values.get(scope, {}).get(field) for scope in REQUIRED_SCOPES]
        if any(value is None for value in values) and field in LOCKED_FRAMEWORK_FIELDS:
            missing_locked_fields.append(field)
        if not _values_equal(values):
            differing_fields.append(field)

    disallowed_differences = [
        field for field in differing_fields if field not in ALLOWED_SCOPE_DIFFERENCES
    ]
    missing_required_locked = [
        field
        for field in LOCKED_FRAMEWORK_FIELDS
        if any(scope_values.get(scope, {}).get(field) is None for scope in REQUIRED_SCOPES)
    ]
    glued = _as_bool(manifest.get("post_hoc_glued_decks", True))
    task_specific_loss = _as_bool(manifest.get("task_specific_loss_recipes", True))
    pass_flag = (
        not missing_scopes
        and not missing_required_locked
        and not missing_locked_fields
        and not disallowed_differences
        and not glued
        and not task_specific_loss
    )
    return {
        "pass": bool(pass_flag),
        "framework_id": manifest.get("framework_id", "unknown"),
        "required_scopes": list(REQUIRED_SCOPES),
        "missing_scopes": missing_scopes,
        "allowed_scope_differences": [
            field for field in differing_fields if field in ALLOWED_SCOPE_DIFFERENCES
        ],
        "disallowed_scope_differences": disallowed_differences,
        "missing_locked_fields": missing_required_locked,
        "post_hoc_glued_decks": glued,
        "task_specific_loss_recipes": task_specific_loss,
    }


def evaluate_general_scorecard(
    *,
    iv_result: dict[str, Any],
    anchor_panel: dict[str, Any],
    joint_panel: dict[str, Any],
    framework_manifest: dict[str, Any],
) -> dict[str, Any]:
    iv = score_iv_suite(iv_result)
    anchor = score_anchor_panel(anchor_panel)
    joint = score_joint_panel(joint_panel)
    framework = score_framework_gate(framework_manifest)
    gate_passes = {
        "iv": bool(iv["pass"]),
        "anchor": bool(anchor["pass"]),
        "joint": bool(joint["pass"]),
        "framework": bool(framework["pass"]),
    }
    return {
        "scorecard_id": "712a_general_conditional_scenario_acceptance",
        "overall_pass": all(gate_passes.values()),
        "gate_passes": gate_passes,
        "iv": iv,
        "anchor": anchor,
        "joint": joint,
        "framework": framework,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 712a General Conditional Scenario Acceptance Scorecard",
        "",
        f"- overall pass: `{report['overall_pass']}`",
        "",
        "| Gate | Pass | Notes |",
        "| --- | --- | --- |",
    ]
    for gate in ("iv", "anchor", "joint", "framework"):
        row = report[gate]
        if gate == "iv":
            notes = f"effective_failed={row['effective_failed_suites']}"
        elif gate == "framework":
            notes = (
                f"disallowed_diffs={row['disallowed_scope_differences']}, "
                f"missing_scopes={row['missing_scopes']}"
            )
        else:
            failed = [name for name, check in row["checks"].items() if not check["pass"]]
            notes = f"failed_checks={failed}"
        lines.append(f"| `{gate}` | `{row['pass']}` | {notes} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iv_json", required=True)
    parser.add_argument("--anchor_panel_json", required=True)
    parser.add_argument("--joint_panel_json", required=True)
    parser.add_argument("--framework_manifest_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md")
    args = parser.parse_args()

    report = evaluate_general_scorecard(
        iv_result=load_json(args.iv_json),
        anchor_panel=load_json(args.anchor_panel_json),
        joint_panel=load_json(args.joint_panel_json),
        framework_manifest=load_json(args.framework_manifest_json),
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    print(json.dumps({"overall_pass": report["overall_pass"], "gate_passes": report["gate_passes"]}, indent=2))


if __name__ == "__main__":
    main()
