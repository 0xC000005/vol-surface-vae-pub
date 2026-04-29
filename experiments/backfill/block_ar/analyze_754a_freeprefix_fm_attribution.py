#!/usr/bin/env python
"""754a: attribute generated-prefix FM regressions before the next model change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCORECARDS = {
    "734a_val_incumbent": "results/block_ar/734a_realvix_framework_lock_baseline/iv_val_full11_s64.json",
    "746a_train_tail_incumbent": "results/block_ar/746a_674a_train_tail_audit/iv_train_tail_full11_s64.json",
    "752a_val_freeprefix_w020": "results/block_ar/752a_iv_freeprefix_fm/iv_val_full11_s64.json",
    "752a_train_tail_freeprefix_w020": "results/block_ar/752a_iv_freeprefix_fm/iv_train_tail_full11_s64.json",
    "753a_val_freeprefix_w005": "results/block_ar/753a_iv_freeprefix_fm_w005/iv_val_full11_s64.json",
    "753a_train_tail_freeprefix_w005": "results/block_ar/753a_iv_freeprefix_fm_w005/iv_train_tail_full11_s64.json",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fnum(value: Any, digits: int = 6) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def full_horizon_active_rate(d: dict[str, Any]) -> float | None:
    full = d["mean_reversion"].get("full_horizon", {})
    if "active_mean_pass_rate" in full:
        return fnum(full["active_mean_pass_rate"])
    per = full.get("per_horizon", {})
    rates: list[float] = []
    for item in per.values():
        if isinstance(item, dict) and "active_pass_rate" in item:
            rates.append(float(item["active_pass_rate"]))
    if not rates:
        return None
    return fnum(sum(rates) / len(rates))


def scorecard_row(name: str, path: str) -> dict[str, Any]:
    d = load_json(path)
    coverage = d["coverage"]
    conditionality = d["conditionality"]
    distributional = d["distributional_fidelity"]
    regime = d["regime_coverage"]
    risk = d["risk_state_allocation"]
    ts = d["time_series"]
    coint = d["cointegration"]
    mr = d["mean_reversion"]
    pathwise = d["pathwise_jump_realism"]
    return {
        "name": name,
        "path": path,
        "n_pass": int(d["summary"]["n_pass"]),
        "failed_suites": list(d["summary"]["failed_suites"]),
        "cov90": fnum(coverage["overall"]["0.9"]),
        "calibration_error": fnum(coverage["calibration_error"]),
        "coverage_pass": bool(coverage["overall_pass"]),
        "conditionality_pass": bool(conditionality["overall_pass"]),
        "mae_reduction_pct": fnum(conditionality.get("mae_reduction_pct")),
        "worst_cell_width_ratio": fnum(conditionality.get("worst_cell_width_ratio")),
        "risk_state_pass": bool(risk["overall_pass"]),
        "history_width_spearman": fnum(risk.get("history_width_spearman")),
        "future_width_spearman": fnum(risk.get("future_width_spearman")),
        "time_series_pass": bool(ts["overall_pass"]),
        "kurtosis_ratio": fnum(ts["kurtosis"]["kurtosis_ratio"]),
        "cointegration_pass": bool(coint["overall_pass"]),
        "cointegration_ratio": fnum(coint["gen_gt_ratio"]),
        "cointegration_worst_cell_ratio": fnum(coint["worst_cell_ratio"]),
        "regime_pass": bool(regime["overall_pass"]),
        "regime_layer2_pass_count": int(regime["layer2_n_passing"]),
        "regime_layer3_catastrophic_rate": fnum(regime["layer3_catastrophic_rate"]),
        "distribution_pass": bool(distributional["overall_pass"]),
        "daily_ks_pass_cells": int(distributional["ks_test"]["n_pass"]),
        "level_ks_pass_cells": int(distributional["ks_level_test"]["n_pass"]),
        "median_bias_pass_cells": int(distributional["median_bias"]["n_pass"]),
        "mean_reversion_pass": bool(mr["overall_pass"]),
        "mean_reversion_ratio": fnum(mr["mr_gt_ratio"]),
        "mean_reversion_full_horizon_active_rate": full_horizon_active_rate(d),
        "pathwise_pass": bool(pathwise["overall_pass"]),
        "pathwise_max_jump_ks": fnum(pathwise["pathwise_max_jump"]["ks_stat"]),
    }


def delta(candidate: dict[str, Any], baseline: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        c = candidate.get(key)
        b = baseline.get(key)
        out[key] = None if c is None or b is None else fnum(float(c) - float(b))
    return out


def build_report(scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in scorecards}
    val_base = by_name["734a_val_incumbent"]
    train_base = by_name["746a_train_tail_incumbent"]
    val_752 = by_name["752a_val_freeprefix_w020"]
    train_752 = by_name["752a_train_tail_freeprefix_w020"]
    val_753 = by_name["753a_val_freeprefix_w005"]
    train_753 = by_name["753a_train_tail_freeprefix_w005"]
    keys = [
        "cov90",
        "calibration_error",
        "level_ks_pass_cells",
        "median_bias_pass_cells",
        "regime_layer2_pass_count",
        "regime_layer3_catastrophic_rate",
        "cointegration_ratio",
        "cointegration_worst_cell_ratio",
        "history_width_spearman",
        "future_width_spearman",
        "mean_reversion_ratio",
        "mean_reversion_full_horizon_active_rate",
        "pathwise_max_jump_ks",
        "kurtosis_ratio",
    ]
    return {
        "scorecards": scorecards,
        "deltas_vs_incumbent": {
            "752a_val_minus_734a_val": delta(val_752, val_base, keys),
            "753a_val_minus_734a_val": delta(val_753, val_base, keys),
            "752a_train_tail_minus_746a_train_tail": delta(train_752, train_base, keys),
            "753a_train_tail_minus_746a_train_tail": delta(train_753, train_base, keys),
        },
        "attribution": {
            "primary_read": "static_generated_prefix_fm_rejected_as_repair",
            "evidence": [
                "Both generated-prefix weights score 5/11 on validation and 5/11 on train-tail, so the issue is not only validation distribution shift.",
                "The intended regime/path axes move partly in the right direction, but coverage, level-KS, median allocation, and full-horizon active mean reversion regress versus the incumbent.",
                "Lowering the weight to 0.05 recovers aggregate mean-reversion ratio but not the active-cell full-horizon gate; it also worsens validation cov90 and level-KS versus 0.2 and versus 734a.",
            ],
            "mechanism_candidates": [
                {
                    "name": "auxiliary_loss_conflict",
                    "status": "plausible",
                    "reason": "The generated-prefix FM target pulls the velocity field toward true future innovations under off-manifold generated prefixes, while level/channel energy separately scores integrated paths. The two losses may give incompatible gradients once the generated prefix is already biased.",
                },
                {
                    "name": "too_much_generated_prefix_too_early",
                    "status": "plausible",
                    "reason": "A static auxiliary term uses generated-prefix states throughout fine-tuning, rather than gradually increasing generated-prefix exposure. This can train on low-quality prefixes before the model has adapted.",
                },
                {
                    "name": "pure_weight_tuning",
                    "status": "rejected",
                    "reason": "Weights 0.2 and 0.05 both fail with similar gate pattern. More scalar search would be a research knob, not a first-principles fix.",
                },
            ],
            "decision": (
                "Do not continue generated-prefix FM scalar tuning. If exposure-bias work continues, make the next test structural and curriculum-based: "
                "generated-prefix exposure should be scheduled from teacher-forced to free-running prefixes, or limited to short prefixes, while keeping the same normalized-innovation AR flow core."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 754a Generated-Prefix FM Attribution",
        "",
        "## Scorecards",
        "",
        "| run | pass | cov90 | calerr | cond MAE% | risk | level KS | bias | coint | worst coint | regime L2 | MR | MR ratio | path KS | kurt |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in report["scorecards"]:
        lines.append(
            "| {name} | {n_pass}/11 | {cov90:.3f} | {calibration_error:.3f} | "
            "{mae_reduction_pct:.2f} | {risk_state_pass} | {level_ks_pass_cells}/25 | "
            "{median_bias_pass_cells}/25 | {cointegration_ratio:.3f} | "
            "{cointegration_worst_cell_ratio:.3f} | {regime_layer2_pass_count}/8 | "
            "{mean_reversion_pass} | {mean_reversion_ratio:.3f} | "
            "{pathwise_max_jump_ks:.3f} | {kurtosis_ratio:.3f} |".format(**row)
        )
    lines.extend(["", "## Deltas Versus Baselines", ""])
    for name, values in report["deltas_vs_incumbent"].items():
        lines.append(f"### {name}")
        for key, value in values.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    attr = report["attribution"]
    lines.extend(
        [
            "## Attribution",
            "",
            f"- primary read: `{attr['primary_read']}`",
        ]
    )
    for item in attr["evidence"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Mechanism Candidates")
    lines.append("")
    for item in attr["mechanism_candidates"]:
        lines.append(f"- `{item['name']}`: {item['status']}. {item['reason']}")
    lines.extend(["", "## Decision", "", attr["decision"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="results/block_ar/754a_freeprefix_fm_attribution")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecards = [scorecard_row(name, path) for name, path in DEFAULT_SCORECARDS.items()]
    report = build_report(scorecards)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_dir / "summary.md")
    print(json.dumps(report["attribution"], indent=2))


if __name__ == "__main__":
    main()
