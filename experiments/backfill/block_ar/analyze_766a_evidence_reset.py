#!/usr/bin/env python
"""Summarize 766a evidence-reset reruns.

This report compares the protected 734a/739a real-VIX tri-scope incumbent with
the 755a short-prefix IV frontier after applying the same short-prefix recipe to
anchor-only and native joint scopes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/block_ar/766a_evidence_reset")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: Any, digits: int = 4) -> Any:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(out):
        return None
    return round(out, digits)


def _coverage90(result: dict[str, Any]) -> float:
    overall = result["coverage"]["overall"]
    return float(overall.get("0.9", overall.get(0.9)))


def iv_row(name: str, path: Path) -> dict[str, Any]:
    result = _load(path)
    dist = result["distributional_fidelity"]
    return {
        "name": name,
        "path": str(path),
        "score": f"{result['summary']['n_pass']}/{result['summary']['n_total']}",
        "failed_suites": list(result["summary"]["failed_suites"]),
        "cov90": _round(_coverage90(result), 4),
        "calibration_error": _round(result["coverage"]["calibration_error"], 4),
        "level_ks_pass": f"{dist['ks_level_test']['n_pass']}/25",
        "median_bias_pass": f"{dist['median_bias']['n_pass']}/25",
        "risk_state_allocation": bool(result["risk_state_allocation"]["overall_pass"]),
        "mean_reversion": bool(result["mean_reversion"]["overall_pass"]),
        "old_cointegration": bool(result["cointegration"]["overall_pass"]),
        "iv_ewma_economic_link": bool(
            result.get("iv_ewma_economic_link", {}).get("overall_pass", False)
        ),
        "pathwise_ks": _round(
            result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"], 4
        ),
        "regime_layer2": (
            f"{result['regime_coverage']['layer2_n_passing']}/"
            f"{result['regime_coverage']['layer2_n_total']}"
        ),
    }


def panel_row(name: str, path: Path) -> dict[str, Any]:
    payload = _load(path)
    result = payload["summary"] if "summary" in payload else payload
    iv_factor_corr = result["iv_factor_corr"]["matrix_corr"]
    return {
        "name": name,
        "path": str(path),
        "finite_rate": _round(result["finite_rate"], 4),
        "factor_delta_ks_mean": _round(result["factor_delta_ks_mean"], 4),
        "factor_delta_ks_pass": (
            f"{result['factor_delta_ks_pass_020']}/{result['n_factors']}"
        ),
        "factor_tail_q99_pass": (
            f"{result['factor_tail_q99_pass_05_20']}/{result['n_factors']}"
        ),
        "factor_factor_corr": _round(result["factor_factor_corr"]["upper_corr"], 4),
        "iv_factor_corr": _round(iv_factor_corr, 4),
        "conditional_mae_reduction_pct": _round(
            result["conditional_panel"]["median_mae_reduction_vs_rolled_pct"], 3
        ),
        "history_width_spearman": _round(
            result["conditional_panel"]["history_activity_width_spearman"], 4
        ),
        "failing_factors_ks_020": [
            item["name"]
            for item in result["per_factor"]
            if float(item["ks_delta"]) >= 0.20
        ],
    }


def scorecard_row(name: str, path: Path) -> dict[str, Any]:
    result = _load(path)
    return {
        "name": name,
        "path": str(path),
        "overall_pass": bool(result["overall_pass"]),
        "gate_passes": result["gate_passes"],
        "iv_effective_failed": result["iv"]["effective_failed_suites"],
        "anchor_failed": [
            key
            for key, value in result["anchor"]["checks"].items()
            if not value["pass"]
        ],
        "joint_failed": [
            key for key, value in result["joint"]["checks"].items() if not value["pass"]
        ],
        "framework_pass": bool(result["framework"]["pass"]),
    }


def build_summary(root: Path) -> dict[str, Any]:
    iv = [
        iv_row("734a_iv_only_incumbent_rerun", root / "734a_iv_val_full11_s64.json"),
        iv_row(
            "734a_native_joint_iv_slice_rerun",
            root / "734a_joint_iv_val_full11_s64.json",
        ),
        iv_row(
            "755a_iv_only_shortprefix_frontier_rerun",
            root / "755a_iv_val_full11_s64.json",
        ),
        iv_row(
            "755a_iv_only_shortprefix_train_tail_rerun",
            root / "755a_iv_train_tail_full11_s64.json",
        ),
        iv_row(
            "766a_native_joint_shortprefix_iv_slice",
            root / "766a_joint_iv_val_full11_s64.json",
        ),
    ]
    panels = [
        panel_row(
            "734a_anchor_only_incumbent_rerun", root / "734a_anchor_val_panel_s64.json"
        ),
        panel_row(
            "734a_native_joint_panel_rerun", root / "734a_joint_val_panel_s64.json"
        ),
        panel_row(
            "766a_anchor_only_shortprefix_transfer",
            root / "766a_anchor_val_panel_s64.json",
        ),
        panel_row(
            "766a_native_joint_shortprefix_transfer",
            root / "766a_joint_val_panel_s64.json",
        ),
    ]
    scorecards = [
        scorecard_row("734a_updated_scorecard", root / "734a_updated_scorecard.json"),
        scorecard_row(
            "766a_shortprefix_updated_scorecard",
            root / "766a_shortprefix_updated_scorecard.json",
        ),
    ]
    return {
        "iteration": "766a",
        "claim": "evidence_reset_tri_scope_rerun",
        "primary_decision": (
            "Do not promote 766a over the protected 734a/739a deployable incumbent. "
            "Use the short-prefix recipe as the active research ingredient only after "
            "tri-scope non-regression, because it improves IV-only validation but does "
            "not fix sticky anchor channels and regresses native joint IV level support."
        ),
        "iv_results": iv,
        "panel_results": panels,
        "scorecards": scorecards,
        "next_research_base": {
            "deployable_incumbent": "734a/739a real-VIX tri-scope framework",
            "active_research_candidate": (
                "766a short-prefix tri-scope transfer, with 755a remaining the "
                "IV-only frontier reference"
            ),
            "do_not_use": (
                "the accidental 2048-window anchor transfer artifact; it was not an "
                "all-train recipe match and is excluded from scorecards"
            ),
        },
        "remaining_bottlenecks": [
            "IV validation coverage/regime layer-2 and level/median allocation",
            "native joint IV level support versus IV-only 755a",
            "sticky low-activity AAA/BBB OAS no-update mass in anchor and joint panels",
            "single-seed summary sensitivity around borderline old cointegration and mean-reversion gates",
        ],
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# 766a Evidence Reset",
        "",
        f"- primary decision: {summary['primary_decision']}",
        f"- deployable incumbent: `{summary['next_research_base']['deployable_incumbent']}`",
        f"- active research candidate: `{summary['next_research_base']['active_research_candidate']}`",
        "",
        "## IV Full-Suite Reruns",
        "",
        "| run | score | failed suites | cov90 | calerr | level KS | median | risk-state | mean reversion | econ-link | path KS | regime L2 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in summary["iv_results"]:
        lines.append(
            "| {name} | {score} | {failed} | {cov90:.4f} | {calerr:.4f} | "
            "{level} | {median} | {risk} | {mr} | {econ} | {pathks:.4f} | {regime} |".format(
                name=row["name"],
                score=row["score"],
                failed=", ".join(row["failed_suites"]) or "none",
                cov90=row["cov90"],
                calerr=row["calibration_error"],
                level=row["level_ks_pass"],
                median=row["median_bias_pass"],
                risk=row["risk_state_allocation"],
                mr=row["mean_reversion"],
                econ=row["iv_ewma_economic_link"],
                pathks=row["pathwise_ks"],
                regime=row["regime_layer2"],
            )
        )

    lines.extend(
        [
            "",
            "## Anchor And Joint Panel Audits",
            "",
            "| run | finite | delta KS mean | KS pass | tail pass | factor corr | IV-factor corr | cond reduction | width rho | failing KS factors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["panel_results"]:
        iv_factor = (
            "n/a" if row["iv_factor_corr"] is None else f"{row['iv_factor_corr']:.4f}"
        )
        lines.append(
            "| {name} | {finite:.4f} | {ksmean:.4f} | {kspass} | {tailpass} | "
            "{fcorr:.4f} | {ivfcorr} | {cond:.3f}% | {rho:.4f} | {failing} |".format(
                name=row["name"],
                finite=row["finite_rate"],
                ksmean=row["factor_delta_ks_mean"],
                kspass=row["factor_delta_ks_pass"],
                tailpass=row["factor_tail_q99_pass"],
                fcorr=row["factor_factor_corr"],
                ivfcorr=iv_factor,
                cond=row["conditional_mae_reduction_pct"],
                rho=row["history_width_spearman"],
                failing=", ".join(row["failing_factors_ks_020"]) or "none",
            )
        )

    lines.extend(["", "## Scorecards", ""])
    for row in summary["scorecards"]:
        lines.extend(
            [
                f"- `{row['name']}` overall pass: `{row['overall_pass']}`; gates: `{row['gate_passes']}`",
                f"- `{row['name']}` IV effective failures: `{row['iv_effective_failed']}`",
                f"- `{row['name']}` anchor failures: `{row['anchor_failed']}`; joint failures: `{row['joint_failed']}`",
            ]
        )

    lines.extend(["", "## Remaining Bottlenecks", ""])
    for item in summary["remaining_bottlenecks"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output_json", type=Path, default=DEFAULT_ROOT / "summary.json"
    )
    parser.add_argument("--output_md", type=Path, default=DEFAULT_ROOT / "summary.md")
    args = parser.parse_args()

    summary = build_summary(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps({"wrote": [str(args.output_json), str(args.output_md)]}, indent=2))


if __name__ == "__main__":
    main()
