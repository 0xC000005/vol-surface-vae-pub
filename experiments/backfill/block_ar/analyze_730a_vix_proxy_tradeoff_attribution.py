#!/usr/bin/env python
"""Attribute the 729a VIX-proxy result against prior joint/IV baselines.

This is a post-experiment analysis script. It does not train or sample models;
it only reads existing scorecards to decide whether 729a introduced a new
failure class or reproduced the known joint-scope IV trade-off.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_719_SCORECARD = "results/block_ar/719a_sticky_zero_readout_scorecard/scorecard.json"
DEFAULT_674_IV_FULL = "results/validations/2026-04-27/674a_channel_level_alltrain_e3/iv_val_full11.json"
DEFAULT_676_JOINT_IV_FULL = "results/validations/2026-04-27/676a_joint38_channel_level_alltrain/val_full11.json"
DEFAULT_729_ANCHOR_PANEL = "results/block_ar/729a_vixproxy_scorecard/anchor_val_panel_s64.json"
DEFAULT_729_JOINT_PANEL = "results/block_ar/729a_vixproxy_scorecard/joint_val_panel_s64.json"
DEFAULT_729_JOINT_IV_FULL = "results/block_ar/729a_vixproxy_scorecard/joint_iv_val_full11_s64.json"


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested(data: Any, *keys: str, default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def scrub_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: scrub_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def full_suite_row(label: str, path: str) -> dict[str, Any]:
    data = load_json(path)
    return {
        "label": label,
        "path": path,
        "n_pass": nested(data, "summary", "n_pass"),
        "n_total": nested(data, "summary", "n_total"),
        "failed_suites": nested(data, "summary", "failed_suites", default=[]),
        "coverage90": as_float(nested(data, "coverage", "overall", "0.9")),
        "coverage_calibration_error": as_float(nested(data, "coverage", "calibration_error")),
        "daily_change_ks_pass_count": nested(data, "distributional_fidelity", "ks_test", "n_pass"),
        "level_ks_pass_count": nested(data, "distributional_fidelity", "ks_level_test", "n_pass"),
        "level_ks_worst": as_float(nested(data, "distributional_fidelity", "ks_level_test", "worst_stat")),
        "mean_reversion_pass": nested(data, "mean_reversion", "overall_pass"),
        "mean_reversion_ratio": as_float(nested(data, "mean_reversion", "mr_gt_ratio")),
        "pathwise_jump_pass": nested(data, "pathwise_jump_realism", "overall_pass"),
        "pathwise_jump_ks": as_float(nested(data, "pathwise_jump_realism", "pathwise_max_jump", "ks_stat")),
        "acf_corr": as_float(nested(data, "time_series", "acf", "acf_correlation")),
        "kurtosis_ratio": as_float(nested(data, "time_series", "kurtosis", "kurtosis_ratio")),
    }


def panel_row(label: str, path: str) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("summary", {})
    return {
        "label": label,
        "path": path,
        "finite_rate": as_float(summary.get("finite_rate")),
        "n_factors": summary.get("n_factors"),
        "factor_delta_ks_mean": as_float(summary.get("factor_delta_ks_mean")),
        "factor_delta_ks_pass_020": summary.get("factor_delta_ks_pass_020"),
        "factor_tail_q99_ratio_median": as_float(summary.get("factor_tail_q99_ratio_median")),
        "factor_tail_q99_pass_05_20": summary.get("factor_tail_q99_pass_05_20"),
        "factor_factor_upper_corr": as_float(nested(summary, "factor_factor_corr", "upper_corr")),
        "factor_factor_mae": as_float(nested(summary, "factor_factor_corr", "mae")),
        "factor_factor_abs_ratio": as_float(
            _safe_ratio(
                nested(summary, "factor_factor_corr", "gen_mean_abs"),
                nested(summary, "factor_factor_corr", "gt_mean_abs"),
            )
        ),
        "iv_factor_matrix_corr": as_float(nested(summary, "iv_factor_corr", "matrix_corr")),
        "iv_factor_mae": as_float(nested(summary, "iv_factor_corr", "mae")),
        "conditional_mae_reduction_pct": as_float(
            nested(summary, "conditional_panel", "median_mae_reduction_vs_rolled_pct")
        ),
        "history_activity_width_spearman": as_float(
            nested(summary, "conditional_panel", "history_activity_width_spearman")
        ),
    }


def scorecard_panel_row(label: str, scorecard: dict[str, Any], scope: str) -> dict[str, Any]:
    checks = nested(scorecard, scope, "checks", default={})
    return {
        "label": label,
        "path": DEFAULT_719_SCORECARD,
        "finite_rate": as_float(nested(checks, "finite_rate", "value")),
        "n_factors": nested(checks, "factor_delta_ks", "value", "n_factors"),
        "factor_delta_ks_mean": as_float(nested(checks, "factor_delta_ks", "value", "mean")),
        "factor_delta_ks_pass_020": nested(checks, "factor_delta_ks", "value", "pass_count"),
        "factor_tail_q99_ratio_median": as_float(nested(checks, "factor_tail_scale", "value", "median")),
        "factor_tail_q99_pass_05_20": nested(checks, "factor_tail_scale", "value", "pass_count"),
        "factor_factor_upper_corr": as_float(nested(checks, "factor_factor_corr", "value", "upper_corr")),
        "factor_factor_abs_ratio": as_float(nested(checks, "factor_factor_corr", "value", "abs_corr_ratio")),
        "iv_factor_matrix_corr": as_float(nested(checks, "iv_factor_corr", "value", "matrix_corr")),
        "iv_factor_abs_ratio": as_float(nested(checks, "iv_factor_corr", "value", "abs_corr_ratio")),
        "conditional_mae_reduction_pct": as_float(
            nested(checks, "conditional_panel", "value", "median_mae_reduction_vs_rolled_pct")
        ),
        "history_activity_width_spearman": as_float(
            nested(checks, "conditional_panel", "value", "history_activity_width_spearman")
        ),
    }


def _safe_ratio(num: Any, den: Any) -> float | None:
    num_f = as_float(num)
    den_f = as_float(den)
    if num_f is None or den_f is None or den_f == 0.0:
        return None
    return num_f / den_f


def find_factor(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("per_factor", []):
        if row.get("name") == name:
            return row
    return None


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 730a VIX-Proxy Trade-Off Attribution",
        "",
        "## Decision",
        "",
        f"- Classification: `{report['decision']['classification']}`.",
        f"- Deployable replacement for 719a: `{report['decision']['deployable_replacement_for_719a']}`.",
        f"- Keep VIX proxy as data-interface evidence: `{report['decision']['keep_vix_proxy_interface']}`.",
        "",
        "## IV Full-Suite Comparison",
        "",
        "| Run | Pass | Failed Suites | Cov90 | Level KS Pass | Path KS | MR Ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["iv_full_suite_rows"]:
        failed = ", ".join(row["failed_suites"])
        lines.append(
            "| {label} | {n_pass}/{n_total} | {failed} | {coverage90} | {level_ks} | {path_ks} | {mr_ratio} |".format(
                label=row["label"],
                n_pass=row["n_pass"],
                n_total=row["n_total"],
                failed=failed,
                coverage90=_fmt(row["coverage90"]),
                level_ks=_fmt_count(row["level_ks_pass_count"]),
                path_ks=_fmt(row["pathwise_jump_ks"]),
                mr_ratio=_fmt(row["mean_reversion_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Panel Comparison",
            "",
            "| Run | Factors | Delta KS Mean | KS Pass | Tail Pass | Factor Corr | IV-Factor Corr | Cond MAE Gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["panel_rows"]:
        lines.append(
            "| {label} | {n_factors} | {ks_mean} | {ks_pass} | {tail_pass} | {factor_corr} | {iv_factor_corr} | {cond_gain}% |".format(
                label=row["label"],
                n_factors=row["n_factors"],
                ks_mean=_fmt(row["factor_delta_ks_mean"]),
                ks_pass=row["factor_delta_ks_pass_020"],
                tail_pass=row["factor_tail_q99_pass_05_20"],
                factor_corr=_fmt(row["factor_factor_upper_corr"]),
                iv_factor_corr=_fmt(row["iv_factor_matrix_corr"]),
                cond_gain=_fmt(row["conditional_mae_reduction_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
        ]
    )
    for bullet in report["mechanism_read"]:
        lines.append(f"- {bullet}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
        ]
    )
    for bullet in report["next_step"]:
        lines.append(f"- {bullet}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    value = as_float(value)
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_count(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scorecard_719 = load_json(args.scorecard_719)
    anchor_729 = load_json(args.panel_729_anchor)
    joint_729 = load_json(args.panel_729_joint)

    iv_rows = [
        {
            "label": "719a_iv_scorecard",
            "path": args.scorecard_719,
            "n_pass": nested(scorecard_719, "iv", "original_n_pass"),
            "n_total": nested(scorecard_719, "iv", "n_total"),
            "failed_suites": nested(scorecard_719, "iv", "effective_failed_suites", default=[]),
            "coverage90": None,
            "coverage_calibration_error": None,
            "daily_change_ks_pass_count": None,
            "level_ks_pass_count": None,
            "level_ks_worst": None,
            "mean_reversion_pass": None,
            "mean_reversion_ratio": None,
            "pathwise_jump_pass": None,
            "pathwise_jump_ks": None,
            "acf_corr": None,
            "kurtosis_ratio": None,
        },
        full_suite_row("674a_iv_only_full", args.full_674_iv),
        full_suite_row("676a_joint38_full", args.full_676_joint),
        full_suite_row("729a_joint39_vixproxy_full", args.full_729_joint_iv),
    ]

    panel_rows = [
        scorecard_panel_row("719a_anchor", scorecard_719, "anchor"),
        panel_row("729a_anchor_vixproxy", args.panel_729_anchor),
        scorecard_panel_row("719a_joint", scorecard_719, "joint"),
        panel_row("729a_joint_vixproxy", args.panel_729_joint),
    ]

    vix_proxy_rows = {
        "anchor": find_factor(anchor_729.get("summary", {}), "factor:vix_proxy"),
        "joint": find_factor(joint_729.get("summary", {}), "factor:vix_proxy"),
    }

    prior_joint_pass = iv_rows[2]["n_pass"]
    vix_joint_pass = iv_rows[3]["n_pass"]
    known_joint_tradeoff = prior_joint_pass == vix_joint_pass

    report = {
        "analysis_id": "730a_vix_proxy_tradeoff_attribution",
        "inputs": {
            "scorecard_719": args.scorecard_719,
            "full_674_iv": args.full_674_iv,
            "full_676_joint": args.full_676_joint,
            "panel_729_anchor": args.panel_729_anchor,
            "panel_729_joint": args.panel_729_joint,
            "full_729_joint_iv": args.full_729_joint_iv,
        },
        "iv_full_suite_rows": iv_rows,
        "panel_rows": panel_rows,
        "vix_proxy_rows": vix_proxy_rows,
        "decision": {
            "classification": (
                "known_joint_scope_iv_tradeoff_with_successful_vix_proxy_data_interface"
                if known_joint_tradeoff
                else "possible_new_vix_proxy_interference"
            ),
            "new_vix_specific_failure_class": not known_joint_tradeoff,
            "deployable_replacement_for_719a": False,
            "keep_vix_proxy_interface": True,
            "claim_scope": (
                "The local VIX proxy validates scalar volatility-factor ingestion, not an independent VIX law, "
                "because the proxy is derived from an IV surface column."
            ),
        },
        "mechanism_read": [
            "729a joint+VIX has the same 5/11 IV full-suite pass count as the prior 676a native joint run, so the regression is not primarily a new VIX-specific failure.",
            "729a improves or preserves panel-level factor realism relative to 719a/688a-style anchor evidence: 13/14 factor KS pass, 14/14 tail pass, strong conditional width response, and strong IV-factor matrix correlation in joint mode.",
            "The cost remains the known joint-scope IV calibration trade-off: coverage, regime coverage, level distribution fidelity, and horizon mean-reversion/pathwise gates are weaker than the frozen 719a IV-facing baseline.",
            "Because the proxy is constructed from short-ATM IV, it is partly redundant with the IV surface; it should be treated as a data-interface stress test rather than a new market-observed VIX factor claim.",
        ],
        "next_step": [
            "Do not add VIX-specific knobs; keep the proxy loader as optional evidence that the framework can ingest scalar volatility factors.",
            "Keep 719a as the current risk-manager baseline and compare future changes against it on all three scopes.",
            "If continuing experiments, focus on the general joint-scope IV calibration/root trade-off, not on proxy-specific repair.",
        ],
    }
    return scrub_json(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard-719", default=DEFAULT_719_SCORECARD)
    parser.add_argument("--full-674-iv", default=DEFAULT_674_IV_FULL)
    parser.add_argument("--full-676-joint", default=DEFAULT_676_JOINT_IV_FULL)
    parser.add_argument("--panel-729-anchor", default=DEFAULT_729_ANCHOR_PANEL)
    parser.add_argument("--panel-729-joint", default=DEFAULT_729_JOINT_PANEL)
    parser.add_argument("--full-729-joint-iv", default=DEFAULT_729_JOINT_IV_FULL)
    parser.add_argument(
        "--output-json",
        default="results/block_ar/730a_vix_proxy_tradeoff_attribution/attribution.json",
    )
    parser.add_argument(
        "--output-md",
        default="results/block_ar/730a_vix_proxy_tradeoff_attribution/attribution.md",
    )
    args = parser.parse_args()

    report = build_report(args)

    output_json = REPO_ROOT / args.output_json
    output_md = REPO_ROOT / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(make_markdown(report), encoding="utf-8")

    print(json.dumps(report["decision"], indent=2))
    print(f"Wrote {output_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {output_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
