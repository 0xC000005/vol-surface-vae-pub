#!/usr/bin/env python
"""Summarize the 661a generalized-510a joint AR patch-energy result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def suite_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    coverage = payload["coverage"]
    conditionality = payload["conditionality"]
    distributional = payload["distributional_fidelity"]
    pathwise = payload["pathwise_jump_realism"]
    cross_cell = payload["cross_cell_correlation"]
    return {
        "score": f"{payload['summary']['n_pass']}/{payload['summary']['n_total']}",
        "failed": ", ".join(payload["summary"]["failed_suites"]) or "none",
        "cov90": keyed(coverage["overall"], 0.9),
        "mae_reduction": conditionality["mae_reduction_pct"],
        "turb_calm": conditionality["turb_calm_ratio"],
        "daily_ks_pass": distributional["ks_test"]["n_pass"],
        "level_ks_pass": distributional["ks_level_test"]["n_pass"],
        "median_bias_pass": distributional["median_bias"]["n_pass"],
        "path_ks": pathwise["pathwise_max_jump"]["ks_stat"],
        "corr_ratio": cross_cell["corr_ratio"],
    }


def joint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", payload)
    return {
        "factor_ks_mean": summary["factor_delta_ks_mean"],
        "factor_ks_pass": summary["factor_delta_ks_pass_020"],
        "factor_q99_median": summary["factor_tail_q99_ratio_median"],
        "factor_q99_pass": summary["factor_tail_q99_pass_05_20"],
        "factor_corr_shape": summary["factor_factor_corr"]["upper_corr"],
        "factor_corr_gt_abs": summary["factor_factor_corr"]["gt_mean_abs"],
        "factor_corr_gen_abs": summary["factor_factor_corr"]["gen_mean_abs"],
        "iv_factor_corr_shape": summary["iv_factor_corr"]["matrix_corr"],
        "iv_factor_gt_abs": summary["iv_factor_corr"]["gt_mean_abs"],
        "iv_factor_gen_abs": summary["iv_factor_corr"]["gen_mean_abs"],
    }


def keyed(mapping: dict[Any, Any], key: Any) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping[str(key)]


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir",
        default="results/autoresearch/661a_generalized_510a_joint_patch_encoded_s661",
    )
    parser.add_argument(
        "--baseline_641_dir",
        default="results/autoresearch/641a_joint38_mixedcoord_scale_e8_w2048_s641",
    )
    parser.add_argument(
        "--baseline_647_dir",
        default="results/autoresearch/647a_joint38_mixed_path_flow_e8_w2048_s647",
    )
    parser.add_argument(
        "--output",
        default="experiments/backfill/block_ar/ANALYSIS_661a_generalized_510a_joint_patch_result.md",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    train_tail = suite_metrics(load_json(result_dir / "train_tail_full11.json"))
    validation = suite_metrics(load_json(result_dir / "full11.json"))
    train_joint = joint_summary(load_json(result_dir / "train_tail_joint_panel.json"))
    val_joint = joint_summary(load_json(result_dir / "joint_panel.json"))

    baseline_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name, base_dir in [
        ("641a mixed-coordinate AR", Path(args.baseline_641_dir)),
        ("647a mixed-coordinate path", Path(args.baseline_647_dir)),
    ]:
        if (base_dir / "full11.json").exists() and (base_dir / "joint_panel.json").exists():
            baseline_rows.append(
                (
                    name,
                    suite_metrics(load_json(base_dir / "full11.json")),
                    joint_summary(load_json(base_dir / "joint_panel.json")),
                )
            )

    lines = [
        "# 661a Generalized-510a Joint AR Patch-Energy Result",
        "",
        "## Question",
        "",
        "661a tested whether the empirically strong 510a-style AR transition can be generalized cleanly to a single native `joint38` model: one checkpoint, one shared causal memory, one shared stochastic source, one transition flow, and no post-hoc IV/factor deck gluing.",
        "",
        "The implementation used the generic empirical-score AR transition over the full 38-channel encoded state panel and added a small patch-energy free-rollout objective. This deliberately kept the architecture simple and reused the old 510a trunk rather than introducing separate IV and anchor-factor mechanisms.",
        "",
        "## Main Result",
        "",
        "| split | IV score | cov90 | cond MAE red. | turb/calm | daily KS | level KS | median bias | path max-jump KS | corr ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| train-tail | {train_tail['score']} | {fmt(train_tail['cov90'])} | "
            f"{fmt(train_tail['mae_reduction'])}% | {fmt(train_tail['turb_calm'])} | "
            f"{train_tail['daily_ks_pass']}/25 | {train_tail['level_ks_pass']}/25 | "
            f"{train_tail['median_bias_pass']}/25 | {fmt(train_tail['path_ks'])} | "
            f"{fmt(train_tail['corr_ratio'])} |"
        ),
        (
            f"| validation | {validation['score']} | {fmt(validation['cov90'])} | "
            f"{fmt(validation['mae_reduction'])}% | {fmt(validation['turb_calm'])} | "
            f"{validation['daily_ks_pass']}/25 | {validation['level_ks_pass']}/25 | "
            f"{validation['median_bias_pass']}/25 | {fmt(validation['path_ks'])} | "
            f"{fmt(validation['corr_ratio'])} |"
        ),
        "",
        f"Train-tail failed suites: `{train_tail['failed']}`.",
        f"Validation failed suites: `{validation['failed']}`.",
        "",
        "## Joint-Panel Read",
        "",
        "| split | factor KS mean | factor KS pass | factor q99 median | factor q99 pass | factor corr shape | factor abs corr GT/gen | IV-factor shape | IV-factor abs corr GT/gen |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| train-tail | {fmt(train_joint['factor_ks_mean'])} | "
            f"{train_joint['factor_ks_pass']}/13 | {fmt(train_joint['factor_q99_median'])} | "
            f"{train_joint['factor_q99_pass']}/13 | {fmt(train_joint['factor_corr_shape'])} | "
            f"{fmt(train_joint['factor_corr_gt_abs'])}/{fmt(train_joint['factor_corr_gen_abs'])} | "
            f"{fmt(train_joint['iv_factor_corr_shape'])} | "
            f"{fmt(train_joint['iv_factor_gt_abs'])}/{fmt(train_joint['iv_factor_gen_abs'])} |"
        ),
        (
            f"| validation | {fmt(val_joint['factor_ks_mean'])} | "
            f"{val_joint['factor_ks_pass']}/13 | {fmt(val_joint['factor_q99_median'])} | "
            f"{val_joint['factor_q99_pass']}/13 | {fmt(val_joint['factor_corr_shape'])} | "
            f"{fmt(val_joint['factor_corr_gt_abs'])}/{fmt(val_joint['factor_corr_gen_abs'])} | "
            f"{fmt(val_joint['iv_factor_corr_shape'])} | "
            f"{fmt(val_joint['iv_factor_gt_abs'])}/{fmt(val_joint['iv_factor_gen_abs'])} |"
        ),
        "",
        "661a is acceptable as a mechanism check but not as the active deployable joint model. In train-tail it preserves many IV mechanics, but the joint-panel validation audit is weaker than the native mixed-coordinate baselines: factor q99 pass falls to 6/13, factor-factor correlation shape falls to 0.584, and generated absolute correlation magnitudes are strongly attenuated.",
        "",
        "## Baseline Comparison",
        "",
        "| model | IV score | cov90 | cond MAE red. | level KS | path KS | factor KS pass | factor q99 pass | factor corr shape | IV-factor shape |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, suite, joint in baseline_rows:
        lines.append(
            f"| {name} | {suite['score']} | {fmt(suite['cov90'])} | "
            f"{fmt(suite['mae_reduction'])}% | {suite['level_ks_pass']}/25 | "
            f"{fmt(suite['path_ks'])} | {joint['factor_ks_pass']}/13 | "
            f"{joint['factor_q99_pass']}/13 | {fmt(joint['factor_corr_shape'])} | "
            f"{fmt(joint['iv_factor_corr_shape'])} |"
        )
    lines.append(
        f"| 661a generalized-510a joint AR | {validation['score']} | {fmt(validation['cov90'])} | "
        f"{fmt(validation['mae_reduction'])}% | {validation['level_ks_pass']}/25 | "
        f"{fmt(validation['path_ks'])} | {val_joint['factor_ks_pass']}/13 | "
        f"{val_joint['factor_q99_pass']}/13 | {fmt(val_joint['factor_corr_shape'])} | "
        f"{fmt(val_joint['iv_factor_corr_shape'])} |"
    )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "The 661a failure is not a shared-source failure. It is a coordinate failure. The empirical-score state decoder is fitted on the training support, so validation anchor factors that move beyond that support are clipped or pulled back toward the training range. The validation joint audit shows this directly: SPX and Nikkei generated maxima remain near the training-era range while the validation realized levels move higher.",
            "",
            "This makes the absolute-level empirical-score coordinate a poor generic state variable for random-walk-like traded factors. It can preserve in-sample IV mechanics, but it is not robust for a general multivariate market panel where levels drift across regimes.",
            "",
            "## Decision",
            "",
            "Do not promote 661a as the active deployable model. Keep it as a negative-but-useful falsifier for the idea that the 510a empirical-score state transition can be directly lifted to a 38-channel level panel.",
            "",
            "The best-supported native joint path remains the mixed-coordinate family: shared source and transition, but generated coordinates selected by data semantics. IV-like bounded mean-reverting surfaces need level-score style support control; anchor factors need movement-from-current coordinates so they can extrapolate with the observed market level.",
            "",
            "The next implementation should not add another post-hoc calibration or separate factor deck. It should either return to the mixed-coordinate state-conditioned AR family or make the 510a trunk operate on movement coordinates with explicit state conditioning, not absolute encoded levels.",
            "",
        ]
    )

    output = Path(args.output)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
