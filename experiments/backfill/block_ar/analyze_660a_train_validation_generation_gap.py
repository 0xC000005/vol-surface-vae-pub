#!/usr/bin/env python
"""660a: compare 658a train-tail and validation generation audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def summarize_full(full: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": get_nested(full, ["summary", "n_pass"]),
        "failed": get_nested(full, ["summary", "failed_suites"], []),
        "cov90": get_nested(full, ["coverage", "overall", "0.9"]),
        "h30_cov90": get_nested(full, ["coverage", "per_horizon", "30", "0.9"]),
        "conditionality_mae_reduction_pct": get_nested(
            full, ["conditionality", "mae_reduction_pct"]
        ),
        "turb_calm_width_ratio": get_nested(full, ["conditionality", "turb_calm_ratio"]),
        "daily_ks_pass": get_nested(
            full, ["distributional_fidelity", "ks_test", "n_pass"]
        ),
        "daily_ks_median": get_nested(
            full, ["distributional_fidelity", "ks_test", "median_stat"]
        ),
        "level_ks_pass": get_nested(
            full, ["distributional_fidelity", "ks_level_test", "n_pass"]
        ),
        "level_ks_median": get_nested(
            full, ["distributional_fidelity", "ks_level_test", "median_stat"]
        ),
        "level_ks_worst": get_nested(
            full, ["distributional_fidelity", "ks_level_test", "worst_stat"]
        ),
        "median_bias_pass": get_nested(
            full, ["distributional_fidelity", "median_bias", "n_pass"]
        ),
        "window_floor_bad_rate": get_nested(
            full, ["distributional_fidelity", "window_floor", "pct_bad"]
        ),
        "kurtosis_ratio": get_nested(full, ["time_series", "kurtosis", "kurtosis_ratio"]),
        "move_size_profile_pass": get_nested(
            full, ["time_series", "move_size_profile", "pass"]
        ),
        "corr_ratio": get_nested(full, ["cross_cell_correlation", "corr_ratio"]),
        "rank_ratio": get_nested(full, ["cross_cell_correlation", "rank_ratio"]),
        "mean_reversion_ratio": get_nested(full, ["mean_reversion", "mr_gt_ratio"]),
        "mean_reversion_pass": get_nested(full, ["mean_reversion", "overall_pass"]),
        "active_mr_pass_rate": get_nested(full, ["mean_reversion", "active_pass_rate"]),
        "pathwise_max_jump_ks": get_nested(
            full, ["pathwise_jump_realism", "pathwise_max_jump", "ks_stat"]
        ),
        "regime_layer2": (
            get_nested(full, ["regime_coverage", "layer2_n_passing"]),
            get_nested(full, ["regime_coverage", "layer2_n_total"]),
        ),
    }


def summarize_joint(joint: dict[str, Any]) -> dict[str, Any]:
    s = joint["summary"]
    return {
        "factor_delta_ks_mean": s["factor_delta_ks_mean"],
        "factor_delta_ks_pass_020": s["factor_delta_ks_pass_020"],
        "factor_tail_q99_pass_05_20": s["factor_tail_q99_pass_05_20"],
        "factor_tail_q99_ratio_median": s["factor_tail_q99_ratio_median"],
        "factor_factor_corr_shape": s["factor_factor_corr"]["upper_corr"],
        "factor_factor_abs_ratio": ratio(
            s["factor_factor_corr"]["gen_mean_abs"],
            s["factor_factor_corr"]["gt_mean_abs"],
        ),
        "iv_factor_corr_shape": s["iv_factor_corr"]["matrix_corr"],
        "iv_factor_abs_ratio": ratio(
            s["iv_factor_corr"]["gen_mean_abs"],
            s["iv_factor_corr"]["gt_mean_abs"],
        ),
        "worst_factor_delta_ks": max(row["ks_delta"] for row in s["per_factor"]),
        "worst_factor_delta_ks_name": max(
            s["per_factor"], key=lambda row: row["ks_delta"]
        )["name"],
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    train = result["iv_full_suite"]["train_tail"]
    val = result["iv_full_suite"]["validation"]
    train_joint = result["joint_panel"]["train_tail"]
    val_joint = result["joint_panel"]["validation"]
    lines = [
        "# 660a Train vs Validation Generation Gap",
        "",
        "## Executive Read",
        "",
        "The current 658a failure is not pure out-of-distribution failure. Train-tail generation improves materially versus validation, so distribution shift is real, but the in-training audit still scores only `5/11` and still fails level occupancy, median bias, per-cell conditionality, regime coverage, and some anchor support. The correct diagnosis is a mixed failure: OOD shift amplifies the problem, but the model has not fully learned the training conditional level/path law either.",
        "",
        "## IV Full-Suite Comparison",
        "",
        "| metric | train-tail | validation | read |",
        "| --- | ---: | ---: | --- |",
        f"| score | {train['score']}/11 | {val['score']}/11 | same headline score, different failure severity |",
        f"| cov90 | {fmt(train['cov90'])} | {fmt(val['cov90'])} | validation undercoverage is much worse |",
        f"| h30 cov90 | {fmt(train['h30_cov90'])} | {fmt(val['h30_cov90'])} | train-tail clears horizon coverage, validation is weak |",
        f"| conditional MAE reduction | {fmt(train['conditionality_mae_reduction_pct'])}% | {fmt(val['conditionality_mae_reduction_pct'])}% | conditional signal works in-sample but not out-of-sample |",
        f"| turb/calm width ratio | {fmt(train['turb_calm_width_ratio'])} | {fmt(val['turb_calm_width_ratio'])} | train-tail learns risk-width ordering; validation reverses it |",
        f"| daily KS pass | {train['daily_ks_pass']}/25 | {val['daily_ks_pass']}/25 | local daily law works on both |",
        f"| level KS pass | {train['level_ks_pass']}/25 | {val['level_ks_pass']}/25 | level law remains bad even in-sample |",
        f"| level KS median | {fmt(train['level_ks_median'])} | {fmt(val['level_ks_median'])} | OOD worsens an existing level-placement failure |",
        f"| median-bias pass | {train['median_bias_pass']}/25 | {val['median_bias_pass']}/25 | median placement is not solved on train |",
        f"| bad window-floor rate | {fmt(train['window_floor_bad_rate'])} | {fmt(val['window_floor_bad_rate'])} | validation has many more severe undercoverage windows |",
        f"| kurtosis ratio | {fmt(train['kurtosis_ratio'])} | {fmt(val['kurtosis_ratio'])} | tail shape is better in-sample |",
        f"| corr ratio | {fmt(train['corr_ratio'])} | {fmt(val['corr_ratio'])} | both acceptable, not the main bottleneck |",
        f"| mean-reversion pass | {fmt(train['mean_reversion_pass'])} | {fmt(val['mean_reversion_pass'])} | validation failure is mostly generalization/profile drift |",
        f"| path max-jump KS | {fmt(train['pathwise_max_jump_ks'])} | {fmt(val['pathwise_max_jump_ks'])} | path extremes generalize poorly but remain inside relaxed gate |",
        f"| regime layer2 | {train['regime_layer2'][0]}/{train['regime_layer2'][1]} | {val['regime_layer2'][0]}/{val['regime_layer2'][1]} | regime-cell coverage is not solved in-sample |",
        "",
        "## Joint Anchor Comparison",
        "",
        "| metric | train-tail | validation | read |",
        "| --- | ---: | ---: | --- |",
        f"| factor delta KS mean | {fmt(train_joint['factor_delta_ks_mean'])} | {fmt(val_joint['factor_delta_ks_mean'])} | train is better, but not perfect |",
        f"| factor delta KS pass | {train_joint['factor_delta_ks_pass_020']}/13 | {val_joint['factor_delta_ks_pass_020']}/13 | two factors fail in both splits |",
        f"| q99 abs-delta pass | {train_joint['factor_tail_q99_pass_05_20']}/13 | {val_joint['factor_tail_q99_pass_05_20']}/13 | tail scale is broadly learned |",
        f"| factor corr shape | {fmt(train_joint['factor_factor_corr_shape'])} | {fmt(val_joint['factor_factor_corr_shape'])} | correlation shape generalizes reasonably |",
        f"| factor abs-corr ratio | {fmt(train_joint['factor_factor_abs_ratio'])} | {fmt(val_joint['factor_factor_abs_ratio'])} | shock amplitude is attenuated in both |",
        f"| IV-factor corr shape | {fmt(train_joint['iv_factor_corr_shape'])} | {fmt(val_joint['iv_factor_corr_shape'])} | shape is better on train, still alive on validation |",
        f"| IV-factor abs-corr ratio | {fmt(train_joint['iv_factor_abs_ratio'])} | {fmt(val_joint['iv_factor_abs_ratio'])} | joint shock amplitude is intrinsically too small |",
        f"| worst factor KS | {train_joint['worst_factor_delta_ks_name']}={fmt(train_joint['worst_factor_delta_ks'])} | {val_joint['worst_factor_delta_ks_name']}={fmt(val_joint['worst_factor_delta_ks'])} | credit-spread factors remain hard |",
        "",
        "## Diagnosis",
        "",
        "1. Not pure OOD: if the model were fundamentally fine and only validation were shifted, train-tail should be close to deployable. It is not; it still fails six suites.",
        "2. Not pure in-sample failure either: validation is materially worse on coverage, conditionality, mean reversion, path jumps, and joint correlation amplitude. Distribution shift is a real amplifier.",
        "3. The robust learned part is local movement: daily IV changes, factor deltas, q99 movement scale, and broad correlation shape are alive on both splits.",
        "4. The unresolved learned part is conditional placement: level occupancy, median placement, per-cell coverage geometry, regime-cell coverage, and absolute joint shock amplitude are not reliable even on training windows.",
        "",
        "## Implication",
        "",
        "Existing validation results should not be read as simply 'the models cannot learn the data.' They show a split-generalization problem layered on top of an incomplete in-sample conditional-law fit. The next experiment should therefore test objective balance or level-placement loss on the train-tail audit first. If train-tail becomes strong but validation stays weak, the bottleneck moves to OOD calibration. If train-tail remains weak, more validation tuning is wasted because the model has not learned the training law.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_full11",
        default="results/block_ar/658a_ar652_joint38_s658/train_tail_full11.json",
    )
    parser.add_argument(
        "--val_full11",
        default="results/block_ar/658a_ar652_joint38_s658/full11.json",
    )
    parser.add_argument(
        "--train_joint",
        default="results/block_ar/658a_ar652_joint38_s658/train_tail_joint_panel_audit.json",
    )
    parser.add_argument(
        "--val_joint",
        default="results/block_ar/658a_ar652_joint38_s658/joint_panel_audit.json",
    )
    parser.add_argument(
        "--output_json",
        default="results/autoresearch/660a_train_validation_generation_gap/diagnostics.json",
    )
    parser.add_argument(
        "--output_md",
        default="experiments/backfill/block_ar/ANALYSIS_660a_train_validation_generation_gap.md",
    )
    args = parser.parse_args()

    result = {
        "context": {
            "model": "658a_ar652_joint38_s658",
            "train_tail_pseudo_validation": {
                "test_start": 4070,
                "val_size": 441,
                "original_window_indices": "3569..4009",
                "note": "These windows are inside the original 658a training range.",
            },
            "validation": {
                "test_start": 4511,
                "val_size": 441,
                "original_window_indices": "4010..4450",
            },
        },
        "iv_full_suite": {
            "train_tail": summarize_full(load_json(args.train_full11)),
            "validation": summarize_full(load_json(args.val_full11)),
        },
        "joint_panel": {
            "train_tail": summarize_joint(load_json(args.train_joint)),
            "validation": summarize_joint(load_json(args.val_joint)),
        },
        "classification": {
            "pure_ood_failure": False,
            "pure_in_sample_modeling_failure": False,
            "mixed_failure": True,
            "primary_train_tail_bottleneck": "conditional level/path placement and regime-cell coverage",
            "primary_validation_amplifier": "train-validation distribution shift in levels and weak regime signal",
        },
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(result["classification"], indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
