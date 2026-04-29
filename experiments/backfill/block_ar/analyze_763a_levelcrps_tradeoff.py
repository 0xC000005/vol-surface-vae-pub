#!/usr/bin/env python
"""763a: attribute the 755a -> 762a level-CRPS trade-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def suite_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": f"{result['summary']['n_pass']}/11",
        "failed": result["summary"]["failed_suites"],
        "cov90": result["coverage"]["overall"]["0.9"],
        "calerr": result["coverage"]["calibration_error"],
        "level_ks": result["distributional_fidelity"]["ks_level_test"]["n_pass"],
        "median_bias": result["distributional_fidelity"]["median_bias"]["n_pass"],
        "daily_ks": result["distributional_fidelity"]["ks_test"]["n_pass"],
        "cointegration_ratio": result["cointegration"]["gen_gt_ratio"],
        "cointegration_worst": result["cointegration"]["worst_cell_ratio"],
        "mean_reversion": result["mean_reversion"]["overall_pass"],
        "mr_active_mean": result["mean_reversion"]["full_horizon"]["mean_active_pass_rate"],
        "pathwise_ks": result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"],
        "kurtosis_ratio": result["time_series"]["kurtosis"]["kurtosis_ratio"],
        "risk_state_allocation": result["risk_state_allocation"]["overall_pass"],
    }


def grid_delta(
    newer: dict[str, Any],
    older: dict[str, Any],
    *keys: str,
) -> dict[str, float | list[int]]:
    new_grid = np.asarray(nested_get(newer, keys), dtype=np.float64)
    old_grid = np.asarray(nested_get(older, keys), dtype=np.float64)
    delta = new_grid - old_grid
    idx = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    return {
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "max_abs_delta": float(delta[idx]),
        "max_abs_delta_cell": [int(idx[0]), int(idx[1])],
    }


def nested_get(value: Any, keys: tuple[str, ...]) -> Any:
    cur = value
    for key in keys:
        cur = cur[key]
    return cur


def best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    return min(history, key=lambda row: float(row.get("val_total", float("inf"))))


def objective_contrib(row: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, weight in weights.items():
        metric = float(row.get(name, 0.0))
        out[name] = metric * float(weight)
    out["sum_tracked"] = float(sum(out.values()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="results/block_ar/763a_levelcrps_tradeoff")
    args = parser.parse_args()

    paths = {
        "755_val": "results/block_ar/755a_iv_shortprefix_fm_k5/iv_val_full11_s64.json",
        "755_train_tail": "results/block_ar/755a_iv_shortprefix_fm_k5/iv_train_tail_full11_s64.json",
        "762_val": "results/block_ar/762a_iv_shortprefix_levelcrps_w005/iv_val_full11_s64.json",
        "762_train_tail": "results/block_ar/762a_iv_shortprefix_levelcrps_w005/iv_train_tail_full11_s64.json",
        "755_history": "models/backfill/755a_iv_shortprefix_fm_k5_w02_e2_s7551/training_history.json",
        "762_history": "models/backfill/762a_iv_shortprefix_levelcrps_w005_s7621/training_history.json",
    }
    data = {name: load_json(path) for name, path in paths.items()}

    hist_755 = best_epoch(data["755_history"])
    hist_762 = best_epoch(data["762_history"])
    weights_755 = {
        "val_fm_loss": 1.0,
        "val_energy": 0.2,
        "val_channel_level_energy": 0.05,
        "val_free_running_fm_loss": 0.2,
    }
    weights_762 = {
        **weights_755,
        "val_level_marginal_crps": 0.05,
    }

    sample_std = {
        "755_val_sample_norm_std": float(hist_755["val_sample_norm_std"]),
        "762_val_sample_norm_std": float(hist_762["val_sample_norm_std"]),
        "755_val_sample_level_std": float(hist_755["val_sample_level_std"]),
        "762_val_sample_level_std": float(hist_762["val_sample_level_std"]),
        "target_norm_std": float(hist_762["val_target_norm_std"]),
        "target_level_std": float(hist_762["val_target_level_std"]),
    }
    sample_std["norm_std_delta_pct"] = (
        sample_std["762_val_sample_norm_std"] / sample_std["755_val_sample_norm_std"] - 1.0
    )
    sample_std["level_std_delta_pct"] = (
        sample_std["762_val_sample_level_std"] / sample_std["755_val_sample_level_std"] - 1.0
    )

    val_summary = {
        "755a": suite_metrics(data["755_val"]),
        "762a": suite_metrics(data["762_val"]),
    }
    train_tail_summary = {
        "755a": suite_metrics(data["755_train_tail"]),
        "762a": suite_metrics(data["762_train_tail"]),
    }
    analysis = {
        "summary": {
            "decision": "reject_762a",
            "mechanism": (
                "The level marginal CRPS term is a proper score but its current scale "
                "narrows generated support and competes with structural dynamics. "
                "It improves some validation cointegration/pathwise geometry while "
                "worsening coverage, level-KS, median-bias, mean reversion, and "
                "train-tail robustness."
            ),
            "next_step": (
                "Do not stack another scalar level loss. Analyze objective/readout "
                "interaction or consider a cleaner reformulation that avoids adding "
                "proper scores with uncontrolled relative scale."
            ),
        },
        "validation": val_summary,
        "train_tail": train_tail_summary,
        "training_objective": {
            "755_best_epoch": int(hist_755["epoch"]),
            "762_best_epoch": int(hist_762["epoch"]),
            "755_val_total": float(hist_755["val_total"]),
            "762_val_total": float(hist_762["val_total"]),
            "755_weighted_components": objective_contrib(hist_755, weights_755),
            "762_weighted_components": objective_contrib(hist_762, weights_762),
            "level_crps_component_vs_channel_energy": float(
                (0.05 * hist_762["val_level_marginal_crps"])
                / max(0.05 * hist_762["val_channel_level_energy"], 1e-12)
            ),
        },
        "sample_spread": sample_std,
        "grid_deltas_val_762_minus_755": {
            "coverage_h30": grid_delta(
                data["762_val"],
                data["755_val"],
                "coverage",
                "per_cell_coverage",
                "30",
            ),
            "level_ks": grid_delta(
                data["762_val"],
                data["755_val"],
                "distributional_fidelity",
                "ks_level_test",
                "ks_grid",
            ),
            "median_above_frac": grid_delta(
                data["762_val"],
                data["755_val"],
                "distributional_fidelity",
                "median_bias",
                "above_frac",
            ),
            "cointegration_ratio": grid_delta(
                data["762_val"],
                data["755_val"],
                "cointegration",
                "per_cell_ratio_grid",
            ),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_md = out_dir / "summary.md"
    out_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps(analysis["summary"], indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


def render_markdown(analysis: dict[str, Any]) -> str:
    val = analysis["validation"]
    train = analysis["train_tail"]
    obj = analysis["training_objective"]
    spread = analysis["sample_spread"]
    deltas = analysis["grid_deltas_val_762_minus_755"]
    lines = [
        "# 763a Level-CRPS Trade-Off Attribution",
        "",
        "## Decision",
        "",
        analysis["summary"]["mechanism"],
        "",
        "## Score Comparison",
        "",
        "| split | model | score | failed | cov90 | level KS | bias | coint worst | MR | kurt |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for split_name, block in [("val", val), ("train-tail", train)]:
        for model_name in ["755a", "762a"]:
            item = block[model_name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_name,
                        model_name,
                        item["score"],
                        ", ".join(item["failed"]),
                        f"{item['cov90']:.3f}",
                        f"{item['level_ks']}/25",
                        f"{item['median_bias']}/25",
                        f"{item['cointegration_worst']:.3f}",
                        str(item["mean_reversion"]).lower(),
                        f"{item['kurtosis_ratio']:.3f}",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Objective Scale",
            "",
            f"- 755a best val objective: `{obj['755_val_total']:.4f}` at epoch `{obj['755_best_epoch']}`.",
            f"- 762a best val objective: `{obj['762_val_total']:.4f}` at epoch `{obj['762_best_epoch']}`.",
            f"- 762a weighted level-CRPS contribution is `{obj['762_weighted_components']['val_level_marginal_crps']:.4f}`.",
            f"- 762a weighted channel-level-energy contribution is `{obj['762_weighted_components']['val_channel_level_energy']:.4f}`.",
            f"- Level-CRPS contribution is `{obj['level_crps_component_vs_channel_energy']:.1f}x` channel-level-energy contribution.",
            "",
            "## Spread Effect",
            "",
            f"- Validation sample normalized std changed `{spread['755_val_sample_norm_std']:.3f} -> {spread['762_val_sample_norm_std']:.3f}` ({spread['norm_std_delta_pct']:+.1%}).",
            f"- Validation sample level std changed `{spread['755_val_sample_level_std']:.3f} -> {spread['762_val_sample_level_std']:.3f}` ({spread['level_std_delta_pct']:+.1%}).",
            "",
            "## Largest Validation Grid Deltas",
            "",
        ]
    )
    for name, item in deltas.items():
        lines.append(
            f"- `{name}`: mean delta `{item['mean_delta']:+.4f}`, "
            f"median delta `{item['median_delta']:+.4f}`, "
            f"max abs delta `{item['max_abs_delta']:+.4f}` at cell `{item['max_abs_delta_cell']}`."
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            analysis["summary"]["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
