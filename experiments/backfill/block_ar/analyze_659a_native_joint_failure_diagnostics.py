#!/usr/bin/env python
"""659a: diagnose native-joint IV + anchor-factor failure mechanisms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.train_609a_unified_ar_transition_flow import (  # noqa: E402
    fit_empirical_quantiles,
)
from experiments.backfill.block_ar.train_628a_unified_ar_increment_transition_flow import (  # noqa: E402
    build_blocks,
)
from experiments.backfill.block_ar.train_629a_state_conditioned_increment_flow import (  # noqa: E402
    select_state_increment_scope,
)


MODEL_ARTIFACTS = {
    "641a_ar_singlehead": {
        "full11": "results/autoresearch/641a_joint38_mixedcoord_scale_e8_w2048_s641/full11.json",
        "joint": "results/autoresearch/641a_joint38_mixedcoord_scale_e8_w2048_s641/joint_panel.json",
        "history": "models/backfill/641a_joint38_mixedcoord_scale_e8_w2048_s641/training_history.json",
    },
    "647a_oneshot_singlehead": {
        "full11": "results/autoresearch/647a_joint38_mixed_path_flow_e8_w2048_s647/full11.json",
        "joint": "results/autoresearch/647a_joint38_mixed_path_flow_e8_w2048_s647/joint_panel.json",
        "history": None,
    },
    "652a_oneshot_multihead_joint": {
        "full11": "results/autoresearch/652a_joint38_multihead_mixed_path_e8_w2048_s652/full11.json",
        "joint": "results/autoresearch/652a_joint38_multihead_mixed_path_e8_w2048_s652/joint_panel.json",
        "history": "models/backfill/652a_joint38_multihead_mixed_path_e8_w2048_s652/training_history.json",
    },
    "652a_oneshot_multihead_ivonly": {
        "full11": "results/autoresearch/652a_ivonly_multihead_mixed_path_e8_w2048_s652/full11.json",
        "joint": None,
        "history": "models/backfill/652a_ivonly_multihead_mixed_path_e8_w2048_s652/training_history.json",
    },
    "658a_ar_multihead": {
        "full11": "results/block_ar/658a_ar652_joint38_s658/full11.json",
        "joint": "results/block_ar/658a_ar652_joint38_s658/joint_panel_audit.json",
        "history": "models/backfill/658a_ar652_joint38_s658/training_history.json",
    },
}


def load_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    values = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, values, side="right") / float(x.size)
    cdf_y = np.searchsorted(y, values, side="right") / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def scipy_free_normal_score(u: np.ndarray) -> np.ndarray:
    # Acklam rational approximation for inverse standard normal CDF.
    # Good enough for diagnostics and avoids adding a SciPy dependency.
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )
    p = np.asarray(u, dtype=np.float64)
    out = np.empty_like(p)
    plow = 0.02425
    phigh = 1.0 - plow
    low = p < plow
    mid = (p >= plow) & (p <= phigh)
    high = p > phigh
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    return out


def values_to_scores_fast(
    values: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray,
    cdf_eps: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    for var in range(values.shape[-1]):
        q = np.asarray(quantiles[var], dtype=np.float64)
        flat = values[..., var].reshape(-1)
        idx = np.searchsorted(q, flat, side="left")
        idx_hi = np.clip(idx, 1, q.shape[0] - 1)
        idx_lo = idx_hi - 1
        q_lo = q[idx_lo]
        q_hi = q[idx_hi]
        u_lo = levels[idx_lo]
        u_hi = levels[idx_hi]
        alpha = np.clip((flat - q_lo) / np.maximum(q_hi - q_lo, 1e-12), 0.0, 1.0)
        u = u_lo + alpha * (u_hi - u_lo)
        u = np.where(flat <= q[0], levels[0], u)
        u = np.where(flat >= q[-1], levels[-1], u)
        u = np.clip(u, cdf_eps, 1.0 - cdf_eps)
        out[..., var] = scipy_free_normal_score(u).reshape(values.shape[:-1])
    return out.astype(np.float32)


def group_stats(arr: np.ndarray, iv_count: int) -> dict[str, Any]:
    iv = arr[..., :iv_count]
    factor = arr[..., iv_count:]
    iv_ms = float(np.mean(np.square(iv)))
    factor_ms = float(np.mean(np.square(factor))) if factor.size else float("nan")
    iv_total = iv_ms * iv_count
    factor_total = factor_ms * max(arr.shape[-1] - iv_count, 0)
    denom = iv_total + factor_total
    return {
        "iv_std": float(np.std(iv)),
        "factor_std": float(np.std(factor)) if factor.size else float("nan"),
        "iv_abs": float(np.mean(np.abs(iv))),
        "factor_abs": float(np.mean(np.abs(factor))) if factor.size else float("nan"),
        "iv_mean_square": iv_ms,
        "factor_mean_square": factor_ms,
        "dimension_weighted_iv_share": float(iv_total / denom) if denom > 0 else float("nan"),
        "dimension_weighted_factor_share": float(factor_total / denom) if denom > 0 else float("nan"),
    }


def panel_daily_changes(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    prev = np.concatenate([history[:, -1:, :], future[:, :-1, :]], axis=1)
    return future - prev


def summarize_shift(
    train_history_raw: np.ndarray,
    train_future_raw: np.ndarray,
    val_history_raw: np.ndarray,
    val_future_raw: np.ndarray,
    iv_count: int,
) -> dict[str, Any]:
    train_delta = panel_daily_changes(train_history_raw, train_future_raw)
    val_delta = panel_daily_changes(val_history_raw, val_future_raw)
    rows = {}
    for name, slc in {"iv": slice(0, iv_count), "anchor": slice(iv_count, None)}.items():
        level_ks = [
            ks_statistic(train_future_raw[..., idx], val_future_raw[..., idx])
            for idx in range(*slc.indices(train_future_raw.shape[-1]))
        ]
        delta_ks = [
            ks_statistic(train_delta[..., idx], val_delta[..., idx])
            for idx in range(*slc.indices(train_delta.shape[-1]))
        ]
        rows[name] = {
            "future_level_ks_mean": float(np.nanmean(level_ks)),
            "future_level_ks_median": float(np.nanmedian(level_ks)),
            "future_level_ks_pass_020": int(np.sum(np.asarray(level_ks) < 0.20)),
            "future_delta_ks_mean": float(np.nanmean(delta_ks)),
            "future_delta_ks_median": float(np.nanmedian(delta_ks)),
            "future_delta_ks_pass_020": int(np.sum(np.asarray(delta_ks) < 0.20)),
        }
    return rows


def summarize_realized_regime_signal(
    val_history_raw: np.ndarray,
    val_future_raw: np.ndarray,
    iv_count: int,
) -> dict[str, float]:
    hist_delta = np.diff(val_history_raw[..., :iv_count], axis=1)
    hist_vov = np.std(hist_delta.reshape(hist_delta.shape[0], -1), axis=1)
    future_delta = panel_daily_changes(
        val_history_raw[..., :iv_count],
        val_future_raw[..., :iv_count],
    )
    future_abs = np.mean(np.abs(future_delta), axis=(1, 2))
    calm = hist_vov <= np.quantile(hist_vov, 0.20)
    turb = hist_vov >= np.quantile(hist_vov, 0.80)
    return {
        "n_calm": int(np.sum(calm)),
        "n_turb": int(np.sum(turb)),
        "future_abs_move_turb_calm_ratio": float(
            np.mean(future_abs[turb]) / max(np.mean(future_abs[calm]), 1e-12)
        ),
        "spearman_proxy_hist_vov_future_abs_corr": float(
            np.corrcoef(np.argsort(np.argsort(hist_vov)), np.argsort(np.argsort(future_abs)))[0, 1]
        ),
        "hist_vov_q20": float(np.quantile(hist_vov, 0.20)),
        "hist_vov_q80": float(np.quantile(hist_vov, 0.80)),
    }


def summarize_suite(name: str, full: dict[str, Any] | None) -> dict[str, Any]:
    if full is None:
        return {"name": name, "missing": True}
    dist = full.get("distributional_fidelity", {})
    return {
        "name": name,
        "n_pass": get_nested(full, ["summary", "n_pass"]),
        "failed": get_nested(full, ["summary", "failed_suites"], []),
        "cov90": get_nested(full, ["coverage", "overall", "0.9"]),
        "conditionality_mae_reduction_pct": get_nested(
            full, ["conditionality", "mae_reduction_pct"]
        ),
        "turb_calm_width_ratio": get_nested(full, ["conditionality", "turb_calm_ratio"]),
        "daily_ks_pass": get_nested(dist, ["ks_test", "n_pass"]),
        "daily_ks_median": get_nested(dist, ["ks_test", "median_stat"]),
        "level_ks_pass": get_nested(dist, ["ks_level_test", "n_pass"]),
        "level_ks_median": get_nested(dist, ["ks_level_test", "median_stat"]),
        "median_bias_pass": get_nested(dist, ["median_bias", "n_pass"]),
        "window_floor_rate": get_nested(dist, ["window_floor", "bad_window_rate"]),
        "corr_ratio": get_nested(full, ["cross_cell_correlation", "corr_ratio"]),
        "rank_ratio": get_nested(full, ["cross_cell_correlation", "rank_ratio"]),
        "kurtosis_ratio": get_nested(full, ["time_series", "kurtosis", "kurtosis_ratio"]),
        "pathwise_max_jump_ks": get_nested(
            full, ["pathwise_jump_realism", "pathwise_max_jump", "ks_stat"]
        ),
        "mean_reversion_ratio": get_nested(full, ["mean_reversion", "mr_gt_ratio"]),
        "regime_layer2": f"{get_nested(full, ['regime_coverage', 'layer2_n_passing'])}/{get_nested(full, ['regime_coverage', 'layer2_n_total'])}",
    }


def flatten_grid(grid: Any, prefix: str = "iv") -> list[dict[str, Any]]:
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 2:
        return []
    rows = []
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            rows.append(
                {
                    "cell": [int(r), int(c)],
                    "name": f"{prefix}:{r * arr.shape[1] + c:02d}",
                    "value": float(arr[r, c]),
                }
            )
    return rows


def top_grid(grid: Any, *, largest: bool = True, n: int = 5) -> list[dict[str, Any]]:
    rows = flatten_grid(grid)
    rows = [row for row in rows if np.isfinite(row["value"])]
    rows.sort(key=lambda row: row["value"], reverse=largest)
    return rows[:n]


def summarize_latest_iv_detail(full: dict[str, Any] | None) -> dict[str, Any]:
    if full is None:
        return {"missing": True}
    dist = full.get("distributional_fidelity", {})
    coverage = full.get("coverage", {})
    conditionality = full.get("conditionality", {})
    time_series = full.get("time_series", {})
    median_above = get_nested(dist, ["median_bias", "above_frac"])
    median_extremes = []
    if median_above is not None:
        for row in flatten_grid(median_above):
            row["distance_from_half"] = float(abs(row["value"] - 0.5))
            median_extremes.append(row)
        median_extremes.sort(key=lambda row: row["distance_from_half"], reverse=True)
    h30_coverage = get_nested(coverage, ["per_cell_coverage", "30"])
    return {
        "worst_daily_ks": top_grid(get_nested(dist, ["ks_test", "ks_grid"]), largest=True),
        "worst_level_ks": top_grid(get_nested(dist, ["ks_level_test", "ks_grid"]), largest=True),
        "lowest_h30_coverage": top_grid(h30_coverage, largest=False),
        "worst_median_bias_frac": median_extremes[:5],
        "worst_conditional_mae_reduction": top_grid(
            conditionality.get("per_cell_mae_reduction"), largest=False
        ),
        "worst_mean_reversion_active_cells": full.get("mean_reversion", {}).get(
            "worst_active_cells", []
        )[:5],
        "kurtosis_ratio": get_nested(time_series, ["kurtosis", "kurtosis_ratio"]),
        "tail_q99_pass": get_nested(time_series, ["tail_scale", "n_pass"]),
        "tail_q99_worst_ratio": get_nested(time_series, ["tail_scale", "worst_cell_ratio"]),
        "tail_q99_best_ratio": get_nested(time_series, ["tail_scale", "best_cell_ratio"]),
    }


def summarize_joint(name: str, joint: dict[str, Any] | None) -> dict[str, Any]:
    if joint is None:
        return {"name": name, "missing": True}
    s = joint.get("summary", {})
    return {
        "name": name,
        "factor_delta_ks_mean": s.get("factor_delta_ks_mean"),
        "factor_delta_ks_pass_020": s.get("factor_delta_ks_pass_020"),
        "factor_q99_pass": s.get("factor_tail_q99_pass_05_20"),
        "factor_factor_corr_shape": get_nested(s, ["factor_factor_corr", "upper_corr"]),
        "factor_factor_gt_mean_abs": get_nested(s, ["factor_factor_corr", "gt_mean_abs"]),
        "factor_factor_gen_mean_abs": get_nested(s, ["factor_factor_corr", "gen_mean_abs"]),
        "iv_factor_corr_shape": get_nested(s, ["iv_factor_corr", "matrix_corr"]),
        "iv_factor_gt_mean_abs": get_nested(s, ["iv_factor_corr", "gt_mean_abs"]),
        "iv_factor_gen_mean_abs": get_nested(s, ["iv_factor_corr", "gen_mean_abs"]),
        "worst_factor_ks": max(
            (row.get("ks_delta", float("nan")) for row in s.get("per_factor", [])),
            default=float("nan"),
        ),
    }


def summarize_latest_anchor_detail(joint: dict[str, Any] | None) -> dict[str, Any]:
    if joint is None:
        return {"missing": True}
    rows = []
    for row in joint.get("summary", {}).get("per_factor", []):
        copied = dict(row)
        gt_range = max(float(copied["gt_max"]) - float(copied["gt_min"]), 1e-12)
        copied["gen_below_gt_range_frac"] = float(
            max(0.0, float(copied["gt_min"]) - float(copied["gen_min"])) / gt_range
        )
        copied["gen_above_gt_range_frac"] = float(
            max(0.0, float(copied["gen_max"]) - float(copied["gt_max"])) / gt_range
        )
        copied["max_range_excursion_frac"] = max(
            copied["gen_below_gt_range_frac"],
            copied["gen_above_gt_range_frac"],
        )
        copied["q99_distance_from_one"] = float(abs(float(copied["q99_abs_delta_ratio"]) - 1.0))
        rows.append(copied)
    by_ks = sorted(rows, key=lambda row: row.get("ks_delta", float("nan")), reverse=True)
    by_range = sorted(rows, key=lambda row: row.get("max_range_excursion_frac", 0.0), reverse=True)
    by_tail = sorted(rows, key=lambda row: row.get("q99_distance_from_one", 0.0), reverse=True)
    return {
        "worst_delta_ks_factors": by_ks[:5],
        "largest_level_range_excursions": by_range[:5],
        "largest_tail_ratio_errors": by_tail[:5],
    }


def build_target_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    data_args = SimpleNamespace(
        state_scope="joint38",
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        iv_count=args.iv_count,
        max_train_windows=args.max_train_windows,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
    )
    _columns, metadata, train_block, val_block = build_blocks(data_args)
    (
        train_level,
        train_increment,
        train_future_level,
        train_future_increment,
        _train_raw,
        train_specs,
    ) = select_state_increment_scope(train_block, "joint38", args.iv_count)
    (
        val_level,
        val_increment,
        val_future_level,
        val_future_increment,
        _val_raw,
        _val_specs,
    ) = select_state_increment_scope(val_block, "joint38", args.iv_count)

    level_quantiles, levels = fit_empirical_quantiles(
        train_level,
        train_future_level,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
    )
    increment_quantiles, _ = fit_empirical_quantiles(
        train_increment,
        train_future_increment,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
    )

    train_level_scores = values_to_scores_fast(
        train_level, level_quantiles, levels, args.cdf_eps
    )
    train_future_level_scores = values_to_scores_fast(
        train_future_level, level_quantiles, levels, args.cdf_eps
    )
    train_future_increment_scores = values_to_scores_fast(
        train_future_increment, increment_quantiles, levels, args.cdf_eps
    )

    val_level_scores = values_to_scores_fast(
        val_level, level_quantiles, levels, args.cdf_eps
    )
    val_future_level_scores = values_to_scores_fast(
        val_future_level, level_quantiles, levels, args.cdf_eps
    )
    val_future_increment_scores = values_to_scores_fast(
        val_future_increment, increment_quantiles, levels, args.cdf_eps
    )

    def ar_target(level_scores: np.ndarray, future_level_scores: np.ndarray, future_inc_scores: np.ndarray) -> np.ndarray:
        prefix_level = np.concatenate([level_scores, future_level_scores[:, :-1]], axis=1)
        current = prefix_level[:, args.history_len - 1 : args.history_len - 1 + args.future_len]
        level_delta = future_level_scores - current
        out = future_inc_scores.copy()
        out[..., : args.iv_count] = level_delta[..., : args.iv_count]
        return out

    def path_target(level_scores: np.ndarray, future_level_scores: np.ndarray, future_inc_scores: np.ndarray) -> np.ndarray:
        level_delta = future_level_scores - level_scores[:, -1:, :]
        out = future_inc_scores.copy()
        out[..., : args.iv_count] = level_delta[..., : args.iv_count]
        return out

    ar_train = ar_target(
        train_level_scores, train_future_level_scores, train_future_increment_scores
    )
    ar_val = ar_target(val_level_scores, val_future_level_scores, val_future_increment_scores)
    path_train = path_target(
        train_level_scores, train_future_level_scores, train_future_increment_scores
    )
    path_val = path_target(val_level_scores, val_future_level_scores, val_future_increment_scores)

    return {
        "metadata": metadata,
        "state_specs": [getattr(spec, "name", str(spec)) for spec in train_specs],
        "ar_mixed_coordinate_train": group_stats(ar_train, args.iv_count),
        "ar_mixed_coordinate_val": group_stats(ar_val, args.iv_count),
        "oneshot_path_coordinate_train": group_stats(path_train, args.iv_count),
        "oneshot_path_coordinate_val": group_stats(path_val, args.iv_count),
        "raw_train_val_shift": summarize_shift(
            train_block.history_state,
            train_block.future_state,
            val_block.history_state,
            val_block.future_state,
            args.iv_count,
        ),
        "realized_iv_regime_signal_val": summarize_realized_regime_signal(
            val_block.history_state,
            val_block.future_state,
            args.iv_count,
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "NA"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(val):
            return "NA"
        return f"{val:.{digits}f}"

    suite = result["suite_comparison"]
    joint = result["joint_panel_comparison"]
    target = result["target_and_data_diagnostics"]
    lines = [
        "# 659a Native-Joint Failure Diagnostics",
        "",
        "## Executive Read",
        "",
        "The native joint models are not mainly failing because IV and anchors cannot share one stochastic source. They are failing because the shared objective sees two very different statistical jobs at once: IV level-score moves are small, mean-reverting, and level-occupancy sensitive, while anchor-factor increment scores are near unit scale and dominate the mixed-coordinate flow target. The models learn local daily realism and broad correlation shape, but underlearn conditional IV level placement and attenuate per-scenario IV-factor shock amplitude.",
        "",
        "## IV Suite Comparison",
        "",
        "| model | score | cov90 | cond MAE red | daily KS | level KS | median bias | kurt ratio | corr ratio | path KS | failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in suite:
        failed = ", ".join(row.get("failed", [])) if not row.get("missing") else "missing"
        lines.append(
            f"| {row['name']} | {row.get('n_pass', 'NA')}/11 | "
            f"{fmt(row.get('cov90'))} | "
            f"{fmt(row.get('conditionality_mae_reduction_pct'), 2)}% | "
            f"{row.get('daily_ks_pass', 'NA')}/25 | {row.get('level_ks_pass', 'NA')}/25 | "
            f"{row.get('median_bias_pass', 'NA')}/25 | "
            f"{fmt(row.get('kurtosis_ratio'))} | "
            f"{fmt(row.get('corr_ratio'))} | "
            f"{fmt(row.get('pathwise_max_jump_ks'))} | {failed} |"
        )
    lines.extend(
        [
            "",
            "## Anchor-Factor Audit",
            "",
            "| model | factor KS mean | factor KS pass | q99 pass | factor corr shape | factor corr abs gen/GT | IV-factor shape | IV-factor abs gen/GT | worst factor KS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in joint:
        if row.get("missing"):
            lines.append(f"| {row['name']} | missing | missing | missing | missing | missing | missing | missing | missing |")
            continue
        factor_ratio = row["factor_factor_gen_mean_abs"] / max(row["factor_factor_gt_mean_abs"], 1e-12)
        iv_factor_ratio = row["iv_factor_gen_mean_abs"] / max(row["iv_factor_gt_mean_abs"], 1e-12)
        lines.append(
            f"| {row['name']} | {row['factor_delta_ks_mean']:.3f} | "
            f"{row['factor_delta_ks_pass_020']}/13 | {row['factor_q99_pass']}/13 | "
            f"{row['factor_factor_corr_shape']:.3f} | {factor_ratio:.3f} | "
            f"{row['iv_factor_corr_shape']:.3f} | {iv_factor_ratio:.3f} | "
            f"{row['worst_factor_ks']:.3f} |"
        )
    ar_train = target["ar_mixed_coordinate_train"]
    path_train = target["oneshot_path_coordinate_train"]
    shift = target["raw_train_val_shift"]
    regime = target["realized_iv_regime_signal_val"]
    detail = result["latest_model_detail"]
    iv_detail = detail["iv_surface_658a"]
    anchor_detail = detail["anchor_658a"]

    def cell_list(rows: list[dict[str, Any]], key: str = "value") -> str:
        return ", ".join(f"{row['name']}={fmt(row.get(key))}" for row in rows)

    def factor_list(rows: list[dict[str, Any]], key: str) -> str:
        return ", ".join(f"{row['name']}={fmt(row.get(key))}" for row in rows)

    lines.extend(
        [
            "",
            "## Target-Scale Interference",
            "",
            f"- AR mixed-coordinate train target: IV std `{ar_train['iv_std']:.3f}`, anchor std `{ar_train['factor_std']:.3f}`, dimension-weighted factor share `{ar_train['dimension_weighted_factor_share']:.3f}`.",
            f"- One-shot path-coordinate train target: IV std `{path_train['iv_std']:.3f}`, anchor std `{path_train['factor_std']:.3f}`, dimension-weighted factor share `{path_train['dimension_weighted_factor_share']:.3f}`.",
            "- Interpretation: in the clean joint objective, factor increment channels carry most of the target energy. IV channels are numerous, but their generated coordinate is much smaller. This makes IV conditional level placement easy to underfit while still achieving good global flow loss and strong factor realism.",
            "",
            "## Data Framing",
            "",
            f"- IV train-vs-val future level KS mean `{shift['iv']['future_level_ks_mean']:.3f}`, delta KS mean `{shift['iv']['future_delta_ks_mean']:.3f}`.",
            f"- Anchor train-vs-val future level KS mean `{shift['anchor']['future_level_ks_mean']:.3f}`, delta KS mean `{shift['anchor']['future_delta_ks_mean']:.3f}`.",
            f"- Validation realized IV turbulent/calm future absolute-move ratio `{regime['future_abs_move_turb_calm_ratio']:.3f}` with rank-corr proxy `{regime['spearman_proxy_hist_vov_future_abs_corr']:.3f}`.",
            "- Interpretation: daily increments are much more stable than levels, so models can learn realistic local movement while failing level occupancy. The validation split does not provide a strong realized future-width signal from history volatility, so regime-width conditionality is weakly identifiable.",
            "",
            "## Failure Mechanism",
            "",
            "1. IV surface: generated daily changes, path jumps, cross-cell structure, and aggregate mean reversion are mostly alive; the failure is conditional level placement and coverage geometry, especially per-cell/regime coverage and level KS.",
            "2. Anchor list: factor daily increments and factor-factor correlation shape are realistic, but generated factor levels can leave plausible validation ranges for some sparse/stale factors, and absolute co-movement amplitude is systematically attenuated.",
            "3. Joint scenario: IV-factor correlation shape is learned, but the per-scenario shock strength is too small. This is why scenarios can look directionally coherent on average but still feel weak as a joint stress story.",
            "4. Interference: the shared flow objective is clean, but not group-balanced. It rewards learning high-energy anchor increments and local daily movement more than the low-amplitude IV level-score decisions that drive coverage, conditionality, and level occupancy.",
            "",
            "## 658a Failure Anatomy",
            "",
            f"- Worst IV level-KS cells: {cell_list(iv_detail['worst_level_ks'])}.",
            f"- Lowest IV h30 coverage cells: {cell_list(iv_detail['lowest_h30_coverage'])}.",
            f"- Worst IV conditional MAE reductions: {cell_list(iv_detail['worst_conditional_mae_reduction'])}.",
            f"- Worst anchor delta-KS factors: {factor_list(anchor_detail['worst_delta_ks_factors'], 'ks_delta')}.",
            f"- Largest anchor level-range excursions: {factor_list(anchor_detail['largest_level_range_excursions'], 'max_range_excursion_frac')}.",
            f"- Largest anchor tail-ratio errors: {factor_list(anchor_detail['largest_tail_ratio_errors'], 'q99_abs_delta_ratio')}.",
            "- Interpretation: the latest coherent joint AR model is not exploding and is not ignoring anchors. Its realism breaks at conditional placement: certain IV surface cells are persistently undercovered at h30, and some macro/market anchors drift outside the validation level range even when their one-day deltas look statistically acceptable.",
            "",
            "## Decision",
            "",
            "Do not add another architecture component yet. The next principled move is a group-balanced training falsifier: same architecture, same shared source, same data, but make IV level-score placement and anchor increments comparably visible to the loss. If that improves IV coverage/level occupancy without damaging anchor KS/correlation, the current paradigm remains viable. If not, the bottleneck is not decoder expressiveness but missing conditional signal/data framing.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument(
        "--output_json",
        default="results/autoresearch/659a_native_joint_failure_diagnostics/diagnostics.json",
    )
    parser.add_argument(
        "--output_md",
        default="experiments/backfill/block_ar/ANALYSIS_659a_native_joint_failure_diagnostics.md",
    )
    args = parser.parse_args()

    suite_rows = []
    joint_rows = []
    histories = {}
    for name, paths in MODEL_ARTIFACTS.items():
        suite_rows.append(summarize_suite(name, load_json(paths["full11"])))
        joint_rows.append(summarize_joint(name, load_json(paths["joint"])))
        history = load_json(paths["history"])
        if history:
            best = min(history, key=lambda row: row.get("val_loss", float("inf")))
            histories[name] = best

    result = {
        "suite_comparison": suite_rows,
        "joint_panel_comparison": joint_rows,
        "best_training_records": histories,
        "latest_model_detail": {
            "iv_surface_658a": summarize_latest_iv_detail(
                load_json(MODEL_ARTIFACTS["658a_ar_multihead"]["full11"])
            ),
            "anchor_658a": summarize_latest_anchor_detail(
                load_json(MODEL_ARTIFACTS["658a_ar_multihead"]["joint"])
            ),
        },
        "target_and_data_diagnostics": build_target_diagnostics(args),
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(make_serializable(result), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
