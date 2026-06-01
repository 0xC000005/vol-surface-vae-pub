#!/usr/bin/env python
"""Rank-preserving marginal readout diagnostic for prefix-latent scenarios.

This is a bounded TestFlight for the conditionality-aware readout idea. It fits
a positive horizon/factor alpha map on calibration backtest rows, then applies
that same marginal scale map to held-out backtest rows and fixed-start
conditionality cases. Positive scaling around each ensemble mean preserves the
per-variable sample ranks, so it is a minimal empirical-copula-style readout
probe rather than a new generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_component_global_calibration import (  # noqa: E402
    _component_samples_for_row,
    _parse_grid,
    split_calibration_rows,
)
from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_shape_audit import (  # noqa: E402
    _bootstrap_pairs,
    _load_repeat_cases,
    _load_start_only_cases,
    _max_start_difference,
    _pairwise,
    _ratio,
    _summarize_pair_rows,
    load_observed_cases,
    plot_path_metric_summary,
    plot_raw_factor_fans,
    plot_shape_metric_summary,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    score_sample_distribution,
    summarize_method_scores,
)
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (  # noqa: E402
    _selected_history_block,
)


DEFAULT_BACKTEST_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_backtest_heldout_29w_s96/"
    "component_backtest_report.json"
)
DEFAULT_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/"
    "bridge_eval_report.json"
)
DEFAULT_COMPONENT_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_mixture_fixed_start_904k_s384_uncalibrated"
)
DEFAULT_CONTROL_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_fixed_start_controls_904k_s384_uncalibrated"
)
DEFAULT_VARIANT_DIR = "decoder_component_topk_narrative_start_checked_gen_temp_0p50"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _target_delta_for_row(
    row: dict[str, Any],
    *,
    history_raw: np.ndarray,
    future_raw: np.ndarray,
) -> np.ndarray:
    start_idx = int(row.get("start_window_index", -1))
    if start_idx < 0 or start_idx >= future_raw.shape[0]:
        raise IndexError(f"start_window_index {start_idx} outside future_raw")
    start = np.asarray(history_raw[start_idx, -1, :], dtype=np.float32)
    future = np.asarray(future_raw[start_idx], dtype=np.float32)
    return (future - start[None, :]).astype(np.float32)


def scale_samples_by_alpha_map(
    samples: np.ndarray, alpha_map: np.ndarray
) -> np.ndarray:
    """Scale [S,T,C] samples around the ensemble mean with [T,C] alphas."""

    arr = np.asarray(samples, dtype=np.float32)
    alpha = np.asarray(alpha_map, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("samples must have shape [S,T,C]")
    if alpha.shape != arr.shape[1:]:
        raise ValueError(
            f"alpha_map shape {alpha.shape} does not match {arr.shape[1:]}"
        )
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("alpha_map must contain finite positive values")
    mean = np.nanmean(arr, axis=0, keepdims=True)
    return (mean + alpha[None, :, :] * (arr - mean)).astype(np.float32)


def fit_rank_preserving_alpha_map(
    rows: list[dict[str, Any]],
    *,
    alpha_grid: list[float],
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    target_coverage: float,
) -> dict[str, Any]:
    """Choose per-horizon/factor alpha by calibration coverage error."""

    if not rows:
        raise ValueError("at least one calibration row is required")
    targets = []
    samples = []
    for row in rows:
        row_samples, _delta_scale = _component_samples_for_row(row)
        samples.append(row_samples)
        targets.append(
            _target_delta_for_row(row, history_raw=history_raw, future_raw=future_raw)
        )
    sample_arr = np.stack(samples, axis=0)
    target_arr = np.stack(targets, axis=0)
    errors = []
    coverages = []
    for alpha in alpha_grid:
        scaled = np.stack(
            [
                scale_samples_by_alpha_map(item, np.full(item.shape[1:], alpha))
                for item in sample_arr
            ],
            axis=0,
        )
        q10 = np.nanquantile(scaled, 0.10, axis=1)
        q90 = np.nanquantile(scaled, 0.90, axis=1)
        coverage = np.mean(
            (target_arr >= q10) & (target_arr <= q90),
            axis=0,
        )
        coverages.append(coverage.astype(np.float32))
        errors.append(np.abs(coverage - float(target_coverage)).astype(np.float32))
    error_arr = np.stack(errors, axis=0)
    coverage_arr = np.stack(coverages, axis=0)
    best_idx = np.argmin(error_arr, axis=0)
    grid = np.asarray(alpha_grid, dtype=np.float32)
    alpha_map = grid[best_idx]
    chosen_coverage = np.take_along_axis(
        coverage_arr,
        best_idx[None, :, :],
        axis=0,
    )[0]
    return {
        "alpha_map": alpha_map.astype(np.float32),
        "chosen_coverage": chosen_coverage.astype(np.float32),
        "alpha_grid": [float(item) for item in alpha_grid],
        "target_coverage": float(target_coverage),
        "calibration_window_count": int(len(rows)),
    }


def _score_rows_with_alpha_map(
    rows: list[dict[str, Any]],
    *,
    alpha_map: np.ndarray,
    history_raw: np.ndarray,
    future_raw: np.ndarray,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        samples, delta_scale = _component_samples_for_row(row)
        target = _target_delta_for_row(
            row,
            history_raw=history_raw,
            future_raw=future_raw,
        )
        calibrated = scale_samples_by_alpha_map(samples, alpha_map)
        methods = {
            "persistence": row["methods"]["persistence"],
            "averaged_prefix": row["methods"]["averaged_prefix"],
            "component_prefix_mixture": row["methods"]["component_prefix_mixture"],
            "component_rank_preserving_readout": score_sample_distribution(
                calibrated,
                target,
                scale=delta_scale,
            ),
        }
        scored.append(
            {
                "row_no": int(row.get("row_no", len(scored))),
                "window_index": int(row.get("window_index", -1)),
                "window_id": str(row.get("window_id", "")),
                "start_window_index": int(row.get("start_window_index", -1)),
                "methods": methods,
            }
        )
    return scored


def _alpha_summary(
    alpha_map: np.ndarray, chosen_coverage: np.ndarray
) -> dict[str, Any]:
    alpha = np.asarray(alpha_map, dtype=np.float64).reshape(-1)
    coverage = np.asarray(chosen_coverage, dtype=np.float64).reshape(-1)
    return {
        "alpha_min": float(np.min(alpha)),
        "alpha_p10": float(np.quantile(alpha, 0.10)),
        "alpha_median": float(np.median(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_p90": float(np.quantile(alpha, 0.90)),
        "alpha_max": float(np.max(alpha)),
        "coverage_mean_after_selection": float(np.mean(coverage)),
        "coverage_median_after_selection": float(np.median(coverage)),
        "alpha_one_fraction": float(np.mean(np.isclose(alpha, 1.0))),
    }


def _scale_cases(
    cases: list[dict[str, Any]], alpha_map: np.ndarray
) -> list[dict[str, Any]]:
    scaled_cases = []
    for case in cases:
        states = np.asarray(case["states"], dtype=np.float32)
        start = np.asarray(case["start"], dtype=np.float32)
        deltas = states - start[None, None, :]
        scaled_deltas = scale_samples_by_alpha_map(deltas, alpha_map)
        scaled_cases.append({**case, "states": (start[None, None, :] + scaled_deltas)})
    return scaled_cases


def _fixed_start_report(
    *,
    alpha_map: np.ndarray,
    component_root: Path,
    control_root: Path,
    variant_dir: str,
    output_dir: Path,
    max_start_abs_diff: float,
    max_repeat_ratio: float,
    max_bootstrap_ratio: float,
    max_start_only_ratio: float,
) -> dict[str, Any]:
    observed_cases = _scale_cases(
        load_observed_cases(component_root, variant_dir=variant_dir, fan_scale=1.0),
        alpha_map,
    )
    start_only_cases = _scale_cases(
        _load_start_only_cases(control_root, fan_scale=1.0),
        alpha_map,
    )
    repeat_cases = _scale_cases(
        _load_repeat_cases(control_root, fan_scale=1.0),
        alpha_map,
    )
    observed = _pairwise(observed_cases, "observed_narrative")
    bootstrap = _bootstrap_pairs(observed_cases)
    start_only = _pairwise(start_only_cases, "start_only_null")
    repeat = _pairwise(repeat_cases, "same_narrative_repeat")
    summaries = {
        "observed_narrative": _summarize_pair_rows(observed),
        "within_run_bootstrap": _summarize_pair_rows(bootstrap),
        "start_only_null": _summarize_pair_rows(start_only),
        "same_narrative_repeat": _summarize_pair_rows(repeat),
    }
    ratios = {
        "repeat_to_observed_path_wasserstein": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            "path_wasserstein_z_mean_median",
        ),
        "bootstrap_to_observed_path_wasserstein": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            "path_wasserstein_z_mean_median",
        ),
        "start_only_to_observed_path_wasserstein": _ratio(
            summaries["start_only_null"],
            summaries["observed_narrative"],
            "path_wasserstein_z_mean_median",
        ),
        "repeat_to_observed_path_variance": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            "path_std_log_ratio_mean_median",
        ),
        "bootstrap_to_observed_path_variance": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            "path_std_log_ratio_mean_median",
        ),
        "start_only_to_observed_path_variance": _ratio(
            summaries["start_only_null"],
            summaries["observed_narrative"],
            "path_std_log_ratio_mean_median",
        ),
        "repeat_to_observed_path_energy": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            "path_energy_distance",
        ),
        "bootstrap_to_observed_path_energy": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            "path_energy_distance",
        ),
        "start_only_to_observed_path_energy": _ratio(
            summaries["start_only_null"],
            summaries["observed_narrative"],
            "path_energy_distance",
        ),
    }
    fixed_start_failures = []
    path_failures = []
    path_warnings = []
    max_start_diff = _max_start_difference(observed_cases)
    if max_start_diff > float(max_start_abs_diff):
        fixed_start_failures.append("fixed_start_not_identical")
    for key, failure_name in [
        ("repeat_to_observed_path_wasserstein", "repeat_path_too_close_to_observed"),
        ("repeat_to_observed_path_variance", "repeat_variance_too_close_to_observed"),
        ("repeat_to_observed_path_energy", "repeat_energy_too_close_to_observed"),
    ]:
        if ratios[key] is not None and ratios[key] > float(max_repeat_ratio):
            path_failures.append(failure_name)
    for key, failure_name in [
        (
            "start_only_to_observed_path_wasserstein",
            "start_only_path_too_close_to_observed",
        ),
        (
            "start_only_to_observed_path_variance",
            "start_only_variance_too_close_to_observed",
        ),
        (
            "start_only_to_observed_path_energy",
            "start_only_energy_too_close_to_observed",
        ),
    ]:
        if ratios[key] is not None and ratios[key] > float(max_start_only_ratio):
            path_failures.append(failure_name)
    for key, warning_name in [
        (
            "bootstrap_to_observed_path_wasserstein",
            "bootstrap_path_noise_close_to_observed",
        ),
        (
            "bootstrap_to_observed_path_variance",
            "bootstrap_variance_noise_close_to_observed",
        ),
        (
            "bootstrap_to_observed_path_energy",
            "bootstrap_energy_noise_close_to_observed",
        ),
    ]:
        if ratios[key] is not None and ratios[key] > float(max_bootstrap_ratio):
            path_warnings.append(warning_name)
    status = (
        "fail"
        if fixed_start_failures or path_failures
        else "warning" if path_warnings else "pass"
    )
    plot_raw_factor_fans(
        observed_cases,
        output_path=output_dir / "rank_preserving_readout_raw_fans.png",
    )
    plot_shape_metric_summary(
        summaries,
        output_path=output_dir / "rank_preserving_readout_shape_metrics.png",
    )
    plot_path_metric_summary(
        summaries,
        output_path=output_dir / "rank_preserving_readout_path_metrics.png",
    )
    return {
        "status": status,
        "max_observed_start_abs_diff": float(max_start_diff),
        "summaries": summaries,
        "ratios": ratios,
        "warnings": fixed_start_failures + path_warnings,
        "failures": fixed_start_failures + path_failures,
        "pair_counts": {
            "observed": int(len(observed)),
            "bootstrap": int(len(bootstrap)),
            "repeat": int(len(repeat)),
            "start_only": int(len(start_only)),
        },
        "artifact_paths": {
            "raw_fan_plot": str(output_dir / "rank_preserving_readout_raw_fans.png"),
            "shape_metric_plot": str(
                output_dir / "rank_preserving_readout_shape_metrics.png"
            ),
            "path_metric_plot": str(
                output_dir / "rank_preserving_readout_path_metrics.png"
            ),
        },
    }


def run_rank_preserving_readout(args: argparse.Namespace) -> dict[str, Any]:
    backtest = _load_json(args.backtest_report)
    rows = backtest.get("window_scores", [])
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("backtest report needs at least four scored rows")
    device = torch.device(
        str(args.device)
        if torch.cuda.is_available() or str(args.device) == "cpu"
        else "cpu"
    )
    block = _selected_history_block(
        checkpoint=str(backtest.get("checkpoint") or args.checkpoint),
        bridge_report=str(backtest.get("bridge_report") or args.bridge_report),
        device=device,
    )
    history_raw = np.asarray(block["history_raw"], dtype=np.float32)
    future_raw = np.asarray(block["future_raw"], dtype=np.float32)
    calibration_count = int(args.calibration_count)
    if calibration_count <= 0:
        calibration_count = max(1, len(rows) // 2)
    calibration_rows, evaluation_rows = split_calibration_rows(
        rows,
        calibration_count=calibration_count,
        split_mode=str(args.split_mode),
    )
    fit = fit_rank_preserving_alpha_map(
        calibration_rows,
        alpha_grid=_parse_grid(str(args.alpha_grid)),
        history_raw=history_raw,
        future_raw=future_raw,
        target_coverage=float(args.target_coverage),
    )
    alpha_map = np.asarray(fit["alpha_map"], dtype=np.float32)
    eval_scores = _score_rows_with_alpha_map(
        evaluation_rows,
        alpha_map=alpha_map,
        history_raw=history_raw,
        future_raw=future_raw,
    )
    eval_summary = summarize_method_scores(eval_scores, baseline="persistence")
    full_scores = _score_rows_with_alpha_map(
        rows,
        alpha_map=alpha_map,
        history_raw=history_raw,
        future_raw=future_raw,
    )
    full_summary = summarize_method_scores(full_scores, baseline="persistence")
    fixed_start = _fixed_start_report(
        alpha_map=alpha_map,
        component_root=Path(args.component_root),
        control_root=Path(args.control_root),
        variant_dir=str(args.variant_dir),
        output_dir=Path(args.output_dir),
        max_start_abs_diff=float(args.max_start_abs_diff),
        max_repeat_ratio=float(args.max_repeat_ratio),
        max_bootstrap_ratio=float(args.max_bootstrap_ratio),
        max_start_only_ratio=float(args.max_start_only_ratio),
    )
    eval_component = eval_summary["component_prefix_mixture"]
    eval_readout = eval_summary["component_rank_preserving_readout"]
    comparison = {
        "eval_readout_minus_component_coverage_80": float(
            eval_readout.get("coverage_80_mean", 0.0)
            - eval_component.get("coverage_80_mean", 0.0)
        ),
        "eval_readout_minus_component_crps": float(
            eval_readout.get("ensemble_crps_z_mean", np.nan)
            - eval_component.get("ensemble_crps_z_mean", np.nan)
        ),
        "eval_readout_minus_component_energy": float(
            eval_readout.get("energy_score_z_mean", np.nan)
            - eval_component.get("energy_score_z_mean", np.nan)
        ),
    }
    status = (
        "pass"
        if fixed_start["status"] == "pass"
        and comparison["eval_readout_minus_component_coverage_80"] >= 0.0
        else "warning"
    )
    output_dir = Path(args.output_dir)
    report = {
        "status": status,
        "research_lane": "experiment",
        "result_status": "candidate" if status == "pass" else "mechanism_found",
        "benchmark_floor_status": "not_applicable",
        "scope_note": (
            "Rank-preserving marginal readout TestFlight. A per-horizon/factor "
            "positive alpha map is fit on calibration backtest rows and applied "
            "without changing per-variable sample ranks. This is a readout "
            "diagnostic, not a new text bridge or generator."
        ),
        "backtest_report": str(args.backtest_report),
        "component_root": str(args.component_root),
        "control_root": str(args.control_root),
        "variant_dir": str(args.variant_dir),
        "split_mode": str(args.split_mode),
        "calibration_window_count": int(len(calibration_rows)),
        "evaluation_window_count": int(len(evaluation_rows)),
        "target_coverage": float(args.target_coverage),
        "rank_preserving_contract": {
            "positive_alpha_map": bool(np.all(alpha_map > 0.0)),
            "per_variable_rank_preserved": True,
            "path_dependence_template": "sample ranks are inherited from the frozen-generator rollout",
        },
        "alpha_summary": _alpha_summary(alpha_map, np.asarray(fit["chosen_coverage"])),
        "evaluation_summary": eval_summary,
        "full_summary_at_selected_alpha_map": full_summary,
        "comparison": comparison,
        "fixed_start_conditionality": fixed_start,
        "interpretation": [
            "A pass would mean the readout improved/held calibration while keeping fixed-start narrative effects above controls.",
            "A warning with better coverage but failed fixed-start conditionality means marginal calibration is still washing out narrative differences.",
            "Because this fits many marginal alphas from few rows, treat this as a mechanism probe, not a promoted production default.",
        ],
        "artifact_paths": {
            "report": str(output_dir / "rank_preserving_readout_report.json"),
            "alpha_map": str(output_dir / "rank_preserving_alpha_map.npz"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "rank_preserving_alpha_map.npz",
        alpha_map=alpha_map.astype(np.float32),
        chosen_coverage=np.asarray(fit["chosen_coverage"], dtype=np.float32),
    )
    _write_json(output_dir / "rank_preserving_readout_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-report", default=DEFAULT_BACKTEST_REPORT)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--component-root", default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--control-root", default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-grid", default="1.0,1.25,1.5,1.75,2.0,2.5,3.0")
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--calibration-count", type=int, default=15)
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "reverse", "even_odd", "odd_even"],
        default="chronological",
    )
    parser.add_argument("--max-start-abs-diff", type=float, default=1e-5)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.75)
    parser.add_argument("--max-bootstrap-ratio", type=float, default=0.75)
    parser.add_argument("--max-start-only-ratio", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_rank_preserving_readout(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison": report["comparison"],
                "fixed_start_status": report["fixed_start_conditionality"]["status"],
                "alpha_summary": report["alpha_summary"],
                "report": report["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
