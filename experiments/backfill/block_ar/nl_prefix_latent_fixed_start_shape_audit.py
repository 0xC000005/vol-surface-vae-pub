#!/usr/bin/env python
"""Audit fixed-start narrative effects on full sampled path distributions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    scale_delta_samples_around_mean,
)
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (  # noqa: E402
    CONTRAST_CASES,
    MARKET_INDEX,
)


DEFAULT_COMPONENT_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_mixture_fixed_start_900a_s96"
)
DEFAULT_CONTROL_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_fixed_start_controls_901a_s96"
)
DEFAULT_VARIANT_DIR = "decoder_component_topk_narrative_start_checked_gen_temp_0p50"
AUDIT_MARKETS = [
    ("SPX", MARKET_INDEX["SPX"]),
    ("VIX", MARKET_INDEX["VIX"]),
    ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
    ("Crude oil", 31),
    ("US10Y", 33),
    ("BBB OAS", 35),
    ("Gold", 37),
    ("DXY", 28),
]
PLOT_MARKETS = [
    ("SPX", MARKET_INDEX["SPX"]),
    ("VIX", MARKET_INDEX["VIX"]),
    ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
    ("Crude oil", 31),
]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _operational_variant_index(report: dict[str, Any], variant_count: int) -> int:
    selected = report.get("selected_start_state", {})
    if isinstance(selected, dict) and selected.get("variant_index") is not None:
        idx = int(selected["variant_index"])
        return idx if 0 <= idx < int(variant_count) else 0
    for idx, row in enumerate(report.get("variant_rows", [])):
        if (
            idx < int(variant_count)
            and isinstance(row, dict)
            and bool(row.get("is_operational"))
        ):
            return int(idx)
    return 0


def _case_dir(component_root: Path, case_name: str, variant_dir: str) -> Path:
    return component_root / case_name / f"fixed_start_{_case_start_index(case_name)}" / variant_dir


def _case_start_index(case_name: str) -> int:
    match = re.search(r"_start(\d+)(?:#.*)?$", str(case_name))
    return int(match.group(1)) if match else 18


def _case_specs_for_component_root(
    component_root: Path,
    variant_dir: str,
) -> list[tuple[str, str, str]]:
    if all(
        (_case_dir(component_root, case_name, variant_dir) / "prefix_latent_story_smoke_report.json").exists()
        for _label, case_name, _color in CONTRAST_CASES
    ):
        return list(CONTRAST_CASES)
    rows: list[tuple[str, str, str]] = []
    for case_root in sorted(Path(component_root).iterdir()):
        if not case_root.is_dir():
            continue
        case_name = case_root.name
        case_dir = _case_dir(component_root, case_name, variant_dir)
        if (case_dir / "prefix_latent_story_smoke_report.json").exists():
            label = re.sub(r"_start\d+$", "", case_name).replace("_", " ")
            rows.append((label, case_name, "#666666"))
    if not rows:
        raise FileNotFoundError(f"no observed fixed-start cases found under {component_root}")
    return rows


def _load_case_from_dir(
    *,
    case_name: str,
    label: str,
    color: str,
    case_dir: Path,
    fan_scale: float,
) -> dict[str, Any]:
    report = _load_json(case_dir / "prefix_latent_story_smoke_report.json")
    arrays = np.load(case_dir / "prefix_latent_story_smoke_arrays.npz")
    generated = np.asarray(arrays["generated_states"], dtype=np.float32)
    requested_raw = np.asarray(arrays["requested_raw"], dtype=np.float32)
    variant_idx = _operational_variant_index(report, generated.shape[0])
    states = np.asarray(generated[variant_idx], dtype=np.float32)
    start = np.asarray(requested_raw[variant_idx], dtype=np.float32)
    if abs(float(fan_scale) - 1.0) > 1e-8:
        deltas = (states - start[None, None, :]).astype(np.float32)
        scaled = scale_delta_samples_around_mean(deltas[None, ...], float(fan_scale))[0]
        states = (start[None, None, :] + scaled).astype(np.float32)
    return {
        "case_name": str(case_name),
        "label": str(label),
        "color": str(color),
        "states": states,
        "start": start,
        "sample_count": int(states.shape[0]),
        "run_report": str(case_dir / "prefix_latent_story_smoke_report.json"),
    }


def load_observed_cases(
    component_root: Path,
    *,
    variant_dir: str,
    fan_scale: float,
) -> list[dict[str, Any]]:
    return [
        _load_case_from_dir(
            case_name=case_name,
            label=label,
            color=color,
            case_dir=_case_dir(component_root, case_name, variant_dir),
            fan_scale=fan_scale,
        )
        for label, case_name, color in _case_specs_for_component_root(
            component_root,
            variant_dir,
        )
    ]


def _load_start_only_cases(
    control_root: Path,
    *,
    fan_scale: float,
) -> list[dict[str, Any]]:
    rows = []
    start_only_root = control_root / "start_only_controls"
    if not start_only_root.exists():
        return rows
    for case_root in sorted(start_only_root.iterdir()):
        if not case_root.is_dir():
            continue
        case_name = case_root.name
        label = re.sub(r"_start\d+$", "", case_name).replace("_", " ")
        color = "#666666"
        case_dir = control_root / "start_only_controls" / case_name
        if case_dir.exists():
            rows.append(
                _load_case_from_dir(
                    case_name=case_name,
                    label=label,
                    color=color,
                    case_dir=case_dir,
                    fan_scale=fan_scale,
                )
            )
    return rows


def _load_repeat_cases(
    control_root: Path,
    *,
    fan_scale: float,
) -> list[dict[str, Any]]:
    rows = []
    repeat_root = control_root / "repeat_controls"
    if not repeat_root.exists():
        return rows
    for case_root in sorted(repeat_root.glob("*_start*")):
        for seed_root in sorted(case_root.glob("seed_*")):
            case_name = f"{case_root.name}#{seed_root.name}"
            rows.append(
                _load_case_from_dir(
                    case_name=case_name,
                    label=case_name,
                    color="#444444",
                    case_dir=seed_root,
                    fan_scale=fan_scale,
                )
            )
    return rows


def standardize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    center = float(np.median(arr))
    q75, q25 = np.quantile(arr, [0.75, 0.25])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(arr))
    if scale <= 1e-12:
        scale = 1.0
    return ((arr - center) / scale).astype(np.float64)


def empirical_ks(left: np.ndarray, right: np.ndarray) -> float:
    a = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    if a.size == 0 or b.size == 0:
        return 0.0
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, values, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def empirical_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    count = max(int(a.size), int(b.size), 100)
    q = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def quantile_shape_l2(left: np.ndarray, right: np.ndarray) -> float:
    q = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    a = np.quantile(standardize_1d(left), q)
    b = np.quantile(standardize_1d(right), q)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def terminal_market_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    start_level: float,
) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    std_a = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    std_b = float(np.std(b, ddof=1)) if b.size > 1 else 0.0
    pooled_std = float(np.sqrt((std_a**2 + std_b**2) / 2.0))
    if pooled_std <= 1e-12:
        pooled_std = 1.0
    down_a = float(np.mean(a < float(start_level)))
    down_b = float(np.mean(b < float(start_level)))
    return {
        "mean_gap_z": float(abs(float(np.mean(a)) - float(np.mean(b))) / pooled_std),
        "std_log_ratio_abs": float(abs(np.log((std_a + 1e-12) / (std_b + 1e-12)))),
        "raw_ks": empirical_ks(a, b),
        "standardized_ks": empirical_ks(standardize_1d(a), standardize_1d(b)),
        "wasserstein_z": empirical_wasserstein(a, b) / pooled_std,
        "quantile_shape_l2": quantile_shape_l2(a, b),
        "down_probability_abs_gap": float(abs(down_a - down_b)),
    }


def _path_with_start(case: dict[str, Any], market_idx: int) -> np.ndarray:
    states = np.asarray(case["states"], dtype=np.float64)[:, :, market_idx]
    start = float(np.asarray(case["start"])[market_idx])
    return np.concatenate(
        [np.full((states.shape[0], 1), start, dtype=np.float64), states],
        axis=1,
    )


def _pooled_horizon_std(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_std = np.std(left, axis=0, ddof=1)
    right_std = np.std(right, axis=0, ddof=1)
    pooled = np.sqrt((left_std**2 + right_std**2) / 2.0)
    finite = pooled[np.isfinite(pooled) & (pooled > 1e-12)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    pooled = np.where(pooled > 1e-12, pooled, fallback)
    return pooled.astype(np.float64)


def horizonwise_path_metrics(
    left_path: np.ndarray,
    right_path: np.ndarray,
) -> dict[str, float]:
    left = np.asarray(left_path, dtype=np.float64)
    right = np.asarray(right_path, dtype=np.float64)
    scale = _pooled_horizon_std(left, right)
    mean_gap_z = np.abs(np.mean(left, axis=0) - np.mean(right, axis=0)) / scale
    std_left = np.std(left, axis=0, ddof=1)
    std_right = np.std(right, axis=0, ddof=1)
    std_log_ratio = np.abs(np.log((std_left + 1e-12) / (std_right + 1e-12)))
    raw_ks = np.asarray(
        [empirical_ks(left[:, step], right[:, step]) for step in range(left.shape[1])],
        dtype=np.float64,
    )
    wasserstein_z = np.asarray(
        [
            empirical_wasserstein(left[:, step], right[:, step]) / scale[step]
            for step in range(left.shape[1])
        ],
        dtype=np.float64,
    )
    return {
        "path_mean_gap_z_mean": float(np.mean(mean_gap_z[1:])),
        "path_mean_gap_z_p90": float(np.quantile(mean_gap_z[1:], 0.90)),
        "path_mean_gap_z_max": float(np.max(mean_gap_z[1:])),
        "path_std_log_ratio_mean": float(np.mean(std_log_ratio[1:])),
        "path_std_log_ratio_p90": float(np.quantile(std_log_ratio[1:], 0.90)),
        "path_std_log_ratio_max": float(np.max(std_log_ratio[1:])),
        "path_raw_ks_mean": float(np.mean(raw_ks[1:])),
        "path_raw_ks_p90": float(np.quantile(raw_ks[1:], 0.90)),
        "path_raw_ks_max": float(np.max(raw_ks[1:])),
        "path_wasserstein_z_mean": float(np.mean(wasserstein_z[1:])),
        "path_wasserstein_z_p90": float(np.quantile(wasserstein_z[1:], 0.90)),
        "path_wasserstein_z_max": float(np.max(wasserstein_z[1:])),
    }


def _sample_drawdown(path: np.ndarray) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float64)
    running_peak = np.maximum.accumulate(arr, axis=1)
    return np.min(arr - running_peak, axis=1)


def _sample_rally(path: np.ndarray) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float64)
    running_trough = np.minimum.accumulate(arr, axis=1)
    return np.max(arr - running_trough, axis=1)


def path_event_metrics(
    left_path: np.ndarray,
    right_path: np.ndarray,
) -> dict[str, float]:
    left = np.asarray(left_path, dtype=np.float64)
    right = np.asarray(right_path, dtype=np.float64)
    terminal_scale = float(np.sqrt((np.var(left[:, -1]) + np.var(right[:, -1])) / 2.0))
    if terminal_scale <= 1e-12 or not np.isfinite(terminal_scale):
        terminal_scale = 1.0
    left_drawdown = _sample_drawdown(left) / terminal_scale
    right_drawdown = _sample_drawdown(right) / terminal_scale
    left_rally = _sample_rally(left) / terminal_scale
    right_rally = _sample_rally(right) / terminal_scale
    left_terminal = (left[:, -1] - left[:, 0]) / terminal_scale
    right_terminal = (right[:, -1] - right[:, 0]) / terminal_scale
    return {
        "path_drawdown_prob_gap_1sigma": float(
            abs(np.mean(left_drawdown <= -1.0) - np.mean(right_drawdown <= -1.0))
        ),
        "path_rally_prob_gap_1sigma": float(
            abs(np.mean(left_rally >= 1.0) - np.mean(right_rally >= 1.0))
        ),
        "terminal_down_prob_gap_1sigma": float(
            abs(np.mean(left_terminal <= -1.0) - np.mean(right_terminal <= -1.0))
        ),
        "terminal_up_prob_gap_1sigma": float(
            abs(np.mean(left_terminal >= 1.0) - np.mean(right_terminal >= 1.0))
        ),
        "drawdown_wasserstein_z": empirical_wasserstein(left_drawdown, right_drawdown),
        "rally_wasserstein_z": empirical_wasserstein(left_rally, right_rally),
    }


def _flatten_case_paths(case: dict[str, Any]) -> np.ndarray:
    states = np.asarray(case["states"], dtype=np.float64)
    start = np.asarray(case["start"], dtype=np.float64)
    market_indices = [idx for _market, idx in AUDIT_MARKETS]
    path = np.concatenate(
        [
            np.broadcast_to(start[None, None, :], (states.shape[0], 1, start.shape[0])),
            states,
        ],
        axis=1,
    )
    deltas = path[:, 1:, market_indices] - path[:, :1, market_indices]
    return deltas.reshape(deltas.shape[0], -1)


def _standardize_pair_features(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([left, right], axis=0)
    scale = np.std(stacked, axis=0, ddof=1)
    finite = scale[np.isfinite(scale) & (scale > 1e-12)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    scale = np.where(scale > 1e-12, scale, fallback)
    center = np.mean(stacked, axis=0)
    return (left - center) / scale, (right - center) / scale


def _mean_pairwise_distance(left: np.ndarray, right: np.ndarray) -> float:
    diff = left[:, None, :] - right[None, :, :]
    return float(np.mean(np.linalg.norm(diff, axis=2)))


def path_energy_distance(
    left_case: dict[str, Any], right_case: dict[str, Any]
) -> float:
    left, right = _standardize_pair_features(
        _flatten_case_paths(left_case),
        _flatten_case_paths(right_case),
    )
    cross = _mean_pairwise_distance(left, right)
    left_self = _mean_pairwise_distance(left, left)
    right_self = _mean_pairwise_distance(right, right)
    dim_scale = float(np.sqrt(max(left.shape[1], 1)))
    return float(max(0.0, 2.0 * cross - left_self - right_self) / dim_scale)


def path_distribution_market_metrics(
    left_case: dict[str, Any],
    right_case: dict[str, Any],
) -> list[dict[str, float | str]]:
    market_rows = []
    for market, idx in AUDIT_MARKETS:
        left_path = _path_with_start(left_case, idx)
        right_path = _path_with_start(right_case, idx)
        market_rows.append(
            {
                "market": market,
                **horizonwise_path_metrics(left_path, right_path),
                **path_event_metrics(left_path, right_path),
            }
        )
    return market_rows


def pair_distribution_metrics(
    left_case: dict[str, Any],
    right_case: dict[str, Any],
) -> dict[str, Any]:
    left_states = np.asarray(left_case["states"], dtype=np.float32)
    right_states = np.asarray(right_case["states"], dtype=np.float32)
    left_terminal = left_states[:, -1, :]
    right_terminal = right_states[:, -1, :]
    market_rows = []
    for market, idx in AUDIT_MARKETS:
        metrics = terminal_market_metrics(
            left_terminal[:, idx],
            right_terminal[:, idx],
            start_level=float(np.asarray(left_case["start"])[idx]),
        )
        market_rows.append({"market": market, **metrics})
    path_market_rows = path_distribution_market_metrics(left_case, right_case)
    aggregate: dict[str, float] = {}
    for key in [
        "mean_gap_z",
        "std_log_ratio_abs",
        "raw_ks",
        "standardized_ks",
        "wasserstein_z",
        "quantile_shape_l2",
        "down_probability_abs_gap",
    ]:
        aggregate[f"{key}_median"] = float(
            median([float(row[key]) for row in market_rows])
        )
        aggregate[f"{key}_mean"] = float(
            np.mean([float(row[key]) for row in market_rows])
        )
    for key in [
        "path_mean_gap_z_mean",
        "path_mean_gap_z_p90",
        "path_std_log_ratio_mean",
        "path_std_log_ratio_p90",
        "path_raw_ks_mean",
        "path_raw_ks_p90",
        "path_wasserstein_z_mean",
        "path_wasserstein_z_p90",
        "path_drawdown_prob_gap_1sigma",
        "path_rally_prob_gap_1sigma",
        "terminal_down_prob_gap_1sigma",
        "terminal_up_prob_gap_1sigma",
        "drawdown_wasserstein_z",
        "rally_wasserstein_z",
    ]:
        aggregate[f"{key}_median"] = float(
            median([float(row[key]) for row in path_market_rows])
        )
        aggregate[f"{key}_mean"] = float(
            np.mean([float(row[key]) for row in path_market_rows])
        )
    aggregate["path_energy_distance"] = path_energy_distance(left_case, right_case)
    return {
        "left_case": str(left_case["case_name"]),
        "right_case": str(right_case["case_name"]),
        "left_label": str(left_case["label"]),
        "right_label": str(right_case["label"]),
        "sample_count_left": int(left_states.shape[0]),
        "sample_count_right": int(right_states.shape[0]),
        "aggregate": aggregate,
        "market_metrics": sorted(
            market_rows,
            key=lambda row: (
                float(row["standardized_ks"]),
                float(row["quantile_shape_l2"]),
            ),
            reverse=True,
        ),
        "path_market_metrics": sorted(
            path_market_rows,
            key=lambda row: (
                float(row["path_wasserstein_z_mean"]),
                float(row["path_std_log_ratio_mean"]),
            ),
            reverse=True,
        ),
    }


def _pairwise(cases: list[dict[str, Any]], control: str) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(cases, 2):
        row = pair_distribution_metrics(left, right)
        row["control"] = control
        rows.append(row)
    return rows


def _bootstrap_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        states = np.asarray(case["states"], dtype=np.float32)
        split = states.shape[0] // 2
        if split < 2:
            continue
        left = {
            **case,
            "case_name": f"{case['case_name']}#left",
            "states": states[:split],
        }
        right = {
            **case,
            "case_name": f"{case['case_name']}#right",
            "states": states[split:],
        }
        row = pair_distribution_metrics(left, right)
        row["control"] = "within_run_bootstrap"
        rows.append(row)
    return rows


def _repeat_pairs(repeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in repeats:
        grouped.setdefault(str(item["case_name"]).split("#seed_")[0], []).append(item)
    rows = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        rows.extend(_pairwise(group, "same_narrative_repeat"))
    return rows


def _summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pair_count": 0}
    keys = list(rows[0]["aggregate"].keys())
    summary: dict[str, Any] = {"pair_count": int(len(rows))}
    for key in keys:
        values = [float(row["aggregate"][key]) for row in rows]
        summary[f"{key}_median_across_pairs"] = float(median(values))
        summary[f"{key}_p90_across_pairs"] = float(np.quantile(values, 0.90))
        summary[f"{key}_max_across_pairs"] = float(max(values))
    return summary


def _ratio(
    numerator: dict[str, Any], denominator: dict[str, Any], key: str
) -> float | None:
    top = numerator.get(f"{key}_median_across_pairs")
    bottom = denominator.get(f"{key}_median_across_pairs")
    if top is None or bottom is None or abs(float(bottom)) <= 1e-12:
        return None
    return float(top) / float(bottom)


def _max_start_difference(cases: list[dict[str, Any]]) -> float:
    if not cases:
        return 0.0
    base = np.asarray(cases[0]["start"], dtype=np.float64)
    return float(
        max(
            np.max(np.abs(np.asarray(case["start"], dtype=np.float64) - base))
            for case in cases
        )
    )


def plot_raw_factor_fans(
    cases: list[dict[str, Any]],
    *,
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.reshape(-1)
    days = np.arange(31)
    for ax, (market, idx) in zip(axes, PLOT_MARKETS, strict=True):
        for case in cases:
            states = np.asarray(case["states"], dtype=np.float64)[:, :, idx]
            start = float(np.asarray(case["start"])[idx])
            path = np.concatenate(
                [np.full((states.shape[0], 1), start, dtype=np.float64), states],
                axis=1,
            )
            p10, p50, p90 = np.quantile(path, [0.10, 0.50, 0.90], axis=0)
            color = str(case["color"])
            ax.fill_between(days, p10, p90, color=color, alpha=0.08, linewidth=0)
            ax.plot(days, p50, color=color, linewidth=1.6, label=str(case["label"]))
        ax.set_title(market)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Forecast day")
        ax.set_ylabel("Raw level")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Fixed-start narrative distribution fans: raw levels")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_shape_metric_summary(
    summaries: dict[str, dict[str, Any]],
    *,
    output_path: str | Path,
) -> None:
    controls = [
        "observed_narrative",
        "same_narrative_repeat",
        "within_run_bootstrap",
        "start_only_null",
    ]
    labels = ["Observed", "Repeat", "Bootstrap", "Start-only"]
    metrics = [
        ("standardized_ks_median_median_across_pairs", "Shape KS"),
        ("quantile_shape_l2_median_median_across_pairs", "Quantile shape L2"),
        ("mean_gap_z_median_median_across_pairs", "Mean gap z"),
        ("std_log_ratio_abs_median_median_across_pairs", "Width log-ratio"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.2), sharey=False)
    for ax, (key, title) in zip(axes, metrics, strict=True):
        values = [
            float(summaries.get(control, {}).get(key, 0.0)) for control in controls
        ]
        ax.bar(labels, values, color=["#1565C0", "#757575", "#9E9E9E", "#BDBDBD"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed-start narrative effect versus controls")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_path_metric_summary(
    summaries: dict[str, dict[str, Any]],
    *,
    output_path: str | Path,
) -> None:
    controls = [
        "observed_narrative",
        "same_narrative_repeat",
        "within_run_bootstrap",
        "start_only_null",
    ]
    labels = ["Observed", "Repeat", "Bootstrap", "Start-only"]
    metrics = [
        (
            "path_wasserstein_z_mean_median_median_across_pairs",
            "Path Wasserstein",
        ),
        (
            "path_std_log_ratio_mean_median_median_across_pairs",
            "Horizon variance",
        ),
        (
            "path_drawdown_prob_gap_1sigma_mean_median_across_pairs",
            "Drawdown prob",
        ),
        (
            "path_energy_distance_median_across_pairs",
            "Path energy",
        ),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.2), sharey=False)
    for ax, (key, title) in zip(axes, metrics, strict=True):
        values = [
            float(summaries.get(control, {}).get(key, 0.0)) for control in controls
        ]
        ax.bar(labels, values, color=["#1565C0", "#757575", "#9E9E9E", "#BDBDBD"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed-start full path-distribution effect versus controls")
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_shape_audit(args: argparse.Namespace) -> dict[str, Any]:
    observed_cases = load_observed_cases(
        Path(args.component_root),
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    start_only_cases = _load_start_only_cases(
        Path(args.control_root),
        fan_scale=float(args.fan_scale),
    )
    repeat_cases = _load_repeat_cases(
        Path(args.control_root),
        fan_scale=float(args.fan_scale),
    )
    observed = _pairwise(observed_cases, "observed_narrative")
    bootstrap = _bootstrap_pairs(observed_cases)
    start_only = _pairwise(start_only_cases, "start_only_null")
    repeat = _repeat_pairs(repeat_cases)
    summaries = {
        "observed_narrative": _summarize_pair_rows(observed),
        "within_run_bootstrap": _summarize_pair_rows(bootstrap),
        "start_only_null": _summarize_pair_rows(start_only),
        "same_narrative_repeat": _summarize_pair_rows(repeat),
    }
    shape_key = "standardized_ks_median"
    quantile_key = "quantile_shape_l2_median"
    ratios = {
        "repeat_to_observed_shape_ks": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            shape_key,
        ),
        "bootstrap_to_observed_shape_ks": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            shape_key,
        ),
        "start_only_to_observed_shape_ks": _ratio(
            summaries["start_only_null"], summaries["observed_narrative"], shape_key
        ),
        "repeat_to_observed_quantile_shape": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            quantile_key,
        ),
        "bootstrap_to_observed_quantile_shape": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            quantile_key,
        ),
        "start_only_to_observed_quantile_shape": _ratio(
            summaries["start_only_null"], summaries["observed_narrative"], quantile_key
        ),
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
    fixed_start_failures: list[str] = []
    shape_failures: list[str] = []
    shape_warnings: list[str] = []
    path_failures: list[str] = []
    path_warnings: list[str] = []
    max_start_diff = _max_start_difference(observed_cases)
    if max_start_diff > float(args.max_start_abs_diff):
        fixed_start_failures.append("fixed_start_not_identical")
    if ratios["repeat_to_observed_shape_ks"] is not None and ratios[
        "repeat_to_observed_shape_ks"
    ] > float(args.max_repeat_ratio):
        shape_failures.append("repeat_shape_too_close_to_observed")
    if ratios["start_only_to_observed_shape_ks"] is not None and ratios[
        "start_only_to_observed_shape_ks"
    ] > float(args.max_start_only_ratio):
        shape_failures.append("start_only_shape_too_close_to_observed")
    if ratios["bootstrap_to_observed_shape_ks"] is not None and ratios[
        "bootstrap_to_observed_shape_ks"
    ] > float(args.max_bootstrap_ratio):
        shape_warnings.append("bootstrap_shape_noise_close_to_observed")
    for key, failure_name in [
        ("repeat_to_observed_path_wasserstein", "repeat_path_too_close_to_observed"),
        ("repeat_to_observed_path_variance", "repeat_variance_too_close_to_observed"),
        ("repeat_to_observed_path_energy", "repeat_energy_too_close_to_observed"),
    ]:
        if ratios[key] is not None and ratios[key] > float(args.max_repeat_ratio):
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
        if ratios[key] is not None and ratios[key] > float(args.max_start_only_ratio):
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
        if ratios[key] is not None and ratios[key] > float(args.max_bootstrap_ratio):
            path_warnings.append(warning_name)
    path_distribution_status = (
        "fail"
        if fixed_start_failures or path_failures
        else "warning" if path_warnings else "pass"
    )
    terminal_shape_only_status = (
        "fail" if shape_failures else "warning" if shape_warnings else "pass"
    )
    status = path_distribution_status
    output_dir = Path(args.output_dir)
    fan_plot = output_dir / "fixed_start_narrative_shape_raw_fans.png"
    metric_plot = output_dir / "fixed_start_narrative_shape_metric_summary.png"
    path_metric_plot = output_dir / "fixed_start_narrative_path_metric_summary.png"
    plot_raw_factor_fans(observed_cases, output_path=fan_plot)
    plot_shape_metric_summary(summaries, output_path=metric_plot)
    plot_path_metric_summary(summaries, output_path=path_metric_plot)
    return {
        "status": status,
        "scope_note": (
            "Fixed-start path-distribution audit. The start level is held fixed; "
            "observed narrative pairs are compared against start-only, repeat, "
            "and within-run bootstrap controls. Terminal shape-only metrics remove "
            "each terminal distribution's own location/scale; path metrics evaluate "
            "the whole sampled 30-day path family."
        ),
        "component_root": str(args.component_root),
        "control_root": str(args.control_root),
        "variant_dir": str(args.variant_dir),
        "fan_scale": float(args.fan_scale),
        "markets": [market for market, _idx in AUDIT_MARKETS],
        "max_observed_start_abs_diff": max_start_diff,
        "thresholds": {
            "max_start_abs_diff": float(args.max_start_abs_diff),
            "max_repeat_ratio": float(args.max_repeat_ratio),
            "max_bootstrap_ratio": float(args.max_bootstrap_ratio),
            "max_start_only_ratio": float(args.max_start_only_ratio),
        },
        "summaries": summaries,
        "ratios": ratios,
        "path_distribution_status": path_distribution_status,
        "terminal_shape_only_status": terminal_shape_only_status,
        "observed_pairwise": observed,
        "bootstrap_pairwise": bootstrap,
        "start_only_pairwise": start_only,
        "repeat_pairwise": repeat,
        "warnings": fixed_start_failures + path_warnings,
        "failures": fixed_start_failures + path_failures,
        "path_distribution_warnings": path_warnings,
        "path_distribution_failures": fixed_start_failures + path_failures,
        "terminal_shape_only_warnings": shape_warnings,
        "terminal_shape_only_failures": shape_failures,
        "interpretation": [
            "Mean gaps answer whether narratives move the center of the distribution.",
            "Width log-ratios answer whether narratives mainly widen or narrow the fan.",
            "Standardized KS and quantile-shape L2 answer whether shape remains different after removing each narrative's own location and scale.",
            "Path Wasserstein, path energy, horizon-wise variance, drawdown, and rally metrics answer whether the sampled 30-day path family responds to the narrative.",
            "Terminal shape-only diagnostics are stricter diagnostics, not the product gate by themselves.",
            "The audit is evidence of narrative sensitivity, not realized-future accuracy.",
        ],
        "artifact_paths": {
            "report": str(output_dir / "fixed_start_shape_audit.json"),
            "raw_fan_plot": str(fan_plot),
            "metric_summary_plot": str(metric_plot),
            "path_metric_summary_plot": str(path_metric_plot),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", default=str(DEFAULT_COMPONENT_ROOT))
    parser.add_argument("--control-root", default=str(DEFAULT_CONTROL_ROOT))
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fan-scale", type=float, default=3.5)
    parser.add_argument("--max-start-abs-diff", type=float, default=1e-5)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.75)
    parser.add_argument("--max-bootstrap-ratio", type=float, default=0.75)
    parser.add_argument("--max-start-only-ratio", type=float, default=0.50)
    args = parser.parse_args()
    report = build_shape_audit(args)
    _write_json(report["artifact_paths"]["report"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "ratios": report["ratios"],
                "warnings": report["warnings"],
                "failures": report["failures"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
