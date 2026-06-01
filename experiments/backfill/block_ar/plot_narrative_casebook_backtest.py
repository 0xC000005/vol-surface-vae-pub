#!/usr/bin/env python
"""Plot raw-level narrative casebook and historical backtest diagnostics.

The figures are intentionally close to the main SNI paper fan-chart convention:
raw market levels, actual history when available, realized future, generated
bands, and selected path samples.  The current default uses the component-
preserving support-mixture rollout, not the older averaged-prefix path, because
averaging support components can visually erase narrative conditionality.  The
narrative casebook uses saved prefix-latent reports and arrays; the backtest
figure additionally samples the frozen SNI generator from the actual historical
prefix for the same fixed start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    component_slices_for_variant,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _reconstruct_states,
    _spec_names,
    sample_normal_generator_for_retrieved_analogues,
)
from experiments.backfill.block_ar.nl_prefix_latent_memory_decoder import (  # noqa: E402
    _future_raw_from_block,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    selected_bridge_window_indices,
)
from experiments.backfill.block_ar.nl_sparse_component_family_view import (  # noqa: E402
    select_sparse_components,
)


CONTROL_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "posterior_ensemble_candidate_966a_professional_start22_s384_d400"
)
DEFAULT_VARIANT_DIR = "cohesive_support_gap30"
DEFAULT_POSTERIOR_MODE = "top3_90"
CASEBOOK_FIGURE = Path(
    "paper/narrative_grounded_scenarios/figures/narrative_casebook_fixed_start.png"
)
BACKTEST_FIGURE = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_backtest_fixed_start_spx_vix.png"
)
SPX_CONDITIONALITY_FIGURE = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_spx_conditionality_fixed_start.png"
)
FACTOR_CONDITIONALITY_FIGURE = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_factor_conditionality_fixed_start.png"
)
QUALITATIVE_SUMMARY_JSON = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_qualitative_casebook_summary.json"
)
QUALITATIVE_SUMMARY_MD = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_qualitative_casebook_summary.md"
)
PORTFOLIO_IMPACT_FIGURE = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_portfolio_impact_fixed_start.png"
)
PORTFOLIO_IMPACT_JSON = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_portfolio_impact_summary.json"
)
PORTFOLIO_IMPACT_MD = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_portfolio_impact_summary.md"
)
BACKTEST_CACHE = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "narrative_backtest_fixed_start_direct_sni_cache.npz"
)


CASES = [
    (
        "Fragile risk-on rebound",
        "fragile_risk_on",
        "#1565C0",
    ),
    (
        "Commodity inflation pressure",
        "commodity_inflation",
        "#EF6C00",
    ),
    (
        "Dollar liquidity squeeze",
        "dollar_liquidity",
        "#6A1B9A",
    ),
]
CONTRAST_CASES = [
    (
        "Fragile risk-on",
        "fragile_risk_on",
        "#1565C0",
    ),
    (
        "Defensive risk-off",
        "defensive_risk_off",
        "#C62828",
    ),
    (
        "Commodity inflation",
        "commodity_inflation",
        "#EF6C00",
    ),
    (
        "Dollar liquidity",
        "dollar_liquidity",
        "#6A1B9A",
    ),
    (
        "Rates selloff",
        "rates_selloff",
        "#00838F",
    ),
    (
        "Safe-haven gold",
        "safe_haven_gold",
        "#2E7D32",
    ),
]
LEGACY_CASE_NAMES = {
    "fragile_risk_on": "fragile_risk_on_start18",
    "defensive_risk_off": "defensive_risk_off_start18",
    "commodity_inflation": "commodity_inflation_start18",
    "dollar_liquidity": "dollar_liquidity_start18",
    "rates_selloff": "rates_selloff_start18",
    "safe_haven_gold": "safe_haven_gold_start18",
}

MARKET_INDEX = {
    "SPX": 25,
    "VIX": 38,
    "IV_ATM_1Y": 17,
}
SHIFT_MARKETS = [
    ("SPX", MARKET_INDEX["SPX"]),
    ("VIX", MARKET_INDEX["VIX"]),
    ("BBB OAS", 35),
    ("US10Y", 33),
    ("DXY", 28),
    ("Gold", 37),
    ("Crude", 31),
    ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
]
PORTFOLIO_EXPOSURES = [
    {
        "market": "SPX",
        "index": MARKET_INDEX["SPX"],
        "sensitivity": 1.00,
        "description": "long equity beta",
    },
    {
        "market": "VIX",
        "index": MARKET_INDEX["VIX"],
        "sensitivity": -0.55,
        "description": "short volatility / convexity exposure",
    },
    {
        "market": "BBB OAS",
        "index": 35,
        "sensitivity": -0.45,
        "description": "long credit risk",
    },
    {
        "market": "US10Y",
        "index": 33,
        "sensitivity": -0.35,
        "description": "long-duration rate exposure",
    },
    {
        "market": "DXY",
        "index": 28,
        "sensitivity": -0.25,
        "description": "liquidity-sensitive short-dollar exposure",
    },
    {
        "market": "Crude",
        "index": 31,
        "sensitivity": 0.20,
        "description": "commodity reflation exposure",
    },
    {
        "market": "Gold",
        "index": 37,
        "sensitivity": 0.15,
        "description": "small safe-haven hedge",
    },
    {
        "market": "1Y ATM IV",
        "index": MARKET_INDEX["IV_ATM_1Y"],
        "sensitivity": -0.30,
        "description": "short implied-volatility carry",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_path(case_dir: Path) -> Path:
    return case_dir / "prefix_latent_story_smoke_report.json"


def _arrays_path(report: dict[str, Any], case_dir: Path) -> Path:
    raw = str(report.get("artifact_paths", {}).get("arrays") or "")
    return Path(raw) if raw else case_dir / "prefix_latent_story_smoke_arrays.npz"


def _operational_variant_index(report: dict[str, Any]) -> int:
    selected = report.get("selected_start_state", {})
    if isinstance(selected, dict) and selected.get("variant_index") is not None:
        return int(selected["variant_index"])
    for idx, row in enumerate(report.get("variant_rows", [])):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return idx
    return 0


def _posterior_sample_indices(
    arrays: Any,
    *,
    variant_idx: int,
    sample_count: int,
    posterior_mode: str,
) -> np.ndarray:
    mode = str(posterior_mode)
    if mode == "full":
        return np.arange(int(sample_count), dtype=np.int64)
    specs = {
        "top1": {"max_components": 1, "min_cumulative_weight": 1.0},
        "top2_80": {"max_components": 2, "min_cumulative_weight": 0.80},
        "top3_90": {"max_components": 3, "min_cumulative_weight": 0.90},
    }
    if mode not in specs:
        raise ValueError(f"unknown posterior_mode {posterior_mode!r}")
    required = {
        "rollout_component_variant_index",
        "rollout_component_window_index",
        "rollout_component_weight",
        "rollout_component_sample_count",
    }
    if not required.issubset(set(arrays.files)):
        return np.arange(int(sample_count), dtype=np.int64)
    components = component_slices_for_variant(
        variant_index=int(variant_idx),
        component_variant_index=np.asarray(
            arrays["rollout_component_variant_index"], dtype=np.int64
        ),
        component_window_index=np.asarray(
            arrays["rollout_component_window_index"], dtype=np.int64
        ),
        component_weight=np.asarray(
            arrays["rollout_component_weight"], dtype=np.float64
        ),
        component_sample_count=np.asarray(
            arrays["rollout_component_sample_count"], dtype=np.int64
        ),
        sample_count=int(sample_count),
    )
    component_rows = [
        {**row, "component_no": int(component_no)}
        for component_no, row in enumerate(components)
    ]
    selected = select_sparse_components(
        component_rows,
        max_components=int(specs[mode]["max_components"]),
        min_cumulative_weight=float(specs[mode]["min_cumulative_weight"]),
    )
    indices: list[int] = []
    for component in selected:
        start_slice, stop_slice = component["sample_slice"]
        indices.extend(range(int(start_slice), int(stop_slice)))
    if not indices:
        return np.arange(int(sample_count), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _case_dir(case_name: str, *, control_root: Path, variant_dir: str) -> Path:
    legacy = LEGACY_CASE_NAMES.get(str(case_name), str(case_name))
    candidates = [
        control_root / str(case_name) / variant_dir,
        control_root / legacy / variant_dir,
        control_root / str(case_name) / "fixed_start_18" / variant_dir,
        control_root / legacy / "fixed_start_18" / variant_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_case(
    case: tuple[str, str, str],
    *,
    control_root: Path,
    variant_dir: str,
    posterior_mode: str = DEFAULT_POSTERIOR_MODE,
) -> dict[str, Any]:
    label, case_name, color = case
    case_dir = _case_dir(case_name, control_root=control_root, variant_dir=variant_dir)
    report = _load_json(_report_path(case_dir))
    with np.load(_arrays_path(report, case_dir), allow_pickle=True) as arrays:
        variant_idx = _operational_variant_index(report)
        full_states = np.asarray(
            arrays["generated_states"][variant_idx], dtype=np.float32
        )
        sample_indices = _posterior_sample_indices(
            arrays,
            variant_idx=int(variant_idx),
            sample_count=int(full_states.shape[0]),
            posterior_mode=str(posterior_mode),
        )
        states = full_states[sample_indices]
        start = np.asarray(arrays["requested_raw"][variant_idx], dtype=np.float32)
    return {
        "label": label,
        "case_name": case_name,
        "color": color,
        "report": report,
        "states": states,
        "start": start,
        "posterior_mode": str(posterior_mode),
        "source_sample_count": int(full_states.shape[0]),
    }


def _load_casebook_cases(
    *,
    control_root: Path = CONTROL_ROOT,
    variant_dir: str = DEFAULT_VARIANT_DIR,
    posterior_mode: str = DEFAULT_POSTERIOR_MODE,
) -> list[dict[str, Any]]:
    return [
        _load_case(
            case,
            control_root=control_root,
            variant_dir=variant_dir,
            posterior_mode=posterior_mode,
        )
        for case in CASES
    ]


def _load_contrast_cases(
    *,
    control_root: Path = CONTROL_ROOT,
    variant_dir: str = DEFAULT_VARIANT_DIR,
    posterior_mode: str = DEFAULT_POSTERIOR_MODE,
) -> list[dict[str, Any]]:
    return [
        _load_case(
            case,
            control_root=control_root,
            variant_dir=variant_dir,
            posterior_mode=posterior_mode,
        )
        for case in CONTRAST_CASES
    ]


def _history_args(max_windows: int = 441) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=DEFAULT_CHECKPOINT,
        state_scope="joint38",
        eval_split="val",
        test_start=4511,
        val_size=441,
        max_windows=int(max_windows),
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )


def _selected_history_block(
    *,
    checkpoint: str,
    bridge_report: str,
    device: torch.device,
) -> dict[str, Any]:
    report = _load_json(Path(bridge_report))
    selected_windows = selected_bridge_window_indices(report)
    model, payload = load_model(checkpoint, device)
    (
        all_history_level,
        all_history_norm,
        all_center,
        all_scale,
        all_drift_feature,
        all_history_raw,
        specs,
        block,
    ) = build_val_block(_history_args(), payload)
    future_raw_all = _future_raw_from_block(
        block,
        int(all_history_raw.shape[0]),
        int(all_history_raw.shape[-1]),
    )
    return {
        "model": model,
        "history_level": all_history_level[selected_windows],
        "history_norm": all_history_norm[selected_windows],
        "center": all_center[selected_windows],
        "scale": all_scale[selected_windows],
        "drift_feature": all_drift_feature[selected_windows],
        "history_raw": all_history_raw[selected_windows],
        "future_raw": future_raw_all[selected_windows],
        "specs": specs,
        "spec_names": np.asarray(_spec_names(specs), dtype=str),
    }


def _cache_item(cache: Any, key: str) -> Any | None:
    keys = set(getattr(cache, "files", []))
    if not keys and isinstance(cache, dict):
        keys = set(cache)
    if key not in keys:
        return None
    value = cache[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _direct_sni_cache_matches(
    cache: Any,
    *,
    start_index: int,
    samples: int,
    checkpoint: str,
    bridge_report: str,
    selected_window_global: int,
) -> bool:
    states = _cache_item(cache, "direct_sni_states")
    if states is None or int(np.asarray(states).shape[0]) < int(samples):
        return False
    checks = {
        "start_index": int(start_index),
        "selected_window_global": int(selected_window_global),
        "checkpoint": str(checkpoint),
        "bridge_report": str(bridge_report),
    }
    for key, expected in checks.items():
        value = _cache_item(cache, key)
        if value is None:
            return False
        if isinstance(expected, int):
            if int(value) != expected:
                return False
        elif str(value) != expected:
            return False
    return True


def _load_or_sample_direct_sni(
    *,
    start_index: int,
    samples: int,
    checkpoint: str,
    bridge_report: str,
    device_name: str,
    refresh: bool,
) -> dict[str, Any]:
    bridge_report_key = str(Path(bridge_report))
    selected_windows = selected_bridge_window_indices(_load_json(Path(bridge_report)))
    selected_window_global = int(selected_windows[int(start_index)])
    if BACKTEST_CACHE.exists() and not refresh:
        cached = np.load(BACKTEST_CACHE)
        if _direct_sni_cache_matches(
            cached,
            start_index=int(start_index),
            samples=int(samples),
            checkpoint=str(checkpoint),
            bridge_report=bridge_report_key,
            selected_window_global=selected_window_global,
        ):
            return {
                "history": np.asarray(cached["history"], dtype=np.float32),
                "future": np.asarray(cached["future"], dtype=np.float32),
                "direct_sni_states": np.asarray(
                    cached["direct_sni_states"][: int(samples)], dtype=np.float32
                ),
                "spec_names": np.asarray(cached["spec_names"], dtype=str),
            }

    device = torch.device(
        device_name if torch.cuda.is_available() or str(device_name) == "cpu" else "cpu"
    )
    block = _selected_history_block(
        checkpoint=checkpoint,
        bridge_report=bridge_report,
        device=device,
    )
    sampled = sample_normal_generator_for_retrieved_analogues(
        block["model"],
        [{"index": int(start_index), "cosine": 1.0}],
        block["history_level"],
        block["history_norm"],
        block["center"],
        block["scale"],
        block["drift_feature"],
        n_samples=int(samples),
        n_steps=30,
        chunk_size=8,
        temperature=0.5,
        device=device,
    )
    states = _reconstruct_states(
        block["history_raw"][[int(start_index)], -1, :],
        sampled["increments"],
        block["specs"],
    )[0]
    BACKTEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        BACKTEST_CACHE,
        start_index=np.asarray(int(start_index), dtype=np.int64),
        history=block["history_raw"][int(start_index)].astype(np.float32),
        future=block["future_raw"][int(start_index)].astype(np.float32),
        direct_sni_states=states.astype(np.float32),
        spec_names=block["spec_names"],
        checkpoint=np.asarray(str(checkpoint)),
        bridge_report=np.asarray(bridge_report_key),
        selected_window_global=np.asarray(selected_window_global, dtype=np.int64),
    )
    return {
        "history": block["history_raw"][int(start_index)].astype(np.float32),
        "future": block["future_raw"][int(start_index)].astype(np.float32),
        "direct_sni_states": states.astype(np.float32),
        "spec_names": block["spec_names"],
    }


def _quantiles(
    paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(paths, dtype=np.float32)
    return (
        np.percentile(arr, 10, axis=0),
        np.percentile(arr, 50, axis=0),
        np.percentile(arr, 90, axis=0),
        np.mean(arr, axis=0),
    )


def _paths_with_initial_value(paths: np.ndarray, initial: float) -> np.ndarray:
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D path array, got shape {arr.shape}")
    initial_col = np.full((arr.shape[0], 1), float(initial), dtype=np.float32)
    return np.concatenate([initial_col, arr], axis=1)


def _level_paths_with_start(states: np.ndarray, idx: int, start: float) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D state array, got shape {arr.shape}")
    return _paths_with_initial_value(arr[:, :, int(idx)], float(start))


def _plot_future_fan(
    ax: plt.Axes,
    states: np.ndarray,
    idx: int,
    *,
    start: float,
    color: str,
    label: str,
    show_samples: bool = True,
    realized: np.ndarray | None = None,
) -> None:
    full_days = np.arange(0, 31)
    level_paths = _level_paths_with_start(states, idx, start)
    q10, q50, q90, mean = _quantiles(level_paths)
    ax.fill_between(full_days, q10, q90, color=color, alpha=0.14, label="10-90% band")
    ax.plot(full_days, q50, color=color, linewidth=2.2, label="Median")
    ax.plot(
        full_days,
        mean,
        color="#455A64",
        linewidth=1.5,
        linestyle="--",
        label="Mean",
    )
    if show_samples:
        path_colors = ["#2E7D32", "#EF6C00", "#6A1B9A", "#00838F"]
        picks = np.linspace(0, states.shape[0] - 1, min(4, states.shape[0]), dtype=int)
        for j, pick in enumerate(picks):
            ax.plot(
                full_days,
                level_paths[pick],
                color=path_colors[j % len(path_colors)],
                linewidth=0.85,
                alpha=0.62,
            )
    if realized is not None:
        ax.plot(
            full_days,
            np.r_[start, realized[:, idx]],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="Realized future",
        )
    ax.scatter(
        [0],
        [float(start)],
        color="black",
        s=22,
        zorder=5,
        label="Accepted start" if realized is None else None,
    )
    ax.axvline(0, color="#9E9E9E", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 30)
    ax.grid(alpha=0.15)
    ax.set_title(label, fontsize=10, fontweight="bold")


def plot_casebook(
    cases: list[dict[str, Any]],
    output: Path,
    *,
    realized_future: np.ndarray | None = None,
) -> None:
    rows = [
        ("SPX", MARKET_INDEX["SPX"]),
        ("VIX", MARKET_INDEX["VIX"]),
        ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
    ]
    fig, axes = plt.subplots(len(rows), len(cases), figsize=(14, 9), sharex=True)
    fig.suptitle(
        "Narrative-conditioned scenario fans at the same accepted start\n"
        "Raw levels, nearest-similar main-regime top3/90 posterior view",
        fontsize=14,
        fontweight="bold",
    )
    for col, case in enumerate(cases):
        for row, (market, idx) in enumerate(rows):
            ax = axes[row, col]
            _plot_future_fan(
                ax,
                case["states"],
                idx,
                start=float(case["start"][idx]),
                color=str(case["color"]),
                label=str(case["label"]) if row == 0 else market,
                realized=realized_future,
            )
            if col == 0:
                ax.set_ylabel(f"{market}\nraw level")
            if row == len(rows) - 1:
                ax.set_xlabel("Forward day")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(
    cases: list[dict[str, Any]],
    direct: dict[str, Any],
    output: Path,
) -> None:
    rows = [
        ("SPX", MARKET_INDEX["SPX"]),
        ("VIX", MARKET_INDEX["VIX"]),
    ]
    hist_days = np.arange(-29, 1)
    full_days = np.arange(0, 31)
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        "Historical backtest view: same history, narrative variants, native SNI",
        fontsize=14,
        fontweight="bold",
    )
    for ax, (market, idx) in zip(axes, rows, strict=True):
        history = direct["history"][:, idx]
        future = direct["future"][:, idx]
        start = float(history[-1])
        direct_states = np.asarray(direct["direct_sni_states"], dtype=np.float32)
        direct_level_paths = _level_paths_with_start(direct_states, idx, start)
        d_q10, d_q50, d_q90, _d_mean = _quantiles(direct_level_paths)
        ax.plot(
            hist_days,
            history,
            color="black",
            linewidth=2.0,
            label="Actual 30-day history",
        )
        ax.plot(
            full_days,
            np.r_[start, future],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Ground-truth future",
        )
        ax.fill_between(
            full_days,
            d_q10,
            d_q90,
            color="#78909C",
            alpha=0.18,
            label="Native SNI from actual history",
        )
        ax.plot(full_days, d_q50, color="#546E7A", linewidth=2.0)
        for case in cases:
            case_start = float(np.asarray(case["start"], dtype=np.float32)[idx])
            case_level_paths = _level_paths_with_start(case["states"], idx, case_start)
            q10, q50, q90, _mean = _quantiles(case_level_paths)
            color = str(case["color"])
            ax.fill_between(full_days, q10, q90, color=color, alpha=0.07)
            ax.plot(
                full_days,
                q50,
                color=color,
                linewidth=1.9,
                label=str(case["label"]),
            )
        ax.axvline(0, color="#9E9E9E", linestyle=":", linewidth=0.9)
        ax.set_ylabel(f"{market}\nraw level")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Day (0 = forecast start)")
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _terminal_stats(states: np.ndarray, idx: int) -> tuple[float, float, float]:
    q10, q50, q90 = np.percentile(states[:, -1, idx], [10, 50, 90])
    return float(q10), float(q50), float(q90)


def _support_summary(case: dict[str, Any]) -> str:
    row = next(
        (
            item
            for item in case["report"].get("variant_rows", [])
            if isinstance(item, dict) and bool(item.get("is_operational"))
        ),
        {},
    )
    cosine = row.get("memory_support_cosine")
    start_z = row.get("memory_prior_weighted_start_distance_z")
    count = row.get("memory_prior_analogue_count")
    return f"support={float(cosine):.2f}, start-z={float(start_z):.1f}, k={int(count)}"


def _grounding_payload(case: dict[str, Any]) -> dict[str, Any]:
    payload = case.get("report", {}).get("cached_query", {}).get("grounding", {})
    return payload if isinstance(payload, dict) else {}


def _clean_condition_text(case: dict[str, Any]) -> str:
    grounding = _grounding_payload(case)
    text = grounding.get("cleaned_conditioning_text")
    if text:
        return str(text)
    nested = grounding.get("condition_only_grounding", {})
    if isinstance(nested, dict):
        text = nested.get("cleaned_conditioning_text")
        if text:
            return str(text)
    return str(case.get("report", {}).get("cached_query", {}).get("narrative_text", ""))


def _grounded_implication_text(case: dict[str, Any]) -> str:
    grounding = _grounding_payload(case)
    implications = grounding.get("market_implications")
    if not isinstance(implications, list):
        nested = grounding.get("condition_only_grounding", {})
        implications = (
            nested.get("current_market_state_implications", [])
            if isinstance(nested, dict)
            else []
        )
    pieces = []
    for item in implications:
        if not isinstance(item, dict):
            continue
        market = str(item.get("market", "")).replace("_", " ")
        direction = str(item.get("direction", ""))
        confidence = str(item.get("confidence", ""))
        if market and direction:
            pieces.append(f"{market} {direction} ({confidence})")
    return "; ".join(pieces)


def _forward_warning_text(case: dict[str, Any]) -> str:
    grounding = _grounding_payload(case)
    rows = grounding.get("non_conditioning_forward_language")
    if not isinstance(rows, list):
        nested = grounding.get("condition_only_grounding", {})
        rows = (
            nested.get("non_conditioning_forward_language", [])
            if isinstance(nested, dict)
            else []
        )
    phrases = []
    for item in rows:
        if isinstance(item, dict) and item.get("phrase"):
            phrases.append(str(item["phrase"]))
    return "; ".join(phrases)


def _top_support(case: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    prior = case.get("report", {}).get("cached_query", {}).get("memory_prior", {})
    details = prior.get("candidate_details", []) if isinstance(prior, dict) else []
    weights = prior.get("weights", []) if isinstance(prior, dict) else []
    rows: list[dict[str, Any]] = []
    for pos, item in enumerate(details[: int(limit)]):
        if not isinstance(item, dict):
            continue
        mismatches = int(item.get("recent_prefix_mismatches", 0) or 0)
        checked = int(item.get("recent_prefix_checked", 0) or 0)
        rows.append(
            {
                "rank": int(item.get("rank", pos + 1) or pos + 1),
                "window_id": str(item.get("window_id", "")),
                "window_index": int(item.get("window_index", -1) or -1),
                "history_end_date": str(item.get("history_end_date", "")),
                "weight": (
                    float(weights[pos])
                    if pos < len(weights)
                    else float(item.get("weight", 0.0) or 0.0)
                ),
                "narrative_match": float(
                    item.get(
                        "memory_support_cosine", item.get("narrative_start_score", 0.0)
                    )
                    or 0.0
                ),
                "start_fit_z": float(item.get("start_distance_z", 0.0) or 0.0),
                "direction_check": (
                    "pass"
                    if checked > 0 and mismatches == 0
                    else f"{mismatches}/{checked} mismatches"
                ),
            }
        )
    return rows


def _terminal_case_levels(case: dict[str, Any]) -> dict[str, dict[str, float]]:
    states = np.asarray(case["states"], dtype=np.float32)
    start = np.asarray(case["start"], dtype=np.float32)
    out: dict[str, dict[str, float]] = {}
    for market, idx in SHIFT_MARKETS:
        q10, q50, q90 = np.percentile(states[:, -1, idx], [10, 50, 90])
        out[market] = {
            "start": float(start[idx]),
            "p10": float(q10),
            "p50": float(q50),
            "p90": float(q90),
            "median_change_from_start": float(q50 - start[idx]),
        }
    return out


def _native_terminal_levels(direct: dict[str, Any]) -> dict[str, float]:
    states = np.asarray(direct["direct_sni_states"], dtype=np.float32)
    terminal = np.percentile(states[:, -1, :], 50, axis=0)
    return {market: float(terminal[idx]) for market, idx in SHIFT_MARKETS}


def _direction_word(value: float) -> str:
    return "higher" if value > 0 else "lower"


def _market_shift_sentence(
    case: dict[str, Any],
    *,
    baseline_levels: dict[str, dict[str, float]],
    top_n: int = 3,
) -> str:
    levels = _terminal_case_levels(case)
    shifts = []
    for market in levels:
        diff = levels[market]["p50"] - baseline_levels[market]["p50"]
        shifts.append((abs(float(diff)), market, float(diff)))
    shifts.sort(reverse=True)
    pieces = [
        f"{market} {_direction_word(diff)} by {abs(diff):.3g}"
        for _abs_diff, market, diff in shifts[: int(top_n)]
    ]
    return (
        f"Relative to the fragile-risk-on reference, the day-30 median is "
        + ", ".join(pieces)
        + "."
    )


def build_qualitative_casebook_summary(
    cases: list[dict[str, Any]],
    contrast_cases: list[dict[str, Any]],
    direct: dict[str, Any],
) -> dict[str, Any]:
    """Create text-plus-distribution evidence for paper/demo interpretation."""

    if not contrast_cases:
        raise ValueError("contrast_cases must be non-empty")
    baseline = contrast_cases[0]
    baseline_levels = _terminal_case_levels(baseline)
    native_levels = _native_terminal_levels(direct)
    summaries = []
    for case in contrast_cases:
        terminal = _terminal_case_levels(case)
        terminal_vs_native = {
            market: {
                **values,
                "median_shift_vs_native_sni": float(
                    values["p50"] - native_levels[market]
                ),
            }
            for market, values in terminal.items()
        }
        summaries.append(
            {
                "case_name": str(case.get("case_name", "")),
                "label": str(case.get("label", "")),
                "narrative_text": str(
                    case.get("report", {})
                    .get("cached_query", {})
                    .get("narrative_text", "")
                ),
                "clean_conditioning_text": _clean_condition_text(case),
                "grounded_implications": _grounded_implication_text(case),
                "forward_language_excluded": _forward_warning_text(case),
                "top_support": _top_support(case),
                "terminal_raw_level_summary": terminal_vs_native,
                "qualitative_read": (
                    "This is the fragile-risk-on reference case."
                    if case is baseline
                    else _market_shift_sentence(case, baseline_levels=baseline_levels)
                ),
            }
        )
    contrast_rows = []
    for case in contrast_cases[1:]:
        left_support = {row["window_index"] for row in _top_support(baseline, limit=8)}
        right_support = {row["window_index"] for row in _top_support(case, limit=8)}
        union = left_support | right_support
        jaccard = len(left_support & right_support) / max(len(union), 1)
        contrast_rows.append(
            {
                "left": str(baseline.get("label", "")),
                "right": str(case.get("label", "")),
                "support_jaccard_vs_reference": float(jaccard),
                "qualitative_read": _market_shift_sentence(
                    case,
                    baseline_levels=baseline_levels,
                ),
            }
        )
    return {
        "scope_note": (
            "Qualitative fixed-start casebook built from saved component-prefix "
            "narrative runs. It reports raw-level terminal fan summaries, support "
            "provenance, grounded current-market claims, and text interpretations."
        ),
        "case_count": int(len(contrast_cases)),
        "paper_case_count": int(len(cases)),
        "same_start_check": {
            "max_abs_start_difference": float(
                max(
                    np.max(
                        np.abs(
                            np.asarray(case["start"], dtype=np.float64)
                            - np.asarray(baseline["start"], dtype=np.float64)
                        )
                    )
                    for case in contrast_cases
                )
            )
        },
        "native_sni_terminal_medians": native_levels,
        "case_summaries": summaries,
        "reference_pairwise_reads": contrast_rows,
    }


def render_qualitative_casebook_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Narrative Qualitative Casebook Summary",
        "",
        str(summary["scope_note"]),
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Same-start max absolute difference: "
        f"`{summary['same_start_check']['max_abs_start_difference']:.6g}`",
        "",
        "## Case Reads",
        "",
    ]
    for case in summary["case_summaries"]:
        support = case.get("top_support", [])
        support_text = ", ".join(
            f"{row['window_id']}@{row['history_end_date']} w={row['weight']:.2f}"
            for row in support[:3]
        )
        terminal = case["terminal_raw_level_summary"]
        terminal_text = ", ".join(
            f"{market} p50={values['p50']:.4g}"
            for market, values in terminal.items()
            if market in {"SPX", "VIX", "BBB OAS", "Crude", "1Y ATM IV"}
        )
        lines.extend(
            [
                f"### {case['label']}",
                "",
                f"- Narrative: {case['narrative_text']}",
                f"- Conditioning text: {case['clean_conditioning_text']}",
                f"- Grounded claims: {case['grounded_implications']}",
                f"- Forward language excluded: {case['forward_language_excluded'] or 'none'}",
                f"- Top support: {support_text}",
                f"- Terminal raw medians: {terminal_text}",
                f"- Read: {case['qualitative_read']}",
                "",
            ]
        )
    lines.extend(["## Reference Contrasts", ""])
    for row in summary["reference_pairwise_reads"]:
        lines.append(
            f"- {row['left']} vs {row['right']}: "
            f"support Jaccard `{row['support_jaccard_vs_reference']:.3f}`; "
            f"{row['qualitative_read']}"
        )
    lines.append("")
    return "\n".join(lines)


def plot_spx_conditionality(
    cases: list[dict[str, Any]],
    direct: dict[str, Any],
    output: Path,
) -> None:
    spx_idx = MARKET_INDEX["SPX"]
    full_days = np.arange(0, 31)
    direct_states = np.asarray(direct["direct_sni_states"], dtype=np.float32)
    start = float(direct["history"][-1, spx_idx])
    future = np.asarray(direct["future"], dtype=np.float32)
    direct_level_paths = _level_paths_with_start(direct_states, spx_idx, start)
    d_q10, d_q50, d_q90, _d_mean = _quantiles(direct_level_paths)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.3, 1.0])
    ax_fan = fig.add_subplot(gs[0, :])
    ax_term = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])
    fig.suptitle(
        "Fixed-start SPX narrative conditionality diagnostic",
        fontsize=14,
        fontweight="bold",
    )

    ax_fan.fill_between(
        full_days,
        d_q10,
        d_q90,
        color="#78909C",
        alpha=0.16,
        label="Native SNI actual-history 10-90%",
    )
    ax_fan.plot(
        full_days,
        d_q50,
        color="#455A64",
        linewidth=2.2,
        label="Native SNI actual-history median",
    )
    ax_fan.plot(
        full_days,
        np.r_[start, future[:, spx_idx]],
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="Realized future",
    )

    terminal_rows: list[tuple[str, str, float, float, float]] = []
    for case in cases:
        color = str(case["color"])
        states = np.asarray(case["states"], dtype=np.float32)
        case_start = float(np.asarray(case["start"], dtype=np.float32)[spx_idx])
        case_level_paths = _level_paths_with_start(states, spx_idx, case_start)
        q10, q50, q90, _mean = _quantiles(case_level_paths)
        ax_fan.fill_between(full_days, q10, q90, color=color, alpha=0.075)
        ax_fan.plot(
            full_days,
            q50,
            color=color,
            linewidth=2.0,
            label=str(case["label"]),
        )
        t10, t50, t90 = _terminal_stats(states, spx_idx)
        terminal_rows.append((str(case["label"]), color, t10, t50, t90))

    ax_fan.axvline(0, color="#9E9E9E", linestyle=":", linewidth=0.9)
    ax_fan.set_xlim(0, 30)
    ax_fan.set_ylabel("SPX raw level")
    ax_fan.set_xlabel("Forward day")
    ax_fan.grid(alpha=0.15)
    ax_fan.legend(fontsize=8, ncol=3, loc="best")
    ax_fan.text(
        0.01,
        0.03,
        f"All narrative runs use the same raw start: SPX={start:.1f}, VIX={float(direct['history'][-1, MARKET_INDEX['VIX']]):.1f}.",
        transform=ax_fan.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )

    y_pos = np.arange(len(terminal_rows))
    for y, (label, color, t10, t50, t90) in zip(y_pos, terminal_rows, strict=True):
        ax_term.plot([t10, t90], [y, y], color=color, linewidth=5, alpha=0.28)
        ax_term.scatter([t50], [y], color=color, s=42, zorder=3)
        ax_term.text(
            t90 + 2.0,
            y,
            _support_summary(next(case for case in cases if case["label"] == label)),
            va="center",
            fontsize=8,
            color="#455A64",
        )
    ax_term.axvline(start, color="black", linewidth=1.0, linestyle=":", label="Start")
    ax_term.axvline(
        float(d_q50[-1]),
        color="#455A64",
        linewidth=1.4,
        linestyle="--",
        label="Native SNI median",
    )
    ax_term.set_yticks(y_pos)
    ax_term.set_yticklabels([row[0] for row in terminal_rows], fontsize=8)
    ax_term.invert_yaxis()
    ax_term.set_xlabel("Terminal SPX raw level, day 30")
    ax_term.set_title(
        "Terminal level intervals by narrative", fontsize=10, fontweight="bold"
    )
    ax_term.grid(axis="x", alpha=0.18)
    ax_term.legend(fontsize=8, loc="lower right")

    direct_terminal = direct_states[:, -1, :]
    direct_median = np.percentile(direct_terminal, 50, axis=0)
    direct_scale = np.maximum(np.std(direct_terminal, axis=0), 1e-6)
    heat = []
    for case in cases:
        terminal = np.asarray(case["states"], dtype=np.float32)[:, -1, :]
        median = np.percentile(terminal, 50, axis=0)
        heat.append(
            [
                float((median[idx] - direct_median[idx]) / direct_scale[idx])
                for _label, idx in SHIFT_MARKETS
            ]
        )
    heat_arr = np.asarray(heat, dtype=np.float32)
    im = ax_heat.imshow(heat_arr, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax_heat.set_yticks(np.arange(len(cases)))
    ax_heat.set_yticklabels([str(case["label"]) for case in cases], fontsize=8)
    ax_heat.set_xticks(np.arange(len(SHIFT_MARKETS)))
    ax_heat.set_xticklabels(
        [label for label, _idx in SHIFT_MARKETS], rotation=45, ha="right", fontsize=8
    )
    ax_heat.set_title(
        "Terminal median shift vs native SNI, in native-SNI std units",
        fontsize=10,
        fontweight="bold",
    )
    for i in range(heat_arr.shape[0]):
        for j in range(heat_arr.shape[1]):
            ax_heat.text(
                j,
                i,
                f"{heat_arr[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(float(heat_arr[i, j])) < 1.0 else "white",
            )
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_factor_conditionality(
    cases: list[dict[str, Any]],
    direct: dict[str, Any],
    output: Path,
) -> None:
    panels = [
        ("SPX", MARKET_INDEX["SPX"]),
        ("Crude oil", 31),
        ("BBB OAS", 35),
        ("VIX", MARKET_INDEX["VIX"]),
        ("Gold", 37),
        ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
    ]
    full_days = np.arange(0, 31)
    direct_states = np.asarray(direct["direct_sni_states"], dtype=np.float32)
    future = np.asarray(direct["future"], dtype=np.float32)
    history = np.asarray(direct["history"], dtype=np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    fig.suptitle(
        "Fixed-start narrative conditionality across raw market factors",
        fontsize=14,
        fontweight="bold",
    )
    for ax, (market, idx) in zip(axes.ravel(), panels, strict=True):
        start = float(history[-1, idx])
        direct_level_paths = _level_paths_with_start(direct_states, idx, start)
        d_q10, d_q50, d_q90, _d_mean = _quantiles(direct_level_paths)
        ax.fill_between(
            full_days,
            d_q10,
            d_q90,
            color="#78909C",
            alpha=0.14,
            label="Native SNI 10-90%" if market == "SPX" else None,
        )
        ax.plot(
            full_days,
            d_q50,
            color="#455A64",
            linewidth=2.1,
            label="Native SNI median" if market == "SPX" else None,
        )
        ax.plot(
            full_days,
            np.r_[start, future[:, idx]],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="Realized future" if market == "SPX" else None,
        )
        for case in cases:
            color = str(case["color"])
            states = np.asarray(case["states"], dtype=np.float32)
            case_start = float(np.asarray(case["start"], dtype=np.float32)[idx])
            case_level_paths = _level_paths_with_start(states, idx, case_start)
            q10, q50, q90, _mean = _quantiles(case_level_paths)
            ax.fill_between(full_days, q10, q90, color=color, alpha=0.055)
            ax.plot(
                full_days,
                q50,
                color=color,
                linewidth=1.9,
                label=str(case["label"]) if market == "SPX" else None,
            )
        ax.axvline(0, color="#9E9E9E", linestyle=":", linewidth=0.9)
        ax.set_title(f"{market} raw level", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 30)
        ax.grid(alpha=0.15)
        ax.set_xlabel("Forward day")
        ax.set_ylabel("Raw level")
    axes[0, 0].legend(fontsize=8, ncol=2, loc="best")
    fig.text(
        0.5,
        0.02,
        (
            "All panels use the same raw start. Bands overlap because these are broad "
            "scenario distributions; use the path-control audit, not visual band "
            "separation alone, to judge fixed-start narrative conditionality."
        ),
        ha="center",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_case_conversion(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        report = case["report"]
        variant_idx = _operational_variant_index(report)
        spx_row = next(
            row
            for row in report["generation"]["path_quantiles"]
            if row.get("market") == "SPX"
            and row.get("analogue_key") == f"RANK_{variant_idx + 1}"
        )
        start = float(case["start"][MARKET_INDEX["SPX"]])
        q10, q50, q90, _mean = _quantiles(case["states"][:, :, MARKET_INDEX["SPX"]])
        row_is_raw = str(spx_row.get("value_kind", "")) == "raw_level"
        row_offset = 0.0 if row_is_raw else start
        errors = {
            "p10": float(np.max(np.abs(np.asarray(spx_row["p10"]) + row_offset - q10))),
            "p50": float(np.max(np.abs(np.asarray(spx_row["p50"]) + row_offset - q50))),
            "p90": float(np.max(np.abs(np.asarray(spx_row["p90"]) + row_offset - q90))),
        }
        all_row = next(
            row
            for row in report["generation"]["path_quantiles"]
            if row.get("market") == "SPX" and row.get("analogue_key") == "ALL"
        )
        all_offset = 0.0 if str(all_row.get("value_kind", "")) == "raw_level" else start
        all_terminal = float(np.asarray(all_row["p50"])[-1] + all_offset)
        rows.append(
            {
                "case": str(case["label"]),
                "operational_variant": int(variant_idx),
                "start_spx": start,
                "max_abs_errors": errors,
                "selected_terminal_p50": float(q50[-1]),
                "pooled_all_terminal_p50_after_selected_start": all_terminal,
                "pooled_minus_selected_terminal_p50": all_terminal - float(q50[-1]),
            }
        )
    return rows


def _empirical_ks(a: np.ndarray, b: np.ndarray) -> float:
    left = np.sort(np.asarray(a, dtype=np.float64).ravel())
    right = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if left.size == 0 or right.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, grid, side="right") / float(left.size)
    right_cdf = np.searchsorted(right, grid, side="right") / float(right.size)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _standardized_sample_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    if left.size != right.size or left.size < 2:
        return float("nan")
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1e-12 or right_std <= 1e-12:
        return float("nan")
    left_z = (left - float(left.mean())) / left_std
    right_z = (right - float(right.mean())) / right_std
    return float(np.corrcoef(left_z, right_z)[0, 1])


def _normalized_quantile_shape_l2(a: np.ndarray, b: np.ndarray) -> float:
    quantiles = np.linspace(0.05, 0.95, 19)

    def norm_q(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).ravel()
        std = float(arr.std())
        if std <= 1e-12:
            return np.zeros_like(quantiles)
        return (np.quantile(arr, quantiles) - float(arr.mean())) / std

    return float(np.sqrt(np.mean((norm_q(a) - norm_q(b)) ** 2)))


def narrative_distribution_shape_diagnostic(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether narrative runs change distribution shape or mostly shift levels."""

    if not cases:
        return {}
    baseline = cases[0]
    factor_panels = [
        ("SPX", MARKET_INDEX["SPX"]),
        ("Crude oil", 31),
        ("BBB OAS", 35),
        ("VIX", MARKET_INDEX["VIX"]),
        ("Gold", 37),
        ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
    ]
    baseline_support = {
        int(row.get("window_index", row.get("bridge_local_index", -1)))
        for row in baseline["report"]["cached_query"]["memory_prior"].get(
            "candidate_details", []
        )
    }
    rows: list[dict[str, Any]] = []
    for case in cases[1:]:
        support = {
            int(row.get("window_index", row.get("bridge_local_index", -1)))
            for row in case["report"]["cached_query"]["memory_prior"].get(
                "candidate_details", []
            )
        }
        overlap = len(baseline_support & support) / max(
            1, len(baseline_support | support)
        )
        for market, idx in factor_panels:
            base_terminal = np.asarray(baseline["states"][:, -1, idx], dtype=np.float64)
            case_terminal = np.asarray(case["states"][:, -1, idx], dtype=np.float64)
            base_width = float(
                np.quantile(base_terminal, 0.9) - np.quantile(base_terminal, 0.1)
            )
            case_width = float(
                np.quantile(case_terminal, 0.9) - np.quantile(case_terminal, 0.1)
            )
            rows.append(
                {
                    "baseline": str(baseline["label"]),
                    "case": str(case["label"]),
                    "market": market,
                    "support_jaccard_vs_baseline": float(overlap),
                    "baseline_terminal_mean": float(base_terminal.mean()),
                    "case_terminal_mean": float(case_terminal.mean()),
                    "mean_diff": float(case_terminal.mean() - base_terminal.mean()),
                    "baseline_terminal_std": float(base_terminal.std()),
                    "case_terminal_std": float(case_terminal.std()),
                    "std_ratio": (
                        float(case_terminal.std() / base_terminal.std())
                        if float(base_terminal.std()) > 1e-12
                        else float("nan")
                    ),
                    "width_10_90_ratio": (
                        float(case_width / base_width)
                        if abs(base_width) > 1e-12
                        else float("nan")
                    ),
                    "standardized_sample_corr": _standardized_sample_corr(
                        base_terminal,
                        case_terminal,
                    ),
                    "normalized_quantile_shape_l2": _normalized_quantile_shape_l2(
                        base_terminal,
                        case_terminal,
                    ),
                    "terminal_ks": _empirical_ks(base_terminal, case_terminal),
                }
            )
    corr_values = [
        float(row["standardized_sample_corr"])
        for row in rows
        if np.isfinite(float(row["standardized_sample_corr"]))
    ]
    qshape_values = [
        float(row["normalized_quantile_shape_l2"])
        for row in rows
        if np.isfinite(float(row["normalized_quantile_shape_l2"]))
    ]
    ks_values = [
        float(row["terminal_ks"])
        for row in rows
        if np.isfinite(float(row["terminal_ks"]))
    ]
    median_corr = float(np.median(corr_values)) if corr_values else float("nan")
    median_shape = float(np.median(qshape_values)) if qshape_values else float("nan")
    if np.isfinite(median_corr) and median_corr > 0.9 and median_shape < 0.06:
        interpretation = (
            "Fixed-start narrative runs mostly shift or scale the same "
            "frozen-generator stochastic family; conditionality is too weak for "
            "risk-manager-facing claims."
        )
    elif np.isfinite(median_corr) and median_corr < 0.5:
        interpretation = (
            "Fixed-start narrative runs are no longer sample-index clones; the "
            "support mixture changes the stochastic family. Use scenario-level "
            "backtests before promoting the method."
        )
    else:
        interpretation = (
            "Fixed-start narrative runs show partial conditionality; inspect "
            "market-level KS, width ratios, and scenario-level backtests before "
            "promotion."
        )
    return {
        "baseline_case": str(baseline["label"]),
        "market_count": int(len(factor_panels)),
        "comparison_count": int(len(rows)),
        "median_standardized_sample_corr": median_corr,
        "median_normalized_quantile_shape_l2": median_shape,
        "median_terminal_ks": (
            float(np.median(ks_values)) if ks_values else float("nan")
        ),
        "rows": rows,
        "interpretation": interpretation,
    }


def _portfolio_factor_matrix(paths: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Return standardized factor moves for the illustrative portfolio layer.

    The casebook portfolio is deliberately not a priced book. It is a simple
    normalized exposure vector so the paper can show how scenario fans translate
    into risk-manager units without introducing a valuation model.
    """

    states = np.asarray(paths, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64)
    factors = []
    for exposure in PORTFOLIO_EXPOSURES:
        idx = int(exposure["index"])
        series = states[:, :, idx]
        base = float(start_arr[idx])
        scale = max(abs(base), 1.0)
        factors.append((series - base) / scale)
    return np.stack(factors, axis=-1)


def _portfolio_pnl(paths: np.ndarray, start: np.ndarray) -> np.ndarray:
    moves = _portfolio_factor_matrix(paths, start)
    weights = np.asarray(
        [float(exposure["sensitivity"]) for exposure in PORTFOLIO_EXPOSURES],
        dtype=np.float64,
    )
    return np.einsum("stf,f->st", moves, weights) * 100.0


def _portfolio_pnl_with_start(paths: np.ndarray, start: np.ndarray) -> np.ndarray:
    return _paths_with_initial_value(_portfolio_pnl(paths, start), 0.0)


def _portfolio_component_pnl(
    paths: np.ndarray,
    start: np.ndarray,
    exposure: dict[str, Any],
) -> np.ndarray:
    idx = int(exposure["index"])
    states = np.asarray(paths, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64)
    base = float(start_arr[idx])
    scale = max(abs(base), 1.0)
    return ((states[:, :, idx] - base) / scale) * float(exposure["sensitivity"]) * 100.0


def _portfolio_stats(pnl: np.ndarray) -> dict[str, float]:
    terminal = np.asarray(pnl, dtype=np.float64)[:, -1]
    return {
        "terminal_p10": float(np.percentile(terminal, 10)),
        "terminal_p50": float(np.percentile(terminal, 50)),
        "terminal_p90": float(np.percentile(terminal, 90)),
        "terminal_mean": float(np.mean(terminal)),
        "terminal_var95_loss": float(-np.percentile(terminal, 5)),
        "terminal_expected_shortfall95_loss": float(
            -np.mean(terminal[terminal <= np.percentile(terminal, 5)])
        ),
        "worst_terminal_pnl": float(np.min(terminal)),
        "best_terminal_pnl": float(np.max(terminal)),
    }


def _portfolio_contributions(
    case: dict[str, Any], top_n: int = 4
) -> list[dict[str, Any]]:
    states = np.asarray(case["states"], dtype=np.float64)
    start = np.asarray(case["start"], dtype=np.float64)
    rows = []
    for exposure in PORTFOLIO_EXPOSURES:
        component = _portfolio_component_pnl(states, start, exposure)
        terminal = component[:, -1]
        rows.append(
            {
                "market": str(exposure["market"]),
                "description": str(exposure["description"]),
                "sensitivity": float(exposure["sensitivity"]),
                "terminal_median_contribution": float(np.percentile(terminal, 50)),
                "terminal_tail_contribution": float(np.percentile(terminal, 5)),
            }
        )
    rows.sort(
        key=lambda row: abs(float(row["terminal_tail_contribution"])), reverse=True
    )
    return rows[: int(top_n)]


def build_portfolio_impact_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_rows = []
    for case in cases:
        pnl = _portfolio_pnl(case["states"], case["start"])
        stats = _portfolio_stats(pnl)
        case_rows.append(
            {
                "case_name": str(case.get("case_name", "")),
                "label": str(case.get("label", "")),
                "portfolio_stats": stats,
                "largest_tail_contributors": _portfolio_contributions(case),
                "top_support": _top_support(case),
            }
        )
    terminal_medians = [
        float(row["portfolio_stats"]["terminal_p50"]) for row in case_rows
    ]
    var95_losses = [
        float(row["portfolio_stats"]["terminal_var95_loss"]) for row in case_rows
    ]
    return {
        "scope_note": (
            "Illustrative normalized portfolio impact layer built on the same "
            "fixed-start narrative scenario paths. Values are not priced P&L; "
            "they are exposure-weighted normalized factor moves in portfolio "
            "risk units, intended to show how narratives change risk outputs."
        ),
        "exposures": PORTFOLIO_EXPOSURES,
        "case_summaries": case_rows,
        "cross_narrative_range": {
            "terminal_p50_range": float(max(terminal_medians) - min(terminal_medians)),
            "var95_loss_range": float(max(var95_losses) - min(var95_losses)),
        },
    }


def render_portfolio_impact_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Narrative Portfolio Impact Summary",
        "",
        str(summary["scope_note"]),
        "",
        "## Exposure Vector",
        "",
    ]
    for exposure in summary["exposures"]:
        lines.append(
            f"- {exposure['market']}: sensitivity `{float(exposure['sensitivity']):.2f}` "
            f"({exposure['description']})"
        )
    lines.extend(["", "## Case Reads", ""])
    for case in summary["case_summaries"]:
        stats = case["portfolio_stats"]
        contributors = ", ".join(
            f"{row['market']} tail={row['terminal_tail_contribution']:.3f}"
            for row in case["largest_tail_contributors"]
        )
        support_text = ", ".join(
            f"{row['window_id']}@{row['history_end_date']} w={row['weight']:.2f}"
            for row in case["top_support"][:3]
        )
        lines.extend(
            [
                f"### {case['label']}",
                "",
                f"- Terminal median risk P&L: `{stats['terminal_p50']:.3f}`",
                f"- 95% loss VaR-style number: `{stats['terminal_var95_loss']:.3f}`",
                f"- 95% expected shortfall-style loss: `{stats['terminal_expected_shortfall95_loss']:.3f}`",
                f"- Worst generated terminal P&L: `{stats['worst_terminal_pnl']:.3f}`",
                f"- Largest tail contributors: {contributors}",
                f"- Top support: {support_text}",
                "",
            ]
        )
    spread = summary["cross_narrative_range"]
    lines.extend(
        [
            "## Cross-Narrative Spread",
            "",
            f"- Terminal median P&L range: `{spread['terminal_p50_range']:.3f}`",
            f"- 95% loss range: `{spread['var95_loss_range']:.3f}`",
            "",
        ]
    )
    return "\n".join(lines)


def plot_portfolio_impact(
    cases: list[dict[str, Any]],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.2, 1.0])
    ax_fan = fig.add_subplot(gs[0, :])
    ax_var = fig.add_subplot(gs[1, 0])
    ax_contrib = fig.add_subplot(gs[1, 1])
    fig.suptitle(
        "Narrative-conditioned portfolio risk impact at the same accepted start",
        fontsize=14,
        fontweight="bold",
    )

    days = np.arange(0, 31)
    terminal_rows = []
    for case in cases:
        pnl = _portfolio_pnl(case["states"], case["start"])
        pnl_display = _portfolio_pnl_with_start(case["states"], case["start"])
        q10, q50, q90, mean = _quantiles(pnl_display)
        color = str(case["color"])
        ax_fan.fill_between(days, q10, q90, color=color, alpha=0.11)
        ax_fan.plot(days, q50, color=color, linewidth=2.2, label=str(case["label"]))
        ax_fan.plot(days, mean, color=color, linewidth=1.1, linestyle="--", alpha=0.65)
        picks = np.linspace(
            0, pnl_display.shape[0] - 1, min(3, pnl_display.shape[0]), dtype=int
        )
        for pick in picks:
            ax_fan.plot(days, pnl_display[pick], color=color, linewidth=0.7, alpha=0.35)
        stats = _portfolio_stats(pnl)
        terminal_rows.append((str(case["label"]), color, stats))
    ax_fan.axhline(0, color="#78909C", linewidth=0.9)
    ax_fan.set_ylabel("Exposure-weighted risk P&L units")
    ax_fan.set_xlabel("Forward day")
    ax_fan.grid(alpha=0.15)
    ax_fan.legend(fontsize=8, ncol=3, loc="best")
    ax_fan.text(
        0.01,
        0.03,
        "Illustrative normalized exposure vector, not a priced portfolio. "
        "Dashed lines show mean paths; bands are 10-90%.",
        transform=ax_fan.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )

    y_pos = np.arange(len(terminal_rows))
    for y, (label, color, stats) in zip(y_pos, terminal_rows, strict=True):
        ax_var.plot(
            [stats["terminal_p10"], stats["terminal_p90"]],
            [y, y],
            color=color,
            linewidth=5,
            alpha=0.25,
        )
        ax_var.scatter([stats["terminal_p50"]], [y], color=color, s=48, zorder=3)
        ax_var.scatter(
            [-stats["terminal_var95_loss"]],
            [y],
            marker="x",
            color="#263238",
            s=42,
            zorder=4,
        )
    ax_var.axvline(0, color="#78909C", linewidth=0.9)
    ax_var.set_yticks(y_pos)
    ax_var.set_yticklabels([row[0] for row in terminal_rows], fontsize=8)
    ax_var.invert_yaxis()
    ax_var.set_xlabel("Terminal P&L risk units")
    ax_var.set_title(
        "Terminal distribution and 95% loss marker", fontsize=10, fontweight="bold"
    )
    ax_var.grid(axis="x", alpha=0.18)

    contributor_map: dict[str, float] = {}
    for case in cases:
        for row in _portfolio_contributions(case, top_n=8):
            contributor_map[row["market"]] = contributor_map.get(
                row["market"], 0.0
            ) + abs(float(row["terminal_tail_contribution"]))
    top_markets = [
        market
        for market, _score in sorted(
            contributor_map.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    ]
    heat = []
    for case in cases:
        row = []
        for market in top_markets:
            exposure = next(
                item for item in PORTFOLIO_EXPOSURES if item["market"] == market
            )
            component = _portfolio_component_pnl(
                case["states"], case["start"], exposure
            )
            row.append(float(np.percentile(component[:, -1], 5)))
        heat.append(row)
    heat_arr = np.asarray(heat, dtype=np.float64)
    vmax = max(float(np.max(np.abs(heat_arr))), 1e-6)
    im = ax_contrib.imshow(
        heat_arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax
    )
    ax_contrib.set_yticks(np.arange(len(cases)))
    ax_contrib.set_yticklabels([str(case["label"]) for case in cases], fontsize=8)
    ax_contrib.set_xticks(np.arange(len(top_markets)))
    ax_contrib.set_xticklabels(top_markets, rotation=45, ha="right", fontsize=8)
    ax_contrib.set_title(
        "5th-percentile terminal contribution by factor", fontsize=10, fontweight="bold"
    )
    for i in range(heat_arr.shape[0]):
        for j in range(heat_arr.shape[1]):
            ax_contrib.text(
                j,
                i,
                f"{heat_arr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(im, ax=ax_contrib, fraction=0.046, pad=0.04)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--control-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument(
        "--posterior-mode",
        choices=["full", "top1", "top2_80", "top3_90"],
        default=DEFAULT_POSTERIOR_MODE,
    )
    parser.add_argument("--start-index", type=int, default=22)
    parser.add_argument("--direct-sni-samples", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-direct-sni", action="store_true")
    parser.add_argument("--casebook-output", default=str(CASEBOOK_FIGURE))
    parser.add_argument("--backtest-output", default=str(BACKTEST_FIGURE))
    parser.add_argument(
        "--spx-conditionality-output", default=str(SPX_CONDITIONALITY_FIGURE)
    )
    parser.add_argument(
        "--factor-conditionality-output",
        default=str(FACTOR_CONDITIONALITY_FIGURE),
    )
    parser.add_argument(
        "--qualitative-summary-json", default=str(QUALITATIVE_SUMMARY_JSON)
    )
    parser.add_argument("--qualitative-summary-md", default=str(QUALITATIVE_SUMMARY_MD))
    parser.add_argument(
        "--portfolio-impact-output", default=str(PORTFOLIO_IMPACT_FIGURE)
    )
    parser.add_argument("--portfolio-impact-json", default=str(PORTFOLIO_IMPACT_JSON))
    parser.add_argument("--portfolio-impact-md", default=str(PORTFOLIO_IMPACT_MD))
    args = parser.parse_args()

    cases = _load_casebook_cases(
        control_root=args.control_root,
        variant_dir=str(args.variant_dir),
        posterior_mode=str(args.posterior_mode),
    )
    contrast_cases = _load_contrast_cases(
        control_root=args.control_root,
        variant_dir=str(args.variant_dir),
        posterior_mode=str(args.posterior_mode),
    )
    conversion = validate_case_conversion(cases)
    direct = _load_or_sample_direct_sni(
        start_index=int(args.start_index),
        samples=int(args.direct_sni_samples),
        checkpoint=str(args.checkpoint),
        bridge_report=str(args.bridge_report),
        device_name=str(args.device),
        refresh=bool(args.refresh_direct_sni),
    )
    # The casebook compares counterfactual narrative-conditioned distributions
    # under the same accepted start. A realized historical future belongs in the
    # historical backtest figure, not in this cross-narrative casebook.
    plot_casebook(cases, Path(args.casebook_output), realized_future=None)
    plot_backtest(cases, direct, Path(args.backtest_output))
    plot_spx_conditionality(
        contrast_cases,
        direct,
        Path(args.spx_conditionality_output),
    )
    plot_factor_conditionality(
        contrast_cases,
        direct,
        Path(args.factor_conditionality_output),
    )
    plot_portfolio_impact(contrast_cases, Path(args.portfolio_impact_output))
    shape_diagnostic = narrative_distribution_shape_diagnostic(contrast_cases)
    qualitative_summary = build_qualitative_casebook_summary(
        cases,
        contrast_cases,
        direct,
    )
    portfolio_summary = build_portfolio_impact_summary(contrast_cases)
    qualitative_json = Path(args.qualitative_summary_json)
    qualitative_md = Path(args.qualitative_summary_md)
    portfolio_json = Path(args.portfolio_impact_json)
    portfolio_md = Path(args.portfolio_impact_md)
    qualitative_json.parent.mkdir(parents=True, exist_ok=True)
    qualitative_json.write_text(
        json.dumps(qualitative_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualitative_md.write_text(
        render_qualitative_casebook_markdown(qualitative_summary),
        encoding="utf-8",
    )
    portfolio_json.parent.mkdir(parents=True, exist_ok=True)
    portfolio_json.write_text(
        json.dumps(portfolio_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    portfolio_md.write_text(
        render_portfolio_impact_markdown(portfolio_summary),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "casebook_figure": str(args.casebook_output),
                "backtest_figure": str(args.backtest_output),
                "spx_conditionality_figure": str(args.spx_conditionality_output),
                "factor_conditionality_figure": str(args.factor_conditionality_output),
                "portfolio_impact_figure": str(args.portfolio_impact_output),
                "control_root": str(args.control_root),
                "variant_dir": str(args.variant_dir),
                "posterior_mode": str(args.posterior_mode),
                "direct_sni_cache": str(BACKTEST_CACHE),
                "conversion_checks": conversion,
                "shape_diagnostic": shape_diagnostic,
                "qualitative_summary_json": str(qualitative_json),
                "qualitative_summary_md": str(qualitative_md),
                "portfolio_impact_json": str(portfolio_json),
                "portfolio_impact_md": str(portfolio_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
