#!/usr/bin/env python
"""VIX level-aware conditionality case study.

This script produces a paper-management diagnostic that compares the proposed
SNI anchor-only generator with a level-unaware bootstrap on two VIX conditions:
normal starting VIX and peak-like high starting VIX. It also exports raw
anchor-factor level paths for the selected conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar._panel_law_535_utils import (  # noqa: E402
    load_aligned_iv_factor_panel,
)
from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    select_rollout_indices,
)
from experiments.backfill.block_ar.audit_576a_unified_increment_panel import (  # noqa: E402
    clean_nonpositive_log_level_factors,
)
from experiments.backfill.block_ar.visualize_norminnov_management_compare import (  # noqa: E402
    GEN_COLOR,
    GT_COLOR,
    PATH_COLORS,
    default_eval_args,
    generate_scope_data,
)


DEFAULT_CHECKPOINT = (
    "models/backfill/734a_anchor_realvix_channel_level_alltrain_w005_e3_s7344/best_model.pt"
)
DEFAULT_OUT = "results/block_ar/vix_level_conditional_case_study"
VIX_NAME = "factor:vix"
SPX_NAME = "factor:spx"
DATA_SETTING = "anchor_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--visible_paths", type=int, default=6)
    parser.add_argument("--raw_export_paths", type=int, default=100)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    rank = np.empty_like(values, dtype=np.float64)
    if len(values) <= 1:
        rank[:] = 0.5
    else:
        rank[order] = np.linspace(0.0, 1.0, len(values))
    return rank


def _coverage(future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    q05 = np.percentile(samples, 5, axis=1)
    q95 = np.percentile(samples, 95, axis=1)
    return np.mean((future >= q05) & (future <= q95), axis=1)


def _train_anchor_bootstrap_reference(
    args: argparse.Namespace,
    payload: dict[str, Any],
    factor_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel, columns, _dates = load_aligned_iv_factor_panel()
    positive_level_policy = payload.get(
        "positive_level_policy",
        payload.get("panel_metadata", {}).get("positive_level_policy", "reference_based"),
    )
    panel, _cleaning_report = clean_nonpositive_log_level_factors(
        panel,
        columns,
        iv_count=25,
        positive_level_policy=positive_level_policy,
    )
    cfg = payload["config"]
    max_train_idx = 4511 - int(cfg["history_len"]) - int(cfg["future_len"])
    train_level_end = max_train_idx - 441 + int(cfg["history_len"]) + int(cfg["future_len"])
    factor_cols = [columns.index(name) for name in factor_names]
    anchor_panel = panel[:train_level_end, factor_cols].astype(np.float64)
    raw_delta = np.diff(anchor_panel, axis=0).astype(np.float64)
    log_delta = np.diff(np.log(np.maximum(anchor_panel, 1e-8)), axis=0).astype(np.float64)
    vix_idx = factor_names.index(VIX_NAME)
    train_vix_levels = anchor_panel[:, vix_idx]
    train_vix_levels = train_vix_levels[np.isfinite(train_vix_levels)]
    return raw_delta, log_delta, train_vix_levels


def _validation_window_dates(payload: dict[str, Any], n_windows: int) -> tuple[np.ndarray, list[str]]:
    _panel, _columns, dates = load_aligned_iv_factor_panel()
    cfg = payload["config"]
    indices = select_rollout_indices(
        test_start=4511,
        val_size=441,
        history_len=int(cfg["history_len"]),
        future_len=int(cfg["future_len"]),
        max_windows=n_windows,
        split="val",
    )
    return indices, [str(d.date()) for d in dates]


def _bootstrap_anchor_panel_paths(
    *,
    start_level: np.ndarray,
    raw_delta: np.ndarray,
    log_delta: np.ndarray,
    factor_names: list[str],
    n_samples: int,
    horizon: int,
    seed: int,
    sampled_transition_indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sampled_transition_indices is None:
        sampled_transition_indices = rng.integers(
            0,
            raw_delta.shape[0],
            size=(int(n_samples), int(horizon)),
            endpoint=False,
        )
    current = np.asarray(start_level, dtype=np.float64).reshape(1, -1).repeat(int(n_samples), axis=0)
    paths = np.empty((int(n_samples), int(horizon), current.shape[1]), dtype=np.float64)
    spx_idx = factor_names.index(SPX_NAME)
    vix_idx = factor_names.index(VIX_NAME)
    for step in range(int(horizon)):
        draw = sampled_transition_indices[:, step]
        next_level = current + raw_delta[draw]
        next_level[:, spx_idx] = current[:, spx_idx] * np.exp(log_delta[draw, spx_idx])
        next_level[:, vix_idx] = current[:, vix_idx] + raw_delta[draw, vix_idx]
        current = np.maximum(next_level, 1e-4)
        paths[:, step, :] = current
    return paths.astype(np.float32)


def _select_cases(
    *,
    history: np.ndarray,
    current: np.ndarray,
    coverage: np.ndarray,
    sni_samples: np.ndarray,
    train_levels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    train_pct = np.asarray([np.mean(train_levels <= x) for x in current], dtype=np.float64)
    q50 = np.percentile(sni_samples, 50, axis=1)
    q95 = np.percentile(sni_samples, 95, axis=1)
    terminal_median_change = q50[:, -1] - current
    upside_tail = np.max(q95, axis=1) - current
    history_peak_ratio = current / np.maximum(np.max(history, axis=1), 1e-8)

    # For presentation, "normal" should not look like a mildly elevated VIX.
    # Target the lower-middle training distribution while avoiding pathological
    # examples where the realized future is completely outside the model band.
    normal = np.where((train_pct >= 0.20) & (train_pct <= 0.40) & (coverage >= 0.70))[0]
    if normal.size == 0:
        normal = np.where((train_pct >= 0.15) & (train_pct <= 0.50) & (coverage >= 0.60))[0]
    if normal.size == 0:
        normal = np.arange(len(current))
    normal_score = -np.abs(train_pct[normal] - 0.30) + 0.25 * coverage[normal]
    normal_idx = int(normal[int(np.argmax(normal_score))])

    # The case study is meant to communicate level-aware conditionality, so the
    # elevated case must be peak-like: high VIX and starting at/near the maximum
    # of the observed conditioning history. We accept lower realized coverage if
    # needed because this figure is a qualitative case study, not a score table.
    elevated = np.where(
        (train_pct >= 0.85)
        & (history_peak_ratio >= 0.95)
        & (terminal_median_change < 0.0)
        & (upside_tail > 0.0)
    )[0]
    if elevated.size == 0:
        elevated = np.where(
            (train_pct >= 0.80)
            & (history_peak_ratio >= 0.90)
            & (terminal_median_change < 0.0)
        )[0]
    if elevated.size == 0:
        elevated = np.argsort(current)[-max(1, min(20, len(current))):]
    elevated_score = (
        2.0 * train_pct[elevated]
        + 0.25 * coverage[elevated]
        + 0.50 * history_peak_ratio[elevated]
    )
    elevated_idx = int(elevated[int(np.argmax(elevated_score))])

    return {
        "normal": {
            "window_index": normal_idx,
            "selection_rule": "normal VIX: train-percentile near 30%, SNI 90% coverage screened when available",
        },
        "elevated": {
            "window_index": elevated_idx,
            "selection_rule": (
                "peak VIX: train-percentile >= 85% when available, current VIX near the local peak, "
                "SNI median reverts below start and still allows upside stress"
            ),
        },
    }


def _pick_visible_paths(samples: np.ndarray, *, seed: int, n_visible: int) -> np.ndarray:
    del seed
    n_keep = min(int(n_visible), int(samples.shape[0]))
    if n_keep <= 0:
        return np.asarray([], dtype=int)
    terminal = samples[:, -1]
    targets = np.linspace(0.08, 0.92, n_keep)
    quantiles = np.quantile(terminal, targets)
    selected: list[int] = []
    for q in quantiles:
        order = np.argsort(np.abs(terminal - q))
        for idx in order:
            candidate = int(idx)
            if candidate not in selected:
                selected.append(candidate)
                break
    return np.asarray(selected, dtype=int)


def _pick_bootstrap_visible_paths(samples: np.ndarray, *, n_visible: int) -> np.ndarray:
    """Pick illustrative bootstrap paths, including high-upside artifacts."""
    n_keep = min(int(n_visible), int(samples.shape[0]))
    if n_keep <= 0:
        return np.asarray([], dtype=int)
    terminal = samples[:, -1]
    max_level = np.max(samples, axis=1)
    min_level = np.min(samples, axis=1)
    selected: list[int] = []

    # Show why the baseline is level-unaware: unconditional raw deltas can move
    # an already-high VIX path even higher because no state-dependent reversion
    # mechanism is learned.
    for candidate in [
        int(np.argmax(max_level)),
        int(np.argmax(terminal)),
        int(np.argmin(terminal)),
        int(np.argmin(min_level)),
    ]:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= n_keep:
            return np.asarray(selected, dtype=int)

    for candidate in _pick_visible_paths(samples, seed=0, n_visible=n_keep):
        if int(candidate) not in selected:
            selected.append(int(candidate))
        if len(selected) >= n_keep:
            break
    return np.asarray(selected, dtype=int)


def _quantiles(samples: np.ndarray) -> dict[str, np.ndarray]:
    labels = ["q05", "q25", "q50", "q75", "q95"]
    values = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)
    return {label: values[i] for i, label in enumerate(labels)}


def _write_long_csv(
    path: Path,
    *,
    cases: dict[str, dict[str, Any]],
    histories: dict[str, np.ndarray],
    futures: dict[str, np.ndarray],
    sni: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
) -> None:
    """Write raw VIX plot inputs only, without diagnostic metadata."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "case",
                "method",
                "series_type",
                "path_id",
                "day",
                "vix_level",
            ]
        )
        for case_name in cases:
            for offset, value in zip(range(-29, 1), histories[case_name], strict=True):
                writer.writerow([case_name, "condition", "history", "history", offset, float(value)])
            for day, value in zip(range(1, 31), futures[case_name], strict=True):
                writer.writerow([case_name, "realized", "realized_future", "ground_truth", day, float(value)])
            for method_name, arr in [("SNI", sni[case_name]), ("Bootstrap", bootstrap[case_name])]:
                for path_idx in range(arr.shape[0]):
                    for day, value in zip(range(1, 31), arr[path_idx], strict=True):
                        writer.writerow(
                            [
                                case_name,
                                method_name,
                                "generated_path",
                                f"path_{path_idx + 1:02d}",
                                day,
                                float(value),
                            ]
                        )


def _write_anchor_raw_level_csv(
    path: Path,
    *,
    cases: dict[str, dict[str, Any]],
    data: dict[str, Any],
    bootstrap_data: dict[str, np.ndarray],
    factor_names: list[str],
    history_dates: dict[str, list[str]],
    future_dates: dict[str, list[str]],
    n_paths: int,
) -> None:
    """Write boss-facing raw level time series for every anchor factor."""
    fieldnames = [
        "case",
        "data_setting",
        "method",
        "series_type",
        "path_id",
        "day",
        "date",
    ] + factor_names
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, meta in cases.items():
            window_idx = int(meta["window_index"])
            history = data["history"][window_idx]
            future = data["future"][window_idx]
            samples = data["samples"][window_idx]
            keep_paths = min(int(n_paths), int(samples.shape[0]))

            for t, day in enumerate(range(-29, 1)):
                writer.writerow(
                    {
                        "case": case_name,
                        "data_setting": DATA_SETTING,
                        "method": "Observed",
                        "series_type": "condition_history",
                        "path_id": "history",
                        "day": day,
                        "date": history_dates[case_name][t],
                        **{name: float(history[t, j]) for j, name in enumerate(factor_names)},
                    }
                )
            for t, day in enumerate(range(1, 31)):
                writer.writerow(
                    {
                        "case": case_name,
                        "data_setting": DATA_SETTING,
                        "method": "Observed",
                        "series_type": "realized_future",
                        "path_id": "ground_truth",
                        "day": day,
                        "date": future_dates[case_name][t],
                        **{name: float(future[t, j]) for j, name in enumerate(factor_names)},
                    }
                )
            for path_idx in range(keep_paths):
                for t, day in enumerate(range(1, 31)):
                    writer.writerow(
                        {
                            "case": case_name,
                            "data_setting": DATA_SETTING,
                            "method": "SNI",
                            "series_type": "generated_future",
                            "path_id": f"path_{path_idx + 1:03d}",
                            "day": day,
                            "date": future_dates[case_name][t],
                            **{name: float(samples[path_idx, t, j]) for j, name in enumerate(factor_names)},
                        }
                    )
            bootstrap_samples = bootstrap_data[case_name]
            keep_bootstrap_paths = min(int(n_paths), int(bootstrap_samples.shape[0]))
            for path_idx in range(keep_bootstrap_paths):
                for t, day in enumerate(range(1, 31)):
                    writer.writerow(
                        {
                            "case": case_name,
                            "data_setting": DATA_SETTING,
                            "method": "Bootstrap",
                            "series_type": "generated_future",
                            "path_id": f"path_{path_idx + 1:03d}",
                            "day": day,
                            "date": future_dates[case_name][t],
                            **{name: float(bootstrap_samples[path_idx, t, j]) for j, name in enumerate(factor_names)},
                        }
                    )


def _plot_case(
    ax: plt.Axes,
    *,
    title: str,
    history: np.ndarray,
    future: np.ndarray,
    samples: np.ndarray,
    visible_idx: np.ndarray,
    color: str,
    coverage: float,
    current_pct: float,
    train_vix_median: float,
    train_vix_q25: float,
    train_vix_q75: float,
) -> None:
    hist_days = np.arange(-29, 1)
    fut_days = np.arange(1, 31)
    qs = _quantiles(samples)
    start_vix = float(history[-1])
    terminal_level = float(qs["q50"][-1])
    terminal_delta = terminal_level - start_vix

    ax.axhspan(
        train_vix_q25,
        train_vix_q75,
        color="gray",
        alpha=0.075,
        label="Training VIX p25-p75 band",
        zorder=0,
    )
    ax.axhline(
        train_vix_median,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="Training median VIX",
    )
    ax.axhline(
        start_vix,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.75,
        label="Start VIX level",
    )
    ax.plot(hist_days, history, color="black", linewidth=1.7, label="History")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.fill_between(fut_days, qs["q05"], qs["q95"], color=color, alpha=0.14, label="Generated 90% band")
    ax.fill_between(fut_days, qs["q25"], qs["q75"], color=color, alpha=0.23, label="Generated IQR")
    for j, path_idx in enumerate(visible_idx):
        ax.plot(
            fut_days,
            samples[path_idx],
            color=PATH_COLORS[j % len(PATH_COLORS)],
            linewidth=1.15,
            alpha=0.86,
            label="Generated sample paths" if j == 0 else None,
        )
    ax.plot(fut_days, qs["q50"], color=color, linewidth=2.0, label="Generated median")
    ax.plot(fut_days, future, color=GT_COLOR, linestyle="--", linewidth=2.1, label="Realized future")
    ax.text(
        0.02,
        0.96,
        f"start={start_vix:.1f} (p{current_pct:.0%})\n"
        f"GT in 90%={coverage:.0%}\n"
        f"median h30 Δ={terminal_delta:+.1f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86),
    )
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Day")
    ax.set_ylabel("VIX level")


def _plot_factor_case(
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str,
    history: np.ndarray,
    future: np.ndarray,
    samples: np.ndarray,
    visible_idx: np.ndarray,
    color: str,
    coverage: float,
    show_coverage: bool = True,
) -> None:
    hist_days = np.arange(-29, 1)
    fut_days = np.arange(1, 31)
    qs = _quantiles(samples)
    start_level = float(history[-1])
    terminal_delta = float(qs["q50"][-1] - start_level)
    ax.axhline(
        start_level,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.75,
        label="Start level",
    )
    ax.plot(hist_days, history, color="black", linewidth=1.7, label="History")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.fill_between(fut_days, qs["q05"], qs["q95"], color=color, alpha=0.14, label="Generated 90% band")
    ax.fill_between(fut_days, qs["q25"], qs["q75"], color=color, alpha=0.23, label="Generated IQR")
    for j, path_idx in enumerate(visible_idx):
        ax.plot(
            fut_days,
            samples[path_idx],
            color=PATH_COLORS[j % len(PATH_COLORS)],
            linewidth=1.15,
            alpha=0.86,
            label="Generated sample paths" if j == 0 else None,
        )
    ax.plot(fut_days, qs["q50"], color=color, linewidth=2.0, label="Generated median")
    ax.plot(fut_days, future, color=GT_COLOR, linestyle="--", linewidth=2.1, label="Realized future")
    text_lines = [
        f"start={start_level:,.0f}",
        f"median h30 Δ={terminal_delta:+,.0f}",
    ]
    if show_coverage:
        text_lines.insert(1, f"GT in 90%={coverage:.0%}")
    ax.text(
        0.02,
        0.96,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86),
    )
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mgmt_args = argparse.Namespace(
        output_root=str(out),
        model_label="vix_level_condition_case",
        iv_checkpoint="",
        anchor_checkpoint=args.checkpoint,
        n_samples=int(args.n_samples),
        max_windows=int(args.max_windows),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=args.device,
        seed=int(args.seed),
        skip_iv=True,
        skip_anchor=False,
    )
    torch.manual_seed(int(args.seed))
    data = generate_scope_data(args.checkpoint, "anchor_only", mgmt_args)
    payload = data["payload"]
    names = data["spec_names"]
    vix_idx = names.index(VIX_NAME)
    spx_idx = names.index(SPX_NAME)

    train_raw_delta, train_log_delta, train_levels = _train_anchor_bootstrap_reference(
        mgmt_args,
        payload,
        names,
    )
    train_vix_q25, train_vix_median, train_vix_q75 = np.percentile(train_levels, [25, 50, 75])
    history_vix = data["history"][:, :, vix_idx].astype(np.float64)
    future_vix = data["future"][:, :, vix_idx].astype(np.float64)
    sni_vix = data["samples"][:, :, :, vix_idx].astype(np.float64)
    current = history_vix[:, -1]
    sni_cov = _coverage(future_vix, sni_vix)
    cases = _select_cases(
        history=history_vix,
        current=current,
        coverage=sni_cov,
        sni_samples=sni_vix,
        train_levels=train_levels,
    )
    val_indices, date_labels = _validation_window_dates(payload, int(data["n_windows"]))

    histories: dict[str, np.ndarray] = {}
    futures: dict[str, np.ndarray] = {}
    sni_cases: dict[str, np.ndarray] = {}
    boot_cases: dict[str, np.ndarray] = {}
    boot_panel_cases: dict[str, np.ndarray] = {}
    history_dates: dict[str, list[str]] = {}
    future_dates: dict[str, list[str]] = {}

    # Use common bootstrap increment draws across normal/elevated cases to make
    # the level-unaware nature visually inspectable.
    common_boot_seed = int(args.seed) + 11_000
    for case_name, meta in cases.items():
        idx = int(meta["window_index"])
        histories[case_name] = history_vix[idx]
        futures[case_name] = future_vix[idx]
        sni_cases[case_name] = sni_vix[idx]
        start_idx = int(val_indices[idx])
        history_dates[case_name] = date_labels[start_idx : start_idx + 30]
        future_dates[case_name] = date_labels[start_idx + 30 : start_idx + 60]
        boot_panel_cases[case_name] = _bootstrap_anchor_panel_paths(
            start_level=data["history"][idx, -1, :],
            raw_delta=train_raw_delta,
            log_delta=train_log_delta,
            factor_names=names,
            n_samples=int(args.n_samples),
            horizon=30,
            seed=common_boot_seed,
        )
        boot_cases[case_name] = boot_panel_cases[case_name][:, :, vix_idx]
        meta["current_vix"] = float(history_vix[idx, -1])
        meta["train_level_percentile"] = float(np.mean(train_levels <= history_vix[idx, -1]))
        meta["history_peak_ratio"] = float(history_vix[idx, -1] / max(float(np.max(history_vix[idx])), 1e-8))
        meta["data_setting"] = DATA_SETTING
        meta["checkpoint"] = args.checkpoint
        meta["sni_90pct_coverage"] = float(sni_cov[idx])
        meta["sni_visible_path_indices"] = [
            int(x) for x in _pick_visible_paths(sni_cases[case_name], seed=int(args.seed) + idx, n_visible=args.visible_paths)
        ]
        meta["bootstrap_visible_path_indices"] = [
            int(x) for x in _pick_bootstrap_visible_paths(boot_cases[case_name], n_visible=args.visible_paths)
        ]
        meta["sni_terminal_median_change"] = float(np.percentile(sni_cases[case_name], 50, axis=0)[-1] - history_vix[idx, -1])
        meta["bootstrap_terminal_median_change"] = float(np.percentile(boot_cases[case_name], 50, axis=0)[-1] - history_vix[idx, -1])

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.2), sharex=True, sharey=True)
    plot_order = [("normal", "Normal VIX"), ("elevated", "Peak VIX")]
    for row, (case_name, case_title) in enumerate(plot_order):
        meta = cases[case_name]
        _plot_case(
            axes[row, 0],
            title=f"{case_title}: SNI conditional generator",
            history=histories[case_name],
            future=futures[case_name],
            samples=sni_cases[case_name],
            visible_idx=np.asarray(meta["sni_visible_path_indices"], dtype=int),
            color=GEN_COLOR,
            coverage=float(meta["sni_90pct_coverage"]),
            current_pct=float(meta["train_level_percentile"]),
            train_vix_median=float(train_vix_median),
            train_vix_q25=float(train_vix_q25),
            train_vix_q75=float(train_vix_q75),
        )
        boot_cov = float(
            np.mean(
                (futures[case_name] >= np.percentile(boot_cases[case_name], 5, axis=0))
                & (futures[case_name] <= np.percentile(boot_cases[case_name], 95, axis=0))
            )
        )
        _plot_case(
            axes[row, 1],
            title=f"{case_title}: level-unaware bootstrap",
            history=histories[case_name],
            future=futures[case_name],
            samples=boot_cases[case_name],
            visible_idx=np.asarray(meta["bootstrap_visible_path_indices"], dtype=int),
            color="#546E7A",
            coverage=boot_cov,
            current_pct=float(meta["train_level_percentile"]),
            train_vix_median=float(train_vix_median),
            train_vix_q25=float(train_vix_q25),
            train_vix_q75=float(train_vix_q75),
        )
        meta["bootstrap_90pct_coverage"] = boot_cov

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(
        "VIX Level-Aware Conditionality: Normal vs Peak Starting State",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Data setting: anchor-only panel with real VIX included; bootstrap samples unconditional raw VIX deltas.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.91])
    png_path = out / "fig_vix_level_conditional_sni_vs_bootstrap.png"
    pdf_path = out / "fig_vix_level_conditional_sni_vs_bootstrap.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    spx_fig, spx_axes = plt.subplots(2, 2, figsize=(16, 9.2), sharex=True, sharey=True)
    for row, (case_name, case_title) in enumerate(plot_order):
        meta = cases[case_name]
        idx = int(meta["window_index"])
        sni_spx = data["samples"][idx, :, :, spx_idx].astype(np.float64)
        boot_spx = boot_panel_cases[case_name][:, :, spx_idx].astype(np.float64)
        future_spx = data["future"][idx, :, spx_idx].astype(np.float64)
        sni_spx_cov = float(
            np.mean(
                (future_spx >= np.percentile(sni_spx, 5, axis=0))
                & (future_spx <= np.percentile(sni_spx, 95, axis=0))
            )
        )
        boot_spx_cov = float(
            np.mean(
                (future_spx >= np.percentile(boot_spx, 5, axis=0))
                & (future_spx <= np.percentile(boot_spx, 95, axis=0))
            )
        )
        _plot_factor_case(
            spx_axes[row, 0],
            title=f"{case_title}: SNI conditional generator",
            ylabel="SPX level",
            history=data["history"][idx, :, spx_idx].astype(np.float64),
            future=future_spx,
            samples=sni_spx,
            visible_idx=np.asarray(meta["sni_visible_path_indices"], dtype=int),
            color=GEN_COLOR,
            coverage=sni_spx_cov,
            show_coverage=False,
        )
        _plot_factor_case(
            spx_axes[row, 1],
            title=f"{case_title}: level-unaware bootstrap",
            ylabel="SPX level",
            history=data["history"][idx, :, spx_idx].astype(np.float64),
            future=future_spx,
            samples=boot_spx,
            visible_idx=np.asarray(meta["bootstrap_visible_path_indices"], dtype=int),
            color="#546E7A",
            coverage=boot_spx_cov,
            show_coverage=False,
        )
        meta["sni_spx_90pct_coverage"] = sni_spx_cov
        meta["bootstrap_spx_90pct_coverage"] = boot_spx_cov

    spx_handles, spx_labels = spx_axes[0, 0].get_legend_handles_labels()
    spx_fig.legend(spx_handles, spx_labels, loc="lower center", ncol=6, fontsize=9)
    spx_fig.suptitle(
        "SPX Raw-Level Companion: Same VIX Conditions, Same Scenario Draw Colors",
        fontsize=16,
        fontweight="bold",
    )
    spx_fig.text(
        0.5,
        0.925,
        "Same normal/peak VIX conditions as the VIX figure; SNI paths are simultaneous raw-level anchor-factor scenarios.",
        ha="center",
        va="center",
        fontsize=9,
    )
    spx_fig.tight_layout(rect=[0, 0.06, 1, 0.91])
    spx_png_path = out / "fig_spx_level_companion_sni_vs_bootstrap.png"
    spx_pdf_path = out / "fig_spx_level_companion_sni_vs_bootstrap.pdf"
    spx_fig.savefig(spx_png_path, dpi=180, bbox_inches="tight")
    spx_fig.savefig(spx_pdf_path, bbox_inches="tight")
    plt.close(spx_fig)

    selection_path = out / "vix_level_conditional_selection.json"
    selection_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    _write_long_csv(
        out / "vix_level_conditional_paths_long.csv",
        cases=cases,
        histories=histories,
        futures=futures,
        sni=sni_cases,
        bootstrap=boot_cases,
    )
    _write_anchor_raw_level_csv(
        out / "vix_level_conditional_anchor_raw_levels_100paths.csv",
        cases=cases,
        data=data,
        bootstrap_data=boot_panel_cases,
        factor_names=names,
        history_dates=history_dates,
        future_dates=future_dates,
        n_paths=int(args.raw_export_paths),
    )

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")
    print(f"Saved plot: {spx_png_path}")
    print(f"Saved plot: {spx_pdf_path}")
    print(f"Saved selection: {selection_path}")
    print(f"Saved raw VIX plot CSV: {out / 'vix_level_conditional_paths_long.csv'}")
    print(f"Saved raw anchor panel CSV: {out / 'vix_level_conditional_anchor_raw_levels_100paths.csv'}")


if __name__ == "__main__":
    main()
