#!/usr/bin/env python
"""IV level-aware conditionality case study.

This diagnostic compares the SNI IV-only generator with a level-unaware
bootstrap on a normal starting IV state and an elevated starting IV state for a
single structured-surface cell. The goal is a visually clean mean-reversion
exhibit, not a replacement for the aggregate validation tables.
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
from experiments.backfill.block_ar.visualize_norminnov_management_compare import (  # noqa: E402
    GEN_COLOR,
    GT_COLOR,
    PATH_COLORS,
    generate_scope_data,
)
import experiments.backfill.block_ar.visualize_management_report as v1  # noqa: E402


DEFAULT_CHECKPOINT = (
    "models/backfill/674a_iv_channel_level_alltrain_w005_e3_s6731/best_model.pt"
)
DEFAULT_OUT = "results/block_ar/iv_level_conditional_case_study"
DATA_SETTING = "iv_only"


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
    parser.add_argument("--row", type=int, default=2, help="0-based maturity row")
    parser.add_argument("--col", type=int, default=4, help="0-based moneyness column")
    return parser.parse_args()


def _cell_label(row: int, col: int) -> str:
    return f"{v1.MATURITY_LABELS[row]} K={v1.MONEYNESS_LABELS[col]}"


def _coverage_1d(future: np.ndarray, samples: np.ndarray) -> np.ndarray:
    q05 = np.percentile(samples, 5, axis=1)
    q95 = np.percentile(samples, 95, axis=1)
    return np.mean((future >= q05) & (future <= q95), axis=1)


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    rank = np.empty_like(values, dtype=np.float64)
    if len(values) <= 1:
        rank[:] = 0.5
    else:
        rank[order] = np.linspace(0.0, 1.0, len(values))
    return rank


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / scale


def _train_iv_reference(
    payload: dict[str, Any],
    *,
    row: int,
    col: int,
) -> tuple[np.ndarray, np.ndarray]:
    panel, _columns, _dates = load_aligned_iv_factor_panel()
    cfg = payload["config"]
    max_train_idx = 4511 - int(cfg["history_len"]) - int(cfg["future_len"])
    train_level_end = max_train_idx - 441 + int(cfg["history_len"]) + int(cfg["future_len"])
    cell_idx = int(row) * 5 + int(col)
    train_levels = panel[:train_level_end, cell_idx].astype(np.float64)
    train_levels = train_levels[np.isfinite(train_levels)]
    train_delta = np.diff(train_levels).astype(np.float64)
    return train_levels, train_delta


def _train_iv_surface_reference(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    panel, _columns, _dates = load_aligned_iv_factor_panel()
    cfg = payload["config"]
    max_train_idx = 4511 - int(cfg["history_len"]) - int(cfg["future_len"])
    train_level_end = max_train_idx - 441 + int(cfg["history_len"]) + int(cfg["future_len"])
    train_surface = panel[:train_level_end, :25].astype(np.float64).reshape(-1, 5, 5)
    train_delta = np.diff(train_surface, axis=0).astype(np.float64)
    return train_surface, train_delta


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


def _bootstrap_iv_cell_paths(
    *,
    start_level: float,
    train_delta: np.ndarray,
    n_samples: int,
    horizon: int,
    seed: int,
    sampled_transition_indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sampled_transition_indices is None:
        sampled_transition_indices = rng.integers(
            0,
            len(train_delta),
            size=(int(n_samples), int(horizon)),
            endpoint=False,
        )
    current = np.full(int(n_samples), float(start_level), dtype=np.float64)
    paths = np.empty((int(n_samples), int(horizon)), dtype=np.float64)
    for step in range(int(horizon)):
        current = np.maximum(current + train_delta[sampled_transition_indices[:, step]], 1e-4)
        paths[:, step] = current
    return paths.astype(np.float32)


def _bootstrap_iv_surface_paths(
    *,
    start_surface: np.ndarray,
    train_delta: np.ndarray,
    n_samples: int,
    horizon: int,
    seed: int,
    sampled_transition_indices: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sampled_transition_indices is None:
        sampled_transition_indices = rng.integers(
            0,
            train_delta.shape[0],
            size=(int(n_samples), int(horizon)),
            endpoint=False,
        )
    current = np.asarray(start_surface, dtype=np.float64).reshape(1, 5, 5).repeat(int(n_samples), axis=0)
    paths = np.empty((int(n_samples), int(horizon), 5, 5), dtype=np.float64)
    for step in range(int(horizon)):
        current = np.maximum(current + train_delta[sampled_transition_indices[:, step]], 1e-4)
        paths[:, step] = current
    return paths.astype(np.float32)


def _pick_visible_quantile_paths(
    samples: np.ndarray,
    *,
    n_visible: int,
    max_quantile_cap: float | None = 0.90,
) -> np.ndarray:
    n_keep = min(int(n_visible), int(samples.shape[0]))
    if n_keep <= 0:
        return np.asarray([], dtype=int)
    eligible = np.arange(samples.shape[0])
    if max_quantile_cap is not None and samples.shape[0] > n_keep:
        path_max = np.max(samples, axis=1)
        for cap in [float(max_quantile_cap), 0.95, 1.0]:
            threshold = np.quantile(path_max, cap)
            candidate = np.where(path_max <= threshold)[0]
            if len(candidate) >= n_keep:
                eligible = candidate
                break
    terminal = np.asarray(samples[eligible, -1], dtype=np.float64)
    targets = np.linspace(0.08, 0.92, n_keep)
    selected: list[int] = []
    for q in np.quantile(terminal, targets):
        for idx in np.argsort(np.abs(terminal - q)):
            candidate = int(eligible[idx])
            if candidate not in selected:
                selected.append(candidate)
                break
    return np.asarray(selected, dtype=int)


def _pick_bootstrap_visible_paths(samples: np.ndarray, *, n_visible: int) -> np.ndarray:
    n_keep = min(int(n_visible), int(samples.shape[0]))
    if n_keep <= 0:
        return np.asarray([], dtype=int)
    terminal = np.asarray(samples[:, -1], dtype=np.float64)
    max_level = np.max(samples, axis=1)
    min_level = np.min(samples, axis=1)
    selected: list[int] = []
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
    for candidate in _pick_visible_quantile_paths(samples, n_visible=n_keep, max_quantile_cap=None):
        if int(candidate) not in selected:
            selected.append(int(candidate))
        if len(selected) >= n_keep:
            break
    return np.asarray(selected, dtype=int)


def _select_iv_cases(
    *,
    history: np.ndarray,
    future: np.ndarray,
    samples: np.ndarray,
    train_levels: np.ndarray,
    coverage: np.ndarray,
) -> dict[str, dict[str, Any]]:
    current = history[:, -1]
    train_pct = np.asarray([np.mean(train_levels <= x) for x in current], dtype=np.float64)
    q50 = np.percentile(samples, 50, axis=1)
    terminal_delta = q50[:, -1] - current
    future_peak = np.max(future, axis=1)
    future_end = future[:, -1]
    realized_reversion = future_peak - future_end
    history_peak_ratio = current / np.maximum(np.max(history, axis=1), 1e-8)
    generated_peak = np.max(q50, axis=1)
    generated_reversion = generated_peak - q50[:, -1]

    normal_candidates = np.where(
        (train_pct >= 0.25)
        & (train_pct <= 0.55)
        & (coverage >= 0.75)
    )[0]
    if normal_candidates.size == 0:
        normal_candidates = np.where((train_pct >= 0.15) & (train_pct <= 0.65) & (coverage >= 0.60))[0]
    if normal_candidates.size == 0:
        normal_candidates = np.arange(len(current))
    normal_score = (
        -np.abs(train_pct[normal_candidates] - 0.40)
        + 0.50 * coverage[normal_candidates]
        - 0.25 * np.abs(_standardize(terminal_delta[normal_candidates]))
    )
    normal_idx = int(normal_candidates[int(np.argmax(normal_score))])

    elevated_candidates = np.where(
        (train_pct >= 0.80)
        & (history_peak_ratio >= 0.80)
        & (terminal_delta < 0.0)
        & (coverage >= 0.60)
    )[0]
    if elevated_candidates.size == 0:
        elevated_candidates = np.where((train_pct >= 0.75) & (terminal_delta < 0.0))[0]
    if elevated_candidates.size == 0:
        elevated_candidates = np.argsort(current)[-max(1, min(30, len(current))):]
    elevated_score = (
        1.50 * train_pct[elevated_candidates]
        + 0.75 * coverage[elevated_candidates]
        + 0.60 * history_peak_ratio[elevated_candidates]
        + 0.60 * _standardize(-terminal_delta[elevated_candidates])
        + 0.35 * _standardize(generated_reversion[elevated_candidates])
        + 0.25 * _standardize(realized_reversion[elevated_candidates])
    )
    elevated_idx = int(elevated_candidates[int(np.argmax(elevated_score))])

    return {
        "normal": {
            "window_index": normal_idx,
            "selection_rule": (
                "normal IV: train-percentile near 40%, high generated 90% coverage, "
                "and no forced directional drift"
            ),
        },
        "elevated": {
            "window_index": elevated_idx,
            "selection_rule": (
                "peak IV: high train-percentile, starting near the local history peak, "
                "generated median reverts downward, and realized/generated reversion are visually present"
            ),
        },
    }


def _quantiles(samples: np.ndarray) -> dict[str, np.ndarray]:
    labels = ["q05", "q25", "q50", "q75", "q95"]
    values = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)
    return {label: values[i] for i, label in enumerate(labels)}


def _plot_iv_case(
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
    train_median: float,
    train_q25: float,
    train_q75: float,
) -> None:
    hist_days = np.arange(-29, 1)
    fut_days = np.arange(1, 31)
    qs = _quantiles(samples)
    start = float(history[-1])
    terminal_delta = float(qs["q50"][-1] - start)
    peak_to_end = float(np.max(qs["q50"]) - qs["q50"][-1])

    ax.axhspan(train_q25, train_q75, color="gray", alpha=0.075, label="Training IV p25-p75 band", zorder=0)
    ax.axhline(train_median, color="gray", linestyle="--", linewidth=1.0, alpha=0.75, label="Training median IV")
    ax.axhline(start, color="black", linestyle=":", linewidth=1.2, alpha=0.75, label="Start IV level")
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
        f"start={start:.3f} (p{current_pct:.0%})\n"
        f"GT in 90%={coverage:.0%}\n"
        f"median h30 Δ={terminal_delta:+.3f}\n"
        f"median peak→end={peak_to_end:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86),
    )
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Day")
    ax.set_ylabel("IV level")


def _write_iv_cell_csv(
    path: Path,
    *,
    cases: dict[str, dict[str, Any]],
    histories: dict[str, np.ndarray],
    futures: dict[str, np.ndarray],
    sni: dict[str, np.ndarray],
    bootstrap: dict[str, np.ndarray],
    history_dates: dict[str, list[str]],
    future_dates: dict[str, list[str]],
    cell_label: str,
    n_paths: int,
) -> None:
    fieldnames = [
        "case",
        "data_setting",
        "method",
        "series_type",
        "path_id",
        "cell",
        "day",
        "date",
        "iv_level",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case_name in cases:
            for t, day in enumerate(range(-29, 1)):
                writer.writerow(
                    {
                        "case": case_name,
                        "data_setting": DATA_SETTING,
                        "method": "Observed",
                        "series_type": "condition_history",
                        "path_id": "history",
                        "cell": cell_label,
                        "day": day,
                        "date": history_dates[case_name][t],
                        "iv_level": float(histories[case_name][t]),
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
                        "cell": cell_label,
                        "day": day,
                        "date": future_dates[case_name][t],
                        "iv_level": float(futures[case_name][t]),
                    }
                )
            for method_name, arr in [("SNI", sni[case_name]), ("Bootstrap", bootstrap[case_name])]:
                keep = min(int(n_paths), int(arr.shape[0]))
                for path_idx in range(keep):
                    for t, day in enumerate(range(1, 31)):
                        writer.writerow(
                            {
                                "case": case_name,
                                "data_setting": DATA_SETTING,
                                "method": method_name,
                                "series_type": "generated_future",
                                "path_id": f"path_{path_idx + 1:03d}",
                                "cell": cell_label,
                                "day": day,
                                "date": future_dates[case_name][t],
                                "iv_level": float(arr[path_idx, t]),
                            }
                        )


def _surface_column_names() -> list[str]:
    columns: list[str] = []
    for r in range(5):
        maturity = str(v1.MATURITY_LABELS[r]).replace(" ", "")
        for c in range(5):
            strike = str(v1.MONEYNESS_LABELS[c])
            columns.append(f"iv_{maturity}_K{strike}")
    return columns


def _surface_row_values(surface: np.ndarray) -> dict[str, float]:
    flat = np.asarray(surface, dtype=np.float64).reshape(25)
    return {name: float(flat[i]) for i, name in enumerate(_surface_column_names())}


def _write_iv_surface_csv(
    path: Path,
    *,
    cases: dict[str, dict[str, Any]],
    data: dict[str, Any],
    bootstrap: dict[str, np.ndarray],
    history_dates: dict[str, list[str]],
    future_dates: dict[str, list[str]],
    n_paths: int,
) -> None:
    surface_columns = _surface_column_names()
    fieldnames = [
        "case",
        "data_setting",
        "method",
        "series_type",
        "path_id",
        "day",
        "date",
    ] + surface_columns
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, meta in cases.items():
            idx = int(meta["window_index"])
            history_surface = data["history"][idx]
            future_surface = data["future"][idx]
            sni_surface = data["samples"][idx]
            boot_surface = bootstrap[case_name]
            keep_sni = min(int(n_paths), int(sni_surface.shape[0]))
            keep_boot = min(int(n_paths), int(boot_surface.shape[0]))

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
                        **_surface_row_values(history_surface[t]),
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
                        **_surface_row_values(future_surface[t]),
                    }
                )
            for path_idx in range(keep_sni):
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
                            **_surface_row_values(sni_surface[path_idx, t]),
                        }
                    )
            for path_idx in range(keep_boot):
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
                            **_surface_row_values(boot_surface[path_idx, t]),
                        }
                    )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = int(args.row)
    col = int(args.col)
    if not (0 <= row < 5 and 0 <= col < 5):
        raise ValueError("--row and --col must be 0-based indices in [0, 4]")

    mgmt_args = argparse.Namespace(
        output_root=str(out),
        model_label="iv_level_condition_case",
        iv_checkpoint=args.checkpoint,
        anchor_checkpoint="",
        n_samples=int(args.n_samples),
        max_windows=int(args.max_windows),
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        device=args.device,
        seed=int(args.seed),
        skip_iv=False,
        skip_anchor=True,
    )
    torch.manual_seed(int(args.seed))
    data = generate_scope_data(args.checkpoint, "iv_only", mgmt_args)
    payload = data["payload"]
    train_levels, _train_delta = _train_iv_reference(payload, row=row, col=col)
    _train_surface, train_surface_delta = _train_iv_surface_reference(payload)
    train_q25, train_median, train_q75 = np.percentile(train_levels, [25, 50, 75])

    history = data["history"][:, :, row, col].astype(np.float64)
    future = data["future"][:, :, row, col].astype(np.float64)
    sni_samples = data["samples"][:, :, :, row, col].astype(np.float64)
    sni_cov = _coverage_1d(future, sni_samples)
    cases = _select_iv_cases(
        history=history,
        future=future,
        samples=sni_samples,
        train_levels=train_levels,
        coverage=sni_cov,
    )

    val_indices, date_labels = _validation_window_dates(payload, int(data["n_windows"]))
    histories: dict[str, np.ndarray] = {}
    futures: dict[str, np.ndarray] = {}
    sni_cases: dict[str, np.ndarray] = {}
    boot_cases: dict[str, np.ndarray] = {}
    boot_surface_cases: dict[str, np.ndarray] = {}
    history_dates: dict[str, list[str]] = {}
    future_dates: dict[str, list[str]] = {}
    common_boot_seed = int(args.seed) + 23_000

    for case_name, meta in cases.items():
        idx = int(meta["window_index"])
        histories[case_name] = history[idx]
        futures[case_name] = future[idx]
        sni_cases[case_name] = sni_samples[idx]
        boot_surface_cases[case_name] = _bootstrap_iv_surface_paths(
            start_surface=data["history"][idx, -1],
            train_delta=train_surface_delta,
            n_samples=int(args.n_samples),
            horizon=30,
            seed=common_boot_seed,
        )
        boot_cases[case_name] = boot_surface_cases[case_name][:, :, row, col]
        start_idx = int(val_indices[idx])
        history_dates[case_name] = date_labels[start_idx : start_idx + 30]
        future_dates[case_name] = date_labels[start_idx + 30 : start_idx + 60]

        train_pct = float(np.mean(train_levels <= history[idx, -1]))
        boot_cov = float(
            np.mean(
                (future[idx] >= np.percentile(boot_cases[case_name], 5, axis=0))
                & (future[idx] <= np.percentile(boot_cases[case_name], 95, axis=0))
            )
        )
        meta.update(
            {
                "checkpoint": args.checkpoint,
                "data_setting": DATA_SETTING,
                "cell": _cell_label(row, col),
                "row": row,
                "col": col,
                "current_iv": float(history[idx, -1]),
                "train_level_percentile": train_pct,
                "history_peak_ratio": float(history[idx, -1] / max(float(np.max(history[idx])), 1e-8)),
                "sni_90pct_coverage": float(sni_cov[idx]),
                "bootstrap_90pct_coverage": boot_cov,
                "sni_visible_path_indices": [
                    int(x) for x in _pick_visible_quantile_paths(sni_cases[case_name], n_visible=args.visible_paths)
                ],
                "bootstrap_visible_path_indices": [
                    int(x) for x in _pick_bootstrap_visible_paths(boot_cases[case_name], n_visible=args.visible_paths)
                ],
                "sni_terminal_median_change": float(
                    np.percentile(sni_cases[case_name], 50, axis=0)[-1] - history[idx, -1]
                ),
                "bootstrap_terminal_median_change": float(
                    np.percentile(boot_cases[case_name], 50, axis=0)[-1] - history[idx, -1]
                ),
                "history_start_date": history_dates[case_name][0],
                "history_end_date": history_dates[case_name][-1],
                "future_start_date": future_dates[case_name][0],
                "future_end_date": future_dates[case_name][-1],
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.2), sharex=True, sharey=True)
    plot_order = [("normal", "Normal IV"), ("elevated", "Peak IV")]
    for row_idx, (case_name, case_title) in enumerate(plot_order):
        meta = cases[case_name]
        _plot_iv_case(
            axes[row_idx, 0],
            title=f"{case_title}: SNI conditional generator",
            history=histories[case_name],
            future=futures[case_name],
            samples=sni_cases[case_name],
            visible_idx=np.asarray(meta["sni_visible_path_indices"], dtype=int),
            color=GEN_COLOR,
            coverage=float(meta["sni_90pct_coverage"]),
            current_pct=float(meta["train_level_percentile"]),
            train_median=float(train_median),
            train_q25=float(train_q25),
            train_q75=float(train_q75),
        )
        _plot_iv_case(
            axes[row_idx, 1],
            title=f"{case_title}: level-unaware bootstrap",
            history=histories[case_name],
            future=futures[case_name],
            samples=boot_cases[case_name],
            visible_idx=np.asarray(meta["bootstrap_visible_path_indices"], dtype=int),
            color="#546E7A",
            coverage=float(meta["bootstrap_90pct_coverage"]),
            current_pct=float(meta["train_level_percentile"]),
            train_median=float(train_median),
            train_q25=float(train_q25),
            train_q75=float(train_q75),
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(
        f"IV Level-Aware Conditionality: Normal vs Peak Starting State ({_cell_label(row, col)})",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        "Data setting: IV-only surface; bootstrap samples unconditional raw IV deltas for the same cell.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.91])
    png_path = out / "fig_iv_level_conditional_sni_vs_bootstrap.png"
    pdf_path = out / "fig_iv_level_conditional_sni_vs_bootstrap.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    selection_path = out / "iv_level_conditional_selection.json"
    selection_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    csv_path = out / "iv_level_conditional_cell_raw_levels_100paths.csv"
    _write_iv_cell_csv(
        csv_path,
        cases=cases,
        histories=histories,
        futures=futures,
        sni=sni_cases,
        bootstrap=boot_cases,
        history_dates=history_dates,
        future_dates=future_dates,
        cell_label=_cell_label(row, col),
        n_paths=int(args.raw_export_paths),
    )
    surface_csv_path = out / "iv_level_conditional_surface_raw_levels_100paths.csv"
    _write_iv_surface_csv(
        surface_csv_path,
        cases=cases,
        data=data,
        bootstrap=boot_surface_cases,
        history_dates=history_dates,
        future_dates=future_dates,
        n_paths=int(args.raw_export_paths),
    )
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")
    print(f"Saved selection: {selection_path}")
    print(f"Saved raw IV cell CSV: {csv_path}")
    print(f"Saved raw IV surface CSV: {surface_csv_path}")


if __name__ == "__main__":
    main()
