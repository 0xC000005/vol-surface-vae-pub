from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.world.evaluation.world_data import (
    SplitName,
    manifest_style_split_indices,
)


FactorColumnFamily = Literal["factor_level", "factor_return"]


@dataclass(frozen=True)
class FactorPanelWindowBatch:
    past_panel: np.ndarray
    future_panel: np.ndarray
    start_index: np.ndarray
    columns: list[str]
    split: str
    metadata: dict[str, int | str | bool]


def _as_column_names(values: np.ndarray, *, prefix: FactorColumnFamily) -> list[str]:
    names = [str(item) for item in values.tolist()]
    return [f"{prefix}:{name}" for name in names]


def _fill_missing_by_column(panel: np.ndarray) -> np.ndarray:
    filled = np.asarray(panel, dtype=np.float32).copy()
    if filled.ndim != 2:
        raise ValueError(f"Expected factor panel shape (T, C), got {filled.shape}")
    for col in range(filled.shape[1]):
        values = filled[:, col]
        finite = np.isfinite(values)
        if finite.all():
            continue
        if not finite.any():
            filled[:, col] = 0.0
            continue
        finite_idx = np.flatnonzero(finite)
        first = int(finite_idx[0])
        last = int(finite_idx[-1])
        values[:first] = values[first]
        for idx in range(first + 1, last + 1):
            if not np.isfinite(values[idx]):
                values[idx] = values[idx - 1]
        values[last + 1 :] = values[last]
        filled[:, col] = values
    return filled


def load_factor_panel_values(
    data_path: str | Path = "data/multi_factor_data.npz",
    *,
    normalize: bool = False,
) -> tuple[np.ndarray, list[str]]:
    data = np.load(Path(data_path))
    required = {"levels", "level_columns", "returns", "return_columns"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{data_path} is missing required arrays: {missing}")

    levels = np.asarray(data["levels"], dtype=np.float32)
    returns = np.asarray(data["returns"], dtype=np.float32)
    if levels.ndim != 2 or returns.ndim != 2:
        raise ValueError("levels and returns must be two-dimensional arrays")
    if levels.shape[0] != returns.shape[0]:
        raise ValueError(
            f"levels/returns length mismatch: {levels.shape[0]} vs {returns.shape[0]}"
        )

    panel = np.concatenate([levels, returns], axis=1).astype(np.float32, copy=False)
    panel = _fill_missing_by_column(panel)
    columns = _as_column_names(data["level_columns"], prefix="factor_level")
    columns += _as_column_names(data["return_columns"], prefix="factor_return")
    if len(columns) != panel.shape[1]:
        raise ValueError(f"column count mismatch: {len(columns)} vs {panel.shape[1]}")

    if normalize:
        mean = panel.mean(axis=0, keepdims=True)
        std = panel.std(axis=0, keepdims=True)
        panel = ((panel - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    return panel.astype(np.float32, copy=False), columns


def build_factor_panel_world_windows(
    data_path: str | Path = "data/multi_factor_data.npz",
    *,
    split: SplitName = "train",
    history_len: int = 30,
    future_len: int = 30,
    test_start: int = 4511,
    val_size: int = 441,
    max_windows: int | None = None,
    stride: int = 1,
    normalize: bool = False,
) -> FactorPanelWindowBatch:
    panel, columns = load_factor_panel_values(data_path, normalize=normalize)
    indices = manifest_style_split_indices(
        n_days=panel.shape[0],
        history_len=history_len,
        future_len=future_len,
        test_start=test_start,
        val_size=val_size,
        split=split,
        stride=stride,
    )
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        indices = indices[:max_windows]

    hist_offsets = np.arange(history_len, dtype=np.int64)
    fut_offsets = history_len + np.arange(future_len, dtype=np.int64)
    past = panel[indices[:, None] + hist_offsets[None, :]]
    future = panel[indices[:, None] + fut_offsets[None, :]]
    return FactorPanelWindowBatch(
        past_panel=past.astype(np.float32, copy=False),
        future_panel=future.astype(np.float32, copy=False),
        start_index=indices,
        columns=columns,
        split=split,
        metadata={
            "data_path": str(data_path),
            "history_len": int(history_len),
            "future_len": int(future_len),
            "test_start": int(test_start),
            "val_size": int(val_size),
            "normalize": bool(normalize),
            "target_scope": "factor_panel_future_downstream_probe_only",
        },
    )


def make_factor_panel_future_targets(
    past_panel: np.ndarray,
    future_panel: np.ndarray,
    *,
    columns: list[str],
) -> dict[str, dict[str, np.ndarray] | dict[str, object]]:
    past = np.asarray(past_panel, dtype=np.float32)
    future = np.asarray(future_panel, dtype=np.float32)
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError("past_panel and future_panel must have shape (N, T, C)")
    if past.shape[0] != future.shape[0] or past.shape[2] != future.shape[2]:
        raise ValueError("past and future must share sample count and channel count")
    if len(columns) != past.shape[2]:
        raise ValueError(f"column count mismatch: {len(columns)} vs {past.shape[2]}")

    last = past[:, -1, :]
    path = np.concatenate([last[:, None, :], future], axis=1)
    step = np.diff(path, axis=1)
    regression = {
        "factor_future_mean_delta": (future.mean(axis=1) - last).astype(np.float32),
        "factor_future_range": (future.max(axis=1) - future.min(axis=1)).astype(
            np.float32
        ),
        "factor_future_terminal_delta": (future[:, -1, :] - last).astype(np.float32),
        "factor_future_max_abs_step": np.max(np.abs(step), axis=1).astype(np.float32),
    }
    return {
        "regression": regression,
        "classification": {},
        "metadata": {
            "target_scope": "factor_panel_future_downstream_probe_only",
            "columns": list(columns),
        },
    }
