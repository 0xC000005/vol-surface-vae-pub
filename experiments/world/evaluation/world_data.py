from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class WorldWindowBatch:
    past_window: np.ndarray
    future_window: np.ndarray
    start_index: np.ndarray
    split: str
    regime_label: np.ndarray | None
    metadata: dict[str, int | str | bool]


def manifest_style_split_indices(
    *,
    n_days: int,
    history_len: int = 30,
    future_len: int = 30,
    test_start: int = 4511,
    val_size: int = 441,
    split: SplitName = "train",
    stride: int = 1,
) -> np.ndarray:
    if history_len <= 0 or future_len <= 0:
        raise ValueError("history_len and future_len must be positive")
    if val_size <= 0:
        raise ValueError("val_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    last_start_exclusive = n_days - history_len - future_len + 1
    max_train_idx = test_start - history_len - future_len
    train_end = max_train_idx - val_size
    if train_end <= 0:
        raise ValueError(
            "Invalid split: test_start/history_len/future_len/val_size leave no train windows"
        )
    if max_train_idx > last_start_exclusive:
        raise ValueError("Invalid split: test_start exceeds available complete windows")

    if split == "train":
        start, end = 0, train_end
    elif split == "val":
        start, end = train_end, max_train_idx
    elif split == "test":
        start, end = max_train_idx, last_start_exclusive
    else:
        raise ValueError(f"Unknown split: {split!r}")

    return np.arange(start, end, stride, dtype=np.int64)


def load_iv_surface_flat(
    data_path: str | Path = "data/vol_surface_with_ret.npz",
    *,
    normalize: bool = True,
) -> np.ndarray:
    data = np.load(Path(data_path))
    if "surface" not in data:
        raise KeyError(f"{data_path} does not contain a 'surface' array")
    surface = np.asarray(data["surface"], dtype=np.float32)
    if surface.ndim != 3 or surface.shape[1:] != (5, 5):
        raise ValueError(f"Expected surface shape (T, 5, 5), got {surface.shape}")
    flat = surface.reshape(surface.shape[0], 25)
    if normalize:
        flat = flat * 2.0 - 1.0
    return flat.astype(np.float32, copy=False)


def _load_regime_labels(
    regime_path: str | Path | None,
    indices: np.ndarray,
    *,
    history_len: int,
    future_len: int,
) -> np.ndarray | None:
    if regime_path is None:
        return None
    path = Path(regime_path)
    if not path.exists():
        return None

    data = np.load(path)
    if "labels" not in data:
        return None
    labels = np.asarray(data["labels"], dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional regime labels, got {labels.shape}")
    if len(indices) == 0:
        return np.empty((0,), dtype=np.int64)
    if int(indices.max()) >= labels.shape[0]:
        return None

    if "history_len" in data and int(data["history_len"]) != history_len:
        return None
    if "future_len" in data and int(data["future_len"]) != future_len:
        return None
    return labels[indices]


def build_iv_world_windows(
    data_path: str | Path = "data/vol_surface_with_ret.npz",
    *,
    split: SplitName = "train",
    history_len: int = 30,
    future_len: int = 30,
    test_start: int = 4511,
    val_size: int = 441,
    max_windows: int | None = None,
    stride: int = 1,
    normalize: bool = True,
    regime_path: str | Path | None = "data/regime_labels.npz",
) -> WorldWindowBatch:
    flat = load_iv_surface_flat(data_path, normalize=normalize)
    indices = manifest_style_split_indices(
        n_days=flat.shape[0],
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
    past = flat[indices[:, None] + hist_offsets[None, :]]
    future = flat[indices[:, None] + fut_offsets[None, :]]
    regime_label = _load_regime_labels(
        regime_path,
        indices,
        history_len=history_len,
        future_len=future_len,
    )

    return WorldWindowBatch(
        past_window=past.astype(np.float32, copy=False),
        future_window=future.astype(np.float32, copy=False),
        start_index=indices,
        split=split,
        regime_label=regime_label,
        metadata={
            "data_path": str(data_path),
            "history_len": int(history_len),
            "future_len": int(future_len),
            "test_start": int(test_start),
            "val_size": int(val_size),
            "normalize": bool(normalize),
        },
    )
