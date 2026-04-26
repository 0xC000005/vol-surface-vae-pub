from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


LOCAL_FACTOR_KEYS = ("ret", "price", "slopes", "skews", "levels")


@dataclass(frozen=True)
class LocalFactorHistoryBlock:
    history_01: torch.Tensor
    future_01: torch.Tensor
    factor_history: torch.Tensor
    indices: np.ndarray
    factor_mean: np.ndarray
    factor_std: np.ndarray
    factor_columns: list[str]


def local_factor_panel_from_mapping(
    raw: Mapping[str, np.ndarray],
    keys: tuple[str, ...] = LOCAL_FACTOR_KEYS,
) -> tuple[np.ndarray, list[str]]:
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"Missing required local factor keys: {missing}")
    columns = list(keys)
    panel = np.stack(
        [np.asarray(raw[key], dtype=np.float32) for key in columns],
        axis=1,
    )
    if panel.ndim != 2:
        raise ValueError("local factor panel must be two-dimensional")
    return panel.astype(np.float32), columns


def factor_histories(
    factor_panel: np.ndarray,
    indices: np.ndarray,
    history_len: int,
) -> np.ndarray:
    panel = np.asarray(factor_panel, dtype=np.float32)
    idx = np.asarray(indices, dtype=np.int64)
    histories = [panel[start : start + int(history_len)] for start in idx]
    out = np.asarray(histories, dtype=np.float32)
    if out.shape != (idx.shape[0], int(history_len), panel.shape[1]):
        raise ValueError(
            "factor histories have an unexpected shape; check indices/history_len"
        )
    return out


def fit_factor_standardizer(
    factor_panel: np.ndarray,
    fit_indices: np.ndarray,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    histories = factor_histories(factor_panel, fit_indices, history_len)
    flat = histories.reshape(-1, histories.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def standardize_factor_histories(
    histories: np.ndarray,
    factor_mean: np.ndarray,
    factor_std: np.ndarray,
) -> np.ndarray:
    return (
        (np.asarray(histories, dtype=np.float32) - factor_mean[None, None, :])
        / factor_std[None, None, :]
    ).astype(np.float32)


def build_local_factor_history_block(
    data_path: str,
    indices: np.ndarray,
    history_len: int,
    future_len: int,
    device: torch.device,
    factor_mean: np.ndarray | None = None,
    factor_std: np.ndarray | None = None,
    fit_indices: np.ndarray | None = None,
) -> LocalFactorHistoryBlock:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    factor_panel, factor_columns = local_factor_panel_from_mapping(raw)
    if factor_panel.shape[0] != surfaces.shape[0]:
        raise ValueError(
            f"Factor/surface length mismatch: {factor_panel.shape[0]} vs {surfaces.shape[0]}"
        )
    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, future_01_flat = build_multistep_windows(
        indices,
        surf_tensor,
        history_len,
        future_len,
    )
    future_01 = future_01_flat.view(history_01.shape[0], future_len, 5, 5)
    if factor_mean is None or factor_std is None:
        if fit_indices is None:
            fit_indices = indices
        factor_mean, factor_std = fit_factor_standardizer(
            factor_panel,
            fit_indices,
            history_len,
        )
    histories = factor_histories(factor_panel, indices, history_len)
    standardized = standardize_factor_histories(histories, factor_mean, factor_std)
    return LocalFactorHistoryBlock(
        history_01=history_01,
        future_01=future_01,
        factor_history=torch.from_numpy(standardized).to(device),
        indices=np.asarray(indices, dtype=np.int64),
        factor_mean=np.asarray(factor_mean, dtype=np.float32),
        factor_std=np.asarray(factor_std, dtype=np.float32),
        factor_columns=factor_columns,
    )


def tensor_dict_summary(block: LocalFactorHistoryBlock) -> dict[str, Any]:
    return {
        "n_windows": int(block.history_01.shape[0]),
        "history_shape": list(block.history_01.shape),
        "future_shape": list(block.future_01.shape),
        "factor_history_shape": list(block.factor_history.shape),
        "index_start": int(block.indices[0]),
        "index_end": int(block.indices[-1]),
        "factor_dim": int(block.factor_history.shape[-1]),
        "factor_columns": list(block.factor_columns),
    }
