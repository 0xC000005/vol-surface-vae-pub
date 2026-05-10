from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.world.evaluation.masked_multiview_data import (
    GeometryTokenMetadata,
    MaskFamily,
    apply_synthetic_mask,
    load_geometry_panel_values,
    sample_typed_synthetic_mask,
)
from experiments.world.evaluation.world_data import SplitName, manifest_style_split_indices


@dataclass(frozen=True)
class ContextTargetJepaBatch:
    clean_values: np.ndarray
    context_values: np.ndarray
    target_values: np.ndarray
    observed_mask: np.ndarray
    context_mask: np.ndarray
    target_mask: np.ndarray
    absolute_index: np.ndarray
    relative_index: np.ndarray
    token_metadata: GeometryTokenMetadata
    target_family: np.ndarray
    start_index: np.ndarray
    split: str
    metadata: dict[str, int | str | bool]


def _target_values(
    clean_values: np.ndarray,
    observed_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    return np.where(observed_mask & target_mask, clean_values, 0.0).astype(np.float32)


def build_context_target_jepa_batch(
    *,
    surface_path: str | Path = "data/vol_surface_with_ret.npz",
    multi_factor_path: str | Path | None = "data/multi_factor_data.npz",
    split: SplitName = "train",
    history_len: int = 30,
    future_len: int = 30,
    test_start: int = 4511,
    val_size: int = 441,
    max_windows: int | None = None,
    stride: int = 1,
    normalize: bool = True,
    seed: int = 0,
    target_families: tuple[MaskFamily, ...] = (
        "surface_maturity",
        "surface_moneyness",
        "surface_rectangle",
        "vol_side_channel",
        "factor_family",
        "time_block",
    ),
) -> ContextTargetJepaBatch:
    values, observed, token_metadata = load_geometry_panel_values(
        surface_path=surface_path,
        multi_factor_path=multi_factor_path,
        normalize=normalize,
    )
    indices = manifest_style_split_indices(
        n_days=values.shape[0],
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

    offsets = np.arange(history_len, dtype=np.int64)
    absolute_index = indices[:, None] + offsets[None, :]
    clean = values[absolute_index].astype(np.float32)
    observed_window = observed[absolute_index].astype(bool)
    rng = np.random.default_rng(seed)

    context_mask = np.empty_like(observed_window, dtype=bool)
    target_mask = np.empty_like(observed_window, dtype=bool)
    target_family: list[str] = []
    for row in range(clean.shape[0]):
        visible, family = sample_typed_synthetic_mask(
            shape=(history_len, token_metadata.n_tokens),
            token_metadata=token_metadata,
            rng=rng,
            families=target_families,
        )
        context_mask[row] = visible
        target_mask[row] = (~visible) & observed_window[row]
        target_family.append(family)

    return ContextTargetJepaBatch(
        clean_values=clean,
        context_values=apply_synthetic_mask(clean, observed_window, context_mask),
        target_values=_target_values(clean, observed_window, target_mask),
        observed_mask=observed_window,
        context_mask=context_mask,
        target_mask=target_mask,
        absolute_index=absolute_index,
        relative_index=offsets,
        token_metadata=token_metadata,
        target_family=np.asarray(target_family, dtype=object),
        start_index=indices,
        split=str(split),
        metadata={
            "objective_family": "context_to_target_jepa",
            "uses_future_targets": False,
            "history_len": int(history_len),
            "future_len_for_split_only": int(future_len),
            "n_windows": int(clean.shape[0]),
            "n_tokens": int(token_metadata.n_tokens),
        },
    )
