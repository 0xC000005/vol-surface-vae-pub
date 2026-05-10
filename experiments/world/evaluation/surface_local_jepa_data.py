from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.world.evaluation.masked_multiview_data import (
    GeometryTokenMetadata,
    apply_synthetic_mask,
    load_geometry_panel_values,
)
from experiments.world.evaluation.world_data import SplitName, manifest_style_split_indices


SurfaceLocalTargetFamily = Literal[
    "surface_wing_moneyness",
    "surface_edge_maturity",
    "surface_atm_strip",
    "surface_rectangle",
    "factor_family",
]


@dataclass(frozen=True)
class SurfaceLocalJepaBatch:
    clean_values: np.ndarray
    context_values: np.ndarray
    target_values: np.ndarray
    observed_mask: np.ndarray
    context_mask: np.ndarray
    target_mask: np.ndarray
    target_positions: np.ndarray
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


def _surface_tokens(
    token_metadata: GeometryTokenMetadata,
    *,
    moneyness: set[int] | None = None,
    maturity: set[int] | None = None,
) -> np.ndarray:
    geom = token_metadata.geometry_id
    coord = token_metadata.geometry_coord
    mask = geom == "iv_surface"
    if moneyness is not None:
        mask &= np.isin(coord[:, 0].astype(np.int64), sorted(moneyness))
    if maturity is not None:
        mask &= np.isin(coord[:, 1].astype(np.int64), sorted(maturity))
    return np.flatnonzero(mask)


def _contiguous_days(rng: np.random.Generator, history_len: int) -> np.ndarray:
    length = int(rng.integers(1, max(2, history_len // 3) + 1))
    start = int(rng.integers(0, history_len - length + 1))
    return np.arange(start, start + length, dtype=np.int64)


def sample_surface_local_target_mask(
    *,
    shape: tuple[int, int],
    token_metadata: GeometryTokenMetadata,
    rng: np.random.Generator,
    families: tuple[SurfaceLocalTargetFamily, ...],
) -> tuple[np.ndarray, str]:
    history_len, n_tokens = shape
    target = np.zeros((history_len, n_tokens), dtype=bool)
    family = str(rng.choice(np.asarray(families, dtype=object)))

    if family == "surface_wing_moneyness":
        chosen = int(rng.choice(np.asarray([0, 4], dtype=np.int64)))
        target[:, _surface_tokens(token_metadata, moneyness={chosen})] = True
    elif family == "surface_edge_maturity":
        chosen = int(rng.choice(np.asarray([0, 4], dtype=np.int64)))
        target[:, _surface_tokens(token_metadata, maturity={chosen})] = True
    elif family == "surface_atm_strip":
        target[:, _surface_tokens(token_metadata, moneyness={2})] = True
    elif family == "surface_rectangle":
        m0 = int(rng.integers(0, 4))
        t0 = int(rng.integers(0, 4))
        target[
            :,
            _surface_tokens(
                token_metadata,
                moneyness={m0, m0 + 1},
                maturity={t0, t0 + 1},
            ),
        ] = True
    elif family == "factor_family":
        geom = token_metadata.geometry_id
        available = sorted(
            {
                str(x)
                for x, g in zip(token_metadata.factor_family.tolist(), geom.tolist())
                if g in {"factor_level", "factor_return"}
            }
        )
        if available:
            chosen = str(rng.choice(np.asarray(available, dtype=object)))
            token_idx = np.flatnonzero(
                np.isin(geom, ["factor_level", "factor_return"])
                & (token_metadata.factor_family == chosen)
            )
            days = _contiguous_days(rng, history_len)
            target[np.ix_(days, token_idx)] = True
    else:
        raise ValueError(f"unknown surface-local target family: {family!r}")

    if not target.any():
        target[:, _surface_tokens(token_metadata, moneyness={0})] = True
        family = "surface_wing_moneyness"
    return ~target, family


def build_surface_local_jepa_batch(
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
    target_families: tuple[SurfaceLocalTargetFamily, ...] = (
        "surface_wing_moneyness",
        "surface_edge_maturity",
        "surface_atm_strip",
        "surface_rectangle",
        "factor_family",
    ),
) -> SurfaceLocalJepaBatch:
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
        visible, family = sample_surface_local_target_mask(
            shape=(history_len, token_metadata.n_tokens),
            token_metadata=token_metadata,
            rng=rng,
            families=target_families,
        )
        context_mask[row] = visible
        target_mask[row] = (~visible) & observed_window[row]
        target_family.append(family)

    return SurfaceLocalJepaBatch(
        clean_values=clean,
        context_values=apply_synthetic_mask(clean, observed_window, context_mask),
        target_values=_target_values(clean, observed_window, target_mask),
        observed_mask=observed_window,
        context_mask=context_mask,
        target_mask=target_mask,
        target_positions=np.argwhere(target_mask).astype(np.int64),
        absolute_index=absolute_index,
        relative_index=offsets,
        token_metadata=token_metadata,
        target_family=np.asarray(target_family, dtype=object),
        start_index=indices,
        split=str(split),
        metadata={
            "objective_family": "token_geometry_level_context_to_target_jepa",
            "target_representation_surface": "token_geometry",
            "uses_future_targets": False,
            "history_len": int(history_len),
            "future_len_for_split_only": int(future_len),
            "n_windows": int(clean.shape[0]),
            "n_tokens": int(token_metadata.n_tokens),
        },
    )
