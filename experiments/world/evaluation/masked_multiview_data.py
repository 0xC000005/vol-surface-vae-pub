from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.world.evaluation.world_data import SplitName, manifest_style_split_indices


MaskFamily = Literal[
    "surface_maturity",
    "surface_moneyness",
    "surface_rectangle",
    "vol_side_channel",
    "factor_family",
    "time_block",
    "sparse",
]


FACTOR_FAMILIES: dict[str, str] = {
    "spx": "equity_risk",
    "nikkei": "equity_risk",
    "vix": "equity_risk",
    "usdcad": "fx",
    "usdjpy": "fx",
    "dxy": "fx",
    "copper": "commodity",
    "wheat": "commodity",
    "crude_oil": "commodity",
    "gold": "commodity",
    "us2y": "rates",
    "us10y": "rates",
    "aaa_oas": "credit",
    "bbb_oas": "credit",
}


@dataclass(frozen=True)
class GeometryTokenMetadata:
    geometry_id: np.ndarray
    factor_id: np.ndarray
    factor_family: np.ndarray
    geometry_coord: np.ndarray

    @property
    def n_tokens(self) -> int:
        return int(self.geometry_id.shape[0])


@dataclass(frozen=True)
class MaskedMultiviewBatch:
    clean_values: np.ndarray
    view_a_values: np.ndarray
    view_b_values: np.ndarray
    observed_mask: np.ndarray
    synthetic_mask_a: np.ndarray
    synthetic_mask_b: np.ndarray
    absolute_index: np.ndarray
    relative_index: np.ndarray
    positive_index: np.ndarray
    token_metadata: GeometryTokenMetadata
    mask_family_a: np.ndarray
    mask_family_b: np.ndarray
    start_index: np.ndarray
    split: str
    metadata: dict[str, int | str | bool]


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    scale = np.nanstd(arr, axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((arr - mean) / scale).astype(np.float32)


def _base_factor_name(column: str) -> str:
    for suffix in ("_logret", "_diff"):
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def build_geometry_token_metadata(
    *,
    level_columns: list[str] | None = None,
    return_columns: list[str] | None = None,
) -> GeometryTokenMetadata:
    geometry_id: list[str] = []
    factor_id: list[str] = []
    factor_family: list[str] = []
    geometry_coord: list[tuple[float, float]] = []

    for moneyness in range(5):
        for maturity in range(5):
            geometry_id.append("iv_surface")
            factor_id.append(f"iv_m{moneyness}_t{maturity}")
            factor_family.append("surface")
            geometry_coord.append((float(moneyness), float(maturity)))

    for name in ("ret", "price", "slopes", "skews", "levels"):
        geometry_id.append("vol_side_channel")
        factor_id.append(name)
        factor_family.append("vol_summary")
        geometry_coord.append((-1.0, -1.0))

    if level_columns:
        for name in level_columns:
            base = _base_factor_name(str(name))
            geometry_id.append("factor_level")
            factor_id.append(base)
            factor_family.append(FACTOR_FAMILIES.get(base, "other_factor"))
            geometry_coord.append((-1.0, -1.0))

    if return_columns:
        for name in return_columns:
            base = _base_factor_name(str(name))
            geometry_id.append("factor_return")
            factor_id.append(base)
            factor_family.append(FACTOR_FAMILIES.get(base, "other_factor"))
            geometry_coord.append((-1.0, -1.0))

    return GeometryTokenMetadata(
        geometry_id=np.asarray(geometry_id, dtype=object),
        factor_id=np.asarray(factor_id, dtype=object),
        factor_family=np.asarray(factor_family, dtype=object),
        geometry_coord=np.asarray(geometry_coord, dtype=np.float32),
    )


def load_geometry_panel_values(
    *,
    surface_path: str | Path = "data/vol_surface_with_ret.npz",
    multi_factor_path: str | Path | None = "data/multi_factor_data.npz",
    normalize: bool = True,
    align_mode: Literal["tail", "head"] = "tail",
) -> tuple[np.ndarray, np.ndarray, GeometryTokenMetadata]:
    surface_data = np.load(Path(surface_path), allow_pickle=True)
    surface = np.asarray(surface_data["surface"], dtype=np.float32)
    if surface.ndim != 3 or surface.shape[1:] != (5, 5):
        raise ValueError(f"Expected surface shape (T, 5, 5), got {surface.shape}")
    surface_flat = surface.reshape(surface.shape[0], 25)
    if normalize:
        surface_flat = surface_flat * 2.0 - 1.0

    side_names = ("ret", "price", "slopes", "skews", "levels")
    side = np.column_stack([np.asarray(surface_data[name], dtype=np.float32) for name in side_names])
    if normalize:
        side = _standardize_columns(side)

    values_parts = [surface_flat.astype(np.float32), side.astype(np.float32)]
    level_columns: list[str] | None = None
    return_columns: list[str] | None = None
    n_days = surface.shape[0]

    if multi_factor_path is not None and Path(multi_factor_path).exists():
        factor_data = np.load(Path(multi_factor_path), allow_pickle=True)
        levels = np.asarray(factor_data["levels"], dtype=np.float32)
        returns = np.asarray(factor_data["returns"], dtype=np.float32)
        level_columns = [str(x) for x in factor_data["level_columns"].tolist()]
        return_columns = [str(x) for x in factor_data["return_columns"].tolist()]
        n_days = min(n_days, levels.shape[0], returns.shape[0])
        if align_mode == "tail":
            surface_slice = slice(surface.shape[0] - n_days, surface.shape[0])
            factor_slice = slice(levels.shape[0] - n_days, levels.shape[0])
        elif align_mode == "head":
            surface_slice = slice(0, n_days)
            factor_slice = slice(0, n_days)
        else:
            raise ValueError(f"Unknown align_mode: {align_mode!r}")
        values_parts = [
            values_parts[0][surface_slice],
            values_parts[1][surface_slice],
        ]
        level_values = levels[factor_slice]
        return_values = returns[factor_slice]
        if normalize:
            level_values = _standardize_columns(level_values)
            return_values = _standardize_columns(return_values)
        values_parts.extend([level_values.astype(np.float32), return_values.astype(np.float32)])

    values = np.concatenate(values_parts, axis=1).astype(np.float32)
    observed_mask = np.isfinite(values)
    values = np.where(observed_mask, values, 0.0).astype(np.float32)
    token_metadata = build_geometry_token_metadata(
        level_columns=level_columns,
        return_columns=return_columns,
    )
    if values.shape[1] != token_metadata.n_tokens:
        raise ValueError(f"values have {values.shape[1]} tokens but metadata has {token_metadata.n_tokens}")
    return values, observed_mask.astype(bool), token_metadata


def _contiguous_range(rng: np.random.Generator, upper: int, max_len: int) -> np.ndarray:
    length = int(rng.integers(1, min(max_len, upper) + 1))
    start = int(rng.integers(0, upper - length + 1))
    return np.arange(start, start + length, dtype=np.int64)


def sample_typed_synthetic_mask(
    *,
    shape: tuple[int, int],
    token_metadata: GeometryTokenMetadata,
    rng: np.random.Generator,
    families: tuple[MaskFamily, ...] = (
        "surface_maturity",
        "surface_moneyness",
        "surface_rectangle",
        "vol_side_channel",
        "factor_family",
        "time_block",
    ),
    sparse_drop_prob: float = 0.05,
) -> tuple[np.ndarray, str]:
    time_len, n_tokens = shape
    visible = np.ones((time_len, n_tokens), dtype=bool)
    family = str(rng.choice(np.asarray(families, dtype=object)))
    geom = token_metadata.geometry_id
    coord = token_metadata.geometry_coord

    if family == "surface_maturity":
        maturity = int(rng.integers(0, 5))
        token_idx = np.where((geom == "iv_surface") & (coord[:, 1] == float(maturity)))[0]
        visible[:, token_idx] = False
    elif family == "surface_moneyness":
        moneyness = int(rng.integers(0, 5))
        token_idx = np.where((geom == "iv_surface") & (coord[:, 0] == float(moneyness)))[0]
        visible[:, token_idx] = False
    elif family == "surface_rectangle":
        rows = _contiguous_range(rng, upper=5, max_len=2)
        cols = _contiguous_range(rng, upper=5, max_len=2)
        row_match = np.isin(coord[:, 0].astype(np.int64), rows)
        col_match = np.isin(coord[:, 1].astype(np.int64), cols)
        token_idx = np.where((geom == "iv_surface") & row_match & col_match)[0]
        visible[:, token_idx] = False
    elif family == "vol_side_channel":
        token_idx = np.where(geom == "vol_side_channel")[0]
        if token_idx.size:
            visible[:, int(rng.choice(token_idx))] = False
    elif family == "factor_family":
        available = sorted(
            {
                str(x)
                for x, g in zip(token_metadata.factor_family.tolist(), geom.tolist())
                if g in {"factor_level", "factor_return"}
            }
        )
        if available:
            chosen = str(rng.choice(np.asarray(available, dtype=object)))
            token_idx = np.where(
                np.isin(geom, ["factor_level", "factor_return"])
                & (token_metadata.factor_family == chosen)
            )[0]
            days = _contiguous_range(rng, upper=time_len, max_len=max(1, time_len // 3))
            visible[np.ix_(days, token_idx)] = False
    elif family == "time_block":
        days = _contiguous_range(rng, upper=time_len, max_len=max(1, time_len // 3))
        visible[days, :] = False
    elif family == "sparse":
        visible &= rng.random(size=(time_len, n_tokens)) >= sparse_drop_prob
    else:
        raise ValueError(f"Unknown mask family: {family!r}")

    return visible, family


def apply_synthetic_mask(
    clean_values: np.ndarray,
    observed_mask: np.ndarray,
    synthetic_mask: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(clean_values, dtype=np.float32)
    return np.where(observed_mask & synthetic_mask, arr, 0.0).astype(np.float32)


def build_masked_multiview_batch(
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
    mask_families: tuple[MaskFamily, ...] = (
        "surface_maturity",
        "surface_moneyness",
        "surface_rectangle",
        "vol_side_channel",
        "factor_family",
        "time_block",
    ),
) -> MaskedMultiviewBatch:
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
    clean = values[absolute_index]
    observed_window = observed[absolute_index]
    rng = np.random.default_rng(seed)

    synth_a = np.empty_like(observed_window, dtype=bool)
    synth_b = np.empty_like(observed_window, dtype=bool)
    family_a: list[str] = []
    family_b: list[str] = []
    for row in range(clean.shape[0]):
        mask_a, name_a = sample_typed_synthetic_mask(
            shape=(history_len, token_metadata.n_tokens),
            token_metadata=token_metadata,
            rng=rng,
            families=mask_families,
        )
        mask_b, name_b = sample_typed_synthetic_mask(
            shape=(history_len, token_metadata.n_tokens),
            token_metadata=token_metadata,
            rng=rng,
            families=mask_families,
        )
        synth_a[row] = mask_a
        synth_b[row] = mask_b
        family_a.append(name_a)
        family_b.append(name_b)

    relative_index = offsets
    positive_index = np.arange(clean.shape[0] * history_len, dtype=np.int64).reshape(
        clean.shape[0], history_len
    )
    return MaskedMultiviewBatch(
        clean_values=clean.astype(np.float32),
        view_a_values=apply_synthetic_mask(clean, observed_window, synth_a),
        view_b_values=apply_synthetic_mask(clean, observed_window, synth_b),
        observed_mask=observed_window.astype(bool),
        synthetic_mask_a=synth_a,
        synthetic_mask_b=synth_b,
        absolute_index=absolute_index,
        relative_index=relative_index,
        positive_index=positive_index,
        token_metadata=token_metadata,
        mask_family_a=np.asarray(family_a, dtype=object),
        mask_family_b=np.asarray(family_b, dtype=object),
        start_index=indices,
        split=split,
        metadata={
            "surface_path": str(surface_path),
            "multi_factor_path": str(multi_factor_path) if multi_factor_path is not None else "",
            "history_len": int(history_len),
            "future_len": int(future_len),
            "test_start": int(test_start),
            "val_size": int(val_size),
            "normalize": bool(normalize),
            "n_tokens": int(token_metadata.n_tokens),
        },
    )
