from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.world.evaluation.masked_multiview_data import (
    GeometryTokenMetadata,
    MaskedMultiviewBatch,
    apply_synthetic_mask,
    build_masked_multiview_batch,
)


HardMaskFamily = str


def _contiguous_range(
    rng: np.random.Generator, upper: int, min_len: int, max_len: int
) -> np.ndarray:
    length = int(rng.integers(min_len, min(max_len, upper) + 1))
    start = int(rng.integers(0, upper - length + 1))
    return np.arange(start, start + length, dtype=np.int64)


def sample_hard_structured_mask(
    *,
    shape: tuple[int, int],
    token_metadata: GeometryTokenMetadata,
    rng: np.random.Generator,
    families: tuple[HardMaskFamily, ...] = (
        "surface_large_rectangle",
        "surface_whole_day_block",
        "factor_family_long_block",
        "cross_family_stress_block",
        "time_block_long",
    ),
) -> tuple[np.ndarray, str]:
    time_len, _n_tokens = shape
    visible = np.ones(shape, dtype=bool)
    family = str(rng.choice(np.asarray(families, dtype=object)))
    geom = token_metadata.geometry_id
    coord = token_metadata.geometry_coord

    if family == "surface_large_rectangle":
        rows = _contiguous_range(rng, upper=5, min_len=2, max_len=4)
        cols = _contiguous_range(rng, upper=5, min_len=2, max_len=4)
        row_match = np.isin(coord[:, 0].astype(np.int64), rows)
        col_match = np.isin(coord[:, 1].astype(np.int64), cols)
        token_idx = np.where((geom == "iv_surface") & row_match & col_match)[0]
        days = _contiguous_range(
            rng, upper=time_len, min_len=time_len // 2, max_len=time_len
        )
        visible[np.ix_(days, token_idx)] = False
    elif family == "surface_whole_day_block":
        token_idx = np.where(geom == "iv_surface")[0]
        days = _contiguous_range(
            rng, upper=time_len, min_len=time_len // 3, max_len=time_len
        )
        visible[np.ix_(days, token_idx)] = False
    elif family == "factor_family_long_block":
        available = sorted(
            {
                str(x)
                for x, g in zip(token_metadata.factor_family.tolist(), geom.tolist())
                if g in {"factor_level", "factor_return"}
            }
        )
        if available:
            n_choose = min(len(available), int(rng.integers(1, 4)))
            chosen = rng.choice(
                np.asarray(available, dtype=object), size=n_choose, replace=False
            )
            token_idx = np.where(
                np.isin(geom, ["factor_level", "factor_return"])
                & np.isin(token_metadata.factor_family, chosen)
            )[0]
            days = _contiguous_range(
                rng, upper=time_len, min_len=time_len // 2, max_len=time_len
            )
            visible[np.ix_(days, token_idx)] = False
    elif family == "cross_family_stress_block":
        surface_tokens = np.where(geom == "iv_surface")[0]
        factor_tokens = np.where(np.isin(geom, ["factor_level", "factor_return"]))[0]
        days = _contiguous_range(
            rng, upper=time_len, min_len=time_len // 3, max_len=2 * time_len // 3
        )
        if surface_tokens.size:
            rows = _contiguous_range(rng, upper=5, min_len=2, max_len=3)
            cols = _contiguous_range(rng, upper=5, min_len=2, max_len=3)
            row_match = np.isin(coord[:, 0].astype(np.int64), rows)
            col_match = np.isin(coord[:, 1].astype(np.int64), cols)
            surface_region = np.where((geom == "iv_surface") & row_match & col_match)[0]
            visible[np.ix_(days, surface_region)] = False
        if factor_tokens.size:
            n_factor_tokens = min(factor_tokens.size, max(2, factor_tokens.size // 3))
            chosen_factor_tokens = rng.choice(
                factor_tokens, size=n_factor_tokens, replace=False
            )
            visible[np.ix_(days, chosen_factor_tokens)] = False
    elif family == "time_block_long":
        days = _contiguous_range(
            rng, upper=time_len, min_len=time_len // 3, max_len=2 * time_len // 3
        )
        visible[days, :] = False
    else:
        raise ValueError(f"unknown hard mask family: {family}")

    return visible, family


def build_hard_masked_batch(
    *,
    split: str,
    history_len: int,
    future_len: int,
    max_windows: int,
    seed: int,
) -> MaskedMultiviewBatch:
    base = build_masked_multiview_batch(
        split=split,
        history_len=history_len,
        future_len=future_len,
        max_windows=max_windows,
        seed=seed,
        normalize=True,
    )
    rng = np.random.default_rng(seed)
    synth_a = np.empty_like(base.synthetic_mask_a, dtype=bool)
    synth_b = np.empty_like(base.synthetic_mask_b, dtype=bool)
    family_a: list[str] = []
    family_b: list[str] = []
    for row in range(base.clean_values.shape[0]):
        mask_a, name_a = sample_hard_structured_mask(
            shape=(history_len, base.token_metadata.n_tokens),
            token_metadata=base.token_metadata,
            rng=rng,
        )
        mask_b, name_b = sample_hard_structured_mask(
            shape=(history_len, base.token_metadata.n_tokens),
            token_metadata=base.token_metadata,
            rng=rng,
        )
        synth_a[row] = mask_a
        synth_b[row] = mask_b
        family_a.append(name_a)
        family_b.append(name_b)
    return replace(
        base,
        view_a_values=apply_synthetic_mask(
            base.clean_values, base.observed_mask, synth_a
        ),
        view_b_values=apply_synthetic_mask(
            base.clean_values, base.observed_mask, synth_b
        ),
        synthetic_mask_a=synth_a,
        synthetic_mask_b=synth_b,
        mask_family_a=np.asarray(family_a, dtype=object),
        mask_family_b=np.asarray(family_b, dtype=object),
    )
