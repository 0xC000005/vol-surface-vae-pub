from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_995d_train_pool_replay_labels import (
    SHARD_ARRAY_KEYS,
    build_causal_pool,
    candidate_future_activity,
    causal_candidate_indices,
    causality_max_excess,
    replay_score_candidate,
    sample_train_query_windows,
    try_load_existing_shard,
)


# --------------------------------------------------------------------------
# Deterministic stratified query sampling.
# --------------------------------------------------------------------------


def test_query_sampling_deterministic_and_sorted() -> None:
    sample_a = sample_train_query_windows(n_queries=1000, lo=110, hi=4009, seed=0)
    sample_b = sample_train_query_windows(n_queries=1000, lo=110, hi=4009, seed=0)
    assert np.array_equal(sample_a, sample_b)
    assert sample_a.size == 1000
    assert np.unique(sample_a).size == 1000
    assert np.all(np.diff(sample_a) > 0)
    assert int(sample_a[0]) >= 110
    assert int(sample_a[-1]) <= 4009


def test_query_sampling_is_evenly_stratified() -> None:
    sample = sample_train_query_windows(n_queries=1000, lo=110, hi=4009, seed=0)
    # each of the 1000 contiguous strata of size 3.9 contributes exactly one
    edges = 110 + np.floor(np.arange(1001, dtype=np.float64) * 3900 / 1000).astype(
        np.int64
    )
    counts, _ = np.histogram(sample, bins=edges)
    assert np.all(counts == 1)


def test_query_sampling_seed_changes_sample() -> None:
    sample_0 = sample_train_query_windows(n_queries=200, lo=110, hi=4009, seed=0)
    sample_1 = sample_train_query_windows(n_queries=200, lo=110, hi=4009, seed=1)
    assert not np.array_equal(sample_0, sample_1)


def test_query_sampling_rejects_oversampling() -> None:
    with pytest.raises(ValueError):
        sample_train_query_windows(n_queries=100, lo=0, hi=50, seed=0)


# --------------------------------------------------------------------------
# Causal candidate universe + pool construction.
# --------------------------------------------------------------------------


def test_causal_candidate_indices_strictly_before_query() -> None:
    candidates = causal_candidate_indices(200, causal_gap=30, universe_max=4009)
    assert candidates.size == 171  # 0..170 inclusive
    assert int(candidates.max()) == 170
    # universe cap binds when the query is deep in the bank
    capped = causal_candidate_indices(5000, causal_gap=30, universe_max=99)
    assert int(capped.max()) == 99
    # no candidates for very early queries
    assert causal_candidate_indices(10, causal_gap=30, universe_max=4009).size == 0


def test_causal_pool_candidates_respect_causality() -> None:
    rng = np.random.default_rng(7)
    terminal = rng.normal(size=(500, 4)).astype(np.float32)
    scale = np.ones(4, dtype=np.float32)
    pool, n_eligible = build_causal_pool(
        query_index=400,
        terminal=terminal,
        scale=scale,
        pool_size=50,
        causal_gap=30,
        universe_max=499,
        min_eligible=60,
    )
    assert pool is not None
    assert n_eligible == 371  # windows 0..370
    assert len(pool) == 50
    indices = [row["window_index"] for row in pool]
    assert all(idx <= 400 - 30 for idx in indices)
    distances = [row["start_distance"] for row in pool]
    assert distances == sorted(distances)
    # matches brute force top-50 over the causal universe
    diff = terminal[:371] - terminal[400][None, :]
    brute = np.sqrt(np.nanmean(np.square(diff / scale[None, :]), axis=1))
    order = np.lexsort((np.arange(371), brute.astype(np.float64)))[:50]
    assert indices == [int(i) for i in order]


def test_causal_pool_skips_when_too_few_eligible() -> None:
    terminal = np.zeros((200, 3), dtype=np.float32)
    scale = np.ones(3, dtype=np.float32)
    pool, n_eligible = build_causal_pool(
        query_index=85,  # universe 0..55 -> 56 candidates < 60
        terminal=terminal,
        scale=scale,
        pool_size=50,
        causal_gap=30,
        universe_max=199,
        min_eligible=60,
    )
    assert pool is None
    assert n_eligible == 56
    # boundary: exactly min_eligible is NOT skipped
    pool_ok, n_ok = build_causal_pool(
        query_index=89,  # universe 0..59 -> 60 candidates
        terminal=terminal,
        scale=scale,
        pool_size=50,
        causal_gap=30,
        universe_max=199,
        min_eligible=60,
    )
    assert pool_ok is not None
    assert n_ok == 60


def test_causality_max_excess_detects_violations() -> None:
    queries = np.asarray([100, 200], dtype=np.int64)
    good = np.asarray([[70, 10], [170, 5]], dtype=np.int64)
    assert causality_max_excess(queries, good, causal_gap=30) == 0
    bad = np.asarray([[70, 10], [171, 5]], dtype=np.int64)
    assert causality_max_excess(queries, bad, causal_gap=30) == 1


# --------------------------------------------------------------------------
# Covariates.
# --------------------------------------------------------------------------


def test_candidate_future_activity_matches_manual() -> None:
    rng = np.random.default_rng(3)
    future_delta = rng.normal(size=(6, 5, 4)).astype(np.float32)
    delta_scale = np.abs(rng.normal(size=(5, 4))).astype(np.float32) + 0.1
    activity = candidate_future_activity(future_delta, delta_scale)
    assert activity.shape == (6,)
    manual = (future_delta[2] / delta_scale).reshape(-1).std()
    assert activity[2] == pytest.approx(float(manual), rel=1e-5)


# --------------------------------------------------------------------------
# CRN determinism of the replay scorer (stub model, no GPU/checkpoint).
# --------------------------------------------------------------------------


class _StubModel:
    """Minimal sample_batched stand-in that consumes the torch RNG stream."""

    def sample_batched(
        self,
        history_level: torch.Tensor,
        history_norm: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        drift_feature: torch.Tensor | None = None,
        *,
        n_samples: int,
        n_steps: int,
        chunk_size: int,
        temperature: float,
    ) -> torch.Tensor:
        b, c = int(history_level.shape[0]), int(history_level.shape[-1])
        noise = torch.randn(b, int(n_samples), int(n_steps), c)
        return noise * float(temperature) + 0.05 * history_level[:, None, -1:, :]


def _stub_reconstruct(
    history_last: np.ndarray, increments: np.ndarray, specs: object
) -> np.ndarray:
    base = np.asarray(history_last, dtype=np.float64)[:, None, None, :]
    return (base + np.cumsum(np.asarray(increments, dtype=np.float64), axis=-2)).astype(
        np.float32
    )


def _stub_inputs(n: int = 40, t: int = 6, c: int = 3, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    history_raw = rng.normal(size=(n, t, c)).astype(np.float32)
    return dict(
        history_level=history_raw.copy(),
        history_norm=rng.normal(size=(n, t, c)).astype(np.float32),
        center=rng.normal(size=(n, c)).astype(np.float32),
        scale=np.abs(rng.normal(size=(n, c))).astype(np.float32) + 0.5,
        drift_feature=np.zeros((n, c), dtype=np.float32),
        history_raw=history_raw,
        target_delta=rng.normal(size=(5, c)).astype(np.float32),
        specs=None,
        delta_scale=np.ones((5, c), dtype=np.float32),
        samples=4,
        n_steps=5,
        chunk_size=2,
        crn_base_seed=8128,
        device=torch.device("cpu"),
        reconstruct_fn=_stub_reconstruct,
    )


def test_replay_score_same_query_candidate_is_bit_identical() -> None:
    model = _StubModel()
    kwargs = _stub_inputs()
    run_1 = replay_score_candidate(
        model, candidate_index=3, query_window=35, start_match_score=0.7, **kwargs
    )
    run_2 = replay_score_candidate(
        model, candidate_index=3, query_window=35, start_match_score=0.7, **kwargs
    )
    assert run_1["ensemble_crps_z"] == run_2["ensemble_crps_z"]
    assert run_1["energy_score_z"] == run_2["energy_score_z"]
    assert run_1["crn_seed"] == run_2["crn_seed"]


def test_replay_score_different_query_changes_noise_stream() -> None:
    model = _StubModel()
    kwargs = _stub_inputs()
    run_a = replay_score_candidate(
        model, candidate_index=3, query_window=35, start_match_score=0.7, **kwargs
    )
    run_b = replay_score_candidate(
        model, candidate_index=3, query_window=36, start_match_score=0.7, **kwargs
    )
    assert run_a["crn_seed"] != run_b["crn_seed"]
    assert run_a["ensemble_crps_z"] != run_b["ensemble_crps_z"]


def test_replay_score_candidates_of_one_query_share_common_random_numbers() -> None:
    """Two candidates with identical inputs differ ONLY via candidate identity.

    With per-candidate seed reset (CRN), making the two candidates' window data
    identical must give bit-identical scores: the noise stream is shared.
    """

    model = _StubModel()
    kwargs = _stub_inputs()
    # make candidate 5's data identical to candidate 3's
    for key in ("history_level", "history_norm", "center", "scale", "history_raw"):
        kwargs[key][5] = kwargs[key][3]
    run_c3 = replay_score_candidate(
        model, candidate_index=3, query_window=35, start_match_score=0.7, **kwargs
    )
    run_c5 = replay_score_candidate(
        model, candidate_index=5, query_window=35, start_match_score=0.7, **kwargs
    )
    assert run_c3["ensemble_crps_z"] == run_c5["ensemble_crps_z"]


# --------------------------------------------------------------------------
# Shard reuse (resume) safety.
# --------------------------------------------------------------------------


def _fake_shard_arrays(
    queries: np.ndarray, pool_size: int, *, causal_gap: int = 30
) -> dict[str, np.ndarray]:
    n = queries.size
    rng = np.random.default_rng(5)
    candidates = np.stack(
        [
            rng.choice(int(q) - causal_gap + 1, size=pool_size, replace=False)
            for q in queries
        ]
    ).astype(np.int64)
    arrays = {
        "query_window": queries.astype(np.int64),
        "n_eligible_candidates": np.full(n, 100, dtype=np.int64),
        "crn_seed": np.arange(n, dtype=np.int64),
        "pool_candidate_window": candidates,
        "replay_crps": rng.uniform(0.2, 0.6, size=(n, pool_size)),
        "replay_energy": rng.uniform(0.2, 0.6, size=(n, pool_size)),
        "replay_coverage_80": rng.uniform(0.5, 1.0, size=(n, pool_size)),
        "cand_start_distance": rng.uniform(0.1, 2.0, size=(n, pool_size)),
        "cand_index_gap": (queries[:, None] - candidates).astype(np.int64),
        "cand_future_activity": rng.uniform(0.5, 2.0, size=(n, pool_size)).astype(
            np.float32
        ),
        "query_future_activity": rng.uniform(0.5, 2.0, size=n).astype(np.float32),
        "query_pool_replay_crps_std": rng.uniform(0.01, 0.1, size=n),
        "pool_min_start_distance": rng.uniform(0.05, 0.2, size=n),
        "pool_median_start_distance": rng.uniform(0.2, 0.6, size=n),
    }
    assert set(arrays) == set(SHARD_ARRAY_KEYS)
    return arrays


def test_try_load_existing_shard_accepts_matching_block(tmp_path) -> None:
    queries = np.asarray([150, 220, 305], dtype=np.int64)
    arrays = _fake_shard_arrays(queries, pool_size=8)
    path = tmp_path / "labels_shard_0000.npz"
    np.savez_compressed(path, **arrays)
    loaded = try_load_existing_shard(path, queries, pool_size=8, causal_gap=30)
    assert loaded is not None
    assert np.array_equal(loaded["replay_crps"], arrays["replay_crps"])


def test_try_load_existing_shard_rejects_mismatch_and_violation(tmp_path) -> None:
    queries = np.asarray([150, 220, 305], dtype=np.int64)
    arrays = _fake_shard_arrays(queries, pool_size=8)
    path = tmp_path / "labels_shard_0000.npz"
    np.savez_compressed(path, **arrays)
    # query-block mismatch -> reject (forces recompute)
    other = np.asarray([150, 220, 306], dtype=np.int64)
    assert try_load_existing_shard(path, other, pool_size=8, causal_gap=30) is None
    # missing file -> reject
    assert (
        try_load_existing_shard(tmp_path / "nope.npz", queries, pool_size=8, causal_gap=30)
        is None
    )
    # causality violation inside the stored shard -> reject
    bad = dict(arrays)
    bad["pool_candidate_window"] = arrays["pool_candidate_window"].copy()
    bad["pool_candidate_window"][0, 0] = int(queries[0]) - 29
    bad_path = tmp_path / "labels_shard_0001.npz"
    np.savez_compressed(bad_path, **bad)
    assert try_load_existing_shard(bad_path, queries, pool_size=8, causal_gap=30) is None
    # non-finite replay CRPS -> reject
    nan_arrays = dict(arrays)
    nan_arrays["replay_crps"] = arrays["replay_crps"].copy()
    nan_arrays["replay_crps"][0, 0] = np.nan
    nan_path = tmp_path / "labels_shard_0002.npz"
    np.savez_compressed(nan_path, **nan_arrays)
    assert try_load_existing_shard(nan_path, queries, pool_size=8, causal_gap=30) is None
