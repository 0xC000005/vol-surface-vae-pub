from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_997a_set_level_teacher_labels import (
    CATEGORY_CODE,
    SET_SHARD_ARRAY_KEYS,
    build_query_sets,
    deployed_set_weights,
    greedy_gap_select,
    replay_score_set,
    sample_feature_spread_set,
    sample_gap_respecting_set,
    sample_stratified_set,
    set_causality_max_excess,
    set_mutual_gap_min,
    solo_informed_probabilities,
    try_load_existing_set_shard,
    tune_solo_informed_temperature,
)


# --------------------------------------------------------------------------
# Synthetic pool fixture: 50 causal candidates for query window 2000.
# --------------------------------------------------------------------------

QUERY = 2000
GAP = 30


def _pool(seed: int = 13) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # 50 distinct causal windows <= QUERY - GAP, some clustered within GAP
    base = rng.choice(np.arange(0, QUERY - GAP + 1), size=40, replace=False)
    clustered = base[:10] + rng.integers(1, GAP, size=10)
    windows = np.unique(np.concatenate([base, clustered]))[:50]
    assert windows.size == 50
    rng.shuffle(windows)
    distances = np.sort(rng.uniform(0.1, 2.0, size=50))  # locality order
    solo = rng.uniform(0.2, 0.8, size=50)
    features = rng.normal(size=(50, 6))
    return {
        "windows": windows.astype(np.int64),
        "distances": distances,
        "solo": solo,
        "features": features,
    }


def _assert_gap_ok(members: list[int], windows: np.ndarray) -> None:
    win = [int(windows[p]) for p in members]
    for i in range(len(win)):
        for j in range(i + 1, len(win)):
            assert abs(win[i] - win[j]) >= GAP


# --------------------------------------------------------------------------
# Deployed weighting convention.
# --------------------------------------------------------------------------


def test_deployed_set_weights_match_manual_softmax_t1() -> None:
    dists = [0.3, 0.5, 1.1]
    weights = deployed_set_weights(dists)
    manual = np.exp(-np.asarray(dists))
    manual = manual / manual.sum()
    assert weights == pytest.approx(manual, rel=1e-12)
    assert weights.sum() == pytest.approx(1.0)
    # closer support gets the larger weight (deployed top3/90 ordering)
    assert weights[0] > weights[1] > weights[2]


# --------------------------------------------------------------------------
# Greedy gap-respecting selection (locality / solo-oracle sets).
# --------------------------------------------------------------------------


def test_greedy_gap_select_respects_gap_and_order() -> None:
    windows = np.asarray([100, 105, 200, 230, 400, 460], dtype=np.int64)
    chosen = greedy_gap_select(np.arange(6), windows, mutual_gap=GAP)
    assert chosen is not None
    _assert_gap_ok(chosen, windows)
    # greedy along the order: 100 taken, 105 skipped (<30), 200 taken,
    # 230 taken (gap exactly 30 is allowed)
    assert chosen == [0, 2, 3]


def test_greedy_gap_select_returns_none_when_impossible() -> None:
    windows = np.asarray([100, 110, 120, 125], dtype=np.int64)
    assert greedy_gap_select(np.arange(4), windows, mutual_gap=GAP) is None


# --------------------------------------------------------------------------
# Sampled set builders: determinism + gap constraint.
# --------------------------------------------------------------------------


def test_sample_gap_respecting_set_deterministic_and_gap_safe() -> None:
    pool = _pool()
    probs = np.full(50, 1.0 / 50)
    sets_a = [
        sample_gap_respecting_set(
            np.random.default_rng([0, QUERY, k]), pool["windows"], probs,
            mutual_gap=GAP,
        )
        for k in range(20)
    ]
    sets_b = [
        sample_gap_respecting_set(
            np.random.default_rng([0, QUERY, k]), pool["windows"], probs,
            mutual_gap=GAP,
        )
        for k in range(20)
    ]
    assert sets_a == sets_b  # same seed => identical
    for members in sets_a:
        assert members is not None
        assert len(members) == 3
        assert len(set(members)) == 3
        _assert_gap_ok(members, pool["windows"])


def test_sample_gap_respecting_set_none_when_pool_too_dense() -> None:
    windows = np.asarray([100, 101, 102, 103], dtype=np.int64)
    probs = np.full(4, 0.25)
    rng = np.random.default_rng(0)
    assert (
        sample_gap_respecting_set(rng, windows, probs, mutual_gap=GAP, max_attempts=5)
        is None
    )


def test_sample_stratified_set_one_member_per_tercile() -> None:
    pool = _pool()
    rng = np.random.default_rng(3)
    edges = np.floor(np.linspace(0, 50, 4)).astype(int)  # implementation edges
    for _ in range(10):
        members = sample_stratified_set(rng, pool["windows"], mutual_gap=GAP)
        assert members is not None
        _assert_gap_ok(members, pool["windows"])
        strata = sorted(
            int(np.searchsorted(edges, p, side="right") - 1) for p in members
        )
        assert strata == [0, 1, 2]  # one per start-distance tercile


def test_sample_feature_spread_set_deterministic_and_gap_safe() -> None:
    pool = _pool()
    run_a = sample_feature_spread_set(
        np.random.default_rng(5), pool["windows"], pool["features"], mutual_gap=GAP
    )
    run_b = sample_feature_spread_set(
        np.random.default_rng(5), pool["windows"], pool["features"], mutual_gap=GAP
    )
    assert run_a == run_b
    assert run_a is not None
    _assert_gap_ok(run_a, pool["windows"])


# --------------------------------------------------------------------------
# Solo-informed probabilities + temperature tuning.
# --------------------------------------------------------------------------


def test_solo_informed_probabilities_prefer_low_crps() -> None:
    solo = np.asarray([0.2, 0.4, 0.6, 0.8])
    probs = solo_informed_probabilities(solo, temperature=1.0)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(np.diff(probs) < 0)  # lower CRPS => higher probability
    # scale invariance via per-query standardization
    probs_scaled = solo_informed_probabilities(10.0 * solo, temperature=1.0)
    assert probs == pytest.approx(probs_scaled, rel=1e-12)


def test_tune_solo_informed_temperature_ess_monotonic_and_recorded() -> None:
    rng = np.random.default_rng(2)
    solo_by_query = [rng.uniform(0.2, 0.8, size=50) for _ in range(30)]
    chosen, diag = tune_solo_informed_temperature(solo_by_query, target_ess=16.0)
    assert chosen in diag["grid"]
    means = [diag["grid_stats"][f"{t:g}"]["mean_ess"] for t in diag["grid"]]
    assert all(a <= b + 1e-9 for a, b in zip(means, means[1:]))  # ESS grows with T
    assert diag["chosen_mean_ess"] == diag["grid_stats"][f"{chosen:g}"]["mean_ess"]


# --------------------------------------------------------------------------
# Full per-query set plan: determinism, dedup, gap, causality, categories.
# --------------------------------------------------------------------------


def _build(pool: dict[str, np.ndarray], seed_key: list[int]) -> tuple[list, int]:
    informed = solo_informed_probabilities(pool["solo"], temperature=1.0)
    return build_query_sets(
        np.random.default_rng(seed_key),
        windows=pool["windows"],
        solo_crps=pool["solo"],
        features=pool["features"],
        informed_probs=informed,
        mutual_gap=GAP,
    )


def test_build_query_sets_deterministic() -> None:
    pool = _pool()
    sets_a, dropped_a = _build(pool, [0, QUERY])
    sets_b, dropped_b = _build(pool, [0, QUERY])
    assert dropped_a == dropped_b
    assert sets_a == sets_b


def test_build_query_sets_dedup_gap_causality_categories() -> None:
    pool = _pool()
    sets, dropped = _build(pool, [0, QUERY])
    assert len(sets) + dropped == 40
    assert len(sets) >= 35  # dedup may drop a few slots, not most
    # dedup: no two sets share the same member-window composition
    keys = [frozenset(int(pool["windows"][p]) for p in s["members"]) for s in sets]
    assert len(keys) == len(set(keys))
    # gap + causality for every set
    for s in sets:
        assert len(s["members"]) == 3
        _assert_gap_ok(s["members"], pool["windows"])
        for p in s["members"]:
            assert int(pool["windows"][p]) <= QUERY - GAP
    # categories: locality first, then solo_oracle, counts bounded by targets
    assert sets[0]["category"] == "locality"
    counts: dict[str, int] = {}
    for s in sets:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    assert counts["locality"] == 1
    assert counts.get("solo_oracle", 0) <= 1
    assert counts.get("solo_informed", 0) <= 14
    assert counts.get("diversity_stratified", 0) <= 6
    assert counts.get("diversity_feature_spread", 0) <= 6
    assert counts.get("uniform_random", 0) <= 12
    assert set(counts) <= set(CATEGORY_CODE)
    # locality set == greedy over locality order (distances are sorted)
    expected_locality = greedy_gap_select(
        np.arange(50), pool["windows"], mutual_gap=GAP
    )
    assert sets[0]["members"] == expected_locality
    # solo-oracle set == greedy over solo-CRPS order
    solo_order = np.lexsort((pool["windows"], pool["solo"]))
    expected_oracle = greedy_gap_select(solo_order, pool["windows"], mutual_gap=GAP)
    oracle_sets = [s for s in sets if s["category"] == "solo_oracle"]
    if oracle_sets:
        assert oracle_sets[0]["members"] == expected_oracle


def test_build_query_sets_unusable_pool_returns_empty() -> None:
    pool = _pool()
    pool = dict(pool)
    pool["windows"] = np.arange(100, 150, dtype=np.int64)  # all within one gap
    sets, dropped = _build(pool, [0, QUERY])
    assert sets == []
    assert dropped == 0


# --------------------------------------------------------------------------
# Causality / mutual-gap checkers on shard-shaped arrays.
# --------------------------------------------------------------------------


def test_set_causality_max_excess_detects_violation_and_masks_invalid() -> None:
    queries = np.asarray([100, 200], dtype=np.int64)
    members = np.full((2, 2, 3), -1, dtype=np.int64)
    valid = np.zeros((2, 2), dtype=bool)
    members[0, 0] = [10, 40, 70]
    valid[0, 0] = True
    members[1, 0] = [100, 140, 170]
    valid[1, 0] = True
    assert set_causality_max_excess(queries, members, valid, causal_gap=30) == 0
    members[1, 0, 2] = 171  # > 200 - 30
    assert set_causality_max_excess(queries, members, valid, causal_gap=30) == 1
    # an invalid slot with a violating window must NOT trip the check
    members[1, 0, 2] = 170
    members[1, 1] = [199, 198, 197]
    assert set_causality_max_excess(queries, members, valid, causal_gap=30) == 0


def test_set_mutual_gap_min() -> None:
    members = np.asarray([[[10, 40, 70]], [[100, 131, 162]]], dtype=np.int64)
    valid = np.ones((2, 1), dtype=bool)
    assert set_mutual_gap_min(members, valid) == 30
    valid[1, 0] = False
    assert set_mutual_gap_min(members, valid) == 30
    valid[0, 0] = False
    assert set_mutual_gap_min(members, valid) == 1 << 30


# --------------------------------------------------------------------------
# CRN determinism of the set replay scorer (stub model, no GPU/checkpoint).
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


def _stub_reconstruct(history_last, increments, specs):
    base = np.asarray(history_last, dtype=np.float64)[:, None, None, :]
    return (
        base + np.cumsum(np.asarray(increments, dtype=np.float64), axis=-2)
    ).astype(np.float32)


def _stub_kwargs(n: int = 40, t: int = 6, c: int = 3, seed: int = 11) -> dict:
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
        samples_per_analogue=4,
        n_steps=5,
        chunk_size=2,
        crn_base_seed=8128,
        device=torch.device("cpu"),
        reconstruct_fn=_stub_reconstruct,
    )


def _score(model, kwargs, *, members=(3, 9, 15), query=35):
    weights = deployed_set_weights([0.3, 0.5, 1.1])
    return replay_score_set(
        model,
        member_windows=list(members),
        member_match_scores=[0.7, 0.6, 0.5],
        member_weights=[float(w) for w in weights],
        query_window=query,
        **kwargs,
    )


def test_replay_score_set_same_query_set_is_bit_identical() -> None:
    model = _StubModel()
    kwargs = _stub_kwargs()
    run_1 = _score(model, kwargs)
    run_2 = _score(model, kwargs)
    assert run_1["ensemble_crps_z"] == run_2["ensemble_crps_z"]
    assert run_1["energy_score_z"] == run_2["energy_score_z"]
    assert run_1["crn_seed"] == run_2["crn_seed"]
    # engine field_weight semantics: total paths = 3 x samples_per_analogue
    assert sum(run_1["sample_counts"]) == 3 * 4
    assert run_1["n_paths_total"] == 3 * 4


def test_replay_score_set_sets_of_one_query_share_common_random_numbers() -> None:
    """Two sets with identical member data must give bit-identical scores."""

    model = _StubModel()
    kwargs = _stub_kwargs()
    for key in ("history_level", "history_norm", "center", "scale", "history_raw"):
        kwargs[key][20] = kwargs[key][3]
        kwargs[key][21] = kwargs[key][9]
        kwargs[key][22] = kwargs[key][15]
    run_a = _score(model, kwargs, members=(3, 9, 15))
    run_b = _score(model, kwargs, members=(20, 21, 22))
    assert run_a["ensemble_crps_z"] == run_b["ensemble_crps_z"]


def test_replay_score_set_different_query_changes_noise_stream() -> None:
    model = _StubModel()
    kwargs = _stub_kwargs()
    run_a = _score(model, kwargs, query=35)
    run_b = _score(model, kwargs, query=36)
    assert run_a["crn_seed"] != run_b["crn_seed"]
    assert run_a["ensemble_crps_z"] != run_b["ensemble_crps_z"]


# --------------------------------------------------------------------------
# Shard reuse (resume) safety.
# --------------------------------------------------------------------------


def _fake_set_shard(queries: np.ndarray, n_sets: int = 4) -> dict[str, np.ndarray]:
    n = queries.size
    rng = np.random.default_rng(5)
    valid = np.ones((n, n_sets), dtype=bool)
    valid[:, -1] = False
    cat = np.full((n, n_sets), -1, dtype=np.int8)
    cat[:, 0] = 0
    cat[:, 1] = 1
    cat[:, 2] = 5
    member = np.full((n, n_sets, 3), -1, dtype=np.int64)
    for i, q in enumerate(queries):
        hi = int(q) - 30
        member[i, 0] = [hi - 90, hi - 45, hi]
        member[i, 1] = [hi - 100, hi - 60, hi - 5]
        member[i, 2] = [hi - 200, hi - 150, hi - 99]
    crps = rng.uniform(0.2, 0.6, size=(n, n_sets))
    crps[~valid] = np.nan
    arrays = {
        "query_window": queries.astype(np.int64),
        "n_eligible_candidates": np.full(n, 100, dtype=np.int64),
        "crn_seed": np.arange(n, dtype=np.int64),
        "n_valid_sets": np.full(n, n_sets - 1, dtype=np.int64),
        "n_dropped_set_slots": np.ones(n, dtype=np.int64),
        "set_valid": valid,
        "set_category": cat,
        "set_member_window": member,
        "set_member_start_distance": rng.uniform(0.1, 2.0, size=(n, n_sets, 3)),
        "set_member_solo_crps": rng.uniform(0.2, 0.6, size=(n, n_sets, 3)),
        "set_weight": np.full((n, n_sets, 3), 1.0 / 3),
        "set_sample_count": np.full((n, n_sets, 3), 16, dtype=np.int64),
        "set_crps": crps,
        "set_energy": rng.uniform(0.2, 0.6, size=(n, n_sets)),
        "set_coverage_80": rng.uniform(0.5, 1.0, size=(n, n_sets)),
        "set_sum_solo_crps": rng.uniform(0.6, 1.8, size=(n, n_sets)),
        "set_weighted_solo_crps": rng.uniform(0.2, 0.6, size=(n, n_sets)),
        "locality_set_crps": crps[:, 0],
        "best_set_crps": np.nanmin(crps, axis=1),
        "set_crps_std": np.nanstd(crps, axis=1),
        "spearman_set_vs_sum_solo": rng.uniform(-1, 1, size=n),
    }
    assert set(arrays) == set(SET_SHARD_ARRAY_KEYS)
    return arrays


def test_try_load_existing_set_shard_accepts_matching_plan(tmp_path) -> None:
    queries = np.asarray([500, 700, 900], dtype=np.int64)
    arrays = _fake_set_shard(queries)
    path = tmp_path / "set_labels_shard_0000.npz"
    np.savez_compressed(path, **arrays)
    loaded = try_load_existing_set_shard(
        path,
        expected_query_window=queries,
        expected_set_valid=arrays["set_valid"],
        expected_set_category=arrays["set_category"],
        expected_set_member_window=arrays["set_member_window"],
        causal_gap=30,
        mutual_gap=30,
    )
    assert loaded is not None
    assert np.array_equal(
        loaded["set_crps"][arrays["set_valid"]],
        arrays["set_crps"][arrays["set_valid"]],
    )


def test_try_load_existing_set_shard_rejects_mismatch_and_violation(tmp_path) -> None:
    queries = np.asarray([500, 700, 900], dtype=np.int64)
    arrays = _fake_set_shard(queries)
    path = tmp_path / "set_labels_shard_0000.npz"
    np.savez_compressed(path, **arrays)

    def _load(**overrides):
        kwargs = dict(
            expected_query_window=queries,
            expected_set_valid=arrays["set_valid"],
            expected_set_category=arrays["set_category"],
            expected_set_member_window=arrays["set_member_window"],
            causal_gap=30,
            mutual_gap=30,
        )
        kwargs.update(overrides)
        return try_load_existing_set_shard(path, **kwargs)

    assert _load() is not None
    # query mismatch
    assert _load(expected_query_window=np.asarray([500, 700, 901])) is None
    # set-plan mismatch (deterministic plan changed => recompute)
    other_member = arrays["set_member_window"].copy()
    other_member[0, 0, 0] -= 1
    assert _load(expected_set_member_window=other_member) is None
    # missing file
    assert (
        try_load_existing_set_shard(
            tmp_path / "nope.npz",
            expected_query_window=queries,
            expected_set_valid=arrays["set_valid"],
            expected_set_category=arrays["set_category"],
            expected_set_member_window=arrays["set_member_window"],
            causal_gap=30,
            mutual_gap=30,
        )
        is None
    )
    # causality violation stored in the shard
    bad = {k: v.copy() for k, v in arrays.items()}
    bad["set_member_window"][0, 0, 2] = int(queries[0]) - 29
    bad_path = tmp_path / "set_labels_shard_0001.npz"
    np.savez_compressed(bad_path, **bad)
    assert (
        try_load_existing_set_shard(
            bad_path,
            expected_query_window=queries,
            expected_set_valid=bad["set_valid"],
            expected_set_category=bad["set_category"],
            expected_set_member_window=bad["set_member_window"],
            causal_gap=30,
            mutual_gap=30,
        )
        is None
    )
    # mutual-gap violation stored in the shard
    dense = {k: v.copy() for k, v in arrays.items()}
    dense["set_member_window"][0, 0] = [100, 110, 200]
    dense_path = tmp_path / "set_labels_shard_0002.npz"
    np.savez_compressed(dense_path, **dense)
    assert (
        try_load_existing_set_shard(
            dense_path,
            expected_query_window=queries,
            expected_set_valid=dense["set_valid"],
            expected_set_category=dense["set_category"],
            expected_set_member_window=dense["set_member_window"],
            causal_gap=30,
            mutual_gap=30,
        )
        is None
    )
    # non-finite CRPS on a valid slot
    nan_arrays = {k: v.copy() for k, v in arrays.items()}
    nan_arrays["set_crps"][0, 0] = np.nan
    nan_path = tmp_path / "set_labels_shard_0003.npz"
    np.savez_compressed(nan_path, **nan_arrays)
    assert (
        try_load_existing_set_shard(
            nan_path,
            expected_query_window=queries,
            expected_set_valid=nan_arrays["set_valid"],
            expected_set_category=nan_arrays["set_category"],
            expected_set_member_window=nan_arrays["set_member_window"],
            causal_gap=30,
            mutual_gap=30,
        )
        is None
    )
