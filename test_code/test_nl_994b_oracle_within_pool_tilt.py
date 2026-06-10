from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_994b_oracle_within_pool_tilt import (
    build_oracle_bridge_report,
    build_start_local_pool,
    choose_weight_temperature,
    kill_condition_assessment,
    oracle_select_top_k,
    softmax_oracle_weights,
    weight_shape_stats,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
    _start_only_ranked_rows,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (
    build_support_sampling_plan,
    select_heldout_query_rows,
)


def _synthetic_terminal(n: int = 140, c: int = 3, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, c)).astype(np.float32)


def test_pool_ranking_query_gap_and_density() -> None:
    n = 140
    terminal = np.zeros((n, 2), dtype=np.float32)
    terminal[:, 0] = np.arange(n, dtype=np.float32)
    scale = np.ones(2, dtype=np.float32)
    train = np.arange(100, dtype=np.int64)
    query = 120
    pool = build_start_local_pool(
        query_index=query,
        terminal=terminal,
        scale=scale,
        train_indices=train,
        pool_size=10,
        query_gap=30,
    )
    indices = [row["window_index"] for row in pool]
    # query gap: candidates 91..99 (|c - 120| < 30) excluded; nearest is 90
    assert indices[0] == 90
    assert all(abs(idx - query) >= 30 for idx in indices)
    # dense: consecutive windows allowed inside the pool (no mutual gap)
    assert indices == list(range(90, 80, -1))
    # distances ascending, locality ranks 1..10
    distances = [row["start_distance"] for row in pool]
    assert distances == sorted(distances)
    assert [row["locality_rank"] for row in pool] == list(range(1, 11))


def test_pool_matches_994a_bridge_reference_ranking() -> None:
    rng = np.random.default_rng(994)
    n, t, c = 130, 4, 5
    history_raw = rng.normal(size=(n, t, c)).astype(np.float32)
    train = list(range(100))
    query = 115
    ref = _start_only_ranked_rows(
        query_index=query,
        history_raw=history_raw,
        train_indices=train,
        top_k=8,
        temporal_gap=30,
        cards_by_index={},
    )
    from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
        _safe_scale,
    )

    terminal = history_raw[:, -1, :]
    pool = build_start_local_pool(
        query_index=query,
        terminal=terminal,
        scale=_safe_scale(terminal, train),
        train_indices=np.asarray(train, dtype=np.int64),
        pool_size=8,
        query_gap=30,
    )
    assert [row["window_index"] for row in pool] == [
        int(row["window_index"]) for row in ref
    ]
    ref_dist = [float(row["score_components"]["start_distance"]) for row in ref]
    pool_dist = [row["start_distance"] for row in pool]
    np.testing.assert_allclose(pool_dist, ref_dist, rtol=1e-5, atol=1e-7)


def test_oracle_selection_respects_mutual_gap() -> None:
    pool = [
        {"window_index": 50, "oracle_ensemble_crps_z": 0.10},
        {"window_index": 52, "oracle_ensemble_crps_z": 0.11},  # gap 2 from 50
        {"window_index": 51, "oracle_ensemble_crps_z": 0.15},  # gap 1 from 50
        {"window_index": 100, "oracle_ensemble_crps_z": 0.20},
        {"window_index": 200, "oracle_ensemble_crps_z": 0.30},
        {"window_index": 205, "oracle_ensemble_crps_z": 0.05},  # best overall
    ]
    selected = oracle_select_top_k(pool, top_k=3, mutual_gap=30)
    assert [row["window_index"] for row in selected] == [205, 50, 100]


def test_oracle_selection_dense_pool_falls_back_to_fewer_rows() -> None:
    pool = [
        {"window_index": 50, "oracle_ensemble_crps_z": 0.10},
        {"window_index": 51, "oracle_ensemble_crps_z": 0.11},
        {"window_index": 52, "oracle_ensemble_crps_z": 0.12},
    ]
    selected = oracle_select_top_k(pool, top_k=3, mutual_gap=30)
    assert [row["window_index"] for row in selected] == [50]


def test_softmax_weights_ordering_and_temperature() -> None:
    crps = [0.30, 0.35, 0.40]
    weights = softmax_oracle_weights(crps, temperature=0.2)
    assert weights.shape == (3,)
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    # lower CRPS -> higher weight
    assert weights[0] > weights[1] > weights[2]
    stats = weight_shape_stats(weights)
    assert 1.0 / 3.0 < stats["max_weight"] < 1.0
    # lower temperature sharpens
    sharper = softmax_oracle_weights(crps, temperature=0.05)
    assert float(sharper[0]) > float(weights[0])


def test_choose_weight_temperature_keeps_default_when_non_degenerate() -> None:
    selected = [[0.30, 0.40, 0.55], [0.25, 0.38, 0.50]]
    chosen, diag = choose_weight_temperature(selected, default_temperature=0.2)
    assert chosen == pytest.approx(0.2)
    assert "default" in diag["reason"]
    band = diag["max_weight_band"]
    used = diag["grid_stats"]["0.2"]["mean_max_weight"]
    assert band[0] <= used <= band[1]


def test_choose_weight_temperature_switches_when_uniform() -> None:
    # spreads of 1e-6 -> near-uniform weights at T=0.2 -> must switch
    selected = [[0.300000, 0.300001, 0.300002] for _ in range(4)]
    chosen, diag = choose_weight_temperature(selected, default_temperature=0.2)
    assert chosen != pytest.approx(0.2)
    assert "degenerate" in diag["reason"]
    chosen_stats = diag["grid_stats"][f"{chosen:g}"]
    assert chosen_stats["mean_max_weight"] > 1.0 / 3.0 + 1e-6


def test_bridge_report_schema_engine_compatible() -> None:
    selections = []
    for query_no, q in enumerate([110, 115]):
        supports = []
        crps_values = [0.20 + 0.05 * rank for rank in range(3)]
        weights = softmax_oracle_weights(crps_values, temperature=0.2)
        for rank in range(3):
            supports.append(
                {
                    "window_index": 10 + 40 * rank + query_no,
                    "start_distance": 0.3 + 0.1 * rank,
                    "start_match_score": 1.0 / (1.3 + 0.1 * rank),
                    "locality_rank": rank + 5,
                    "oracle_ensemble_crps_z": crps_values[rank],
                    "oracle_energy_score_z": 0.5 + 0.1 * rank,
                    "weight": float(weights[rank]),
                }
            )
        selections.append({"window_index": q, "supports": supports})
    report = build_oracle_bridge_report(
        selections=selections,
        train_indices=list(range(100)),
        query_indices=[110, 115],
        n_block_windows=130,
        arrays_path="bank.npz",
        pool_size=50,
        query_gap=30,
        mutual_gap=30,
        weight_temperature=0.2,
        crn_base_seed=8128,
        replay_samples=16,
    )
    # leakage marking is mandatory
    assert report["leakage_only_oracle_diagnostic"] is True
    assert report["split"]["train_indices"] == list(range(100))
    assert report["window_indices"] == list(range(130))
    assert "support_overlap" in report["evaluation"]

    # the unchanged engine must be able to consume the rows
    rows = select_heldout_query_rows(report, role="anchor")
    assert [row["window_index"] for row in rows] == [110, 115]
    top_items = rows[0]["top_train_pool"][:3]
    assert [item["rank"] for item in top_items] == [1, 2, 3]
    weights = [item["weight"] for item in top_items]
    assert abs(sum(weights) - 1.0) < 1e-9
    # weights follow oracle CRPS order (lower CRPS -> higher weight)
    assert weights[0] > weights[1] > weights[2]
    analogue_rows = [
        {
            "index": item["window_index"],
            "cosine": float(item["cosine"]),
            "weight": float(item["weight"]),
        }
        for item in top_items
    ]
    plan = build_support_sampling_plan(
        analogue_rows,
        samples_per_analogue=16,
        mode="field_weight",
        weight_temperature=1.0,
    )
    assert sum(plan["sample_counts"]) == 48
    assert len(plan["analogue_rows"]) == 48
    # oracle replay labels travel in score_components for audit
    components = top_items[0]["score_components"]
    assert components["method"] == "oracle_within_pool_replay_tilt"
    assert "oracle_replay_ensemble_crps_z" in components


def test_kill_condition_assessment_verdicts() -> None:
    beat = {
        "L6": {"mean_delta": -0.02, "ci_low": -0.03, "ci_high": -0.01},
        "L30": {"mean_delta": -0.02, "ci_low": -0.035, "ci_high": -0.005},
    }
    assert (
        kill_condition_assessment(beat)["verdict"]
        == "oracle_beats_start_only_tilt_family_alive"
    )
    dead = {
        "L6": {"mean_delta": 0.02, "ci_low": 0.01, "ci_high": 0.03},
        "L30": {"mean_delta": 0.02, "ci_low": 0.005, "ci_high": 0.035},
    }
    assert (
        kill_condition_assessment(dead)["verdict"]
        == "improvement_excluded_tilt_family_dead"
    )
    straddle = {
        "L6": {"mean_delta": -0.005, "ci_low": -0.02, "ci_high": 0.01},
        "L30": {"mean_delta": -0.005, "ci_low": -0.03, "ci_high": 0.02},
    }
    verdict = kill_condition_assessment(straddle)
    assert verdict["verdict"] == (
        "no_established_improvement_inconclusive_or_dead_by_strict_reading"
    )
    assert verdict["improvement_established_all_block_lengths"] is False
