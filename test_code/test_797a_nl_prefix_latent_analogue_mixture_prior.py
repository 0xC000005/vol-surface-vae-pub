import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")

import experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior as prior_module
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (
    build_mixture_memory_prior,
    candidate_support_table,
    direction_check_for_mixture,
    run_analogue_mixture_prior,
    start_distances_to_query_start,
    weighted_prefix_terminal_rows,
)
from experiments.backfill.block_ar.nl_prefix_latent_market_alignment import (
    is_terminal_direction_checkable,
    market_implication_alignment,
)


def _spec_names() -> list[str]:
    return [f"iv:{idx}" for idx in range(25)] + ["factor:spx", "factor:vix"]


def _history() -> np.ndarray:
    history = np.zeros((3, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    # Candidate 0 matches risk-on: SPX up, VIX down.
    history[0, -1, 25] = 2.0
    history[0, -1, 26] = -1.0
    # Candidate 1 is opposite.
    history[1, -1, 25] = -2.0
    history[1, -1, 26] = 1.0
    # Candidate 2 is partly aligned.
    history[2, -1, 25] = 1.0
    history[2, -1, 26] = 1.0
    return history


def _grounding() -> dict:
    return {
        "market_implications": [
            {"market": "SPX", "direction": "up", "confidence": "high"},
            {"market": "VIX", "direction": "down", "confidence": "high"},
        ]
    }


def test_weighted_prefix_terminal_rows_blends_selected_analogues() -> None:
    rows = weighted_prefix_terminal_rows(
        history_level=_history(),
        window_indices=np.asarray([0, 1]),
        weights=np.asarray([0.75, 0.25], dtype=np.float32),
        spec_names=_spec_names(),
    )

    by_market = {row["Market"]: row["Mean Terminal Delta"] for row in rows}

    assert by_market["SPX"] > 0.0
    assert by_market["VIX"] < 0.0


def test_candidate_mixture_components_preserve_selection_weights() -> None:
    candidates = candidate_support_table(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=1.0,
    )
    candidate = {
        "query_id": "weighted_test",
        "candidate_mixture_rank": 0,
        "top_train_pool": [
            {"window_index": 0, "weight": 0.8},
            {"window_index": 2, "weight": 0.2},
        ],
    }

    components = prior_module._candidate_mixture_to_prior_components(
        candidate=candidate,
        candidates=candidates,
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            dtype=np.float32,
        ),
        history_level=_history(),
        grounding=_grounding(),
        spec_names=_spec_names(),
    )

    assert components is not None
    assert np.allclose(components["weights"], [0.8, 0.2])
    assert np.allclose(components["memory"], [0.96, 0.04], atol=1.0e-6)


def test_candidate_support_table_combines_memory_and_implication_alignment() -> None:
    rows = candidate_support_table(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=1.0,
    )

    by_idx = {row["window_index"]: row for row in rows}

    assert by_idx[0]["recent_prefix_alignment_score"] == 1.0
    assert by_idx[1]["recent_prefix_alignment_score"] == -1.0
    assert by_idx[1]["narrative_start_score"] > by_idx[0]["narrative_start_score"]
    assert by_idx[1]["recent_prefix_alignment"]["mismatch_count"] == 2
    assert by_idx[0]["combined_score"] > by_idx[1]["combined_score"]
    assert by_idx[0]["start_only_score"] == 0.0


def test_candidate_support_table_can_condition_on_supplied_start_state() -> None:
    query_start = _history()[2, -1, :]

    distances = start_distances_to_query_start(
        start_state=_history()[:, -1, :],
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        query_window_index=0,
        query_start_state=query_start,
    )
    rows = candidate_support_table(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        query_start_state=query_start,
        grounding=_grounding(),
        spec_names=_spec_names(),
        start_distance_threshold_z=0.0,
        start_distance_penalty=10.0,
        implication_alignment_weight=0.0,
    )

    by_idx = {row["window_index"]: row for row in rows}

    assert distances[2] == 0.0
    assert by_idx[2]["start_distance_z"] == 0.0
    assert by_idx[0]["start_distance_cost"] > 0.0
    assert by_idx[2]["combined_score"] > by_idx[0]["combined_score"]


def test_start_only_control_prior_ignores_narrative_memory_for_ranking() -> None:
    query_start = _history()[2, -1, :]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        query_start_state=query_start,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_start_only",
        top_k=1,
        temperature=0.2,
        start_distance_threshold_z=0.0,
        start_distance_penalty=10.0,
        implication_alignment_weight=100.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["mode"] == "soft_topk_start_only"
    assert result["window_indices"] == [2]
    assert result["candidate_details"][0]["start_distance_z"] == 0.0


def test_build_mixture_memory_prior_returns_weighted_memory_and_support() -> None:
    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_combined",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=1.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["mode"] == "soft_topk_combined"
    assert result["memory"].shape == (2,)
    assert result["analogue_count"] == 2
    assert result["support_diversity_policy"]["policy"] == "score_ranked_support"
    assert result["support_diversity_policy"]["temporal_non_overlap_enforced"] is False
    assert abs(sum(result["weights"]) - 1.0) < 1e-6
    assert result["support_alignment"]["checked_count"] == 2
    assert result["direction_check"]["status"] != "reject"
    assert result["query_start_source"] == "query_window_index"


def test_narrative_start_mode_keeps_grounding_as_direction_check_only() -> None:
    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_narrative_start",
        top_k=1,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=100.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["window_indices"] == [1]
    assert result["candidate_details"][0]["recent_prefix_mismatches"] == 2
    assert result["direction_check"]["status"] == "reject"
    assert (
        result["direction_check"]["reason"] == "final_mixed_prefix_direction_mismatch"
    )


def test_narrative_start_checked_mode_uses_grounding_as_hard_gate() -> None:
    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="soft_topk_narrative_start_checked",
        top_k=1,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=100.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["window_indices"] == [0]
    assert result["candidate_details"][0]["recent_prefix_mismatches"] == 0
    assert result["direction_check"]["status"] != "reject"


def test_diverse_narrative_start_checked_mode_avoids_duplicate_support() -> None:
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.8, 1.2, -2.0]
    history[:, -1, 26] = [-1.0, -0.8, -0.4, 1.0]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [
                [1.00, 0.00],
                [0.99, 0.01],
                [0.00, 1.00],
                [1.00, 0.00],
            ],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="diverse_topk_narrative_start_checked",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=0.95,
    )

    assert result["window_indices"] == [0, 2]
    assert all(
        row["recent_prefix_mismatches"] == 0 for row in result["candidate_details"]
    )
    assert result["direction_check"]["status"] == "pass"


def test_diverse_narrative_start_checked_mode_enforces_temporal_gap() -> None:
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8, 0.8]
    history[:, -1, 26] = [-1.0, -0.9, -0.8, -0.2]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [
                [1.00, 0.00],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.20, 0.80],
            ],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="diverse_topk_narrative_start_checked",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        diverse_min_index_gap=3,
    )

    assert result["window_indices"] == [0, 3]
    assert abs(result["window_indices"][1] - result["window_indices"][0]) >= 3
    assert (
        result["support_diversity_policy"]["policy"]
        == "direction_checked_latent_temporal_diverse_support"
    )
    assert result["support_diversity_policy"]["temporal_min_index_gap"] == 3
    assert result["support_diversity_policy"]["temporal_non_overlap_enforced"] is True
    assert result["support_diversity_policy"]["padding_with_temporal_overlaps"] is False


def test_diverse_narrative_start_checked_mode_does_not_pad_with_overlaps() -> None:
    history = np.zeros((3, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8]
    history[:, -1, 26] = [-1.0, -0.9, -0.8]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.00, 0.00], [0.99, 0.01], [0.98, 0.02]],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="diverse_topk_narrative_start_checked",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        diverse_min_index_gap=3,
    )

    assert result["window_indices"] == [0]
    assert result["analogue_count"] == 1
    assert result["support_diversity_policy"]["selected_count"] == 1
    assert result["support_diversity_policy"]["requested_top_k"] == 3


def test_cohesive_narrative_start_checked_mode_keeps_anchor_family() -> None:
    history = np.zeros((5, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8, 1.0, 0.9]
    history[:, -1, 26] = [-1.0, -0.9, -0.8, -0.4, -0.3]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [
                [1.00, 0.00],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.05, 0.95],
                [0.02, 0.98],
            ],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3, 4]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="cohesive_topk_narrative_start_checked",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=0.95,
    )

    assert result["window_indices"] == [0, 1, 2]
    assert result["support_diversity_policy"]["policy"] == (
        "direction_checked_anchor_cohesive_support"
    )
    assert result["support_diversity_policy"]["anchor_window_index"] == 0
    assert result["direction_check"]["status"] == "pass"


def test_cluster_family_narrative_start_checked_mode_prefers_coherent_family() -> None:
    history = np.zeros((5, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.8, 1.7, 1.6, 0.9]
    history[:, -1, 26] = [-1.0, -0.8, -0.7, -0.6, -0.3]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [
                [1.00, 0.00],
                [0.82, 0.58],
                [0.80, 0.60],
                [0.78, 0.62],
                [0.00, 1.00],
            ],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3, 4]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="cluster_family_narrative_start_checked",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=0.95,
    )

    assert result["window_indices"] == [1, 2, 3]
    assert result["support_diversity_policy"]["policy"] == (
        "direction_checked_latent_family_support"
    )
    assert result["support_diversity_policy"]["family_size"] == 3
    assert result["direction_check"]["status"] == "pass"


def test_similarity_kernel_narrative_start_checked_mode_concentrates_weights() -> None:
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.0, 0.8]
    history[:, -1, 26] = [-1.0, -0.9, -0.4, -0.2]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [
                [1.00, 0.00],
                [0.70, 0.714],
                [0.20, 0.980],
                [0.10, 0.90],
            ],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="kernel_topk_narrative_start_checked",
        top_k=3,
        temperature=0.05,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=0.99,
    )

    assert result["window_indices"] == [0, 1, 2]
    assert result["weights"][0] > 0.70
    assert result["weights"][0] > result["weights"][1] > result["weights"][2]
    assert abs(sum(result["weights"]) - 1.0) < 1e-6
    assert result["support_diversity_policy"]["policy"] == (
        "direction_checked_low_temperature_similarity_kernel_support"
    )


def test_portfolio_quality_guard_mode_reweights_live_support_candidates() -> None:
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8, 1.7]
    history[:, -1, 26] = [-1.0, -0.9, -0.8, -0.7]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06], [0.1, 0.9]],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="portfolio_quality_guard_924e",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "portfolio_prior": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    weights = dict(zip(result["window_indices"], result["weights"], strict=True))
    assert result["mode"] == "portfolio_quality_guard_924e"
    assert (
        result["portfolio_quality_guard_policy"]["fallback_to_equal_support"] is False
    )
    assert result["support_diversity_policy"]["portfolio_quality_guard_active"] is True
    assert weights[2] > weights[0]
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_direction_first_quality_guard_selects_atomic_direction_safe_candidate() -> (
    None
):
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8, 1.7]
    history[:, -1, 26] = [-1.0, -0.9, -0.8, -0.7]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06], [0.1, 0.9]],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="portfolio_direction_first_quality_guard_938a",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "portfolio_prior": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    policy = result["portfolio_quality_guard_policy"]
    diversity = result["support_diversity_policy"]
    assert result["mode"] == "portfolio_direction_first_quality_guard_938a"
    assert policy["fallback_to_equal_support"] is False
    assert policy["direction_first_candidate_selection"] is True
    assert result["direction_check"]["status"] == "pass"
    assert diversity["portfolio_quality_guard_direction_first"] is True
    assert result["analogue_count"] == 2
    assert 2 in result["window_indices"]
    np.testing.assert_allclose(result["weights"], [0.5, 0.5])


def test_portfolio_quality_guard_mode_falls_back_to_base_prior() -> None:
    base = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="diverse_topk_narrative_start_checked",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
    )
    guarded = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="portfolio_quality_guard_924e",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "portfolio_prior": {0: 0.0, 1: 0.0, 2: 1.0},
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2},
            "probability_temperature": 10.0,
            "min_support_weight_max_threshold": 1.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    assert guarded["window_indices"] == base["window_indices"]
    np.testing.assert_allclose(guarded["weights"], base["weights"])
    assert (
        guarded["portfolio_quality_guard_policy"]["fallback_to_equal_support"] is True
    )
    assert (
        guarded["support_diversity_policy"]["portfolio_quality_guard_fallback"] is True
    )


def test_portfolio_quality_guard_mode_falls_back_on_insufficient_candidate_breadth() -> (
    None
):
    guarded = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06]],
            dtype=np.float32,
        ),
        history_level=_history(),
        train_indices=np.asarray([0, 1, 2]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="portfolio_quality_guard_924e",
        top_k=2,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "portfolio_prior": {0: 0.0, 1: 0.0, 2: 1.0},
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
        quality_guard_min_candidate_mixtures=4,
    )

    assert (
        guarded["portfolio_quality_guard_policy"]["fallback_to_equal_support"] is True
    )
    assert (
        guarded["portfolio_quality_guard_policy"]["fallback_reason"]
        == "insufficient_live_candidate_mixtures"
    )
    assert (
        guarded["support_diversity_policy"]["portfolio_quality_guard_active"] is False
    )


def test_portfolio_quality_guard_falls_back_on_final_direction_reject(monkeypatch):
    original_check = prior_module.direction_check_for_mixture
    calls = {"count": 0}

    def reject_after_base(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] >= 2:
            return {
                "status": "reject",
                "reason": "final_mixed_prefix_direction_mismatch",
                "final_mixture_checked_count": 2,
                "final_mixture_mismatch_count": 1,
            }
        return original_check(*args, **kwargs)

    monkeypatch.setattr(
        prior_module,
        "direction_check_for_mixture",
        reject_after_base,
    )

    guarded = prior_module.build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06], [0.1, 0.9]],
            dtype=np.float32,
        ),
        history_level=np.asarray(
            [
                _history()[0],
                _history()[0],
                _history()[0],
                _history()[0],
            ],
            dtype=np.float32,
        ),
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="portfolio_quality_guard_924e",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "portfolio_prior": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    policy = guarded["portfolio_quality_guard_policy"]
    assert policy["fallback_to_equal_support"] is True
    assert policy["fallback_reason"] == "final_mixed_prefix_direction_mismatch"
    assert (
        guarded["support_diversity_policy"]["portfolio_quality_guard_active"] is False
    )
    assert (
        guarded["support_diversity_policy"]["portfolio_quality_guard_fallback"] is True
    )


def test_portfolio_quality_guard_uses_direction_safe_candidate_after_marginal_reject(
    monkeypatch,
) -> None:
    original_check = prior_module.direction_check_for_mixture
    calls = {"count": 0}

    def reject_only_marginal_selection(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            return {
                "status": "reject",
                "reason": "final_mixed_prefix_direction_mismatch",
                "final_mixture_checked_count": 2,
                "final_mixture_mismatch_count": 1,
            }
        return original_check(*args, **kwargs)

    monkeypatch.setattr(
        prior_module,
        "direction_check_for_mixture",
        reject_only_marginal_selection,
    )

    guarded = prior_module.build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06], [0.1, 0.9]],
            dtype=np.float32,
        ),
        history_level=np.asarray(
            [
                _history()[0],
                _history()[0],
                _history()[0],
                _history()[0],
            ],
            dtype=np.float32,
        ),
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding=_grounding(),
        spec_names=_spec_names(),
        mode="portfolio_quality_guard_924e",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "portfolio_prior": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    policy = guarded["portfolio_quality_guard_policy"]
    diversity = guarded["support_diversity_policy"]
    assert policy["fallback_to_equal_support"] is False
    assert policy["direction_safe_candidate_fallback"] is True
    assert policy["initial_marginal_direction_check"]["status"] == "reject"
    assert guarded["direction_check"]["status"] == "pass"
    assert diversity["portfolio_quality_guard_active"] is True
    assert diversity["portfolio_quality_guard_fallback"] is False
    assert (
        diversity["portfolio_quality_guard_direction_safe_candidate_fallback"] is True
    )


def test_narrative_book_quality_guard_mode_uses_grounded_book_weights() -> None:
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8, 1.7]
    history[:, -1, 26] = [-1.0, -0.9, -0.8, -0.7]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06], [0.1, 0.9]],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding={
            "condition_only_grounding": {
                "current_market_state_implications": [
                    {"market": "CRUDE_OIL", "direction": "up", "confidence": "high"},
                    {"market": "US10Y", "direction": "up", "confidence": "high"},
                ]
            }
        },
        spec_names=_spec_names(),
        mode="narrative_book_quality_guard_926b",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "priors_by_book": {
                "commodity_inflation": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "dollar_liquidity": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
            },
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    weights = dict(zip(result["window_indices"], result["weights"], strict=True))
    policy = result["narrative_book_response_policy"]
    assert result["mode"] == "narrative_book_quality_guard_926b"
    assert policy["fallback_to_equal_support"] is False
    assert policy["book_weights"]["commodity_inflation"] > 0.0
    assert (
        result["support_diversity_policy"]["narrative_book_quality_guard_active"]
        is True
    )
    assert weights[2] > weights[0]
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_narrative_book_direction_first_quality_guard_selects_atomic_candidate() -> (
    None
):
    history = np.zeros((4, 30, 27), dtype=np.float32)
    history[:, :, :25] = 0.2
    history[:, -1, 25] = [2.0, 1.9, 1.8, 1.7]
    history[:, -1, 26] = [-1.0, -0.9, -0.8, -0.7]

    result = build_mixture_memory_prior(
        query_memory=np.asarray([1.0, 0.0], dtype=np.float32),
        memory_targets=np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.94, 0.06], [0.1, 0.9]],
            dtype=np.float32,
        ),
        history_level=history,
        train_indices=np.asarray([0, 1, 2, 3]),
        query_window_index=0,
        grounding={
            "condition_only_grounding": {
                "current_market_state_implications": [
                    {"market": "CRUDE_OIL", "direction": "up", "confidence": "high"},
                    {"market": "US10Y", "direction": "up", "confidence": "high"},
                ]
            }
        },
        spec_names=_spec_names(),
        mode="narrative_book_direction_first_quality_guard_938c",
        top_k=3,
        temperature=0.2,
        start_distance_threshold_z=100.0,
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_max_pairwise_cosine=1.0,
        quality_guard_context={
            "priors_by_book": {
                "commodity_inflation": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "dollar_liquidity": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
            },
            "crps_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "energy_prior": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2},
            "probability_temperature": 0.25,
            "min_support_weight_max_threshold": 0.0,
            "min_support_weight_max_quantile": 0.25,
        },
        quality_guard_candidate_pool_size=3,
        quality_guard_mixture_size=2,
    )

    policy = result["narrative_book_response_policy"]
    diversity = result["support_diversity_policy"]
    assert result["mode"] == "narrative_book_direction_first_quality_guard_938c"
    assert policy["fallback_to_equal_support"] is False
    assert policy["direction_first_candidate_selection"] is True
    assert result["direction_check"]["status"] != "reject"
    assert diversity["narrative_book_quality_guard_direction_first"] is True
    assert result["analogue_count"] == 2
    assert 2 in result["window_indices"]
    np.testing.assert_allclose(result["weights"], [0.5, 0.5])


def test_direction_check_warns_on_weak_support_before_final_mismatch() -> None:
    check = direction_check_for_mixture(
        candidate_details=[
            {
                "window_index": 0,
                "recent_prefix_checked": 2,
                "recent_prefix_match_count": 1,
                "recent_prefix_mismatches": 1,
                "recent_prefix_alignment_status": "warning",
            }
        ],
        weights=np.asarray([1.0], dtype=np.float32),
        final_mixture_alignment={
            "checked_count": 2,
            "match_count": 2,
            "mismatch_count": 0,
            "status": "pass",
        },
        min_support_match_rate=0.75,
    )

    assert check["status"] == "warning"
    assert check["reason"] == "selected_support_direction_weak"


def test_weighted_rows_are_compatible_with_alignment_helper() -> None:
    rows = weighted_prefix_terminal_rows(
        history_level=_history(),
        window_indices=np.asarray([0, 2]),
        weights=np.asarray([0.8, 0.2], dtype=np.float32),
        spec_names=_spec_names(),
    )

    alignment = market_implication_alignment(
        grounding=_grounding(),
        scenario_rows=rows,
    )

    assert alignment["status"] == "pass"
    assert alignment["mismatch_count"] == 0


def test_market_alignment_skips_static_current_state_level_language() -> None:
    assert not is_terminal_direction_checkable(
        {
            "market": "VIX",
            "direction": "up",
            "horizon": "current_state",
            "evidence": ["volatility remains elevated"],
        }
    )
    alignment = market_implication_alignment(
        grounding={
            "market_implications": [
                {
                    "market": "VIX",
                    "direction": "up",
                    "horizon": "current_state",
                    "evidence": ["volatility remains elevated"],
                },
                {
                    "market": "USDJPY",
                    "direction": "up",
                    "horizon": "current_state",
                    "evidence": ["USDJPY is moving higher"],
                },
            ]
        },
        scenario_rows=[
            {"Market": "VIX", "Mean Terminal Delta": -1.0},
            {"Market": "USDJPY", "Mean Terminal Delta": 1.0},
        ],
    )

    assert alignment["checked_count"] == 1
    assert alignment["skipped_count"] == 1
    assert alignment["status"] == "pass"


def test_run_analogue_mixture_prior_writes_summary(tmp_path) -> None:
    report_path = tmp_path / "case" / "prefix_latent_story_smoke_report.json"
    arrays_path = tmp_path / "case" / "prefix_latent_story_smoke_arrays.npz"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "cached_query": {
                    "window_index": 0,
                    "grounding": _grounding(),
                }
            }
        ),
        encoding="utf-8",
    )
    np.savez(arrays_path, text_memory=np.asarray([[1.0, 0.0]], dtype=np.float32))
    casebook_path = tmp_path / "casebook.json"
    casebook_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "risk_on",
                        "story": "risk-on",
                        "selected_start_status": "warning",
                        "market_alignment": {
                            "checked_count": 2,
                            "mismatch_count": 1,
                        },
                        "artifact_paths": {"prefix_report": str(report_path)},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.npz"
    np.savez(
        oracle_path,
        history_level=_history(),
        true_memory=np.asarray(
            [[0.7, 0.3], [1.0, 0.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"state_specs": [{"name": name} for name in _spec_names()]},
        checkpoint_path,
    )

    summary = run_analogue_mixture_prior(
        SimpleNamespace(
            casebook_summary=str(casebook_path),
            oracle_arrays=str(oracle_path),
            checkpoint=str(checkpoint_path),
            output_dir=str(tmp_path / "out"),
            top_k=2,
            temperature=0.2,
            start_distance_threshold_z=100.0,
            start_distance_penalty=0.0,
            implication_alignment_weight=1.0,
            diverse_max_pairwise_cosine=0.99,
        )
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 1
    assert "soft_topk_combined" in summary["variant_totals"]
    assert (tmp_path / "out" / "analogue_mixture_prior_summary.json").exists()
