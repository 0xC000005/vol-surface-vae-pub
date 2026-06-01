import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_quality_guard_policy import (
    build_quality_guard_bridge,
    quality_guard_scores,
    select_quality_guard_support,
)


def test_quality_guard_scores_penalize_only_negative_broad_quality():
    support_rows = [[1], [2], [3]]
    scores = quality_guard_scores(
        support_rows=support_rows,
        portfolio_prior={1: 0.5, 2: 0.5, 3: 0.1},
        crps_prior={1: 0.2, 2: -0.4, 3: 0.8},
        energy_prior={1: 0.3, 2: -0.1, 3: -0.2},
    )

    np.testing.assert_allclose(scores["quality_penalty"], [0.0, -0.5, -0.2])
    np.testing.assert_allclose(scores["total"], [0.5, 0.0, -0.1])


def test_quality_guard_scores_keeps_portfolio_order_when_quality_is_ok():
    support_rows = [[1], [2]]
    scores = quality_guard_scores(
        support_rows=support_rows,
        portfolio_prior={1: 0.1, 2: 0.9},
        crps_prior={1: 0.2, 2: 0.1},
        energy_prior={1: 0.3, 2: 0.1},
    )

    assert scores["total"][1] > scores["total"][0]


def test_select_quality_guard_support_is_reusable_policy_core():
    candidates = [
        _candidate_row("candidate_a", 4, [1, 2]),
        _candidate_row("candidate_b", 4, [1, 3]),
    ]

    selection = select_quality_guard_support(
        candidates,
        portfolio_prior={1: 0.1, 2: 0.1, 3: 0.9},
        crps_prior={1: 0.2, 2: 0.2, 3: 0.2},
        energy_prior={1: 0.2, 2: 0.2, 3: 0.2},
        probability_temperature=0.25,
    )

    assert selection["fallback"] is False
    policy = selection["support_policy"]
    assert policy["name"] == "portfolio_response_quality_guard_prior"
    assert policy["selected_query_id"] == "candidate_b"
    assert policy["fallback_to_equal_support"] is False
    weights = dict(
        zip(
            policy["selected_support_window_indices"],
            policy["selected_support_weights"],
            strict=True,
        )
    )
    assert weights[3] > weights[2]


def test_select_quality_guard_support_keeps_fallback_contract():
    candidates = [
        _candidate_row("candidate_a", 4, [1, 2]),
        _candidate_row("candidate_b", 4, [1, 3]),
    ]

    selection = select_quality_guard_support(
        candidates,
        portfolio_prior={1: 0.1, 2: 0.1, 3: 0.9},
        crps_prior={1: 0.2, 2: 0.2, 3: 0.2},
        energy_prior={1: 0.2, 2: 0.2, 3: 0.2},
        probability_temperature=10.0,
        min_support_weight_max_threshold=1.0,
        min_support_weight_max_quantile=0.25,
    )

    assert selection["fallback"] is True
    assert selection["selected"] is None
    assert selection["support_policy"]["fallback_to_equal_support"] is True
    assert selection["support_policy"]["support_weight_max"] <= 1.0


def _candidate_row(query_id: str, window_index: int, supports: list[int]) -> dict:
    return {
        "query_id": query_id,
        "window_index": window_index,
        "window_id": f"q{window_index}",
        "embedding_index": 0,
        "candidate_mixture_rank": 1,
        "candidate_mixture_positions": [1, 2],
        "candidate_support_window_indices": supports,
        "top_train_cosines": [0.9, 0.8],
        "top_train_pool": [
            {"window_index": support, "window_id": f"s{support}", "cosine": 0.9}
            for support in supports
        ],
    }


def _scenario_report(rows: list[tuple[str, float, float, float]]) -> dict:
    return {
        "window_scores": [
            {
                "query_id": query_id,
                "methods": {
                    "narrative_generator_topk": {
                        "portfolio_reliable_path_score_z": portfolio,
                        "ensemble_crps_z": crps,
                        "energy_score_z": energy,
                    }
                },
            }
            for query_id, portfolio, crps, energy in rows
        ]
    }


def test_build_quality_guard_bridge_records_sources_and_reweights_supports():
    train_bridge = {
        "evaluation": {
            "heldout_examples": [
                _candidate_row("train_a", 0, [1, 2]),
                _candidate_row("train_b", 0, [1, 3]),
            ]
        }
    }
    # train_a and train_b have the same portfolio loss, but train_a has worse
    # broad quality. The guard should prefer train_b's support composition.
    train_report = _scenario_report(
        [
            ("train_a", 0.3, 5.0, 5.0),
            ("train_b", 0.3, 0.3, 0.3),
        ]
    )
    candidate_bridge = {
        "evaluation": {
            "heldout_examples": [
                _candidate_row("test_a", 4, [1, 2]),
                _candidate_row("test_b", 4, [1, 3]),
            ]
        }
    }
    base_bridge = {
        "evaluation": {
            "heldout_examples": [
                {"window_index": 4, "window_id": "q4", "embedding_index": 0}
            ]
        }
    }
    bridge = build_quality_guard_bridge(
        train_candidate_bridge=train_bridge,
        train_label_report=train_report,
        base_bridge_report=base_bridge,
        candidate_bridge_report=candidate_bridge,
        bridge_arrays={
            "condition_vectors": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "memory_targets": np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.5, 0.5]],
                dtype=np.float32,
            ),
        },
        history_arrays={"history_level": np.zeros((5, 2, 39), dtype=np.float32)},
        source_artifacts={"train_label_report": "train.json"},
    )

    row = bridge["evaluation"]["heldout_examples"][0]
    assert row["support_policy"]["name"] == "portfolio_response_quality_guard_prior"
    assert row["support_policy"]["selected_query_id"] == "test_b"
    weights = {item["window_index"]: item["weight"] for item in row["top_train_pool"]}
    assert weights[3] > weights[2]
    assert bridge["mixture_policy"]["source_artifacts"]["train_label_report"] == "train.json"


def test_build_quality_guard_bridge_can_fallback_on_low_support_concentration():
    train_bridge = {
        "evaluation": {
            "heldout_examples": [
                _candidate_row("train_a", 0, [1, 2]),
                _candidate_row("train_b", 0, [1, 3]),
            ]
        }
    }
    train_report = _scenario_report(
        [
            ("train_a", 0.3, 0.3, 0.3),
            ("train_b", 0.3, 0.4, 0.4),
        ]
    )
    candidate_bridge = {
        "evaluation": {
            "heldout_examples": [
                _candidate_row("test_a", 4, [1, 2]),
                _candidate_row("test_b", 4, [1, 3]),
            ]
        }
    }
    base_bridge = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 4,
                    "window_id": "q4",
                    "embedding_index": 0,
                    "top_train_pool": [{"window_index": 99}],
                }
            ]
        }
    }
    bridge = build_quality_guard_bridge(
        train_candidate_bridge=train_bridge,
        train_label_report=train_report,
        base_bridge_report=base_bridge,
        candidate_bridge_report=candidate_bridge,
        bridge_arrays={
            "condition_vectors": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "memory_targets": np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.5, 0.5]],
                dtype=np.float32,
            ),
        },
        history_arrays={"history_level": np.zeros((5, 2, 39), dtype=np.float32)},
        min_support_weight_max_quantile=1.0,
    )

    row = bridge["evaluation"]["heldout_examples"][0]
    assert row["support_policy"]["fallback_to_equal_support"] is True
    assert row["top_train_pool"] == [{"window_index": 99}]
