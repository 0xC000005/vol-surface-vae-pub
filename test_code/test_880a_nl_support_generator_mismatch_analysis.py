import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_support_generator_mismatch_analysis import (
    attach_actual_rollout_decomposition,
    compare_support_policies,
    extract_generator_self_quality,
)


def _toy_arrays():
    history = np.zeros((5, 30, 2), dtype=np.float32)
    history[:, -1, 0] = np.asarray([0.0, 0.1, 0.2, 2.0, 2.2], dtype=np.float32)
    future = np.zeros((5, 3, 2), dtype=np.float32)
    future[0, :, 0] = [1.0, 1.0, 1.0]
    future[1, :, 0] = [1.1, 1.1, 1.1]
    future[2, :, 0] = [1.2, 1.2, 1.2]
    future[3, :, 0] = [0.9, 0.9, 0.9]
    future[4, :, 0] = [0.8, 0.8, 0.8]
    delta_scale = np.ones((3, 2), dtype=np.float32)
    return history, future, delta_scale


def test_extract_generator_self_quality_maps_train_windows() -> None:
    report = {
        "window_scores": [
            {
                "window_index": 7,
                "window_id": "joint39_val_0007",
                "methods": {
                    "narrative_generator_topk": {
                        "energy_score_z": 0.9,
                        "ensemble_crps_z": 0.7,
                        "coverage_80": 0.6,
                    }
                },
            }
        ]
    }

    quality = extract_generator_self_quality(report)

    assert quality[7]["energy_score_z"] == 0.9
    assert quality[7]["ensemble_crps_z"] == 0.7
    assert quality[7]["coverage_80"] == 0.6


def test_compare_support_policies_detects_replay_generator_mismatch() -> None:
    history, future, delta_scale = _toy_arrays()
    baseline = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [
                        {"window_index": 1, "window_id": "support_1", "cosine": 0.8},
                        {"window_index": 2, "window_id": "support_2", "cosine": 0.7},
                    ],
                }
            ]
        }
    }
    candidate = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [
                        {"window_index": 3, "window_id": "support_3", "cosine": 0.6},
                        {"window_index": 4, "window_id": "support_4", "cosine": 0.5},
                    ],
                }
            ]
        }
    }
    # Candidate supports replay the query future slightly better, but the frozen
    # generator is known to be less calibrated on those support regimes.
    generator_quality = {
        1: {"energy_score_z": 0.6, "ensemble_crps_z": 0.5, "coverage_80": 0.7},
        2: {"energy_score_z": 0.7, "ensemble_crps_z": 0.6, "coverage_80": 0.7},
        3: {"energy_score_z": 1.0, "ensemble_crps_z": 0.9, "coverage_80": 0.4},
        4: {"energy_score_z": 1.1, "ensemble_crps_z": 1.0, "coverage_80": 0.4},
    }

    report = compare_support_policies(
        baseline_bridge=baseline,
        candidate_bridge=candidate,
        history_level=history,
        future_delta=future,
        delta_scale=delta_scale,
        generator_quality=generator_quality,
        top_k=2,
    )

    assert report["summary"]["replay_loss_delta_mean"] < 0.0
    assert report["summary"]["generator_energy_delta_mean"] > 0.0
    assert report["summary"]["generator_coverage_delta_mean"] < 0.0
    assert report["decision"]["mechanism"] == "replay_better_generator_worse"


def test_compare_support_policies_marks_generator_aligned_candidate() -> None:
    history, future, delta_scale = _toy_arrays()
    baseline = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [
                        {"window_index": 1, "window_id": "support_1", "cosine": 0.8}
                    ],
                }
            ]
        }
    }
    candidate = {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 0,
                    "window_id": "q0",
                    "top_train_pool": [
                        {"window_index": 2, "window_id": "support_2", "cosine": 0.7}
                    ],
                }
            ]
        }
    }
    generator_quality = {
        1: {"energy_score_z": 0.8, "ensemble_crps_z": 0.8, "coverage_80": 0.5},
        2: {"energy_score_z": 0.6, "ensemble_crps_z": 0.6, "coverage_80": 0.7},
    }

    report = compare_support_policies(
        baseline_bridge=baseline,
        candidate_bridge=candidate,
        history_level=history,
        future_delta=future,
        delta_scale=delta_scale,
        generator_quality=generator_quality,
        top_k=1,
    )

    assert report["summary"]["generator_energy_delta_mean"] < 0.0
    assert report["decision"]["mechanism"] == "generator_aligned_candidate"


def test_attach_actual_rollout_decomposition_detects_proxy_false_positive() -> None:
    report = {
        "summary": {
            "replay_loss_delta_mean": -0.1,
            "generator_energy_delta_mean": -0.1,
            "generator_crps_delta_mean": -0.1,
        },
        "decision": {
            "mechanism": "generator_aligned_candidate",
            "promote_candidate": False,
            "next_step": "",
        },
    }
    decomposition = {
        "summary": {
            "summary_improvement_deltas": {
                "historical_replay_topk_crps": 0.02,
                "historical_replay_topk_energy": 0.01,
                "narrative_generator_topk_crps": -0.003,
                "narrative_generator_topk_energy": -0.002,
            }
        }
    }

    updated = attach_actual_rollout_decomposition(report, decomposition)

    assert updated["decision"]["mechanism"] == "generator_proxy_false_positive"
    assert updated["actual_rollout_comparison"]["candidate_generator_regressed"] is True
