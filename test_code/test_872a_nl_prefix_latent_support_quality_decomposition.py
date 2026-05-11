import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_support_quality_decomposition import (
    decompose_support_quality,
    support_overlap_fraction,
)


def _bridge_report(top_windows: list[str]) -> dict:
    return {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 10,
                    "window_id": "q10",
                    "role": "anchor",
                    "top_train_pool": [
                        {
                            "window_id": window_id,
                            "window_index": idx,
                            "cosine": 0.9 - 0.01 * idx,
                        }
                        for idx, window_id in enumerate(top_windows)
                    ],
                }
            ]
        }
    }


def _scenario_report(
    *,
    replay_energy: float,
    generator_energy: float,
    replay_improvement: float,
    generator_improvement: float,
) -> dict:
    return {
        "summary": {
            "historical_replay_topk": {
                "energy_score_z_improvement_vs_persistence": replay_improvement,
                "ensemble_crps_z_improvement_vs_persistence": replay_improvement,
            },
            "narrative_generator_topk": {
                "energy_score_z_improvement_vs_persistence": generator_improvement,
                "ensemble_crps_z_improvement_vs_persistence": generator_improvement,
            },
        },
        "window_scores": [
            {
                "window_index": 10,
                "methods": {
                    "historical_replay_topk": {
                        "energy_score_z": replay_energy,
                        "ensemble_crps_z": replay_energy,
                        "coverage_80": 0.4,
                        "mean_path_mae_z": 1.0,
                        "terminal_mae_z": 1.0,
                    },
                    "narrative_generator_topk": {
                        "energy_score_z": generator_energy,
                        "ensemble_crps_z": generator_energy,
                        "coverage_80": 0.6,
                        "mean_path_mae_z": 1.0,
                        "terminal_mae_z": 1.0,
                    },
                },
            }
        ],
    }


def test_support_overlap_fraction_uses_smaller_support_set() -> None:
    assert support_overlap_fraction(["a", "b", "c"], ["b", "c"]) == 1.0
    assert support_overlap_fraction(["a", "b"], ["c", "d"]) == 0.0


def test_decompose_support_quality_identifies_replay_generator_mismatch() -> None:
    report = decompose_support_quality(
        base_bridge_report=_bridge_report(["a", "b", "c"]),
        candidate_bridge_report=_bridge_report(["a", "d", "e"]),
        base_scenario_report=_scenario_report(
            replay_energy=1.0,
            generator_energy=1.0,
            replay_improvement=0.10,
            generator_improvement=0.20,
        ),
        candidate_scenario_report=_scenario_report(
            replay_energy=0.9,
            generator_energy=1.1,
            replay_improvement=0.12,
            generator_improvement=0.18,
        ),
        top_k=3,
    )

    assert report["summary"]["window_count"] == 1
    assert np.isclose(report["summary"]["mean_topk_overlap_fraction"], 1 / 3)
    assert report["summary"]["replay_better_generator_worse_count"] == 1
    assert report["decision"]["mechanism"] == "support_replay_generator_mismatch"
