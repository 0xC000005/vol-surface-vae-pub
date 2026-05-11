import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_generator_calibrated_support import (
    build_self_calibration_bridge_report,
    extract_generator_quality,
    rerank_bridge_report_by_quality,
    rerank_pool_by_quality,
    train_indices_from_report,
)


def _source_bridge_report() -> dict:
    return {
        "window_metadata": [
            {"window_index": 0, "window_id": "w0", "manifest_split": "train"},
            {"window_index": 1, "window_id": "w1", "manifest_split": "train"},
            {"window_index": 2, "window_id": "w2", "manifest_split": "test"},
        ],
        "window_indices": [0, 1, 2],
        "split": {"train_indices": [0, 1], "test_indices": [2]},
    }


def test_train_indices_from_report_prefers_split() -> None:
    assert train_indices_from_report(_source_bridge_report()) == [0, 1]


def test_build_self_calibration_bridge_report_retrieves_self() -> None:
    report = build_self_calibration_bridge_report(
        _source_bridge_report(),
        max_windows=1,
    )

    rows = report["evaluation"]["heldout_examples"]
    assert len(rows) == 1
    assert rows[0]["window_index"] == 0
    assert rows[0]["top_train_pool"][0]["window_index"] == 0
    assert report["split"]["source"] == "train_self_calibration"


def test_extract_generator_quality_reads_scenario_scores() -> None:
    quality = extract_generator_quality(
        {
            "window_scores": [
                {
                    "window_index": 7,
                    "window_id": "w7",
                    "methods": {
                        "narrative_generator_topk": {
                            "energy_score_z": 0.8,
                            "ensemble_crps_z": 0.6,
                            "coverage_80": 0.5,
                        }
                    },
                }
            ]
        }
    )

    assert quality[7]["quality_score"] == 0.8
    assert quality[7]["ensemble_crps_z"] == 0.6


def test_rerank_pool_by_quality_moves_measured_good_candidates_first() -> None:
    pool = [
        {"window_index": 10, "window_id": "bad", "cosine": 0.99},
        {"window_index": 11, "window_id": "good", "cosine": 0.80},
        {"window_index": 12, "window_id": "unknown", "cosine": 0.79},
    ]
    quality = {
        10: {"quality_score": 2.0, "metric": "energy_score_z"},
        11: {"quality_score": 1.0, "metric": "energy_score_z"},
    }

    reranked = rerank_pool_by_quality(pool, quality)

    assert [row["window_index"] for row in reranked] == [11, 10, 12]
    assert reranked[0]["original_support_rank"] == 2
    assert reranked[2]["generator_calibration_score"] is None


def test_rerank_bridge_report_updates_summary_and_policy() -> None:
    candidate = {
        "evaluation": {
            "heldout_window_count": 1,
            "heldout_example_count": 1,
            "heldout_examples": [
                {
                    "window_index": 20,
                    "window_id": "q20",
                    "embedding_index": 0,
                    "role": "anchor",
                    "kind": "query",
                    "target_cosine": 0.5,
                    "target_mse": 0.1,
                    "true_rank_full_pool": 2,
                    "true_rank_test_pool": 1,
                    "top_train_pool": [
                        {"window_index": 10, "window_id": "bad", "cosine": 0.99},
                        {"window_index": 11, "window_id": "good", "cosine": 0.80},
                    ],
                }
            ],
            "hard_negative_separation": {
                "window_count": 0,
                "mean_hard_margin": None,
                "mean_negative_gap": None,
                "windows": [],
            },
        }
    }

    report = rerank_bridge_report_by_quality(
        candidate,
        {
            10: {"quality_score": 2.0, "metric": "energy_score_z"},
            11: {"quality_score": 1.0, "metric": "energy_score_z"},
        },
    )

    pool = report["evaluation"]["heldout_examples"][0]["top_train_pool"]
    assert [row["window_index"] for row in pool] == [11, 10]
    assert report["support_policy"]["candidate_rows_reranked"] == 1
    assert report["summary"]["heldout_example_count"] == 1
