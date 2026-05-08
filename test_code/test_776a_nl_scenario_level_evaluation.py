import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_scenario_level_evaluation import (
    build_delta_scale,
    bridge_local_to_block_indices,
    bridge_report_arrays_path,
    direct_memory_condition_for_query,
    future_delta_paths,
    memory_residual_method_name,
    parse_memory_residual_alphas,
    score_sample_distribution,
    select_heldout_query_rows,
    summarize_method_scores,
)


def _bridge_report() -> dict:
    return {
        "evaluation": {
            "heldout_examples": [
                {
                    "window_index": 2,
                    "window_id": "w2",
                    "role": "anchor",
                    "kind": "primary",
                    "top_train_pool": [
                        {"window_index": 0, "window_id": "w0", "cosine": 0.9},
                        {"window_index": 1, "window_id": "w1", "cosine": 0.8},
                    ],
                },
                {
                    "window_index": 2,
                    "window_id": "w2",
                    "role": "positive",
                    "kind": "market",
                    "top_train_pool": [
                        {"window_index": 1, "window_id": "w1", "cosine": 0.7},
                    ],
                },
                {
                    "window_index": 3,
                    "window_id": "w3",
                    "role": "anchor",
                    "kind": "primary",
                    "top_train_pool": [
                        {"window_index": 1, "window_id": "w1", "cosine": 0.6},
                    ],
                },
            ]
        }
    }


def test_select_heldout_query_rows_keeps_one_anchor_per_window() -> None:
    rows = select_heldout_query_rows(_bridge_report(), role="anchor", max_windows=1)

    assert len(rows) == 1
    assert rows[0]["window_index"] == 2
    assert [item["window_index"] for item in rows[0]["top_train_pool"]] == [0, 1]


def test_bridge_local_to_block_indices_uses_manifest_window_indices() -> None:
    report = {"window_indices": [4, 1, 8]}

    mapping = bridge_local_to_block_indices(report, n_windows=3)

    assert mapping.tolist() == [4, 1, 8]


def test_bridge_report_arrays_path_prefers_explicit_then_artifact() -> None:
    report = {"artifact_paths": {"arrays": "outputs/bridge_arrays.npz"}}

    assert str(bridge_report_arrays_path(report, explicit_path="manual.npz")) == "manual.npz"
    assert str(bridge_report_arrays_path(report, explicit_path=None)) == "outputs/bridge_arrays.npz"


def test_direct_memory_condition_for_query_uses_embedding_index() -> None:
    condition_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    condition = direct_memory_condition_for_query(
        {"embedding_index": 1},
        condition_vectors,
    )

    np.testing.assert_allclose(condition, [0.0, 2.0, 0.0])


def test_parse_memory_residual_alphas_accepts_comma_separated_values() -> None:
    alphas = parse_memory_residual_alphas("0, 0.10,0.25, 1")

    assert alphas == [0.0, 0.1, 0.25, 1.0]


def test_memory_residual_method_name_is_stable_for_report_keys() -> None:
    assert memory_residual_method_name("narrative_residual_memory", 0.25) == (
        "narrative_residual_memory_a025"
    )


def test_future_delta_paths_subtracts_last_history_state() -> None:
    history = np.asarray(
        [
            [[1.0, 2.0], [2.0, 3.0]],
            [[10.0, 20.0], [11.0, 19.0]],
        ],
        dtype=np.float32,
    )
    future = np.asarray(
        [
            [[3.0, 1.0], [5.0, 4.0]],
            [[10.0, 18.0], [13.0, 21.0]],
        ],
        dtype=np.float32,
    )

    delta = future_delta_paths(history, future)

    np.testing.assert_allclose(
        delta,
        np.asarray(
            [
                [[1.0, -2.0], [3.0, 1.0]],
                [[-1.0, -1.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_score_sample_distribution_rewards_target_like_samples() -> None:
    target = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    scale = np.ones_like(target)
    good_samples = np.stack([target - 0.1, target, target + 0.1], axis=0)
    bad_samples = np.zeros((3, 2, 2), dtype=np.float32)

    good = score_sample_distribution(good_samples, target, scale=scale)
    bad = score_sample_distribution(bad_samples, target, scale=scale)

    assert good["mean_path_mae_z"] < bad["mean_path_mae_z"]
    assert good["energy_score_z"] < bad["energy_score_z"]
    assert good["coverage_80"] == 1.0


def test_build_delta_scale_uses_train_variability_floor() -> None:
    train_delta = np.asarray(
        [
            [[1.0, 1.0], [2.0, 5.0]],
            [[1.0, 3.0], [2.0, 9.0]],
        ],
        dtype=np.float32,
    )

    scale = build_delta_scale(train_delta, floor=0.5)

    assert scale.shape == (2, 2)
    assert np.all(scale >= 0.5)
    assert scale[0, 1] > scale[0, 0]


def test_summarize_method_scores_computes_improvement_against_persistence() -> None:
    window_scores = [
        {
            "methods": {
                "persistence": {"mean_path_mae_z": 2.0, "energy_score_z": 3.0},
                "narrative_generator": {"mean_path_mae_z": 1.0, "energy_score_z": 2.0},
            }
        },
        {
            "methods": {
                "persistence": {"mean_path_mae_z": 4.0, "energy_score_z": 5.0},
                "narrative_generator": {"mean_path_mae_z": 3.0, "energy_score_z": 4.0},
            }
        },
    ]

    summary = summarize_method_scores(window_scores, baseline="persistence")

    assert summary["persistence"]["mean_path_mae_z_mean"] == 3.0
    assert summary["narrative_generator"]["mean_path_mae_z_mean"] == 2.0
    assert np.isclose(
        summary["narrative_generator"]["mean_path_mae_z_improvement_vs_persistence"],
        1.0 / 3.0,
    )
