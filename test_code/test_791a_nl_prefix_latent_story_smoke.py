import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (
    build_live_story_variant_rows,
    generated_delta_samples_to_states,
    narrative_text_for_query,
    resolve_start_window_index,
    select_cached_story_query,
    window_metadata_by_bridge_local_index,
)


def _bridge_report() -> dict:
    return {
        "window_metadata": [
            {
                "window_id": "joint39_val_0000",
                "window_index": 0,
                "source_index": 4000,
                "manifest_split": "train",
            },
            {
                "window_id": "joint39_val_0100",
                "window_index": 100,
                "source_index": 4100,
                "manifest_split": "test",
            },
        ],
        "evaluation": {
            "heldout_examples": [
                {
                    "role": "positive",
                    "kind": "description_risk_manager",
                    "window_id": "joint39_val_0100",
                    "window_index": 5,
                    "embedding_index": 100,
                },
                {
                    "role": "anchor",
                    "kind": "revised_market_description",
                    "window_id": "joint39_val_0100",
                    "window_index": 5,
                    "embedding_index": 101,
                },
                {
                    "role": "anchor",
                    "kind": "revised_market_description",
                    "window_id": "joint39_val_0200",
                    "window_index": 9,
                    "embedding_index": 102,
                },
            ]
        }
    }


def _pipeline_report() -> dict:
    return {
        "narrative_bundles": [
            {
                "window_id": "joint39_val_0100",
                "window_index": 5,
                "narratives": [
                    {"id": "description_risk_manager", "text": "Risk-manager story."},
                    {
                        "id": "revised_market_description",
                        "text": "Grounded market description.",
                    },
                ],
            }
        ]
    }


def test_select_cached_story_query_filters_role_kind_and_window_id() -> None:
    row = select_cached_story_query(
        _bridge_report(),
        role="anchor",
        kind="revised_market_description",
        window_id="joint39_val_0200",
        query_index=0,
    )

    assert row["window_index"] == 9
    assert row["embedding_index"] == 102


def test_narrative_text_for_query_uses_matching_narrative_kind() -> None:
    row = {
        "window_id": "joint39_val_0100",
        "kind": "description_risk_manager",
    }

    assert narrative_text_for_query(_pipeline_report(), row) == "Risk-manager story."


def test_window_metadata_by_bridge_local_index_uses_bridge_local_rows() -> None:
    metadata = window_metadata_by_bridge_local_index(_bridge_report())

    assert metadata[1]["window_id"] == "joint39_val_0100"
    assert metadata[1]["window_index"] == 100
    assert metadata[1]["source_index"] == 4100


def test_resolve_start_window_index_supports_nearest_and_explicit_start() -> None:
    start = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [4.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float32,
    )

    nearest = resolve_start_window_index(
        query_window_index=0,
        start_state=start,
        train_indices=np.asarray([1, 3], dtype=np.int64),
        start_mode="nearest_train_start",
    )
    explicit = resolve_start_window_index(
        query_window_index=0,
        start_state=start,
        train_indices=np.asarray([1, 3], dtype=np.int64),
        start_mode="explicit_start_window",
        explicit_start_window_index=2,
    )

    assert nearest["start_window_index"] == 1
    assert nearest["variant"] == "nearest_train_start"
    assert nearest["start_distance_z"] >= 0.0
    assert explicit["start_window_index"] == 2
    assert explicit["variant"] == "explicit_start_window"


def test_build_live_story_variant_rows_adds_original_baseline_for_changed_start() -> None:
    start = np.asarray([[0.0], [1.0], [4.0]], dtype=np.float32)
    rows = build_live_story_variant_rows(
        query_row={"window_index": 0, "window_id": "joint39_val_0000"},
        start_state=start,
        train_indices=np.asarray([1, 2], dtype=np.int64),
        start_mode="farthest_train_start",
        include_original_baseline=True,
    )

    assert [row["variant"] for row in rows] == ["original", "farthest_train_start"]
    assert rows[0]["start_window_index"] == 0
    assert rows[1]["query_window_index"] == 0
    assert rows[1]["start_window_index"] in {1, 2}


def test_generated_delta_samples_to_states_adds_current_state_per_variant() -> None:
    deltas = np.ones((2, 3, 4, 5), dtype=np.float32)
    current = np.asarray(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=np.float32,
    )

    states = generated_delta_samples_to_states(deltas, current)

    assert states.shape == (2, 3, 4, 5)
    assert np.allclose(states[0, 0, 0], current[0] + 1.0)
    assert np.allclose(states[1, 2, 3], current[1] + 1.0)
