import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from test_code.test_784a_nl_risk_manager_story_smoke import _grounding

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (
    _render_markdown,
    build_live_story_variant_rows,
    build_live_story_condition_memory,
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


def test_resolve_start_window_index_supports_memory_nearest_start() -> None:
    start = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    memory_targets = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    selected = resolve_start_window_index(
        query_window_index=0,
        start_state=start,
        train_indices=np.asarray([1, 2], dtype=np.int64),
        start_mode="memory_nearest_start",
        query_memory=np.asarray([0.0, 1.0], dtype=np.float32),
        memory_targets=memory_targets,
    )

    assert selected["variant"] == "memory_nearest_start"
    assert selected["start_window_index"] == 2
    assert selected["memory_support_cosine"] > 0.99


def test_resolve_start_window_index_supports_balanced_memory_start() -> None:
    start = np.asarray([[0.0], [1.0], [10.0]], dtype=np.float32)
    memory_targets = np.asarray(
        [
            [1.0, 0.0],
            [0.4, 0.8],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    selected = resolve_start_window_index(
        query_window_index=0,
        start_state=start,
        train_indices=np.asarray([1, 2], dtype=np.int64),
        start_mode="balanced_memory_start",
        query_memory=np.asarray([0.0, 1.0], dtype=np.float32),
        memory_targets=memory_targets,
        start_distance_threshold_z=0.5,
        start_distance_penalty=0.02,
    )

    assert selected["variant"] == "balanced_memory_start"
    assert selected["start_window_index"] == 1
    assert selected["start_selection_method"] == "max_memory_inside_start_threshold"
    assert selected["memory_support_rank"] == 2
    assert selected["candidate_count_inside_distance"] == 1


def test_build_live_story_variant_rows_adds_original_baseline_for_changed_start() -> None:
    start = np.asarray([[0.0], [1.0], [4.0]], dtype=np.float32)
    memory_targets = np.asarray([[1.0, 0.0], [0.4, 0.8], [0.0, 1.0]], dtype=np.float32)
    rows = build_live_story_variant_rows(
        query_row={"window_index": 0, "window_id": "joint39_val_0000"},
        start_state=start,
        train_indices=np.asarray([1, 2], dtype=np.int64),
        start_mode="memory_nearest_start",
        include_original_baseline=True,
        query_memory=np.asarray([0.0, 1.0], dtype=np.float32),
        memory_targets=memory_targets,
    )

    assert [row["variant"] for row in rows] == ["original", "memory_nearest_start"]
    assert rows[0]["start_window_index"] == 0
    assert rows[1]["query_window_index"] == 0
    assert rows[1]["start_window_index"] == 2
    assert rows[1]["memory_support_cosine"] > rows[0]["memory_support_cosine"]


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


def test_build_live_story_condition_memory_uses_embedder_and_adapter() -> None:
    calls = {}

    def fake_embedder(texts, *, model, dotenv_path, batch_size):
        calls["texts"] = texts
        calls["model"] = model
        calls["dotenv_path"] = dotenv_path
        calls["batch_size"] = batch_size
        return np.asarray([[3.0, 4.0]], dtype=np.float32)

    class FakeAdapter:
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            calls["adapter_input_norm"] = float(torch.linalg.norm(value).item())
            return torch.asarray([[1.0, 2.0, 3.0]], dtype=torch.float32)

    def fake_loader(path, *, embedding_dim, condition_dim):
        calls["adapter_path"] = path
        calls["embedding_dim"] = embedding_dim
        calls["condition_dim"] = condition_dim
        return FakeAdapter()

    result = build_live_story_condition_memory(
        story="A fragile risk-on rebound.",
        grounding=_grounding(),
        embedding_model="fake-embedding-model",
        bridge_adapter="fake_adapter.pt",
        condition_dim=3,
        dotenv_path=".env.test",
        embedder=fake_embedder,
        adapter_loader=fake_loader,
    )

    assert result["query_condition"].shape == (3,)
    assert result["embedding_metadata"]["embedding_dim"] == 2
    assert result["embedding_metadata"]["condition_dim"] == 3
    assert calls["model"] == "fake-embedding-model"
    assert calls["adapter_path"] == "fake_adapter.pt"
    assert abs(calls["adapter_input_norm"] - 1.0) < 1e-6
    assert "A fragile risk-on rebound." in calls["texts"][0]


def test_render_markdown_labels_live_story_condition() -> None:
    markdown = _render_markdown(
        {
            "cached_query": {
                "condition_source": "live_openai_story",
                "narrative_text": "A live risk-manager story.",
            },
            "variant_rows": [],
            "validation_gate": {},
            "decoder": {},
            "generation": {},
        }
    )

    assert "## Live Narrative" in markdown
    assert "A live risk-manager story." in markdown
