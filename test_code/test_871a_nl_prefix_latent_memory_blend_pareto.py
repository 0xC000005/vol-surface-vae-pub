import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_memory_blend_pareto import (
    alpha_slug,
    blend_condition_memory,
    parse_alpha_grid,
    write_blended_bridge_artifacts,
)


def test_parse_alpha_grid_adds_valid_interior_values() -> None:
    assert parse_alpha_grid("0.25, 0.5,0.75") == [0.25, 0.5, 0.75]


def test_parse_alpha_grid_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="blend alphas"):
        parse_alpha_grid("0.2,1.2")


def test_blend_condition_memory_is_convex() -> None:
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    blended = blend_condition_memory(text, support, support_alpha=0.25)

    np.testing.assert_allclose(
        blended,
        np.asarray([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32),
    )


def test_alpha_slug_is_stable() -> None:
    assert alpha_slug(0.25) == "alpha250"


def test_write_blended_bridge_artifacts_round_trips(tmp_path) -> None:
    examples = [
        {
            "window_index": 0,
            "embedding_index": 0,
            "role": "anchor",
            "kind": "factual",
            "window_id": "w0",
        },
        {
            "window_index": 1,
            "embedding_index": 1,
            "role": "anchor",
            "kind": "factual",
            "window_id": "w1",
        },
    ]
    condition = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bridge_arrays = {
        "condition_vectors": np.zeros_like(condition),
        "memory_targets": condition.copy(),
        "text_embeddings": condition.copy(),
        "train_indices": np.asarray([0], dtype=np.int64),
        "test_indices": np.asarray([1], dtype=np.int64),
    }

    artifacts = write_blended_bridge_artifacts(
        output_dir=tmp_path,
        source_report={"window_indices": [0, 1], "window_metadata": []},
        examples=examples,
        bridge_arrays=bridge_arrays,
        condition_vectors=condition,
        train_indices=bridge_arrays["train_indices"],
        test_indices=bridge_arrays["test_indices"],
        support_alpha=0.25,
        eval_top_k=1,
    )

    assert artifacts["arrays"].endswith("bridge_eval_arrays_blend_alpha250.npz")
    assert artifacts["report"].endswith("bridge_eval_report_blend_alpha250.json")
    with np.load(artifacts["arrays"]) as payload:
        np.testing.assert_allclose(payload["condition_vectors"], condition)
