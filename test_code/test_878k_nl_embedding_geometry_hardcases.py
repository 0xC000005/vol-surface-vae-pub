import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_embedding_geometry_hardcases import (
    contrast_geometry_for_window,
)


def test_contrast_geometry_for_window_compares_raw_and_condition_margins():
    examples = [
        {"window_id": "w0", "role": "anchor", "embedding_index": 0},
        {"window_id": "w0", "role": "positive", "embedding_index": 1},
        {"window_id": "w0", "role": "negative", "embedding_index": 2},
    ]
    raw = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    condition = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    result = contrast_geometry_for_window(
        examples,
        raw_embeddings=raw,
        condition_vectors=condition,
        window_id="w0",
    )

    assert result["window_id"] == "w0"
    assert result["raw"]["hard_margin"] > 1.0
    assert result["condition"]["hard_margin"] < 0.2
    assert result["diagnosis"] == "adapter_collapses_raw_separation"
