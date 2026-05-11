import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_residual_refinement_testflight import (
    build_residual_inputs,
    build_support_mixture_memory,
    train_residual_refiner,
)


def _examples() -> list[dict[str, object]]:
    rows = []
    for idx in range(4):
        rows.append(
            {
                "window_id": f"w{idx}",
                "window_index": idx,
                "embedding_index": idx,
                "target_index": idx,
                "role": "anchor",
            }
        )
    rows.append(
        {
            "window_id": "w3",
            "window_index": 3,
            "embedding_index": 4,
            "target_index": None,
            "role": "negative",
        }
    )
    return rows


def test_support_mixture_excludes_self_for_train_rows() -> None:
    examples = _examples()
    memory = np.eye(4, dtype=np.float32)
    query = memory[[0, 1, 2, 3, 3]]
    starts = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    mixture, details = build_support_mixture_memory(
        examples=examples,
        query_memory=query,
        memory_targets=memory,
        start_state=starts,
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        top_k=1,
        temperature=0.25,
        start_distance_penalty=0.0,
        exclude_self=True,
    )

    assert mixture.shape == query.shape
    assert details["sample_support_rows"][0]["selected_windows"] != [0]


def test_residual_refiner_learns_toy_support_residual() -> None:
    examples = _examples()
    train_indices = np.asarray([0, 1, 2], dtype=np.int64)
    memory = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    query = np.vstack([memory, [[-0.5, -0.5]]]).astype(np.float32)
    starts = np.asarray(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float32,
    )
    mixture = np.full_like(query, 0.1, dtype=np.float32)
    inputs, _ = build_residual_inputs(
        query_memory=query,
        mixture_memory=mixture,
        start_state=starts,
        examples=examples,
        train_indices=train_indices,
    )

    before = float(np.mean((mixture[:3] - memory[:3]) ** 2))
    result = train_residual_refiner(
        inputs=inputs,
        mixture_memory=mixture,
        memory_targets=memory,
        examples=examples,
        train_indices=train_indices,
        hidden_dim=16,
        steps=200,
        batch_size=3,
        lr=1e-2,
        seed=1,
        device="cpu",
    )
    after = float(np.mean((result["condition_vectors"][:3] - memory[:3]) ** 2))

    assert result["train_row_count"] == 3
    assert after < before
