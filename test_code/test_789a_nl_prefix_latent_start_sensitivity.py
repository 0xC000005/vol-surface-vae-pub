import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_sensitivity import (
    build_start_variant_rows,
    endpoint_alignment_summary,
    rollout_sensitivity_summary,
)


def test_build_start_variant_rows_uses_train_starts_for_substitutions() -> None:
    start = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=np.float32,
    )

    rows = build_start_variant_rows(
        query_window_indices=np.asarray([0, 2], dtype=np.int64),
        train_indices=np.asarray([1, 3], dtype=np.int64),
        start_state=start,
    )

    assert [row["variant"] for row in rows[:3]] == [
        "original",
        "nearest_train_start",
        "farthest_train_start",
    ]
    assert rows[0]["start_window_index"] == 0
    assert rows[1]["start_window_index"] in {1, 3}
    assert rows[2]["start_window_index"] in {1, 3}
    assert all(row["query_window_index"] in {0, 2} for row in rows)


def test_endpoint_alignment_summary_detects_start_pin_error() -> None:
    prefix = np.zeros((2, 4, 3), dtype=np.float32)
    requested = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    prefix[:, -1, :] = requested

    summary = endpoint_alignment_summary(prefix, requested)

    assert summary["max_abs_error"] == 0.0
    assert summary["mean_abs_error"] == 0.0


def test_rollout_sensitivity_summary_is_finite_for_changed_samples() -> None:
    original = np.zeros((2, 3, 5, 4), dtype=np.float32)
    variant = original.copy()
    variant[1] += 0.5
    scale = np.ones((5, 4), dtype=np.float32)
    variant_rows = [
        {"query_window_index": 7, "variant": "original"},
        {"query_window_index": 7, "variant": "farthest_train_start"},
    ]

    summary = rollout_sensitivity_summary(
        samples=variant,
        variant_rows=variant_rows,
        scale=scale,
    )

    assert summary["farthest_train_start"]["mean_abs_delta_z"] > 0.0
    assert np.isfinite(summary["farthest_train_start"]["terminal_mean_abs_delta_z"])
