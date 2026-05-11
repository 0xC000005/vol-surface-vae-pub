import numpy as np
import pytest
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_text_start_memory_diagnostic import (
    build_input_features,
    caption_coverage_audit,
    standardize_start_features,
)


def test_standardize_start_features_uses_example_window_for_negatives():
    starts = np.asarray(
        [
            [10.0, 100.0],
            [20.0, 110.0],
            [30.0, 130.0],
        ],
        dtype=np.float32,
    )
    example_windows = np.asarray([0, 0, 1, 2], dtype=np.int64)

    features, stats = standardize_start_features(
        starts,
        example_windows,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
    )

    np.testing.assert_allclose(stats["start_mean"], [[15.0, 105.0]])
    np.testing.assert_allclose(stats["start_std"], [[5.0, 5.0]])
    np.testing.assert_allclose(features[0], [-1.0, -1.0])
    np.testing.assert_allclose(features[1], [-1.0, -1.0])
    np.testing.assert_allclose(features[2], [1.0, 1.0])
    np.testing.assert_allclose(features[3], [3.0, 5.0])


def test_build_input_features_modes_and_validation():
    examples = [
        {"window_index": 0, "role": "anchor", "kind": "main"},
        {"window_index": 1, "role": "positive", "kind": "alt"},
    ]
    text = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    starts = np.asarray([[10.0], [20.0]], dtype=np.float32)

    text_only, _ = build_input_features(
        text,
        starts,
        examples,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
        input_mode="text_only",
    )
    start_only, _ = build_input_features(
        text,
        starts,
        examples,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
        input_mode="start_only",
    )
    text_start, _ = build_input_features(
        text,
        starts,
        examples,
        fit_window_indices=np.asarray([0, 1], dtype=np.int64),
        input_mode="text_start",
        start_feature_weight=0.5,
    )

    assert text_only.shape == (2, 2)
    assert start_only.shape == (2, 1)
    assert text_start.shape == (2, 3)
    np.testing.assert_allclose(text_only[0], [0.6, 0.8])
    np.testing.assert_allclose(text_start[:, :2], text_only)
    np.testing.assert_allclose(text_start[:, 2], [-0.5, 0.5])

    with pytest.raises(ValueError, match="input_mode"):
        build_input_features(
            text,
            starts,
            examples,
            fit_window_indices=np.asarray([0], dtype=np.int64),
            input_mode="bad",
        )


def test_caption_coverage_audit_counts_roles_by_split():
    examples = [
        {"window_index": 0, "role": "anchor", "kind": "main"},
        {"window_index": 0, "role": "positive", "kind": "risk_manager"},
        {"window_index": 0, "role": "negative", "kind": "opposite"},
        {"window_index": 1, "role": "anchor", "kind": "main"},
        {"window_index": 1, "role": "negative", "kind": "partial"},
    ]
    split = {
        "train_indices": [0],
        "test_indices": [1],
        "excluded_indices": [],
    }

    audit = caption_coverage_audit(examples, split)

    assert audit["window_count"] == 2
    assert audit["example_count"] == 5
    assert audit["role_counts"] == {"anchor": 2, "negative": 2, "positive": 1}
    assert audit["split_role_counts"]["train"] == {
        "anchor": 1,
        "negative": 1,
        "positive": 1,
    }
    assert audit["split_role_counts"]["test"] == {"anchor": 1, "negative": 1}
    assert audit["positive_captions_per_window"]["mean"] == 0.5
    assert audit["hard_negatives_per_window"]["mean"] == 1.0
