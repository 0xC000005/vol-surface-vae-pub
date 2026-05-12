import numpy as np
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_hybrid_direction_feature_bridge import (
    build_hybrid_feature_matrix,
    direction_features_for_examples,
    direction_features_from_text,
)


def test_direction_features_from_text_reads_explicit_market_tokens():
    vector, names = direction_features_from_text(
        "MARKET_IMPLICATIONS: SPX: UP LARGE; VIX: DOWN MEDIUM; BBB_OAS: WIDER SMALL"
    )

    values = dict(zip(names, vector))
    assert values["SPX_signed"] == 1.0
    assert values["SPX_present"] == 1.0
    assert values["VIX_signed"] == -0.66
    assert values["BBB_OAS_signed"] == 0.33


def test_direction_features_for_examples_fills_opposite_negative_from_anchor():
    examples = [
        {
            "window_id": "w1",
            "role": "anchor",
            "kind": "primary",
            "text": "MARKET_IMPLICATIONS: SPX: UP LARGE; VIX: DOWN MEDIUM",
        },
        {
            "window_id": "w1",
            "role": "negative",
            "kind": "opposite",
            "text": "Opposite narrative without machine tokens.",
        },
    ]

    features, names = direction_features_for_examples(examples)
    values = [dict(zip(names, row)) for row in features]

    assert values[0]["SPX_signed"] == 1.0
    assert values[1]["SPX_signed"] == -1.0
    assert values[0]["VIX_signed"] == -0.66
    assert values[1]["VIX_signed"] == 0.66
    assert values[1]["SPX_present"] == 1.0


def test_build_hybrid_feature_matrix_keeps_text_and_direction_channels():
    text_embeddings = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    direction_features = np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    hybrid = build_hybrid_feature_matrix(text_embeddings, direction_features)

    assert hybrid.shape == (2, 4)
    assert np.all(np.isfinite(hybrid))
    assert np.isclose(np.linalg.norm(hybrid[0, :2]), 2**-0.5, atol=1e-6)
    assert np.isclose(np.linalg.norm(hybrid[0, 2:]), 2**-0.5, atol=1e-6)
    assert np.isclose(np.linalg.norm(hybrid[1, :2]), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(hybrid[1, 2:]), 0.0, atol=1e-6)
