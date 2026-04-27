import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_576a_unified_increment_panel import encode_state  # noqa: E402
from experiments.backfill.block_ar.increment_coordinate_628_utils import (  # noqa: E402
    build_increment_coordinate_block,
    reconstruct_state_from_increments,
)


def test_increment_coordinate_reconstructs_future_state() -> None:
    panel = np.array(
        [
            [10.0, 20.0, 100.0, 0.00, 1.00, 0.00],
            [11.0, 19.0, 101.0, 0.01, 1.10, 0.10],
            [12.0, 18.0, 103.0, 0.02, 1.00, -0.10],
            [13.0, 17.0, 106.0, 0.03, 1.20, 0.20],
            [14.0, 16.0, 110.0, 0.04, 1.30, 0.10],
            [15.0, 15.0, 115.0, 0.05, 1.20, -0.10],
        ],
        dtype=np.float32,
    )
    columns = [
        "iv:0",
        "iv:1",
        "factor:equity",
        "factor:equity_logret",
        "factor:spread",
        "factor:spread_diff",
    ]

    block = build_increment_coordinate_block(
        panel,
        columns,
        indices=[1],
        history_len=3,
        future_len=2,
        iv_count=2,
    )

    reconstructed = reconstruct_state_from_increments(
        block.history_state[:, -1, :],
        block.future_increment,
        block.specs,
    )

    assert block.history_increment.shape == (1, 3, 4)
    assert block.future_increment.shape == (1, 2, 4)
    assert np.allclose(reconstructed, block.future_state, atol=1e-5)


def test_history_increment_uses_previous_encoded_state_when_available() -> None:
    panel = np.array(
        [
            [10.0, 100.0],
            [11.0, 121.0],
            [12.0, 144.0],
            [13.0, 169.0],
        ],
        dtype=np.float32,
    )
    columns = ["iv:0", "factor:equity"]
    block = build_increment_coordinate_block(
        panel,
        columns,
        indices=[1],
        history_len=2,
        future_len=1,
        iv_count=1,
    )

    specs = block.specs
    encoded = encode_state(panel[:, [0, 1]], specs)
    expected_first = encoded[1] - encoded[0]

    assert np.allclose(block.history_increment[0, 0], expected_first, atol=1e-6)


def test_reconstruct_state_from_increments_broadcasts_sample_axis() -> None:
    panel = np.array(
        [
            [10.0, 100.0],
            [11.0, 121.0],
            [12.0, 144.0],
        ],
        dtype=np.float32,
    )
    columns = ["iv:0", "factor:equity"]
    block = build_increment_coordinate_block(
        panel,
        columns,
        indices=[1],
        history_len=1,
        future_len=1,
        iv_count=1,
    )
    sampled_increments = np.repeat(block.future_increment[:, None, :, :], 3, axis=1)

    reconstructed = reconstruct_state_from_increments(
        block.history_state[:, -1, :],
        sampled_increments,
        block.specs,
    )

    assert reconstructed.shape == (1, 3, 1, 2)
    assert np.allclose(reconstructed[:, 0], block.future_state, atol=1e-5)
