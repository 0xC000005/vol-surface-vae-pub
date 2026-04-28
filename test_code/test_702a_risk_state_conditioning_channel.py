import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.audit_702a_risk_state_conditioning_channel import (
    future_risk_targets_np,
    width_by_window,
)


def test_future_risk_targets_np_activity_and_peak():
    future = np.array(
        [
            [[1.0, -1.0], [4.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )

    targets = future_risk_targets_np(future)

    assert np.allclose(targets["activity"], [4.5, 0.25])
    assert np.allclose(targets["mean_abs"], [1.5, 0.5])
    assert np.allclose(targets["max_abs"], [4.0, 0.5])
    assert targets["temporal_peak"][0] > targets["temporal_peak"][1]


def test_width_by_window_uses_sample_quantiles():
    samples = np.array(
        [
            [
                [[0.0], [0.0]],
                [[1.0], [1.0]],
                [[2.0], [2.0]],
            ],
            [
                [[10.0], [10.0]],
                [[10.0], [10.0]],
                [[10.0], [10.0]],
            ],
        ],
        dtype=np.float32,
    )

    widths = width_by_window(samples)

    assert widths.shape == (2,)
    assert widths[0] > 0.0
    assert widths[1] == 0.0
