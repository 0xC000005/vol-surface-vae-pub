import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.evaluate_604a_asymmetric_tail_postforecast_adapter import (  # noqa: E402
    apply_asymmetric_tail_scaling,
    fit_asymmetric_tail_scale_tables,
    tail_miss_rates,
)


def test_tail_miss_rates_separates_lower_and_upper_misses() -> None:
    samples = np.array(
        [
            [[[[0.45]]], [[[0.50]]], [[[0.55]]]],
            [[[[0.45]]], [[[0.50]]], [[[0.55]]]],
        ],
        dtype=np.float32,
    )
    target = np.array([[[[0.40]]], [[[0.60]]]], dtype=np.float32)

    rates = tail_miss_rates(samples, target, lower_scale=1.0, upper_scale=1.0)

    assert rates["lower_miss"] == 0.5
    assert rates["upper_miss"] == 0.5
    assert rates["coverage"] == 0.0


def test_fit_asymmetric_tail_scale_tables_expands_only_needed_side() -> None:
    # Lower targets sit below the current q05 while upper targets are already covered.
    samples = np.array(
        [
            [[[[0.45]]], [[[0.50]]], [[[0.55]]]],
            [[[[0.45]]], [[[0.50]]], [[[0.55]]]],
            [[[[0.45]]], [[[0.50]]], [[[0.55]]]],
            [[[[0.45]]], [[[0.50]]], [[[0.55]]]],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [[[[0.40]]], [[[0.41]]], [[[0.50]]], [[[0.52]]]],
        dtype=np.float32,
    )

    tables = fit_asymmetric_tail_scale_tables(
        calib_samples=samples,
        calib_future=target,
        lower_tail_target=0.05,
        upper_tail_target=0.05,
        scale_min=1.0,
        scale_max=3.0,
        scale_steps=9,
    )

    assert tables.lower_scales.shape == (1, 1, 1)
    assert tables.upper_scales.shape == (1, 1, 1)
    assert tables.lower_scales[0, 0, 0] > tables.upper_scales[0, 0, 0]


def test_apply_asymmetric_tail_scaling_preserves_median_and_side_order() -> None:
    samples = np.array(
        [[[[[0.40]]], [[[0.50]]], [[[0.60]]]]],
        dtype=np.float32,
    )
    lower = np.array([[[2.0]]], dtype=np.float32)
    upper = np.array([[[1.5]]], dtype=np.float32)

    scaled = apply_asymmetric_tail_scaling(
        samples=samples,
        lower_scales=lower,
        upper_scales=upper,
        alpha=1.0,
    )

    assert np.allclose(np.median(scaled, axis=1), np.median(samples, axis=1))
    assert scaled[0, 0, 0, 0, 0] < samples[0, 0, 0, 0, 0]
    assert scaled[0, 2, 0, 0, 0] > samples[0, 2, 0, 0, 0]
    assert scaled[0, 0, 0, 0, 0] < scaled[0, 1, 0, 0, 0] < scaled[0, 2, 0, 0, 0]
