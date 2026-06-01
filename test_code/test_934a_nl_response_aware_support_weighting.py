import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    FACTOR_INDEX,
)
from experiments.backfill.block_ar.nl_response_aware_support_weighting_testflight import (  # noqa: E402
    _allocate_counts,
    _component_response_score,
    _preview_and_pool_states,
    response_channels,
)


def test_response_channels_preserve_grounded_direction_and_fallbacks():
    grounding = {
        "current_market_state_implications": [
            {
                "market": "SPX",
                "direction": "down",
                "confidence": "high",
                "magnitude": "medium",
            }
        ]
    }

    channels = response_channels(grounding, fallback_factors=("SPX", "VIX"))

    assert channels[0]["factor"] == "SPX"
    assert channels[0]["sign"] == -1.0
    assert any(row["factor"] == "VIX" and row["sign"] == 0.0 for row in channels)


def test_allocate_counts_sums_to_requested_samples():
    counts = _allocate_counts(np.asarray([0.2, 0.3, 0.5]), 17)

    assert counts.sum() == 17
    assert counts.tolist() == [3, 5, 9]


def test_component_response_score_rewards_signed_channel_move():
    start = np.zeros(39, dtype=np.float64)
    start[FACTOR_INDEX["SPX"]] = 100.0
    up_component = np.zeros((4, 2, 39), dtype=np.float64)
    down_component = np.zeros((4, 2, 39), dtype=np.float64)
    up_component[:, :, FACTOR_INDEX["SPX"]] = np.asarray(
        [[100.0, 104.0], [100.0, 105.0], [100.0, 106.0], [100.0, 107.0]]
    )
    down_component[:, :, FACTOR_INDEX["SPX"]] = np.asarray(
        [[100.0, 96.0], [100.0, 95.0], [100.0, 94.0], [100.0, 93.0]]
    )

    channels = [{"factor": "SPX", "sign": 1.0, "weight": 1.0}]

    assert _component_response_score(
        component_states=up_component,
        start=start,
        channels=channels,
    ) > _component_response_score(
        component_states=down_component,
        start=start,
        channels=channels,
    )


def test_preview_and_pool_states_separates_scoring_from_pool_when_possible():
    states = np.arange(10 * 2 * 3, dtype=np.float32).reshape(10, 2, 3)
    score, pool, info = _preview_and_pool_states(
        states,
        pilot_samples=4,
        min_pool_samples=2,
        rng=np.random.default_rng(7),
    )

    assert score.shape == (4, 2, 3)
    assert pool.shape == (6, 2, 3)
    assert info["preview_sample_count"] == 4
    assert info["pool_sample_count"] == 6
    assert info["pool_reuses_preview"] is False


def test_preview_and_pool_states_preserves_full_sample_default():
    states = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3)
    score, pool, info = _preview_and_pool_states(
        states,
        pilot_samples=0,
        min_pool_samples=2,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_array_equal(score, states)
    np.testing.assert_array_equal(pool, states)
    assert info["pool_reuses_preview"] is True
