import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_risk_state_allocation_tests,
)


def _synthetic_panel(width_tracks_state: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_windows = 12
    n_samples = 21
    history_len = 4
    horizon = 5
    offsets = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)

    history = np.zeros((n_windows, history_len, 1, 1), dtype=np.float32)
    ground_truth = np.zeros((n_windows, horizon, 1, 1), dtype=np.float32)
    samples = np.zeros((n_windows, n_samples, horizon, 1, 1), dtype=np.float32)
    for i in range(n_windows):
        activity = 0.01 * float(i + 1)
        history[i, :, 0, 0] = 0.5 + activity * np.arange(history_len)
        ground_truth[i, :, 0, 0] = history[i, -1, 0, 0] + activity * np.arange(1, horizon + 1)
        sample_width = activity if width_tracks_state else 0.05
        samples[i, :, :, 0, 0] = ground_truth[i, None, :, 0, 0] + offsets[:, None] * sample_width

    return samples, ground_truth, history


def test_risk_state_allocation_passes_when_width_tracks_population_activity() -> None:
    samples, ground_truth, history = _synthetic_panel(width_tracks_state=True)

    result = run_risk_state_allocation_tests(
        samples,
        ground_truth,
        history,
        n_buckets=4,
        min_bucket_size=2,
    )

    assert result["overall_pass"]
    assert result["history_width_spearman"] > 0.95
    assert result["future_width_spearman"] > 0.95
    assert result["history_bucket_width_monotonicity"] == 1.0
    assert result["future_bucket_width_monotonicity"] == 1.0


def test_risk_state_allocation_fails_when_width_is_state_insensitive() -> None:
    samples, ground_truth, history = _synthetic_panel(width_tracks_state=False)

    result = run_risk_state_allocation_tests(
        samples,
        ground_truth,
        history,
        n_buckets=4,
        min_bucket_size=2,
    )

    assert not result["overall_pass"]
    assert abs(result["future_width_spearman"]) < 1e-8
    assert result["future_low_high_width_ratio"] == 1.0


def test_risk_state_allocation_counts_first_future_step_from_history() -> None:
    n_windows = 10
    n_samples = 21
    horizon = 4
    offsets = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    history = np.full((n_windows, 3, 1, 1), 0.5, dtype=np.float32)
    ground_truth = np.zeros((n_windows, horizon, 1, 1), dtype=np.float32)
    samples = np.zeros((n_windows, n_samples, horizon, 1, 1), dtype=np.float32)

    for i in range(n_windows):
        first_step = 0.01 * float(i + 1)
        future_level = 0.5 + first_step
        ground_truth[i, :, 0, 0] = future_level
        samples[i, :, :, 0, 0] = future_level + offsets[:, None] * first_step

    result = run_risk_state_allocation_tests(
        samples,
        ground_truth,
        history,
        n_buckets=5,
        min_bucket_size=2,
    )

    assert result["future_width_spearman"] > 0.95
    assert result["future_low_high_width_ratio"] > 4.0
    assert result["oracle_future_alignment_pass"]
    assert not result["observable_state_response_pass"]
    assert not result["overall_pass"]


def test_risk_state_allocation_overall_tracks_observable_state_when_future_signal_absent() -> None:
    n_windows = 12
    n_samples = 21
    history_len = 4
    horizon = 5
    offsets = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)

    history = np.zeros((n_windows, history_len, 1, 1), dtype=np.float32)
    ground_truth = np.zeros((n_windows, horizon, 1, 1), dtype=np.float32)
    samples = np.zeros((n_windows, n_samples, horizon, 1, 1), dtype=np.float32)
    for i in range(n_windows):
        history_activity = 0.01 * float(i + 1)
        future_activity = 0.01 * float(n_windows - i)
        history[i, :, 0, 0] = 0.5 + history_activity * np.arange(history_len)
        ground_truth[i, :, 0, 0] = history[i, -1, 0, 0] + future_activity * np.arange(1, horizon + 1)
        samples[i, :, :, 0, 0] = ground_truth[i, None, :, 0, 0] + offsets[:, None] * history_activity

    result = run_risk_state_allocation_tests(
        samples,
        ground_truth,
        history,
        n_buckets=4,
        min_bucket_size=2,
    )

    assert result["history_future_activity_spearman"] < 0.0
    assert result["observable_state_response_pass"]
    assert not result["oracle_future_alignment_pass"]
    assert result["overall_pass"]
