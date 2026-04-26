import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.evaluate_564a_stress_selected_510a import (
    severity_stratified_indices,
    select_severity_stratified_paths,
)


def test_severity_stratified_indices_are_deterministic_and_tail_inclusive() -> None:
    severity = np.arange(10, dtype=np.float64)

    indices = severity_stratified_indices(severity, n_select=6)

    assert indices.tolist() == [0, 1, 4, 5, 8, 9]


def test_select_severity_stratified_paths_preserves_shape_and_extremes() -> None:
    candidates = np.zeros((2, 10, 3, 2, 2), dtype=np.float32)
    for sample_idx in range(10):
        candidates[:, sample_idx] = float(sample_idx)

    selected = select_severity_stratified_paths(candidates, n_select=6)

    assert selected.shape == (2, 6, 3, 2, 2)
    assert np.allclose(selected[0, :, 0, 0, 0], np.array([0, 1, 4, 5, 8, 9], dtype=np.float32))
    assert np.allclose(selected[1, :, 0, 0, 0], np.array([0, 1, 4, 5, 8, 9], dtype=np.float32))
