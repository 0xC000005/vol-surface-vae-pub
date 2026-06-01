import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_conditionality_stress_test import (  # noqa: E402
    CASE_RELEVANT_FACTORS,
    FACTOR_INDEX,
    _sample_energy_distance,
)


def test_sample_energy_distance_is_zero_for_identical_sample_clouds():
    samples = np.arange(48, dtype=np.float64).reshape(4, 12)

    assert _sample_energy_distance(samples, samples.copy(), max_samples=2) == 0.0


def test_case_relevant_factor_panels_use_known_factors():
    for factors in CASE_RELEVANT_FACTORS.values():
        assert factors
        assert set(factors).issubset(FACTOR_INDEX)
