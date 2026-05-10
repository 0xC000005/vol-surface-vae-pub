import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.world.part1_jepa_latent.analyze_additive_probe_standardization import (  # noqa: E402
    standardize_train_val,
)


def test_standardize_train_val_uses_train_statistics_and_handles_constant_columns():
    train = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
    val = np.array([[7.0, 2.0]])

    train_std, val_std = standardize_train_val(train, val)

    assert np.allclose(train_std[:, 0].mean(), 0.0)
    assert np.allclose(train_std[:, 0].std(), 1.0)
    assert np.allclose(train_std[:, 1], 0.0)
    assert np.allclose(val_std[:, 1], 0.0)
