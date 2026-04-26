import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar._local_factor_conditioning_555_utils import (
    LOCAL_FACTOR_KEYS,
    factor_histories,
    fit_factor_standardizer,
    local_factor_panel_from_mapping,
    standardize_factor_histories,
)


def test_local_factor_panel_uses_only_available_npz_factor_keys_in_order() -> None:
    raw = {
        "surface": np.zeros((4, 1, 1), dtype=np.float32),
        "ret": np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        "price": np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64),
        "slopes": np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64),
        "skews": np.array([-1.0, -0.9, -0.8, -0.7], dtype=np.float64),
        "levels": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    }

    panel, columns = local_factor_panel_from_mapping(raw)

    assert columns == list(LOCAL_FACTOR_KEYS)
    assert panel.shape == (4, 5)
    assert np.allclose(panel[:, 0], raw["ret"])
    assert np.allclose(panel[:, -1], raw["levels"])


def test_factor_histories_are_history_only() -> None:
    panel = np.arange(20, dtype=np.float32).reshape(10, 2)

    histories = factor_histories(panel, indices=np.array([0, 2]), history_len=3)

    assert histories.shape == (2, 3, 2)
    assert np.allclose(histories[0], panel[0:3])
    assert np.allclose(histories[1], panel[2:5])


def test_standardizer_is_fit_on_training_histories_only() -> None:
    panel = np.arange(12, dtype=np.float32).reshape(6, 2)
    fit_indices = np.array([0, 1])
    histories = factor_histories(panel, indices=np.array([2]), history_len=2)

    mean, std = fit_factor_standardizer(panel, fit_indices=fit_indices, history_len=2)
    standardized = standardize_factor_histories(histories, mean, std)

    expected_fit = np.concatenate([panel[0:2], panel[1:3]], axis=0)
    assert np.allclose(mean, expected_fit.mean(axis=0))
    assert np.all(std > 0)
    assert standardized.shape == histories.shape
