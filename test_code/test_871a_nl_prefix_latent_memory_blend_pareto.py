import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_memory_blend_pareto import (
    blend_condition_memory,
    parse_alpha_grid,
)


def test_parse_alpha_grid_adds_valid_interior_values() -> None:
    assert parse_alpha_grid("0.25, 0.5,0.75") == [0.25, 0.5, 0.75]


def test_parse_alpha_grid_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="blend alphas"):
        parse_alpha_grid("0.2,1.2")


def test_blend_condition_memory_is_convex() -> None:
    text = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    support = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    blended = blend_condition_memory(text, support, support_alpha=0.25)

    np.testing.assert_allclose(
        blended,
        np.asarray([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32),
    )
