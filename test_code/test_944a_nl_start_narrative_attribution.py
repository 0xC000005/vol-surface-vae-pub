from __future__ import annotations

import numpy as np
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_start_narrative_attribution import (
    two_way_feature_attribution,
)


def test_two_way_feature_attribution_start_main_effect() -> None:
    cube = np.zeros((3, 4, 2), dtype=np.float64)
    cube[:, :, 0] = np.arange(3, dtype=np.float64)[:, None]

    result = two_way_feature_attribution(cube)

    assert result["start_share"] > 0.99
    assert result["narrative_share"] < 1.0e-10
    assert result["interaction_share"] < 1.0e-10


def test_two_way_feature_attribution_narrative_main_effect() -> None:
    cube = np.zeros((3, 4, 2), dtype=np.float64)
    cube[:, :, 0] = np.arange(4, dtype=np.float64)[None, :]

    result = two_way_feature_attribution(cube)

    assert result["narrative_share"] > 0.99
    assert result["start_share"] < 1.0e-10
    assert result["interaction_share"] < 1.0e-10


def test_two_way_feature_attribution_interaction_effect() -> None:
    cube = np.zeros((2, 2, 1), dtype=np.float64)
    cube[:, :, 0] = np.asarray([[1.0, -1.0], [-1.0, 1.0]])

    result = two_way_feature_attribution(cube)

    assert result["interaction_share"] > 0.99
    assert result["start_share"] < 1.0e-10
    assert result["narrative_share"] < 1.0e-10
