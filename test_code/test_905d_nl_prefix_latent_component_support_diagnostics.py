import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_component_support_diagnostics import (  # noqa: E402
    build_support_diagnostics,
    support_jaccard,
    weighted_overlap,
)


def test_weighted_overlap_and_jaccard() -> None:
    assert weighted_overlap({1: 0.7, 2: 0.3}, {1: 0.2, 3: 0.8}) == 0.2
    assert support_jaccard([1, 2], [2, 3]) == 1 / 3


def test_build_support_diagnostics_groups_by_start(monkeypatch) -> None:
    def fake_discover(root, *, variant_dir):
        del root, variant_dir
        return [
            {
                "case_name": "risk_on_start18",
                "narrative": "risk_on",
                "start_index": 18,
                "support_weights": {1: 0.75, 2: 0.25},
                "support_windows": [1, 2],
                "top_window": 1,
                "effective_n": 1.6,
                "support_count": 2,
                "_decoded_prefix": np.zeros((2, 3), dtype=np.float32),
                "_states": np.zeros((4, 3, 39), dtype=np.float32),
                "_start": np.zeros(39, dtype=np.float32),
            },
            {
                "case_name": "risk_off_start18",
                "narrative": "risk_off",
                "start_index": 18,
                "support_weights": {2: 0.5, 3: 0.5},
                "support_windows": [2, 3],
                "top_window": 2,
                "effective_n": 2.0,
                "support_count": 2,
                "_decoded_prefix": np.ones((2, 3), dtype=np.float32),
                "_states": np.ones((4, 3, 39), dtype=np.float32),
                "_start": np.zeros(39, dtype=np.float32),
            },
        ]

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_component_support_diagnostics.discover_support_cases",
        fake_discover,
    )

    report = build_support_diagnostics({"demo": Path("unused")})
    summary = report["start_summaries"]["demo:start18"]

    assert report["case_count"] == 2
    assert report["pair_count"] == 1
    assert summary["weighted_overlap_median"] == 0.25
    assert summary["jaccard_median"] == 1 / 3
    assert summary["same_top_window_rate"] == 0.0
    assert abs(summary["decoded_prefix_l2_median"] - 1.0) < 1e-6
    assert "generated_path_energy_median" in summary
    assert "_states" not in report["cases"][0]
