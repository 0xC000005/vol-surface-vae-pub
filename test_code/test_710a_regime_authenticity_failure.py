import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_710a_regime_authenticity_failure import (
    analyze_regime_authenticity_failures,
)


def test_analyze_regime_authenticity_failures_links_undercoverage_to_authenticity() -> None:
    result = {
        "regime_coverage": {
            "layer2_regime_cell": {
                "calm": {
                    "1": {
                        "grid": [[0.80, 0.69], [0.95, 0.71]],
                        "worst": 0.69,
                        "worst_cell": [0, 1],
                    }
                },
                "turb": {
                    "30": {
                        "grid": [[0.72, 0.55], [0.90, 0.66]],
                        "worst": 0.55,
                        "worst_cell": [0, 1],
                    }
                },
            }
        },
        "distributional_fidelity": {
            "ks_level_test": {
                "ks_grid": [[0.10, 0.30], [0.08, 0.20]],
                "ks_gate": 0.15,
            },
            "cell_mae": {
                "mae_grid": [[1.0, 12.0], [2.0, 8.0]],
            },
        },
        "cointegration": {
            "per_cell_ratio_grid": [[0.60, 0.10], [0.80, 0.20]],
            "worst_cell_pass": False,
        },
    }

    report = analyze_regime_authenticity_failures(result, min_regime_cell=0.70)

    assert report["n_undercovered_slices"] == 3
    assert report["worst_undercovered_slices"][0]["coverage"] == 0.55
    assert report["worst_undercovered_slices"][0]["cell"] == [0, 1]
    assert report["undercovered_cell_overlap"]["level_ks_fail_rate"] == 1.0
    assert report["undercovered_cell_overlap"]["cointegration_fail_rate"] == 1.0
