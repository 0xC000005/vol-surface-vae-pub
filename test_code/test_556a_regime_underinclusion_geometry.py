import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.audit_556a_regime_underinclusion_geometry import (
    extract_regime_cell_rows,
    summarize_candidate_regime_geometry,
)


def _result() -> dict:
    return {
        "summary": {"n_pass": 8, "failed_suites": ["regime_coverage"]},
        "regime_coverage": {
            "layer1_pass": True,
            "layer2_n_passing": 1,
            "layer2_n_total": 2,
            "layer3_pass": True,
            "layer2_regime_cell": {
                "calm": {
                    "1": {"worst": 0.72, "best": 0.94, "worst_cell": [1, 0], "best_cell": [4, 4]},
                    "30": {"worst": 0.66, "best": 0.98, "worst_cell": [0, 3], "best_cell": [1, 1]},
                }
            },
        },
    }


def test_extract_regime_cell_rows_marks_under_and_over_failures() -> None:
    rows = extract_regime_cell_rows("demo", _result())
    by_horizon = {row["horizon"]: row for row in rows}

    assert by_horizon["1"]["undercovered"] is False
    assert by_horizon["1"]["overcovered"] is False
    assert by_horizon["30"]["undercovered"] is True
    assert by_horizon["30"]["overcovered"] is True
    assert by_horizon["30"]["worst_cell"] == [0, 3]


def test_summarize_candidate_regime_geometry_reports_localized_failure() -> None:
    summary = summarize_candidate_regime_geometry("demo", _result())

    assert summary["name"] == "demo"
    assert summary["layer1_pass"] is True
    assert summary["layer3_pass"] is True
    assert summary["n_undercovered_layer2"] == 1
    assert summary["n_overcovered_layer2"] == 1
    assert summary["worst_layer2"]["worst"] == 0.66
