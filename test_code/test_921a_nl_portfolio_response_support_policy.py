import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_support_policy_testflight import (
    PORTFOLIO_LABEL_METRIC,
    build_portfolio_label_scenario_report,
    compare_policy_to_equal_floor,
    portfolio_candidate_scores,
)


def _arrays(*, good: bool = True):
    future = np.zeros((2, 30, 39), dtype=np.float32)
    future[0, :, 25] = 0.10
    future[0, :, 38] = -0.05
    scale = np.ones((30, 39), dtype=np.float32)
    samples = np.zeros((4, 30, 39), dtype=np.float32)
    if good:
        samples[:, :, 25] = 0.10
        samples[:, :, 38] = -0.05
    else:
        samples[:, :, 25] = -0.10
        samples[:, :, 38] = 0.05
    return {
        "future_delta": future,
        "delta_scale": scale,
        "narrative_0000_q0": samples,
        "narrative_0": samples,
        "evaluated_indices": np.asarray([0], dtype=np.int64),
        "evaluated_block_indices": np.asarray([0], dtype=np.int64),
    }


def _scenario_report(*, crps=1.0, energy=1.0, coverage=0.5):
    return {
        "artifact_paths": {"report": "synthetic.json"},
        "summary": {
            "narrative_generator_topk": {
                "window_count": 1,
                "ensemble_crps_z_mean": crps,
                "energy_score_z_mean": energy,
                "coverage_80_mean": coverage,
            }
        },
        "window_scores": [
            {
                "row_no": 0,
                "query_id": "q0",
                "window_index": 0,
                "block_window_index": 0,
                "methods": {"narrative_generator_topk": {}},
            }
        ],
    }


def test_portfolio_candidate_scores_reward_matching_path():
    good = portfolio_candidate_scores(
        samples=_arrays(good=True)["narrative_0000_q0"],
        target=_arrays(good=True)["future_delta"][0],
        delta_scale=_arrays(good=True)["delta_scale"],
        books=[{"name": "x", "label": "x", "exposures": {"SPX": 1.0, "VIX": -1.0}}],
    )
    bad = portfolio_candidate_scores(
        samples=_arrays(good=False)["narrative_0000_q0"],
        target=_arrays(good=False)["future_delta"][0],
        delta_scale=_arrays(good=False)["delta_scale"],
        books=[{"name": "x", "label": "x", "exposures": {"SPX": 1.0, "VIX": -1.0}}],
    )

    assert good[PORTFOLIO_LABEL_METRIC] < bad[PORTFOLIO_LABEL_METRIC]


def test_build_label_report_adds_portfolio_metric():
    report = build_portfolio_label_scenario_report(
        scenario_report=_scenario_report(),
        arrays=_arrays(good=True),
        book_names=("equity_beta_carry",),
    )

    row = report["window_scores"][0]["methods"]["narrative_generator_topk"]
    assert report["summary"]["row_count"] == 1
    assert PORTFOLIO_LABEL_METRIC in row
    assert row["portfolio_response_books"][0]["book"] == "equity_beta_carry"


def test_compare_policy_marks_clean_portfolio_quality_gain():
    equal = _scenario_report(crps=1.0, energy=1.0, coverage=0.5)
    candidate = _scenario_report(crps=0.9, energy=0.8, coverage=0.55)
    comparison = compare_policy_to_equal_floor(
        equal_report=equal,
        candidate_report=candidate,
        equal_arrays=_arrays(good=False),
        candidate_arrays=_arrays(good=True),
        book_names=("equity_beta_carry",),
    )

    assert comparison["status"] == "candidate_beats_equal_floor"
    assert comparison["benchmark_floor_status"] == "beats_floor"
    assert comparison["portfolio_delta_candidate_minus_equal"][
        "portfolio_reliable_path_score_z"
    ] < 0.0
