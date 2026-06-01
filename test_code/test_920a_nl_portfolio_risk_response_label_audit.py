import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_risk_response_label_audit import (
    PORTFOLIO_BOOKS,
    build_portfolio_risk_response_report,
    portfolio_pnl,
    portfolio_response_label,
)


def _case(name, spx_shift, *, samples=16):
    states = np.ones((samples, 30, 39), dtype=np.float64)
    start = np.ones(39, dtype=np.float64)
    states[:, :, 25] = 1.0 + spx_shift
    states[:, :, 38] = 1.0 - spx_shift
    return {
        "case_name": name,
        "label": name.replace("_", " "),
        "states": states,
        "start": start,
    }


def test_portfolio_pnl_respects_exposure_direction():
    book = {
        "name": "test",
        "label": "test",
        "exposures": {"SPX": 1.0, "VIX": -1.0},
    }
    case = _case("risk_on", 0.10)

    pnl = portfolio_pnl(case["states"], case["start"], book)

    assert pnl.shape == (16, 30)
    assert np.allclose(pnl, 20.0)


def test_response_label_reports_tail_metrics():
    label = portfolio_response_label(_case("risk_off", -0.05), PORTFOLIO_BOOKS[0])

    assert label["book"] == "equity_beta_carry"
    assert "var95_loss" in label
    assert "es95_loss" in label
    assert label["sample_count"] == 16


def test_report_detects_observed_portfolio_response_above_controls():
    observed = [_case("risk_on", 0.10), _case("risk_off", -0.10)]
    repeats = [_case("risk_on#seed_1", 0.10), _case("risk_on#seed_2", 0.10)]
    start_only = [_case("start_a", 0.0), _case("start_b", 0.0)]
    book = {
        "name": "test_book",
        "label": "Test book",
        "purpose": "Synthetic direction test.",
        "exposures": {"SPX": 1.0, "VIX": -1.0},
    }

    report = build_portfolio_risk_response_report(
        observed_cases=observed,
        repeat_cases=repeats,
        start_only_cases=start_only,
        books=[book],
    )

    assert report["book_summaries"][0]["status"] == "pass"
    assert report["book_summaries"][0]["ratios"]["path_vs_repeat"] == float("inf")
    assert report["status_counts"] == {"pass": 1}
