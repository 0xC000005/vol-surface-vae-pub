import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_portfolio_quality_guard_paired_rollout_summary import (
    build_pair_summary,
)


def _case(offset: float) -> dict:
    start = np.ones(39, dtype=np.float64)
    states = np.ones((4, 3, 39), dtype=np.float64)
    states[:, :, 25] += offset
    states[:, :, 38] -= offset
    return {
        "root": f"case_{offset}",
        "states": states,
        "start": start,
        "support_indices": [1, 2],
        "support_weights": [0.5, 0.5],
        "quality_guard_policy": None,
    }


def test_build_pair_summary_reports_factor_and_portfolio_deltas() -> None:
    base = _case(0.0)
    guarded = _case(1.0)
    guarded["support_indices"] = [2, 3]
    guarded["quality_guard_policy"] = {"fallback_to_equal_support": False}

    report = build_pair_summary(
        case_name="unit",
        base=base,
        guarded=guarded,
        factors=["SPX", "VIX"],
        book_names=["equity_beta_carry"],
    )

    assert report["support_jaccard"] == 1 / 3
    factor_rows = {row["factor"]: row for row in report["factor_terminal_rows"]}
    assert factor_rows["SPX"]["terminal_mean_delta_qg_minus_base"] == 1.0
    assert factor_rows["VIX"]["terminal_mean_delta_qg_minus_base"] == -1.0
    assert report["portfolio_rows"][0]["book"] == "equity_beta_carry"
