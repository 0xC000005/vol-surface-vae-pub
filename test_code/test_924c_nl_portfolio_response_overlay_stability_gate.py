import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_overlay_stability_gate import (
    build_overlay_stability_gate,
)


def _report(*, crps: float, energy: float, coverage: float, portfolio: float) -> dict:
    return {
        "status": "candidate",
        "scenario_delta_candidate_minus_equal": {
            "ensemble_crps_z_mean": crps,
            "energy_score_z_mean": energy,
            "coverage_80_mean": coverage,
        },
        "portfolio_delta_candidate_minus_equal": {
            "portfolio_reliable_path_score_z": portfolio,
        },
    }


def test_stability_gate_promotes_default_only_when_all_seeds_are_clean():
    gate = build_overlay_stability_gate(
        [
            _report(crps=-0.1, energy=-0.1, coverage=0.1, portfolio=-0.1),
            _report(crps=-0.2, energy=-0.2, coverage=0.2, portfolio=-0.2),
        ]
    )

    assert gate["result_status"] == "overlay_default_candidate"
    assert gate["decision"]["promote_default"] is True


def test_stability_gate_keeps_tiny_energy_regression_as_overlay_only():
    gate = build_overlay_stability_gate(
        [
            _report(crps=-0.1, energy=-0.1, coverage=0.1, portfolio=-0.1),
            _report(crps=-0.2, energy=0.0002, coverage=0.2, portfolio=-0.2),
        ]
    )

    assert gate["result_status"] == "overlay_portfolio_candidate"
    assert gate["decision"]["promote_default"] is False
    assert gate["decision"]["promote_portfolio_overlay"] is True


def test_stability_gate_rejects_inconsistent_portfolio_response():
    gate = build_overlay_stability_gate(
        [
            _report(crps=-0.1, energy=-0.1, coverage=0.1, portfolio=-0.1),
            _report(crps=-0.2, energy=-0.2, coverage=0.2, portfolio=0.2),
        ]
    )

    assert gate["result_status"] == "overlay_not_current_lever"
