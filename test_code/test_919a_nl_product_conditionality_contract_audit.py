import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_product_conditionality_contract_audit import (  # noqa: E402
    _overall_status,
    factor_distribution_gate,
    portfolio_tail_gate,
    support_gate,
)


def test_overall_status_warns_if_any_gate_warns_without_failures() -> None:
    gates = [
        {"status": "pass"},
        {"status": "warning"},
        {"status": "pass"},
    ]
    assert _overall_status(gates) == "warning"


def test_support_gate_passes_when_support_changes_and_controls_do_not() -> None:
    gate = support_gate(
        {
            "summaries": {
                "observed_cross_narrative": {
                    "support_tv_distance_median": 1.0,
                },
            },
            "decision": {
                "ratios": {
                    "repeat_support_tv_to_observed": 0.0,
                    "start_only_support_tv_to_observed": 0.0,
                },
            },
        }
    )

    assert gate["status"] == "pass"


def test_factor_gate_warns_when_bootstrap_noise_is_close() -> None:
    gate = factor_distribution_gate(
        {
            "summaries": {
                "observed_cross_narrative": {
                    "rollout_path_energy_distance_median": 0.05,
                },
            },
            "decision": {
                "ratios": {
                    "repeat_rollout_energy_to_observed": 0.1,
                    "bootstrap_rollout_energy_to_observed": 0.95,
                    "start_only_rollout_energy_to_observed": 0.0,
                },
            },
        }
    )

    assert gate["status"] == "warning"


def test_portfolio_gate_warns_on_tail_even_if_path_passes() -> None:
    gate = portfolio_tail_gate(
        {
            "decision": {
                "portfolio_status": "warning",
                "key_ratios": {
                    "portfolio_path_vs_repeat": 2.0,
                    "portfolio_path_vs_bootstrap": 1.1,
                    "portfolio_var95_vs_repeat": 0.8,
                    "portfolio_var95_vs_bootstrap": 1.2,
                },
            },
        }
    )

    assert gate["status"] == "warning"
