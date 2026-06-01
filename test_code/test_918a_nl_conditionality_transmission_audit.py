import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_conditionality_transmission_audit import (  # noqa: E402
    build_transmission_decision,
    scenario_gain_persists,
    support_metrics,
)


def test_scenario_gain_persists_requires_all_models() -> None:
    gate = {
        "passes_all_models": True,
        "checks": {
            "small": {"passes": True},
            "large": {"passes": True},
        },
    }
    assert scenario_gain_persists(gate)
    gate["checks"]["large"]["passes"] = False
    assert not scenario_gain_persists(gate)


def test_support_metrics_weighted_overlap_and_tv() -> None:
    metrics = support_metrics({1: 0.6, 2: 0.4}, {2: 0.25, 3: 0.75})
    assert metrics["support_shared_count"] == 1
    assert metrics["support_union_count"] == 3
    assert metrics["support_jaccard"] == 1 / 3
    assert metrics["support_weighted_overlap"] == 0.25
    assert metrics["support_tv_distance"] == 0.75


def test_decision_identifies_rollout_bottleneck_after_support_and_prefix() -> None:
    summaries = {
        "observed_cross_narrative": {
            "support_tv_distance_median": 0.8,
            "decoded_prefix_norm_rmse_median": 1.0,
            "rollout_path_energy_distance_median": 1.0,
            "rollout_path_wasserstein_z_median": 1.0,
        },
        "same_narrative_repeat": {
            "support_tv_distance_median": 0.0,
            "decoded_prefix_norm_rmse_median": 0.0,
            "rollout_path_energy_distance_median": 0.2,
            "rollout_path_wasserstein_z_median": 0.2,
        },
        "within_run_bootstrap": {
            "support_tv_distance_median": None,
            "decoded_prefix_norm_rmse_median": None,
            "rollout_path_energy_distance_median": 0.95,
            "rollout_path_wasserstein_z_median": 0.90,
        },
        "start_only_null": {
            "support_tv_distance_median": 0.0,
            "decoded_prefix_norm_rmse_median": 0.0,
            "rollout_path_energy_distance_median": 0.0,
            "rollout_path_wasserstein_z_median": 0.0,
        },
    }
    benchmark = {
        "decision": {
            "key_ratios": {
                "portfolio_var95_vs_repeat": 0.7,
            }
        }
    }
    decision = build_transmission_decision(summaries, benchmark)
    assert decision["verdict"] == "support_and_prefix_preserved_rollout_tail_bottleneck"
    assert "within_run_rollout_bootstrap_noise_close_to_observed" in decision["bottlenecks"]
    assert "portfolio_tail_separation_below_repeat_control" in decision["bottlenecks"]
