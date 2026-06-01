import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_generator_response_weighting_frontier import (  # noqa: E402
    _clean_improvement,
    _oracle_summary,
    decide_frontier,
)


def test_clean_improvement_requires_crps_energy_and_nonnegative_coverage() -> None:
    assert _clean_improvement(
        {
            "crps_delta_candidate_minus_baseline": -0.1,
            "energy_delta_candidate_minus_baseline": -0.2,
            "coverage_delta_candidate_minus_baseline": 0.01,
        }
    )
    assert not _clean_improvement(
        {
            "crps_delta_candidate_minus_baseline": -0.1,
            "energy_delta_candidate_minus_baseline": 0.2,
            "coverage_delta_candidate_minus_baseline": 0.01,
        }
    )


def test_oracle_summary_infers_window_count_from_sample_count_patterns() -> None:
    summary = _oracle_summary(
        {
            "summary": {
                "equal_top5": {
                    "crps_mean": 1.0,
                    "energy_mean": 2.0,
                    "coverage_80_mean": 0.4,
                    "sample_count_patterns": {"24,24,24,24,24": 2},
                },
                "oracle_weighted_top5": {
                    "crps_mean": 0.9,
                    "energy_mean": 1.8,
                    "coverage_80_mean": 0.5,
                    "sample_count_patterns": {
                        "12,18,24,30,36": 3,
                        "8,16,24,32,40": 4,
                    },
                },
            }
        }
    )

    assert summary["window_count"] == 7
    assert summary["crps_delta_candidate_minus_baseline"] == pytest.approx(-0.1)


def test_decide_frontier_keeps_oracle_but_rejects_weak_learned_policy() -> None:
    decision = decide_frontier(
        transmission={
            "decision": {
                "verdict": "support_and_prefix_preserved_rollout_tail_bottleneck",
                "bottlenecks": ["within_run_rollout_bootstrap_noise_close_to_observed"],
            }
        },
        oracle={
            "crps_delta_candidate_minus_baseline": -0.01,
            "energy_delta_candidate_minus_baseline": -0.02,
            "coverage_delta_candidate_minus_baseline": 0.01,
        },
        learned={
            "crps_delta_candidate_minus_baseline": -0.001,
            "energy_delta_candidate_minus_baseline": 0.001,
            "coverage_delta_candidate_minus_baseline": 0.002,
        },
    )

    assert decision["status"] == "upper_bound_found_learned_policy_insufficient"
    assert decision["oracle_clean_improvement"]
    assert not decision["learned_clean_improvement"]
    assert "within-run rollout bootstrap noise" in decision["recommendation"]
