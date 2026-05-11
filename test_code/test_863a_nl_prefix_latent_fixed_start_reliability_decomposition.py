import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_reliability_decomposition import (
    build_reliability_decomposition,
    support_summary,
)


def _case_summary(name: str, windows: list[int], cosine: float = 0.9) -> dict[str, object]:
    return {
        "start_name": name,
        "support_match_rate": 1.0,
        "top_support": [
            {"window_index": window, "weight": 1.0 / len(windows), "memory_support_cosine": cosine}
            for window in windows
        ],
    }


def test_support_summary_measures_support_diversity() -> None:
    summary = support_summary(
        [
            _case_summary("fixed_start_1", [1, 2, 3]),
            _case_summary("fixed_start_1", [3, 4, 5]),
        ]
    )

    assert summary["top1_unique_count"] == 2
    assert summary["support_window_unique_count"] == 5
    assert summary["median_support_jaccard_distance"] == 0.8
    assert summary["mean_top_support_cosine"] == 0.9


def test_reliability_decomposition_flags_repeat_and_sampling_noise() -> None:
    report = build_reliability_decomposition(
        control_report={
            "thresholds": {"max_bootstrap_ratio": 0.75, "max_repeat_ratio": 0.75},
            "per_start_controls": [
                {
                    "start_name": "fixed_start_1",
                    "status": "fail",
                    "observed_median_gap": 1.0,
                    "start_only_ratio": 0.0,
                    "bootstrap_ratio": 0.9,
                    "repeat_ratio": 0.8,
                }
            ],
        },
        observed_bakeoff={
            "rows": [
                {
                    "start_name": "fixed_start_1",
                    "memory_prior_weighted_start_distance_z": 13.0,
                    "memory_prior_support_weighted_match_rate": 1.0,
                    "scenario_metrics": {
                        "ensemble_crps_z_improvement_vs_persistence": 0.2,
                        "energy_score_z_improvement_vs_persistence": 0.1,
                    },
                    "validation_operational": "pass",
                }
            ]
        },
        observed_contrast={
            "case_summaries": [
                _case_summary("fixed_start_1", [1, 2, 3], cosine=0.76),
                _case_summary("fixed_start_1", [4, 5, 6], cosine=0.77),
            ]
        },
    )

    row = report["start_rows"][0]
    assert report["status"] == "warning"
    assert "repeat_seed_instability" in row["mechanism_flags"]
    assert "rollout_sampling_noise_close_to_narrative_gap" in row["mechanism_flags"]
    assert "weak_text_memory_support" in row["mechanism_flags"]
    assert "support_pool_far_from_fixed_start" in row["mechanism_flags"]


def test_reliability_decomposition_keeps_clean_start_unflagged() -> None:
    report = build_reliability_decomposition(
        control_report={
            "thresholds": {"max_bootstrap_ratio": 0.75, "max_repeat_ratio": 0.75},
            "per_start_controls": [
                {
                    "start_name": "fixed_start_1",
                    "status": "pass",
                    "observed_median_gap": 1.0,
                    "start_only_ratio": 0.0,
                    "bootstrap_ratio": 0.3,
                    "repeat_ratio": 0.2,
                }
            ],
        },
        observed_bakeoff={
            "rows": [
                {
                    "start_name": "fixed_start_1",
                    "memory_prior_weighted_start_distance_z": 5.0,
                    "memory_prior_support_weighted_match_rate": 1.0,
                    "scenario_metrics": {},
                    "validation_operational": "pass",
                }
            ]
        },
        observed_contrast={
            "case_summaries": [
                _case_summary("fixed_start_1", [1, 2, 3], cosine=0.9),
                _case_summary("fixed_start_1", [4, 5, 6], cosine=0.9),
            ]
        },
    )

    assert report["start_rows"][0]["mechanism_flags"] == [
        "no_blocking_mechanism_detected"
    ]
