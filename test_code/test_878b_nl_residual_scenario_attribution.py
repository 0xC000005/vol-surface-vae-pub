import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_residual_scenario_attribution import (
    analyze_report,
)


def _row(window_id, top1, incumbent_crps, candidate_crps, incumbent_cov, candidate_cov):
    return {
        "window_id": window_id,
        "window_index": int(window_id.rsplit("_", 1)[-1]),
        "block_window_index": int(window_id.rsplit("_", 1)[-1]) + 100,
        "top_train_cosines": [top1, top1 - 0.05, top1 - 0.10],
        "top_train_indices": [1, 2, 3],
        "top_train_window_ids": ["a", "b", "c"],
        "methods": {
            "incumbent": {
                "ensemble_crps_z": incumbent_crps,
                "coverage_80": incumbent_cov,
            },
            "candidate": {
                "ensemble_crps_z": candidate_crps,
                "coverage_80": candidate_cov,
            },
        },
    }


def test_analyze_report_uses_positive_delta_as_candidate_better():
    report = {
        "window_scores": [
            _row("joint39_val_0001", 0.90, 1.0, 0.8, 0.40, 0.60),
            _row("joint39_val_0002", 0.75, 1.0, 1.2, 0.40, 0.30),
        ]
    }

    result = analyze_report(
        report,
        incumbent_method="incumbent",
        candidate_method="candidate",
        metrics=("ensemble_crps_z", "coverage_80"),
    )

    crps = result["metric_summary"]["ensemble_crps_z"]
    coverage = result["metric_summary"]["coverage_80"]
    assert crps["mean_delta_positive_is_better"] == 0.0
    assert crps["win_rate"] == 0.5
    assert coverage["mean_delta_positive_is_better"] == 0.05
    assert coverage["win_rate"] == 0.5
    assert result["best_windows_by_crps"][0]["window_id"] == "joint39_val_0001"
    assert result["worst_windows_by_crps"][0]["window_id"] == "joint39_val_0002"


def test_analyze_report_keeps_support_similarity_diagnostics():
    report = {
        "window_scores": [
            _row("joint39_val_0001", 0.95, 1.0, 0.9, 0.50, 0.55),
            _row("joint39_val_0002", 0.85, 1.0, 0.8, 0.50, 0.60),
            _row("joint39_val_0003", 0.70, 1.0, 1.2, 0.50, 0.40),
        ]
    }

    result = analyze_report(
        report,
        incumbent_method="incumbent",
        candidate_method="candidate",
        metrics=("ensemble_crps_z",),
    )

    assert result["window_count"] == 3
    assert result["support_diagnostics"]["top1_cosine_mean"] == 0.833333333333
    assert result["support_diagnostics"]["top1_cosine_gain_correlation"] > 0
