import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_calibration_conditionality_analysis import (
    build_calibration_conditionality_report,
    safe_ratio,
)


def _audit(status: str, observed_energy: float, bootstrap_energy: float) -> dict:
    return {
        "path_distribution_status": status,
        "warnings": [] if status == "pass" else ["warning"],
        "summaries": {
            "observed_narrative": {
                "path_energy_distance_median_across_pairs": observed_energy,
                "path_wasserstein_z_mean_median_median_across_pairs": 1.0,
                "path_std_log_ratio_mean_median_median_across_pairs": 1.0,
                "path_drawdown_prob_gap_1sigma_mean_median_across_pairs": 1.0,
            },
            "same_narrative_repeat": {
                "path_energy_distance_median_across_pairs": observed_energy / 4.0,
                "path_wasserstein_z_mean_median_median_across_pairs": 0.25,
                "path_std_log_ratio_mean_median_median_across_pairs": 0.25,
                "path_drawdown_prob_gap_1sigma_mean_median_across_pairs": 0.25,
            },
            "within_run_bootstrap": {
                "path_energy_distance_median_across_pairs": bootstrap_energy,
                "path_wasserstein_z_mean_median_median_across_pairs": 0.5,
                "path_std_log_ratio_mean_median_median_across_pairs": 0.5,
                "path_drawdown_prob_gap_1sigma_mean_median_across_pairs": 0.5,
            },
            "start_only_null": {
                "path_energy_distance_median_across_pairs": 0.0,
                "path_wasserstein_z_mean_median_median_across_pairs": 0.0,
                "path_std_log_ratio_mean_median_median_across_pairs": 0.0,
                "path_drawdown_prob_gap_1sigma_mean_median_across_pairs": 0.0,
            },
        },
    }


def test_safe_ratio_handles_missing_and_zero_denominator() -> None:
    assert safe_ratio(None, 1.0) is None
    assert safe_ratio(1.0, None) is None
    assert safe_ratio(1.0, 0.0) is None
    assert safe_ratio(2.0, 4.0) == 0.5


def test_calibration_report_marks_base_pass_display_warning() -> None:
    report = build_calibration_conditionality_report(
        base_report=_audit("pass", observed_energy=2.0, bootstrap_energy=0.5),
        calibrated_report=_audit("warning", observed_energy=1.0, bootstrap_energy=0.5),
        base_report_path="base.json",
        calibrated_report_path="cal.json",
    )

    assert report["decision"] == "base_conditioning_passes_calibrated_display_warns"
    assert report["status"] == "warning"
    assert report["observed_signal_retention"]["path_energy"] == 0.5
    assert report["observed_signal_retention"]["path_wasserstein"] == 1.0
    assert report["observed_signal_retention"]["path_variance"] == 1.0
    assert report["bootstrap_signal_retention"]["path_energy"] == 1.0
    assert any(row["metric"] == "path_energy" for row in report["metric_rows"])
