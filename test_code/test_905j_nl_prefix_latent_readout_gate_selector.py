import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_readout_gate_selector import (
    choose_readout_candidate,
)


def _calibration_report(alpha, coverage, crps, energy):
    return {
        "selected_alpha": alpha,
        "comparison": {
            "eval_calibrated_minus_component_coverage_80": coverage,
            "eval_calibrated_minus_component_crps": crps,
            "eval_calibrated_minus_component_energy": energy,
        },
    }


def _path_report(status="pass", warnings=None, failures=None):
    return {
        "status": status,
        "warnings": list(warnings or []),
        "failures": list(failures or []),
        "ratios": {
            "bootstrap_to_observed_path_wasserstein": 0.70,
            "repeat_to_observed_path_energy": 0.12,
        },
    }


def test_selector_rejects_warning_candidate_even_with_better_quality():
    selected = choose_readout_candidate(
        [
            {
                "name": "alpha1p05",
                "calibration_report": _calibration_report(1.05, 0.014, -0.004, -0.007),
                "path_report": _path_report(),
            },
            {
                "name": "alpha1p25",
                "calibration_report": _calibration_report(1.25, 0.068, -0.019, -0.035),
                "path_report": _path_report(warnings=["bootstrap_noise"]),
            },
        ]
    )

    assert selected["selected"]["name"] == "alpha1p05"
    assert selected["selected"]["gate_status"] == "pass"
    assert selected["rejected"][0]["name"] == "alpha1p25"
    assert selected["rejected"][0]["gate_status"] == "warning"


def test_selector_falls_back_to_uncalibrated_when_no_candidate_passes_gate():
    selected = choose_readout_candidate(
        [
            {
                "name": "alpha1p10",
                "calibration_report": _calibration_report(1.10, 0.027, -0.008, -0.015),
                "path_report": _path_report(warnings=["bootstrap_noise"]),
            }
        ],
        fallback_name="uncalibrated",
    )

    assert selected["selected"]["name"] == "uncalibrated"
    assert selected["selected"]["gate_status"] == "fallback"
    assert selected["recommendation"] == "keep_uncalibrated_default"
