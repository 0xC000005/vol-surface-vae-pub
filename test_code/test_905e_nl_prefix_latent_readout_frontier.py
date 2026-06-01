import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_readout_frontier import (
    build_readout_frontier_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _shape_audit(path_energy_ratio: float) -> dict:
    return {
        "ratios": {
            "bootstrap_to_observed_path_energy": path_energy_ratio,
            "bootstrap_to_observed_path_wasserstein": path_energy_ratio,
        }
    }


def _calibration_report(
    *,
    status: str,
    calibrated_report: Path,
    energy_retention: float,
    wasserstein_retention: float,
) -> dict:
    return {
        "status": status,
        "base_report": "base.json",
        "calibrated_report": str(calibrated_report),
        "calibrated_status": status,
        "observed_signal_retention": {
            "path_energy": energy_retention,
            "path_wasserstein": wasserstein_retention,
            "path_variance": 1.0,
            "drawdown_probability": 0.9,
        },
        "bootstrap_signal_retention": {
            "path_energy": 1.0,
            "path_wasserstein": 1.0,
        },
    }


def test_readout_frontier_rejects_low_retention_candidate(tmp_path: Path) -> None:
    audit = _write_json(tmp_path / "audit.json", _shape_audit(1.2))
    cal = _write_json(
        tmp_path / "calibration.json",
        _calibration_report(
            status="warning",
            calibrated_report=audit,
            energy_retention=0.3,
            wasserstein_retention=0.7,
        ),
    )

    report = build_readout_frontier_report(calibration_reports=[("alpha3", str(cal))])

    assert report["status"] == "warning"
    assert (
        report["decision"]
        == "global_readout_not_sufficient_continue_ecc_style_candidate"
    )
    assert (
        report["candidate_rows"][0]["candidate_decision"] == "reject_readout_candidate"
    )
    assert (
        "low_observed_path_energy_retention" in report["candidate_rows"][0]["failures"]
    )


def test_readout_frontier_accepts_candidate_above_thresholds(tmp_path: Path) -> None:
    audit = _write_json(tmp_path / "audit.json", _shape_audit(0.3))
    cal = _write_json(
        tmp_path / "calibration.json",
        _calibration_report(
            status="pass",
            calibrated_report=audit,
            energy_retention=0.8,
            wasserstein_retention=0.9,
        ),
    )

    report = build_readout_frontier_report(calibration_reports=[("good", str(cal))])

    assert report["status"] == "pass"
    assert report["decision"] == "global_readout_candidate_viable"
    assert report["best_candidate"]["label"] == "good"
    assert report["candidate_rows"][0]["candidate_decision"] == "candidate_viable"
