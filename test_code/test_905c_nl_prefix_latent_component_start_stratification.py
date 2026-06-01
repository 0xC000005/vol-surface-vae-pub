import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_component_start_stratification import (  # noqa: E402
    build_stratification,
    classify_audit,
)


def _report(status="pass", warnings=None, failures=None, ratio=0.25):
    return {
        "status": status,
        "promotion_status": status,
        "promotion_warnings": list(warnings or []),
        "failures": list(failures or []),
        "ratios": {
            "repeat_to_observed_path_energy": ratio,
            "bootstrap_to_observed_path_energy": ratio,
            "bootstrap_to_sample_matched_observed_path_energy": ratio,
            "start_only_to_observed_path_energy": 0.0,
        },
        "start_reliability": {
            "case_rows": [{"start_index": 18}],
            "warning_counts": {},
            "failure_counts": {},
            "start_distance_z_median": 13.0,
        },
        "observed_pairwise": [
            {"sample_count_left": 192, "sample_count_right": 192}
        ],
    }


def test_build_stratification_parses_start_from_pair_names(tmp_path: Path) -> None:
    import json

    path = tmp_path / "shape.json"
    report = _report()
    report["start_reliability"] = {}
    report["observed_pairwise"] = [
        {
            "left_case": "fragile_risk_on_start18",
            "right_case": "defensive_risk_off_start18",
            "sample_count_left": 384,
            "sample_count_right": 384,
        }
    ]
    path.write_text(json.dumps(report), encoding="utf-8")

    stratification = build_stratification([path])

    assert stratification["rows"][0]["starts"] == [18]


def test_classify_audit_separates_pass_warning_and_repeat_failure() -> None:
    assert classify_audit(_report()) == ("pass", "clean_pass")
    assert classify_audit(
        _report(status="warning", warnings=["bootstrap_path_noise_close_to_observed"])
    ) == ("warning", "bootstrap_readout_warning")
    raw_warning_report = _report(status="warning")
    raw_warning_report.pop("promotion_warnings")
    raw_warning_report["warnings"] = ["bootstrap_energy_noise_close_to_observed"]
    assert classify_audit(raw_warning_report) == (
        "warning",
        "bootstrap_readout_warning",
    )
    assert classify_audit(
        _report(status="fail", failures=["repeat_path_too_close_to_observed"])
    ) == ("fail", "repeat_not_separated")
    start_warning_report = _report(
        status="fail",
        failures=["repeat_path_too_close_to_observed"],
    )
    start_warning_report["start_reliability"]["warning_counts"] = {
        "large_start_distance": 6
    }
    assert classify_audit(start_warning_report) == ("fail", "start_incompatible")


def test_build_stratification_counts_rows(tmp_path: Path) -> None:
    pass_path = tmp_path / "pass.json"
    warning_path = tmp_path / "warning.json"
    import json

    pass_path.write_text(json.dumps(_report()), encoding="utf-8")
    warning_path.write_text(
        json.dumps(
            _report(
                status="warning",
                warnings=["bootstrap_energy_noise_close_to_observed"],
                ratio=0.9,
            )
        ),
        encoding="utf-8",
    )

    report = build_stratification([pass_path, warning_path])

    assert report["audit_count"] == 2
    assert report["status_counts"] == {"pass": 1, "warning": 1}
    assert report["reason_counts"]["bootstrap_readout_warning"] == 1
    assert report["rows"][0]["starts"] == [18]
    assert report["rows"][0]["sample_count"] == 192
