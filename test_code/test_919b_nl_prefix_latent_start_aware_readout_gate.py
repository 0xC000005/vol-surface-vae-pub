import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_start_aware_readout_gate import (
    build_start_aware_readout_gate,
)


def _selection():
    return {
        "selected": {
            "name": "alpha1p05",
            "gate_status": "pass",
            "quality_deltas": {
                "coverage_80_delta": 0.01,
                "crps_delta": -0.01,
                "energy_delta": -0.01,
            },
        }
    }


def _audit_report(path, *, status="pass", warnings=None, failures=None):
    path.write_text(
        """
{
  "status": "%s",
  "warnings": %s,
  "failures": %s,
  "ratios": {
    "repeat_to_observed_path_energy": 0.1,
    "bootstrap_to_observed_path_energy": 0.5,
    "start_only_to_observed_path_energy": 0.0
  },
  "observed_pairwise": [
    {"left_case": "case_start18#a", "right_case": "case_start18#b", "sample_count_left": 64, "sample_count_right": 64}
  ]
}
"""
        % (
            status,
            "[]" if warnings is None else str(warnings).replace("'", '"'),
            "[]" if failures is None else str(failures).replace("'", '"'),
        ),
        encoding="utf-8",
    )
    return path


def test_broad_promotion_requires_multiple_clean_starts(tmp_path):
    paths = [
        _audit_report(tmp_path / "a.json"),
        _audit_report(tmp_path / "b.json"),
    ]
    report = build_start_aware_readout_gate(
        _selection(),
        paths,
        min_pass_starts=2,
    )

    assert report["status"] == "pass"
    assert report["recommendation"] == "promote_selected_readout_broadly"
    assert report["broad_promotion"] is True


def test_single_clean_start_keeps_readout_local(tmp_path):
    paths = [
        _audit_report(tmp_path / "a.json"),
        _audit_report(
            tmp_path / "b.json",
            status="warning",
            warnings=["bootstrap_path_noise_close_to_observed"],
        ),
    ]
    report = build_start_aware_readout_gate(
        _selection(),
        paths,
        min_pass_starts=2,
    )

    assert report["status"] == "warning"
    assert report["recommendation"] == "keep_selected_readout_as_local_candidate_only"
    assert report["broad_promotion"] is False
    assert "too_few_clean_starts_for_broad_promotion" in report["findings"]
    assert "warning_starts_remain_bootstrap_limited" in report["findings"]
