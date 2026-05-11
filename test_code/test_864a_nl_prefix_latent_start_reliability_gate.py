import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_reliability_gate import (
    build_start_reliability_manifest,
    evaluate_start_reliability,
)


def test_start_reliability_manifest_preserves_pass_and_high_warning() -> None:
    manifest = build_start_reliability_manifest(
        control_report={
            "status": "fail",
            "per_start_controls": [
                {
                    "start_name": "fixed_start_good",
                    "status": "pass",
                    "observed_median_gap": 1.0,
                    "start_only_ratio": 0.0,
                    "bootstrap_ratio": 0.3,
                    "repeat_ratio": 0.2,
                    "warnings": [],
                    "failures": [],
                },
                {
                    "start_name": "fixed_start_weak",
                    "status": "fail",
                    "observed_median_gap": 0.5,
                    "start_only_ratio": 0.0,
                    "bootstrap_ratio": 0.6,
                    "repeat_ratio": 0.9,
                    "warnings": [],
                    "failures": ["repeat_noise_close"],
                },
            ],
        },
        sample_count=192,
        min_sample_count=192,
    )

    assert manifest["status"] == "warning"
    assert manifest["status_counts"] == {
        "pass": 1,
        "warn_high_instability": 1,
    }
    weak = evaluate_start_reliability(manifest, "fixed_start_weak")
    assert weak["product_status"] == "warn_high_instability"
    assert weak["failures"] == ["repeat_noise_close"]


def test_start_reliability_manifest_warns_on_low_sample_count() -> None:
    manifest = build_start_reliability_manifest(
        control_report={
            "status": "pass",
            "per_start_controls": [
                {
                    "start_name": "fixed_start_low_sample",
                    "status": "pass",
                    "warnings": [],
                    "failures": [],
                }
            ],
        },
        sample_count=96,
        min_sample_count=192,
    )

    row = manifest["starts"][0]
    assert row["product_status"] == "warn_needs_stronger_evidence"
    assert "scenario_sample_count_below_reliability_floor" in row["warnings"]


def test_evaluate_start_reliability_warns_for_unknown_start() -> None:
    manifest = build_start_reliability_manifest(
        control_report={"per_start_controls": []},
        sample_count=192,
    )

    result = evaluate_start_reliability(manifest, "missing")

    assert result["source"] == "fallback"
    assert result["product_status"] == "warn_needs_stronger_evidence"
    assert result["warnings"] == ["start_reliability_evidence_missing"]
