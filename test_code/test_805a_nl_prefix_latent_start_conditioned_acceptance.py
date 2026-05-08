import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_acceptance import (
    _operational_score_row,
    _score_metrics,
    apply_start_modifier,
    render_markdown,
    summarize_matrix,
)


def test_apply_start_modifier_creates_out_of_support_payload() -> None:
    payload = {
        "label": "base",
        "coordinate": "raw_state",
        "values_by_name": {
            "iv:1m_100": 0.2,
            "factor:spx": 2000.0,
            "factor:vix": 15.0,
            "factor:bbb_oas": 2.0,
        },
    }

    modified = apply_start_modifier(payload, "extreme_out_of_support")
    values = modified["values_by_name"]

    assert modified["label"] == "extreme_user_start_out_of_support"
    assert np.isclose(values["iv:1m_100"], 0.5)
    assert values["factor:spx"] == 6000.0
    assert values["factor:vix"] == 80.0
    assert values["factor:bbb_oas"] == 14.0


def test_summarize_matrix_uses_expected_operational_statuses() -> None:
    summary = summarize_matrix(
        [
            {
                "case_name": "fragile",
                "start_name": "recommended",
                "expected_operational_status": "pass",
                "validation_operational": "pass",
                "expectation_met": True,
            },
            {
                "case_name": "fragile",
                "start_name": "extreme",
                "expected_operational_status": "fail",
                "validation_operational": "warning",
                "expectation_met": False,
            },
        ]
    )

    assert summary["overall_status"] == "fail"
    assert summary["case_count"] == 2
    assert summary["expectation_fail_count"] == 1


def test_render_markdown_lists_start_matrix_rows() -> None:
    text = render_markdown(
        {
            "overall_status": "pass",
            "case_count": 1,
            "expectation_fail_count": 0,
            "cases": [
                {
                    "case_name": "fragile",
                    "start_name": "recommended",
                    "expected_operational_status": "pass",
                    "validation_operational": "pass",
                    "expectation_met": True,
                    "start_type": "historical_candidate",
                    "start_distance_z": 0.0,
                    "memory_prior_weighted_start_distance_z": 2.0,
                    "target_available": True,
                    "scenario_metrics": {
                        "energy_score_z": 1.25,
                        "ensemble_crps_z": 0.5,
                    },
                    "run_report": "run.json",
                }
            ],
        }
    )

    assert "Start-Conditioned Prefix-Latent Acceptance" in text
    assert "`fragile`" in text
    assert "`recommended`" in text
    assert "`True`" in text
    assert "`1.250`" in text
    assert "`0.500`" in text


def test_operational_score_row_matches_variant_and_start() -> None:
    report = {
        "generation": {
            "window_scores": [
                {
                    "variant": "other",
                    "start_window_index": 18,
                    "methods": {},
                },
                {
                    "variant": "explicit_start",
                    "start_window_index": 18,
                    "methods": {"text_memory_plus_start_prefix_decoder": {}},
                },
            ]
        }
    }

    row = _operational_score_row(
        report,
        {"variant": "explicit_start", "start_window_index": 18},
    )

    assert row["variant"] == "explicit_start"


def test_score_metrics_extracts_model_and_persistence_improvement() -> None:
    metrics = _score_metrics(
        {
            "methods": {
                "text_memory_plus_start_prefix_decoder": {
                    "coverage_80": 0.75,
                    "energy_score_z": 4.0,
                    "ensemble_crps_z": 2.0,
                    "mean_path_mae_z": 3.0,
                    "terminal_mae_z": 5.0,
                },
                "persistence": {
                    "energy_score_z": 5.0,
                    "ensemble_crps_z": 4.0,
                    "mean_path_mae_z": 2.0,
                },
            }
        }
    )

    assert metrics["target_available"] is True
    assert np.isclose(metrics["coverage_80"], 0.75)
    assert np.isclose(metrics["energy_score_z_improvement_vs_persistence"], 0.2)
    assert np.isclose(metrics["ensemble_crps_z_improvement_vs_persistence"], 0.5)
    assert np.isclose(metrics["mean_path_mae_z_improvement_vs_persistence"], -0.5)
