import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_bakeoff import (
    render_markdown,
    row_from_report,
    selected_historical_cases,
    summarize_by_variant,
)


def test_selected_historical_cases_excludes_modified_user_start() -> None:
    rows = selected_historical_cases()

    assert rows
    assert all("candidate_index" in row for row in rows)
    assert all(row["start_name"] != "extreme_user_start" for row in rows)


def test_row_from_report_extracts_operational_metrics() -> None:
    row = row_from_report(
        case={
            "case_name": "fragile",
            "start_name": "recommended",
            "candidate_index": 18,
        },
        variant={
            "variant_name": "decoder_soft_topk_combined",
            "memory_prior_mode": "soft_topk_combined",
            "prefix_prior_mode": "decoder",
            "top_k": 8,
            "temperature": 0.2,
        },
        report={
            "artifact_paths": {"report": "run.json"},
            "validation_gate": {
                "operational_status": "pass",
                "overall_status": "pass",
            },
            "variant_rows": [
                {
                    "variant": "explicit_start",
                    "is_operational": True,
                    "start_window_index": 18,
                    "start_distance_z": 1.5,
                    "memory_prior_weighted_start_distance_z": 2.5,
                    "memory_prior_analogue_count": 8,
                }
            ],
            "generation": {
                "window_scores": [
                    {
                        "variant": "explicit_start",
                        "start_window_index": 18,
                        "methods": {
                            "text_memory_plus_start_prefix_decoder": {
                                "energy_score_z": 4.0,
                                "ensemble_crps_z": 2.0,
                            },
                            "persistence": {
                                "energy_score_z": 5.0,
                                "ensemble_crps_z": 4.0,
                            },
                        },
                    }
                ]
            },
        },
    )

    assert row["target_available"] is True
    assert row["run_report"] == "run.json"
    assert np.isclose(row["scenario_metrics"]["energy_score_z"], 4.0)
    assert np.isclose(
        row["scenario_metrics"]["ensemble_crps_z_improvement_vs_persistence"], 0.5
    )


def test_summarize_by_variant_sorts_by_crps() -> None:
    summary = summarize_by_variant(
        [
            {
                "variant_name": "b",
                "validation_operational": "pass",
                "target_available": True,
                "memory_prior_weighted_start_distance_z": 2.0,
                "scenario_metrics": {
                    "energy_score_z": 2.0,
                    "ensemble_crps_z": 2.0,
                    "energy_score_z_improvement_vs_persistence": -0.1,
                    "ensemble_crps_z_improvement_vs_persistence": -0.2,
                },
            },
            {
                "variant_name": "a",
                "validation_operational": "pass",
                "target_available": True,
                "memory_prior_weighted_start_distance_z": 1.0,
                "scenario_metrics": {
                    "energy_score_z": 1.0,
                    "ensemble_crps_z": 1.0,
                    "energy_score_z_improvement_vs_persistence": 0.1,
                    "ensemble_crps_z_improvement_vs_persistence": 0.2,
                },
            },
        ]
    )

    assert [row["variant_name"] for row in summary] == ["a", "b"]
    assert summary[0]["operational_status_counts"] == {"pass": 1}


def test_render_markdown_lists_variant_and_case_rows() -> None:
    text = render_markdown(
        {
            "status": "pass",
            "case_count": 1,
            "variant_count": 1,
            "run_count": 1,
            "variant_summary": [
                {
                    "variant_name": "decoder_soft_topk_combined",
                    "run_count": 1,
                    "target_count": 1,
                    "operational_status_counts": {"pass": 1},
                    "mean_energy_score_z": 1.0,
                    "mean_ensemble_crps_z": 0.5,
                    "mean_energy_improvement_vs_persistence": 0.1,
                    "mean_crps_improvement_vs_persistence": 0.2,
                    "mean_weighted_start_distance_z": 2.0,
                }
            ],
            "rows": [
                {
                    "case_name": "fragile",
                    "start_name": "recommended",
                    "variant_name": "decoder_soft_topk_combined",
                    "validation_operational": "pass",
                    "target_available": True,
                    "memory_prior_weighted_start_distance_z": 2.0,
                    "scenario_metrics": {
                        "energy_score_z": 1.0,
                        "ensemble_crps_z": 0.5,
                        "energy_score_z_improvement_vs_persistence": 0.1,
                        "ensemble_crps_z_improvement_vs_persistence": 0.2,
                    },
                    "run_report": "run.json",
                }
            ],
        }
    )

    assert "Fixed-Start Prefix-Mixture Bakeoff" in text
    assert "`decoder_soft_topk_combined`" in text
    assert "`fragile`" in text
