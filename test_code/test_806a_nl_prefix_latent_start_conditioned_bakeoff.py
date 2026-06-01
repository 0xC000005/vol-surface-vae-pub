import json
import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_bakeoff import (
    oracle_select_best_rows,
    render_markdown,
    row_from_report,
    run_bakeoff,
    selected_historical_cases,
    selected_variants,
    summarize_oracle_selection,
    summarize_by_variant,
)


def test_selected_historical_cases_excludes_modified_user_start() -> None:
    rows = selected_historical_cases()

    assert rows
    assert all("candidate_index" in row for row in rows)
    assert all(row["start_name"] != "extreme_user_start" for row in rows)


def test_selected_historical_cases_supports_expanded_case_set() -> None:
    rows = selected_historical_cases(case_set="expanded")

    assert len(rows) == 8
    assert {row["case_name"] for row in rows} == {
        "fragile_risk_on",
        "defensive_risk_off",
        "rates_selloff",
    }
    assert any(row["candidate_index"] == 178 for row in rows)


def test_selected_historical_cases_supports_custom_case_spec(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "commodity_inflation_pressure",
                        "start_name": "explicit_18",
                        "condition_report": "condition_report.json",
                        "candidate_index": 18,
                    },
                    {
                        "case_name": "commodity_inflation_pressure",
                        "start_name": "explicit_40",
                        "condition_report": "condition_report.json",
                        "candidate_index": 40,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = selected_historical_cases(case_spec_json=path)

    assert [row["start_name"] for row in rows] == ["explicit_18", "explicit_40"]
    assert all(row["condition_report"] == "condition_report.json" for row in rows)


def test_selected_variants_supports_temperature_calibration_set() -> None:
    rows = selected_variants(2, variant_set="temperature")

    assert len(rows) == 2
    assert rows[0]["generator_temperature"] == 0.5
    assert rows[1]["generator_temperature"] == 0.75


def test_selected_variants_supports_direction_check_set() -> None:
    rows = selected_variants(1, variant_set="direction_check")

    assert rows[0]["memory_prior_mode"] == "soft_topk_narrative_start_checked"
    assert rows[0]["generator_temperature"] == 0.5
    assert rows[0]["rollout_mixture_mode"] == "averaged_prefix"


def test_selected_variants_supports_control_sets() -> None:
    start_only = selected_variants(1, variant_set="start_only_control")
    controls = selected_variants(3, variant_set="controls")

    assert start_only[0]["memory_prior_mode"] == "soft_topk_start_only"
    assert [row["memory_prior_mode"] for row in controls] == [
        "soft_topk_narrative_start_checked",
        "soft_topk_start_only",
        "soft_topk_memory",
    ]


def test_selected_variants_supports_component_sets() -> None:
    narrative = selected_variants(1, variant_set="component_direction_check")
    temperature = selected_variants(3, variant_set="component_support_temperature")
    generator_temperature = selected_variants(
        3,
        variant_set="component_generator_temperature",
    )
    start_only = selected_variants(1, variant_set="component_start_only_control")
    start_only_generator_temperature = selected_variants(
        1,
        variant_set="component_start_only_generator_temperature",
    )

    assert narrative[0]["rollout_mixture_mode"] == "component_prefix_mixture"
    assert narrative[0]["memory_prior_mode"] == "soft_topk_narrative_start_checked"
    assert [row["temperature"] for row in temperature] == [0.2, 0.1, 0.05]
    assert {
        row["rollout_mixture_mode"] for row in temperature
    } == {"component_prefix_mixture"}
    assert [row["generator_temperature"] for row in generator_temperature] == [
        0.25,
        0.5,
        0.75,
    ]
    assert {
        row["rollout_mixture_mode"] for row in generator_temperature
    } == {"component_prefix_mixture"}
    assert start_only[0]["rollout_mixture_mode"] == "component_prefix_mixture"
    assert start_only[0]["memory_prior_mode"] == "soft_topk_start_only"
    assert start_only_generator_temperature[0]["generator_temperature"] == 0.25
    assert (
        start_only_generator_temperature[0]["rollout_mixture_mode"]
        == "component_prefix_mixture"
    )


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
            "rollout_mixture_mode": "averaged_prefix",
            "top_k": 8,
            "temperature": 0.2,
            "generator_temperature": 0.75,
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
                    "memory_prior_direction_status": "pass",
                    "memory_prior_direction_reason": "ok",
                    "memory_prior_support_weighted_match_rate": 0.875,
                    "memory_prior_final_mixture_mismatch_count": 0,
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
    assert row["generator_temperature"] == 0.75
    assert row["rollout_mixture_mode"] == "averaged_prefix"
    assert row["memory_prior_direction_status"] == "pass"
    assert row["memory_prior_support_weighted_match_rate"] == 0.875
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
                "memory_prior_direction_status": "pass",
                "memory_prior_support_weighted_match_rate": 0.9,
                "memory_prior_final_mixture_mismatch_count": 0,
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
                "memory_prior_direction_status": "reject",
                "memory_prior_support_weighted_match_rate": 0.4,
                "memory_prior_final_mixture_mismatch_count": 1,
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
    assert summary[0]["direction_status_counts"] == {"reject": 1}
    assert summary[0]["total_final_mixture_mismatches"] == 1


def test_oracle_select_best_rows_chooses_lowest_crps_per_case_start() -> None:
    rows = [
        {
            "case_name": "fragile",
            "start_name": "recommended",
            "candidate_index": 18,
            "variant_name": "a",
            "target_available": True,
            "scenario_metrics": {"ensemble_crps_z": 2.0},
        },
        {
            "case_name": "fragile",
            "start_name": "recommended",
            "candidate_index": 18,
            "variant_name": "b",
            "target_available": True,
            "scenario_metrics": {"ensemble_crps_z": 1.0},
        },
        {
            "case_name": "fragile",
            "start_name": "user",
            "candidate_index": 19,
            "variant_name": "c",
            "target_available": False,
            "scenario_metrics": {},
        },
    ]

    selected = oracle_select_best_rows(rows)

    assert len(selected) == 1
    assert selected[0]["variant_name"] == "b"


def test_summarize_oracle_selection_reports_upper_bound_metrics() -> None:
    summary = summarize_oracle_selection(
        [
            {
                "case_name": "fragile",
                "start_name": "recommended",
                "candidate_index": 18,
                "variant_name": "a",
                "target_available": True,
                "scenario_metrics": {
                    "energy_score_z": 3.0,
                    "ensemble_crps_z": 2.0,
                    "energy_score_z_improvement_vs_persistence": -0.5,
                    "ensemble_crps_z_improvement_vs_persistence": -0.25,
                },
            },
            {
                "case_name": "fragile",
                "start_name": "recommended",
                "candidate_index": 18,
                "variant_name": "b",
                "target_available": True,
                "scenario_metrics": {
                    "energy_score_z": 1.0,
                    "ensemble_crps_z": 0.5,
                    "energy_score_z_improvement_vs_persistence": 0.1,
                    "ensemble_crps_z_improvement_vs_persistence": 0.2,
                },
            },
        ]
    )

    assert summary["selector"] == "realized_future_best_crps_upper_bound"
    assert summary["chosen_variant_counts"] == {"b": 1}
    assert np.isclose(summary["mean_ensemble_crps_z"], 0.5)
    assert np.isclose(summary["mean_crps_improvement_vs_persistence"], 0.2)


def test_run_bakeoff_sets_explicit_seed(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_prefix_latent_story_smoke(args):
        captured["seed"] = args.seed
        captured["rollout_mixture_mode"] = args.rollout_mixture_mode
        return {
            "artifact_paths": {"report": str(tmp_path / "run.json")},
            "validation_gate": {
                "operational_status": "pass",
                "overall_status": "pass",
            },
            "variant_rows": [
                {
                    "variant": "explicit_start",
                    "is_operational": True,
                    "start_window_index": 18,
                }
            ],
            "generation": {"window_scores": []},
        }

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_bakeoff.run_prefix_latent_story_smoke",
        fake_run_prefix_latent_story_smoke,
    )
    case_spec = tmp_path / "cases.json"
    case_spec.write_text(
        json.dumps(
            [
                {
                    "case_name": "fragile",
                    "start_name": "fixed_start_18",
                    "condition_report": "condition.json",
                    "candidate_index": 18,
                }
            ]
        ),
        encoding="utf-8",
    )

    class Args:
        output_dir = str(tmp_path / "out")
        case_count = 1
        case_set = "default"
        case_spec_json = str(case_spec)
        variant_count = 1
        variant_set = "component_direction_check"
        samples = 2
        steps = 1
        chunk_size = 2
        device = "cpu"
        seed = 1234

    run_bakeoff(Args())

    assert captured["seed"] == 1234
    assert captured["rollout_mixture_mode"] == "component_prefix_mixture"


def test_render_markdown_lists_variant_and_case_rows() -> None:
    text = render_markdown(
        {
            "status": "pass",
            "case_set": "default",
            "variant_set": "prior",
            "case_count": 1,
            "variant_count": 1,
            "run_count": 1,
            "variant_summary": [
                {
                    "variant_name": "decoder_soft_topk_combined",
                    "run_count": 1,
                    "target_count": 1,
                    "operational_status_counts": {"pass": 1},
                    "direction_status_counts": {"pass": 1},
                    "mean_energy_score_z": 1.0,
                    "mean_ensemble_crps_z": 0.5,
                    "mean_energy_improvement_vs_persistence": 0.1,
                    "mean_crps_improvement_vs_persistence": 0.2,
                    "mean_weighted_start_distance_z": 2.0,
                    "mean_support_weighted_match_rate": 0.875,
                    "total_final_mixture_mismatches": 0,
                }
            ],
            "oracle_selection_summary": {
                "selector": "realized_future_best_crps_upper_bound",
                "scope_note": "Diagnostic only.",
                "selected_count": 1,
                "chosen_variant_counts": {"decoder_soft_topk_combined": 1},
                "mean_energy_score_z": 1.0,
                "mean_ensemble_crps_z": 0.5,
                "mean_energy_improvement_vs_persistence": 0.1,
                "mean_crps_improvement_vs_persistence": 0.2,
            },
            "rows": [
                {
                    "case_name": "fragile",
                    "start_name": "recommended",
                    "variant_name": "decoder_soft_topk_combined",
                    "validation_operational": "pass",
                    "memory_prior_direction_status": "pass",
                    "memory_prior_support_weighted_match_rate": 0.875,
                    "memory_prior_final_mixture_mismatch_count": 0,
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
    assert "Realized-Future Oracle Selector" in text
    assert "`fragile`" in text
