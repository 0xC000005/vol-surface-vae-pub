import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_boss_demo_pack import (
    build_demo_pack,
    live_casebook_snapshot,
    render_markdown,
    validation_snapshot,
)


def _validation_report() -> dict:
    return {
        "case_count": 2,
        "case_set": "expanded",
        "variant_set": "temperature",
        "run_count": 2,
        "variant_summary": [
            {
                "variant_name": "decoder_soft_topk_combined_gen_temp_0p50",
                "operational_status_counts": {"pass": 1, "warning": 1},
                "mean_energy_score_z": 0.7,
                "mean_ensemble_crps_z": 0.5,
                "mean_energy_improvement_vs_persistence": 0.1,
                "mean_crps_improvement_vs_persistence": 0.2,
            }
        ],
        "rows": [
            {
                "case_name": "commodity_inflation_pressure",
                "start_name": "explicit_start_18",
                "validation_operational": "pass",
                "scenario_metrics": {
                    "ensemble_crps_z_improvement_vs_persistence": 0.1,
                    "energy_score_z_improvement_vs_persistence": 0.2,
                },
            },
            {
                "case_name": "dollar_liquidity_squeeze",
                "start_name": "balanced_policy_start_77",
                "validation_operational": "warning",
                "scenario_metrics": {
                    "ensemble_crps_z_improvement_vs_persistence": 0.3,
                    "energy_score_z_improvement_vs_persistence": 0.4,
                },
            },
        ],
    }


def _live_casebook_report() -> dict:
    return {
        "status": "ok",
        "case_count": 2,
        "pass_count": 2,
        "total_openai_tokens": 4000,
        "min_support_candidate_count": 8,
        "artifact_paths": {"summary": "live_casebook.json"},
        "cases": [
            {
                "case_name": "commodity_inflation_pressure_18",
                "casebook_choice": "commodity_inflation_pressure:18",
                "status": "ok",
                "expected_start_index": 18,
                "condition_only_validation_status": "pass",
                "selected_start_status": "pass",
                "overall_status": "pass",
                "condition_only_forward_warning_count": 1,
                "support_candidate_count": 8,
                "support_prior_mode": "soft_topk_combined",
                "grounding_model": "gpt-5.4-mini",
                "embedding_model": "text-embedding-3-small",
                "summary_path": "commodity/gradio_api_smoke_summary.json",
            },
            {
                "case_name": "dollar_liquidity_squeeze_22",
                "casebook_choice": "dollar_liquidity_squeeze:22",
                "status": "ok",
                "expected_start_index": 22,
                "condition_only_validation_status": "pass",
                "selected_start_status": "pass",
                "overall_status": "pass",
                "condition_only_forward_warning_count": 1,
                "support_candidate_count": 8,
                "support_prior_mode": "soft_topk_combined",
                "grounding_model": "gpt-5.4-mini",
                "embedding_model": "text-embedding-3-small",
                "summary_path": "dollar/gradio_api_smoke_summary.json",
            },
        ],
    }


def test_validation_snapshot_extracts_demo_metrics() -> None:
    snapshot = validation_snapshot(_validation_report())

    assert snapshot["case_set"] == "expanded"
    assert snapshot["run_count"] == 2
    assert snapshot["pass_rows"] == 1
    assert snapshot["warning_rows"] == 1
    assert snapshot["narrative_family_count"] == 2
    assert snapshot["fixed_start_count"] == 2
    assert snapshot["improved_crps_rows"] == 2
    assert snapshot["improved_energy_rows"] == 2
    assert snapshot["mean_crps_improvement_vs_persistence"] == 0.2


def test_live_casebook_snapshot_extracts_provenance() -> None:
    snapshot = live_casebook_snapshot(_live_casebook_report())

    assert snapshot["status"] == "ok"
    assert snapshot["case_count"] == 2
    assert snapshot["pass_count"] == 2
    assert snapshot["total_openai_tokens"] == 4000
    assert snapshot["grounding_models"] == ["gpt-5.4-mini"]
    assert snapshot["embedding_models"] == ["text-embedding-3-small"]
    assert snapshot["case_rows"][0]["support_candidate_count"] == 8


def test_render_markdown_explains_workflow_and_warning_semantics() -> None:
    summary = {
        "validation_report": "validation.json",
        "validation_snapshot": validation_snapshot(_validation_report()),
    }

    text = render_markdown(summary)

    assert "Narrative-Conditioned Scenario Generator Evidence Pack" in text
    assert "Fix the initial joint39 level" in text
    assert "Narrative families: `2`" in text
    assert "Rows improving energy vs persistence: `2/2`" in text
    assert "Rows improving CRPS vs persistence: `2/2`" in text
    assert "warning is a trust caveat" in text


def test_render_markdown_includes_live_casebook_section() -> None:
    summary = {
        "validation_report": "validation.json",
        "validation_snapshot": validation_snapshot(_validation_report()),
        "live_casebook_report": "live_casebook.json",
        "live_casebook_snapshot": live_casebook_snapshot(_live_casebook_report()),
    }

    text = render_markdown(summary)

    assert "Live Gradio API Casebook" in text
    assert "Total OpenAI tokens: `4000`" in text
    assert "commodity_inflation_pressure_18" in text
    assert "text-embedding-3-small" in text


def test_build_demo_pack_writes_json_and_markdown(tmp_path) -> None:
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(_validation_report()), encoding="utf-8")

    summary = build_demo_pack(
        SimpleNamespace(
            validation_report=str(validation_path),
            live_casebook_report="",
            output_dir=str(tmp_path / "out"),
        )
    )

    assert summary["status"] == "ok"
    assert summary["validation_snapshot"]["improved_crps_rows"] == 2
    assert (tmp_path / "out" / "boss_demo_pack.json").exists()
    assert (tmp_path / "out" / "boss_demo_pack.md").exists()


def test_build_demo_pack_writes_live_casebook_snapshot(tmp_path) -> None:
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(_validation_report()), encoding="utf-8")
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(_live_casebook_report()), encoding="utf-8")

    summary = build_demo_pack(
        SimpleNamespace(
            validation_report=str(validation_path),
            live_casebook_report=str(live_path),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert summary["status"] == "ok"
    assert summary["live_casebook_snapshot"]["pass_count"] == 2
    markdown = (tmp_path / "out" / "boss_demo_pack.md").read_text(encoding="utf-8")
    assert "Live Gradio API Casebook" in markdown
