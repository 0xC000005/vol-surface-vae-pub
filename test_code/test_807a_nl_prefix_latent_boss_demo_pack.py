import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_boss_demo_pack import (
    build_demo_pack,
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
                "validation_operational": "pass",
                "scenario_metrics": {"ensemble_crps_z_improvement_vs_persistence": 0.1},
            },
            {
                "validation_operational": "warning",
                "scenario_metrics": {"ensemble_crps_z_improvement_vs_persistence": 0.3},
            },
        ],
    }


def test_validation_snapshot_extracts_demo_metrics() -> None:
    snapshot = validation_snapshot(_validation_report())

    assert snapshot["case_set"] == "expanded"
    assert snapshot["run_count"] == 2
    assert snapshot["pass_rows"] == 1
    assert snapshot["warning_rows"] == 1
    assert snapshot["improved_crps_rows"] == 2
    assert snapshot["mean_crps_improvement_vs_persistence"] == 0.2


def test_render_markdown_explains_workflow_and_warning_semantics() -> None:
    summary = {
        "validation_report": "validation.json",
        "validation_snapshot": validation_snapshot(_validation_report()),
    }

    text = render_markdown(summary)

    assert "Narrative-Conditioned Scenario Generator Evidence Pack" in text
    assert "Fix the initial joint39 level" in text
    assert "Rows improving CRPS vs persistence: `2/2`" in text
    assert "warning is a trust caveat" in text


def test_build_demo_pack_writes_json_and_markdown(tmp_path) -> None:
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(_validation_report()), encoding="utf-8")

    summary = build_demo_pack(
        SimpleNamespace(
            validation_report=str(validation_path),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert summary["status"] == "ok"
    assert summary["validation_snapshot"]["improved_crps_rows"] == 2
    assert (tmp_path / "out" / "boss_demo_pack.json").exists()
    assert (tmp_path / "out" / "boss_demo_pack.md").exists()
