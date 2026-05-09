import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_run_record import (
    build_prefix_run_record,
    render_markdown,
    write_prefix_run_record,
)


def _report(tmp_path: Path) -> dict:
    report_path = tmp_path / "prefix_report.json"
    markdown_path = tmp_path / "prefix_report.md"
    arrays_path = tmp_path / "arrays.npz"
    report_path.write_text("{}", encoding="utf-8")
    markdown_path.write_text("# Report\n", encoding="utf-8")
    arrays_path.write_bytes(b"arrays")
    return {
        "artifact_paths": {
            "report": str(report_path),
            "markdown": str(markdown_path),
            "arrays": str(arrays_path),
        },
        "cached_query": {
            "window_id": "joint39_val_0370",
            "kind": "revised_market_description",
            "narrative_text": "Private narrative text should not be retained.",
            "text_memory_dim": 128,
            "condition_source": "condition_only_openai_story",
            "grounding": {
                "market_implications": [{"market": "SPX"}],
                "non_conditioning_forward_language": [
                    {"phrase": "volatility may reverse"}
                ],
            },
            "memory_prior": {
                "candidate_details": [
                    {
                        "rank": 1,
                        "window_id": "joint39_val_0269",
                        "bridge_local_index": 269,
                        "source_index": 4762,
                        "history_start_date": "2020-02-01",
                        "history_end_date": "2020-03-12",
                        "manifest_split": "train",
                        "weight": 0.42,
                        "memory_support_cosine": 0.887,
                        "start_distance_z": 6.94,
                        "recent_prefix_alignment_score": 0.8,
                        "recent_prefix_mismatches": 1,
                        "combined_score": 0.73,
                    }
                ]
            },
        },
        "variant_rows": [
            {
                "variant": "original",
                "start_window_id": "joint39_val_0370",
                "is_operational": False,
            },
            {
                "variant": "nearest_train_start",
                "start_window_id": "joint39_val_0269",
                "start_window_index": 269,
                "start_manifest_split": "train",
                "start_selection_method": "max_memory_inside_start_threshold",
                "start_distance_z": 6.94,
                "memory_support_cosine": 0.887,
                "is_operational": True,
            },
        ],
        "validation_gate": {
            "overall_status": "pass",
            "selected_start_status": "pass",
            "operational_status": "pass",
            "stress_status": "pass",
            "endpoint_max_abs_error": 0.0,
        },
        "generation": {
            "generated_state_shape": [2, 16, 30, 39],
            "finite_rate": 1.0,
            "sample_count": 16,
            "rollout_temperature": 0.5,
        },
        "condition_only_product_gate": {
            "production_decision": {
                "decision": "supported_calibrated_scenario",
                "ui_guidance": "Show as supported.",
            }
        },
    }


def test_build_prefix_run_record_keeps_hash_not_raw_narrative(tmp_path: Path) -> None:
    report = _report(tmp_path)

    record = build_prefix_run_record(report)

    assert record["record_type"] == "prefix_latent_narrative_scenario_run"
    assert record["status"] == "pass"
    assert record["condition"]["text_memory_dim"] == 128
    assert record["condition"]["market_implication_count"] == 1
    assert record["condition"]["forward_warning_count"] == 1
    assert len(record["condition"]["narrative_text_sha256"]) == 64
    assert record["selected_start"]["variant"] == "nearest_train_start"
    assert record["support"]["candidate_count"] == 1
    assert record["artifacts"][0]["exists"] is True
    assert "Private narrative text" not in json.dumps(record)


def test_write_prefix_run_record_writes_json_and_markdown(tmp_path: Path) -> None:
    record = write_prefix_run_record(
        _report(tmp_path),
        output_dir=tmp_path / "run_record",
    )

    json_path = Path(record["artifact_paths"]["json"])
    markdown_path = Path(record["artifact_paths"]["markdown"])
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert json.loads(json_path.read_text())["record_id"] == record["record_id"]
    assert "Prefix-Latent Run Record" in markdown_path.read_text()


def test_render_markdown_summarizes_support_and_artifacts(tmp_path: Path) -> None:
    markdown = render_markdown(build_prefix_run_record(_report(tmp_path)))

    assert "Prefix-Latent Run Record" in markdown
    assert "joint39_val_0269" in markdown
    assert "Raw narrative text is not stored" in markdown
