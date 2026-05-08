import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_policy_audit import (
    AuditCase,
    aggregate_rows,
    choose_recommendation,
    parse_case,
    slugify,
    story_smoke_command,
    summarize_report,
)


def fake_report(path: Path, *, status: str, warning: str | None = None) -> None:
    warnings = [warning] if warning else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cached_query": {
                    "memory_prior": {
                        "support_alignment": {
                            "status": "pass",
                            "checked_count": 3,
                            "match_count": 3,
                            "mismatch_count": 0,
                        }
                    }
                },
                "variant_rows": [
                    {
                        "is_operational": True,
                        "start_window_id": "joint39_val_0040",
                        "start_source_index": 4050,
                        "start_manifest_split": "train",
                        "start_selection_method": "max_memory_support",
                    }
                ],
                "validation_gate": {
                    "overall_status": status,
                    "operational_status": status,
                    "selected_start_status": status,
                    "cases": [
                        {
                            "is_operational": True,
                            "query_window_index": 153,
                            "start_window_index": 40,
                            "start_distance_z": 14.0,
                            "input_memory_cosine": 0.97,
                            "mean_abs_delta_z": 0.8,
                            "terminal_mean_abs_delta_z": 0.9,
                            "warnings": warnings,
                            "failures": [],
                        }
                    ],
                },
            }
        )
    )


def test_parse_case_requires_name_and_path() -> None:
    parsed = parse_case("Risk On=/tmp/report.json")
    assert parsed.name == "Risk_On"
    assert parsed.condition_report == Path("/tmp/report.json")
    with pytest.raises(Exception):
        parse_case("/tmp/report.json")


def test_summarize_report_extracts_operational_metrics(tmp_path: Path) -> None:
    report = tmp_path / "run" / "prefix_latent_story_smoke_report.json"
    fake_report(report, status="warning", warning="large_rollout_shift")
    row = summarize_report("defensive", "balanced_memory_start", report)
    assert row["selected_start_status"] == "warning"
    assert row["warning_reasons"] == ["large_rollout_shift"]
    assert row["support_mismatch_count"] == 0
    assert row["start_window_id"] == "joint39_val_0040"


def test_aggregate_rows_prefers_lower_warning_count() -> None:
    rows = [
        {
            "start_mode": "balanced_memory_start",
            "selected_start_status": "warning",
            "warning_reasons": ["large_rollout_shift"],
            "failure_reasons": [],
            "start_distance_z": 14.0,
            "input_memory_cosine": 0.97,
            "terminal_mean_abs_delta_z": 1.1,
            "support_mismatch_count": 0,
            "support_checked_count": 3,
        },
        {
            "start_mode": "memory_nearest_start",
            "selected_start_status": "pass",
            "warning_reasons": [],
            "failure_reasons": [],
            "start_distance_z": 18.0,
            "input_memory_cosine": 0.99,
            "terminal_mean_abs_delta_z": 0.9,
            "support_mismatch_count": 0,
            "support_checked_count": 3,
        },
    ]
    aggregates = aggregate_rows(rows)
    recommendation = choose_recommendation(aggregates)
    assert recommendation["start_mode"] == "memory_nearest_start"


def test_story_smoke_command_contains_condition_report(tmp_path: Path) -> None:
    args = Namespace(
        memory_prior_mode="soft_topk_combined",
        memory_prior_top_k=8,
        memory_prior_temperature=0.2,
        implication_alignment_weight=0.25,
        steps=100,
        samples=2,
        chunk_size=2,
        device="cuda",
    )
    command = story_smoke_command(
        condition_report=tmp_path / "condition.json",
        output_dir=tmp_path / "out",
        start_mode="balanced_memory_start",
        args=args,
    )
    assert "--condition-report" in command
    assert "balanced_memory_start" in command


def test_slugify_has_safe_fallback() -> None:
    assert slugify("fragile risk-on") == "fragile_risk-on"
    assert slugify("!!!") == "case"
