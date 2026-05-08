import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_live_casebook import (
    build_case_command,
    default_casebook_stories,
    summarize_case_report,
)


def test_default_casebook_stories_are_story_like() -> None:
    stories = default_casebook_stories()

    assert len(stories) >= 3
    assert all("story" in item and "name" in item for item in stories)
    assert any("risk-on" in item["story"] for item in stories)


def test_build_case_command_sets_live_story_and_output_dir() -> None:
    args = SimpleNamespace(
        script="experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py",
        steps=50,
        samples=2,
        chunk_size=2,
        start_mode="balanced_memory_start",
        device="cuda",
        grounding_model="gpt-5.4-mini",
        embedding_model="text-embedding-3-small",
    )

    command = build_case_command(
        story="A live casebook story.",
        output_dir="outputs/case_01",
        args=args,
    )

    assert "--live-story" in command
    assert command[command.index("--story") + 1] == "A live casebook story."
    assert command[command.index("--output-dir") + 1] == "outputs/case_01"
    assert command[command.index("--steps") + 1] == "50"
    assert command[command.index("--start-mode") + 1] == "balanced_memory_start"


def test_summarize_case_report_extracts_gate_and_grounding() -> None:
    summary = summarize_case_report(
        case_name="risk_on",
        report_path="outputs/report.json",
        report={
            "cached_query": {
                "condition_source": "live_openai_story",
                "grounding": {
                    "narrative_frame": "fragile risk-on rebound",
                    "grounding_warnings": [{"code": "FORWARD_RISK"}],
                },
            },
            "validation_gate": {
                "overall_status": "warning",
                "operational_status": "warning",
                "selected_start_status": "pass",
                "diagnostic_baseline_status": "warning",
                "warning_counts": {"low_memory_compatibility": 1},
                "cases": [
                    {
                        "input_memory_cosine": 0.77,
                        "is_operational": False,
                        "warnings": ["low_memory_compatibility"],
                    },
                    {
                        "input_memory_cosine": 0.85,
                        "is_operational": True,
                        "terminal_mean_abs_delta_z": 0.4,
                        "warnings": [],
                    },
                ],
            },
            "generation": {"generated_state_shape": [2, 2, 30, 39]},
        },
    )

    assert summary["case_name"] == "risk_on"
    assert summary["overall_status"] == "warning"
    assert summary["selected_start_status"] == "pass"
    assert summary["diagnostic_baseline_status"] == "warning"
    assert summary["selected_start_memory_cosine"] == 0.85
    assert summary["selected_start_warnings"] == []
    assert summary["min_memory_cosine"] == 0.77
    assert summary["grounding_warning_count"] == 1
