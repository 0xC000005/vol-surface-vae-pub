import json
import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_grounding_channel_ablation import (
    condition_case_from_report,
    load_case_spec,
    render_markdown,
    summarize_channel_bakeoff,
    write_channel_case_specs,
)


def _condition_report_payload() -> dict:
    return {
        "case_name": "safe_haven_gold_bid",
        "cached_query": {
            "window_id": "safe_haven_gold_bid",
            "narrative_text": "Gold is bid while equities are choppy.",
            "query_text": "CLEANED_CONDITIONING_TEXT: Gold up, VIX up.",
            "embedding_metadata": {
                "grounding_model": "fixture-grounder",
                "query_channel": "grounded_condition",
            },
            "grounding": {
                "narrative_frame": "safe haven bid",
                "condition_only_grounding": {
                    "narrative_frame": "safe haven bid",
                    "cleaned_conditioning_text": "Gold up, VIX up.",
                    "current_market_state_implications": [
                        {
                            "market": "GOLD",
                            "direction": "up",
                            "magnitude": "medium",
                            "confidence": "high",
                            "evidence": ["Gold is bid"],
                            "inferred": False,
                            "horizon": "current_state",
                            "target_use": "support_prior",
                        }
                    ],
                    "recent_regime_implications": [],
                    "grounding_warnings": [],
                    "unsupported_claims": [],
                    "non_conditioning_forward_language": [],
                },
                "condition_only_validation": {"status": "pass"},
                "story_split": {"conditioning_sentences": ["Gold is bid."]},
            },
        },
    }


def test_condition_case_from_report_preserves_story_and_grounding(tmp_path) -> None:
    report_path = tmp_path / "condition_report.json"
    report_path.write_text(json.dumps(_condition_report_payload()), encoding="utf-8")

    case = condition_case_from_report(report_path)

    assert case["case_name"] == "safe_haven_gold_bid"
    assert "Gold is bid" in case["story"]
    assert case["condition_only_grounding"]["narrative_frame"] == "safe haven bid"
    assert case["metadata"]["model"] == "fixture-grounder"


def test_load_case_spec_limits_rows(tmp_path) -> None:
    path = tmp_path / "case_spec.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_name": "a",
                        "start_name": "s0",
                        "condition_report": "a.json",
                        "candidate_index": 1,
                    },
                    {
                        "case_name": "b",
                        "start_name": "s1",
                        "condition_report": "b.json",
                        "candidate_index": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = load_case_spec(path, case_count=1)

    assert rows == [
        {
            "case_name": "a",
            "start_name": "s0",
            "condition_report": "a.json",
            "candidate_index": 1,
        }
    ]


def test_write_channel_case_specs_replaces_condition_reports(tmp_path) -> None:
    case_rows = [
        {
            "case_name": "safe_haven_gold_bid",
            "start_name": "explicit_start_18",
            "condition_report": "old.json",
            "candidate_index": 18,
        }
    ]
    paths = write_channel_case_specs(
        case_rows=case_rows,
        report_paths={
            "safe_haven_gold_bid": {
                "raw_narrative": "new_raw.json",
                "narrative_plus_grounding": "new_combined.json",
            }
        },
        query_channels=["raw_narrative", "narrative_plus_grounding"],
        output_dir=tmp_path,
    )

    raw_spec = json.loads(open(paths["raw_narrative"], encoding="utf-8").read())
    combined_spec = json.loads(
        open(paths["narrative_plus_grounding"], encoding="utf-8").read()
    )

    assert raw_spec["cases"][0]["case_name"].endswith("__raw_narrative")
    assert raw_spec["cases"][0]["condition_report"] == "new_raw.json"
    assert combined_spec["cases"][0]["condition_report"] == "new_combined.json"


def test_summarize_channel_bakeoff_aggregates_metrics() -> None:
    summary = summarize_channel_bakeoff(
        channel="narrative_plus_grounding",
        bakeoff={
            "run_count": 2,
            "rows": [
                {
                    "target_available": True,
                    "validation_operational": "pass",
                    "memory_prior_weighted_start_distance_z": 1.0,
                    "scenario_metrics": {
                        "energy_score_z_improvement_vs_persistence": 0.2,
                        "ensemble_crps_z_improvement_vs_persistence": 0.1,
                    },
                },
                {
                    "target_available": True,
                    "validation_operational": "warning",
                    "memory_prior_weighted_start_distance_z": 3.0,
                    "scenario_metrics": {
                        "energy_score_z_improvement_vs_persistence": 0.4,
                        "ensemble_crps_z_improvement_vs_persistence": 0.3,
                    },
                },
            ],
            "artifact_paths": {"report": "report.json"},
        },
    )

    assert summary["query_channel"] == "narrative_plus_grounding"
    assert summary["target_count"] == 2
    assert summary["operational_status_counts"] == {"pass": 1, "warning": 1}
    assert abs(summary["mean_crps_improvement_vs_persistence"] - 0.2) < 1e-12
    assert abs(summary["mean_weighted_start_distance_z"] - 2.0) < 1e-12


def test_render_markdown_mentions_narrative_plus_implications() -> None:
    markdown = render_markdown(
        {
            "status": "pass",
            "case_count": 1,
            "query_channels": ["narrative_plus_implications"],
            "samples": 2,
            "steps": 100,
            "device": "cpu",
            "channel_summary": [
                {
                    "query_channel": "narrative_plus_implications",
                    "run_count": 1,
                    "target_count": 1,
                    "operational_status_counts": {"pass": 1},
                    "mean_energy_improvement_vs_persistence": 0.1,
                    "mean_crps_improvement_vs_persistence": 0.2,
                    "mean_weighted_start_distance_z": 2.0,
                    "bakeoff_report": "report.json",
                }
            ],
        }
    )

    assert "raw narrative plus explicit implications" in markdown
    assert "`narrative_plus_implications`" in markdown
