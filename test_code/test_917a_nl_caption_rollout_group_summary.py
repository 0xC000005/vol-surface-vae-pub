import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_caption_rollout_group_summary import (
    parse_query_id,
    summarize_groups,
)


def test_parse_query_id_splits_from_right() -> None:
    assert parse_query_id("joint39_val_0001::caption::with::colons::text-embedding-3-small") == (
        "joint39_val_0001::caption::with",
        "colons",
        "text-embedding-3-small",
    )


def test_summarize_groups_reports_scenario_quality_by_variant_group() -> None:
    reverse_report = {
        "model_reports": [
            {
                "embedding_model": "text-embedding-3-small",
                "rows": [
                    {
                        "window_id": "joint39_val_0001",
                        "variant_id": "legacy_anchor",
                        "variant_group": "simple",
                        "provider": "legacy",
                        "target_cosine": 0.2,
                        "mean_support_cosine": 0.5,
                    },
                    {
                        "window_id": "joint39_val_0001",
                        "variant_id": "fused_caption",
                        "variant_group": "fused_codex",
                        "provider": "codex",
                        "target_cosine": 0.7,
                        "mean_support_cosine": 0.8,
                    },
                ],
            }
        ]
    }
    rollout_report = {
        "window_scores": [
            {
                "query_id": "joint39_val_0001::legacy_anchor::text-embedding-3-small",
                "methods": {
                    "narrative_generator_topk": {
                        "ensemble_crps_z": 0.9,
                        "energy_score_z": 1.1,
                        "coverage_80": 0.4,
                        "terminal_mae_z": 1.0,
                    },
                    "persistence": {
                        "ensemble_crps_z": 1.0,
                        "energy_score_z": 1.2,
                    },
                },
            },
            {
                "query_id": "joint39_val_0001::fused_caption::text-embedding-3-small",
                "methods": {
                    "narrative_generator_topk": {
                        "ensemble_crps_z": 0.8,
                        "energy_score_z": 1.0,
                        "coverage_80": 0.5,
                        "terminal_mae_z": 0.9,
                    },
                    "persistence": {
                        "ensemble_crps_z": 1.0,
                        "energy_score_z": 1.25,
                    },
                },
            },
        ]
    }

    report = summarize_groups(
        reverse_report,
        rollout_report,
        embedding_model="text-embedding-3-small",
    )

    assert report["group_summaries"]["simple"]["crps_improvement_vs_persistence"] == 0.1
    assert report["group_summaries"]["fused_codex"]["crps_improvement_vs_persistence"] == 0.2
    assert report["group_summaries"]["fused_codex"]["mean_target_cosine"] == 0.7
