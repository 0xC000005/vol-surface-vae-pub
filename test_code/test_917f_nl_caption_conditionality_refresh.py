import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_caption_conditionality_refresh import (
    gain_persists,
    parse_query_id,
    select_qualitative_window,
)


def test_parse_query_id_handles_embedding_model_suffix():
    assert parse_query_id("joint39_val_0001::simple_fact_tokens::text-embedding-3-small") == (
        "joint39_val_0001",
        "simple_fact_tokens",
        "text-embedding-3-small",
    )


def test_gain_persists_requires_crps_and_energy():
    small = {
        "group_summaries": {
            "simple": {
                "crps_improvement_vs_persistence": 0.10,
                "energy_improvement_vs_persistence": 0.12,
            },
            "fused_codex": {
                "crps_improvement_vs_persistence": 0.14,
                "energy_improvement_vs_persistence": 0.15,
            },
        }
    }
    large = {
        "group_summaries": {
            "simple": {
                "crps_improvement_vs_persistence": 0.11,
                "energy_improvement_vs_persistence": 0.13,
            },
            "fused_codex": {
                "crps_improvement_vs_persistence": 0.12,
                "energy_improvement_vs_persistence": 0.14,
            },
        }
    }
    report = gain_persists(small, large)
    assert report["passes_all_models"] is True
    assert report["checks"]["small"]["crps_gain"] > 0
    assert report["checks"]["large"]["energy_gain"] > 0


def test_select_qualitative_window_prefers_non_train_improvement():
    reverse = {
        "model_reports": [
            {
                "embedding_model": "text-embedding-3-small",
                "rows": [
                    {
                        "window_id": "w1",
                        "variant_id": "simple_fact_tokens",
                        "split": "test",
                        "support_rows": [{"window_index": 1}, {"window_index": 2}],
                    },
                    {
                        "window_id": "w1",
                        "variant_id": "codex_v2_fused_fact_training_caption",
                        "split": "test",
                        "support_rows": [{"window_index": 5}, {"window_index": 6}],
                    },
                ],
            }
        ]
    }
    rollout = {
        "window_scores": [
            {
                "query_id": "w1::simple_fact_tokens::text-embedding-3-small",
                "window_id": "w1",
                "window_index": 1,
                "block_window_index": 11,
                "methods": {
                    "narrative_generator_topk": {
                        "ensemble_crps_z": 2.0,
                        "energy_score_z": 3.0,
                    },
                    "persistence": {
                        "ensemble_crps_z": 2.0,
                        "energy_score_z": 3.0,
                    },
                },
            },
            {
                "query_id": "w1::codex_v2_fused_fact_training_caption::text-embedding-3-small",
                "window_id": "w1",
                "window_index": 1,
                "block_window_index": 11,
                "methods": {
                    "narrative_generator_topk": {
                        "ensemble_crps_z": 1.2,
                        "energy_score_z": 1.8,
                    },
                    "persistence": {
                        "ensemble_crps_z": 2.0,
                        "energy_score_z": 3.0,
                    },
                },
            },
        ]
    }
    selected = select_qualitative_window(reverse, rollout, embedding_model="text-embedding-3-small")
    assert selected["window_id"] == "w1"
    assert selected["split"] == "test"
    assert selected["crps_delta_simple_minus_fused"] == 0.8
    assert selected["support_jaccard_simple_fused"] == 0.0
