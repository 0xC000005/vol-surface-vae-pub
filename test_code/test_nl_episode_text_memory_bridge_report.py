from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_text_memory_bridge_report import (
    build_text_memory_bridge_report,
)


def _card(idx: int, text: str, split: str) -> dict:
    return {
        "window_id": f"joint39_train_{idx:04d}",
        "window_index": idx,
        "split": split,
        "scenario_title": text.split()[0],
        "archetype": "unit",
        "valid_for_training_retrieval": True,
        "narrative_authoring": {"local_template_prose_used": False},
        "views": {
            "sparse_user_query": text,
            "weekly_risk_monitor": f"Weekly monitor: {text}",
            "full_professional": f"Professional memo: {text}",
        },
    }


def test_text_memory_bridge_report_trains_hash_adapter_and_emits_scenario_rows(
    tmp_path: Path,
) -> None:
    cards = [
        _card(0, "equity volatility stress with credit widening", "support_train"),
        _card(10, "commodity inflation pressure with oil higher", "support_train"),
        _card(40, "equity volatility stress with hedging demand", "support_decoder_test"),
    ]
    support_report = {
        "window_metadata": [
            {
                "window_index": 0,
                "window_id": "joint39_train_0000",
                "manifest_split": "support_train",
            },
            {
                "window_index": 10,
                "window_id": "joint39_train_0010",
                "manifest_split": "support_train",
            },
            {
                "window_index": 40,
                "window_id": "joint39_train_0040",
                "manifest_split": "support_decoder_test",
            },
        ]
    }
    memory_targets = np.zeros((41, 4), dtype=np.float32)
    memory_targets[0] = [1.0, 0.0, 0.0, 0.0]
    memory_targets[10] = [0.0, 1.0, 0.0, 0.0]
    memory_targets[40] = [1.0, 0.1, 0.0, 0.0]
    support_arrays = tmp_path / "support_arrays.npz"
    np.savez_compressed(
        support_arrays,
        memory_targets=memory_targets,
        history_raw=np.zeros((41, 30, 3), dtype=np.float32),
        train_indices=np.asarray([0, 10], dtype=np.int64),
        test_indices=np.asarray([40], dtype=np.int64),
    )

    report, arrays = build_text_memory_bridge_report(
        cards=cards,
        support_report=support_report,
        support_arrays_path=support_arrays,
        output_dir=tmp_path / "out",
        embedding_backend="hash",
        embedding_model="unit-hash",
        dotenv_path=".env",
        embedding_batch_size=16,
        hash_dim=64,
        adapter_steps=8,
        adapter_batch_size=4,
        top_k=1,
        temporal_gap=0,
        max_query_windows=0,
        device="cpu",
    )

    row = report["evaluation"]["heldout_examples"][0]
    assert report["schema_version"] == "nl_episode_text_memory_bridge_report_v1"
    assert report["adapter_training"]["train_example_count"] > 0
    assert row["role"] == "anchor"
    assert row["kind"] == "text_memory_bridge"
    assert row["embedding_index"] >= 0
    assert len(row["top_train_pool"]) == 1
    assert row["top_train_pool"][0]["score_components"]["method"] == "text_memory_bridge"
    assert arrays["condition_vectors"].shape[1] == 4
    assert arrays["memory_targets"].shape == (41, 4)


def test_text_memory_bridge_grounded_top3_90_keeps_direction_checked_support(
    tmp_path: Path,
) -> None:
    cards = [
        _card(0, "equity volatility stress with credit widening", "support_train"),
        _card(10, "equity calm with credit tightening", "support_train"),
        _card(40, "equity volatility stress with hedging demand", "support_decoder_test"),
    ]
    cards[0]["views"]["factor_list_baseline"] = (
        "Mechanical baseline: SPX down large; VIX up large; BBB_OAS wider large."
    )
    cards[1]["views"]["factor_list_baseline"] = (
        "Mechanical baseline: SPX up large; VIX down large; BBB_OAS tighter large."
    )
    cards[2]["views"]["factor_list_baseline"] = (
        "Mechanical baseline: SPX down large; VIX up large; BBB_OAS wider large."
    )
    support_report = {
        "window_metadata": [
            {"window_index": 0, "window_id": "joint39_train_0000", "manifest_split": "support_train"},
            {"window_index": 10, "window_id": "joint39_train_0010", "manifest_split": "support_train"},
            {"window_index": 40, "window_id": "joint39_train_0040", "manifest_split": "support_decoder_test"},
        ]
    }
    memory_targets = np.zeros((41, 4), dtype=np.float32)
    memory_targets[0] = [1.0, 0.0, 0.0, 0.0]
    memory_targets[10] = [0.0, 1.0, 0.0, 0.0]
    memory_targets[40] = [1.0, 0.1, 0.0, 0.0]
    support_arrays = tmp_path / "support_arrays.npz"
    np.savez_compressed(
        support_arrays,
        memory_targets=memory_targets,
        history_raw=np.zeros((41, 30, 3), dtype=np.float32),
        train_indices=np.asarray([0, 10], dtype=np.int64),
        test_indices=np.asarray([40], dtype=np.int64),
    )

    report, _arrays = build_text_memory_bridge_report(
        cards=cards,
        support_report=support_report,
        support_arrays_path=support_arrays,
        output_dir=tmp_path / "out",
        embedding_backend="hash",
        embedding_model="unit-hash",
        dotenv_path=".env",
        embedding_batch_size=16,
        hash_dim=64,
        adapter_steps=8,
        adapter_batch_size=4,
        top_k=3,
        temporal_gap=0,
        max_query_windows=0,
        device="cpu",
        grounding_gate=True,
        max_grounding_claims=3,
        max_grounding_mismatches=0,
        top3_90=True,
        support_pool_size=3,
    )

    row = report["evaluation"]["heldout_examples"][0]
    assert row["kind"] == "text_memory_bridge_grounded_top3_90"
    assert row["required_grounding_claims"]
    assert [item["window_index"] for item in row["top_train_pool"]] == [0]
    selected = row["top_train_pool"][0]
    assert selected["posterior_role"] == "top3_90_selected"
    assert selected["score_components"]["method"] == "text_memory_bridge_grounded_top3_90"
    assert (
        selected["score_components"]["direction_check"]["status"] == "pass"
    )
