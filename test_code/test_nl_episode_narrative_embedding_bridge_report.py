from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (
    build_embedding_bridge_report,
)


def _card(idx: int, text: str, split: str) -> dict:
    return {
        "window_id": f"joint39_train_{idx:04d}",
        "window_index": idx,
        "split": split,
        "scenario_title": text.split()[0],
        "archetype": "test",
        "views": {
            "full_professional": text,
            "sparse_user_query": text,
            "technical_factor_evidence": text,
        },
    }


def test_embedding_bridge_report_hash_backend_is_evaluator_compatible(tmp_path: Path) -> None:
    cards = [
        _card(0, "equity volatility stress with credit widening", "support_train"),
        _card(10, "commodity inflation pressure with oil higher", "support_train"),
        _card(40, "equity volatility stress with hedging demand", "support_decoder_test"),
    ]
    support_report = {
        "window_metadata": [
            {"window_index": 0, "window_id": "joint39_train_0000", "manifest_split": "support_train"},
            {"window_index": 10, "window_id": "joint39_train_0010", "manifest_split": "support_train"},
            {"window_index": 40, "window_id": "joint39_train_0040", "manifest_split": "support_decoder_test"},
        ]
    }
    support_arrays = tmp_path / "support_arrays.npz"
    np.savez_compressed(
        support_arrays,
        history_raw=np.zeros((41, 30, 3), dtype=np.float32),
    )

    report, arrays = build_embedding_bridge_report(
        cards=cards,
        support_report=support_report,
        support_arrays_path=support_arrays,
        output_dir=tmp_path / "out",
        selector_mode="episode_embedding",
        embedding_backend="hash",
        embedding_model="unit-hash",
        dotenv_path=".env",
        embedding_batch_size=16,
        hash_dim=64,
        top_k=1,
        temporal_gap=0,
        max_query_windows=0,
        text_candidate_k=2,
        text_weight=0.25,
        start_weight=0.75,
    )

    row = report["evaluation"]["heldout_examples"][0]
    assert report["schema_version"] == "nl_episode_embedding_bridge_report_v1"
    assert row["top_train_pool"][0]["window_index"] == 0
    assert row["top_train_pool"][0]["score_components"]["method"] == "semantic_embedding"
    assert arrays["text_embeddings"].shape[0] >= 3
