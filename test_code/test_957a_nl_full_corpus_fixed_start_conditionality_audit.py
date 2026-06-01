import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_full_corpus_fixed_start_conditionality_audit import (
    fixed_start_decision,
    select_diverse_caption_cases,
    summarize_condition_group,
)


def _example(window_id: str, window_index: int, kind: str, embedding_index: int):
    return {
        "role": "anchor",
        "kind": kind,
        "window_id": window_id,
        "window_index": window_index,
        "embedding_index": embedding_index,
    }


def test_select_diverse_caption_cases_pairs_professional_and_simple_non_train():
    report = {
        "split": {"train_indices": [0], "test_indices": [1, 2, 3]},
        "evaluation": {
            "heldout_examples": [
                _example("w0", 0, "simple_fact_tokens", 0),
                _example("w0", 0, "codex_v2_fused_fact_training_caption", 1),
                _example("w1", 1, "simple_fact_tokens", 2),
                _example("w1", 1, "codex_v2_fused_fact_training_caption", 3),
                _example("w2", 2, "simple_fact_tokens", 4),
                _example("w2", 2, "codex_v2_fused_fact_training_caption", 5),
                _example("w3", 3, "simple_fact_tokens", 6),
                _example("w3", 3, "codex_v2_fused_fact_training_caption", 7),
            ]
        },
    }
    condition_vectors = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    cases = select_diverse_caption_cases(
        report,
        condition_vectors=condition_vectors,
        case_count=2,
    )

    assert len(cases) == 2
    assert {case["window_id"] for case in cases}.issubset({"w1", "w2", "w3"})
    assert all(case["professional_kind"] == "codex_v2_fused_fact_training_caption" for case in cases)
    assert all(case["simple_kind"] == "simple_fact_tokens" for case in cases)


def test_fixed_start_decision_requires_professional_effect_above_start_only():
    start = np.zeros(3, dtype=np.float32)
    prof_a = {
        "case": "a",
        "states": np.zeros((4, 2, 3), dtype=np.float32),
        "start": start,
        "support_indices": [1, 2],
    }
    prof_b = {
        "case": "b",
        "states": np.ones((4, 2, 3), dtype=np.float32),
        "start": start,
        "support_indices": [3, 4],
    }
    start_only_a = {
        "case": "a",
        "states": np.zeros((4, 2, 3), dtype=np.float32),
        "start": start,
        "support_indices": [9, 10],
    }
    start_only_b = {
        "case": "b",
        "states": np.zeros((4, 2, 3), dtype=np.float32),
        "start": start,
        "support_indices": [9, 10],
    }

    professional = summarize_condition_group([prof_a, prof_b], factor_indices={"A": 0, "B": 1})
    start_only = summarize_condition_group([start_only_a, start_only_b], factor_indices={"A": 0, "B": 1})
    decision = fixed_start_decision(
        professional_summary=professional,
        simple_summary=professional,
        start_only_summary=start_only,
    )

    assert professional["mean_factor_terminal_ks"] > start_only["mean_factor_terminal_ks"]
    assert decision["status"] == "pass"
