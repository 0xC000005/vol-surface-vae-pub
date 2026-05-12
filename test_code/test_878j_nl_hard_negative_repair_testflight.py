import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_hard_negative_repair_testflight import (
    repair_negative_text,
    select_repair_indices,
)


def test_select_repair_indices_keeps_only_negative_rows_for_requested_windows():
    examples = [
        {"window_id": "w0", "role": "anchor", "embedding_index": 0},
        {"window_id": "w0", "role": "negative", "embedding_index": 1},
        {"window_id": "w1", "role": "negative", "embedding_index": 2},
    ]

    assert select_repair_indices(examples, {"w0"}) == [1]


def test_repair_negative_text_makes_contrastive_role_explicit():
    repaired = repair_negative_text(
        original_text="SPX: UP LARGE; VIX: DOWN LARGE",
        observed_fact_tokens="SPX: DOWN LARGE; VIX: UP LARGE",
    )

    assert "HARD_NEGATIVE_CONTROL" in repaired
    assert "not the observed market window" in repaired
    assert "SPX: UP LARGE" in repaired
    assert "SPX: DOWN LARGE" in repaired
