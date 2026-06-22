#!/usr/bin/env python
"""End-to-end regression tests for the #48 causal-gap guard in the NL bridge.

A causal-leakage fix routes support selection in both
``build_hybrid_start_text_bridge_report`` and
``build_episode_retrieval_bridge_report`` through ``_temporal_gap_filter(...,
query_index=idx)``. This guarantees ``query_window_index - support_window_index
>= temporal_gap`` -- i.e. every selected support window must END at or before the
query window's start, so no near-future / overlapping window leaks into the
support pool.

These tests are deliberately DISCRIMINATING: the within-gap candidates are
constructed to score HIGHEST (identical text + identical terminal state to the
query). Without the ``query_index=`` causal argument, ``_temporal_gap_filter``
would select the highest-scoring near-future window (149) and the gap assertion
would FAIL. The guard must drop them pre-emptively. A boundary candidate at
exactly ``gap`` away (120, gap==30) must be KEPT, catching an off-by-one (<=)
regression of the ">= gap" semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    build_episode_retrieval_bridge_report,
    build_hybrid_start_text_bridge_report,
)

TEMPORAL_GAP = 30
QUERY_INDEX = 150
TOP_K = 3
HISTORY_LEN = 30
FEATURE_DIM = 5
# Cover absolute indices 0..200 so terminal[query_index] / terminal[candidate]
# are always in-bounds.
N_WINDOWS = 201

# Identical "matching" narrative shared by the query and the leak candidates so
# that, absent the causal guard, the near-future windows rank first.
MATCH_TEXT = (
    "Equity volatility spiked as credit spreads widened sharply and the dollar "
    "rallied on a flight to safe-haven assets; funding liquidity tightened while "
    "treasury yields fell on a defensive de-risking rotation."
)
# Distinct (lower-scoring) text for the valid earlier supports.
FILLER_TEXT = (
    "Calm range-bound session; gold and copper drifted with muted realized "
    "volatility and steady carry positioning across commodities."
)
# Partial-match text for the boundary window: scores ABOVE the fillers (so it is
# reliably the top valid candidate and survives the post-causal mutual-diversity
# filter) but BELOW the perfect-match leak windows (so the test stays
# discriminating -- without the guard the leak windows still outrank it).
BOUNDARY_TEXT = (
    "Equity volatility rose as credit spreads widened on a defensive "
    "de-risking move."
)

# Candidate windows. The three near-future windows (149/145/140) get the
# matching text; the valid earlier windows get filler text. 120 sits at EXACTLY
# the gap boundary (150 - 120 == 30) and must be KEPT.
LEAK_WINDOWS = [149, 145, 140]            # gap 1/5/10  -> MUST be excluded
BOUNDARY_WINDOW = 120                      # gap exactly 30 -> MUST be allowed
VALID_WINDOWS = [BOUNDARY_WINDOW, 80, 40, 10]
TRAIN_INDICES = sorted(LEAK_WINDOWS + VALID_WINDOWS)


def _card(window_index: int, text: str) -> dict:
    return {
        "window_id": f"win_{window_index:04d}",
        "support_metadata": {"window_index": int(window_index)},
        "views": {"full_professional": text},
        "scenario_title": f"scenario {window_index}",
        "split": "support_decoder_train",
        "archetype": "test",
    }


def _candidate_text(idx: int) -> str:
    if idx in LEAK_WINDOWS:
        return MATCH_TEXT
    if idx == BOUNDARY_WINDOW:
        return BOUNDARY_TEXT
    return FILLER_TEXT


def _build_train_cards() -> list[dict]:
    return [_card(idx, _candidate_text(idx)) for idx in TRAIN_INDICES]


def _build_query_card() -> dict:
    # Query carries the matching narrative; valid earlier supports do NOT, so the
    # ONLY high-text-score candidates are the (illegal) near-future windows.
    return _card(QUERY_INDEX, MATCH_TEXT)


def _build_history_raw() -> np.ndarray:
    """(N, T, D). Terminal row of the query and of every leak window is made
    identical so that the start-fit term ALSO favours the near-future windows in
    the hybrid path; valid earlier windows get a distinct terminal state."""

    rng = np.random.default_rng(0)
    history = rng.normal(size=(N_WINDOWS, HISTORY_LEN, FEATURE_DIM)).astype(np.float32)
    query_terminal = np.linspace(0.5, 1.5, FEATURE_DIM, dtype=np.float32)
    history[QUERY_INDEX, -1, :] = query_terminal
    for idx in LEAK_WINDOWS:
        history[idx, -1, :] = query_terminal  # perfect start match -> ranks top
    return history


def _metadata() -> list[dict]:
    return [
        {"window_index": idx, "window_id": f"win_{idx:04d}"}
        for idx in range(N_WINDOWS)
    ]


def _assert_pool_causal(report: dict) -> None:
    heldout = report["evaluation"]["heldout_examples"]
    assert heldout, "expected at least one held-out query row"
    for row in heldout:
        query_idx = int(row["window_index"])
        assert query_idx == QUERY_INDEX
        pool = row["top_train_pool"]
        # Non-empty: "all supports satisfy X" over [] is vacuously true.
        assert len(pool) >= 1, f"empty support pool for query {query_idx}"
        selected = {int(item["window_index"]) for item in pool}
        # No near-future / overlapping window leaked in.
        for leak in LEAK_WINDOWS:
            assert leak not in selected, (
                f"leak window {leak} (gap {query_idx - leak}) selected for "
                f"query {query_idx}; causal guard failed"
            )
        # Hard gap assertion: every support strictly earlier by >= temporal_gap.
        for support_idx in selected:
            gap = query_idx - support_idx
            assert gap >= TEMPORAL_GAP, (
                f"support {support_idx} violates causal gap "
                f"(query {query_idx} - support {support_idx} = {gap} < {TEMPORAL_GAP})"
            )
        # Boundary candidate at exactly the gap must be allowed (catches <= off-by-one).
        assert BOUNDARY_WINDOW in selected, (
            f"boundary window {BOUNDARY_WINDOW} (gap exactly {TEMPORAL_GAP}) was "
            "not selected; the guard must keep gap == temporal_gap"
        )


def test_episode_retrieval_bridge_report_enforces_causal_gap() -> None:
    report = build_episode_retrieval_bridge_report(
        train_cards=_build_train_cards(),
        query_cards=[_build_query_card()],
        train_indices=list(TRAIN_INDICES),
        test_indices=[QUERY_INDEX],
        method="hybrid",
        top_k=TOP_K,
        temporal_gap=TEMPORAL_GAP,
        cards_path="synthetic://episode_cards",
    )
    _assert_pool_causal(report)


def test_hybrid_start_text_bridge_report_enforces_causal_gap() -> None:
    report = build_hybrid_start_text_bridge_report(
        train_cards=_build_train_cards(),
        query_cards=[_build_query_card()],
        history_raw=_build_history_raw(),
        train_indices=list(TRAIN_INDICES),
        test_indices=[QUERY_INDEX],
        method="hybrid",
        top_k=TOP_K,
        temporal_gap=TEMPORAL_GAP,
        text_candidate_k=256,
        text_weight=0.5,
        start_weight=0.5,
        cards_path="synthetic://episode_cards",
        arrays_path="synthetic://support_arrays.npz",
    )
    _assert_pool_causal(report)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
