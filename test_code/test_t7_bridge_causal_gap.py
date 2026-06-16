"""Regression guard: the embedding-bridge retrievers must enforce the #48 causal gap
(query_window_index - candidate_window_index >= temporal_gap), not just mutual diversity.

Run: PYTHONPATH=. python -m pytest test_code/test_t7_bridge_causal_gap.py -q
"""
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import _temporal_gap_filter


def test_temporal_gap_filter_causal_gap_drops_near_future():
    ranked = [
        {"window_index": 3937},  # gap 7 from query 3944 -> causal-gap violator (overlaps query window)
        {"window_index": 3800},  # gap 144 -> ok
        {"window_index": 3900},  # gap 44 -> ok
    ]
    out = [r["window_index"] for r in _temporal_gap_filter(
        ranked, top_k=3, temporal_gap=30, query_index=3944)]
    assert 3937 not in out                 # leaky candidate dropped
    assert 3800 in out and 3900 in out


def test_temporal_gap_filter_backward_compatible_without_query_index():
    # No query_index -> old mutual-diversity-only behavior (keeps the near-future window).
    ranked = [{"window_index": 3937}, {"window_index": 3800}, {"window_index": 3900}]
    out = [r["window_index"] for r in _temporal_gap_filter(ranked, top_k=3, temporal_gap=30)]
    assert 3937 in out
