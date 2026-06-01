import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.backfill.block_ar.plot_vix_level_conditional_case_study import (
    _bootstrap_anchor_panel_paths,
)
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (
    build_qualitative_casebook_summary,
    build_portfolio_impact_summary,
    _direct_sni_cache_matches,
    _level_paths_with_start,
    _portfolio_pnl_with_start,
)
from experiments.backfill.block_ar.nl_prefix_latent_portfolio_conditionality_audit import (
    build_portfolio_conditionality_report,
)


def test_bootstrap_anchor_panel_uses_raw_vix_delta_and_log_spx_return():
    start = np.array([100.0, 20.0], dtype=np.float64)
    raw_delta = np.array([[10.0, 2.0]], dtype=np.float64)
    log_delta = np.array([[np.log(1.10), np.log(2.0)]], dtype=np.float64)
    paths = _bootstrap_anchor_panel_paths(
        start_level=start,
        raw_delta=raw_delta,
        log_delta=log_delta,
        factor_names=["factor:spx", "factor:vix"],
        n_samples=1,
        horizon=2,
        seed=7,
        sampled_transition_indices=np.array([[0, 0]], dtype=np.int64),
    )

    np.testing.assert_allclose(paths[0, :, 0], np.array([110.0, 121.0]), rtol=1e-6)
    np.testing.assert_allclose(paths[0, :, 1], np.array([22.0, 24.0]), rtol=1e-6)


def test_narrative_casebook_summary_keeps_text_support_and_raw_levels():
    def make_case(label: str, support_index: int, spx_terminal: float) -> dict:
        states = np.zeros((2, 30, 39), dtype=np.float32)
        states[:, -1, 25] = spx_terminal
        start = np.zeros(39, dtype=np.float32)
        start[25] = 100.0
        return {
            "label": label,
            "case_name": label.lower().replace(" ", "_"),
            "states": states,
            "start": start,
            "report": {
                "cached_query": {
                    "narrative_text": f"{label} narrative",
                    "condition_text": f"{label} condition",
                    "grounding": {
                        "market_implications": [
                            {
                                "market": "SPX",
                                "direction": "up",
                                "confidence": "high",
                            }
                        ],
                        "non_conditioning_forward_language": [
                            {"phrase": "future-risk phrase"}
                        ],
                    },
                    "memory_prior": {
                        "weights": [1.0],
                        "candidate_details": [
                            {
                                "rank": 1,
                                "window_id": f"window_{support_index}",
                                "window_index": support_index,
                                "history_end_date": "2020-01-31",
                                "memory_support_cosine": 0.9,
                                "start_distance_z": 0.0,
                                "recent_prefix_checked": 1,
                                "recent_prefix_mismatches": 0,
                            }
                        ],
                    },
                }
            },
        }

    direct = {"direct_sni_states": np.zeros((2, 30, 39), dtype=np.float32)}
    summary = build_qualitative_casebook_summary(
        [make_case("Reference", 1, 101.0)],
        [make_case("Reference", 1, 101.0), make_case("Stress", 5, 97.0)],
        direct,
    )

    assert summary["same_start_check"]["max_abs_start_difference"] == 0.0
    stress = summary["case_summaries"][1]
    assert stress["grounded_implications"] == "SPX up (high)"
    assert stress["forward_language_excluded"] == "future-risk phrase"
    assert stress["top_support"][0]["direction_check"] == "pass"
    assert stress["terminal_raw_level_summary"]["SPX"]["p50"] == 97.0
    assert summary["reference_pairwise_reads"][0]["support_jaccard_vs_reference"] == 0.0


def test_level_paths_with_start_anchors_generated_fan_to_condition():
    states = np.zeros((2, 3, 4), dtype=np.float32)
    states[0, :, 2] = np.array([101.0, 102.0, 103.0], dtype=np.float32)
    states[1, :, 2] = np.array([99.0, 98.0, 97.0], dtype=np.float32)

    anchored = _level_paths_with_start(states, idx=2, start=100.0)

    assert anchored.shape == (2, 4)
    np.testing.assert_allclose(anchored[:, 0], np.array([100.0, 100.0]))
    np.testing.assert_allclose(anchored[0], np.array([100.0, 101.0, 102.0, 103.0]))
    np.testing.assert_allclose(anchored[1], np.array([100.0, 99.0, 98.0, 97.0]))


def test_portfolio_pnl_with_start_anchors_display_at_zero():
    start = np.ones(39, dtype=np.float32)
    start[25] = 100.0
    start[38] = 20.0
    states = np.broadcast_to(start, (2, 3, 39)).copy().astype(np.float32)
    states[:, :, 25] = np.array(
        [[101.0, 102.0, 103.0], [99.0, 98.0, 97.0]],
        dtype=np.float32,
    )

    pnl = _portfolio_pnl_with_start(states, start)

    assert pnl.shape == (2, 4)
    np.testing.assert_allclose(pnl[:, 0], np.array([0.0, 0.0]))
    assert pnl[0, -1] > 0.0
    assert pnl[1, -1] < 0.0


def test_direct_sni_cache_key_includes_bridge_report_and_global_window():
    cache = {
        "start_index": np.asarray(18, dtype=np.int64),
        "selected_window_global": np.asarray(36, dtype=np.int64),
        "checkpoint": np.asarray("checkpoint-a"),
        "bridge_report": np.asarray("report-a.json"),
        "direct_sni_states": np.zeros((96, 30, 39), dtype=np.float32),
    }

    assert _direct_sni_cache_matches(
        cache,
        start_index=18,
        samples=32,
        checkpoint="checkpoint-a",
        bridge_report="report-a.json",
        selected_window_global=36,
    )
    assert not _direct_sni_cache_matches(
        cache,
        start_index=18,
        samples=32,
        checkpoint="checkpoint-a",
        bridge_report="report-b.json",
        selected_window_global=36,
    )
    assert not _direct_sni_cache_matches(
        cache,
        start_index=18,
        samples=32,
        checkpoint="checkpoint-a",
        bridge_report="report-a.json",
        selected_window_global=40,
    )


def test_portfolio_impact_summary_reports_cross_narrative_risk_spread():
    def make_case(label: str, spx_terminal: float, vix_terminal: float) -> dict:
        start = np.ones(39, dtype=np.float32)
        start[25] = 100.0
        start[38] = 20.0
        states = np.broadcast_to(start, (4, 30, 39)).copy().astype(np.float32)
        states[:, -1, 25] = spx_terminal
        states[:, -1, 38] = vix_terminal
        return {
            "label": label,
            "case_name": label.lower().replace(" ", "_"),
            "states": states,
            "start": start,
            "report": {
                "cached_query": {
                    "memory_prior": {
                        "weights": [1.0],
                        "candidate_details": [
                            {
                                "rank": 1,
                                "window_id": f"{label}_window",
                                "window_index": 1,
                                "history_end_date": "2020-01-31",
                                "memory_support_cosine": 0.9,
                                "start_distance_z": 0.0,
                                "recent_prefix_checked": 1,
                                "recent_prefix_mismatches": 0,
                            }
                        ],
                    }
                }
            },
        }

    summary = build_portfolio_impact_summary(
        [
            make_case("Risk On", 110.0, 18.0),
            make_case("Risk Off", 95.0, 30.0),
        ]
    )

    assert summary["case_summaries"][0]["portfolio_stats"]["terminal_p50"] > 0.0
    assert summary["case_summaries"][1]["portfolio_stats"]["terminal_p50"] < 0.0
    assert summary["cross_narrative_range"]["terminal_p50_range"] > 0.0
    assert summary["case_summaries"][1]["largest_tail_contributors"]


def test_portfolio_conditionality_report_compares_observed_to_controls():
    def make_case(label: str, shift: float) -> dict:
        start = np.ones(39, dtype=np.float32)
        start[25] = 100.0
        start[38] = 20.0
        states = np.broadcast_to(start, (6, 30, 39)).copy().astype(np.float32)
        states[:, -1, 25] = start[25] + shift
        states[:, -1, 38] = start[38] - shift * 0.05
        return {
            "label": label,
            "case_name": label,
            "states": states,
            "start": start,
        }

    observed = [make_case("a", 10.0), make_case("b", -10.0), make_case("c", 2.0)]
    start_only = [make_case("a", 0.0), make_case("b", 0.0), make_case("c", 0.0)]
    repeats = [
        make_case("a#seed_1", 10.0),
        make_case("a#seed_2", 10.2),
        make_case("b#seed_1", -10.0),
        make_case("b#seed_2", -9.8),
    ]
    report = build_portfolio_conditionality_report(
        observed_cases=observed,
        start_only_cases=start_only,
        repeat_cases=repeats,
    )

    assert report["summaries"]["observed_cross_narrative"]["pair_count"] == 3
    assert report["summaries"]["same_narrative_repeat"]["pair_count"] == 2
    assert report["ratios"]["path_vs_repeat"] > 1.0
    assert report["status"] in {"pass", "warning", "fail"}
