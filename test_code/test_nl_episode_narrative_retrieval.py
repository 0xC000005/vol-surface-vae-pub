from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_cards import (
    build_episode_card,
    summarize_corpus_quality,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (
    assert_cards_allowed_for_retrieval,
    build_query,
    pairwise_jaccard_summary,
    rank_episode_cards,
)
from experiments.backfill.block_ar.nl_episode_narrative_support_cards import (
    build_support_episode_cards,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (
    build_episode_retrieval_bridge_report,
    build_hybrid_start_text_bridge_report,
    build_start_only_bridge_report,
)
from experiments.backfill.block_ar.nl_episode_narrative_conditionality_lift import (
    build_conditionality_lift_report,
)
from experiments.backfill.block_ar.nl_episode_card_v3_testflight import (
    DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE,
    build_episode_card_v3,
    infer_supported_angles,
)


def _caption(**overrides):
    base = {
        "window_id": "joint39_val_0001",
        "scenario_title": "Liquidity-Led Carry Rebound",
        "archetype": "liquidity_surge",
        "archetype_confidence": "medium",
        "mechanical_summary": (
            "Equities recovered, volatility compressed, credit spreads tightened, "
            "and carry-sensitive exposures recovered after prior stress."
        ),
        "current_market_state": (
            "SPX up, VIX down, BBB OAS tighter, Treasury yields stable, "
            "DXY mixed, gold mixed, crude oil firmer."
        ),
        "trigger": (
            "The supplied prefix does not identify a named catalyst; the visible "
            "trigger is reduced demand for hedges and improved risk appetite."
        ),
        "transmission": (
            "Lower volatility and calmer credit conditions reduce defensive "
            "hedging demand, supporting equity beta and carry."
        ),
        "cross_asset_reaction": (
            "SPX is higher, VIX is lower, BBB spreads are tighter, rates are "
            "stable, DXY is mixed, gold is mixed, and crude is firmer."
        ),
        "sequence": (
            "Volatility eased first, equities recovered, and credit spreads "
            "tightened as investors rebuilt carry exposure."
        ),
        "portfolio_vulnerability": (
            "Short-volatility and long-equity positions benefit; defensive "
            "underweights and credit hedges can lag."
        ),
        "risk_manager_implication": (
            "Risk managers should monitor whether liquidity-led risk appetite is "
            "masking renewed downside convexity."
        ),
        "evidence_used": [
            "SPX 30-day change up medium.",
            "VIX 30-day change down medium.",
            "BBB OAS 30-day change tighter small.",
        ],
        "ambiguity_flags": [
            "The exact macro catalyst is not confirmed by supplied data."
        ],
        "leakage_exclusions": [
            "No realized post-window move used.",
            "No generated scenario outcome used.",
        ],
        "no_forecast_caveat": (
            "This describes current/recent conditions only and is not a forecast."
        ),
        "training_caption": (
            "Recent markets show a liquidity-led carry rebound. SPX is up, VIX is "
            "down, and credit spreads are tighter as lower hedge demand supports "
            "equity beta and carry exposure."
        ),
        "contrastive_captions": [
            "Defensive risk-off with SPX lower, VIX higher, and spreads wider.",
            "Rates-led tightening with yields higher and equity duration pressured.",
        ],
    }
    base.update(overrides)
    return base


def test_build_episode_card_creates_sparse_and_mechanism_views():
    card = build_episode_card(_caption(), source_path="captions.jsonl")

    assert card["window_id"] == "joint39_val_0001"
    assert set(card["views"]) >= {
        "full_professional",
        "sparse_user_query",
        "mechanism_first",
        "one_or_two_factor_headline",
        "factor_list_baseline",
    }
    assert "lower volatility" in card["views"]["mechanism_first"].lower()
    assert "not a forecast" in card["views"]["full_professional"].lower()

    sparse = card["views"]["sparse_user_query"]
    factor_list = card["views"]["factor_list_baseline"]
    assert (
        card["view_metrics"]["sparse_user_query"]["factor_term_count"]
        < card["view_metrics"]["factor_list_baseline"]["factor_term_count"]
    )
    assert len(sparse.split()) < len(card["views"]["full_professional"].split())
    assert "SPX" in factor_list and "VIX" in factor_list


def test_build_episode_card_flags_future_leakage_in_views():
    card = build_episode_card(
        _caption(
            training_caption=(
                "SPX is up and VIX is down. The next 30 days will rally and "
                "the generated scenario has lower VaR."
            )
        ),
        source_path="captions.jsonl",
    )

    assert card["leakage"]["has_leakage"] is True
    assert any("next 30" in hit["snippet"].lower() for hit in card["leakage"]["hits"])
    assert any(
        "generated scenario" in hit["snippet"].lower()
        for hit in card["leakage"]["hits"]
    )


def test_summarize_corpus_quality_detects_factor_list_bias_and_sparse_gap():
    cards = [
        build_episode_card(
            _caption(window_id=f"joint39_val_{idx:04d}"), source_path="x"
        )
        for idx in range(3)
    ]

    report = summarize_corpus_quality(cards, source_path="x")

    assert report["card_count"] == 3
    assert report["training_caption"]["median_factor_term_count"] >= 3
    assert report["training_caption"]["median_word_count"] > 10
    assert report["mechanism_coverage"]["training_caption_with_mechanism_share"] > 0
    assert report["recommendation"]["requires_multi_view_enrichment"] is True
    assert "factor-list" in report["recommendation"]["reason"].lower()


def test_rank_episode_cards_retrieves_matching_sparse_mechanism():
    risk_on = build_episode_card(_caption(window_id="risk_on"), source_path="x")
    risk_off = build_episode_card(
        _caption(
            window_id="risk_off",
            scenario_title="Defensive De-Risking",
            archetype="financial_accident",
            trigger="A confidence shock causes investors to reduce risk exposure.",
            transmission=(
                "Higher hedging demand and weaker risk appetite transmit from "
                "equities into credit and defensive assets."
            ),
            risk_manager_implication=(
                "Risk managers should check equity drawdown, spread widening, "
                "volatility shorts, and safe-haven hedges."
            ),
            training_caption=(
                "Risk-off de-risking with SPX down, VIX up, credit spreads wider, "
                "Treasury yields lower, gold higher, and crude weaker."
            ),
        ),
        source_path="x",
    )

    query = build_query(
        label="risk_off_query",
        text=(
            "A confidence shock is forcing de-risking. Hedging demand is rising, "
            "credit risk appetite is weaker, and defensive assets are in demand."
        ),
    )
    ranked = rank_episode_cards(query, [risk_on, risk_off], method="hybrid", top_k=2)

    assert ranked[0]["window_id"] == "risk_off"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["score_components"]["lexical"] > 0


def test_pairwise_jaccard_summary_tracks_retrieval_diversity():
    result_sets = {
        "risk_on": ["a", "b", "c"],
        "risk_off": ["d", "e", "f"],
        "liquidity": ["a", "d", "g"],
    }

    summary = pairwise_jaccard_summary(result_sets)

    assert summary["pair_count"] == 3
    assert summary["min_jaccard"] == 0.0
    assert summary["max_jaccard"] == 0.2


def test_rank_episode_cards_can_enforce_temporal_gap():
    cards = []
    for idx in [100, 105, 180]:
        card = build_episode_card(
            _caption(window_id=f"joint39_train_{idx:04d}"), source_path="x"
        )
        card["support_metadata"] = {"window_index": idx}
        cards.append(card)
    query = build_query(
        label="risk_on_query",
        text="Risk appetite is returning as equities recover and volatility compresses.",
    )

    ranked = rank_episode_cards(query, cards, method="hybrid", top_k=2, temporal_gap=30)

    assert [row["window_index"] for row in ranked] == [100, 180]


def test_retrieval_script_can_run_from_repo_root():
    script = ROOT / "experiments/backfill/block_ar/nl_episode_narrative_retrieval.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Local narrative-to-narrative retrieval" in result.stdout


def test_build_support_episode_cards_from_raw_history():
    import numpy as np

    history_raw = np.zeros((2, 30, 39), dtype=np.float32)
    history_raw[:, :, 25] = 100.0
    history_raw[:, :, 38] = 20.0
    history_raw[:, :, 35] = 200.0
    history_raw[:, :, 31] = 50.0
    history_raw[:, :, 33] = 3.0
    history_raw[:, :, 28] = 100.0
    history_raw[:, :, 37] = 1500.0

    history_raw[0, -1, 25] = 110.0
    history_raw[0, -1, 38] = 15.0
    history_raw[0, -1, 35] = 180.0
    history_raw[0, -1, 31] = 52.0

    history_raw[1, -1, 25] = 92.0
    history_raw[1, -1, 38] = 28.0
    history_raw[1, -1, 35] = 230.0
    history_raw[1, -1, 37] = 1540.0

    metadata = [
        {
            "window_id": "joint39_train_0000",
            "window_index": 0,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-03",
            "calendar_end_date": "2000-02-14",
        },
        {
            "window_id": "joint39_train_0001",
            "window_index": 1,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-04",
            "calendar_end_date": "2000-02-15",
        },
    ]

    cards, report = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=[0, 1],
        source_path="support_bank_arrays.npz",
    )

    assert len(cards) == 2
    assert report["card_count"] == 2
    assert report["source_inventory"] == "support_bank_raw_history"
    assert cards[0]["window_id"] == "joint39_train_0000"
    assert "risk-on" in cards[0]["scenario_title"].lower()
    assert "volatility" in cards[0]["views"]["full_professional"].lower()
    assert "credit" in cards[0]["views"]["factor_list_baseline"].lower()


def test_support_cards_do_not_call_gold_duration_risk_on_safe_haven():
    import numpy as np

    history_raw = np.zeros((3, 30, 39), dtype=np.float32)
    history_raw[:, :, 25] = 100.0
    history_raw[:, :, 38] = 20.0
    history_raw[:, :, 33] = 3.0
    history_raw[:, :, 37] = 1500.0

    # Query-like support: Gold and duration bid, but SPX rallies and VIX falls.
    history_raw[0, -1, 25] = 120.0
    history_raw[0, -1, 38] = 12.0
    history_raw[0, -1, 33] = 2.6
    history_raw[0, -1, 37] = 1560.0

    # Opposite examples give the synthetic corpus nonzero scales.
    history_raw[1, -1, 25] = 80.0
    history_raw[1, -1, 38] = 28.0
    history_raw[1, -1, 33] = 3.4
    history_raw[1, -1, 37] = 1440.0

    history_raw[2, -1, 25] = 92.0
    history_raw[2, -1, 38] = 30.0
    history_raw[2, -1, 33] = 2.5
    history_raw[2, -1, 37] = 1580.0

    metadata = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "window_index": idx,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-03",
            "calendar_end_date": "2000-02-14",
        }
        for idx in range(3)
    ]

    cards, _ = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=[0, 1, 2],
        source_path="support_bank_arrays.npz",
    )

    assert cards[0]["scenario_title"] == "Gold up in risk-on relief"
    assert cards[0]["archetype_confidence"] == "low"
    assert "Safe-haven gold bid" not in cards[0]["scenario_title"]
    evidence = " ".join(cards[0]["caption_fields"]["evidence_used"])
    assert "GOLD" in evidence
    assert "SPX" in evidence
    assert "VIX" in evidence


def test_support_cards_identify_classic_safe_haven_gold_with_high_confidence():
    import numpy as np

    history_raw = np.zeros((3, 30, 39), dtype=np.float32)
    history_raw[:, :, 25] = 100.0
    history_raw[:, :, 38] = 20.0
    history_raw[:, :, 33] = 3.0
    history_raw[:, :, 37] = 1500.0

    history_raw[0, -1, 25] = 80.0
    history_raw[0, -1, 38] = 30.0
    history_raw[0, -1, 33] = 2.5
    history_raw[0, -1, 37] = 1550.0
    history_raw[1, -1, 25] = 120.0
    history_raw[1, -1, 38] = 12.0
    history_raw[1, -1, 33] = 3.5
    history_raw[1, -1, 37] = 1450.0

    metadata = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "window_index": idx,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-03",
            "calendar_end_date": "2000-02-14",
        }
        for idx in range(3)
    ]

    cards, _ = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=[0, 1, 2],
        source_path="support_bank_arrays.npz",
    )

    assert cards[0]["scenario_title"] == "Classic safe-haven gold risk-off"
    assert cards[0]["archetype_confidence"] == "high"
    assert "Gold is higher" in cards[0]["caption_fields"]["transmission"]


def test_support_cards_force_anchor_evidence_for_commodity_and_dollar_titles():
    import numpy as np

    history_raw = np.zeros((3, 30, 39), dtype=np.float32)
    history_raw[:, :, 25] = 100.0
    history_raw[:, :, 28] = 100.0
    history_raw[:, :, 31] = 50.0
    history_raw[:, :, 32] = 2.0
    history_raw[:, :, 33] = 3.0
    history_raw[:, :, 38] = 20.0

    # Commodity/rates pressure where other channels are larger than crude.
    history_raw[0, -1, 31] = 53.0
    history_raw[0, -1, 33] = 3.4
    history_raw[0, -1, 28] = 108.0

    # Dollar liquidity where other channels are larger than DXY.
    history_raw[1, -1, 28] = 103.0
    history_raw[1, -1, 25] = 80.0
    history_raw[1, -1, 38] = 30.0

    history_raw[2, -1, 31] = 47.0
    history_raw[2, -1, 33] = 2.6
    history_raw[2, -1, 28] = 97.0

    metadata = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "window_index": idx,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-03",
            "calendar_end_date": "2000-02-14",
        }
        for idx in range(3)
    ]

    cards, _ = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=[0, 1, 2],
        source_path="support_bank_arrays.npz",
    )

    commodity = cards[0]
    assert commodity["scenario_title"] == "Commodity-inflation pressure"
    assert "CRUDE_OIL" in " ".join(commodity["caption_fields"]["evidence_used"])

    dollar = cards[1]
    assert dollar["scenario_title"] == "Dollar-liquidity squeeze"
    assert "DXY" in " ".join(dollar["caption_fields"]["evidence_used"])


def test_episode_card_v3_local_narrative_builder_is_disabled():
    import numpy as np

    history_raw = np.zeros((3, 30, 39), dtype=np.float32)
    history_raw[:, :, 25] = 100.0
    history_raw[:, :, 38] = 20.0
    history_raw[:, :, 35] = 200.0
    history_raw[:, :, 33] = 3.0
    history_raw[:, :, 37] = 1500.0

    history_raw[0, -1, 25] = 80.0
    history_raw[0, -1, 38] = 31.0
    history_raw[0, -1, 35] = 230.0
    history_raw[0, -1, 33] = 2.5
    history_raw[0, -1, 37] = 1560.0
    history_raw[1, -1, 25] = 120.0
    history_raw[1, -1, 38] = 12.0
    history_raw[1, -1, 35] = 180.0
    history_raw[1, -1, 33] = 3.5
    history_raw[1, -1, 37] = 1440.0

    metadata = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "window_index": idx,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-03",
            "calendar_end_date": "2000-02-14",
        }
        for idx in range(3)
    ]
    source_cards, _ = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=[0, 1, 2],
        source_path="support_bank_arrays.npz",
    )

    with pytest.raises(RuntimeError, match="deterministic/template"):
        build_episode_card_v3(source_cards[0], rich=True)
    assert "smoke tests" in DETERMINISTIC_NARRATIVE_DISABLED_MESSAGE


def test_retrieval_rejects_invalid_local_episode_card_v3_artifacts(tmp_path: Path):
    cards = [
        {
            "schema_version": "nl_episode_card_v3",
            "window_id": "joint39_train_0001",
            "views": {"sparse_user_query": "Locally rendered narrative."},
        }
    ]

    with pytest.raises(ValueError, match="directly Codex/GPT-authored"):
        assert_cards_allowed_for_retrieval(cards, path=tmp_path / "cards.jsonl")


def test_infer_supported_angles_splits_gold_duration_from_classic_safe_haven():
    import numpy as np

    history_raw = np.zeros((3, 30, 39), dtype=np.float32)
    history_raw[:, :, 25] = 100.0
    history_raw[:, :, 38] = 20.0
    history_raw[:, :, 33] = 3.0
    history_raw[:, :, 37] = 1500.0
    history_raw[0, -1, 25] = 120.0
    history_raw[0, -1, 38] = 12.0
    history_raw[0, -1, 33] = 2.5
    history_raw[0, -1, 37] = 1560.0
    history_raw[1, -1, 25] = 80.0
    history_raw[1, -1, 38] = 30.0
    history_raw[1, -1, 33] = 3.5
    history_raw[1, -1, 37] = 1440.0
    metadata = [
        {
            "window_id": f"joint39_train_{idx:04d}",
            "window_index": idx,
            "manifest_split": "support_train",
            "calendar_start_date": "2000-01-03",
            "calendar_end_date": "2000-02-14",
        }
        for idx in range(3)
    ]
    source_cards, _ = build_support_episode_cards(
        history_raw,
        metadata,
        train_indices=[0, 1, 2],
        source_path="support_bank_arrays.npz",
    )
    names = [spec["angle_name"] for spec in infer_supported_angles(source_cards[0])]

    assert "Gold up in risk-on relief" in names
    assert "Classic safe-haven gold risk-off" not in names


def test_build_episode_retrieval_bridge_report_uses_test_queries_and_train_support():
    train_a = build_episode_card(
        _caption(window_id="joint39_train_0000"), source_path="x"
    )
    train_a["support_metadata"] = {"window_index": 0}
    train_b = build_episode_card(
        _caption(window_id="joint39_train_0040"), source_path="x"
    )
    train_b["support_metadata"] = {"window_index": 40}
    test = build_episode_card(_caption(window_id="joint39_train_0080"), source_path="x")
    test["support_metadata"] = {"window_index": 80}

    report = build_episode_retrieval_bridge_report(
        train_cards=[train_a, train_b],
        query_cards=[test],
        train_indices=[0, 40],
        test_indices=[80],
        method="hybrid",
        top_k=2,
        temporal_gap=30,
        cards_path="cards.jsonl",
    )

    rows = report["evaluation"]["heldout_examples"]
    assert len(rows) == 1
    assert rows[0]["role"] == "anchor"
    assert rows[0]["window_index"] == 80
    assert [item["window_index"] for item in rows[0]["top_train_pool"]] == [0, 40]
    assert report["split"]["train_indices"] == [0, 40]


def test_build_start_only_bridge_report_ranks_by_terminal_state_distance():
    import numpy as np

    history_raw = np.zeros((4, 30, 3), dtype=np.float32)
    history_raw[0, -1] = [0.0, 0.0, 0.0]
    history_raw[1, -1] = [10.0, 0.0, 0.0]
    history_raw[2, -1] = [0.2, 0.0, 0.0]
    history_raw[3, -1] = [9.8, 0.0, 0.0]
    metadata = [
        {"window_id": f"joint39_train_{idx:04d}", "window_index": idx}
        for idx in range(4)
    ]

    report = build_start_only_bridge_report(
        history_raw=history_raw,
        metadata=metadata,
        train_indices=[0, 1],
        test_indices=[2, 3],
        top_k=1,
        temporal_gap=0,
        cards_by_index={},
        arrays_path="arrays.npz",
    )

    rows = report["evaluation"]["heldout_examples"]
    assert rows[0]["window_index"] == 2
    assert rows[0]["top_train_pool"][0]["window_index"] == 0
    assert rows[1]["window_index"] == 3
    assert rows[1]["top_train_pool"][0]["window_index"] == 1
    assert rows[0]["kind"] == "start_only_terminal_state_retrieval"


def test_hybrid_start_text_bridge_report_combines_text_and_start_fit():
    import numpy as np

    far_text = build_episode_card(
        _caption(
            window_id="joint39_train_0000",
            scenario_title="Risk-on",
            training_caption="Risk appetite is returning and equities are recovering.",
        ),
        source_path="x",
    )
    far_text["support_metadata"] = {"window_index": 0}
    close_start = build_episode_card(
        _caption(
            window_id="joint39_train_0001",
            scenario_title="Risk-on weaker text",
            training_caption="Markets are mixed with only mild recovery evidence.",
        ),
        source_path="x",
    )
    close_start["support_metadata"] = {"window_index": 1}
    query_card = build_episode_card(
        _caption(
            window_id="joint39_train_0002",
            training_caption="Risk appetite is returning and equities are recovering.",
        ),
        source_path="x",
    )
    query_card["support_metadata"] = {"window_index": 2}

    history_raw = np.zeros((3, 30, 2), dtype=np.float32)
    history_raw[0, -1] = [100.0, 0.0]
    history_raw[1, -1] = [2.0, 0.0]
    history_raw[2, -1] = [2.1, 0.0]

    report = build_hybrid_start_text_bridge_report(
        train_cards=[far_text, close_start],
        query_cards=[query_card],
        history_raw=history_raw,
        train_indices=[0, 1],
        test_indices=[2],
        method="hybrid",
        top_k=1,
        temporal_gap=0,
        text_candidate_k=2,
        text_weight=0.1,
        start_weight=0.9,
        cards_path="cards.jsonl",
        arrays_path="arrays.npz",
    )

    row = report["evaluation"]["heldout_examples"][0]
    assert row["kind"] == "hybrid_start_text_episode_retrieval"
    assert row["top_train_pool"][0]["window_index"] == 1
    assert row["top_train_pool"][0]["score_components"]["text_score"] > 0.0


def test_conditionality_lift_report_compares_narrative_to_start_only():
    import numpy as np

    narrative_report = {
        "summary": {
            "narrative_generator_topk": {
                "ensemble_crps_z_mean": 0.8,
                "energy_score_z_mean": 0.9,
            }
        },
        "window_scores": [
            {
                "window_index": 10,
                "top_train_window_ids": ["a", "b"],
                "top_train_indices": [0, 1],
            }
        ],
    }
    start_only_report = {
        "summary": {
            "narrative_generator_topk": {
                "ensemble_crps_z_mean": 1.0,
                "energy_score_z_mean": 1.0,
            }
        },
        "window_scores": [
            {
                "window_index": 10,
                "top_train_window_ids": ["c", "d"],
                "top_train_indices": [2, 3],
            }
        ],
    }
    narrative_arrays = {
        "delta_scale": np.ones((2, 3), dtype=np.float32),
        "narrative_10": np.asarray(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
    }
    start_only_arrays = {
        "delta_scale": np.ones((2, 3), dtype=np.float32),
        "narrative_10": np.asarray(
            [[[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
    }

    report = build_conditionality_lift_report(
        narrative_report=narrative_report,
        start_only_report=start_only_report,
        narrative_arrays=narrative_arrays,
        start_only_arrays=start_only_arrays,
    )

    assert report["window_count"] == 1
    assert report["aggregate"]["mean_support_jaccard"] == 0.0
    assert report["aggregate"]["mean_terminal_factor_ks"] > 0.0
    assert report["quality_guardrail"]["crps_delta_vs_start_only"] < 0.0
    assert report["verdict"] == "conditionality_lift_detected"
