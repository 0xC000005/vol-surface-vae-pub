import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_book_conditioned_quality_guard import (
    blend_book_priors,
    narrative_book_relevance_weights,
    support_priors_by_book,
)


def test_narrative_book_relevance_weights_focus_commodity_story():
    grounding = {
        "condition_only_grounding": {
            "current_market_state_implications": [
                {"market": "CRUDE_OIL", "direction": "up", "confidence": "high"},
                {"market": "US10Y", "direction": "up", "confidence": "high"},
                {"market": "SPX", "direction": "down", "confidence": "medium"},
            ]
        }
    }

    weights = narrative_book_relevance_weights(grounding)

    assert weights["commodity_inflation"] > weights["dollar_liquidity"]
    assert sum(weights.values()) == 1.0


def test_narrative_book_relevance_weights_uses_default_when_no_market_signal():
    weights = narrative_book_relevance_weights({"current_market_state_implications": []})

    assert weights["equity_beta_carry"] > 0.0
    assert weights["dollar_liquidity"] > 0.0
    assert weights["short_volatility"] > 0.0
    assert weights["commodity_inflation"] == 0.0


def test_support_priors_by_book_reads_nested_book_scores():
    train_bridge = {
        "evaluation": {
            "heldout_examples": [
                _candidate("q1", 1, [10, 20]),
                _candidate("q2", 1, [10, 30]),
                _candidate("q3", 2, [40, 50]),
                _candidate("q4", 2, [40, 60]),
            ]
        }
    }
    label_report = {
        "window_scores": [
            _score("q1", "commodity_inflation", 2.0),
            _score("q2", "commodity_inflation", 1.0),
            _score("q3", "commodity_inflation", 1.0),
            _score("q4", "commodity_inflation", 2.0),
        ]
    }

    priors = support_priors_by_book(
        train_candidate_bridge=train_bridge,
        train_label_report=label_report,
    )

    assert priors["commodity_inflation"][30] > priors["commodity_inflation"][20]
    assert priors["commodity_inflation"][50] > priors["commodity_inflation"][60]


def test_blend_book_priors_uses_book_weights():
    blended = blend_book_priors(
        {
            "commodity_inflation": {1: 1.0, 2: 0.0},
            "dollar_liquidity": {1: 0.0, 2: 1.0},
        },
        {"commodity_inflation": 0.75, "dollar_liquidity": 0.25},
    )

    assert blended[1] == 0.75
    assert blended[2] == 0.25


def _candidate(query_id: str, window_index: int, supports: list[int]) -> dict:
    return {
        "query_id": query_id,
        "window_index": window_index,
        "candidate_support_window_indices": supports,
        "top_train_pool": [{"window_index": idx} for idx in supports],
    }


def _score(query_id: str, book: str, composite: float) -> dict:
    return {
        "query_id": query_id,
        "methods": {
            "narrative_generator_topk": {
                "portfolio_response_books": [
                    {"book": book, "composite_path_score_z": composite}
                ]
            }
        },
    }
