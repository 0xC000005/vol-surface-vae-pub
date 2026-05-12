import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_embedding_model_hardcase_ablation import (
    representation_text,
)


def test_representation_text_extracts_factor_tokens_from_full_example():
    text = (
        "NARRATIVE: risk-on story\n"
        "GROUNDING_STATUS: market_fact_supported\n"
        "MARKET_IMPLICATIONS: SPX: UP LARGE; VIX: DOWN LARGE\n"
        "NARRATIVE_CATALYSTS: catalyst"
    )

    assert (
        representation_text(text, representation="factor_tokens")
        == "SPX: UP LARGE; VIX: DOWN LARGE"
    )


def test_representation_text_keeps_negative_text_as_factor_tokens():
    text = "SPX: DOWN LARGE; VIX: UP LARGE"

    assert representation_text(text, representation="factor_tokens") == text
