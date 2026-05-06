import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_text_conditioning import (
    TextConditionAdapter,
    anchor_similarity_metrics,
    build_conditioning_text,
    build_contrast_texts,
    build_contrastive_examples,
    cosine_similarity,
    load_description_records,
    normalize_rows,
    project_embeddings_with_adapter,
    train_contrastive_projection,
)


def _description_record() -> dict:
    return {
        "window_id": "smoke_joint39_001",
        "panel_version": "joint39",
        "canonical_machine_text": (
            "SPX: DOWN LARGE, VIX: UP LARGE, US2Y: DOWN LARGE, "
            "BBB_OAS: WIDER LARGE"
        ),
        "revised_description": (
            "A pronounced risk-off window with equities down, volatility up, "
            "rates lower, and credit spreads wider."
        ),
        "descriptions": [
            {
                "style": "risk_manager",
                "text": (
                    "Equities were down sharply, volatility rose, and credit "
                    "spreads widened."
                ),
            }
        ],
        "contrastive": {
            "opposite": (
                "SPX: UP LARGE, VIX: DOWN LARGE, US2Y: UP LARGE, "
                "BBB_OAS: TIGHTER LARGE"
            ),
            "partial": ["SPX: DOWN LARGE, VIX: UP LARGE, US2Y: UP LARGE"],
            "magnitude": ["SPX: DOWN SMALL, VIX: UP SMALL, BBB_OAS: WIDER SMALL"],
        },
    }


def test_load_description_records_accepts_json_and_jsonl(tmp_path: Path) -> None:
    record = _description_record()
    json_path = tmp_path / "one.json"
    json_path.write_text(json.dumps(record), encoding="utf-8")
    assert load_description_records(json_path) == [record]

    jsonl_path = tmp_path / "many.jsonl"
    jsonl_path.write_text(
        json.dumps(record) + "\n\n" + json.dumps({"window_id": "w2"}) + "\n",
        encoding="utf-8",
    )
    assert load_description_records(jsonl_path) == [record, {"window_id": "w2"}]


def test_build_conditioning_text_preserves_canonical_direction_tokens() -> None:
    text = build_conditioning_text(_description_record())
    assert "PANEL: joint39" in text
    assert "CANONICAL: SPX: DOWN LARGE" in text
    assert "SUMMARY: A pronounced risk-off window" in text
    assert "free-form variant" not in text


def test_build_conditioning_text_can_include_free_form_variants() -> None:
    text = build_conditioning_text(_description_record(), include_free_form=True)
    assert "FREE_FORM risk_manager:" in text
    assert "Equities were down sharply" in text


def test_build_contrast_texts_flattens_opposite_partial_and_magnitude() -> None:
    contrasts = build_contrast_texts(_description_record())
    assert [item["kind"] for item in contrasts] == ["opposite", "partial", "magnitude"]
    assert contrasts[0]["text"].startswith("SPX: UP LARGE")


def test_build_contrastive_examples_has_anchor_positive_and_negative_roles() -> None:
    examples = build_contrastive_examples(_description_record())
    roles = [example["role"] for example in examples]
    assert roles == ["anchor", "positive", "negative", "negative", "negative"]
    assert examples[0]["text"].startswith("WINDOW_ID: smoke_joint39_001")
    assert examples[1]["text"].startswith("Equities were down sharply")


def test_normalize_rows_and_cosine_similarity() -> None:
    normalized = normalize_rows(np.array([[3.0, 4.0], [0.0, -2.0]]))
    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])
    assert cosine_similarity(normalized[0], normalized[0]) == 1.0
    assert np.isclose(cosine_similarity(normalized[0], -normalized[0]), -1.0)


def test_normalize_rows_rejects_zero_rows() -> None:
    try:
        normalize_rows(np.array([[0.0, 0.0]]))
    except ValueError as exc:
        assert "zero-norm" in str(exc)
    else:
        raise AssertionError("expected zero row to be rejected")


def test_text_condition_adapter_shape_and_determinism() -> None:
    torch.manual_seed(7)
    adapter_a = TextConditionAdapter(embedding_dim=6, condition_dim=4, hidden_dim=8)
    torch.manual_seed(7)
    adapter_b = TextConditionAdapter(embedding_dim=6, condition_dim=4, hidden_dim=8)
    embeddings = torch.randn(3, 6)

    out_a = adapter_a(embeddings)
    out_b = adapter_b(embeddings)

    assert out_a.shape == (3, 4)
    assert torch.allclose(out_a, out_b)


def test_project_embeddings_with_adapter_returns_numpy_condition_vectors() -> None:
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    condition = project_embeddings_with_adapter(
        embeddings,
        condition_dim=5,
        hidden_dim=7,
        seed=11,
    )
    assert condition.shape == (2, 5)
    assert np.isfinite(condition).all()


def test_anchor_similarity_metrics_separates_positive_and_negative_roles() -> None:
    embeddings = normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.2, 0.0, 0.9],
            ],
            dtype=np.float32,
        )
    )
    metrics = anchor_similarity_metrics(
        embeddings, ["anchor", "positive", "negative"]
    )
    assert metrics["positive_mean_cosine"] > metrics["negative_mean_cosine"]
    assert metrics["separation_mean"] > 0.0


def test_train_contrastive_projection_reduces_loss_on_toy_hard_negatives() -> None:
    embeddings = normalize_rows(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.92, 0.10, 0.0, 0.0],
                [0.88, -0.10, 0.0, 0.0],
                [0.80, 0.0, 0.60, 0.0],
                [0.78, 0.0, -0.62, 0.0],
            ],
            dtype=np.float32,
        )
    )
    result = train_contrastive_projection(
        embeddings,
        ["anchor", "positive", "positive", "negative", "negative"],
        metric_dim=3,
        hidden_dim=8,
        steps=120,
        lr=2e-2,
        margin=0.4,
        seed=3,
    )
    assert result["projected"].shape == (5, 3)
    assert result["loss_last"] < result["loss_first"]
    assert result["projected_metrics"]["separation_mean"] > result["raw_metrics"][
        "separation_mean"
    ]
