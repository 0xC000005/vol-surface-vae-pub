import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (
    NarrativeAdapter,
    _append_label_cache,
    _enforce_production_label_policy,
    _load_selection_manifest_rows,
    _read_label_cache,
    apply_manifest_window_selection,
    bundle_validation_errors,
    build_narrative_training_examples,
    description_bundle_to_narrative_bundle,
    build_scenario_narrative_bundle,
    key_market_summary,
    refresh_cached_label_bundle,
    retrieval_diagnostics,
    retrieval_weights,
    sample_normal_generator_for_retrieved_analogues,
    train_narrative_adapter,
)
from experiments.backfill.block_ar.nl_scenario_descriptions import (
    ContrastiveDescriptions,
    FreeFormDescription,
    MarketMoveAudit,
    ScenarioDescriptionBundle,
)


def _spec_names() -> list[str]:
    return [f"iv:{idx:02d}" for idx in range(25)] + [
        "factor:spx",
        "factor:usdcad",
        "factor:usdjpy",
        "factor:dxy",
        "factor:copper",
        "factor:wheat",
        "factor:crude_oil",
        "factor:us2y",
        "factor:us10y",
        "factor:aaa_oas",
        "factor:bbb_oas",
        "factor:nikkei",
        "factor:gold",
        "factor:vix",
    ]


def _risk_off_history() -> np.ndarray:
    history = np.zeros((30, 39), dtype=np.float32)
    history[:, :25] = np.linspace(0.0, 0.25, 30)[:, None]
    history[:, 25] = np.linspace(0.0, -1.0, 30)  # SPX
    history[:, 32] = np.linspace(0.0, -0.4, 30)  # US2Y
    history[:, 33] = np.linspace(0.0, -0.3, 30)  # US10Y
    history[:, 35] = np.linspace(0.0, 0.8, 30)  # BBB OAS
    history[:, 37] = np.linspace(0.0, 0.2, 30)  # gold
    history[:, 38] = np.linspace(0.0, 1.2, 30)  # VIX
    return history


def test_key_market_summary_detects_risk_off_signs() -> None:
    summary = key_market_summary(_risk_off_history(), _spec_names())
    by_market = {item["market"]: item for item in summary}
    assert by_market["SPX"]["direction"] == "down"
    assert by_market["VIX"]["direction"] == "up"
    assert by_market["BBB_OAS"]["direction"] == "wider"
    assert by_market["US2Y"]["direction"] == "down"
    assert by_market["IV_SURFACE"]["direction"] == "up"


def test_build_scenario_narrative_bundle_is_hallucination_aware() -> None:
    bundle = build_scenario_narrative_bundle(
        "w0",
        _risk_off_history(),
        _spec_names(),
    )
    assert bundle["grounding_policy"] == "market_facts_first_no_external_news"
    assert bundle["observed_market_facts"]
    assert len(bundle["narratives"]) >= 3
    assert len(bundle["contrastive_narratives"]) >= 2
    assert any(
        narrative["unsupported_claims"] for narrative in bundle["narratives"]
    )
    assert "COVID" in bundle["narratives"][0]["text"]
    assert bundle["narratives"][0]["grounding_status"] == "analogy_not_observed_fact"


def test_build_narrative_training_examples_marks_roles_and_targets() -> None:
    bundle = build_scenario_narrative_bundle("w0", _risk_off_history(), _spec_names())
    examples = build_narrative_training_examples(bundle)
    roles = [example["role"] for example in examples]
    assert "anchor" in roles
    assert "positive" in roles
    assert "negative" in roles
    assert all(example["target_index"] == 0 for example in examples if example["role"] != "negative")
    assert all(example["target_index"] is None for example in examples if example["role"] == "negative")


def test_description_bundle_to_narrative_bundle_uses_openai_text() -> None:
    description = ScenarioDescriptionBundle(
        window_id="w_openai",
        panel_version="joint39",
        canonical_machine_text="SPX: DOWN LARGE; VIX: UP LARGE; BBB_OAS: WIDER MEDIUM",
        descriptions=[
            FreeFormDescription(
                style="risk_manager",
                text="A model-written risk-manager narrative with equities lower and vol higher.",
            )
        ],
        structured_audit=[
            MarketMoveAudit(
                market="SPX",
                direction="down",
                magnitude="large",
                confidence="high",
                evidence=["factor:spx_30d_change=-1.2"],
                inferred=False,
            )
        ],
        contrastive=ContrastiveDescriptions(
            opposite="SPX: UP LARGE; VIX: DOWN LARGE",
            partial=["SPX: DOWN LARGE; VIX: DOWN SMALL"],
            magnitude=["SPX: DOWN SMALL; VIX: UP SMALL"],
        ),
        critique=["No unsupported news claims."],
        revised_description="OpenAI revised description for a grounded market selloff.",
    )
    bundle = description_bundle_to_narrative_bundle(description)
    assert bundle["grounding_policy"] == "openai_market_facts_first_no_external_news"
    assert bundle["narratives"][0]["text"] == description.revised_description
    assert bundle["narratives"][1]["text"] == description.descriptions[0].text
    assert {item["kind"] for item in bundle["contrastive_narratives"]} == {
        "opposite",
        "partial",
        "magnitude",
    }
    examples = build_narrative_training_examples(bundle)
    assert any("OpenAI revised description" in example["text"] for example in examples)


def test_label_cache_roundtrip(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    bundle = {
        "window_id": "w_cache",
        "narratives": [{"id": "a", "text": "cached", "grounding_status": "ok"}],
    }
    _append_label_cache(path, bundle)
    assert _read_label_cache(path)["w_cache"] == bundle


def test_bundle_validation_errors_returns_only_errors() -> None:
    bundle = {
        "hallucination_audit": {
            "validation_issues": [
                {"code": "warning_only", "severity": "warning"},
                {"code": "bad_news", "severity": "error"},
            ]
        }
    }
    assert bundle_validation_errors(bundle) == [{"code": "bad_news", "severity": "error"}]


def test_refresh_cached_label_bundle_recomputes_validation() -> None:
    description = ScenarioDescriptionBundle(
        window_id="w_cached",
        panel_version="joint39",
        canonical_machine_text="SPX: DOWN LARGE; VIX: UP LARGE",
        descriptions=[
            FreeFormDescription(
                style="risk_manager",
                text="Risk-off market state with no confirmed external trigger.",
            )
        ],
        structured_audit=[],
        contrastive=ContrastiveDescriptions(
            opposite="SPX: UP LARGE; VIX: DOWN LARGE",
            partial=["SPX: DOWN LARGE; VIX: DOWN SMALL"],
            magnitude=["SPX: DOWN SMALL; VIX: UP SMALL"],
        ),
        revised_description="Grounded market selloff without a confirmed catalyst.",
    )
    stale = {
        "window_id": "w_cached",
        "hallucination_audit": {
            "validation_issues": [
                {"code": "unsupported_catalyst", "severity": "error"}
            ]
        },
        "source_description_bundle": description.model_dump(),
    }
    refreshed = refresh_cached_label_bundle(stale)
    issue_codes = {
        (issue["code"], issue["severity"])
        for issue in refreshed["hallucination_audit"]["validation_issues"]
    }
    assert ("unsupported_catalyst", "error") not in issue_codes


def test_production_label_policy_requires_openai_unless_local_test() -> None:
    with pytest.raises(ValueError, match="Production narrative runs require --label-backend openai"):
        _enforce_production_label_policy(
            SimpleNamespace(label_backend="rule", local_test=False)
        )

    _enforce_production_label_policy(
        SimpleNamespace(label_backend="openai", local_test=False)
    )
    _enforce_production_label_policy(
        SimpleNamespace(label_backend="rule", local_test=True)
    )


def test_load_selection_manifest_rows_preserves_split_order(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "splits": {
    "train": [{"window_index": 2, "window_id": "w2", "source_index": 102}],
    "validation": [{"window_index": 5, "window_id": "w5", "source_index": 105}],
    "test": [{"window_index": 9, "window_id": "w9", "source_index": 109}]
  }
}
""".strip(),
        encoding="utf-8",
    )

    rows = _load_selection_manifest_rows(manifest_path, "all")

    assert [row["window_id"] for row in rows] == ["w2", "w5", "w9"]
    assert [row["manifest_split"] for row in rows] == ["train", "validation", "test"]


def test_apply_manifest_window_selection_preserves_original_indices() -> None:
    base = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    history_raw = np.arange(6 * 3 * 2, dtype=np.float32).reshape(6, 3, 2)
    window_metadata = [
        {"source_index": 100 + idx, "calendar_end_date": f"2020-01-{idx + 1:02d}"}
        for idx in range(6)
    ]
    manifest_rows = [
        {
            "window_index": 4,
            "window_id": "joint39_val_0004",
            "source_index": 104,
            "selection_reasons": ["eventful"],
            "manifest_split": "train",
        },
        {
            "window_index": 1,
            "window_id": "joint39_val_0001",
            "source_index": 101,
            "selection_reasons": ["test"],
            "manifest_split": "test",
        },
    ]

    selected = apply_manifest_window_selection(
        history_level=base,
        history_norm=base + 10,
        center=base + 20,
        scale=base + 30,
        drift_feature=base + 40,
        history_raw=history_raw,
        window_metadata=window_metadata,
        manifest_rows=manifest_rows,
    )

    np.testing.assert_allclose(selected["history_level"], base[[4, 1]])
    assert selected["source_indices"].tolist() == [104, 101]
    assert selected["window_indices"].tolist() == [4, 1]
    assert selected["window_metadata"][0]["window_id"] == "joint39_val_0004"
    assert selected["window_metadata"][0]["selection_reasons"] == ["eventful"]
    assert selected["window_metadata"][1]["manifest_split"] == "test"


def test_train_narrative_adapter_reduces_alignment_loss() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(6, 8)).astype(np.float32)
    text_embeddings = base / np.linalg.norm(base, axis=1, keepdims=True)
    targets = rng.normal(size=(2, 4)).astype(np.float32)
    target_indices = np.array([0, 0, -1, 1, 1, -1], dtype=np.int64)
    roles = ["anchor", "positive", "negative", "anchor", "positive", "negative"]
    groups = ["a", "a", "a", "b", "b", "b"]
    result = train_narrative_adapter(
        text_embeddings,
        targets,
        target_indices,
        roles,
        groups,
        condition_dim=4,
        hidden_dim=12,
        steps=160,
        lr=2e-2,
        contrastive_weight=0.5,
        seed=9,
    )
    assert isinstance(result["adapter"], NarrativeAdapter)
    assert result["condition_vectors"].shape == (6, 4)
    assert result["loss_last"] < result["loss_first"]
    assert np.isfinite(result["condition_vectors"]).all()


def test_retrieval_weights_are_softmax_normalized() -> None:
    rows = [
        {"index": 4, "cosine": 0.9},
        {"index": 2, "cosine": 0.8},
        {"index": 7, "cosine": 0.1},
    ]
    weights = retrieval_weights(rows, temperature=0.05)
    assert [row["index"] for row in weights] == [4, 2, 7]
    assert np.isclose(sum(row["weight"] for row in weights), 1.0)
    assert weights[0]["weight"] > weights[1]["weight"] > weights[2]["weight"]


def test_retrieval_diagnostics_flags_ood_low_similarity() -> None:
    warning = retrieval_diagnostics(
        [{"index": 0, "cosine": 0.61}, {"index": 1, "cosine": 0.60}],
        ood_threshold=0.75,
    )
    assert warning["ood_warning"] is True
    assert warning["top_cosine"] == 0.61
    assert np.isclose(warning["top_gap"], 0.01)


def test_sample_normal_generator_for_retrieved_analogues_uses_sample_batched() -> None:
    class DummyModel:
        def sample_batched(
            self,
            history_level_values,
            history_normalized_innovation,
            center,
            scale,
            *,
            drift_feature,
            n_samples,
            n_steps,
            chunk_size,
            temperature,
        ):
            assert history_level_values.shape[0] == 2
            assert history_normalized_innovation.shape[0] == 2
            assert center.shape[0] == 2
            assert scale.shape[0] == 2
            assert drift_feature.shape[0] == 2
            return history_level_values.new_zeros((2, n_samples, n_steps, 3))

    result = sample_normal_generator_for_retrieved_analogues(
        DummyModel(),
        [{"index": 3, "cosine": 0.9}, {"index": 1, "cosine": 0.8}],
        np.zeros((5, 4, 3), dtype=np.float32),
        np.zeros((5, 4, 3), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
        np.ones((5, 3), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
        n_samples=4,
        n_steps=6,
        chunk_size=2,
        temperature=1.0,
        device="cpu",
    )
    assert result["indices"] == [3, 1]
    assert result["increments"].shape == (2, 4, 6, 3)
