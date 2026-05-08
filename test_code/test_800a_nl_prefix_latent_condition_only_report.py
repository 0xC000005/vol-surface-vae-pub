import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_condition_only_report import (
    build_condition_report,
    compatible_grounding_from_condition_case,
    project_condition_query_text,
    run_condition_only_report,
    select_condition_case,
)


def _case_payload() -> dict:
    return {
        "case_name": "fragile_risk_on_rebound",
        "story": "Equities are recovering. The forward risk is ignored.",
        "candidate_query_text": (
            "CURRENT_SUPPORT_IMPLICATIONS:\n"
            "current: SPX up medium confidence=high horizon=current_state "
            "target=support_prior evidence=equities are recovering"
        ),
        "condition_only_validation": {"status": "pass", "future_target_count": 0},
        "condition_only_grounding": {
            "narrative_frame": "fragile risk-on rebound",
            "cleaned_conditioning_text": "Equities are recovering.",
            "current_market_state_implications": [
                {
                    "market": "SPX",
                    "direction": "up",
                    "magnitude": "medium",
                    "confidence": "high",
                    "evidence": ["equities are recovering"],
                    "inferred": False,
                    "horizon": "current_state",
                    "target_use": "support_prior",
                }
            ],
            "recent_regime_implications": [],
            "grounding_warnings": [{"code": "FWD", "message": "ignored"}],
            "unsupported_claims": [],
            "non_conditioning_forward_language": [
                {"phrase": "The forward risk is ignored."}
            ],
        },
        "metadata": {"model": "fixture-grounding-model"},
        "story_split": {
            "conditioning_sentences": ["Equities are recovering."],
            "non_conditioning_forward_sentences": [
                "The forward risk is ignored."
            ],
        },
    }


def test_select_condition_case_loads_summary_by_name(tmp_path) -> None:
    summary = {"cases": [_case_payload(), {"case_name": "other"}]}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    case = select_condition_case(
        case_json=None,
        summary_json=path,
        case_name="fragile_risk_on_rebound",
        case_index=0,
    )

    assert case["case_name"] == "fragile_risk_on_rebound"


def test_compatible_grounding_uses_condition_implications() -> None:
    grounding = compatible_grounding_from_condition_case(_case_payload())

    assert grounding["narrative_frame"] == "fragile risk-on rebound"
    assert len(grounding["market_implications"]) == 1
    assert grounding["market_implications"][0]["market"] == "SPX"
    assert grounding["condition_only_validation"]["status"] == "pass"


def test_project_condition_query_text_uses_embedder_and_adapter() -> None:
    calls = {}

    def fake_embedder(texts, *, model, dotenv_path, batch_size):
        calls["texts"] = texts
        calls["model"] = model
        calls["dotenv_path"] = dotenv_path
        calls["batch_size"] = batch_size
        return np.asarray([[3.0, 4.0]], dtype=np.float32)

    class FakeAdapter:
        def eval(self):
            calls["eval"] = True

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            calls["adapter_input_norm"] = float(torch.linalg.norm(value).item())
            return torch.asarray([[1.0, 2.0, 3.0]], dtype=torch.float32)

    def fake_loader(path, *, embedding_dim, condition_dim):
        calls["adapter_path"] = str(path)
        calls["embedding_dim"] = embedding_dim
        calls["condition_dim"] = condition_dim
        return FakeAdapter()

    result = project_condition_query_text(
        query_text="clean condition text",
        embedding_model="fake-embedding",
        bridge_adapter="adapter.pt",
        condition_dim=3,
        dotenv_path=".env.test",
        embedder=fake_embedder,
        adapter_loader=fake_loader,
    )

    assert result["query_condition"].shape == (3,)
    assert result["query_embedding"].shape == (2,)
    assert result["embedding_metadata"]["embedding_dim"] == 2
    assert result["embedding_metadata"]["condition_dim"] == 3
    assert calls["texts"] == ["clean condition text"]
    assert calls["eval"] is True
    assert abs(calls["adapter_input_norm"] - 1.0) < 1e-6


def test_build_condition_report_writes_report_and_arrays(tmp_path) -> None:
    projection = {
        "query_condition": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "query_embedding": np.asarray([0.5, 0.25], dtype=np.float32),
        "embedding_metadata": {
            "embedding_model": "fake",
            "embedding_dim": 2,
            "condition_dim": 3,
        },
    }

    report = build_condition_report(
        case=_case_payload(),
        projection=projection,
        output_dir=tmp_path,
        bridge_arrays="bridge_arrays.npz",
        bridge_adapter="adapter.pt",
    )

    arrays = np.load(report["artifact_paths"]["arrays"])
    assert report["cached_query"]["condition_source"] == "condition_only_openai_story"
    assert report["cached_query"]["grounding"]["market_implications"][0]["market"] == "SPX"
    assert arrays["text_memory"].shape == (1, 3)
    assert arrays["query_embedding"].shape == (1, 2)


def test_run_condition_only_report_uses_bridge_arrays_for_condition_dim(tmp_path, monkeypatch) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")
    bridge_arrays = tmp_path / "bridge_arrays.npz"
    np.savez_compressed(
        bridge_arrays,
        condition_vectors=np.zeros((2, 3), dtype=np.float32),
        memory_targets=np.zeros((2, 5), dtype=np.float32),
    )

    def fake_project(**kwargs):
        assert kwargs["condition_dim"] == 5
        return {
            "query_condition": np.ones(5, dtype=np.float32),
            "query_embedding": np.ones(2, dtype=np.float32),
            "embedding_metadata": {
                "embedding_model": kwargs["embedding_model"],
                "embedding_dim": 2,
                "condition_dim": 5,
            },
        }

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_condition_only_report."
        "project_condition_query_text",
        fake_project,
    )
    args = SimpleNamespace(
        case_json=str(case_path),
        summary_json=None,
        case_name=None,
        case_index=0,
        output_dir=str(tmp_path / "out"),
        bridge_arrays=str(bridge_arrays),
        bridge_adapter="adapter.pt",
        embedding_model="fake",
        dotenv=".env",
    )

    report = run_condition_only_report(args)

    assert report["cached_query"]["embedding_metadata"]["condition_dim"] == 5
    assert np.load(report["artifact_paths"]["arrays"])["text_memory"].shape == (1, 5)
