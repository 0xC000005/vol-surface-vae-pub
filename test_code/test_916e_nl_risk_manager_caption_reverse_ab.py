import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_caption_reverse_ab import (
    _codex_caption_paths,
    _matched_caption_window_ids,
    build_text_variants,
    compare_variant_groups,
    select_diverse_support_rows,
)
from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import RiskManagerCaptionV2


def _caption(window_id: str) -> RiskManagerCaptionV2:
    return RiskManagerCaptionV2(
        window_id=window_id,
        schema_version="risk_manager_caption_v2",
        scenario_title="Risk-off dollar squeeze",
        archetype="financial_accident",
        archetype_confidence="medium",
        mechanical_summary="Equities weakened, volatility rose, and the dollar firmed.",
        current_market_state="The current prefix has equity drawdown and higher volatility.",
        trigger="Funding pressure is inferred from cross-asset prices.",
        transmission="Risk reduction moves from equities into volatility and FX.",
        cross_asset_reaction="SPX fell, VIX rose, and USDJPY increased.",
        sequence="Equity weakness came first, followed by higher volatility.",
        portfolio_vulnerability="Long equity and short volatility exposures are vulnerable.",
        risk_manager_implication="Watch whether risk reduction broadens into credit.",
        evidence_used=["SPX down large", "VIX up medium", "USDJPY up medium"],
        ambiguity_flags=["The catalyst is inferred, not observed."],
        leakage_exclusions=["future path not used"],
        no_forecast_caveat="This describes the prefix and is not a forecast.",
        training_caption=(
            "A risk-off dollar-squeeze prefix with equities lower, volatility "
            "higher, and the dollar firmer against yen."
        ),
        contrastive_captions=[
            "Opposite setup: equities rally, volatility compresses, and the dollar softens."
        ],
        quality_self_critique=[],
    )


def test_build_text_variants_adds_fused_fact_token_caption_channels() -> None:
    bundle = {
        "narratives": [
            {
                "id": "revised_market_description",
                "text": "Equities are down and volatility is up.",
                "observed_fact_tokens": "SPX: DOWN LARGE; VIX: UP MEDIUM; USDJPY: UP MEDIUM",
            }
        ]
    }

    variants = build_text_variants(
        window_id="joint39_val_0001",
        bundle=bundle,
        api_caption=_caption("joint39_val_0001"),
        codex_caption=None,
        include_generic_demo=False,
    )

    by_id = {row["variant_id"]: row for row in variants}
    fused = by_id["api_v2_fused_fact_structured_caption"]

    assert fused["variant_group"] == "fused_api"
    assert fused["text"].startswith("FACT_TOKENS: SPX: DOWN LARGE")
    assert "TITLE: Risk-off dollar squeeze" in fused["text"]
    assert "TRAINING_CAPTION:" in fused["text"]


def test_select_diverse_support_rows_enforces_temporal_gap() -> None:
    memory = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.7, 0.7],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    rows = select_diverse_support_rows(
        np.asarray([1.0, 0.0], dtype=np.float32),
        memory,
        candidate_indices=[0, 1, 2, 3, 4],
        top_k=3,
        temporal_gap=2,
    )

    assert [row["window_index"] for row in rows] == [0, 3]
    assert rows[0]["rank"] == 1
    assert rows[0]["cosine"] > rows[1]["cosine"]


def test_compare_variant_groups_reports_rich_caption_delta() -> None:
    rows = [
        {
            "window_id": "w0",
            "variant_group": "simple",
            "target_cosine": 0.45,
            "true_rank_full_pool": 8,
            "top_support_hit": False,
        },
        {
            "window_id": "w0",
            "variant_group": "rich",
            "target_cosine": 0.70,
            "true_rank_full_pool": 3,
            "top_support_hit": True,
        },
        {
            "window_id": "w1",
            "variant_group": "simple",
            "target_cosine": 0.55,
            "true_rank_full_pool": 7,
            "top_support_hit": False,
        },
        {
            "window_id": "w1",
            "variant_group": "rich",
            "target_cosine": 0.65,
            "true_rank_full_pool": 5,
            "top_support_hit": False,
        },
    ]

    summary = compare_variant_groups(rows, baseline_group="simple")

    assert summary["groups"]["simple"]["count"] == 2
    assert summary["groups"]["rich"]["count"] == 2
    assert summary["deltas_vs_simple"]["rich"]["mean_target_cosine_delta"] == 0.175
    assert summary["deltas_vs_simple"]["rich"]["mean_true_rank_full_pool_delta"] == -3.5
    assert summary["deltas_vs_simple"]["rich"]["top_support_hit_rate_delta"] == 0.5


def test_codex_caption_paths_accepts_nested_batch_layout(tmp_path: Path) -> None:
    nested = tmp_path / "captions"
    nested.mkdir()
    (nested / "codex_gpt55_caption_joint39_val_0001.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "codex_gpt55_caption_failed_validator.json").write_text(
        "{}", encoding="utf-8"
    )

    paths = _codex_caption_paths(tmp_path)

    assert [path.name for path in paths] == ["codex_gpt55_caption_joint39_val_0001.json"]


def test_matched_caption_window_ids_uses_codex_only_windows() -> None:
    api = {"joint39_val_0001": _caption("joint39_val_0001")}
    codex = {"joint39_val_0002": _caption("joint39_val_0002")}
    bundles = {"joint39_val_0001": {}, "joint39_val_0002": {}, "missing": {}}
    window_index_by_id = {"joint39_val_0001": 1, "joint39_val_0002": 2}

    assert _matched_caption_window_ids(
        api_captions=api,
        codex_captions=codex,
        bundles=bundles,
        window_index_by_id=window_index_by_id,
    ) == ["joint39_val_0001", "joint39_val_0002"]
