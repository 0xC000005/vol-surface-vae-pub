import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_risk_manager_caption_v2 import (
    REQUIRED_SPECIALIST_DOCS,
    RiskManagerCaptionV2,
    build_caption_messages,
    load_specialist_standards,
    run_testflight,
    select_testflight_bundles,
    validate_caption_v2,
)


def _bundle(window_id: str, *, split: str, text: str) -> dict:
    return {
        "window_id": window_id,
        "manifest_split": split,
        "source_index": 1000 + int(window_id.rsplit("_", 1)[-1]),
        "window_index": int(window_id.rsplit("_", 1)[-1]),
        "calendar": {
            "calendar_start_date": "2016-01-01",
            "calendar_end_date": "2016-02-15",
            "forecast_start_date": "2016-02-16",
            "forecast_end_date": "2016-03-30",
        },
        "market_implications": [
            {
                "market": "SPX",
                "direction": "down",
                "magnitude": "large",
                "horizon": "30d conditioning window",
                "confidence": "high",
                "evidence": ["SPX fell materially over the prefix."],
            },
            {
                "market": "VIX",
                "direction": "up",
                "magnitude": "medium",
                "horizon": "30d conditioning window",
                "confidence": "high",
                "evidence": ["VIX rose over the prefix."],
            },
        ],
        "narratives": [
            {
                "id": "revised_market_description",
                "text": text,
                "grounding_status": "market_fact_supported",
                "observed_fact_tokens": "SPX: DOWN LARGE; VIX: UP MEDIUM",
            }
        ],
        "source_description_bundle": {
            "canonical_machine_text": "SPX: DOWN LARGE; VIX: UP MEDIUM",
            "revised_description": text,
        },
    }


def test_specialist_standards_loads_both_word_documents() -> None:
    standards = load_specialist_standards()

    assert len(REQUIRED_SPECIALIST_DOCS) == 2
    assert {item.path.name for item in standards.documents} == {
        "quant generated scenarios story narrative.docx",
        "quant generated scenarios story narrative 2.docx",
    }
    assert "trigger" in standards.rubric_text.lower()
    assert "transmission" in standards.rubric_text.lower()
    assert "portfolio" in standards.rubric_text.lower()


def test_build_caption_messages_includes_specialist_sources_and_no_forecast_guard() -> None:
    standards = load_specialist_standards()
    bundle = _bundle("joint39_val_0001", split="train", text="A broad risk-off tape.")

    messages = build_caption_messages(bundle, standards=standards)
    combined = "\n".join(item["content"] for item in messages)

    assert "quant generated scenarios story narrative.docx" in combined
    assert "quant generated scenarios story narrative 2.docx" in combined
    assert "Do not use realized future outcomes" in combined
    assert "current/recent 30-day prefix" in combined
    assert "Scenario title" in combined


def test_caption_validation_rejects_future_target_leakage() -> None:
    caption = RiskManagerCaptionV2(
        window_id="joint39_val_0001",
        schema_version="risk_manager_caption_v2",
        scenario_title="Demand recession setup",
        archetype="demand_recession",
        archetype_confidence="medium",
        mechanical_summary="SPX was down and VIX was up over the current prefix.",
        current_market_state="The current/recent prefix is risk-off.",
        trigger="Demand concern is the interpretation, not a confirmed event.",
        transmission="Lower risk appetite moved through equities and volatility.",
        cross_asset_reaction="Equities weakened while volatility rose.",
        sequence="Risk appetite weakened first, then volatility rose.",
        portfolio_vulnerability="Long equity beta would be vulnerable.",
        risk_manager_implication="Monitor whether the stress broadens into credit.",
        evidence_used=["SPX down large", "VIX up medium"],
        ambiguity_flags=[],
        leakage_exclusions=["future path not used"],
        no_forecast_caveat="This describes the prefix and is not a forecast.",
        training_caption=(
            "The next 30 days will see SPX fall another 8 percent and VIX spike."
        ),
        contrastive_captions=[
            "Opposite setup: equities are rallying and volatility is falling."
        ],
        quality_self_critique=["Contains future target language."],
    )

    issues = validate_caption_v2(caption)

    assert any(issue.code == "future_target_leakage" for issue in issues)


def test_caption_validation_requires_risk_manager_story_sections() -> None:
    caption = RiskManagerCaptionV2(
        window_id="joint39_val_0001",
        schema_version="risk_manager_caption_v2",
        scenario_title="Risk-off setup",
        archetype="financial_accident",
        archetype_confidence="medium",
        mechanical_summary="SPX down, VIX up.",
        current_market_state="Equities weakened while volatility rose.",
        trigger="",
        transmission="",
        cross_asset_reaction="Equities down, volatility up.",
        sequence="",
        portfolio_vulnerability="",
        risk_manager_implication="",
        evidence_used=["SPX down large"],
        ambiguity_flags=[],
        leakage_exclusions=[],
        no_forecast_caveat="Not a forecast.",
        training_caption="A risk-off prefix with equities lower and volatility higher.",
        contrastive_captions=[],
        quality_self_critique=[],
    )

    issues = validate_caption_v2(caption)
    codes = {issue.code for issue in issues}

    assert "missing_trigger" in codes
    assert "missing_transmission" in codes
    assert "missing_sequence" in codes
    assert "missing_portfolio_vulnerability" in codes
    assert "missing_contrastive_caption" in codes


def test_select_testflight_bundles_is_split_balanced_and_deterministic(tmp_path: Path) -> None:
    bundles = [
        _bundle(f"joint39_val_{idx:04d}", split=split, text=f"caption {idx}")
        for idx, split in enumerate(
            ["train", "train", "train", "validation", "validation", "test", "test", "test", "train", "test", "validation", "train"],
            start=1,
        )
    ]
    report = {"narrative_bundles": bundles}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    selected = select_testflight_bundles(report_path, count=6)

    assert [item["window_id"] for item in selected] == [
        "joint39_val_0001",
        "joint39_val_0004",
        "joint39_val_0006",
        "joint39_val_0002",
        "joint39_val_0005",
        "joint39_val_0007",
    ]
    assert {item["manifest_split"] for item in selected} == {"train", "validation", "test"}


def test_openai_testflight_continue_on_error_writes_partial_report(
    tmp_path: Path, monkeypatch
) -> None:
    bundles = [
        _bundle("joint39_val_0001", split="train", text="A broad risk-off tape."),
        _bundle("joint39_val_0002", split="validation", text="A calmer risk-on tape."),
    ]
    report_path = tmp_path / "pipeline_report.json"
    report_path.write_text(json.dumps({"narrative_bundles": bundles}), encoding="utf-8")

    def fake_caption_one(bundle, **kwargs):
        if bundle["window_id"] == "joint39_val_0002":
            raise RuntimeError("quota exhausted")
        return (
            RiskManagerCaptionV2(
                window_id=bundle["window_id"],
                schema_version="risk_manager_caption_v2",
                scenario_title="Risk-off setup",
                archetype="financial_accident",
                archetype_confidence="medium",
                mechanical_summary="SPX was down and VIX was up over the prefix.",
                current_market_state="The current/recent prefix is risk-off.",
                trigger="A risk appetite break is inferred from market prices.",
                transmission="Equity weakness transmitted into volatility and credit.",
                cross_asset_reaction="Equities weakened while volatility rose.",
                sequence="Risk appetite weakened first, then volatility rose.",
                portfolio_vulnerability="Long equity beta would be vulnerable.",
                risk_manager_implication="Monitor whether the stress broadens into credit.",
                evidence_used=["SPX down large", "VIX up medium"],
                ambiguity_flags=[],
                leakage_exclusions=["future path not used"],
                no_forecast_caveat="This describes the prefix and is not a forecast.",
                training_caption=(
                    "A risk-off current-condition prefix with equities materially "
                    "lower, volatility higher, and portfolio risk concentrated in "
                    "long equity beta and short-volatility exposures."
                ),
                contrastive_captions=[
                    "Opposite setup: equities are rallying and volatility is falling."
                ],
                quality_self_critique=[],
            ),
            {
                "model": "fake-premium",
                "prompt_version": "test",
                "response_id": "resp_test",
                "usage": {"total_tokens": 123},
            },
        )

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_risk_manager_caption_v2.caption_one_with_openai",
        fake_caption_one,
    )
    args = Namespace(
        pipeline_report=report_path,
        output_dir=tmp_path / "out",
        count=2,
        backend="openai",
        model="fake-premium",
        dotenv=".env",
        max_output_tokens=100,
        continue_on_error=True,
    )

    report = run_testflight(args)
    saved = json.loads(Path(report["artifact_paths"]["report"]).read_text(encoding="utf-8"))

    assert report["status"] == "partial_fail"
    assert report["requested_count"] == 2
    assert report["caption_count"] == 1
    assert report["api_error_count"] == 1
    assert saved["api_errors"][0]["window_id"] == "joint39_val_0002"
