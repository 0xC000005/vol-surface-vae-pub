import sys
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_gradio_live_casebook import (
    _bounded_live_runner,
    market_implication_alignment,
    run_gradio_live_casebook,
    summarize_gradio_case,
)


def _fake_report() -> dict:
    return {
        "status": "ok",
        "cached_query": {
            "condition_source": "live_openai_story",
            "grounding": {
                "narrative_frame": "fragile risk-on rebound",
                "grounding_warnings": [{"code": "interpretive_phrase"}],
                "market_implications": [
                    {"market": "SPX", "direction": "up"},
                    {"market": "VIX", "direction": "down"},
                ],
            },
        },
        "validation_gate": {
            "overall_status": "warning",
            "operational_status": "pass",
            "selected_start_status": "pass",
            "diagnostic_baseline_status": "warning",
            "stress_status": "pass",
            "warning_counts": {"low_memory_compatibility": 1},
            "fail_counts": {},
        },
        "generation": {
            "generated_state_shape": [2, 2, 30, 39],
            "path_quantiles": [
                {"analogue_label": "All start variants"},
                {"analogue_label": "Diagnostic baseline: joint39_val_0370"},
                {"analogue_label": "Selected start: joint39_val_0115"},
            ],
        },
    }


def _fake_final_outputs(report: dict | None = None) -> tuple:
    return (
        "markdown",
        "## Prefix-Latent Run Status\n\n- Selected-start: `pass`",
        pd.DataFrame([{"Variant": "balanced_memory_start"}]),
        pd.DataFrame([{"Variant": "original"}]),
        pd.DataFrame([{"Status": "pass"}]),
        pd.DataFrame([{"Market": "SPX", "Mean Terminal Delta": -1.0}]),
        go.Figure(data=[go.Scatter(y=[1, 2, 3])]),
        '{"status": "ok"}',
        report or _fake_report(),
        {"choices": [("All", "ALL")]},
    )


def _fake_first_outputs() -> tuple:
    return (
        "in progress",
        "## Prefix-Latent Run Status\n\n- Prefix-latent run started: `0.0s ago`",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        go.Figure(),
        "{}",
        {},
        None,
    )


def test_bounded_live_runner_forces_live_story_and_bounded_controls(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_run_prefix_latent_story_smoke(args):
        captured.update(vars(args))
        return {"status": "ok"}

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_gradio_live_casebook."
        "run_prefix_latent_story_smoke",
        fake_run_prefix_latent_story_smoke,
    )
    runner = _bounded_live_runner(
        output_dir=tmp_path / "prefix_run",
        steps=100,
        samples=2,
        chunk_size=2,
    )
    result = runner(
        SimpleNamespace(
            live_story=False, output_dir="old", steps=1000, samples=8, chunk_size=8
        )
    )

    assert result["status"] == "ok"
    assert captured["live_story"] is True
    assert captured["output_dir"] == str(tmp_path / "prefix_run")
    assert captured["steps"] == 100
    assert captured["samples"] == 2
    assert captured["chunk_size"] == 2


def test_summarize_gradio_case_checks_product_contract(tmp_path) -> None:
    summary = summarize_gradio_case(
        case_name="fragile_risk_on_rebound",
        story="A story.",
        case_dir=tmp_path,
        first_outputs=_fake_first_outputs(),
        final_outputs=_fake_final_outputs(),
    )

    assert summary["status"] == "ok"
    assert summary["condition_source"] == "live_openai_story"
    assert summary["selected_start_status"] == "pass"
    assert summary["diagnostic_baseline_status"] == "warning"
    assert summary["grounding_warning_count"] == 1
    assert summary["market_implication_count"] == 2
    assert summary["market_alignment"]["mismatch_count"] == 1
    assert summary["fan_trace_count"] == 1
    assert any("Selected start:" in label for label in summary["path_labels"])
    assert summary["scenario_rows"][0]["Market"] == "SPX"


def test_market_implication_alignment_flags_direction_mismatch() -> None:
    alignment = market_implication_alignment(
        grounding={
            "market_implications": [
                {"market": "SPX", "direction": "up", "confidence": "high"},
                {"market": "BBB_OAS", "direction": "wider", "confidence": "high"},
                {"market": "IV_SURFACE", "direction": "down", "confidence": "medium"},
            ]
        },
        scenario_rows=[
            {"Market": "SPX", "Mean Terminal Delta": "-10.0"},
            {"Market": "BBB_OAS", "Mean Terminal Delta": "0.5"},
            {"Market": "IV_SURFACE", "Mean Terminal Delta": "-0.01"},
        ],
    )

    assert alignment["status"] == "warning"
    assert alignment["checked_count"] == 3
    assert alignment["mismatch_count"] == 1
    assert alignment["mismatches"][0]["market"] == "SPX"


def test_run_gradio_live_casebook_aggregates_cases(tmp_path) -> None:
    captured = []

    def fake_app_runner(**kwargs):
        captured.append(kwargs)
        yield _fake_first_outputs()
        yield _fake_final_outputs()

    summary = run_gradio_live_casebook(
        SimpleNamespace(
            output_dir=str(tmp_path),
            case_count=2,
            steps=100,
            samples=2,
            chunk_size=2,
            start_mode="balanced_memory_start",
            fan_market="SPX",
        ),
        app_runner=fake_app_runner,
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 2
    assert summary["selected_start_status_counts"] == {"pass": 2}
    assert summary["market_alignment_status_counts"] == {"warning": 2}
    assert all(item["live_story"] is True for item in captured)
    assert all(item["start_mode"] == "balanced_memory_start" for item in captured)
    assert (tmp_path / "gradio_live_casebook_summary.json").exists()
