import sys
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_gradio_cached_smoke import (
    _table_rows,
    run_gradio_cached_smoke,
)


def test_table_rows_handles_dataframes_and_lists() -> None:
    assert _table_rows(pd.DataFrame([{"a": 1}, {"a": 2}])) == 2
    assert _table_rows([1, 2, 3]) == 3
    assert _table_rows(object()) == 0


def test_run_gradio_cached_smoke_validates_wrapper_outputs(
    tmp_path, monkeypatch
) -> None:
    def fake_run_prefix_latent_for_app(**kwargs):
        yield (
            "in progress",
            "## Prefix-Latent Run Status\n\n- Prefix-latent run started: `0.0s ago`",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            go.Figure(),
            "{}",
            {},
            None,
        )
        yield (
            "markdown",
            "## Prefix-Latent Run Status\n\n- Selected-start: `pass`",
            pd.DataFrame([{"Variant": "balanced_memory_start"}]),
            pd.DataFrame([{"Variant": "original"}]),
            pd.DataFrame([{"Status": "pass"}]),
            pd.DataFrame([{"Market": "SPX"}]),
            go.Figure(data=[go.Scatter(y=[1, 2, 3])]),
            '{"status": "ok"}',
            {
                "status": "ok",
                "validation_gate": {
                    "selected_start_status": "pass",
                    "diagnostic_baseline_status": "warning",
                    "overall_status": "warning",
                },
                "cached_query": {"condition_source": "cached_bridge_query"},
                "generation": {
                    "path_quantiles": [
                        {
                            "market": "SPX",
                            "analogue_label": "Diagnostic baseline: joint39_val_0370",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                        {
                            "market": "SPX",
                            "analogue_label": "Selected start: joint39_val_0063",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                    ]
                },
            },
            {"choices": [("All", "ALL")]},
        )

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_gradio_cached_smoke."
        "run_prefix_latent_for_app",
        fake_run_prefix_latent_for_app,
    )
    summary = run_gradio_cached_smoke(
        SimpleNamespace(
            output_dir=str(tmp_path),
            start_mode="balanced_memory_start",
            samples=2,
            fan_market="SPX",
            story="A cached story.",
            live_story=False,
        )
    )

    assert summary["status"] == "ok"
    assert summary["selected_table_rows"] == 1
    assert summary["diagnostic_table_rows"] == 1
    assert summary["selected_start_status"] == "pass"
    assert summary["has_selected_start_label"] is True
    assert summary["has_diagnostic_baseline_label"] is True
    assert (tmp_path / "gradio_cached_smoke_summary.json").exists()


def test_run_gradio_live_smoke_requires_live_condition_source(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_run_prefix_latent_for_app(**kwargs):
        captured.update(kwargs)
        yield (
            "in progress",
            "## Prefix-Latent Run Status\n\n- Prefix-latent run started: `0.0s ago`",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            go.Figure(),
            "{}",
            {},
            None,
        )
        yield (
            "markdown",
            "## Prefix-Latent Run Status\n\n- Selected-start: `pass`",
            pd.DataFrame([{"Variant": "balanced_memory_start"}]),
            pd.DataFrame([{"Variant": "original"}]),
            pd.DataFrame([{"Status": "pass"}]),
            pd.DataFrame([{"Market": "SPX"}]),
            go.Figure(data=[go.Scatter(y=[1, 2, 3])]),
            '{"status": "ok"}',
            {
                "status": "ok",
                "cached_query": {"condition_source": "live_openai_story"},
                "validation_gate": {
                    "selected_start_status": "pass",
                    "diagnostic_baseline_status": "warning",
                    "overall_status": "warning",
                },
                "generation": {
                    "path_quantiles": [
                        {
                            "market": "SPX",
                            "analogue_label": "Diagnostic baseline: joint39_val_0370",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                        {
                            "market": "SPX",
                            "analogue_label": "Selected start: joint39_val_0063",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                    ]
                },
            },
            {"choices": [("All", "ALL")]},
        )

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_gradio_cached_smoke."
        "run_prefix_latent_for_app",
        fake_run_prefix_latent_for_app,
    )
    summary = run_gradio_cached_smoke(
        SimpleNamespace(
            output_dir=str(tmp_path),
            start_mode="balanced_memory_start",
            samples=2,
            fan_market="SPX",
            story="A live story.",
            live_story=True,
        )
    )

    assert captured["live_story"] is True
    assert summary["status"] == "ok"
    assert summary["mode"] == "live_story"
    assert summary["condition_source"] == "live_openai_story"
    assert (tmp_path / "gradio_live_smoke_summary.json").exists()


def test_run_gradio_cached_casebook_smoke_uses_cached_report(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_run_prefix_latent_for_app(**kwargs):
        captured.update(kwargs)
        yield (
            "in progress",
            "## Prefix-Latent Run Status\n\n- Prefix-latent run started: `0.0s ago`",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            go.Figure(),
            "{}",
            {},
            None,
        )
        yield (
            "markdown",
            "## Prefix-Latent Run Status\n\n- Selected-start: `pass`",
            pd.DataFrame([{"Variant": "explicit_start_window"}]),
            pd.DataFrame([{"Variant": "original"}]),
            pd.DataFrame([{"Status": "pass"}]),
            pd.DataFrame([{"Market": "SPX"}]),
            go.Figure(data=[go.Scatter(y=[1, 2, 3])]),
            '{"status": "ok"}',
            {
                "status": "ok",
                "cached_query": {"condition_source": "external_condition_report"},
                "validation_gate": {
                    "selected_start_status": "pass",
                    "diagnostic_baseline_status": "pass",
                    "overall_status": "pass",
                },
                "generation": {
                    "path_quantiles": [
                        {
                            "market": "SPX",
                            "analogue_label": "Diagnostic baseline: joint39_val_0370",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                        {
                            "market": "SPX",
                            "analogue_label": "Selected start: joint39_val_0063",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                        {
                            "market": "IV_ATM_3M",
                            "analogue_label": "All start variants",
                            "days": [1, 2],
                            "p10": [0.0, 0.0],
                            "p50": [0.1, 0.2],
                            "p90": [0.3, 0.4],
                        },
                    ]
                },
            },
            {"choices": [("All", "ALL")]},
        )

    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_gradio_cached_smoke."
        "run_prefix_latent_for_app",
        fake_run_prefix_latent_for_app,
    )
    summary = run_gradio_cached_smoke(
        SimpleNamespace(
            output_dir=str(tmp_path),
            start_mode="balanced_memory_start",
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
            story="A typed story that should be replaced.",
            live_story=True,
            cached_casebook_choice="safe_haven_gold_bid:18",
        )
    )

    assert captured["live_story"] is False
    assert captured["condition_only_story"] is False
    assert captured["use_explicit_start"] is True
    assert captured["explicit_start_window_index"] == 18
    assert (
        "condition_only_report_823b_safe_haven" in captured["cached_condition_report"]
    )
    assert "safe-haven" in captured["story"].lower()
    assert summary["mode"] == "cached_casebook"
    assert summary["condition_source"] == "external_condition_report"
    assert summary["redraw_fan_trace_count"] > 0
    assert (tmp_path / "gradio_cached_casebook_smoke_summary.json").exists()
