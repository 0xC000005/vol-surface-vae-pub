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


def test_run_gradio_cached_smoke_validates_wrapper_outputs(tmp_path, monkeypatch) -> None:
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
                "generation": {
                    "path_quantiles": [
                        {"analogue_label": "Diagnostic baseline: joint39_val_0370"},
                        {"analogue_label": "Selected start: joint39_val_0063"},
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
        )
    )

    assert summary["status"] == "ok"
    assert summary["selected_table_rows"] == 1
    assert summary["diagnostic_table_rows"] == 1
    assert summary["selected_start_status"] == "pass"
    assert summary["has_selected_start_label"] is True
    assert summary["has_diagnostic_baseline_label"] is True
    assert (tmp_path / "gradio_cached_smoke_summary.json").exists()
