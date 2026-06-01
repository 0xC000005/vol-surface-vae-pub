import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar import nl_prefix_latent_gradio_api_smoke as smoke


def _plot(traces: int = 1) -> dict:
    return {
        "type": "plotly",
        "plot": json.dumps({"data": [{"y": [1, 2]} for _ in range(traces)]}),
    }


def _frame(rows: int = 1) -> dict:
    return {"headers": ["a"], "data": [[i] for i in range(rows)], "metadata": None}


def test_plot_trace_count_handles_plotly_payload() -> None:
    assert smoke._plot_trace_count(_plot(3)) == 3
    assert smoke._plot_trace_count({"type": "plotly", "plot": ""}) == 0
    assert smoke._plot_trace_count(object()) == 0


def test_update_value_reads_gradio_update_dict() -> None:
    assert (
        smoke._update_value(
            {"choices": [["Selected", "RANK_2"]], "value": "RANK_2"}, "ALL"
        )
        == "RANK_2"
    )
    assert smoke._update_value({"choices": [["All", "ALL"]]}, "ALL") == "ALL"


def test_resolve_client_auth_defaults_to_none() -> None:
    args = SimpleNamespace()

    assert smoke.resolve_client_auth(args, env={}) is None


def test_resolve_client_auth_reads_env_names() -> None:
    args = SimpleNamespace(
        auth_user_env="DEMO_USER",
        auth_password_env="DEMO_PASSWORD",
        require_auth=False,
    )

    assert smoke.resolve_client_auth(
        args,
        env={"DEMO_USER": "risk", "DEMO_PASSWORD": "manager"},
    ) == ("risk", "manager")


def test_resolve_client_auth_require_auth_fails_on_missing_password() -> None:
    args = SimpleNamespace(
        auth_user_env="DEMO_USER",
        auth_password_env="DEMO_PASSWORD",
        require_auth=True,
    )

    try:
        smoke.resolve_client_auth(args, env={"DEMO_USER": "risk"})
    except RuntimeError as error:
        assert "DEMO_PASSWORD" in str(error)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected missing auth password to fail")


def test_make_client_passes_auth_only_when_supplied(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def __init__(self, url: str, auth=None):
            calls.append((url, auth))

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)

    smoke.make_client("http://demo")
    smoke.make_client("http://demo", auth=("risk", "manager"))

    assert calls == [
        ("http://demo", None),
        ("http://demo", ("risk", "manager")),
    ]


def test_run_gradio_api_smoke_uses_live_fixed_start_endpoint(
    monkeypatch,
    tmp_path,
) -> None:
    calls = []

    class FakeClient:
        def __init__(self, url: str):
            self.url = url

        def predict(self, *args, api_name: str):
            calls.append((api_name, args))
            if api_name == "/run_live_openai_prefix_for_app":
                return (
                    "report markdown",
                    "## Scenario Workflow Status\n\n- Story support: `pass`",
                    _frame(1),
                    _frame(1),
                    _frame(2),
                    _frame(11),
                    _plot(8),
                    json.dumps(
                        {
                            "status": "ok",
                            "condition_only_case": {
                                "condition_only_validation": {
                                    "status": "pass",
                                    "forward_warning_count": 1,
                                }
                            },
                            "generation": {
                                "narrative_ensemble_calibration": {
                                    "applied": True,
                                    "mode": "support_gated_directional_delta_calibration",
                                    "effective_beta": 0.25,
                                    "support_gate": 1.0,
                                    "active_direction_count": 1,
                                }
                            },
                            "artifact_paths": {
                                "report": "prefix_report.json",
                                "markdown": "prefix_report.md",
                                "arrays": "prefix_arrays.npz",
                                "run_record": "prefix_run_record.json",
                            },
                            "cached_query": {
                                "condition_source": "external_condition_report",
                                "embedding_metadata": {
                                    "condition_report": "condition_report.json",
                                    "condition_arrays": "condition_arrays.npz",
                                    "embedding_dim": 1536,
                                    "condition_dim": 128,
                                },
                                "grounding": {
                                    "market_implications": [
                                        {
                                            "market": "GOLD",
                                            "direction": "up",
                                            "target_use": "support_prior",
                                        }
                                    ],
                                    "non_conditioning_forward_language": [
                                        {
                                            "phrase": "forward risk",
                                            "handling": "ignore_for_conditioning",
                                        }
                                    ],
                                },
                                "memory_prior": {
                                    "mode": "soft_topk_combined",
                                    "support_alignment": {"status": "pass"},
                                    "candidate_details": [
                                        {
                                            "rank": 1,
                                            "window_id": "joint39_val_0036",
                                            "window_index": 18,
                                            "weight": 1.0,
                                            "memory_support_cosine": 0.9,
                                        }
                                    ],
                                },
                            },
                            "validation_gate": {
                                "selected_start_status": "pass",
                                "diagnostic_baseline_status": "pass",
                                "overall_status": "pass",
                            },
                        }
                    ),
                    {"choices": [["All retrieved analogues", "ALL"]]},
                    _frame(5),
                    _frame(2),
                    _frame(5),
                    _frame(8),
                    _frame(8),
                    _frame(0),
                    {"choices": [["joint39_val_0036", "18"]]},
                )
            if api_name == "/refresh_fan_chart":
                return _plot(8)
            raise AssertionError(f"unexpected api_name: {api_name}")

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)
    summary = smoke.run_gradio_api_smoke(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            mode="live_condition_only",
            story="Safe-haven story.",
            expected_start_index=18,
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
        )
    )

    assert summary["status"] == "ok"
    assert summary["condition_source"] == "external_condition_report"
    assert summary["selected_start_status"] == "pass"
    assert summary["fan_trace_count"] == 8
    assert summary["redraw_trace_count"] == 8
    assert summary["prefix_run_record_path"] == "prefix_run_record.json"
    assert summary["prefix_report_snapshot_path"].endswith(
        "prefix_report_snapshot.json"
    )
    assert summary["prefix_markdown_snapshot_path"].endswith(
        "prefix_report_snapshot.md"
    )
    assert summary["support_candidate_count"] == 1
    assert summary["support_top_candidates"][0]["window_id"] == "joint39_val_0036"
    assert summary["market_implications"][0]["market"] == "GOLD"
    assert summary["forward_warnings"][0]["handling"] == "ignore_for_conditioning"
    assert calls[0] == (
        "/run_live_openai_prefix_for_app",
        ("SPX", "ALL", "Safe-haven story.", 18),
    )
    assert calls[1] == ("/refresh_fan_chart", ("IV_ATM_3M", "ALL"))
    assert (tmp_path / "gradio_api_smoke_summary.json").exists()
    assert (tmp_path / "prefix_report_snapshot.json").exists()
    assert (tmp_path / "prefix_report_snapshot.md").exists()


def test_run_gradio_api_smoke_live_condition_only(monkeypatch, tmp_path) -> None:
    captured_run_args = None

    class FakeClient:
        def __init__(self, url: str):
            self.url = url

        def predict(self, *args, api_name: str):
            nonlocal captured_run_args
            if api_name == "/run_live_openai_prefix_for_app":
                captured_run_args = args
                return (
                    "report markdown",
                    "## Scenario Workflow Status\n\n- Story support: `pass`",
                    _frame(1),
                    _frame(1),
                    _frame(2),
                    _frame(11),
                    _plot(8),
                    json.dumps(
                        {
                            "status": "ok",
                            "condition_only_case": {
                                "condition_only_validation": {
                                    "status": "pass",
                                    "forward_warning_count": 1,
                                }
                            },
                            "generation": {
                                "narrative_ensemble_calibration": {
                                    "applied": True,
                                    "mode": "support_gated_directional_delta_calibration",
                                    "effective_beta": 0.25,
                                    "support_gate": 1.0,
                                    "active_direction_count": 1,
                                }
                            },
                            "cached_query": {
                                "condition_source": "external_condition_report",
                                "embedding_metadata": {
                                    "condition_report": "condition_report.json",
                                    "condition_arrays": "condition_arrays.npz",
                                    "embedding_dim": 1536,
                                    "condition_dim": 128,
                                },
                                "grounding": {
                                    "market_implications": [
                                        {
                                            "market": "GOLD",
                                            "direction": "up",
                                            "target_use": "support_prior",
                                        }
                                    ],
                                    "non_conditioning_forward_language": [
                                        {
                                            "phrase": "forward risk",
                                            "handling": "ignore_for_conditioning",
                                        }
                                    ],
                                },
                                "memory_prior": {
                                    "mode": "soft_topk_combined",
                                    "support_alignment": {"status": "pass"},
                                    "candidate_details": [
                                        {
                                            "rank": 1,
                                            "window_id": "joint39_val_0036",
                                            "window_index": 18,
                                            "weight": 1.0,
                                            "memory_support_cosine": 0.9,
                                        }
                                    ],
                                },
                            },
                            "validation_gate": {
                                "selected_start_status": "pass",
                                "diagnostic_baseline_status": "pass",
                                "overall_status": "pass",
                            },
                        }
                    ),
                    {"choices": [["All retrieved analogues", "ALL"]]},
                    _frame(5),
                    _frame(2),
                    _frame(5),
                    _frame(8),
                    _frame(8),
                    _frame(0),
                    {"choices": [["joint39_val_0036", "18"]]},
                )
            if api_name == "/refresh_fan_chart":
                return _plot(8)
            raise AssertionError(f"unexpected api_name: {api_name}")

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)
    summary = smoke.run_gradio_api_smoke(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            mode="live_condition_only",
            story="A live story with a forward risk warning.",
            expected_start_index=18,
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
        )
    )

    assert captured_run_args is not None
    assert captured_run_args == (
        "SPX",
        "ALL",
        "A live story with a forward risk warning.",
        18,
    )
    assert summary["mode"] == "live_condition_only"
    assert summary["condition_only_validation_status"] == "pass"
    assert summary["condition_only_forward_warning_count"] == 1
    assert summary["support_candidate_count"] == 1
    assert summary["condition_report_path"] == "condition_report.json"
    assert (tmp_path / "prefix_report_snapshot.json").exists()
    assert summary["narrative_calibration_applied"] is True
    assert summary["narrative_calibration_effective_beta"] == 0.25
    assert summary["forward_warnings"][0]["phrase"] == "forward risk"
    assert summary["status"] == "ok"


def test_run_gradio_api_smoke_can_allow_warning_level_selected_start(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeClient:
        def __init__(self, url: str):
            self.url = url

        def predict(self, *args, api_name: str):
            if api_name == "/run_live_openai_prefix_for_app":
                return (
                    "report markdown",
                    "## Scenario Workflow Status\n\n- Story support: `warning`",
                    _frame(1),
                    _frame(1),
                    _frame(2),
                    _frame(11),
                    _plot(8),
                    json.dumps(
                        {
                            "status": "ok",
                            "condition_only_case": {
                                "condition_only_validation": {
                                    "status": "pass",
                                    "forward_warning_count": 1,
                                }
                            },
                            "generation": {
                                "narrative_ensemble_calibration": {
                                    "applied": True,
                                    "mode": "support_gated_directional_delta_calibration",
                                    "effective_beta": 0.25,
                                    "support_gate": 1.0,
                                    "active_direction_count": 1,
                                }
                            },
                            "cached_query": {
                                "condition_source": "external_condition_report",
                                "embedding_metadata": {
                                    "embedding_dim": 1536,
                                    "condition_dim": 128,
                                },
                                "grounding": {
                                    "market_implications": [
                                        {
                                            "market": "SPX",
                                            "direction": "up",
                                            "target_use": "support_prior",
                                        }
                                    ],
                                    "non_conditioning_forward_language": [
                                        {"phrase": "forward risk"}
                                    ],
                                },
                                "memory_prior": {
                                    "mode": "diverse_topk_narrative_start_checked",
                                    "support_alignment": {"status": "pass"},
                                    "candidate_details": [
                                        {
                                            "rank": 1,
                                            "window_id": "joint39_val_0033",
                                            "weight": 1.0,
                                        }
                                    ],
                                },
                            },
                            "validation_gate": {
                                "selected_start_status": "warning",
                                "diagnostic_baseline_status": "pass",
                                "overall_status": "warning",
                            },
                        }
                    ),
                    {"choices": [["Selected", "RANK_1"]], "value": "RANK_1"},
                    _frame(5),
                    _frame(2),
                    _frame(5),
                    _frame(8),
                    _frame(8),
                    _frame(0),
                    {"choices": [["joint39_val_0033", "17"]]},
                )
            if api_name == "/refresh_fan_chart":
                return _plot(8)
            raise AssertionError(f"unexpected api_name: {api_name}")

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)

    summary = smoke.run_gradio_api_smoke(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            mode="live_condition_only",
            story="Fragile risk-on story with forward risk.",
            expected_start_index=0,
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
            allow_start_warning=True,
        )
    )

    assert summary["status"] == "ok"
    assert summary["selected_start_status"] == "warning"
    assert summary["overall_status"] == "warning"


def test_run_gradio_api_smoke_can_allow_non_leaking_condition_warning(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeClient:
        def __init__(self, url: str):
            self.url = url

        def predict(self, *args, api_name: str):
            if api_name == "/run_live_openai_prefix_for_app":
                return (
                    "report markdown",
                    "## Scenario Workflow Status\n\n- Story support: `pass`",
                    _frame(1),
                    _frame(1),
                    _frame(2),
                    _frame(11),
                    _plot(8),
                    json.dumps(
                        {
                            "status": "ok",
                            "condition_only_case": {
                                "condition_only_validation": {
                                    "status": "warning",
                                    "forward_warning_count": 1,
                                    "future_target_count": 0,
                                    "forward_warning_leakage_count": 0,
                                    "reasons": ["condition_role_errors"],
                                }
                            },
                            "generation": {
                                "narrative_ensemble_calibration": {
                                    "applied": True,
                                    "effective_beta": 0.25,
                                    "support_gate": 1.0,
                                    "active_direction_count": 1,
                                }
                            },
                            "cached_query": {
                                "condition_source": "external_condition_report",
                                "embedding_metadata": {
                                    "embedding_dim": 1536,
                                    "condition_dim": 128,
                                },
                                "grounding": {
                                    "market_implications": [
                                        {
                                            "market": "BBB_OAS",
                                            "direction": "tighter",
                                            "target_use": "support_prior",
                                        }
                                    ],
                                    "non_conditioning_forward_language": [
                                        {"phrase": "forward risk"}
                                    ],
                                },
                                "memory_prior": {
                                    "mode": "diverse_topk_narrative_start_checked",
                                    "support_alignment": {"status": "pass"},
                                    "candidate_details": [
                                        {
                                            "rank": 1,
                                            "window_id": "joint39_val_0036",
                                            "weight": 1.0,
                                        }
                                    ],
                                },
                            },
                            "validation_gate": {
                                "selected_start_status": "pass",
                                "diagnostic_baseline_status": "pass",
                                "overall_status": "pass",
                            },
                        }
                    ),
                    {"choices": [["Selected", "RANK_1"]], "value": "RANK_1"},
                    _frame(5),
                    _frame(2),
                    _frame(5),
                    _frame(8),
                    _frame(8),
                    _frame(0),
                    {"choices": [["joint39_val_0036", "18"]]},
                )
            if api_name == "/refresh_fan_chart":
                return _plot(8)
            raise AssertionError(f"unexpected api_name: {api_name}")

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)

    summary = smoke.run_gradio_api_smoke(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            mode="live_condition_only",
            story="Fragile risk-on story with forward risk.",
            expected_start_index=18,
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
            allow_condition_warning=True,
        )
    )

    assert summary["status"] == "ok"
    assert summary["condition_only_validation_status"] == "warning"
    assert summary["allow_condition_warning"] is True
