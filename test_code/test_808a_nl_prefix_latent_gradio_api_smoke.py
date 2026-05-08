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


def test_run_gradio_api_smoke_uses_cached_casebook(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeClient:
        def __init__(self, url: str):
            self.url = url

        def predict(self, *args, api_name: str):
            calls.append((api_name, args))
            if api_name == "/cached_prefix_casebook_update":
                return (
                    "Safe-haven story.",
                    True,
                    18,
                    False,
                    False,
                    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
                    "prefix_latent_condition_only_report_823b_safe_haven/"
                    "condition_only_report.json",
                    "## Cached Casebook\n\n- OpenAI calls: `none for this cached run`",
                )
            if api_name == "/run_prefix_latent_for_app":
                return (
                    "report markdown",
                    "## Prefix-Latent Run Status\n\n- Selected-start: `pass`",
                    _frame(1),
                    _frame(1),
                    _frame(2),
                    _frame(11),
                    _plot(8),
                    json.dumps(
                        {
                            "status": "ok",
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
            if api_name == "/refresh_fan_chart_2":
                return _plot(8)
            raise AssertionError(f"unexpected api_name: {api_name}")

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)
    summary = smoke.run_gradio_api_smoke(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            casebook_choice="safe_haven_gold_bid:18",
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
    assert summary["support_candidate_count"] == 1
    assert summary["support_top_candidates"][0]["window_id"] == "joint39_val_0036"
    assert summary["market_implications"][0]["market"] == "GOLD"
    assert summary["forward_warnings"][0]["handling"] == "ignore_for_conditioning"
    assert calls[0][0] == "/cached_prefix_casebook_update"
    assert calls[1][0] == "/run_prefix_latent_for_app"
    assert calls[2] == ("/refresh_fan_chart_2", ("IV_ATM_3M", "ALL"))
    assert (tmp_path / "gradio_api_smoke_summary.json").exists()


def test_run_gradio_api_smoke_live_condition_only(monkeypatch, tmp_path) -> None:
    captured_run_args = None

    class FakeClient:
        def __init__(self, url: str):
            self.url = url

        def predict(self, *args, api_name: str):
            nonlocal captured_run_args
            if api_name == "/cached_prefix_casebook_update":
                raise AssertionError("live mode should not call cached casebook")
            if api_name == "/run_prefix_latent_for_app":
                captured_run_args = args
                return (
                    "report markdown",
                    "## Prefix-Latent Run Status\n\n- Selected-start: `pass`",
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
            if api_name == "/refresh_fan_chart_2":
                return _plot(8)
            raise AssertionError(f"unexpected api_name: {api_name}")

    monkeypatch.setattr(smoke, "_client_class", lambda: FakeClient)
    summary = smoke.run_gradio_api_smoke(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            mode="live_condition_only",
            casebook_choice="safe_haven_gold_bid:18",
            story="A live story with a forward risk warning.",
            expected_start_index=18,
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
        )
    )

    assert captured_run_args is not None
    assert captured_run_args[4] is True
    assert captured_run_args[6] == ""
    assert captured_run_args[7] is True
    assert captured_run_args[8] is True
    assert captured_run_args[9] == 18
    assert summary["mode"] == "live_condition_only"
    assert summary["condition_only_validation_status"] == "pass"
    assert summary["condition_only_forward_warning_count"] == 1
    assert summary["support_candidate_count"] == 1
    assert summary["condition_report_path"] == "condition_report.json"
    assert summary["forward_warnings"][0]["phrase"] == "forward risk"
    assert summary["status"] == "ok"
