import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar import (
    nl_prefix_latent_gradio_live_api_casebook as casebook,
)


def test_run_live_api_casebook_aggregates_cases(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_casebook_update(choice: str):
        start = 22 if choice.endswith(":22") else 18
        return (
            f"Story for {choice}. The forward risk is warning-only.",
            True,
            start,
            False,
            False,
            "",
            "status",
        )

    def fake_run_gradio_api_smoke(args):
        calls.append(args)
        return {
            "status": "ok",
            "errors": [],
            "artifact_paths": {
                "summary": f"{args.output_dir}/gradio_api_smoke_summary.json"
            },
            "condition_only_validation_status": "pass",
            "condition_only_forward_warning_count": 1,
            "selected_start_status": "pass",
            "overall_status": "pass",
            "support_candidate_count": 8,
            "support_prior_mode": "soft_topk_combined",
            "redraw_trace_count": 8,
            "openai_usage": {"total_tokens": 123},
        }

    monkeypatch.setattr(casebook, "cached_prefix_casebook_update", fake_casebook_update)
    monkeypatch.setattr(casebook, "run_gradio_api_smoke", fake_run_gradio_api_smoke)

    summary = casebook.run_live_api_casebook(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
            casebook_choices=[
                "commodity_inflation_pressure:18",
                "dollar_liquidity_squeeze:22",
            ],
            continue_on_error=False,
        )
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 2
    assert summary["pass_count"] == 2
    assert summary["total_openai_tokens"] == 246
    assert summary["min_support_candidate_count"] == 8
    assert calls[0].mode == "live_condition_only"
    assert calls[0].expected_start_index == 18
    assert calls[1].expected_start_index == 22
    assert (tmp_path / "gradio_live_api_casebook_summary.json").exists()
    assert (tmp_path / "gradio_live_api_casebook_summary.md").exists()


def test_run_live_api_casebook_fails_on_warning_leak_contract(
    monkeypatch, tmp_path
) -> None:
    def fake_casebook_update(choice: str):
        return ("Story.", True, 18, False, False, "", "status")

    def fake_run_gradio_api_smoke(args):
        return {
            "status": "ok",
            "errors": [],
            "artifact_paths": {
                "summary": f"{args.output_dir}/gradio_api_smoke_summary.json"
            },
            "condition_only_validation_status": "pass",
            "condition_only_forward_warning_count": 0,
            "selected_start_status": "pass",
            "overall_status": "pass",
            "support_candidate_count": 8,
            "support_prior_mode": "soft_topk_combined",
            "redraw_trace_count": 8,
            "openai_usage": {"total_tokens": 123},
        }

    monkeypatch.setattr(casebook, "cached_prefix_casebook_update", fake_casebook_update)
    monkeypatch.setattr(casebook, "run_gradio_api_smoke", fake_run_gradio_api_smoke)

    try:
        casebook.run_live_api_casebook(
            SimpleNamespace(
                url="http://127.0.0.1:7861",
                output_dir=str(tmp_path),
                samples=2,
                fan_market="SPX",
                redraw_market="IV_ATM_3M",
                casebook_choices=["safe_haven_gold_bid:18"],
                continue_on_error=False,
            )
        )
    except RuntimeError as error:
        assert "forward warning missing" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected RuntimeError")
