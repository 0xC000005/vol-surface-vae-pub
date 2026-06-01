import json
import sys
from pathlib import Path
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
            "narrative_calibration_applied": True,
            "narrative_calibration_effective_beta": 0.25,
            "narrative_calibration_support_gate": 1.0,
            "narrative_calibration_active_direction_count": 2,
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
            use_default_story_deck=False,
            allow_unqualified_narratives=True,
            fixed_start_index=22,
            continue_on_error=False,
        )
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 2
    assert summary["pass_count"] == 2
    assert summary["total_openai_tokens"] == 246
    assert summary["min_support_candidate_count"] == 8
    assert summary["calibration_applied_count"] == 2
    assert summary["min_calibration_support_gate"] == 1.0
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
            "narrative_calibration_applied": True,
            "narrative_calibration_effective_beta": 0.25,
            "narrative_calibration_support_gate": 1.0,
            "narrative_calibration_active_direction_count": 2,
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
                use_default_story_deck=False,
                allow_unqualified_narratives=True,
                fixed_start_index=22,
                continue_on_error=False,
            )
        )
    except RuntimeError as error:
        assert "forward warning missing" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected RuntimeError")


def test_run_live_api_casebook_can_use_fixed_start_story_deck(
    monkeypatch, tmp_path
) -> None:
    calls = []

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
            "support_candidate_count": 3,
            "support_prior_mode": "diverse_topk_narrative_start_checked",
            "redraw_trace_count": 8,
            "openai_usage": {"total_tokens": 111},
            "narrative_calibration_applied": True,
            "narrative_calibration_effective_beta": 0.25,
            "narrative_calibration_support_gate": 1.0,
            "narrative_calibration_active_direction_count": 2,
        }

    monkeypatch.setattr(casebook, "run_gradio_api_smoke", fake_run_gradio_api_smoke)

    summary = casebook.run_live_api_casebook(
        SimpleNamespace(
            url="http://127.0.0.1:7861",
            output_dir=str(tmp_path),
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
            casebook_choices=None,
            use_default_story_deck=True,
            default_story_cases=[
                "fragile_risk_on_rebound",
                "safe_haven_gold_bid",
            ],
            fixed_start_index=22,
            allow_unqualified_narratives=False,
            continue_on_error=False,
        )
    )

    assert summary["status"] == "ok"
    assert summary["use_default_story_deck"] is True
    assert summary["fixed_start_index"] == 22
    assert summary["case_count"] == 2
    assert calls[0].expected_start_index == 22
    assert calls[1].expected_start_index == 22
    assert calls[0].story != calls[1].story


def test_run_live_api_casebook_rejects_unqualified_cached_story_by_default(
    monkeypatch, tmp_path
) -> None:
    def fake_casebook_update(choice: str):
        return ("Short unqualified story.", True, 18, False, False, "", "status")

    monkeypatch.setattr(casebook, "cached_prefix_casebook_update", fake_casebook_update)

    try:
        casebook.run_live_api_casebook(
            SimpleNamespace(
                url="http://127.0.0.1:7861",
                output_dir=str(tmp_path),
                samples=2,
                fan_market="SPX",
                redraw_market="IV_ATM_3M",
                casebook_choices=["safe_haven_gold_bid:18"],
                use_default_story_deck=False,
                allow_unqualified_narratives=False,
                fixed_start_index=22,
                continue_on_error=False,
            )
        )
    except casebook.UnqualifiedNarrativeError as error:
        assert "casebook_choice:safe_haven_gold_bid:18" in str(error)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("expected UnqualifiedNarrativeError")


def test_run_live_api_casebook_preserves_failed_smoke_artifacts(
    monkeypatch, tmp_path
) -> None:
    def fake_run_gradio_api_smoke(args):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "fail",
            "errors": ["calibration_not_applied"],
            "artifact_paths": {
                "summary": str(output_dir / "gradio_api_smoke_summary.json")
            },
            "condition_only_validation_status": "pass",
            "condition_only_forward_warning_count": 1,
            "selected_start_status": "warning",
            "overall_status": "warning",
            "support_candidate_count": 4,
            "support_prior_mode": "diverse_topk_narrative_start_checked",
            "redraw_trace_count": 11,
            "openai_usage": {"total_tokens": 123},
            "narrative_calibration_applied": False,
            "narrative_calibration_effective_beta": 0.0,
            "narrative_calibration_support_gate": 0.0,
            "narrative_calibration_active_direction_count": 4,
            "prefix_report_snapshot_path": str(
                output_dir / "prefix_report_snapshot.json"
            ),
            "prefix_arrays_snapshot_path": str(
                output_dir / "prefix_arrays_snapshot.npz"
            ),
        }
        Path(payload["prefix_report_snapshot_path"]).write_text(
            json.dumps({"generation": {"narrative_ensemble_calibration": {}}}),
            encoding="utf-8",
        )
        Path(payload["prefix_arrays_snapshot_path"]).write_bytes(b"npz")
        Path(payload["artifact_paths"]["summary"]).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        raise RuntimeError("Gradio API smoke failed: ['calibration_not_applied']")

    monkeypatch.setattr(casebook, "run_gradio_api_smoke", fake_run_gradio_api_smoke)

    try:
        casebook.run_live_api_casebook(
            SimpleNamespace(
                url="http://127.0.0.1:7861",
                output_dir=str(tmp_path),
                samples=2,
                fan_market="SPX",
                redraw_market="VIX",
                casebook_choices=None,
                use_default_story_deck=True,
                default_story_cases=["dollar_liquidity_squeeze"],
                fixed_start_index=22,
                allow_start_warning=True,
                allow_condition_warning=True,
                allow_unqualified_narratives=False,
                continue_on_error=True,
            )
        )
    except RuntimeError as error:
        assert "calibration_not_applied" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected RuntimeError")

    summary = json.loads(
        (tmp_path / "gradio_live_api_casebook_summary.json").read_text()
    )
    case = summary["cases"][0]
    assert case["status"] == "fail"
    assert case["errors"] == ["calibration_not_applied"]
    assert case["condition_only_validation_status"] == "pass"
    assert case["support_candidate_count"] == 4
    assert case["prefix_report_snapshot_path"].endswith("prefix_report_snapshot.json")
    assert case["summary_path"].endswith("gradio_api_smoke_summary.json")
    assert summary["total_openai_tokens"] == 123
