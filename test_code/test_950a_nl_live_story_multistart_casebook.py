import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

from experiments.backfill.block_ar import nl_live_story_multistart_casebook as multi


def test_multistart_casebook_runs_each_start_and_aggregates(monkeypatch, tmp_path):
    calls = []

    def fake_run_live_api_casebook(args):
        calls.append(args)
        start = int(args.fixed_start_index)
        return {
            "status": "ok",
            "artifact_paths": {
                "summary": str(tmp_path / f"start_{start}" / "summary.json")
            },
            "case_count": 2,
            "pass_count": 2,
            "total_openai_tokens": 100 + start,
            "calibration_applied_count": 2,
            "min_calibration_support_gate": 1.0,
            "fixed_start_index": start,
        }

    def fake_build_analysis(summary_path, markets=None):
        start = int(str(summary_path).split("/")[-2].removeprefix("start_"))
        return {
            "status": "ok",
            "case_count": 2,
            "pass_count": 2,
            "fixed_start_index": start,
            "calibration_applied_count": 2,
            "min_calibration_support_gate": 1.0,
            "max_pairwise_support_jaccard": 0.1 * start,
            "mean_pairwise_support_jaccard": 0.05 * start,
            "total_openai_tokens": 100 + start,
        }

    monkeypatch.setattr(multi, "run_live_api_casebook", fake_run_live_api_casebook)
    monkeypatch.setattr(multi, "build_live_story_deck_analysis", fake_build_analysis)
    monkeypatch.setattr(multi, "plot_terminal_delta_panel", lambda report, path: str(path))

    summary = multi.run_multistart_live_story_deck(
        SimpleNamespace(
            url="http://127.0.0.1:7860",
            output_dir=str(tmp_path),
            start_indices=[0, 22],
            samples=4,
            fan_market="SPX",
            redraw_market="VIX",
            default_story_cases=["fragile_risk_on_rebound", "safe_haven_gold_bid"],
            continue_on_error=False,
            no_plot=True,
        )
    )

    assert summary["status"] == "ok"
    assert summary["start_count"] == 2
    assert summary["case_count"] == 4
    assert summary["pass_count"] == 4
    assert summary["calibration_applied_count"] == 4
    assert summary["total_openai_tokens"] == 222
    assert summary["min_calibration_support_gate"] == 1.0
    assert summary["max_pairwise_support_jaccard"] == pytest.approx(2.2)
    assert [call.fixed_start_index for call in calls] == [0, 22]
    assert all(call.use_default_story_deck for call in calls)
    assert (tmp_path / "multi_start_live_story_deck_summary.json").exists()
    assert (tmp_path / "multi_start_live_story_deck_summary.md").exists()


def test_multistart_casebook_records_failed_start_when_continuing(
    monkeypatch, tmp_path
):
    def fake_run_live_api_casebook(args):
        if int(args.fixed_start_index) == 18:
            raise RuntimeError("boom")
        return {
            "status": "ok",
            "artifact_paths": {"summary": str(tmp_path / "start_0" / "summary.json")},
            "case_count": 1,
            "pass_count": 1,
            "total_openai_tokens": 10,
            "calibration_applied_count": 1,
            "min_calibration_support_gate": 1.0,
            "fixed_start_index": 0,
        }

    monkeypatch.setattr(multi, "run_live_api_casebook", fake_run_live_api_casebook)
    monkeypatch.setattr(
        multi,
        "build_live_story_deck_analysis",
        lambda summary_path, markets=None: {
            "status": "ok",
            "case_count": 1,
            "pass_count": 1,
            "fixed_start_index": 0,
            "calibration_applied_count": 1,
            "min_calibration_support_gate": 1.0,
            "max_pairwise_support_jaccard": 0.0,
            "mean_pairwise_support_jaccard": 0.0,
            "total_openai_tokens": 10,
        },
    )

    summary = multi.run_multistart_live_story_deck(
        SimpleNamespace(
            url="http://127.0.0.1:7860",
            output_dir=str(tmp_path),
            start_indices=[0, 18],
            samples=2,
            fan_market="SPX",
            redraw_market="VIX",
            default_story_cases=None,
            continue_on_error=True,
            no_plot=True,
        )
    )

    assert summary["status"] == "fail"
    assert summary["start_count"] == 2
    assert summary["completed_start_count"] == 1
    assert summary["case_count"] == 1
    assert "start 18" in summary["errors"][0]


def test_multistart_casebook_analyzes_failed_start_with_written_summary(
    monkeypatch, tmp_path
):
    def fake_run_live_api_casebook(args):
        start_dir = tmp_path / f"start_{int(args.fixed_start_index)}"
        start_dir.mkdir(parents=True, exist_ok=True)
        summary_path = start_dir / "gradio_live_api_casebook_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "fail",
                    "case_count": 2,
                    "pass_count": 1,
                    "total_openai_tokens": 25,
                    "calibration_applied_count": 1,
                    "min_calibration_support_gate": 0.0,
                    "selected_start_warning_count": 1,
                    "condition_validation_warning_count": 0,
                    "artifact_paths": {"summary": str(summary_path)},
                    "cases": [],
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError("Live Gradio API casebook failed")

    def fake_build_analysis(summary_path, markets=None):
        return {
            "status": "fail",
            "case_count": 2,
            "pass_count": 1,
            "fixed_start_index": 22,
            "calibration_applied_count": 1,
            "min_calibration_support_gate": 0.0,
            "max_pairwise_support_jaccard": 0.3,
            "mean_pairwise_support_jaccard": 0.1,
            "total_openai_tokens": 25,
        }

    monkeypatch.setattr(multi, "run_live_api_casebook", fake_run_live_api_casebook)
    monkeypatch.setattr(multi, "build_live_story_deck_analysis", fake_build_analysis)
    monkeypatch.setattr(multi, "plot_terminal_delta_panel", lambda report, path: str(path))

    summary = multi.run_multistart_live_story_deck(
        SimpleNamespace(
            url="http://127.0.0.1:7860",
            output_dir=str(tmp_path),
            start_indices=[22],
            samples=2,
            fan_market="SPX",
            redraw_market="VIX",
            default_story_cases=None,
            continue_on_error=True,
            no_plot=True,
        )
    )

    assert summary["status"] == "fail"
    assert summary["completed_start_count"] == 0
    assert summary["analyzed_start_count"] == 1
    assert summary["case_count"] == 2
    assert summary["pass_count"] == 1
    assert summary["calibration_applied_count"] == 1
    assert summary["total_openai_tokens"] == 25
    assert summary["starts"][0]["analysis_path"].endswith(
        "fixed_start_live_story_deck_analysis.json"
    )
