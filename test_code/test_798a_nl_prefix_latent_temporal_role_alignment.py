import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_temporal_role_alignment import (
    evaluate_temporal_report,
    run_temporal_alignment,
    split_grounding_by_temporal_role,
    temporal_role_for_implication,
)


def test_temporal_role_for_implication_splits_current_forward_and_ambiguous() -> None:
    assert (
        temporal_role_for_implication({"evidence": ["equities are recovering"]})
        == "current_regime"
    )
    assert (
        temporal_role_for_implication({"evidence": ["forward risk could unwind"]})
        == "forward_risk"
    )
    assert (
        temporal_role_for_implication(
            {"evidence": ["volatility is compressing", "forward risk reversal"]}
        )
        == "ambiguous_mixed"
    )


def test_split_grounding_by_temporal_role_preserves_implications() -> None:
    split = split_grounding_by_temporal_role(
        {
            "market_implications": [
                {"market": "SPX", "direction": "up", "evidence": ["recovering"]},
                {
                    "market": "VIX",
                    "direction": "up",
                    "evidence": ["future volatility shock"],
                },
            ]
        }
    )

    assert len(split["current_regime"]) == 1
    assert len(split["forward_risk"]) == 1
    assert split["current_regime"][0]["temporal_role"] == "current_regime"


def test_evaluate_temporal_report_scores_support_and_future_separately(tmp_path) -> None:
    report = {
        "cached_query": {
            "condition_source": "external_condition_report",
            "memory_prior_mode": "soft_topk_combined",
            "grounding": {
                "market_implications": [
                    {
                        "market": "SPX",
                        "direction": "up",
                        "evidence": ["equities are recovering"],
                    },
                    {
                        "market": "VIX",
                        "direction": "up",
                        "evidence": ["forward risk volatility shock"],
                    },
                ]
            },
            "memory_prior": {
                "terminal_rows": [
                    {"Market": "SPX", "Mean Terminal Delta": 1.0},
                    {"Market": "VIX", "Mean Terminal Delta": -1.0},
                ]
            },
        },
        "generation": {
            "terminal_delta_summary": [
                {"market": "SPX", "mean_terminal_delta": -1.0},
                {"market": "VIX", "mean_terminal_delta": 2.0},
            ]
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = evaluate_temporal_report(path)

    assert result["current_support_alignment"]["mismatch_count"] == 0
    assert result["future_generated_alignment"]["mismatch_count"] == 0
    assert (
        result["legacy_all_implications_future_alignment"]["mismatch_count"] == 1
    )


def test_run_temporal_alignment_expands_casebook_and_writes_summary(tmp_path) -> None:
    report_path = tmp_path / "case" / "report.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "cached_query": {
                    "grounding": {
                        "market_implications": [
                            {
                                "market": "SPX",
                                "direction": "up",
                                "evidence": ["equities are recovering"],
                            }
                        ]
                    },
                    "memory_prior": {
                        "terminal_rows": [
                            {"Market": "SPX", "Mean Terminal Delta": 1.0}
                        ]
                    },
                },
                "generation": {"terminal_delta_summary": []},
            }
        ),
        encoding="utf-8",
    )
    casebook_path = tmp_path / "casebook.json"
    casebook_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"artifact_paths": {"prefix_report": str(report_path)}}
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = run_temporal_alignment(
        SimpleNamespace(input=[str(casebook_path)], output_dir=str(tmp_path / "out"))
    )

    assert summary["status"] == "ok"
    assert summary["case_count"] == 1
    assert summary["totals"]["current_mismatch_rate"] == 0.0
    assert (tmp_path / "out" / "temporal_role_alignment_summary.json").exists()
