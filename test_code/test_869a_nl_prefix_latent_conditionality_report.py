import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_conditionality_report import (
    build_conditionality_report,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_conditionality_report_summarizes_fixed_start_controls(tmp_path) -> None:
    contrast_path = tmp_path / "contrast.json"
    control_path = tmp_path / "controls.json"
    output_dir = tmp_path / "out"
    _write(
        contrast_path,
        {
            "start_max_abs_diff": 0.0,
            "case_summaries": [
                {
                    "case_name": "fragile_risk_on_start18",
                    "terminal_summary": {
                        "SPX": {"mean": 10.0},
                        "VIX": {"mean": -1.0},
                    },
                    "top_support": [{"rank": 1, "window_index": 7}],
                },
                {
                    "case_name": "safe_haven_gold_start18",
                    "terminal_summary": {
                        "SPX": {"mean": -12.0},
                        "VIX": {"mean": 2.0},
                    },
                    "top_support": [{"rank": 1, "window_index": 8}],
                },
            ],
            "pairwise_contrasts": [
                {
                    "start_name": "fixed_start_18",
                    "left_case": "fragile_risk_on_start18",
                    "right_case": "safe_haven_gold_start18",
                    "standardized_l2_gap": 1.25,
                    "largest_abs_market_gaps": [
                        {"market": "SPX", "standardized_mean_gap": 1.1},
                        {"market": "VIX", "standardized_mean_gap": -0.8},
                    ],
                }
            ],
        },
    )
    _write(
        control_path,
        {
            "observed": {
                "gap_summary": {
                    "overall_median_gap": 1.25,
                    "pair_count": 1,
                }
            },
            "controls": {
                "start_only": {"gap_summary": {"overall_median_gap": 0.0}},
                "same_narrative_repeat": {
                    "gap_summary": {"overall_median_gap": 0.25}
                },
                "within_run_bootstrap": {
                    "gap_summary": {"overall_median_gap": 0.3}
                },
            },
            "per_start_controls": [
                {
                    "start_name": "fixed_start_18",
                    "status": "pass",
                    "observed_median_gap": 1.25,
                    "repeat_ratio": 0.2,
                    "bootstrap_ratio": 0.24,
                },
                {
                    "start_name": "fixed_start_178",
                    "status": "fail",
                    "observed_median_gap": 0.4,
                    "repeat_ratio": 0.8,
                },
                {"start_name": "fixed_start_22", "status": "pass"},
                {"start_name": "fixed_start_40", "status": "pass"},
            ],
        },
    )

    report = build_conditionality_report(
        contrast_report=contrast_path,
        control_report=control_path,
        output_dir=output_dir,
        top_pairs=4,
    )

    assert report["status"] == "pass"
    assert report["headline"]["start_max_abs_diff"] == 0.0
    assert report["headline"]["start_only_median_gap"] == 0.0
    assert report["headline"]["reliable_start_count"] == 3
    assert report["top_pairwise_contrasts"][0]["left_family"] == "fragile_risk_on"
    assert report["recommended_casebook_examples"][0]["left_terminal_mean"]["SPX"] == 10.0
    assert Path(report["artifact_paths"]["json"]).exists()
    assert "Narrative Conditionality Report" in Path(
        report["artifact_paths"]["markdown"]
    ).read_text(encoding="utf-8")
