import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_start_damping_diagnostic import (
    build_start_damping_diagnostic,
    render_markdown,
)


def _write_run_report(path: Path, *, cosine: float, top_window: int) -> str:
    path.write_text(
        json.dumps(
            {
                "variant_rows": [
                    {"is_operational": False, "memory_support_cosine": 0.99},
                    {
                        "is_operational": True,
                        "memory_support_cosine": cosine,
                        "memory_prior_top_window_index": top_window,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _bakeoff_report(tmp_path: Path) -> dict:
    rows = []
    for idx, case in enumerate(["risk_on", "risk_off"]):
        rows.append(
            {
                "start_name": "fixed_start_good",
                "case_name": case,
                "run_report": _write_run_report(
                    tmp_path / f"good_{idx}.json", cosine=0.82 + idx * 0.01, top_window=10 + idx
                ),
                "validation_operational": "pass",
                "memory_prior_direction_status": "pass",
                "start_distance_z": 3.0,
                "memory_prior_weighted_start_distance_z": 4.0,
                "memory_prior_support_weighted_match_rate": 1.0,
                "scenario_metrics": {
                    "energy_score_z_improvement_vs_persistence": 0.12,
                    "ensemble_crps_z_improvement_vs_persistence": 0.10,
                },
            }
        )
    for idx, case in enumerate(["risk_on", "risk_off"]):
        rows.append(
            {
                "start_name": "fixed_start_damped",
                "case_name": case,
                "run_report": _write_run_report(
                    tmp_path / f"damped_{idx}.json", cosine=0.55 + idx * 0.01, top_window=20
                ),
                "validation_operational": "warning",
                "memory_prior_direction_status": "pass",
                "start_distance_z": 20.0,
                "memory_prior_weighted_start_distance_z": 12.0,
                "memory_prior_support_weighted_match_rate": 1.0,
                "scenario_metrics": {
                    "energy_score_z_improvement_vs_persistence": 0.08,
                    "ensemble_crps_z_improvement_vs_persistence": 0.07,
                },
            }
        )
    return {"rows": rows}


def _contrast_report() -> dict:
    return {
        "pairwise_contrasts": [
            {
                "start_name": "fixed_start_good",
                "left_case": "risk_on",
                "right_case": "risk_off",
                "standardized_l2_gap": 2.1,
            },
            {
                "start_name": "fixed_start_damped",
                "left_case": "risk_on",
                "right_case": "risk_off",
                "standardized_l2_gap": 0.6,
            },
        ]
    }


def _gate_report() -> dict:
    return {
        "start_block_assessments": [
            {
                "start_name": "fixed_start_good",
                "status": "pass",
                "max_standardized_l2_gap": 2.1,
                "median_standardized_l2_gap": 2.1,
                "min_standardized_l2_gap": 2.1,
                "pair_count": 1,
                "warnings": [],
                "failures": [],
            },
            {
                "start_name": "fixed_start_damped",
                "status": "warning",
                "max_standardized_l2_gap": 0.6,
                "median_standardized_l2_gap": 0.6,
                "min_standardized_l2_gap": 0.6,
                "pair_count": 1,
                "warnings": ["start_dampens_narrative_influence"],
                "failures": [],
            },
        ]
    }


def test_build_start_damping_diagnostic_identifies_mechanism(tmp_path: Path) -> None:
    report = build_start_damping_diagnostic(
        bakeoff_report=_bakeoff_report(tmp_path),
        contrast_report=_contrast_report(),
        gate_report=_gate_report(),
    )

    assert report["status"] == "warning"
    assert report["headline"]["damped_starts"] == ["fixed_start_damped"]
    rows = {row["start_name"]: row for row in report["start_summaries"]}
    damped_notes = "\n".join(rows["fixed_start_damped"]["mechanism_notes"])
    assert "start_compatibility_warning" in damped_notes
    assert "low_narrative_separation" in damped_notes
    assert "weaker_memory_support" in damped_notes
    assert rows["fixed_start_damped"]["unique_top_support_window_count"] == 1


def test_render_markdown_explains_warning_not_hard_failure(tmp_path: Path) -> None:
    report = build_start_damping_diagnostic(
        bakeoff_report=_bakeoff_report(tmp_path),
        contrast_report=_contrast_report(),
        gate_report=_gate_report(),
    )

    text = render_markdown(report)

    assert "not a hard model failure" in text
    assert "fixed_start_damped" in text
    assert "Start Diagnostics" in text
