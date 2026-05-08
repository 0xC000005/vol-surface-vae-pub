import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_hard_case_decomposition import (
    decompose_report,
    display_factor_name,
    hard_case_rows,
    production_decision,
)


def write_fake_report(root: Path) -> Path:
    arrays = root / "prefix_latent_story_smoke_arrays.npz"
    samples = np.zeros((2, 2, 30, 4), dtype=np.float32)
    samples[1, :, -1, 2] = 2.0
    samples[1, :, :, 1] = 0.5
    np.savez_compressed(
        arrays,
        samples=samples,
        delta_scale=np.ones((30, 4), dtype=np.float32),
    )
    report = root / "prefix_latent_story_smoke_report.json"
    report.write_text(
        json.dumps(
            {
                "artifact_inputs": {},
                "artifact_paths": {"arrays": str(arrays)},
                "cached_query": {
                    "memory_prior": {
                        "support_alignment": {
                            "status": "pass",
                            "checked_count": 5,
                            "mismatch_count": 0,
                        }
                    }
                },
                "validation_gate": {
                    "overall_status": "warning",
                    "operational_status": "warning",
                    "endpoint_max_abs_error": 0.0,
                    "thresholds": {
                        "endpoint_abs_fail": 1e-6,
                        "memory_cosine_warn": 0.8,
                        "memory_cosine_fail": 0.65,
                        "start_distance_warn": 15.0,
                        "start_distance_fail": 32.0,
                        "rollout_shift_warn": 1.0,
                        "rollout_shift_fail": 2.0,
                    },
                    "cases": [
                        {
                            "case_index": 0,
                            "variant": "original",
                            "is_operational": False,
                            "status": "pass",
                            "warnings": [],
                            "failures": [],
                        },
                        {
                            "case_index": 1,
                            "variant": "balanced_memory_start",
                            "is_operational": True,
                            "status": "warning",
                            "warnings": ["large_rollout_shift"],
                            "failures": [],
                            "input_memory_cosine": 0.97,
                            "start_distance_z": 14.0,
                            "mean_abs_delta_z": 0.2,
                            "terminal_mean_abs_delta_z": 1.2,
                            "start_window_index": 40,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return report


def test_decompose_report_classifies_components(tmp_path: Path) -> None:
    report = write_fake_report(tmp_path)
    decomposition = decompose_report(report, top_factors=2)
    assert decomposition["selected_status"] == "warning"
    assert decomposition["components"]["support_prior"]["status"] == "pass"
    assert decomposition["components"]["start_distance"]["status"] == "pass"
    assert decomposition["components"]["rollout_shift"]["status"] == "warning"
    assert decomposition["top_rollout_shift_factors"][0]["factor_index"] == 2


def test_production_decision_warns_when_only_rollout_is_sensitive(tmp_path: Path) -> None:
    report = write_fake_report(tmp_path)
    decomposition = decompose_report(report, top_factors=2)
    decision = production_decision([decomposition])
    assert decision["decision"] == "warn_and_continue_for_narrative_only"


def test_production_decision_accepts_clean_viable_policy(tmp_path: Path) -> None:
    report = write_fake_report(tmp_path)
    decomposition = decompose_report(report, top_factors=2)
    decomposition["selected_warnings"] = []
    decomposition["components"]["rollout_shift"]["status"] = "pass"
    decomposition["components"]["rollout_shift"]["terminal_mean_abs_delta_z"] = 0.5
    decision = production_decision([decomposition])
    assert decision["decision"] == "accept_for_narrative_only"


def test_production_decision_ignores_nonviable_start_policy(tmp_path: Path) -> None:
    report = write_fake_report(tmp_path)
    viable = decompose_report(report, top_factors=2)
    nonviable = json.loads(json.dumps(viable))
    nonviable["start_mode"] = "memory_nearest_start"
    nonviable["components"]["start_distance"]["status"] = "warning"
    nonviable["selected_warnings"] = ["large_start_distance", "large_rollout_shift"]
    decision = production_decision([viable, nonviable])
    assert decision["decision"] == "warn_and_continue_for_narrative_only"
    assert decision["viable_start_modes"] == ["balanced_memory_start"]


def test_hard_case_rows_filters_case_name() -> None:
    summary = {"rows": [{"case_name": "a"}, {"case_name": "b"}]}
    assert hard_case_rows(summary, "b") == [{"case_name": "b"}]


def test_display_factor_name_humanizes_common_names() -> None:
    assert display_factor_name(7, "iv:07") == "IV 3M K=1.00"
    assert display_factor_name(25, "factor:spx") == "SPX"
