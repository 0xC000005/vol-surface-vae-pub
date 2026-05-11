import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

import experiments.backfill.block_ar.nl_prefix_latent_fixed_start_control_suite as control_suite
from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_control_suite import (
    bootstrap_gap_report,
    build_control_suite,
    repeat_gap_report,
)


def _contrast(gaps: list[float]) -> dict[str, object]:
    return {
        "pairwise_contrasts": [
            {
                "start_name": "fixed_start_1",
                "left_case": f"left_{idx}",
                "right_case": f"right_{idx}",
                "standardized_l2_gap": gap,
            }
            for idx, gap in enumerate(gaps)
        ]
    }


def test_build_control_suite_passes_when_controls_are_small() -> None:
    report = build_control_suite(
        observed_contrast=_contrast([2.0, 2.4, 1.6]),
        start_only_contrast={
            "gap_summary": {
                "pair_count": 3,
                "overall_median_gap": 0.1,
                "overall_mean_gap": 0.1,
                "overall_p90_gap": 0.1,
                "overall_max_gap": 0.1,
            }
        },
        bootstrap_report={
            "gap_summary": {
                "pair_count": 3,
                "overall_median_gap": 0.15,
                "overall_mean_gap": 0.15,
                "overall_p90_gap": 0.18,
                "overall_max_gap": 0.2,
            }
        },
        repeat_report={
            "gap_summary": {
                "pair_count": 3,
                "overall_median_gap": 0.2,
                "overall_mean_gap": 0.2,
                "overall_p90_gap": 0.25,
                "overall_max_gap": 0.3,
            }
        },
        max_start_only_ratio=0.5,
        max_bootstrap_ratio=0.75,
        max_repeat_ratio=0.75,
    )

    assert report["status"] == "pass"
    assert report["failures"] == []
    assert report["controls"]["start_only"]["ratio_to_observed_median_gap"] < 0.1
    assert report["per_start_controls"][0]["status"] == "pass"


def test_build_control_suite_fails_when_start_only_gap_matches_observed() -> None:
    report = build_control_suite(
        observed_contrast=_contrast([1.0, 1.2, 0.8]),
        start_only_contrast={
            "gap_summary": {
                "pair_count": 3,
                "overall_median_gap": 0.9,
                "overall_mean_gap": 0.9,
                "overall_p90_gap": 1.0,
                "overall_max_gap": 1.0,
                "by_start": [
                    {
                        "start_name": "fixed_start_1",
                        "median_gap": 0.9,
                    }
                ],
            }
        },
        bootstrap_report=None,
        repeat_report=None,
        max_start_only_ratio=0.5,
        max_bootstrap_ratio=0.75,
        max_repeat_ratio=0.75,
    )

    assert report["status"] == "fail"
    assert "start_only_gap_too_close_to_narrative_gap" in report["failures"]
    assert "same_narrative_repeat_control_missing" in report["warnings"]
    assert "within_run_bootstrap_control_missing" in report["warnings"]
    assert report["per_start_controls"][0]["status"] == "fail"


def test_build_control_suite_fails_when_any_per_start_control_fails() -> None:
    observed = {
        "pairwise_contrasts": [
            {"start_name": "start_a", "standardized_l2_gap": 1.0},
            {"start_name": "start_b", "standardized_l2_gap": 1.0},
        ]
    }
    repeat = {
        "gap_summary": {
            "pair_count": 2,
            "overall_median_gap": 0.4,
            "overall_mean_gap": 0.4,
            "overall_p90_gap": 0.5,
            "overall_max_gap": 0.5,
            "by_start": [
                {"start_name": "start_a", "median_gap": 0.8},
                {"start_name": "start_b", "median_gap": 0.1},
            ],
        }
    }

    report = build_control_suite(
        observed_contrast=observed,
        start_only_contrast={
            "gap_summary": {
                "pair_count": 2,
                "overall_median_gap": 0.0,
                "overall_mean_gap": 0.0,
                "overall_p90_gap": 0.0,
                "overall_max_gap": 0.0,
                "by_start": [
                    {"start_name": "start_a", "median_gap": 0.0},
                    {"start_name": "start_b", "median_gap": 0.0},
                ],
            }
        },
        bootstrap_report=None,
        repeat_report=repeat,
        max_start_only_ratio=0.5,
        max_bootstrap_ratio=0.75,
        max_repeat_ratio=0.75,
    )

    assert report["status"] == "fail"
    assert "per_start_control_failure" in report["failures"]
    assert report["per_start_controls"][0]["status"] == "fail"


def _write_repeat_run(tmp_path: Path, name: str, spx_mean: float) -> str:
    path = tmp_path / f"{name}.json"
    payload = {
        "generation": {
            "terminal_delta_summary": [
                {
                    "market": "SPX",
                    "mean_terminal_delta": spx_mean,
                    "p10": spx_mean - 1.0,
                    "p50": spx_mean,
                    "p90": spx_mean + 1.0,
                }
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_repeat_gap_report_compares_same_case_different_seed(tmp_path: Path) -> None:
    repeat = {
        "rows": [
            {
                "case_name": "fragile",
                "start_name": "fixed_start_1",
                "candidate_index": 18,
                "repeat_seed": 1,
                "run_report": _write_repeat_run(tmp_path, "seed1", 1.0),
            },
            {
                "case_name": "fragile",
                "start_name": "fixed_start_1",
                "candidate_index": 18,
                "repeat_seed": 2,
                "run_report": _write_repeat_run(tmp_path, "seed2", 2.0),
            },
        ]
    }

    report = repeat_gap_report(repeat, markets=["SPX"])

    assert report["status"] == "pass"
    assert report["gap_summary"]["pair_count"] == 1
    assert report["pairwise_contrasts"][0]["left_seed"] == 1
    assert report["pairwise_contrasts"][0]["right_seed"] == 2


def test_bootstrap_gap_report_splits_saved_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        control_suite,
        "load_state_spec_names",
        lambda _checkpoint: [f"iv:{idx}" for idx in range(25)]
        + ["factor:spx", "factor:vix"],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "prefix_latent_story_smoke_report.json"
    report_path.write_text(
        json.dumps(
            {
                "variant_rows": [{"is_operational": True}],
                "generation": {"terminal_delta_summary": []},
            }
        ),
        encoding="utf-8",
    )
    samples = [
        [
            [[0.0] * 27 for _ in range(29)] + [[0.0] * 25 + [1.0, 0.0]],
            [[0.0] * 27 for _ in range(29)] + [[0.0] * 25 + [1.2, 0.0]],
            [[0.0] * 27 for _ in range(29)] + [[0.0] * 25 + [2.0, 0.0]],
            [[0.0] * 27 for _ in range(29)] + [[0.0] * 25 + [2.2, 0.0]],
        ]
    ]
    np.savez(
        run_dir / "prefix_latent_story_smoke_arrays.npz",
        samples=np.asarray(samples, dtype=np.float32),
    )

    report = bootstrap_gap_report(
        {
            "rows": [
                {
                    "case_name": "fragile",
                    "start_name": "s",
                    "candidate_index": 1,
                    "run_report": str(report_path),
                }
            ]
        },
        markets=["SPX"],
        checkpoint=tmp_path / "fake.pt",
    )

    assert report["status"] == "pass"
    assert report["gap_summary"]["pair_count"] == 1
