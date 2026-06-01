import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (
    build_component_pooling_diagnostic,
    component_slices_for_variant,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_case(
    tmp_path: Path,
    name: str,
    *,
    component_offsets: list[float],
) -> tuple[Path, Path]:
    sample_count = 4
    width = 3
    horizon = 2
    samples = np.zeros((2, sample_count, horizon, width), dtype=np.float32)
    for component_no, offset in enumerate(component_offsets):
        start = component_no * 2
        samples[1, start : start + 2, :, 0] = offset
        samples[1, start : start + 2, :, 1] = offset * 2
    generated_states = samples.copy()
    arrays_path = tmp_path / name / "arrays.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_path,
        samples=samples,
        generated_states=generated_states,
        delta_scale=np.ones((horizon, width), dtype=np.float32),
        requested_raw=np.zeros((2, width), dtype=np.float32),
        rollout_component_variant_index=np.asarray([1, 1], dtype=np.int64),
        rollout_component_window_index=np.asarray([10, 20], dtype=np.int64),
        rollout_component_weight=np.asarray([0.7, 0.3], dtype=np.float32),
        rollout_component_sample_count=np.asarray([2, 2], dtype=np.int64),
    )
    report_path = _write_json(
        tmp_path / name / "report.json",
        {
            "generation": {
                "narrative_ensemble_calibration": {
                    "operational_variant_index": 1,
                }
            }
        },
    )
    return arrays_path, report_path


def test_component_slices_for_variant_reconstructs_sample_ranges() -> None:
    variant = np.asarray([0, 1, 1], dtype=np.int64)
    windows = np.asarray([3, 4, 5], dtype=np.int64)
    weights = np.asarray([1.0, 0.6, 0.4], dtype=np.float32)
    counts = np.asarray([1, 2, 3], dtype=np.int64)

    rows = component_slices_for_variant(
        variant_index=1,
        component_variant_index=variant,
        component_window_index=windows,
        component_weight=weights,
        component_sample_count=counts,
        sample_count=5,
    )

    assert [(row["window_index"], row["sample_slice"]) for row in rows] == [
        (4, [0, 2]),
        (5, [2, 5]),
    ]
    assert [row["weight"] for row in rows] == pytest.approx([0.6, 0.4])


def test_component_slices_rejects_bad_count_sum() -> None:
    with pytest.raises(ValueError, match="component sample counts sum"):
        component_slices_for_variant(
            variant_index=1,
            component_variant_index=np.asarray([1], dtype=np.int64),
            component_window_index=np.asarray([4], dtype=np.int64),
            component_weight=np.asarray([1.0], dtype=np.float32),
            component_sample_count=np.asarray([3], dtype=np.int64),
            sample_count=5,
        )


def test_component_pooling_diagnostic_detects_component_separation(
    tmp_path: Path,
) -> None:
    arrays_a, report_a = _write_case(tmp_path, "case_a", component_offsets=[1.0, 3.0])
    arrays_b, report_b = _write_case(tmp_path, "case_b", component_offsets=[10.0, 12.0])
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "risk_on_22",
                    "prefix_arrays_snapshot_path": str(arrays_a),
                    "prefix_report_snapshot_path": str(report_a),
                    "support_top_candidates": [{"window_id": "a"}, {"window_id": "b"}],
                },
                {
                    "case_name": "risk_off_22",
                    "prefix_arrays_snapshot_path": str(arrays_b),
                    "prefix_report_snapshot_path": str(report_b),
                    "support_top_candidates": [{"window_id": "c"}, {"window_id": "d"}],
                },
            ],
        },
    )

    report = build_component_pooling_diagnostic(
        summary,
        markets=[("A", 0), ("B", 1)],
    )

    assert report["case_count"] == 2
    assert report["support_jaccard"]["max"] == 0.0
    assert report["component_count"] == 4
    assert report["pooling_diagnosis"]["terminal_median_component_vs_pooled_ratio"]["A"] > 1.0
    assert report["pooled_cross_narrative"]["path_energy_median"] > 0.0
    assert len(report["within_narrative_component"]["path_energy_values"]) == 2
