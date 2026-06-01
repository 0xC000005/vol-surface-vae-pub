import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_sparse_component_family_view import (
    build_sparse_component_family_view,
    select_sparse_components,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_case(
    tmp_path: Path,
    name: str,
    *,
    weights: list[float],
    terminal_offsets: list[float],
) -> tuple[Path, Path]:
    sample_count = 6
    horizon = 3
    width = 2
    samples = np.zeros((2, sample_count, horizon, width), dtype=np.float32)
    generated_states = np.zeros_like(samples)
    requested_raw = np.asarray([[100.0, 20.0], [100.0, 20.0]], dtype=np.float32)
    cursor = 0
    counts = [2, 2, 2]
    for offset, count in zip(terminal_offsets, counts, strict=True):
        stop = cursor + count
        path = np.linspace(0.0, offset, horizon, dtype=np.float32)
        samples[1, cursor:stop, :, 0] = path
        samples[1, cursor:stop, :, 1] = -path
        generated_states[1, cursor:stop, :, :] = requested_raw[1][None, None, :]
        generated_states[1, cursor:stop, :, :] += samples[1, cursor:stop, :, :]
        cursor = stop

    arrays_path = tmp_path / name / "arrays.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_path,
        samples=samples,
        generated_states=generated_states,
        delta_scale=np.ones((horizon, width), dtype=np.float32),
        requested_raw=requested_raw,
        rollout_component_variant_index=np.asarray([1, 1, 1], dtype=np.int64),
        rollout_component_window_index=np.asarray([10, 20, 30], dtype=np.int64),
        rollout_component_weight=np.asarray(weights, dtype=np.float32),
        rollout_component_sample_count=np.asarray(counts, dtype=np.int64),
    )
    report_path = _write_json(
        tmp_path / name / "report.json",
        {"selected_start_state": {"variant_index": 1}},
    )
    return arrays_path, report_path


def test_select_sparse_components_keeps_top_until_coverage() -> None:
    components = [
        {"component_no": 0, "weight": 0.50},
        {"component_no": 1, "weight": 0.25},
        {"component_no": 2, "weight": 0.20},
        {"component_no": 3, "weight": 0.05},
    ]

    selected = select_sparse_components(
        components,
        max_components=3,
        min_cumulative_weight=0.70,
    )

    assert [item["component_no"] for item in selected] == [0, 1]
    assert [item["sparse_weight"] for item in selected] == pytest.approx(
        [2.0 / 3.0, 1.0 / 3.0]
    )


def test_sparse_component_family_view_keeps_component_families(
    tmp_path: Path,
) -> None:
    arrays_a, report_a = _write_case(
        tmp_path,
        "case_a",
        weights=[0.60, 0.25, 0.15],
        terminal_offsets=[1.0, 5.0, 9.0],
    )
    arrays_b, report_b = _write_case(
        tmp_path,
        "case_b",
        weights=[0.55, 0.30, 0.15],
        terminal_offsets=[-1.0, -5.0, -9.0],
    )
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "risk_on_22",
                    "prefix_arrays_snapshot_path": str(arrays_a),
                    "prefix_report_snapshot_path": str(report_a),
                },
                {
                    "case_name": "risk_off_22",
                    "prefix_arrays_snapshot_path": str(arrays_b),
                    "prefix_report_snapshot_path": str(report_b),
                },
            ],
        },
    )

    report = build_sparse_component_family_view(
        summary,
        markets=[("A", 0), ("B", 1)],
        max_components=2,
        min_cumulative_weight=0.80,
    )

    assert report["case_count"] == 2
    assert report["total_selected_components"] == 4
    assert report["cases"][0]["kept_original_weight"] == pytest.approx(0.85)
    assert [item["window_index"] for item in report["cases"][0]["selected_components"]] == [
        10,
        20,
    ]
    assert report["cases"][0]["full_pooled_terminal_standardized"]["A"]["p50"] == pytest.approx(5.0)
    assert report["cases"][0]["sparse_pooled_terminal_standardized"]["A"]["p50"] == pytest.approx(3.0)
    assert report["cases"][1]["sparse_pooled_terminal_standardized"]["A"]["p50"] == pytest.approx(-3.0)
    assert report["terminal_p50_range_comparison"]["A"]["full_pooled"] == pytest.approx(10.0)
    assert report["terminal_p50_range_comparison"]["A"]["sparse_pooled"] == pytest.approx(6.0)
    assert report["terminal_p50_range_comparison"]["A"]["selected_component"] == pytest.approx(10.0)
