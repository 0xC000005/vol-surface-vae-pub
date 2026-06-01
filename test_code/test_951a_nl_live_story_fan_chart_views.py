import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_live_story_fan_chart_views import (
    build_fan_chart_view_data,
    plot_combined_fan_chart_views,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_case(tmp_path: Path, name: str, offset: float) -> tuple[Path, Path]:
    arrays_path = tmp_path / name / "arrays.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    requested_raw = np.asarray([[10.0, 20.0], [100.0, 200.0]], dtype=np.float32)
    moves = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
                [[3.0, 4.0], [5.0, 6.0]],
            ],
            [
                [[10.0 + offset, -5.0], [20.0 + offset, -10.0]],
                [[12.0 + offset, -7.0], [22.0 + offset, -12.0]],
                [[14.0 + offset, -9.0], [24.0 + offset, -14.0]],
            ],
        ],
        dtype=np.float32,
    )
    generated_states = requested_raw[:, None, None, :] + moves
    delta_scale = np.asarray([[2.0, 5.0], [4.0, 10.0]], dtype=np.float32)
    np.savez(
        arrays_path,
        requested_raw=requested_raw,
        samples=moves,
        generated_states=generated_states,
        delta_scale=delta_scale,
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


def test_fan_chart_view_data_builds_raw_delta_and_standardized_views(
    tmp_path: Path,
) -> None:
    arrays_a, report_a = _write_case(tmp_path, "case_a", offset=0.0)
    arrays_b, report_b = _write_case(tmp_path, "case_b", offset=10.0)
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "fragile_risk_on_rebound_22",
                    "prefix_arrays_snapshot_path": str(arrays_a),
                    "prefix_report_snapshot_path": str(report_a),
                },
                {
                    "case_name": "commodity_inflation_pressure_22",
                    "prefix_arrays_snapshot_path": str(arrays_b),
                    "prefix_report_snapshot_path": str(report_b),
                },
            ],
        },
    )

    view = build_fan_chart_view_data(
        summary,
        markets=[("Synthetic A", 0), ("Synthetic B", 1)],
    )

    assert view["fixed_start_index"] == 22
    assert [case["label"] for case in view["cases"]] == [
        "Fragile risk-on rebound",
        "Commodity inflation pressure",
    ]

    case_a = view["cases"][0]
    raw_a = case_a["views"]["raw_level"]["Synthetic A"]
    move_a = case_a["views"]["start_relative_move"]["Synthetic A"]
    std_a = case_a["views"]["delta_scale_standardized_move"]["Synthetic A"]
    assert raw_a["days"] == [0, 1, 2]
    assert raw_a["p50"] == pytest.approx([100.0, 112.0, 122.0])
    assert move_a["p50"] == pytest.approx([0.0, 12.0, 22.0])
    assert std_a["p50"] == pytest.approx([0.0, 6.0, 5.5])

    raw_b = view["cases"][1]["views"]["raw_level"]["Synthetic A"]
    assert raw_b["p50"] == pytest.approx([100.0, 122.0, 132.0])


def test_fan_chart_view_data_requires_generated_states(tmp_path: Path) -> None:
    arrays_path = tmp_path / "case" / "arrays.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arrays_path, requested_raw=np.zeros((1, 2), dtype=np.float32))
    report_path = _write_json(tmp_path / "case" / "report.json", {})
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "cases": [
                {
                    "case_name": "case",
                    "prefix_arrays_snapshot_path": str(arrays_path),
                    "prefix_report_snapshot_path": str(report_path),
                }
            ]
        },
    )

    with pytest.raises(KeyError, match="generated_states"):
        build_fan_chart_view_data(summary, markets=[("Synthetic A", 0)])


def test_plot_combined_fan_chart_views_writes_artifact(tmp_path: Path) -> None:
    arrays_a, report_a = _write_case(tmp_path, "case_a", offset=0.0)
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "fragile_risk_on_rebound_22",
                    "prefix_arrays_snapshot_path": str(arrays_a),
                    "prefix_report_snapshot_path": str(report_a),
                },
            ],
        },
    )
    view = build_fan_chart_view_data(summary, markets=[("Synthetic A", 0)])

    path = plot_combined_fan_chart_views(
        view,
        tmp_path / "combined.png",
        max_markets=1,
    )

    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
