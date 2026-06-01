import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_component_aware_scenario_view_audit import (
    build_component_aware_audit,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_case(
    root: Path,
    name: str,
    *,
    weights: list[float],
    terminal_offsets: list[float],
) -> tuple[Path, Path]:
    sample_count = len(weights) * 2
    horizon = 3
    width = 39
    samples = np.zeros((2, sample_count, horizon, width), dtype=np.float32)
    generated = np.zeros_like(samples)
    requested_raw = np.zeros((2, width), dtype=np.float32)
    requested_raw[:, 0] = 100.0
    requested_raw[:, 1] = 20.0
    requested_raw[:, 2] = 5.0
    counts = [2 for _ in weights]
    cursor = 0
    for offset, count in zip(terminal_offsets, counts, strict=True):
        stop = cursor + count
        path = np.linspace(0.0, offset, horizon, dtype=np.float32)
        samples[1, cursor:stop, :, 25] = path
        samples[1, cursor:stop, :, 38] = -path
        generated[1, cursor:stop, :, :] = requested_raw[1][None, None, :]
        generated[1, cursor:stop, :, 25] += samples[1, cursor:stop, :, 25]
        generated[1, cursor:stop, :, 38] += samples[1, cursor:stop, :, 38]
        cursor = stop
    arrays_path = root / name / "arrays.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_path,
        samples=samples,
        generated_states=generated,
        delta_scale=np.ones((horizon, width), dtype=np.float32),
        requested_raw=requested_raw,
        rollout_component_variant_index=np.asarray([1 for _ in weights], dtype=np.int64),
        rollout_component_window_index=np.asarray(
            [10 + idx for idx in range(len(weights))], dtype=np.int64
        ),
        rollout_component_weight=np.asarray(weights, dtype=np.float32),
        rollout_component_sample_count=np.asarray(counts, dtype=np.int64),
    )
    report_path = _write_json(
        root / name / "report.json",
        {"selected_start_state": {"variant_index": 1}},
    )
    return arrays_path, report_path


def test_component_aware_audit_quantifies_pooling_loss(tmp_path: Path) -> None:
    root = tmp_path / "deck"
    start_dir = root / "start_22"
    arrays_a, report_a = _write_case(
        start_dir,
        "case_a",
        weights=[0.55, 0.30, 0.15],
        terminal_offsets=[1.0, 5.0, 9.0],
    )
    arrays_b, report_b = _write_case(
        start_dir,
        "case_b",
        weights=[0.55, 0.30, 0.15],
        terminal_offsets=[-1.0, -5.0, -9.0],
    )
    _write_json(
        start_dir / "gradio_live_api_casebook_summary.json",
        {
            "fixed_start_index": 22,
            "cases": [
                {
                    "case_name": "risk_on_22",
                    "prefix_arrays_snapshot_path": str(arrays_a),
                    "prefix_report_snapshot_path": str(report_a),
                    "support_top_candidates": [{"window_id": "a"}],
                },
                {
                    "case_name": "risk_off_22",
                    "prefix_arrays_snapshot_path": str(arrays_b),
                    "prefix_report_snapshot_path": str(report_b),
                    "support_top_candidates": [{"window_id": "b"}],
                },
            ],
        },
    )

    report = build_component_aware_audit(
        root=root,
        output_dir=tmp_path / "out",
        primary_start=22,
        quality_report=tmp_path / "missing_quality.json",
        max_plot_markets=1,
    )

    assert report["start_count"] == 1
    assert report["start_reports"][0]["case_count"] == 2
    assert report["start_reports"][0]["support_jaccard_max"] == 0.0
    assert report["aggregate"]["mean_component_to_pooled_path_energy_ratio"] is not None
    assert report["diagnosis"]["status"] == "pooling_dampens_visible_conditionality"
    top1 = report["aggregate"]["sparse_policy_summaries"][0]
    assert top1["name"] == "top1_component"
    assert top1["mean_kept_original_weight"] == pytest.approx(0.55)
    assert top1["median_sparse_vs_full_terminal_range_ratio"] is not None
