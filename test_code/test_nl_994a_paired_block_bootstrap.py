"""Unit tests for the 994a paired moving-block bootstrap summarizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_994a_paired_block_bootstrap import (  # noqa: E402
    load_method_scores,
    moving_block_bootstrap_mean,
    paired_deltas,
    run_paired_block_bootstrap,
    summarize_paired_metric,
)


def _write_report(
    path: Path,
    *,
    windows: list[int],
    method_values: dict[str, dict[str, list[float]]],
) -> Path:
    rows = []
    for pos, window in enumerate(windows):
        methods = {
            method: {metric: values[metric][pos] for metric in values}
            for method, values in method_values.items()
        }
        rows.append(
            {
                "row_no": pos,
                "window_index": int(window),
                "methods": methods,
            }
        )
    payload = {"status": "ok", "window_scores": rows}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _synthetic_reports(tmp_path: Path, *, n: int = 90, shift: float = -0.05):
    rng = np.random.default_rng(123)
    windows = [4010 + 5 * i for i in range(n)]
    base = 0.5 + 0.05 * rng.standard_normal(n)
    noise = 0.005 * rng.standard_normal(n)
    metrics_a = {
        "ensemble_crps_z": base.tolist(),
        "energy_score_z": (base * 1.3).tolist(),
        "coverage_80": np.clip(0.7 + 0.05 * rng.standard_normal(n), 0, 1).tolist(),
    }
    metrics_b = {
        "ensemble_crps_z": (base + shift + noise).tolist(),
        "energy_score_z": (base * 1.3 + shift + noise).tolist(),
        "coverage_80": metrics_a["coverage_80"],
    }
    report_a = _write_report(
        tmp_path / "report_a.json",
        windows=windows,
        method_values={"narrative_generator_topk": metrics_a},
    )
    report_b = _write_report(
        tmp_path / "report_b.json",
        windows=windows,
        method_values={"narrative_generator_topk": metrics_b},
    )
    return report_a, report_b, shift, noise


def test_moving_block_bootstrap_deterministic_and_shaped():
    series = np.sin(np.arange(60) / 3.0)
    a = moving_block_bootstrap_mean(series, block_length=30, n_boot=500, seed=7)
    b = moving_block_bootstrap_mean(series, block_length=30, n_boot=500, seed=7)
    c = moving_block_bootstrap_mean(series, block_length=30, n_boot=500, seed=8)
    assert a.shape == (500,)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_moving_block_bootstrap_clamps_block_length():
    series = np.array([1.0, 2.0, 3.0])
    means = moving_block_bootstrap_mean(series, block_length=30, n_boot=50, seed=0)
    # block length clamps to n=3 -> only one block (the series itself)
    assert np.allclose(means, series.mean())


def test_constant_delta_gives_degenerate_ci():
    deltas = np.full(60, -0.05)
    summary = summarize_paired_metric(deltas, block_length=30, n_boot=200, seed=1)
    assert summary["mean_delta"] == pytest.approx(-0.05)
    assert summary["ci_low"] == pytest.approx(-0.05)
    assert summary["ci_high"] == pytest.approx(-0.05)
    assert summary["ci_excludes_zero"] is True
    assert summary["n_effective_blocks"] == pytest.approx(2.0)


def test_paired_bootstrap_recovers_known_shift(tmp_path):
    report_a, report_b, shift, noise = _synthetic_reports(tmp_path, shift=-0.05)
    payload = run_paired_block_bootstrap(
        report_a=report_a,
        report_b=report_b,
        method_a="narrative_generator_topk",
        method_b="narrative_generator_topk",
        metrics=("ensemble_crps_z", "energy_score_z"),
        block_length=30,
        n_boot=2000,
        seed=994,
    )
    for metric in ("ensemble_crps_z", "energy_score_z"):
        block = payload["metrics"][metric]
        assert block["n_windows"] == 90
        assert block["mean_delta"] == pytest.approx(shift + noise.mean(), abs=1e-9)
        # CI must bracket the empirical mean delta and sit near the true shift
        assert block["ci_low"] <= block["mean_delta"] <= block["ci_high"]
        assert block["ci_low"] == pytest.approx(shift, abs=0.005)
        assert block["ci_high"] == pytest.approx(shift, abs=0.005)
        assert block["ci_excludes_zero"] is True
        assert block["block_length_used"] == 30
        assert block["n_effective_blocks"] == pytest.approx(3.0)
        assert block["frac_boot_means_below_zero"] == pytest.approx(1.0)
    assert payload["shared_window_indices"][0] == 4010
    assert len(payload["shared_window_indices"]) == 90


def test_paired_deltas_uses_only_shared_windows(tmp_path):
    rng = np.random.default_rng(5)
    windows_a = [0, 5, 10, 15, 20]
    windows_b = [5, 15, 20, 25]
    values_a = {"ensemble_crps_z": rng.standard_normal(len(windows_a)).tolist()}
    values_b = {"ensemble_crps_z": rng.standard_normal(len(windows_b)).tolist()}
    report_a = _write_report(
        tmp_path / "a.json",
        windows=windows_a,
        method_values={"m": values_a},
    )
    report_b = _write_report(
        tmp_path / "b.json",
        windows=windows_b,
        method_values={"m": values_b},
    )
    scores_a = load_method_scores(report_a, method="m", metrics=("ensemble_crps_z",))
    scores_b = load_method_scores(report_b, method="m", metrics=("ensemble_crps_z",))
    deltas, shared = paired_deltas(scores_a, scores_b, metric="ensemble_crps_z")
    assert shared == [5, 15, 20]
    assert deltas.shape == (3,)
    expected = [
        scores_b[idx]["ensemble_crps_z"] - scores_a[idx]["ensemble_crps_z"]
        for idx in shared
    ]
    assert np.allclose(deltas, expected)


def test_missing_method_raises(tmp_path):
    report = _write_report(
        tmp_path / "r.json",
        windows=[0, 5],
        method_values={"m": {"ensemble_crps_z": [0.1, 0.2]}},
    )
    with pytest.raises(ValueError, match="lacks"):
        load_method_scores(report, method="other", metrics=("ensemble_crps_z",))
