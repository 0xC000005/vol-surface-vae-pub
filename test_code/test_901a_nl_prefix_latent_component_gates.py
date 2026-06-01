import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_backtest_calibration import (
    coverage_matrix,
    pit_matrix,
)
from experiments.backfill.block_ar.nl_prefix_latent_component_fixed_start_controls import (
    _case_dir,
    _condition_report_for_case_name,
    _gap_summary,
)
from experiments.backfill.block_ar.nl_prefix_latent_component_global_calibration import (
    scale_samples_around_mean,
    split_calibration_rows,
)
from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_shape_audit import (
    _case_dir as _shape_case_dir,
    empirical_ks,
    path_energy_distance,
    path_event_metrics,
    path_distribution_market_metrics,
    quantile_shape_l2,
    standardize_1d,
    terminal_market_metrics,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (
    scale_delta_samples_around_mean,
)


def test_coverage_matrix_uses_central_80_band() -> None:
    samples = np.asarray(
        [
            [[0.0, 10.0], [0.0, 10.0]],
            [[1.0, 11.0], [1.0, 11.0]],
            [[2.0, 12.0], [2.0, 12.0]],
            [[3.0, 13.0], [3.0, 13.0]],
            [[4.0, 14.0], [4.0, 14.0]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([[2.0, 99.0], [2.0, 12.0]], dtype=np.float32)

    covered = coverage_matrix(samples, target)

    np.testing.assert_array_equal(
        covered,
        np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )


def test_pit_matrix_places_low_and_high_targets_near_edges() -> None:
    samples = np.asarray(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[3.0], [3.0]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([[-1.0], [4.0]], dtype=np.float32)

    pit = pit_matrix(samples, target)

    assert pit[0, 0] < 0.2
    assert pit[1, 0] > 0.8


def test_gap_summary_reports_median_and_tail() -> None:
    rows = [
        {"standardized_l2_gap": 0.1},
        {"standardized_l2_gap": 0.3},
        {"standardized_l2_gap": 0.9},
    ]

    summary = _gap_summary(rows)

    assert summary["pair_count"] == 3
    assert summary["median_gap"] == 0.3
    assert summary["max_gap"] == 0.9


def test_component_control_case_dir_uses_case_start_suffix() -> None:
    root = Path("root")

    assert _case_dir(root, "fragile_risk_on_start22", "variant") == (
        root / "fragile_risk_on_start22" / "fixed_start_22" / "variant"
    )


def test_component_control_condition_report_reuses_base_narrative_for_new_start():
    assert _condition_report_for_case_name("fragile_risk_on_start22") == (
        _condition_report_for_case_name("fragile_risk_on_start18")
    )


def test_shape_audit_case_dir_uses_case_start_suffix() -> None:
    root = Path("root")

    assert _shape_case_dir(root, "defensive_risk_off_start77", "variant") == (
        root / "defensive_risk_off_start77" / "fixed_start_77" / "variant"
    )


def test_scale_samples_around_mean_preserves_mean_and_scales_spread() -> None:
    samples = np.asarray(
        [
            [[0.0], [2.0]],
            [[2.0], [4.0]],
        ],
        dtype=np.float32,
    )

    scaled = scale_samples_around_mean(samples, 2.0)

    np.testing.assert_allclose(scaled.mean(axis=0), samples.mean(axis=0))
    np.testing.assert_allclose(scaled[:, :, 0], np.asarray([[-1.0, 1.0], [3.0, 5.0]]))


def test_scale_delta_samples_around_mean_preserves_variant_means() -> None:
    samples = np.asarray(
        [
            [
                [[0.0], [2.0]],
                [[2.0], [4.0]],
            ],
            [
                [[10.0], [12.0]],
                [[14.0], [16.0]],
            ],
        ],
        dtype=np.float32,
    )

    scaled = scale_delta_samples_around_mean(samples, 1.5)

    np.testing.assert_allclose(scaled.mean(axis=1), samples.mean(axis=1))
    assert float(np.std(scaled[0])) > float(np.std(samples[0]))


def test_split_calibration_rows_supports_reverse_and_even_odd() -> None:
    rows = [{"i": i} for i in range(6)]

    cal, eval_rows = split_calibration_rows(
        rows,
        calibration_count=2,
        split_mode="reverse",
    )
    assert [row["i"] for row in cal] == [4, 5]
    assert [row["i"] for row in eval_rows] == [0, 1, 2, 3]

    cal, eval_rows = split_calibration_rows(
        rows,
        calibration_count=2,
        split_mode="even_odd",
    )
    assert [row["i"] for row in cal] == [0, 2, 4]
    assert [row["i"] for row in eval_rows] == [1, 3, 5]


def test_shape_audit_standardizes_terminal_distributions() -> None:
    left = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    shifted_scaled = np.asarray([10.0, 12.0, 14.0, 16.0], dtype=np.float32)

    np.testing.assert_allclose(
        standardize_1d(left),
        standardize_1d(shifted_scaled),
    )
    assert empirical_ks(standardize_1d(left), standardize_1d(shifted_scaled)) == 0.0
    assert quantile_shape_l2(left, shifted_scaled) == 0.0


def test_terminal_market_metrics_separate_width_and_shape() -> None:
    left = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    wider_same_shape = np.asarray([-1.5, 0.5, 2.5, 4.5], dtype=np.float32)

    metrics = terminal_market_metrics(left, wider_same_shape, start_level=0.0)

    assert metrics["std_log_ratio_abs"] > 0.0
    assert metrics["quantile_shape_l2"] == 0.0
    assert metrics["standardized_ks"] == 0.0


def test_path_distribution_metrics_detect_path_family_changes() -> None:
    base = np.zeros((2, 3, 39), dtype=np.float32)
    stressed = np.zeros((2, 3, 39), dtype=np.float32)
    base[:, :, 25] = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]])
    stressed[:, :, 25] = np.asarray([[0.0, -1.0, -2.0], [0.0, -1.5, -3.0]])
    left = {
        "case_name": "risk_on",
        "label": "risk_on",
        "states": base,
        "start": np.zeros(39, dtype=np.float32),
    }
    right = {
        "case_name": "stress",
        "label": "stress",
        "states": stressed,
        "start": np.zeros(39, dtype=np.float32),
    }

    energy = path_energy_distance(left, right)
    rows = path_distribution_market_metrics(left, right)
    events = path_event_metrics(base[:, :, 25], stressed[:, :, 25])

    assert energy > 0.0
    assert rows[0]["path_wasserstein_z_mean"] > 0.0
    assert events["path_drawdown_prob_gap_1sigma"] > 0.0
