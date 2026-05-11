import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_validation_gate import (
    DEFAULT_GATE_THRESHOLDS,
    case_rollout_shift_rows,
    evaluate_start_case_gates,
)


def test_case_rollout_shift_rows_compares_each_variant_to_original() -> None:
    samples = np.zeros((3, 2, 4, 2), dtype=np.float32)
    samples[1] += 0.25
    samples[2] += 1.00
    rows = [
        {"query_window_index": 10, "variant": "original"},
        {"query_window_index": 10, "variant": "nearest_train_start"},
        {"query_window_index": 10, "variant": "farthest_train_start"},
    ]
    scale = np.ones((4, 2), dtype=np.float32)

    shifts = case_rollout_shift_rows(samples=samples, variant_rows=rows, scale=scale)

    assert shifts[0]["mean_abs_delta_z"] == 0.0
    assert shifts[1]["mean_abs_delta_z"] > 0.0
    assert shifts[2]["mean_abs_delta_z"] > shifts[1]["mean_abs_delta_z"]


def test_evaluate_start_case_gates_flags_low_memory_and_far_start() -> None:
    decoded_memory = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.4358899],
            [0.7, 0.71414286],
        ],
        dtype=np.float32,
    )
    text_memory = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows = [
        {
            "query_window_index": 10,
            "start_window_index": 10,
            "variant": "original",
            "start_distance_z": 0.0,
        },
        {
            "query_window_index": 10,
            "start_window_index": 11,
            "variant": "nearest_train_start",
            "start_distance_z": 4.0,
        },
        {
            "query_window_index": 10,
            "start_window_index": 99,
            "variant": "farthest_train_start",
            "start_distance_z": 25.0,
        },
    ]
    shifts = [
        {"mean_abs_delta_z": 0.0, "terminal_mean_abs_delta_z": 0.0},
        {"mean_abs_delta_z": 0.2, "terminal_mean_abs_delta_z": 0.3},
        {"mean_abs_delta_z": 1.2, "terminal_mean_abs_delta_z": 1.4},
    ]

    report = evaluate_start_case_gates(
        variant_rows=rows,
        decoded_memory=decoded_memory,
        text_memory=text_memory,
        rollout_shifts=shifts,
        endpoint_max_abs_error=0.0,
        thresholds=DEFAULT_GATE_THRESHOLDS,
        hard_case_count=2,
    )

    assert report["overall_status"] == "warning"
    assert report["operational_status"] == "pass"
    assert report["stress_status"] == "warning"
    assert report["warning_counts"]["low_memory_compatibility"] == 1
    assert report["warning_counts"]["large_start_distance"] == 1
    assert report["warning_counts"]["large_rollout_shift"] == 1
    assert report["hard_cases"][0]["variant"] == "farthest_train_start"


def test_evaluate_start_case_gates_separates_diagnostic_and_selected_rows() -> None:
    decoded_memory = np.asarray(
        [
            [0.7, 0.71414286],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    text_memory = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    rows = [
        {
            "query_window_index": 10,
            "start_window_index": 10,
            "variant": "original",
            "case_role": "diagnostic_original_start",
            "is_operational": False,
            "start_distance_z": 0.0,
        },
        {
            "query_window_index": 10,
            "start_window_index": 12,
            "variant": "balanced_memory_start",
            "case_role": "operational_selected_start",
            "is_operational": True,
            "start_distance_z": 8.0,
        },
    ]
    shifts = [
        {"mean_abs_delta_z": 0.0, "terminal_mean_abs_delta_z": 0.0},
        {"mean_abs_delta_z": 0.2, "terminal_mean_abs_delta_z": 0.3},
    ]

    report = evaluate_start_case_gates(
        variant_rows=rows,
        decoded_memory=decoded_memory,
        text_memory=text_memory,
        rollout_shifts=shifts,
        endpoint_max_abs_error=0.0,
        thresholds=DEFAULT_GATE_THRESHOLDS,
    )

    assert report["overall_status"] == "warning"
    assert report["selected_start_status"] == "pass"
    assert report["operational_status"] == "pass"
    assert report["diagnostic_baseline_status"] == "warning"
    assert report["operational_case_count"] == 1
    assert report["diagnostic_case_count"] == 1
    assert report["cases"][0]["case_role"] == "diagnostic_original_start"
    assert report["cases"][1]["is_operational"] is True


def test_evaluate_start_case_gates_fails_on_endpoint_error() -> None:
    rows = [{"query_window_index": 1, "start_window_index": 1, "variant": "original"}]
    memory = np.ones((1, 2), dtype=np.float32)
    shifts = [{"mean_abs_delta_z": 0.0, "terminal_mean_abs_delta_z": 0.0}]

    report = evaluate_start_case_gates(
        variant_rows=rows,
        decoded_memory=memory,
        text_memory=memory,
        rollout_shifts=shifts,
        endpoint_max_abs_error=1e-2,
        thresholds=DEFAULT_GATE_THRESHOLDS,
    )

    assert report["overall_status"] == "fail"
    assert "endpoint_not_pinned" in report["hard_fail_reasons"]


def test_evaluate_start_case_gates_warns_on_direction_mismatch() -> None:
    rows = [
        {
            "query_window_index": 1,
            "start_window_index": 1,
            "variant": "explicit_start",
            "is_operational": True,
            "memory_prior_direction_status": "reject",
        }
    ]
    memory = np.ones((1, 2), dtype=np.float32)
    shifts = [{"mean_abs_delta_z": 0.0, "terminal_mean_abs_delta_z": 0.0}]

    report = evaluate_start_case_gates(
        variant_rows=rows,
        decoded_memory=memory,
        text_memory=memory,
        rollout_shifts=shifts,
        endpoint_max_abs_error=0.0,
        thresholds=DEFAULT_GATE_THRESHOLDS,
    )

    assert report["overall_status"] == "warning"
    assert report["operational_status"] == "warning"
    assert report["warning_counts"]["memory_prior_direction_reject"] == 1
    assert "memory_prior_direction_reject" not in report["fail_counts"]
    assert report["cases"][0]["memory_prior_direction_status"] == "reject"
    assert report["cases"][0]["warnings"] == ["memory_prior_direction_reject"]
    assert report["cases"][0]["failures"] == []
