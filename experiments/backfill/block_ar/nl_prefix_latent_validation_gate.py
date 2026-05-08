#!/usr/bin/env python
"""Validation gate for narrative prefix-latent start sensitivity reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GATE_THRESHOLDS = {
    "endpoint_abs_fail": 1e-6,
    "memory_cosine_warn": 0.80,
    "memory_cosine_fail": 0.65,
    "start_distance_warn": 15.0,
    "start_distance_fail": 32.0,
    "rollout_shift_warn": 1.0,
    "rollout_shift_fail": 2.0,
}


def _cosine_vector(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("left and right must have matching shape [N,D]")
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8)
    return (np.sum(a * b, axis=1) / denom).astype(np.float32)


def case_rollout_shift_rows(
    *,
    samples: np.ndarray,
    variant_rows: list[dict[str, Any]],
    scale: np.ndarray,
) -> list[dict[str, float]]:
    """Return per-case z-unit rollout shifts versus each query's original start."""

    sample_arr = np.asarray(samples, dtype=np.float32)
    scale_arr = np.maximum(np.asarray(scale, dtype=np.float32), 1e-8)
    if sample_arr.size == 0:
        return [
            {"mean_abs_delta_z": 0.0, "terminal_mean_abs_delta_z": 0.0}
            for _row in variant_rows
        ]
    if sample_arr.ndim != 4:
        raise ValueError("samples must have shape [K,S,T,C]")
    if len(variant_rows) != sample_arr.shape[0]:
        raise ValueError("variant_rows length must match samples")
    if scale_arr.shape != sample_arr.shape[2:]:
        raise ValueError("scale must have shape [T,C]")
    sample_z = sample_arr / scale_arr[None, None, :, :]
    original_by_query: dict[int, np.ndarray] = {}
    for idx, row in enumerate(variant_rows):
        if str(row.get("variant")) == "original":
            original_by_query[int(row["query_window_index"])] = sample_z[idx].mean(axis=0)
    shifts: list[dict[str, float]] = []
    for idx, row in enumerate(variant_rows):
        query_idx = int(row["query_window_index"])
        original = original_by_query.get(query_idx)
        if original is None or str(row.get("variant")) == "original":
            shifts.append({"mean_abs_delta_z": 0.0, "terminal_mean_abs_delta_z": 0.0})
            continue
        diff = sample_z[idx].mean(axis=0) - original
        shifts.append(
            {
                "mean_abs_delta_z": float(np.mean(np.abs(diff))),
                "terminal_mean_abs_delta_z": float(np.mean(np.abs(diff[-1]))),
            }
        )
    return shifts


def _status_from_flags(fail: bool, warning_count: int) -> str:
    if fail:
        return "fail"
    return "warning" if warning_count else "pass"


def _status_for_cases(cases: list[dict[str, Any]], hard_fail_reasons: list[str]) -> str:
    if hard_fail_reasons or any(case["failures"] for case in cases):
        return "fail"
    if any(case["warnings"] for case in cases):
        return "warning"
    return "pass"


def evaluate_start_case_gates(
    *,
    variant_rows: list[dict[str, Any]],
    decoded_memory: np.ndarray,
    text_memory: np.ndarray,
    rollout_shifts: list[dict[str, float]],
    endpoint_max_abs_error: float,
    thresholds: dict[str, float] | None = None,
    hard_case_count: int = 8,
) -> dict[str, Any]:
    """Evaluate per-case validation warnings and hard failures."""

    th = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds:
        th.update({key: float(value) for key, value in thresholds.items()})
    cos = _cosine_vector(decoded_memory, text_memory)
    if len(variant_rows) != len(cos) or len(rollout_shifts) != len(variant_rows):
        raise ValueError("variant rows, memories, and rollout shifts must align")
    hard_fail_reasons: list[str] = []
    if float(endpoint_max_abs_error) > float(th["endpoint_abs_fail"]):
        hard_fail_reasons.append("endpoint_not_pinned")
    cases: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    for idx, row in enumerate(variant_rows):
        warnings: list[str] = []
        failures: list[str] = []
        memory_cosine = float(cos[idx])
        start_distance = float(row.get("start_distance_z", 0.0))
        mean_shift = float(rollout_shifts[idx].get("mean_abs_delta_z", 0.0))
        terminal_shift = float(rollout_shifts[idx].get("terminal_mean_abs_delta_z", 0.0))
        if memory_cosine < float(th["memory_cosine_fail"]):
            failures.append("memory_compatibility_fail")
        elif memory_cosine < float(th["memory_cosine_warn"]):
            warnings.append("low_memory_compatibility")
        if start_distance > float(th["start_distance_fail"]):
            failures.append("start_distance_fail")
        elif start_distance > float(th["start_distance_warn"]):
            warnings.append("large_start_distance")
        if max(mean_shift, terminal_shift) > float(th["rollout_shift_fail"]):
            failures.append("rollout_shift_fail")
        elif max(mean_shift, terminal_shift) > float(th["rollout_shift_warn"]):
            warnings.append("large_rollout_shift")
        warning_counts.update(warnings)
        fail_counts.update(failures)
        risk_score = (
            max(0.0, float(th["memory_cosine_warn"]) - memory_cosine)
            + max(0.0, start_distance / max(float(th["start_distance_warn"]), 1e-8) - 1.0)
            + max(0.0, max(mean_shift, terminal_shift) / max(float(th["rollout_shift_warn"]), 1e-8) - 1.0)
            + 2.0 * len(failures)
        )
        cases.append(
            {
                "case_index": int(idx),
                "query_window_index": int(row.get("query_window_index", -1)),
                "start_window_index": int(row.get("start_window_index", -1)),
                "variant": str(row.get("variant", "")),
                "case_role": str(row.get("case_role", "")),
                "is_operational": bool(
                    row.get(
                        "is_operational",
                        str(row.get("variant", "")) != "farthest_train_start",
                    )
                ),
                "start_distance_z": start_distance,
                "input_memory_cosine": memory_cosine,
                "mean_abs_delta_z": mean_shift,
                "terminal_mean_abs_delta_z": terminal_shift,
                "warnings": warnings,
                "failures": failures,
                "status": _status_from_flags(bool(failures), len(warnings)),
                "risk_score": float(risk_score),
            }
        )
    overall_status = _status_for_cases(cases, hard_fail_reasons)
    operational_cases = [case for case in cases if bool(case.get("is_operational"))]
    diagnostic_cases = [case for case in cases if not bool(case.get("is_operational"))]
    stress_cases = [
        case for case in cases if str(case.get("variant")) == "farthest_train_start"
    ]
    hard_cases = sorted(cases, key=lambda item: float(item["risk_score"]), reverse=True)[
        : int(hard_case_count)
    ]
    return {
        "overall_status": overall_status,
        "operational_status": _status_for_cases(operational_cases, hard_fail_reasons),
        "selected_start_status": _status_for_cases(
            operational_cases,
            hard_fail_reasons,
        ),
        "diagnostic_baseline_status": _status_for_cases(diagnostic_cases, []),
        "stress_status": _status_for_cases(stress_cases, []),
        "thresholds": th,
        "endpoint_max_abs_error": float(endpoint_max_abs_error),
        "hard_fail_reasons": hard_fail_reasons,
        "warning_counts": dict(warning_counts),
        "fail_counts": dict(fail_counts),
        "case_count": int(len(cases)),
        "operational_case_count": int(len(operational_cases)),
        "diagnostic_case_count": int(len(diagnostic_cases)),
        "status_counts": dict(Counter(case["status"] for case in cases)),
        "hard_cases": hard_cases,
        "cases": cases,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_validation_gate(args: argparse.Namespace) -> dict[str, Any]:
    report = _load_json(args.start_sensitivity_report)
    with np.load(args.start_sensitivity_arrays) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if "delta_scale" in arrays:
        delta_scale = arrays["delta_scale"]
    else:
        samples = arrays.get("samples", np.empty((0,), dtype=np.float32))
        if samples.ndim != 4:
            raise ValueError("arrays must include delta_scale or 4-D samples")
        delta_scale = np.ones(samples.shape[2:], dtype=np.float32)
    rollout_shifts = case_rollout_shift_rows(
        samples=arrays.get("samples", np.empty((0,), dtype=np.float32)),
        variant_rows=report["variant_rows"],
        scale=delta_scale,
    )
    gate = evaluate_start_case_gates(
        variant_rows=report["variant_rows"],
        decoded_memory=arrays["decoded_memory"],
        text_memory=arrays["text_memory"],
        rollout_shifts=rollout_shifts,
        endpoint_max_abs_error=float(report["endpoint_alignment"]["max_abs_error"]),
        hard_case_count=int(args.hard_case_count),
    )
    output = {
        "status": "ok",
        "scope_note": (
            "Validation gate over cached start-sensitivity artifacts. No OpenAI API "
            "calls are made."
        ),
        "start_sensitivity_report": str(args.start_sensitivity_report),
        "start_sensitivity_arrays": str(args.start_sensitivity_arrays),
        "gate": gate,
        "artifact_paths": {
            "report": str(Path(args.output_dir) / "prefix_latent_validation_gate_report.json")
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "prefix_latent_validation_gate_report.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-sensitivity-report",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_start_sensitivity_fullheldout_789b/"
            "prefix_latent_start_sensitivity_report.json"
        ),
    )
    parser.add_argument(
        "--start-sensitivity-arrays",
        default=(
            "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
            "prefix_latent_start_sensitivity_fullheldout_789b/"
            "prefix_latent_start_sensitivity_arrays.npz"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-case-count", type=int, default=8)
    args = parser.parse_args()
    output = run_validation_gate(args)
    print(
        json.dumps(
            {
                "report": output["artifact_paths"]["report"],
                "overall_status": output["gate"]["overall_status"],
                "operational_status": output["gate"]["operational_status"],
                "stress_status": output["gate"]["stress_status"],
                "warning_counts": output["gate"]["warning_counts"],
                "fail_counts": output["gate"]["fail_counts"],
                "hard_fail_reasons": output["gate"]["hard_fail_reasons"],
                "hard_cases": output["gate"]["hard_cases"][:3],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
