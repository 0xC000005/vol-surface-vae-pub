#!/usr/bin/env python
"""Analyze per-window deltas for a support policy versus equal support.

This diagnostic consumes completed scenario evaluations. It does not call the
generator. It is meant to answer whether a deployable support-policy feature can
gate a candidate policy: use the candidate when it improves portfolio response,
fall back to equal support when it would hurt broad scenario quality.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_support_policy_testflight import (  # noqa: E402
    PORTFOLIO_LABEL_METRIC,
    RELIABLE_BOOKS,
    _load_npz,
    _narrative_array_key,
    _selected_books,
    portfolio_candidate_scores,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_policy_delta_analysis_923f"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _pearson(left: list[float], right: list[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return None
    aa = a[mask] - float(np.mean(a[mask]))
    bb = b[mask] - float(np.mean(b[mask]))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return float(np.dot(aa, bb) / denom)


def _row_lookup(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in report.get("window_scores", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            rows[int(row["window_index"])] = row
    return rows


def _bridge_row_lookup(bridge_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in bridge_report.get("evaluation", {}).get("heldout_examples", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            rows[int(row["window_index"])] = row
    return rows


def _portfolio_score_for_row(
    *,
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    books: list[dict[str, Any]],
) -> float:
    key = _narrative_array_key(row, arrays)
    return float(
        portfolio_candidate_scores(
            samples=np.asarray(arrays[key], dtype=np.float32),
            target=np.asarray(arrays["future_delta"], dtype=np.float32)[
                int(row["block_window_index"])
            ],
            delta_scale=np.asarray(arrays["delta_scale"], dtype=np.float32),
            books=books,
        )[PORTFOLIO_LABEL_METRIC]
    )


def _entropy(weights: list[float]) -> float:
    arr = np.asarray(weights, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 0.0
    arr = arr / float(np.sum(arr))
    return float(-np.sum(arr * np.log(arr)))


def _policy_features(row: dict[str, Any]) -> dict[str, float]:
    policy = row.get("support_policy", {}) if isinstance(row, dict) else {}
    weights = [float(x) for x in policy.get("selected_support_weights", [])]
    probs = [
        float(item.get("probability", 0.0))
        for item in policy.get("candidate_probabilities", [])
        if isinstance(item, dict)
    ]
    scores = [
        float(item.get("score", 0.0))
        for item in policy.get("candidate_probabilities", [])
        if isinstance(item, dict)
    ]
    if not weights:
        weights = [1.0]
    weight_arr = np.asarray(weights, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64) if scores else np.asarray([0.0])
    return {
        "support_weight_max": float(np.max(weight_arr)),
        "support_weight_min": float(np.min(weight_arr)),
        "support_weight_entropy": _entropy(weights),
        "support_effective_n": float(1.0 / np.sum((weight_arr / weight_arr.sum()) ** 2)),
        "candidate_probability_entropy": _entropy(probs),
        "candidate_score_spread": float(np.max(score_arr) - np.min(score_arr)),
    }


def _method(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("methods", {}).get("narrative_generator_topk", {})


def _mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def _gate_report(
    *,
    rows: list[dict[str, Any]],
    feature: str,
    threshold: float,
    direction: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        value = float(row["policy_features"][feature])
        active = value >= threshold if direction == "ge" else value <= threshold
        selected.append(row if active else {**row, "use_equal": True})
    active_count = sum(0 if row.get("use_equal") else 1 for row in selected)

    def choose(row: dict[str, Any], key: str) -> float:
        if row.get("use_equal"):
            return float(row[f"equal_{key}"])
        return float(row[f"candidate_{key}"])

    return {
        "feature": feature,
        "threshold": float(threshold),
        "direction": direction,
        "active_count": int(active_count),
        "coverage_80_delta": _mean(
            [choose(row, "coverage_80") - float(row["equal_coverage_80"]) for row in selected]
        ),
        "crps_delta": _mean(
            [choose(row, "crps") - float(row["equal_crps"]) for row in selected]
        ),
        "energy_delta": _mean(
            [choose(row, "energy") - float(row["equal_energy"]) for row in selected]
        ),
        "portfolio_delta": _mean(
            [
                choose(row, "portfolio_path_score")
                - float(row["equal_portfolio_path_score"])
                for row in selected
            ]
        ),
    }


def delta_analysis_report(
    *,
    equal_report: dict[str, Any],
    candidate_report: dict[str, Any],
    equal_arrays: dict[str, np.ndarray],
    candidate_arrays: dict[str, np.ndarray],
    candidate_bridge: dict[str, Any],
    book_names: tuple[str, ...] | list[str] = RELIABLE_BOOKS,
) -> dict[str, Any]:
    books = _selected_books(book_names)
    equal_rows = _row_lookup(equal_report)
    candidate_rows = _row_lookup(candidate_report)
    bridge_rows = _bridge_row_lookup(candidate_bridge)
    common = sorted(set(equal_rows).intersection(candidate_rows))
    rows: list[dict[str, Any]] = []
    for window_index in common:
        equal = equal_rows[window_index]
        candidate = candidate_rows[window_index]
        bridge = bridge_rows.get(window_index, {})
        em = _method(equal)
        cm = _method(candidate)
        equal_port = _portfolio_score_for_row(row=equal, arrays=equal_arrays, books=books)
        candidate_port = _portfolio_score_for_row(
            row=candidate,
            arrays=candidate_arrays,
            books=books,
        )
        rows.append(
            {
                "window_index": int(window_index),
                "window_id": str(candidate.get("window_id", "")),
                "equal_crps": float(em["ensemble_crps_z"]),
                "candidate_crps": float(cm["ensemble_crps_z"]),
                "crps_delta": float(cm["ensemble_crps_z"] - em["ensemble_crps_z"]),
                "equal_energy": float(em["energy_score_z"]),
                "candidate_energy": float(cm["energy_score_z"]),
                "energy_delta": float(cm["energy_score_z"] - em["energy_score_z"]),
                "equal_coverage_80": float(em.get("coverage_80") or 0.0),
                "candidate_coverage_80": float(cm.get("coverage_80") or 0.0),
                "coverage_80_delta": float(
                    (cm.get("coverage_80") or 0.0) - (em.get("coverage_80") or 0.0)
                ),
                "equal_portfolio_path_score": equal_port,
                "candidate_portfolio_path_score": candidate_port,
                "portfolio_delta": float(candidate_port - equal_port),
                "policy_features": _policy_features(bridge),
            }
        )
    if not rows:
        raise ValueError("no matched window rows")

    feature_names = sorted(rows[0]["policy_features"])
    correlations = {
        name: {
            "with_portfolio_delta": _pearson(
                [row["policy_features"][name] for row in rows],
                [row["portfolio_delta"] for row in rows],
            ),
            "with_energy_delta": _pearson(
                [row["policy_features"][name] for row in rows],
                [row["energy_delta"] for row in rows],
            ),
            "with_crps_delta": _pearson(
                [row["policy_features"][name] for row in rows],
                [row["crps_delta"] for row in rows],
            ),
        }
        for name in feature_names
    }
    gates: list[dict[str, Any]] = []
    for name in feature_names:
        values = np.asarray([row["policy_features"][name] for row in rows], dtype=np.float64)
        for quantile in [0.25, 0.50, 0.75]:
            threshold = float(np.quantile(values, quantile))
            for direction in ["ge", "le"]:
                gate = _gate_report(
                    rows=rows,
                    feature=name,
                    threshold=threshold,
                    direction=direction,
                )
                gate["quantile"] = float(quantile)
                gates.append(gate)
    gates.sort(
        key=lambda item: (
            1 if item["portfolio_delta"] is not None and item["portfolio_delta"] < 0 else 0,
            -abs(float(item["energy_delta"] or 0.0)),
            -abs(float(item["crps_delta"] or 0.0)),
        ),
        reverse=True,
    )
    viable = [
        item
        for item in gates
        if item["portfolio_delta"] is not None
        and item["portfolio_delta"] < 0.0
        and item["crps_delta"] is not None
        and item["crps_delta"] <= 0.0
        and item["energy_delta"] is not None
        and item["energy_delta"] <= 0.0
    ]
    return {
        "status": "ok",
        "research_lane": "candidate",
        "result_status": "deployable_gate_found" if viable else "no_simple_gate_found",
        "benchmark_floor_status": "not_tested",
        "row_count": int(len(rows)),
        "aggregate_delta": {
            "crps_delta": _mean([row["crps_delta"] for row in rows]),
            "energy_delta": _mean([row["energy_delta"] for row in rows]),
            "coverage_80_delta": _mean([row["coverage_80_delta"] for row in rows]),
            "portfolio_delta": _mean([row["portfolio_delta"] for row in rows]),
        },
        "feature_correlations": correlations,
        "best_simple_gates": gates[:10],
        "viable_simple_gates": viable[:10],
        "window_rows": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Portfolio-Response Policy Delta Analysis",
        "",
        f"Status: `{report['result_status']}`",
        "",
        "## Aggregate Delta",
        "",
    ]
    for key, value in report["aggregate_delta"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Viable Simple Gates", ""])
    if not report["viable_simple_gates"]:
        lines.append("No simple deployable threshold gate improved portfolio, CRPS, and energy together.")
    else:
        for gate in report["viable_simple_gates"][:5]:
            lines.append(
                "- "
                f"{gate['feature']} {gate['direction']} {gate['threshold']}: "
                f"portfolio {gate['portfolio_delta']}, CRPS {gate['crps_delta']}, "
                f"energy {gate['energy_delta']}, active {gate['active_count']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equal-report", required=True)
    parser.add_argument("--equal-arrays", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--candidate-arrays", required=True)
    parser.add_argument("--candidate-bridge", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--book-names", default=",".join(RELIABLE_BOOKS))
    args = parser.parse_args()

    report = delta_analysis_report(
        equal_report=_load_json(args.equal_report),
        candidate_report=_load_json(args.candidate_report),
        equal_arrays=_load_npz(args.equal_arrays),
        candidate_arrays=_load_npz(args.candidate_arrays),
        candidate_bridge=_load_json(args.candidate_bridge),
        book_names=tuple(
            item.strip() for item in str(args.book_names).split(",") if item.strip()
        ),
    )
    report["artifact_paths"] = {
        "equal_report": str(args.equal_report),
        "candidate_report": str(args.candidate_report),
        "candidate_bridge": str(args.candidate_bridge),
        "report": str(args.output_dir / "portfolio_response_policy_delta_analysis.json"),
        "markdown": str(args.output_dir / "portfolio_response_policy_delta_analysis.md"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "portfolio_response_policy_delta_analysis.json", report)
    _write_text(args.output_dir / "portfolio_response_policy_delta_analysis.md", _markdown(report))
    print(
        json.dumps(
            {
                "status": report["result_status"],
                "report": report["artifact_paths"]["report"],
                "aggregate_delta": report["aggregate_delta"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
