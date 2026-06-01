#!/usr/bin/env python
"""Portfolio-response-aware support-policy TestFlight for NL scenarios.

This candidate keeps the support-grounded contract intact:

1. use the existing narrative/start candidate support mixtures;
2. label candidate mixtures by portfolio-risk response quality on historical
   backtests;
3. train the existing compact kernel-listwise support policy on that label;
4. compare against the equal/simple support mixture floor after generator
   rollout.

No OpenAI calls are made. This script consumes existing scenario-evaluation
arrays produced by ``nl_scenario_level_evaluation.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_risk_response_label_audit import (  # noqa: E402
    PORTFOLIO_BOOKS,
    MARKET_INDEX,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    score_sample_distribution,
)


DEFAULT_TRAIN_SCENARIO_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_train_mixture_labels/scenario_eval/"
    "scenario_level_eval_report.json"
)
DEFAULT_TRAIN_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_train_mixture_labels/scenario_eval/"
    "scenario_level_eval_arrays.npz"
)
DEFAULT_EQUAL_TEST_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_kernel_listwise_full_906e_equal_top5_test_s2_seed906/"
    "scenario_level_eval_report.json"
)
DEFAULT_EQUAL_TEST_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_kernel_listwise_full_906e_equal_top5_test_s2_seed906/"
    "scenario_level_eval_arrays.npz"
)
DEFAULT_CANDIDATE_TEST_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_kernel_listwise_921a_scenario_eval/"
    "scenario_level_eval_report.json"
)
DEFAULT_CANDIDATE_TEST_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_kernel_listwise_921a_scenario_eval/"
    "scenario_level_eval_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_support_policy_921a"
)

RELIABLE_BOOKS = ("equity_beta_carry", "dollar_liquidity", "short_volatility")
PORTFOLIO_LABEL_METRIC = "portfolio_reliable_path_score_z"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _safe_query_id(value: Any) -> str:
    raw = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def _narrative_array_key(row: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    """Return the saved generated-sample key for a scenario-eval row."""

    row_no = int(row.get("row_no", -1))
    query_id = str(row.get("query_id", ""))
    exact = f"narrative_{row_no:04d}_{_safe_query_id(query_id)}"
    if exact in arrays:
        return exact
    window_key = f"narrative_{int(row['window_index'])}"
    if window_key in arrays:
        return window_key
    prefix = f"narrative_{row_no:04d}_"
    matches = [name for name in arrays if name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"could not find narrative array for row_no={row_no}, "
        f"window_index={row.get('window_index')}, query_id={query_id!r}"
    )


def _selected_books(names: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    wanted = {str(name) for name in names}
    books = [book for book in PORTFOLIO_BOOKS if str(book["name"]) in wanted]
    missing = sorted(wanted.difference(str(book["name"]) for book in books))
    if missing:
        raise ValueError(f"unknown portfolio books: {missing}")
    if not books:
        raise ValueError("at least one portfolio book is required")
    return books


def _portfolio_path_z(
    paths: np.ndarray,
    delta_scale: np.ndarray,
    book: dict[str, Any],
) -> np.ndarray:
    """Convert delta paths to normalized portfolio-risk-unit paths."""

    arr = np.asarray(paths, dtype=np.float32)
    scale = np.maximum(np.asarray(delta_scale, dtype=np.float32), 1e-8)
    if arr.shape[-2:] != scale.shape:
        raise ValueError("path and delta_scale dimensions are inconsistent")
    result = np.zeros(arr.shape[:-1], dtype=np.float32)
    for market, exposure in dict(book["exposures"]).items():
        idx = int(MARKET_INDEX[market])
        result += float(exposure) * (arr[..., idx] / scale[:, idx])
    return result.astype(np.float32)


def portfolio_candidate_scores(
    *,
    samples: np.ndarray,
    target: np.ndarray,
    delta_scale: np.ndarray,
    books: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return lower-is-better portfolio-response scores for one candidate."""

    book_rows: list[dict[str, Any]] = []
    composite_terms: list[float] = []
    for book in books:
        sample_pnl = _portfolio_path_z(samples, delta_scale, book)[:, :, None]
        target_pnl = _portfolio_path_z(target, delta_scale, book)[:, None]
        scored = score_sample_distribution(
            sample_pnl,
            target_pnl,
            scale=np.ones_like(target_pnl, dtype=np.float32),
        )
        crps = float(scored["ensemble_crps_z"])
        energy = float(scored["energy_score_z"])
        composite = 0.5 * (crps + energy)
        composite_terms.append(composite)
        book_rows.append(
            {
                "book": str(book["name"]),
                "book_label": str(book["label"]),
                "ensemble_crps_z": crps,
                "energy_score_z": energy,
                "composite_path_score_z": float(composite),
                "coverage_80": scored.get("coverage_80"),
            }
        )
    return {
        PORTFOLIO_LABEL_METRIC: float(np.mean(composite_terms)),
        "portfolio_reliable_path_score_std": float(np.std(composite_terms)),
        "books": book_rows,
    }


def build_portfolio_label_scenario_report(
    *,
    scenario_report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    book_names: tuple[str, ...] | list[str] = RELIABLE_BOOKS,
) -> dict[str, Any]:
    """Build a scenario-report-shaped label file for learned support policies."""

    books = _selected_books(book_names)
    future_delta = np.asarray(arrays["future_delta"], dtype=np.float32)
    delta_scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    label_values: list[float] = []
    for row in scenario_report.get("window_scores", []):
        if not isinstance(row, dict):
            continue
        key = _narrative_array_key(row, arrays)
        block_index = int(row["block_window_index"])
        target = future_delta[block_index]
        score_block = portfolio_candidate_scores(
            samples=np.asarray(arrays[key], dtype=np.float32),
            target=target,
            delta_scale=delta_scale,
            books=books,
        )
        value = float(score_block[PORTFOLIO_LABEL_METRIC])
        label_values.append(value)
        row_copy = json.loads(json.dumps(row))
        row_copy.setdefault("methods", {}).setdefault("narrative_generator_topk", {})
        row_copy["methods"]["narrative_generator_topk"].update(
            {
                PORTFOLIO_LABEL_METRIC: value,
                "portfolio_reliable_path_score_std": float(
                    score_block["portfolio_reliable_path_score_std"]
                ),
                "portfolio_response_books": score_block["books"],
            }
        )
        rows.append(row_copy)
    if not rows:
        raise ValueError("no portfolio label rows were built")
    values = np.asarray(label_values, dtype=np.float64)
    return {
        "status": "ok",
        "research_lane": "candidate",
        "result_status": "portfolio_response_label_report_built",
        "benchmark_floor_status": "not_tested",
        "scope_note": (
            "Scenario-report-shaped training labels for a portfolio-risk-aware "
            "support policy. Lower labels are better. Labels are computed only "
            "from historical backtest arrays and realized future deltas; they "
            "are not available at inference."
        ),
        "label_metric": PORTFOLIO_LABEL_METRIC,
        "book_names": [str(book["name"]) for book in books],
        "source_scenario_report": str(scenario_report.get("artifact_paths", {}).get("report", "")),
        "window_scores": rows,
        "summary": {
            "row_count": int(len(rows)),
            "query_count": int(len({int(row["window_index"]) for row in rows})),
            "label_mean": float(np.mean(values)),
            "label_median": float(np.median(values)),
            "label_std": float(np.std(values)),
            "label_min": float(np.min(values)),
            "label_max": float(np.max(values)),
        },
    }


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _method_summary(report: dict[str, Any], method: str = "narrative_generator_topk") -> dict[str, Any]:
    row = report.get("summary", {}).get(method, {})
    if not isinstance(row, dict):
        return {}
    keys = [
        "window_count",
        "ensemble_crps_z_mean",
        "energy_score_z_mean",
        "coverage_80_mean",
        "ensemble_crps_z_improvement_vs_persistence",
        "energy_score_z_improvement_vs_persistence",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _portfolio_summary_for_eval(
    *,
    report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    book_names: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    books = _selected_books(book_names)
    future_delta = np.asarray(arrays["future_delta"], dtype=np.float32)
    delta_scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    values: dict[str, list[float]] = {
        "portfolio_reliable_path_score_z": [],
        "portfolio_reliable_crps_z": [],
        "portfolio_reliable_energy_z": [],
    }
    by_book: dict[str, list[float]] = {str(book["name"]): [] for book in books}
    for row in report.get("window_scores", []):
        if not isinstance(row, dict):
            continue
        key = _narrative_array_key(row, arrays)
        block_index = int(row["block_window_index"])
        scored = portfolio_candidate_scores(
            samples=np.asarray(arrays[key], dtype=np.float32),
            target=future_delta[block_index],
            delta_scale=delta_scale,
            books=books,
        )
        values["portfolio_reliable_path_score_z"].append(
            float(scored[PORTFOLIO_LABEL_METRIC])
        )
        book_crps = [float(book["ensemble_crps_z"]) for book in scored["books"]]
        book_energy = [float(book["energy_score_z"]) for book in scored["books"]]
        values["portfolio_reliable_crps_z"].append(float(np.mean(book_crps)))
        values["portfolio_reliable_energy_z"].append(float(np.mean(book_energy)))
        for book in scored["books"]:
            by_book[str(book["book"])].append(float(book["composite_path_score_z"]))
    return {
        "window_count": int(len(values["portfolio_reliable_path_score_z"])),
        "book_names": list(book_names),
        "means": {
            key: float(np.mean(raw)) if raw else None for key, raw in values.items()
        },
        "by_book_mean": {
            key: float(np.mean(raw)) if raw else None for key, raw in by_book.items()
        },
    }


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _relative_reduction(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or abs(float(baseline)) <= 1e-12:
        return None
    return (float(baseline) - float(candidate)) / float(baseline)


def compare_policy_to_equal_floor(
    *,
    equal_report: dict[str, Any],
    candidate_report: dict[str, Any],
    equal_arrays: dict[str, np.ndarray],
    candidate_arrays: dict[str, np.ndarray],
    book_names: tuple[str, ...] | list[str] = RELIABLE_BOOKS,
) -> dict[str, Any]:
    equal_scenario = _method_summary(equal_report)
    candidate_scenario = _method_summary(candidate_report)
    equal_portfolio = _portfolio_summary_for_eval(
        report=equal_report,
        arrays=equal_arrays,
        book_names=book_names,
    )
    candidate_portfolio = _portfolio_summary_for_eval(
        report=candidate_report,
        arrays=candidate_arrays,
        book_names=book_names,
    )
    scenario_delta = {
        key: _delta(candidate_scenario.get(key), equal_scenario.get(key))
        for key in [
            "ensemble_crps_z_mean",
            "energy_score_z_mean",
            "coverage_80_mean",
        ]
    }
    scenario_relative = {
        "ensemble_crps_z_mean_relative_reduction_vs_equal": _relative_reduction(
            candidate_scenario.get("ensemble_crps_z_mean"),
            equal_scenario.get("ensemble_crps_z_mean"),
        ),
        "energy_score_z_mean_relative_reduction_vs_equal": _relative_reduction(
            candidate_scenario.get("energy_score_z_mean"),
            equal_scenario.get("energy_score_z_mean"),
        ),
    }
    portfolio_delta = {
        key: _delta(
            candidate_portfolio["means"].get(key),
            equal_portfolio["means"].get(key),
        )
        for key in sorted(equal_portfolio["means"])
    }
    portfolio_relative = {
        f"{key}_relative_reduction_vs_equal": _relative_reduction(
            candidate_portfolio["means"].get(key),
            equal_portfolio["means"].get(key),
        )
        for key in sorted(equal_portfolio["means"])
    }
    crps_delta = scenario_delta["ensemble_crps_z_mean"]
    energy_delta = scenario_delta["energy_score_z_mean"]
    portfolio_score_delta = portfolio_delta["portfolio_reliable_path_score_z"]
    clean_quality = (
        crps_delta is not None
        and energy_delta is not None
        and crps_delta <= 0.0
        and energy_delta <= 0.0
    )
    portfolio_improves = (
        portfolio_score_delta is not None and portfolio_score_delta < 0.0
    )
    severe_quality_regression = (
        crps_delta is not None
        and energy_delta is not None
        and (crps_delta > 0.005 or energy_delta > 0.005)
    )
    if clean_quality and portfolio_improves:
        status = "candidate_beats_equal_floor"
        benchmark_floor_status = "beats_floor"
    elif portfolio_improves and not severe_quality_regression:
        status = "candidate_tradeoff_not_promoted"
        benchmark_floor_status = "competitive"
    elif severe_quality_regression:
        status = "candidate_rejected_quality_regression"
        benchmark_floor_status = "below_floor"
    else:
        status = "candidate_not_current_lever"
        benchmark_floor_status = "below_floor"
    return {
        "status": status,
        "research_lane": "candidate",
        "result_status": status,
        "benchmark_floor_status": benchmark_floor_status,
        "scope_note": (
            "Portfolio-response-aware support policy compared against the "
            "equal/simple support mixture floor. Lower CRPS, energy, and "
            "portfolio path scores are better; higher coverage is better."
        ),
        "equal_scenario": equal_scenario,
        "candidate_scenario": candidate_scenario,
        "scenario_delta_candidate_minus_equal": scenario_delta,
        "scenario_relative_reduction": scenario_relative,
        "equal_portfolio": equal_portfolio,
        "candidate_portfolio": candidate_portfolio,
        "portfolio_delta_candidate_minus_equal": portfolio_delta,
        "portfolio_relative_reduction": portfolio_relative,
        "decision": {
            "clean_quality_improvement": bool(clean_quality),
            "portfolio_score_improves": bool(portfolio_improves),
            "severe_quality_regression": bool(severe_quality_regression),
            "interpretation": (
                "Promote only if portfolio response improves without weakening "
                "held-out CRPS/energy/coverage. A portfolio-only gain with "
                "quality regression remains diagnostic."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    scen = report["scenario_delta_candidate_minus_equal"]
    port = report["portfolio_delta_candidate_minus_equal"]
    rel = report["portfolio_relative_reduction"]
    return "\n".join(
        [
            "# Portfolio-Response Support-Policy TestFlight",
            "",
            f"Status: `{report['status']}`",
            f"Benchmark floor status: `{report['benchmark_floor_status']}`",
            "",
            "## Scenario Quality Delta",
            "",
            "| Metric | Candidate - equal floor |",
            "|---|---:|",
            f"| Ensemble CRPS z | {scen.get('ensemble_crps_z_mean')} |",
            f"| Energy score z | {scen.get('energy_score_z_mean')} |",
            f"| 80% coverage | {scen.get('coverage_80_mean')} |",
            "",
            "## Portfolio Response Delta",
            "",
            "| Metric | Candidate - equal floor | Relative reduction |",
            "|---|---:|---:|",
            (
                "| Reliable portfolio path score z | "
                f"{port.get('portfolio_reliable_path_score_z')} | "
                f"{rel.get('portfolio_reliable_path_score_z_relative_reduction_vs_equal')} |"
            ),
            (
                "| Reliable portfolio CRPS z | "
                f"{port.get('portfolio_reliable_crps_z')} | "
                f"{rel.get('portfolio_reliable_crps_z_relative_reduction_vs_equal')} |"
            ),
            (
                "| Reliable portfolio energy z | "
                f"{port.get('portfolio_reliable_energy_z')} | "
                f"{rel.get('portfolio_reliable_energy_z_relative_reduction_vs_equal')} |"
            ),
            "",
            "## Decision",
            "",
            str(report["decision"]["interpretation"]),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenario-report", type=Path, default=DEFAULT_TRAIN_SCENARIO_REPORT)
    parser.add_argument("--train-arrays", type=Path, default=DEFAULT_TRAIN_ARRAYS)
    parser.add_argument("--equal-test-report", type=Path, default=DEFAULT_EQUAL_TEST_REPORT)
    parser.add_argument("--equal-test-arrays", type=Path, default=DEFAULT_EQUAL_TEST_ARRAYS)
    parser.add_argument("--candidate-test-report", type=Path, default=DEFAULT_CANDIDATE_TEST_REPORT)
    parser.add_argument("--candidate-test-arrays", type=Path, default=DEFAULT_CANDIDATE_TEST_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--book-names",
        default=",".join(RELIABLE_BOOKS),
        help="Comma-separated portfolio book names for the reliable label.",
    )
    parser.add_argument(
        "--skip-label-report",
        action="store_true",
        help="Only run the candidate/equal comparison.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Only build the train label report.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    book_names = tuple(
        item.strip() for item in str(args.book_names).split(",") if item.strip()
    )
    label_path = output_dir / "portfolio_response_label_scenario_report.json"
    comparison_path = output_dir / "portfolio_response_support_policy_comparison.json"
    markdown_path = output_dir / "portfolio_response_support_policy_comparison.md"
    result: dict[str, Any] = {
        "label_report": None,
        "comparison": None,
    }
    if not bool(args.skip_label_report):
        label_report = build_portfolio_label_scenario_report(
            scenario_report=_load_json(args.train_scenario_report),
            arrays=_load_npz(args.train_arrays),
            book_names=book_names,
        )
        label_report["artifact_paths"] = {
            "report": str(label_path),
            "source_scenario_report": str(args.train_scenario_report),
            "source_arrays": str(args.train_arrays),
        }
        _write_json(label_path, label_report)
        result["label_report"] = str(label_path)
    if not bool(args.skip_comparison):
        comparison = compare_policy_to_equal_floor(
            equal_report=_load_json(args.equal_test_report),
            candidate_report=_load_json(args.candidate_test_report),
            equal_arrays=_load_npz(args.equal_test_arrays),
            candidate_arrays=_load_npz(args.candidate_test_arrays),
            book_names=book_names,
        )
        comparison["artifact_paths"] = {
            "report": str(comparison_path),
            "markdown": str(markdown_path),
            "equal_test_report": str(args.equal_test_report),
            "candidate_test_report": str(args.candidate_test_report),
            "equal_test_arrays": str(args.equal_test_arrays),
            "candidate_test_arrays": str(args.candidate_test_arrays),
        }
        _write_json(comparison_path, comparison)
        _write_text(markdown_path, _markdown(comparison))
        result["comparison"] = str(comparison_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
