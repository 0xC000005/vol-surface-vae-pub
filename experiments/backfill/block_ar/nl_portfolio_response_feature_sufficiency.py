#!/usr/bin/env python
"""Diagnose deployable signals for portfolio-response support weighting.

This is a post-experiment diagnostic. It does not run the generator and does
not call OpenAI. It asks whether the stable CRN portfolio-response labels are
predictable from deployable candidate information:

1. simple candidate rank order;
2. the existing hand-built candidate features;
3. a train-only support reliability prior learned from historical support IDs;
4. a small least-squares blend of rank and support reliability.

The goal is to decide whether the next support-policy iteration should improve
features/scoring or whether the equal support-mixture floor is already close to
the best deployable signal available from the current candidate set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_learned_mixture_policy_testflight import (  # noqa: E402
    build_mixture_policy_training_table,
    fit_kernel_listwise_mixture_policy,
)
from experiments.backfill.block_ar.nl_portfolio_response_policy_postmortem import (  # noqa: E402
    pairwise_preference_accuracy,
    selection_regret,
)
from experiments.backfill.block_ar.nl_portfolio_response_support_policy_testflight import (  # noqa: E402
    PORTFOLIO_LABEL_METRIC,
)


DEFAULT_TRAIN_CANDIDATE_BRIDGE = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_train_mixture_labels/"
    "mixture_label_bridge_report.json"
)
DEFAULT_TRAIN_LABEL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_train_mixture_labels_922f_crn_full/"
    "portfolio_response_label_scenario_report.json"
)
DEFAULT_TEST_CANDIDATE_BRIDGE = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_test_mixture_candidates/"
    "mixture_label_bridge_report.json"
)
DEFAULT_TEST_LABEL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_test_mixture_candidates_922e_crn_full/"
    "portfolio_response_label_scenario_report.json"
)
DEFAULT_BRIDGE_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_arrays.npz"
)
DEFAULT_HISTORY_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_full_906b_history_level/history_level_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_feature_sufficiency_923a"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


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


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return None
    aa = a[mask] - float(np.mean(a[mask]))
    bb = b[mask] - float(np.mean(b[mask]))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return float(np.dot(aa, bb) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(raw, kind="mergesort")
    ranks = np.empty(raw.shape[0], dtype=np.float64)
    start = 0
    while start < raw.shape[0]:
        end = start + 1
        while end < raw.shape[0] and raw[order[end]] == raw[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def _query_groups(query_ids: list[Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, query_id in enumerate(query_ids):
        groups[str(query_id)].append(idx)
    return dict(groups)


def _query_standardized(values: np.ndarray, query_ids: list[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arr)
    for indices in _query_groups(query_ids).values():
        idx = np.asarray(indices, dtype=np.int64)
        local = arr[idx]
        scale = float(np.std(local))
        if scale <= 1e-12:
            out[idx] = 0.0
        else:
            out[idx] = (local - float(np.mean(local))) / scale
    return out


def _query_positions(query_ids: list[Any]) -> np.ndarray:
    pos = np.zeros(len(query_ids), dtype=np.float64)
    for indices in _query_groups(query_ids).values():
        for local_no, idx in enumerate(indices, start=1):
            pos[idx] = float(local_no)
    return pos


def _support_reliability_prior(
    *,
    support_rows: list[list[int]],
    labels: np.ndarray,
    query_ids: list[Any],
) -> dict[int, float]:
    standardized = _query_standardized(labels, query_ids)
    buckets: dict[int, list[float]] = defaultdict(list)
    for support, value in zip(support_rows, standardized, strict=True):
        for support_id in support:
            buckets[int(support_id)].append(float(value))
    return {key: float(np.mean(values)) for key, values in buckets.items() if values}


def _score_support_prior(
    support_rows: list[list[int]],
    prior: dict[int, float],
) -> np.ndarray:
    scores: list[float] = []
    for support in support_rows:
        values = [float(prior.get(int(support_id), 0.0)) for support_id in support]
        scores.append(float(np.mean(values)) if values else 0.0)
    return np.asarray(scores, dtype=np.float64)


def _fit_linear_blend(train_scores: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    x = np.asarray(train_scores, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("train_scores must be [N,D] and align with labels")
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    ridge = 1e-4 * np.eye(design.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + ridge, design.T @ y)


def _predict_linear_blend(scores: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return (design @ np.asarray(coef, dtype=np.float64)).astype(np.float64)


def _evaluate_scorer(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    query_ids: list[Any],
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    pred = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        "pearson": _pearson(pred, truth),
        "spearman": _spearman(pred, truth),
        "pairwise": pairwise_preference_accuracy(
            predicted_scores=pred,
            true_scores=truth,
            query_ids=query_ids,
        ),
        "selection_regret": selection_regret(
            predicted_scores=pred,
            raw_losses=-truth,
            query_ids=query_ids,
        ),
    }


def _build_table(
    *,
    candidate_bridge: dict[str, Any],
    label_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
) -> Any:
    return build_mixture_policy_training_table(
        candidate_bridge=candidate_bridge,
        scenario_report=label_report,
        condition_vectors=np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32),
        memory_targets=np.asarray(bridge_arrays["memory_targets"], dtype=np.float32),
        history_level=np.asarray(history_arrays["history_level"], dtype=np.float32),
        method="narrative_generator_topk",
        metric=PORTFOLIO_LABEL_METRIC,
    )


def feature_sufficiency_report(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
    test_candidate_bridge: dict[str, Any],
    test_label_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
    k_neighbors: int = 32,
    probability_temperature: float = 0.20,
) -> dict[str, Any]:
    train = _build_table(
        candidate_bridge=train_candidate_bridge,
        label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
    )
    test = _build_table(
        candidate_bridge=test_candidate_bridge,
        label_report=test_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
    )
    train_labels = np.asarray(train.labels, dtype=np.float64)
    test_labels = np.asarray(test.labels, dtype=np.float64)
    train_query_ids = [row["window_index"] for row in train.rows]
    test_query_ids = [row["window_index"] for row in test.rows]
    train_support = [list(map(int, row["support_window_indices"])) for row in train.rows]
    test_support = [list(map(int, row["support_window_indices"])) for row in test.rows]

    train_rank = -_query_positions(train_query_ids)
    test_rank = -_query_positions(test_query_ids)

    support_prior = _support_reliability_prior(
        support_rows=train_support,
        labels=train_labels,
        query_ids=train_query_ids,
    )
    train_support_score = _score_support_prior(train_support, support_prior)
    test_support_score = _score_support_prior(test_support, support_prior)
    unknown_supports = sorted(
        {
            int(support_id)
            for support in test_support
            for support_id in support
            if int(support_id) not in support_prior
        }
    )

    blend_coef = _fit_linear_blend(
        np.stack(
            [
                _query_standardized(train_rank, train_query_ids),
                _query_standardized(train_support_score, train_query_ids),
            ],
            axis=1,
        ),
        _query_standardized(train_labels, train_query_ids),
    )
    train_blend = _predict_linear_blend(
        np.stack(
            [
                _query_standardized(train_rank, train_query_ids),
                _query_standardized(train_support_score, train_query_ids),
            ],
            axis=1,
        ),
        blend_coef,
    )
    test_blend = _predict_linear_blend(
        np.stack(
            [
                _query_standardized(test_rank, test_query_ids),
                _query_standardized(test_support_score, test_query_ids),
            ],
            axis=1,
        ),
        blend_coef,
    )

    kernel = fit_kernel_listwise_mixture_policy(
        train.features,
        train.labels,
        query_ids=train_query_ids,
        feature_names=train.feature_names,
        k_neighbors=int(k_neighbors),
        probability_temperature=float(probability_temperature),
    )
    train_kernel = kernel.predict(train.features)
    test_kernel = kernel.predict(test.features)

    train_feature_plus = np.concatenate(
        [train.features, train_support_score[:, None].astype(np.float32)],
        axis=1,
    )
    test_feature_plus = np.concatenate(
        [test.features, test_support_score[:, None].astype(np.float32)],
        axis=1,
    )
    kernel_plus = fit_kernel_listwise_mixture_policy(
        train_feature_plus,
        train.labels,
        query_ids=train_query_ids,
        feature_names=[*train.feature_names, "train_support_reliability_prior"],
        k_neighbors=int(k_neighbors),
        probability_temperature=float(probability_temperature),
    )
    train_kernel_plus = kernel_plus.predict(train_feature_plus)
    test_kernel_plus = kernel_plus.predict(test_feature_plus)

    scorers = {
        "candidate_rank": (train_rank, test_rank),
        "support_reliability_prior": (train_support_score, test_support_score),
        "rank_support_prior_linear_blend": (train_blend, test_blend),
        "existing_kernel_listwise_features": (train_kernel, test_kernel),
        "kernel_listwise_plus_support_prior": (train_kernel_plus, test_kernel_plus),
    }
    train_eval = {
        name: _evaluate_scorer(scores=train_scores, labels=train_labels, query_ids=train_query_ids)
        for name, (train_scores, _) in scorers.items()
    }
    test_eval = {
        name: _evaluate_scorer(scores=test_scores, labels=test_labels, query_ids=test_query_ids)
        for name, (_, test_scores) in scorers.items()
    }

    rank_pairwise = (
        test_eval["candidate_rank"]["pairwise"]["accuracy"]
        if test_eval["candidate_rank"]["pairwise"]["accuracy"] is not None
        else 0.0
    )
    best_name = max(
        test_eval,
        key=lambda name: (
            -999.0
            if test_eval[name]["pairwise"]["accuracy"] is None
            else float(test_eval[name]["pairwise"]["accuracy"])
        ),
    )
    best_pairwise = test_eval[best_name]["pairwise"]["accuracy"] or 0.0
    best_regret = test_eval[best_name]["selection_regret"]["mean_regret"]
    rank_regret = test_eval["candidate_rank"]["selection_regret"]["mean_regret"]
    if best_pairwise >= rank_pairwise + 0.03 and (
        best_regret is not None
        and rank_regret is not None
        and float(best_regret) < float(rank_regret)
    ):
        result_status = "deployable_feature_signal_found"
    elif best_pairwise >= rank_pairwise + 0.01:
        result_status = "weak_deployable_feature_signal"
    else:
        result_status = "feature_surface_below_rank_floor"

    return {
        "status": "ok",
        "research_lane": "candidate",
        "result_status": result_status,
        "benchmark_floor_status": (
            "beats_floor" if result_status == "deployable_feature_signal_found"
            else "competitive" if result_status == "weak_deployable_feature_signal"
            else "below_floor"
        ),
        "scope_note": (
            "Feature-sufficiency diagnostic for stable CRN portfolio-response "
            "labels. Labels use historical realized futures for training/eval; "
            "reported scorers use only deployable candidate/support metadata."
        ),
        "train_rows": int(train.features.shape[0]),
        "train_queries": int(len(set(train_query_ids))),
        "test_rows": int(test.features.shape[0]),
        "test_queries": int(len(set(test_query_ids))),
        "feature_names": list(train.feature_names),
        "support_prior": {
            "known_support_count": int(len(support_prior)),
            "unknown_test_support_count": int(len(unknown_supports)),
            "unknown_test_support_ids": unknown_supports[:20],
        },
        "rank_support_blend": {
            "intercept": float(blend_coef[0]),
            "rank_coef": float(blend_coef[1]),
            "support_prior_coef": float(blend_coef[2]),
        },
        "kernel_policy": {
            "k_neighbors": int(kernel.k_neighbors),
            "bandwidth": float(kernel.bandwidth),
            "probability_temperature": float(kernel.probability_temperature),
        },
        "train": train_eval,
        "test": test_eval,
        "decision": {
            "best_test_scorer": best_name,
            "best_test_pairwise": float(best_pairwise),
            "rank_test_pairwise": float(rank_pairwise),
            "interpretation": (
                "If support reliability or feature+support-prior beats rank "
                "cleanly, the next candidate can add that signal to the support "
                "policy and run scenario-level evaluation. If not, avoid more "
                "ranker capacity and revisit candidate support generation or "
                "the value target."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Portfolio-Response Feature Sufficiency",
        "",
        f"Status: `{report['result_status']}`",
        f"Benchmark floor: `{report['benchmark_floor_status']}`",
        "",
        "## Test Scorers",
        "",
        "| Scorer | Pairwise | Mean Regret | Median Regret | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["test"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["pairwise"]["accuracy"]),
                    str(row["selection_regret"]["mean_regret"]),
                    str(row["selection_regret"]["median_regret"]),
                    str(row["spearman"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- best test scorer: `{report['decision']['best_test_scorer']}`",
            f"- best pairwise: `{report['decision']['best_test_pairwise']}`",
            f"- rank pairwise: `{report['decision']['rank_test_pairwise']}`",
            "",
            report["decision"]["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidate-bridge", type=Path, default=DEFAULT_TRAIN_CANDIDATE_BRIDGE)
    parser.add_argument("--train-label-report", type=Path, default=DEFAULT_TRAIN_LABEL_REPORT)
    parser.add_argument("--test-candidate-bridge", type=Path, default=DEFAULT_TEST_CANDIDATE_BRIDGE)
    parser.add_argument("--test-label-report", type=Path, default=DEFAULT_TEST_LABEL_REPORT)
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--history-arrays", type=Path, default=DEFAULT_HISTORY_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kernel-k-neighbors", type=int, default=32)
    parser.add_argument("--kernel-probability-temperature", type=float, default=0.20)
    args = parser.parse_args()

    report = feature_sufficiency_report(
        train_candidate_bridge=_load_json(args.train_candidate_bridge),
        train_label_report=_load_json(args.train_label_report),
        test_candidate_bridge=_load_json(args.test_candidate_bridge),
        test_label_report=_load_json(args.test_label_report),
        bridge_arrays=_load_npz(args.bridge_arrays),
        history_arrays=_load_npz(args.history_arrays),
        k_neighbors=args.kernel_k_neighbors,
        probability_temperature=args.kernel_probability_temperature,
    )
    report["artifact_paths"] = {
        "train_candidate_bridge": str(args.train_candidate_bridge),
        "train_label_report": str(args.train_label_report),
        "test_candidate_bridge": str(args.test_candidate_bridge),
        "test_label_report": str(args.test_label_report),
        "report": str(args.output_dir / "portfolio_response_feature_sufficiency.json"),
        "markdown": str(args.output_dir / "portfolio_response_feature_sufficiency.md"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "portfolio_response_feature_sufficiency.json", report)
    _write_text(args.output_dir / "portfolio_response_feature_sufficiency.md", _markdown(report))
    print(
        json.dumps(
            {
                "status": report["result_status"],
                "benchmark_floor_status": report["benchmark_floor_status"],
                "report": report["artifact_paths"]["report"],
                "best_test_scorer": report["decision"]["best_test_scorer"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
