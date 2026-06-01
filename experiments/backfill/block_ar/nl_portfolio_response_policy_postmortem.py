#!/usr/bin/env python
"""Postmortem for portfolio-response support-policy failures.

This is a post-experiment analysis script, not a new policy. It attributes why
the deployable portfolio-response support policies did not close the gap to the
positive oracle gate:

1. Is the response label predictable from deployable candidate features?
2. Are labels built from low-sample generator rollouts too noisy?
3. Does the candidate support list contain enough oracle headroom?
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
from experiments.backfill.block_ar.nl_portfolio_response_support_policy_testflight import (  # noqa: E402
    PORTFOLIO_LABEL_METRIC,
    RELIABLE_BOOKS,
    _load_npz,
    _narrative_array_key,
    _selected_books,
    portfolio_candidate_scores,
)


DEFAULT_TRAIN_CANDIDATE_BRIDGE = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_train_mixture_labels/"
    "mixture_label_bridge_report.json"
)
DEFAULT_TRAIN_LABEL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_support_policy_921a/"
    "portfolio_response_label_scenario_report.json"
)
DEFAULT_TEST_CANDIDATE_BRIDGE = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_test_mixture_candidates/"
    "mixture_label_bridge_report.json"
)
DEFAULT_TEST_LABEL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_test_mixture_candidates_921b/"
    "portfolio_response_label_scenario_report.json"
)
DEFAULT_TEST_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_test_mixture_candidates_921b/"
    "scenario_eval/scenario_level_eval_arrays.npz"
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
    "nl_portfolio_response_policy_postmortem_922a"
)


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
        rank_value = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank_value
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def _group_indices(query_ids: list[Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, query_id in enumerate(query_ids):
        groups[str(query_id)].append(idx)
    return dict(groups)


def pairwise_preference_accuracy(
    *,
    predicted_scores: np.ndarray,
    true_scores: np.ndarray,
    query_ids: list[Any],
    min_gap: float = 1e-8,
) -> dict[str, Any]:
    """Return within-query pairwise accuracy for higher-is-better scores."""

    pred = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(true_scores, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape or pred.shape[0] != len(query_ids):
        raise ValueError("predicted_scores, true_scores, and query_ids must match")
    correct = 0
    total = 0
    ties = 0
    for indices in _group_indices(query_ids).values():
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                true_gap = float(truth[left] - truth[right])
                if abs(true_gap) <= float(min_gap):
                    continue
                pred_gap = float(pred[left] - pred[right])
                if abs(pred_gap) <= 1e-12:
                    ties += 1
                    continue
                total += 1
                if np.sign(pred_gap) == np.sign(true_gap):
                    correct += 1
    return {
        "pair_count": int(total),
        "tie_count": int(ties),
        "accuracy": float(correct / total) if total else None,
    }


def selection_regret(
    *,
    predicted_scores: np.ndarray,
    raw_losses: np.ndarray,
    query_ids: list[Any],
) -> dict[str, Any]:
    """Measure loss regret from selecting max predicted score per query."""

    pred = np.asarray(predicted_scores, dtype=np.float64).reshape(-1)
    losses = np.asarray(raw_losses, dtype=np.float64).reshape(-1)
    regrets: list[float] = []
    selected_positions: list[int] = []
    oracle_positions: list[int] = []
    for indices in _group_indices(query_ids).values():
        idx = np.asarray(indices, dtype=np.int64)
        selected_local = int(np.argmax(pred[idx]))
        oracle_local = int(np.argmin(losses[idx]))
        selected_positions.append(selected_local + 1)
        oracle_positions.append(oracle_local + 1)
        regrets.append(float(losses[idx[selected_local]] - losses[idx[oracle_local]]))
    arr = np.asarray(regrets, dtype=np.float64)
    return {
        "query_count": int(arr.size),
        "mean_regret": float(np.mean(arr)) if arr.size else None,
        "median_regret": float(np.median(arr)) if arr.size else None,
        "p90_regret": float(np.quantile(arr, 0.90)) if arr.size else None,
        "selected_position_mean": float(np.mean(selected_positions)) if selected_positions else None,
        "oracle_position_mean": float(np.mean(oracle_positions)) if oracle_positions else None,
        "exact_oracle_selection_rate": float(np.mean(arr <= 1e-12)) if arr.size else None,
    }


def _candidate_rank_score(rows: list[dict[str, Any]]) -> np.ndarray:
    scores: list[float] = []
    for row in rows:
        rank = int(row.get("candidate_mixture_rank", len(scores) + 1) or 1)
        scores.append(-float(rank))
    return np.asarray(scores, dtype=np.float64)


def _table_from_reports(
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


def predictability_report(
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
    train_table = _table_from_reports(
        candidate_bridge=train_candidate_bridge,
        label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
    )
    test_table = _table_from_reports(
        candidate_bridge=test_candidate_bridge,
        label_report=test_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
    )
    model = fit_kernel_listwise_mixture_policy(
        train_table.features,
        train_table.labels,
        query_ids=[row["window_index"] for row in train_table.rows],
        feature_names=train_table.feature_names,
        k_neighbors=int(k_neighbors),
        probability_temperature=float(probability_temperature),
    )
    train_pred = model.predict(train_table.features)
    test_pred = model.predict(test_table.features)
    train_truth = np.asarray(train_table.labels, dtype=np.float64)
    test_truth = np.asarray(test_table.labels, dtype=np.float64)
    train_queries = [row["window_index"] for row in train_table.rows]
    test_queries = [row["window_index"] for row in test_table.rows]
    rank_score = _candidate_rank_score(test_table.rows)
    return {
        "scope_note": (
            "Predictability uses higher-is-better labels, i.e. negative "
            "portfolio response loss. It is diagnostic and uses realized "
            "future paths only for label construction."
        ),
        "train_rows": int(train_table.features.shape[0]),
        "train_queries": int(len(set(train_queries))),
        "test_rows": int(test_table.features.shape[0]),
        "test_queries": int(len(set(test_queries))),
        "feature_count": int(train_table.features.shape[1]),
        "feature_names": list(train_table.feature_names),
        "kernel_policy": {
            "k_neighbors": int(model.k_neighbors),
            "bandwidth": float(model.bandwidth),
            "probability_temperature": float(model.probability_temperature),
        },
        "train": {
            "pearson": _pearson(train_pred, train_truth),
            "spearman": _spearman(train_pred, train_truth),
            "pairwise": pairwise_preference_accuracy(
                predicted_scores=train_pred,
                true_scores=train_truth,
                query_ids=train_queries,
            ),
            "selection_regret": selection_regret(
                predicted_scores=train_pred,
                raw_losses=-train_truth,
                query_ids=train_queries,
            ),
        },
        "test": {
            "pearson": _pearson(test_pred, test_truth),
            "spearman": _spearman(test_pred, test_truth),
            "pairwise": pairwise_preference_accuracy(
                predicted_scores=test_pred,
                true_scores=test_truth,
                query_ids=test_queries,
            ),
            "selection_regret": selection_regret(
                predicted_scores=test_pred,
                raw_losses=-test_truth,
                query_ids=test_queries,
            ),
            "rank_baseline_pairwise": pairwise_preference_accuracy(
                predicted_scores=rank_score,
                true_scores=test_truth,
                query_ids=test_queries,
            ),
            "rank_baseline_selection_regret": selection_regret(
                predicted_scores=rank_score,
                raw_losses=-test_truth,
                query_ids=test_queries,
            ),
        },
    }


def label_noise_report(
    *,
    candidate_label_report: dict[str, Any],
    arrays: dict[str, np.ndarray],
    book_names: tuple[str, ...] | list[str] = RELIABLE_BOOKS,
) -> dict[str, Any]:
    future_delta = np.asarray(arrays["future_delta"], dtype=np.float32)
    delta_scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    full_scores: list[float] = []
    half_a_scores: list[float] = []
    half_b_scores: list[float] = []
    query_ids: list[int] = []
    rows = [
        row for row in candidate_label_report.get("window_scores", []) if isinstance(row, dict)
    ]
    books = _selected_books(book_names)
    for row in rows:
        key = _narrative_array_key(row, arrays)
        samples = np.asarray(arrays[key], dtype=np.float32)
        if int(samples.shape[0]) < 4:
            continue
        split = int(samples.shape[0] // 2)
        target = future_delta[int(row["block_window_index"])]
        full = portfolio_candidate_scores(
            samples=samples,
            target=target,
            delta_scale=delta_scale,
            books=books,
        )[PORTFOLIO_LABEL_METRIC]
        left = portfolio_candidate_scores(
            samples=samples[:split],
            target=target,
            delta_scale=delta_scale,
            books=books,
        )[PORTFOLIO_LABEL_METRIC]
        right = portfolio_candidate_scores(
            samples=samples[split : split * 2],
            target=target,
            delta_scale=delta_scale,
            books=books,
        )[PORTFOLIO_LABEL_METRIC]
        full_scores.append(float(full))
        half_a_scores.append(float(left))
        half_b_scores.append(float(right))
        query_ids.append(int(row["window_index"]))
    full_arr = np.asarray(full_scores, dtype=np.float64)
    a_arr = np.asarray(half_a_scores, dtype=np.float64)
    b_arr = np.asarray(half_b_scores, dtype=np.float64)
    half_abs = np.abs(a_arr - b_arr)
    spreads: list[float] = []
    for indices in _group_indices(query_ids).values():
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size:
            spreads.append(float(np.max(full_arr[idx]) - np.min(full_arr[idx])))
    spread_arr = np.asarray(spreads, dtype=np.float64)
    return {
        "row_count": int(full_arr.size),
        "query_count": int(len(set(query_ids))),
        "sample_count_per_candidate": int(
            next(
                (
                    np.asarray(arrays[_narrative_array_key(row, arrays)]).shape[0]
                    for row in rows
                    if _narrative_array_key(row, arrays) in arrays
                ),
                0,
            )
        ),
        "full_vs_half_a_pearson": _pearson(full_arr, a_arr),
        "half_a_vs_half_b_pearson": _pearson(a_arr, b_arr),
        "full_vs_half_a_spearman": _spearman(full_arr, a_arr),
        "half_a_vs_half_b_spearman": _spearman(a_arr, b_arr),
        "median_abs_half_difference": float(np.median(half_abs)) if half_abs.size else None,
        "p90_abs_half_difference": float(np.quantile(half_abs, 0.90)) if half_abs.size else None,
        "median_within_query_spread": float(np.median(spread_arr)) if spread_arr.size else None,
        "half_diff_to_query_spread_ratio_median": (
            float(np.median(half_abs) / np.median(spread_arr))
            if half_abs.size and spread_arr.size and float(np.median(spread_arr)) > 1e-12
            else None
        ),
        "half_pairwise_agreement": pairwise_preference_accuracy(
            predicted_scores=-a_arr,
            true_scores=-b_arr,
            query_ids=query_ids,
        ),
    }


def candidate_headroom_report(candidate_label_report: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in candidate_label_report.get("window_scores", []) if isinstance(row, dict)
    ]
    losses: list[float] = []
    query_ids: list[int] = []
    for row in rows:
        metric = (
            row.get("methods", {})
            .get("narrative_generator_topk", {})
            .get(PORTFOLIO_LABEL_METRIC)
        )
        if metric is None:
            continue
        losses.append(float(metric))
        query_ids.append(int(row["window_index"]))
    loss_arr = np.asarray(losses, dtype=np.float64)
    groups = _group_indices(query_ids)
    top1_gaps: list[float] = []
    oracle_positions: list[int] = []
    spreads: list[float] = []
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        local = loss_arr[idx]
        top1_gaps.append(float(local[0] - np.min(local)))
        oracle_positions.append(int(np.argmin(local)) + 1)
        spreads.append(float(np.max(local) - np.min(local)))
    return {
        "query_count": int(len(groups)),
        "candidate_rows": int(loss_arr.size),
        "mean_top1_minus_oracle": float(np.mean(top1_gaps)) if top1_gaps else None,
        "median_top1_minus_oracle": float(np.median(top1_gaps)) if top1_gaps else None,
        "p90_top1_minus_oracle": float(np.quantile(top1_gaps, 0.90)) if top1_gaps else None,
        "oracle_position_mean": float(np.mean(oracle_positions)) if oracle_positions else None,
        "oracle_position_median": float(np.median(oracle_positions)) if oracle_positions else None,
        "oracle_position_counts": {
            str(pos): int(oracle_positions.count(pos)) for pos in sorted(set(oracle_positions))
        },
        "mean_within_query_spread": float(np.mean(spreads)) if spreads else None,
        "median_within_query_spread": float(np.median(spreads)) if spreads else None,
    }


def decide_postmortem(report: dict[str, Any]) -> dict[str, Any]:
    pred = report["predictability"]["test"]
    noise = report["label_noise"]
    headroom = report["candidate_headroom"]
    pairwise = pred["pairwise"]["accuracy"] or 0.0
    rank_pairwise = pred["rank_baseline_pairwise"]["accuracy"] or 0.0
    half_corr = noise["half_a_vs_half_b_pearson"] or 0.0
    half_ratio = noise["half_diff_to_query_spread_ratio_median"] or 0.0
    oracle_gap = headroom["median_top1_minus_oracle"] or 0.0
    if half_corr < 0.65 or half_ratio > 0.50:
        status = "label_noise_is_primary_bottleneck"
        next_step = (
            "Increase candidate label sample count or use common-random-number "
            "labeling before training another support policy."
        )
    elif pairwise <= max(0.55, rank_pairwise + 0.02) and oracle_gap > 0.05:
        status = "feature_predictability_is_primary_bottleneck"
        next_step = (
            "Improve deployable candidate features or response-surface modeling; "
            "do not add capacity without better out-of-sample pairwise accuracy."
        )
    elif oracle_gap <= 0.05:
        status = "candidate_headroom_too_small"
        next_step = (
            "Improve candidate support generation before training another "
            "support policy."
        )
    else:
        status = "mixed_bottleneck"
        next_step = (
            "Run one controlled higher-sample labeling pass and then remeasure "
            "feature predictability."
        )
    return {
        "status": status,
        "next_step": next_step,
        "evidence": {
            "test_pairwise_accuracy": pairwise,
            "rank_baseline_pairwise_accuracy": rank_pairwise,
            "half_label_pearson": half_corr,
            "half_diff_to_query_spread_ratio_median": half_ratio,
            "median_top1_minus_oracle": oracle_gap,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    pred = report["predictability"]
    noise = report["label_noise"]
    headroom = report["candidate_headroom"]
    return "\n".join(
        [
            "# Portfolio-Response Policy Postmortem",
            "",
            f"Status: `{decision['status']}`",
            "",
            "## Predictability",
            "",
            f"- Train pairwise accuracy: `{pred['train']['pairwise']['accuracy']}`",
            f"- Test pairwise accuracy: `{pred['test']['pairwise']['accuracy']}`",
            f"- Test rank-baseline pairwise accuracy: `{pred['test']['rank_baseline_pairwise']['accuracy']}`",
            f"- Test mean selection regret: `{pred['test']['selection_regret']['mean_regret']}`",
            "",
            "## Label Noise",
            "",
            f"- Half-label Pearson: `{noise['half_a_vs_half_b_pearson']}`",
            f"- Median half-label absolute difference: `{noise['median_abs_half_difference']}`",
            f"- Median within-query spread: `{noise['median_within_query_spread']}`",
            f"- Half-diff / query-spread ratio: `{noise['half_diff_to_query_spread_ratio_median']}`",
            "",
            "## Candidate Headroom",
            "",
            f"- Median top1 minus oracle: `{headroom['median_top1_minus_oracle']}`",
            f"- Oracle position median: `{headroom['oracle_position_median']}`",
            f"- Oracle position counts: `{json.dumps(headroom['oracle_position_counts'], sort_keys=True)}`",
            "",
            "## Next Step",
            "",
            decision["next_step"],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidate-bridge", type=Path, default=DEFAULT_TRAIN_CANDIDATE_BRIDGE)
    parser.add_argument("--train-label-report", type=Path, default=DEFAULT_TRAIN_LABEL_REPORT)
    parser.add_argument("--test-candidate-bridge", type=Path, default=DEFAULT_TEST_CANDIDATE_BRIDGE)
    parser.add_argument("--test-label-report", type=Path, default=DEFAULT_TEST_LABEL_REPORT)
    parser.add_argument("--test-arrays", type=Path, default=DEFAULT_TEST_ARRAYS)
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--history-arrays", type=Path, default=DEFAULT_HISTORY_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    train_candidate_bridge = _load_json(args.train_candidate_bridge)
    train_label_report = _load_json(args.train_label_report)
    test_candidate_bridge = _load_json(args.test_candidate_bridge)
    test_label_report = _load_json(args.test_label_report)
    bridge_arrays = _load_npz(args.bridge_arrays)
    history_arrays = _load_npz(args.history_arrays)
    test_arrays = _load_npz(args.test_arrays)

    predictability = predictability_report(
        train_candidate_bridge=train_candidate_bridge,
        train_label_report=train_label_report,
        test_candidate_bridge=test_candidate_bridge,
        test_label_report=test_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
    )
    noise = label_noise_report(
        candidate_label_report=test_label_report,
        arrays=test_arrays,
    )
    headroom = candidate_headroom_report(test_label_report)
    report = {
        "status": "ok",
        "research_lane": "candidate",
        "result_status": "post_experiment_analysis",
        "benchmark_floor_status": "not_applicable",
        "scope_note": (
            "Post-experiment attribution for portfolio-response support-policy "
            "failures. No OpenAI calls and no generator reruns."
        ),
        "input_artifacts": {
            "train_candidate_bridge": str(args.train_candidate_bridge),
            "train_label_report": str(args.train_label_report),
            "test_candidate_bridge": str(args.test_candidate_bridge),
            "test_label_report": str(args.test_label_report),
            "test_arrays": str(args.test_arrays),
            "bridge_arrays": str(args.bridge_arrays),
            "history_arrays": str(args.history_arrays),
        },
        "predictability": predictability,
        "label_noise": noise,
        "candidate_headroom": headroom,
    }
    report["decision"] = decide_postmortem(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_response_policy_postmortem.json"
    markdown_path = output_dir / "portfolio_response_policy_postmortem.md"
    report["artifact_paths"] = {
        "report": str(json_path),
        "markdown": str(markdown_path),
    }
    _write_json(json_path, report)
    _write_text(markdown_path, _markdown(report))
    print(json.dumps({"report": str(json_path), "status": report["decision"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
