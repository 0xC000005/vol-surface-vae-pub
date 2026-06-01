#!/usr/bin/env python
"""Compare portfolio-response candidate labels across rollout seeds.

The learned support-policy branch should only be promoted if the label surface
is stable enough to learn. This diagnostic matches candidate mixtures by
``query_id`` across two scenario-label reports and measures whether candidate
ordering survives a seed change.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LABEL_METRIC = "portfolio_reliable_path_score_z"
DEFAULT_METHOD = "narrative_generator_topk"


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


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0.0:
        return None
    return float(np.dot(aa, bb) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    return _pearson(_rankdata(a), _rankdata(b))


def _extract_rows(
    report: dict[str, Any],
    *,
    method: str,
    label_metric: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in report.get("window_scores", []):
        if not isinstance(row, dict):
            continue
        query_id = str(row.get("query_id", ""))
        methods = row.get("methods", {})
        block = methods.get(method, {}) if isinstance(methods, dict) else {}
        if not query_id or label_metric not in block:
            continue
        rows[query_id] = {
            "query_id": query_id,
            "window_id": row.get("window_id"),
            "window_index": row.get("window_index"),
            "block_window_index": row.get("block_window_index"),
            "row_no": row.get("row_no"),
            "top_train_window_ids": row.get("top_train_window_ids", []),
            "label": float(block[label_metric]),
        }
    return rows


def _base_window_id(query_id: str) -> str:
    marker = "__mixture_"
    if marker in query_id:
        return query_id.split(marker, 1)[0]
    return query_id


def _pairwise_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    agree = 0
    total = 0
    ties = 0
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            da = first[i] - first[j]
            db = second[i] - second[j]
            if da == 0.0 or db == 0.0:
                ties += 1
                continue
            total += 1
            if (da < 0.0 and db < 0.0) or (da > 0.0 and db > 0.0):
                agree += 1
    return {
        "accuracy": float(agree / total) if total else None,
        "pair_count": int(total),
        "tie_count": int(ties),
    }


def _selection_regret(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if source.size == 0:
        return {}
    selected = int(np.argmin(source))
    oracle = int(np.argmin(target))
    regret = float(target[selected] - target[oracle])
    return {
        "selected_position": selected + 1,
        "oracle_position": oracle + 1,
        "regret": regret,
        "exact_oracle": bool(selected == oracle),
    }


def compare_label_seed_stability(
    first_report: dict[str, Any],
    second_report: dict[str, Any],
    *,
    method: str = DEFAULT_METHOD,
    label_metric: str = DEFAULT_LABEL_METRIC,
) -> dict[str, Any]:
    first_rows = _extract_rows(first_report, method=method, label_metric=label_metric)
    second_rows = _extract_rows(second_report, method=method, label_metric=label_metric)
    common_ids = sorted(set(first_rows).intersection(second_rows))
    if not common_ids:
        raise ValueError("no matched query_id rows found")

    first_values = np.asarray([first_rows[q]["label"] for q in common_ids], dtype=np.float64)
    second_values = np.asarray([second_rows[q]["label"] for q in common_ids], dtype=np.float64)
    abs_diff = np.abs(first_values - second_values)

    groups: dict[str, list[str]] = defaultdict(list)
    for query_id in common_ids:
        groups[_base_window_id(query_id)].append(query_id)

    pairwise_rows: list[dict[str, Any]] = []
    first_to_second_regrets: list[float] = []
    second_to_first_regrets: list[float] = []
    top1_matches = 0
    query_spreads: list[float] = []
    abs_diff_to_spread: list[float] = []
    for base_id, ids in sorted(groups.items()):
        ids = sorted(ids)
        a = np.asarray([first_rows[q]["label"] for q in ids], dtype=np.float64)
        b = np.asarray([second_rows[q]["label"] for q in ids], dtype=np.float64)
        spread = float(max(np.ptp(a), np.ptp(b), 1e-12))
        query_spreads.append(spread)
        abs_diff_to_spread.append(float(np.median(np.abs(a - b)) / spread))
        pairwise = _pairwise_agreement(a, b)
        regret_ab = _selection_regret(a, b)
        regret_ba = _selection_regret(b, a)
        first_to_second_regrets.append(float(regret_ab["regret"]))
        second_to_first_regrets.append(float(regret_ba["regret"]))
        top1_matches += int(regret_ab["exact_oracle"])
        pairwise_rows.append(
            {
                "window_id": base_id,
                "candidate_count": len(ids),
                "pairwise_accuracy": pairwise["accuracy"],
                "pair_count": pairwise["pair_count"],
                "first_to_second_regret": regret_ab["regret"],
                "second_to_first_regret": regret_ba["regret"],
                "top1_match": regret_ab["exact_oracle"],
                "median_abs_diff_to_spread": abs_diff_to_spread[-1],
            }
        )

    pair_counts = np.asarray([row["pair_count"] for row in pairwise_rows], dtype=np.float64)
    pair_acc = np.asarray(
        [
            0.0 if row["pairwise_accuracy"] is None else float(row["pairwise_accuracy"])
            for row in pairwise_rows
        ],
        dtype=np.float64,
    )
    valid = pair_counts > 0
    weighted_pairwise = (
        float(np.sum(pair_acc[valid] * pair_counts[valid]) / np.sum(pair_counts[valid]))
        if np.any(valid)
        else None
    )
    mean_top1_regret = float(np.mean(first_to_second_regrets))
    median_top1_regret = float(np.median(first_to_second_regrets))
    top1_match_rate = float(top1_matches / len(pairwise_rows))
    result_status = "seed_stable_labels" if (
        weighted_pairwise is not None
        and weighted_pairwise >= 0.60
        and top1_match_rate >= 0.25
        and median_top1_regret <= 0.05
    ) else "seed_sensitive_labels"

    return {
        "status": "ok",
        "result_status": result_status,
        "method": method,
        "label_metric": label_metric,
        "matched_rows": len(common_ids),
        "query_count": len(pairwise_rows),
        "global": {
            "pearson": _pearson(first_values, second_values),
            "spearman": _spearman(first_values, second_values),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "median_abs_diff": float(np.median(abs_diff)),
            "mean_query_spread": float(np.mean(query_spreads)),
            "median_query_spread": float(np.median(query_spreads)),
            "median_abs_diff_to_query_spread": float(np.median(abs_diff_to_spread)),
        },
        "within_query": {
            "weighted_pairwise_accuracy": weighted_pairwise,
            "top1_match_rate": top1_match_rate,
            "first_to_second_mean_regret": mean_top1_regret,
            "first_to_second_median_regret": median_top1_regret,
            "second_to_first_mean_regret": float(np.mean(second_to_first_regrets)),
            "second_to_first_median_regret": float(np.median(second_to_first_regrets)),
        },
        "query_rows": pairwise_rows,
        "decision": {
            "promote_support_policy_training": False,
            "interpretation": (
                "Use this as a gate on whether candidate labels are stable enough "
                "for learned support weighting. A seed-sensitive label surface "
                "means the next policy should average seeds, raise samples, or "
                "learn a smoother response target before promotion."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    global_block = report["global"]
    within = report["within_query"]
    return "\n".join(
        [
            "# Portfolio-Response Label Seed Stability",
            "",
            f"Status: `{report['result_status']}`",
            "",
            "## Global Matched-Row Agreement",
            "",
            f"- matched rows: `{report['matched_rows']}`",
            f"- query count: `{report['query_count']}`",
            f"- Pearson: `{global_block['pearson']}`",
            f"- Spearman: `{global_block['spearman']}`",
            f"- median abs diff: `{global_block['median_abs_diff']}`",
            f"- median abs diff / query spread: `{global_block['median_abs_diff_to_query_spread']}`",
            "",
            "## Within-Query Ranking Stability",
            "",
            f"- weighted pairwise accuracy: `{within['weighted_pairwise_accuracy']}`",
            f"- top-1 match rate: `{within['top1_match_rate']}`",
            f"- first-to-second median regret: `{within['first_to_second_median_regret']}`",
            f"- second-to-first median regret: `{within['second_to_first_median_regret']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-report", required=True)
    parser.add_argument("--second-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--label-metric", default=DEFAULT_LABEL_METRIC)
    args = parser.parse_args()

    report = compare_label_seed_stability(
        _load_json(args.first_report),
        _load_json(args.second_report),
        method=args.method,
        label_metric=args.label_metric,
    )
    report["artifact_paths"] = {
        "first_report": str(args.first_report),
        "second_report": str(args.second_report),
        "report": str(Path(args.output_dir) / "portfolio_response_label_seed_stability.json"),
        "markdown": str(Path(args.output_dir) / "portfolio_response_label_seed_stability.md"),
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "portfolio_response_label_seed_stability.json", report)
    _write_text(output_dir / "portfolio_response_label_seed_stability.md", _markdown(report))
    print(json.dumps({"report": report["artifact_paths"]["report"], "status": report["result_status"]}))


if __name__ == "__main__":
    main()
