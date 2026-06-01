#!/usr/bin/env python
"""Build a support-reliability-prior bridge for NL scenario evaluation.

The policy is intentionally small and auditable:

1. build train candidate labels from historical backtests;
2. learn a train-only reliability prior for each support window ID;
3. score each candidate support mixture by the mean reliability of its supports;
4. either select the best candidate or softly marginalize candidate probabilities
   into support weights;
5. write a bridge report consumable by ``nl_scenario_level_evaluation.py``.

No OpenAI calls and no generator calls are made here. The output bridge can then
be evaluated by the standard frozen SNI scenario-level evaluation script.
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

from experiments.backfill.block_ar.nl_learned_mixture_policy_testflight import (  # noqa: E402
    build_mixture_policy_training_table,
    listwise_support_weights_for_candidates,
)
from experiments.backfill.block_ar.nl_portfolio_response_feature_sufficiency import (  # noqa: E402
    _score_support_prior,
    _support_reliability_prior,
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
DEFAULT_BASE_BRIDGE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_test_query_bridge/query_bridge_report.json"
)
DEFAULT_CANDIDATE_BRIDGE_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_full_906e_test_mixture_candidates/"
    "mixture_label_bridge_report.json"
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
    "nl_portfolio_response_support_reliability_923b"
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


def _support_indices(row: dict[str, Any]) -> list[int]:
    raw = row.get("candidate_support_window_indices")
    if isinstance(raw, list) and raw:
        return [int(value) for value in raw]
    return [
        int(item["window_index"])
        for item in row.get("top_train_pool", [])
        if isinstance(item, dict) and item.get("window_index") is not None
    ]


def _group_candidate_rows(candidate_bridge: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_bridge.get("evaluation", {}).get("heldout_examples", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            groups.setdefault(int(row["window_index"]), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: int(item.get("candidate_mixture_rank", 10**9)))
    return groups


def _build_train_table(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
) -> Any:
    return build_mixture_policy_training_table(
        candidate_bridge=train_candidate_bridge,
        scenario_report=train_label_report,
        condition_vectors=np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32),
        memory_targets=np.asarray(bridge_arrays["memory_targets"], dtype=np.float32),
        history_level=np.asarray(history_arrays["history_level"], dtype=np.float32),
        method="narrative_generator_topk",
        metric=PORTFOLIO_LABEL_METRIC,
    )


def build_support_reliability_bridge(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
    base_bridge_report: dict[str, Any],
    candidate_bridge_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
    selection_mode: str = "candidate_softmax",
    probability_temperature: float = 1.0,
    max_candidate_entropy_quantile: float | None = None,
) -> dict[str, Any]:
    train = _build_train_table(
        train_candidate_bridge=train_candidate_bridge,
        train_label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
    )
    train_query_ids = [row["window_index"] for row in train.rows]
    train_support_rows = [
        list(map(int, row["support_window_indices"])) for row in train.rows
    ]
    prior = _support_reliability_prior(
        support_rows=train_support_rows,
        labels=np.asarray(train.labels, dtype=np.float64),
        query_ids=train_query_ids,
    )
    train_groups = _group_candidate_rows(train_candidate_bridge)
    entropy_threshold: float | None = None
    if max_candidate_entropy_quantile is not None:
        entropies: list[float] = []
        for candidates in train_groups.values():
            support_rows = [_support_indices(candidate) for candidate in candidates]
            scores = _score_support_prior(support_rows, prior)
            probs = _softmax(scores, temperature=float(probability_temperature))
            entropies.append(_entropy(probs))
        if not entropies:
            raise ValueError("could not compute train candidate entropy threshold")
        entropy_threshold = float(
            np.quantile(
                np.asarray(entropies, dtype=np.float64),
                float(max_candidate_entropy_quantile),
            )
        )
    groups = _group_candidate_rows(candidate_bridge_report)
    output = json.loads(json.dumps(base_bridge_report))
    rows = output.get("evaluation", {}).get("heldout_examples", [])
    changed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        window_index = int(row.get("window_index", -1))
        candidates = groups.get(window_index)
        if not candidates:
            continue
        support_rows = [_support_indices(candidate) for candidate in candidates]
        scores = _score_support_prior(support_rows, prior)
        probs_for_gate = _softmax(scores, temperature=float(probability_temperature))
        entropy_for_gate = _entropy(probs_for_gate)
        if entropy_threshold is not None and entropy_for_gate > entropy_threshold:
            row["support_policy"] = {
                "name": "portfolio_response_support_reliability_prior",
                "policy_kind": "train_only_support_reliability_prior",
                "selection_mode": str(selection_mode),
                "probability_temperature": float(probability_temperature),
                "candidate_entropy": float(entropy_for_gate),
                "max_candidate_entropy_threshold": float(entropy_threshold),
                "max_candidate_entropy_quantile": float(max_candidate_entropy_quantile),
                "fallback_to_equal_support": True,
                "candidate_count": len(candidates),
            }
            changed += 1
            continue
        if selection_mode == "best_candidate":
            best_pos = int(np.argmax(scores))
            selected = json.loads(json.dumps(candidates[best_pos].get("top_train_pool", [])))
            count = max(len(selected), 1)
            for item in selected:
                item["weight"] = float(1.0 / count)
            candidate_probabilities = [
                {
                    "query_id": str(candidate.get("query_id", "")),
                    "probability": float(1.0 if idx == best_pos else 0.0),
                    "score": float(score),
                }
                for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
            ]
            row["top_train_pool"] = selected
        elif selection_mode == "candidate_softmax":
            selected, probs = listwise_support_weights_for_candidates(
                candidates,
                scores,
                probability_temperature=float(probability_temperature),
            )
            best_pos = int(np.argmax(probs))
            candidate_probabilities = [
                {
                    "query_id": str(candidate.get("query_id", "")),
                    "probability": float(prob),
                    "score": float(score),
                }
                for candidate, prob, score in zip(candidates, probs, scores, strict=True)
            ]
            row["top_train_pool"] = selected
        else:
            raise ValueError(f"unsupported selection mode: {selection_mode}")
        row["support_policy"] = {
            "name": "portfolio_response_support_reliability_prior",
            "policy_kind": "train_only_support_reliability_prior",
            "selection_mode": str(selection_mode),
            "probability_temperature": float(probability_temperature),
            "candidate_entropy": float(entropy_for_gate),
            "max_candidate_entropy_threshold": (
                None if entropy_threshold is None else float(entropy_threshold)
            ),
            "max_candidate_entropy_quantile": (
                None
                if max_candidate_entropy_quantile is None
                else float(max_candidate_entropy_quantile)
            ),
            "fallback_to_equal_support": False,
            "candidate_count": len(candidates),
            "selected_query_id": str(candidates[best_pos].get("query_id", "")),
            "selected_score": float(scores[best_pos]),
            "candidate_probabilities": candidate_probabilities,
            "selected_support_window_indices": [
                int(item["window_index"]) for item in row.get("top_train_pool", [])
            ],
            "selected_support_weights": [
                float(item.get("weight", 0.0) or 0.0)
                for item in row.get("top_train_pool", [])
            ],
        }
        changed += 1
    output["mixture_policy"] = {
        "name": "portfolio_response_support_reliability_prior",
        "research_lane": "candidate",
        "policy_kind": "train_only_support_reliability_prior",
        "selection_mode": str(selection_mode),
        "candidate_rows_reranked": int(changed),
        "train_rows": int(len(train.rows)),
        "train_queries": int(len(set(train_query_ids))),
        "known_support_count": int(len(prior)),
        "probability_temperature": float(probability_temperature),
        "max_candidate_entropy_quantile": (
            None
            if max_candidate_entropy_quantile is None
            else float(max_candidate_entropy_quantile)
        ),
        "max_candidate_entropy_threshold": (
            None if entropy_threshold is None else float(entropy_threshold)
        ),
        "label_metric": PORTFOLIO_LABEL_METRIC,
    }
    return output


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-8)
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    shifted = (raw - float(np.max(raw))) / temp
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(raw.shape, 1.0 / max(raw.size, 1), dtype=np.float64)
    return exp_values / denom


def _entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 0.0
    arr = arr / float(np.sum(arr))
    return float(-np.sum(arr * np.log(arr)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidate-bridge", type=Path, default=DEFAULT_TRAIN_CANDIDATE_BRIDGE)
    parser.add_argument("--train-label-report", type=Path, default=DEFAULT_TRAIN_LABEL_REPORT)
    parser.add_argument("--base-bridge-report", type=Path, default=DEFAULT_BASE_BRIDGE_REPORT)
    parser.add_argument("--candidate-bridge-report", type=Path, default=DEFAULT_CANDIDATE_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--history-arrays", type=Path, default=DEFAULT_HISTORY_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection-mode",
        choices=["candidate_softmax", "best_candidate"],
        default="candidate_softmax",
    )
    parser.add_argument("--probability-temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-candidate-entropy-quantile",
        type=float,
        default=None,
        help=(
            "Optional train-set entropy quantile guard. If a held-out query's "
            "candidate-probability entropy is above this train quantile, leave "
            "the base/equal support pool unchanged."
        ),
    )
    args = parser.parse_args()

    bridge = build_support_reliability_bridge(
        train_candidate_bridge=_load_json(args.train_candidate_bridge),
        train_label_report=_load_json(args.train_label_report),
        base_bridge_report=_load_json(args.base_bridge_report),
        candidate_bridge_report=_load_json(args.candidate_bridge_report),
        bridge_arrays=_load_npz(args.bridge_arrays),
        history_arrays=_load_npz(args.history_arrays),
        selection_mode=args.selection_mode,
        probability_temperature=args.probability_temperature,
        max_candidate_entropy_quantile=args.max_candidate_entropy_quantile,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "support_reliability_policy_bridge_report.json"
    _write_json(report_path, bridge)
    print(
        json.dumps(
            {
                "bridge_report": str(report_path),
                "candidate_rows_reranked": bridge["mixture_policy"]["candidate_rows_reranked"],
                "known_support_count": bridge["mixture_policy"]["known_support_count"],
                "selection_mode": args.selection_mode,
                "max_candidate_entropy_quantile": args.max_candidate_entropy_quantile,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
