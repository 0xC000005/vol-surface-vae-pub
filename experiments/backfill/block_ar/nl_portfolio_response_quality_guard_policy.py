#!/usr/bin/env python
"""Build a quality-guarded portfolio-response support-policy bridge.

This is a constrained follow-up to the support-reliability prior.  It keeps the
auditable support-mixture contract, but scores candidate mixtures with two
train-only signals:

1. portfolio-response utility, learned from historical portfolio backtests;
2. broad scenario-quality utility, learned from CRPS and energy backtests.

The candidate utility is portfolio utility plus a one-sided penalty when broad
quality utilities are negative.  This avoids adding another learned ranker while
testing the central question: can the portfolio-response overlay keep its
portfolio gain without weakening the frozen generator's broad scenario quality?

No OpenAI calls and no generator calls are made here.  The output bridge is
consumed by ``nl_scenario_level_evaluation.py``.
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
from experiments.backfill.block_ar.nl_portfolio_response_support_reliability_policy import (  # noqa: E402
    DEFAULT_BASE_BRIDGE_REPORT,
    DEFAULT_BRIDGE_ARRAYS,
    DEFAULT_CANDIDATE_BRIDGE_REPORT,
    DEFAULT_HISTORY_ARRAYS,
    DEFAULT_TRAIN_CANDIDATE_BRIDGE,
    DEFAULT_TRAIN_LABEL_REPORT,
    _entropy,
    _group_candidate_rows,
    _load_json,
    _load_npz,
    _softmax,
    _support_indices,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_quality_guard_924a"
)
BROAD_QUALITY_METRICS = ("ensemble_crps_z", "energy_score_z")


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


def _build_train_table_for_metric(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
    metric: str,
) -> Any:
    return build_mixture_policy_training_table(
        candidate_bridge=train_candidate_bridge,
        scenario_report=train_label_report,
        condition_vectors=np.asarray(bridge_arrays["condition_vectors"], dtype=np.float32),
        memory_targets=np.asarray(bridge_arrays["memory_targets"], dtype=np.float32),
        history_level=np.asarray(history_arrays["history_level"], dtype=np.float32),
        method="narrative_generator_topk",
        metric=metric,
    )


def _support_prior_for_metric(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
    metric: str,
) -> dict[int, float]:
    table = _build_train_table_for_metric(
        train_candidate_bridge=train_candidate_bridge,
        train_label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
        metric=metric,
    )
    return _support_reliability_prior(
        support_rows=[
            list(map(int, row["support_window_indices"])) for row in table.rows
        ],
        labels=np.asarray(table.labels, dtype=np.float64),
        query_ids=[row["window_index"] for row in table.rows],
    )


def quality_guard_scores(
    *,
    support_rows: list[list[int]],
    portfolio_prior: dict[int, float],
    crps_prior: dict[int, float],
    energy_prior: dict[int, float],
) -> dict[str, np.ndarray]:
    """Return one-sided quality-guard candidate scores.

    All priors are higher-is-better utilities because the training table stores
    negative rollout losses.  The portfolio signal drives ranking, while CRPS
    and energy only penalize candidates whose train-derived support quality is
    below average.  There is deliberately no tunable penalty weight in this
    first gate.
    """

    portfolio = _score_support_prior(support_rows, portfolio_prior)
    crps = _score_support_prior(support_rows, crps_prior)
    energy = _score_support_prior(support_rows, energy_prior)
    quality_penalty = np.minimum(crps, 0.0) + np.minimum(energy, 0.0)
    total = portfolio + quality_penalty
    return {
        "total": np.asarray(total, dtype=np.float64),
        "portfolio": np.asarray(portfolio, dtype=np.float64),
        "crps": np.asarray(crps, dtype=np.float64),
        "energy": np.asarray(energy, dtype=np.float64),
        "quality_penalty": np.asarray(quality_penalty, dtype=np.float64),
    }


def select_quality_guard_support(
    candidates: list[dict[str, Any]],
    *,
    portfolio_prior: dict[int, float],
    crps_prior: dict[int, float],
    energy_prior: dict[int, float],
    probability_temperature: float = 0.25,
    max_candidate_entropy_threshold: float | None = None,
    max_candidate_entropy_quantile: float | None = None,
    min_support_weight_max_threshold: float | None = None,
    min_support_weight_max_quantile: float | None = None,
) -> dict[str, Any]:
    """Select support with the train-only quality guard.

    This is the reusable 924e policy core.  It is deliberately independent of
    bridge-report mutation so the same contract can be used by offline
    historical-query bridges and, later, by live narrative support selection.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")
    support_rows = [_support_indices(candidate) for candidate in candidates]
    score_block = quality_guard_scores(
        support_rows=support_rows,
        portfolio_prior=portfolio_prior,
        crps_prior=crps_prior,
        energy_prior=energy_prior,
    )
    scores = score_block["total"]
    probs_for_gate = _softmax(scores, temperature=float(probability_temperature))
    entropy_for_gate = _entropy(probs_for_gate)
    if (
        max_candidate_entropy_threshold is not None
        and entropy_for_gate > float(max_candidate_entropy_threshold)
    ):
        return {
            "fallback": True,
            "selected": None,
            "candidate_probabilities": [],
            "support_policy": {
                "name": "portfolio_response_quality_guard_prior",
                "policy_kind": "train_only_portfolio_plus_quality_guard",
                "probability_temperature": float(probability_temperature),
                "candidate_entropy": float(entropy_for_gate),
                "max_candidate_entropy_threshold": float(
                    max_candidate_entropy_threshold
                ),
                "max_candidate_entropy_quantile": (
                    None
                    if max_candidate_entropy_quantile is None
                    else float(max_candidate_entropy_quantile)
                ),
                "fallback_to_equal_support": True,
                "candidate_count": len(candidates),
            },
        }

    selected, probs = listwise_support_weights_for_candidates(
        candidates,
        scores,
        probability_temperature=float(probability_temperature),
    )
    selected_weights = [
        float(item.get("weight", 0.0) or 0.0)
        for item in selected
        if isinstance(item, dict)
    ]
    support_weight_max = float(max(selected_weights)) if selected_weights else 0.0
    if (
        min_support_weight_max_threshold is not None
        and support_weight_max <= float(min_support_weight_max_threshold)
    ):
        return {
            "fallback": True,
            "selected": None,
            "candidate_probabilities": [],
            "support_policy": {
                "name": "portfolio_response_quality_guard_prior",
                "policy_kind": "train_only_portfolio_plus_quality_guard",
                "probability_temperature": float(probability_temperature),
                "candidate_entropy": float(entropy_for_gate),
                "max_candidate_entropy_threshold": (
                    None
                    if max_candidate_entropy_threshold is None
                    else float(max_candidate_entropy_threshold)
                ),
                "max_candidate_entropy_quantile": (
                    None
                    if max_candidate_entropy_quantile is None
                    else float(max_candidate_entropy_quantile)
                ),
                "support_weight_max": float(support_weight_max),
                "min_support_weight_max_threshold": float(
                    min_support_weight_max_threshold
                ),
                "min_support_weight_max_quantile": (
                    None
                    if min_support_weight_max_quantile is None
                    else float(min_support_weight_max_quantile)
                ),
                "fallback_to_equal_support": True,
                "candidate_count": len(candidates),
            },
        }

    best_pos = int(np.argmax(probs))
    candidate_probabilities = [
        {
            "query_id": str(candidate.get("query_id", "")),
            "probability": float(prob),
            "score": float(score),
            "portfolio_score": float(portfolio_score),
            "crps_score": float(crps_score),
            "energy_score": float(energy_score),
            "quality_penalty": float(quality_penalty),
        }
        for candidate, prob, score, portfolio_score, crps_score, energy_score, quality_penalty in zip(
            candidates,
            probs,
            score_block["total"],
            score_block["portfolio"],
            score_block["crps"],
            score_block["energy"],
            score_block["quality_penalty"],
            strict=True,
        )
    ]
    return {
        "fallback": False,
        "selected": selected,
        "candidate_probabilities": candidate_probabilities,
        "support_policy": {
            "name": "portfolio_response_quality_guard_prior",
            "policy_kind": "train_only_portfolio_plus_quality_guard",
            "probability_temperature": float(probability_temperature),
            "candidate_entropy": float(entropy_for_gate),
            "max_candidate_entropy_threshold": (
                None
                if max_candidate_entropy_threshold is None
                else float(max_candidate_entropy_threshold)
            ),
            "max_candidate_entropy_quantile": (
                None
                if max_candidate_entropy_quantile is None
                else float(max_candidate_entropy_quantile)
            ),
            "fallback_to_equal_support": False,
            "candidate_count": len(candidates),
            "support_weight_max": float(support_weight_max),
            "min_support_weight_max_threshold": (
                None
                if min_support_weight_max_threshold is None
                else float(min_support_weight_max_threshold)
            ),
            "min_support_weight_max_quantile": (
                None
                if min_support_weight_max_quantile is None
                else float(min_support_weight_max_quantile)
            ),
            "selected_query_id": str(candidates[best_pos].get("query_id", "")),
            "selected_score": float(scores[best_pos]),
            "candidate_probabilities": candidate_probabilities,
            "selected_support_window_indices": [
                int(item["window_index"]) for item in selected
            ],
            "selected_support_weights": [
                float(item.get("weight", 0.0) or 0.0) for item in selected
            ],
        },
    }


def build_quality_guard_policy_context(
    *,
    train_candidate_bridge: dict[str, Any] | None = None,
    train_label_report: dict[str, Any] | None = None,
    bridge_arrays: dict[str, np.ndarray] | None = None,
    history_arrays: dict[str, np.ndarray] | None = None,
    train_candidate_bridge_path: str | Path = DEFAULT_TRAIN_CANDIDATE_BRIDGE,
    train_label_report_path: str | Path = DEFAULT_TRAIN_LABEL_REPORT,
    bridge_arrays_path: str | Path = DEFAULT_BRIDGE_ARRAYS,
    history_arrays_path: str | Path = DEFAULT_HISTORY_ARRAYS,
    probability_temperature: float = 0.25,
    max_candidate_entropy_quantile: float | None = None,
    min_support_weight_max_quantile: float | None = 0.25,
) -> dict[str, Any]:
    """Build the train-only policy context used by the 924e guard."""

    train_candidate = (
        train_candidate_bridge
        if train_candidate_bridge is not None
        else _load_json(train_candidate_bridge_path)
    )
    train_labels = (
        train_label_report
        if train_label_report is not None
        else _load_json(train_label_report_path)
    )
    bridge_npz = (
        bridge_arrays if bridge_arrays is not None else _load_npz(bridge_arrays_path)
    )
    history_npz = (
        history_arrays if history_arrays is not None else _load_npz(history_arrays_path)
    )
    portfolio_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=train_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric=PORTFOLIO_LABEL_METRIC,
    )
    crps_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=train_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric="ensemble_crps_z",
    )
    energy_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=train_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric="energy_score_z",
    )
    train_groups = _group_candidate_rows(train_candidate)

    entropy_threshold: float | None = None
    if max_candidate_entropy_quantile is not None:
        entropies: list[float] = []
        for candidates in train_groups.values():
            support_rows = [_support_indices(candidate) for candidate in candidates]
            scores = quality_guard_scores(
                support_rows=support_rows,
                portfolio_prior=portfolio_prior,
                crps_prior=crps_prior,
                energy_prior=energy_prior,
            )["total"]
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

    support_weight_max_threshold: float | None = None
    if min_support_weight_max_quantile is not None:
        max_weights: list[float] = []
        for candidates in train_groups.values():
            support_rows = [_support_indices(candidate) for candidate in candidates]
            scores = quality_guard_scores(
                support_rows=support_rows,
                portfolio_prior=portfolio_prior,
                crps_prior=crps_prior,
                energy_prior=energy_prior,
            )["total"]
            selected, _ = listwise_support_weights_for_candidates(
                candidates,
                scores,
                probability_temperature=float(probability_temperature),
            )
            weights = [
                float(item.get("weight", 0.0) or 0.0)
                for item in selected
                if isinstance(item, dict)
            ]
            if weights:
                max_weights.append(float(max(weights)))
        if not max_weights:
            raise ValueError("could not compute train support max-weight threshold")
        support_weight_max_threshold = float(
            np.quantile(
                np.asarray(max_weights, dtype=np.float64),
                float(min_support_weight_max_quantile),
            )
        )

    return {
        "portfolio_prior": portfolio_prior,
        "crps_prior": crps_prior,
        "energy_prior": energy_prior,
        "probability_temperature": float(probability_temperature),
        "max_candidate_entropy_quantile": (
            None
            if max_candidate_entropy_quantile is None
            else float(max_candidate_entropy_quantile)
        ),
        "max_candidate_entropy_threshold": entropy_threshold,
        "min_support_weight_max_quantile": (
            None
            if min_support_weight_max_quantile is None
            else float(min_support_weight_max_quantile)
        ),
        "min_support_weight_max_threshold": support_weight_max_threshold,
        "source_artifacts": {
            "train_candidate_bridge": str(train_candidate_bridge_path),
            "train_label_report": str(train_label_report_path),
            "bridge_arrays": str(bridge_arrays_path),
            "history_arrays": str(history_arrays_path),
        },
    }


def build_quality_guard_bridge(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
    base_bridge_report: dict[str, Any],
    candidate_bridge_report: dict[str, Any],
    bridge_arrays: dict[str, np.ndarray],
    history_arrays: dict[str, np.ndarray],
    probability_temperature: float = 0.25,
    max_candidate_entropy_quantile: float | None = None,
    min_support_weight_max_quantile: float | None = None,
    source_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    portfolio_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate_bridge,
        train_label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
        metric=PORTFOLIO_LABEL_METRIC,
    )
    crps_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate_bridge,
        train_label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
        metric="ensemble_crps_z",
    )
    energy_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate_bridge,
        train_label_report=train_label_report,
        bridge_arrays=bridge_arrays,
        history_arrays=history_arrays,
        metric="energy_score_z",
    )

    train_groups = _group_candidate_rows(train_candidate_bridge)
    entropy_threshold: float | None = None
    support_weight_max_threshold: float | None = None
    if max_candidate_entropy_quantile is not None:
        entropies: list[float] = []
        for candidates in train_groups.values():
            support_rows = [_support_indices(candidate) for candidate in candidates]
            scores = quality_guard_scores(
                support_rows=support_rows,
                portfolio_prior=portfolio_prior,
                crps_prior=crps_prior,
                energy_prior=energy_prior,
            )["total"]
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
    if min_support_weight_max_quantile is not None:
        max_weights: list[float] = []
        for candidates in train_groups.values():
            support_rows = [_support_indices(candidate) for candidate in candidates]
            scores = quality_guard_scores(
                support_rows=support_rows,
                portfolio_prior=portfolio_prior,
                crps_prior=crps_prior,
                energy_prior=energy_prior,
            )["total"]
            selected, _ = listwise_support_weights_for_candidates(
                candidates,
                scores,
                probability_temperature=float(probability_temperature),
            )
            weights = [
                float(item.get("weight", 0.0) or 0.0)
                for item in selected
                if isinstance(item, dict)
            ]
            if weights:
                max_weights.append(float(max(weights)))
        if not max_weights:
            raise ValueError("could not compute train support max-weight threshold")
        support_weight_max_threshold = float(
            np.quantile(
                np.asarray(max_weights, dtype=np.float64),
                float(min_support_weight_max_quantile),
            )
        )

    groups = _group_candidate_rows(candidate_bridge_report)
    output = json.loads(json.dumps(base_bridge_report))
    rows = output.get("evaluation", {}).get("heldout_examples", [])
    changed = 0
    fallback_count = 0
    active_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        window_index = int(row.get("window_index", -1))
        candidates = groups.get(window_index)
        if not candidates:
            continue
        selection = select_quality_guard_support(
            candidates,
            portfolio_prior=portfolio_prior,
            crps_prior=crps_prior,
            energy_prior=energy_prior,
            probability_temperature=float(probability_temperature),
            max_candidate_entropy_threshold=entropy_threshold,
            max_candidate_entropy_quantile=max_candidate_entropy_quantile,
            min_support_weight_max_threshold=support_weight_max_threshold,
            min_support_weight_max_quantile=min_support_weight_max_quantile,
        )
        row["support_policy"] = selection["support_policy"]
        if bool(selection["fallback"]):
            changed += 1
            fallback_count += 1
            continue
        row["top_train_pool"] = selection["selected"]
        changed += 1
        active_count += 1

    output["mixture_policy"] = {
        "name": "portfolio_response_quality_guard_prior",
        "research_lane": "candidate",
        "policy_kind": "train_only_portfolio_plus_quality_guard",
        "candidate_rows_reranked": int(changed),
        "active_rows": int(active_count),
        "fallback_rows": int(fallback_count),
        "known_support_count": {
            "portfolio": int(len(portfolio_prior)),
            "crps": int(len(crps_prior)),
            "energy": int(len(energy_prior)),
        },
        "probability_temperature": float(probability_temperature),
        "max_candidate_entropy_quantile": (
            None
            if max_candidate_entropy_quantile is None
            else float(max_candidate_entropy_quantile)
        ),
        "max_candidate_entropy_threshold": (
            None if entropy_threshold is None else float(entropy_threshold)
        ),
        "min_support_weight_max_quantile": (
            None
            if min_support_weight_max_quantile is None
            else float(min_support_weight_max_quantile)
        ),
        "min_support_weight_max_threshold": (
            None
            if support_weight_max_threshold is None
            else float(support_weight_max_threshold)
        ),
        "label_metric": PORTFOLIO_LABEL_METRIC,
        "quality_metrics": list(BROAD_QUALITY_METRICS),
        "score_contract": (
            "portfolio utility plus one-sided penalties for negative "
            "train-derived CRPS and energy support utility"
        ),
        "source_artifacts": dict(source_artifacts or {}),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidate-bridge", type=Path, default=DEFAULT_TRAIN_CANDIDATE_BRIDGE)
    parser.add_argument("--train-label-report", type=Path, default=DEFAULT_TRAIN_LABEL_REPORT)
    parser.add_argument("--base-bridge-report", type=Path, default=DEFAULT_BASE_BRIDGE_REPORT)
    parser.add_argument("--candidate-bridge-report", type=Path, default=DEFAULT_CANDIDATE_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--history-arrays", type=Path, default=DEFAULT_HISTORY_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probability-temperature", type=float, default=0.25)
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
    parser.add_argument(
        "--min-support-weight-max-quantile",
        type=float,
        default=None,
        help=(
            "Optional train-set support-concentration gate. If the held-out "
            "final support distribution's max weight is below this train "
            "quantile, leave the base/equal support pool unchanged."
        ),
    )
    args = parser.parse_args()

    bridge = build_quality_guard_bridge(
        train_candidate_bridge=_load_json(args.train_candidate_bridge),
        train_label_report=_load_json(args.train_label_report),
        base_bridge_report=_load_json(args.base_bridge_report),
        candidate_bridge_report=_load_json(args.candidate_bridge_report),
        bridge_arrays=_load_npz(args.bridge_arrays),
        history_arrays=_load_npz(args.history_arrays),
        probability_temperature=args.probability_temperature,
        max_candidate_entropy_quantile=args.max_candidate_entropy_quantile,
        min_support_weight_max_quantile=args.min_support_weight_max_quantile,
        source_artifacts={
            "train_candidate_bridge": str(args.train_candidate_bridge),
            "train_label_report": str(args.train_label_report),
            "base_bridge_report": str(args.base_bridge_report),
            "candidate_bridge_report": str(args.candidate_bridge_report),
            "bridge_arrays": str(args.bridge_arrays),
            "history_arrays": str(args.history_arrays),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "quality_guard_policy_bridge_report.json"
    _write_json(report_path, bridge)
    print(
        json.dumps(
            {
                "bridge_report": str(report_path),
                "candidate_rows_reranked": bridge["mixture_policy"]["candidate_rows_reranked"],
                "active_rows": bridge["mixture_policy"]["active_rows"],
                "fallback_rows": bridge["mixture_policy"]["fallback_rows"],
                "probability_temperature": args.probability_temperature,
                "max_candidate_entropy_quantile": args.max_candidate_entropy_quantile,
                "min_support_weight_max_quantile": args.min_support_weight_max_quantile,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
