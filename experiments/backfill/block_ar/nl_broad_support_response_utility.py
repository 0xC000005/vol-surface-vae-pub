#!/usr/bin/env python
"""Train a broad-bank response-utility prior for NL support selection.

This is a cheap TestFlight for the next response-aware support-weighting idea.
It does not call OpenAI and does not run the frozen generator.  Instead, it
builds broad support-bank candidate mixtures and labels them with historical
future-delta replay against held-out historical query windows.  The resulting
support-level utility priors can be consumed by the story-smoke support selector
as a pre-rollout scorer.

The label is deliberately proxy-level, not a promotion claim: it asks whether
the broad 4010-window support inventory contains a trainable signal that can
rank candidate support sets before final SNI rollout.
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

from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    FACTOR_INDEX,
    PORTFOLIO_EXPOSURES,
)
from experiments.backfill.block_ar.nl_learned_mixture_policy_testflight import (  # noqa: E402
    listwise_support_weights_for_candidates,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KEY_FACTOR_NAMES,
)
from experiments.backfill.block_ar.nl_portfolio_response_policy_postmortem import (  # noqa: E402
    pairwise_preference_accuracy,
    selection_regret,
)
from experiments.backfill.block_ar.nl_portfolio_response_quality_guard_policy import (  # noqa: E402
    _entropy,
    _softmax,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (  # noqa: E402
    _portfolio_quality_guard_candidate_mixtures,
    candidate_support_table,
    load_state_spec_names,
)


DEFAULT_SUPPORT_BANK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_SUPPORT_BANK_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_broad_support_response_utility_940a"
)
DEFAULT_CONTEXT_PATH = DEFAULT_OUTPUT_DIR / "broad_support_response_utility_context.json"

MARKET_SIGN = {
    "up": 1,
    "higher": 1,
    "wider": 1,
    "steeper": 1,
    "down": -1,
    "lower": -1,
    "tighter": -1,
    "narrower": -1,
    "flatter": -1,
}


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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _direction_from_delta(delta: float, *, positive_word: str = "up") -> str:
    if abs(float(delta)) <= 1.0e-8:
        return "flat"
    if positive_word == "wider":
        return "wider" if delta > 0 else "tighter"
    return "up" if delta > 0 else "down"


def _magnitude(delta: float, scale: float) -> str:
    ratio = abs(float(delta)) / max(abs(float(scale)), 1.0)
    if ratio >= 0.06:
        return "large"
    if ratio >= 0.02:
        return "medium"
    return "small"


def grounding_from_history(
    *,
    history_level: np.ndarray,
    window_index: int,
    spec_names: list[str],
) -> dict[str, Any]:
    """Build a factual grounding sidecar from the observed 30-day prefix."""

    history = np.asarray(history_level, dtype=np.float32)
    idx = int(window_index)
    start = history[idx, 0]
    terminal = history[idx, -1]
    spec_index = {name: pos for pos, name in enumerate(spec_names)}
    rows: list[dict[str, Any]] = []
    for market, spec_name in KEY_FACTOR_NAMES.items():
        col = spec_index.get(spec_name)
        if col is None:
            continue
        delta = float(terminal[col] - start[col])
        positive_word = "wider" if market in {"BBB_OAS", "AAA_OAS"} else "up"
        direction = _direction_from_delta(delta, positive_word=positive_word)
        if direction == "flat":
            continue
        rows.append(
            {
                "market": market,
                "direction": direction,
                "magnitude": _magnitude(delta, float(start[col])),
                "confidence": "high",
                "horizon": "30 historical trading days ending at the conditioning date",
                "evidence": [f"{market}_30d_delta={delta:.6g}"],
                "target_use": "support_prior",
            }
        )
    return {
        "condition_only_grounding": {
            "current_market_state_implications": rows,
            "non_conditioning_forward_language": [],
            "grounding_warnings": [],
        }
    }


def _channels_from_grounding(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    payload = grounding.get("condition_only_grounding", grounding)
    rows = payload.get("current_market_state_implications", [])
    channels: list[dict[str, Any]] = []
    factor_by_market = {market.upper(): name for market, name in {
        "SPX": "SPX",
        "VIX": "VIX",
        "DXY": "DXY",
        "USDJPY": "DXY",
        "CRUDE_OIL": "Crude",
        "OIL": "Crude",
        "US10Y": "US10Y",
        "BBB_OAS": "BBB_OAS",
        "GOLD": "Gold",
        "IV_ATM_1Y": "IV_ATM_1Y",
    }.items()}
    for row in rows:
        if not isinstance(row, dict):
            continue
        factor = factor_by_market.get(str(row.get("market", "")).upper())
        if factor not in FACTOR_INDEX:
            continue
        direction = str(row.get("direction", "")).lower()
        sign = MARKET_SIGN.get(direction, 0)
        if sign == 0:
            continue
        weight = 1.0
        if str(row.get("magnitude", "")).lower() == "large":
            weight = 1.2
        elif str(row.get("magnitude", "")).lower() == "small":
            weight = 0.75
        channels.append({"factor": factor, "sign": float(sign), "weight": weight})
    return channels


def _normalize_future_delta(
    future_delta: np.ndarray,
    *,
    scale: np.ndarray,
) -> np.ndarray:
    denom = np.maximum(np.asarray(scale, dtype=np.float64).reshape(1, 1, -1), 1.0)
    return np.asarray(future_delta, dtype=np.float64) / denom


def _ensemble_crps(samples: np.ndarray, observed: np.ndarray) -> float:
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64).reshape(1, *x.shape[1:])
    term1 = float(np.mean(np.abs(x - y)))
    pair = np.abs(x[:, None, :, :] - x[None, :, :, :])
    term2 = 0.5 * float(np.mean(pair))
    return term1 - term2


def _energy_score(samples: np.ndarray, observed: np.ndarray) -> float:
    x = np.asarray(samples, dtype=np.float64).reshape(samples.shape[0], -1)
    y = np.asarray(observed, dtype=np.float64).reshape(1, -1)
    term1 = float(np.mean(np.linalg.norm(x - y, axis=1)))
    pair = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    term2 = 0.5 * float(np.mean(pair))
    return term1 - term2


def _portfolio_path_from_delta(
    *,
    future_delta: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(future_delta, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    terms = []
    for exposure in PORTFOLIO_EXPOSURES:
        idx = int(exposure["index"])
        scale = max(abs(float(start_arr[idx])), 1.0)
        terms.append(delta[:, :, idx] / scale * float(exposure["sensitivity"]) * 100.0)
    return np.sum(np.stack(terms, axis=-1), axis=-1)


def _response_score(
    *,
    samples_delta: np.ndarray,
    start: np.ndarray,
    channels: list[dict[str, Any]],
) -> float:
    if not channels:
        return 0.0
    delta = np.asarray(samples_delta, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    pieces = []
    weights = []
    for channel in channels:
        idx = FACTOR_INDEX[str(channel["factor"])]
        scale = max(abs(float(start_arr[idx])), 1.0)
        terminal = delta[:, -1, idx] / scale
        path = delta[:, :, idx] / scale
        sign = float(channel["sign"])
        signed = sign * float(np.median(terminal))
        activation = abs(float(np.median(terminal)))
        width = float(np.percentile(path, 90) - np.percentile(path, 10))
        pieces.append(0.55 * signed + 0.25 * activation + 0.20 * width)
        weights.append(float(channel.get("weight", 1.0)))
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(pieces, dtype=np.float64)
    return float(np.sum(w * v) / max(float(np.sum(w)), 1.0e-8))


def _query_standardized(values: np.ndarray, query_ids: list[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr)
    groups: dict[int, list[int]] = defaultdict(list)
    for pos, query_id in enumerate(query_ids):
        groups[int(query_id)].append(pos)
    for positions in groups.values():
        idx = np.asarray(positions, dtype=np.int64)
        local = arr[idx]
        scale = float(np.std(local))
        out[idx] = 0.0 if scale <= 1.0e-12 else (local - float(np.mean(local))) / scale
    return out


def _support_prior(
    *,
    support_rows: list[list[int]],
    labels: np.ndarray,
    query_ids: list[int],
) -> dict[int, float]:
    standardized = _query_standardized(labels, query_ids)
    buckets: dict[int, list[float]] = defaultdict(list)
    for support, value in zip(support_rows, standardized, strict=True):
        for support_id in support:
            buckets[int(support_id)].append(float(value))
    return {
        int(support_id): float(np.mean(values))
        for support_id, values in buckets.items()
        if values
    }


def _score_support_rows(
    support_rows: list[list[int]],
    prior: dict[int, float],
) -> np.ndarray:
    scores = []
    for support in support_rows:
        values = [float(prior.get(int(idx), 0.0)) for idx in support]
        scores.append(float(np.mean(values)) if values else 0.0)
    return np.asarray(scores, dtype=np.float64)


FEATURE_NAMES = (
    "bias",
    "mean_memory_cosine",
    "min_memory_cosine",
    "mean_narrative_start_score",
    "max_narrative_start_score",
    "mean_negative_start_distance",
    "max_negative_start_distance",
    "direction_pass_rate",
    "temporal_spread",
    "prefix_response_score",
    "prefix_direction_alignment_mean",
    "prefix_direction_alignment_min",
    "prefix_direction_alignment_std",
    "prefix_terminal_abs_mean",
    "prefix_terminal_abs_max",
    "prefix_channel_width_mean",
    "prefix_channel_count",
)


def _prefix_dynamic_feature_block(
    *,
    support_indices: list[int],
    history_level: np.ndarray | None,
    query_start_state: np.ndarray | None,
    grounding: dict[str, Any] | None,
) -> list[float]:
    if history_level is None or query_start_state is None or grounding is None:
        return [0.0] * 8
    if not support_indices:
        return [0.0] * 8
    channels = _channels_from_grounding(grounding)
    if not channels:
        return [0.0] * 8
    history = np.asarray(history_level, dtype=np.float64)
    support = np.asarray(support_indices, dtype=np.int64)
    support_history = history[support]
    prefix_delta = support_history - support_history[:, [0], :]
    response = _response_score(
        samples_delta=prefix_delta,
        start=np.asarray(query_start_state, dtype=np.float64),
        channels=channels,
    )
    start = np.asarray(query_start_state, dtype=np.float64).reshape(-1)
    signed_terminal: list[float] = []
    abs_terminal: list[float] = []
    widths: list[float] = []
    for channel in channels:
        idx = FACTOR_INDEX[str(channel["factor"])]
        scale = max(abs(float(start[idx])), 1.0)
        terminal = prefix_delta[:, -1, idx] / scale
        path = prefix_delta[:, :, idx] / scale
        sign = float(channel["sign"])
        signed_terminal.extend([float(sign * value) for value in terminal])
        abs_terminal.extend([float(abs(value)) for value in terminal])
        widths.append(float(np.percentile(path, 90) - np.percentile(path, 10)))
    signed = np.asarray(signed_terminal, dtype=np.float64)
    abs_values = np.asarray(abs_terminal, dtype=np.float64)
    width_values = np.asarray(widths, dtype=np.float64)
    return [
        float(response),
        float(np.mean(signed)) if signed.size else 0.0,
        float(np.min(signed)) if signed.size else 0.0,
        float(np.std(signed)) if signed.size else 0.0,
        float(np.mean(abs_values)) if abs_values.size else 0.0,
        float(np.max(abs_values)) if abs_values.size else 0.0,
        float(np.mean(width_values)) if width_values.size else 0.0,
        float(len(channels)),
    ]


def features_for_candidate_mixture(
    candidate: dict[str, Any],
    *,
    history_level: np.ndarray | None = None,
    query_start_state: np.ndarray | None = None,
    grounding: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return candidate-set features available before final generation."""

    items = [
        item
        for item in candidate.get("top_train_pool", [])
        if isinstance(item, dict) and item.get("window_index") is not None
    ]
    if not items:
        return np.asarray([1.0] + [0.0] * (len(FEATURE_NAMES) - 1), dtype=np.float64)
    support_indices = [int(item["window_index"]) for item in items]
    cos = np.asarray([float(item.get("cosine", 0.0) or 0.0) for item in items])
    score = np.asarray([float(item.get("score", 0.0) or 0.0) for item in items])
    dist = np.asarray(
        [float(item.get("start_distance_z", 0.0) or 0.0) for item in items]
    )
    passed = np.asarray(
        [
            1.0
            if str(item.get("recent_prefix_alignment_status", "")).lower() == "pass"
            else 0.0
            for item in items
        ],
        dtype=np.float64,
    )
    index = np.asarray([float(item["window_index"]) for item in items])
    spread = float(np.std(index) / 1000.0) if index.size > 1 else 0.0
    dynamic_features = _prefix_dynamic_feature_block(
        support_indices=support_indices,
        history_level=history_level,
        query_start_state=query_start_state,
        grounding=grounding,
    )
    return np.asarray(
        [
            1.0,
            float(np.mean(cos)),
            float(np.min(cos)),
            float(np.mean(score)),
            float(np.max(score)),
            float(np.mean(-dist)),
            float(np.max(-dist)),
            float(np.mean(passed)),
            spread,
            *dynamic_features,
        ],
        dtype=np.float64,
    )


def _fit_ridge(features: np.ndarray, labels: np.ndarray, query_ids: list[int]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = _query_standardized(np.asarray(labels, dtype=np.float64), query_ids)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must align")
    ridge = 1.0e-3 * np.eye(x.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def _linear_score(features: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ np.asarray(coef, dtype=np.float64)


def _evaluate_scores(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    query_ids: list[int],
) -> dict[str, Any]:
    return {
        "pairwise_accuracy": pairwise_preference_accuracy(
            predicted_scores=np.asarray(scores, dtype=np.float64),
            true_scores=np.asarray(labels, dtype=np.float64),
            query_ids=query_ids,
        ),
        "selection_regret": selection_regret(
            predicted_scores=np.asarray(scores, dtype=np.float64),
            raw_losses=-np.asarray(labels, dtype=np.float64),
            query_ids=query_ids,
        ),
    }


def _build_labeled_rows(
    *,
    query_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    spec_names: list[str],
    max_queries: int,
    candidate_pool_size: int,
    mixture_size: int,
    max_mixtures: int,
    diverse_max_pairwise_cosine: float,
    diverse_min_index_gap: int,
    start_distance_penalty: float,
    implication_alignment_weight: float,
) -> list[dict[str, Any]]:
    memory = np.asarray(arrays["memory_targets"], dtype=np.float32)
    history = np.asarray(arrays["history_level"], dtype=np.float32)
    future_delta = np.asarray(arrays["future_delta"], dtype=np.float32)
    scale = np.asarray(arrays["scale"], dtype=np.float32)
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    selected_queries = np.asarray(query_indices, dtype=np.int64)[: int(max_queries)]
    for query_pos, query_idx in enumerate(selected_queries, start=1):
        query_idx = int(query_idx)
        grounding = grounding_from_history(
            history_level=history,
            window_index=query_idx,
            spec_names=spec_names,
        )
        candidate_train = train_indices[
            np.abs(train_indices.astype(np.int64) - query_idx) >= int(diverse_min_index_gap)
        ]
        candidates = candidate_support_table(
            query_memory=memory[query_idx],
            memory_targets=memory,
            history_level=history,
            train_indices=candidate_train,
            query_window_index=query_idx,
            query_start_state=history[query_idx, -1],
            grounding=grounding,
            spec_names=spec_names,
            start_distance_threshold_z=1.0e9,
            start_distance_penalty=float(start_distance_penalty),
            implication_alignment_weight=float(implication_alignment_weight),
        )
        mixtures = _portfolio_quality_guard_candidate_mixtures(
            candidates=candidates,
            memory_targets=memory,
            top_k=8,
            candidate_pool_size=int(candidate_pool_size),
            mixture_size=int(mixture_size),
            max_mixtures=int(max_mixtures),
            diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(diverse_min_index_gap),
        )
        if not mixtures:
            continue
        query_scale = scale[query_idx]
        observed_delta = future_delta[query_idx]
        observed_norm = _normalize_future_delta(
            observed_delta[None, :, :],
            scale=query_scale,
        )[0]
        channels = _channels_from_grounding(grounding)
        for mixture in mixtures:
            support = [
                int(item["window_index"])
                for item in mixture.get("top_train_pool", [])
                if isinstance(item, dict)
            ]
            if not support:
                continue
            support_delta = future_delta[np.asarray(support, dtype=np.int64)]
            support_norm = _normalize_future_delta(support_delta, scale=query_scale)
            crps = _ensemble_crps(support_norm, observed_norm)
            energy = _energy_score(support_norm, observed_norm)
            response = _response_score(
                samples_delta=support_delta,
                start=history[query_idx, -1],
                channels=channels,
            )
            support_portfolio = _portfolio_path_from_delta(
                future_delta=support_delta,
                start=history[query_idx, -1],
            )
            observed_portfolio = _portfolio_path_from_delta(
                future_delta=observed_delta[None, :, :],
                start=history[query_idx, -1],
            )[0]
            portfolio_energy = _energy_score(
                support_portfolio[:, :, None],
                observed_portfolio[:, None],
            )
            rows.append(
                {
                    "query_index": int(query_idx),
                    "query_position": int(query_pos),
                    "support": support,
                    "query_id": str(mixture.get("query_id", "")),
                    "crps_utility": float(-crps),
                    "energy_utility": float(-energy),
                    "portfolio_utility": float(response - 0.25 * portfolio_energy),
                    "response_score": float(response),
                    "portfolio_energy": float(portfolio_energy),
                    "features": [
                        float(value)
                        for value in features_for_candidate_mixture(
                            mixture,
                            history_level=history,
                            query_start_state=history[query_idx, -1],
                            grounding=grounding,
                        )
                    ],
                }
            )
    return rows


def _context_from_rows(
    *,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    probability_temperature: float,
    source_artifacts: dict[str, str],
) -> dict[str, Any]:
    train_support = [list(map(int, row["support"])) for row in train_rows]
    train_queries = [int(row["query_index"]) for row in train_rows]
    train_features = np.asarray(
        [row["features"] for row in train_rows], dtype=np.float64
    )
    eval_features = np.asarray([row["features"] for row in eval_rows], dtype=np.float64)
    portfolio_prior = _support_prior(
        support_rows=train_support,
        labels=np.asarray([row["portfolio_utility"] for row in train_rows]),
        query_ids=train_queries,
    )
    crps_prior = _support_prior(
        support_rows=train_support,
        labels=np.asarray([row["crps_utility"] for row in train_rows]),
        query_ids=train_queries,
    )
    energy_prior = _support_prior(
        support_rows=train_support,
        labels=np.asarray([row["energy_utility"] for row in train_rows]),
        query_ids=train_queries,
    )
    eval_support = [list(map(int, row["support"])) for row in eval_rows]
    eval_queries = [int(row["query_index"]) for row in eval_rows]
    eval_portfolio = np.asarray([row["portfolio_utility"] for row in eval_rows])
    eval_crps = np.asarray([row["crps_utility"] for row in eval_rows])
    eval_energy = np.asarray([row["energy_utility"] for row in eval_rows])
    portfolio_scores = _score_support_rows(eval_support, portfolio_prior)
    crps_scores = _score_support_rows(eval_support, crps_prior)
    energy_scores = _score_support_rows(eval_support, energy_prior)
    quality_scores = portfolio_scores + np.minimum(crps_scores, 0.0) + np.minimum(
        energy_scores, 0.0
    )
    portfolio_coef = _fit_ridge(
        train_features,
        np.asarray([row["portfolio_utility"] for row in train_rows]),
        train_queries,
    )
    crps_coef = _fit_ridge(
        train_features,
        np.asarray([row["crps_utility"] for row in train_rows]),
        train_queries,
    )
    energy_coef = _fit_ridge(
        train_features,
        np.asarray([row["energy_utility"] for row in train_rows]),
        train_queries,
    )
    feature_portfolio_scores = _linear_score(eval_features, portfolio_coef)
    feature_crps_scores = _linear_score(eval_features, crps_coef)
    feature_energy_scores = _linear_score(eval_features, energy_coef)
    feature_quality_scores = (
        feature_portfolio_scores
        + np.minimum(feature_crps_scores, 0.0)
        + np.minimum(feature_energy_scores, 0.0)
    )
    group_entropies: list[float] = []
    for query in sorted(set(eval_queries)):
        mask = np.asarray([item == query for item in eval_queries], dtype=bool)
        if int(mask.sum()) > 1:
            group_entropies.append(
                _entropy(_softmax(quality_scores[mask], temperature=probability_temperature))
            )
    combined_pairwise = _evaluate_scores(
        scores=quality_scores,
        labels=eval_portfolio,
        query_ids=eval_queries,
    )["pairwise_accuracy"]["accuracy"]
    feature_pairwise = _evaluate_scores(
        scores=feature_quality_scores,
        labels=eval_portfolio,
        query_ids=eval_queries,
    )["pairwise_accuracy"]["accuracy"]
    preferred_selector = (
        "feature_model"
        if float(feature_pairwise) > float(combined_pairwise)
        else "support_prior"
    )
    return {
        "status": "ok",
        "scope_note": (
            "Broad support-bank response utility trained from historical future-delta "
            "replay labels. This is a pre-rollout support scorer TestFlight, not a "
            "promotion claim."
        ),
        "preferred_selector": preferred_selector,
        "portfolio_prior": portfolio_prior,
        "crps_prior": crps_prior,
        "energy_prior": energy_prior,
        "probability_temperature": float(probability_temperature),
        "max_candidate_entropy_quantile": None,
        "max_candidate_entropy_threshold": None,
        "min_support_weight_max_quantile": None,
        "min_support_weight_max_threshold": None,
        "source_artifacts": source_artifacts,
        "feature_model": {
            "feature_names": list(FEATURE_NAMES),
            "portfolio_coef": [float(value) for value in portfolio_coef],
            "crps_coef": [float(value) for value in crps_coef],
            "energy_coef": [float(value) for value in energy_coef],
        },
        "training": {
            "train_query_count": len(set(int(row["query_index"]) for row in train_rows)),
            "train_candidate_count": len(train_rows),
            "eval_query_count": len(set(eval_queries)),
            "eval_candidate_count": len(eval_rows),
            "portfolio_prior_support_count": len(portfolio_prior),
            "crps_prior_support_count": len(crps_prior),
            "energy_prior_support_count": len(energy_prior),
        },
        "evaluation": {
            "portfolio_prior": _evaluate_scores(
                scores=portfolio_scores,
                labels=eval_portfolio,
                query_ids=eval_queries,
            ),
            "crps_prior": _evaluate_scores(
                scores=crps_scores,
                labels=eval_crps,
                query_ids=eval_queries,
            ),
            "energy_prior": _evaluate_scores(
                scores=energy_scores,
                labels=eval_energy,
                query_ids=eval_queries,
            ),
            "combined_quality_score_vs_portfolio_label": _evaluate_scores(
                scores=quality_scores,
                labels=eval_portfolio,
                query_ids=eval_queries,
            ),
            "feature_quality_score_vs_portfolio_label": _evaluate_scores(
                scores=feature_quality_scores,
                labels=eval_portfolio,
                query_ids=eval_queries,
            ),
            "feature_crps_score": _evaluate_scores(
                scores=feature_crps_scores,
                labels=eval_crps,
                query_ids=eval_queries,
            ),
            "feature_energy_score": _evaluate_scores(
                scores=feature_energy_scores,
                labels=eval_energy,
                query_ids=eval_queries,
            ),
            "mean_candidate_entropy": (
                None if not group_entropies else float(np.mean(group_entropies))
            ),
        },
    }


def load_broad_response_utility_context(
    path: str | Path = DEFAULT_CONTEXT_PATH,
) -> dict[str, Any]:
    context = _load_json(path)
    for key in ("portfolio_prior", "crps_prior", "energy_prior"):
        raw = context.get(key, {})
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: {key} must be a JSON object")
        context[key] = {int(k): float(v) for k, v in raw.items()}
    return context


def feature_quality_scores_for_candidates(
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    history_level: np.ndarray | None = None,
    query_start_state: np.ndarray | None = None,
    grounding: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    model = context.get("feature_model", {})
    if not isinstance(model, dict):
        raise ValueError("context missing feature_model")
    features = np.asarray(
        [
            features_for_candidate_mixture(
                candidate,
                history_level=history_level,
                query_start_state=query_start_state,
                grounding=grounding,
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    portfolio = _linear_score(features, np.asarray(model["portfolio_coef"]))
    crps = _linear_score(features, np.asarray(model["crps_coef"]))
    energy = _linear_score(features, np.asarray(model["energy_coef"]))
    total = portfolio + np.minimum(crps, 0.0) + np.minimum(energy, 0.0)
    return {
        "total": total,
        "portfolio": portfolio,
        "crps": crps,
        "energy": energy,
        "quality_penalty": np.minimum(crps, 0.0) + np.minimum(energy, 0.0),
    }


def select_feature_quality_guard_support(
    candidates: list[dict[str, Any]],
    *,
    context: dict[str, Any],
    history_level: np.ndarray | None = None,
    query_start_state: np.ndarray | None = None,
    grounding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select support using the broad-bank feature quality model."""

    if not candidates:
        raise ValueError("candidates must be non-empty")
    probability_temperature = float(context.get("probability_temperature", 0.25))
    score_block = feature_quality_scores_for_candidates(
        candidates,
        context,
        history_level=history_level,
        query_start_state=query_start_state,
        grounding=grounding,
    )
    scores = np.asarray(score_block["total"], dtype=np.float64)
    probs_for_gate = _softmax(scores, temperature=probability_temperature)
    entropy_for_gate = _entropy(probs_for_gate)
    selected, probs = listwise_support_weights_for_candidates(
        candidates,
        scores,
        probability_temperature=probability_temperature,
    )
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
            "name": "broad_replay_response_guard_prior",
            "policy_kind": "broad_replay_feature_quality_guard",
            "probability_temperature": probability_temperature,
            "candidate_entropy": float(entropy_for_gate),
            "fallback_to_equal_support": False,
            "candidate_count": len(candidates),
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


def train_broad_response_utility(args: argparse.Namespace) -> dict[str, Any]:
    arrays = _load_npz(args.support_bank_arrays)
    report = _load_json(args.support_bank_report)
    spec_names = load_state_spec_names(args.checkpoint)
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    test_indices = np.asarray(arrays["test_indices"], dtype=np.int64)
    rng = np.random.default_rng(int(args.seed))
    train_queries = rng.choice(
        train_indices,
        size=min(int(args.train_queries), train_indices.size),
        replace=False,
    )
    eval_queries = rng.choice(
        test_indices if test_indices.size else train_indices,
        size=min(int(args.eval_queries), int(test_indices.size or train_indices.size)),
        replace=False,
    )
    train_rows = _build_labeled_rows(
        query_indices=train_queries,
        arrays=arrays,
        spec_names=spec_names,
        max_queries=int(args.train_queries),
        candidate_pool_size=int(args.candidate_pool_size),
        mixture_size=int(args.mixture_size),
        max_mixtures=int(args.max_mixtures),
        diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
        diverse_min_index_gap=int(args.diverse_min_index_gap),
        start_distance_penalty=float(args.start_distance_penalty),
        implication_alignment_weight=float(args.implication_alignment_weight),
    )
    eval_rows = _build_labeled_rows(
        query_indices=eval_queries,
        arrays=arrays,
        spec_names=spec_names,
        max_queries=int(args.eval_queries),
        candidate_pool_size=int(args.candidate_pool_size),
        mixture_size=int(args.mixture_size),
        max_mixtures=int(args.max_mixtures),
        diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
        diverse_min_index_gap=int(args.diverse_min_index_gap),
        start_distance_penalty=float(args.start_distance_penalty),
        implication_alignment_weight=float(args.implication_alignment_weight),
    )
    context = _context_from_rows(
        train_rows=train_rows,
        eval_rows=eval_rows,
        probability_temperature=float(args.probability_temperature),
        source_artifacts={
            "support_bank_report": str(args.support_bank_report),
            "support_bank_arrays": str(args.support_bank_arrays),
            "checkpoint": str(args.checkpoint),
        },
    )
    context["support_bank_counts"] = report.get("counts", {})
    context["config"] = {
        "seed": int(args.seed),
        "candidate_pool_size": int(args.candidate_pool_size),
        "mixture_size": int(args.mixture_size),
        "max_mixtures": int(args.max_mixtures),
        "diverse_max_pairwise_cosine": float(args.diverse_max_pairwise_cosine),
        "diverse_min_index_gap": int(args.diverse_min_index_gap),
        "start_distance_penalty": float(args.start_distance_penalty),
        "implication_alignment_weight": float(args.implication_alignment_weight),
    }
    output_dir = Path(args.output_dir)
    context_path = output_dir / "broad_support_response_utility_context.json"
    report_path = output_dir / "broad_support_response_utility_report.json"
    slim_report = {
        key: value
        for key, value in context.items()
        if key not in {"portfolio_prior", "crps_prior", "energy_prior"}
    }
    slim_report["artifact_paths"] = {
        "context": str(context_path),
        "report": str(report_path),
    }
    context["artifact_paths"] = dict(slim_report["artifact_paths"])
    _write_json(context_path, context)
    _write_json(report_path, slim_report)
    print(json.dumps(slim_report["evaluation"], indent=2, sort_keys=True))
    print(f"wrote {context_path}")
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-bank-report", default=str(DEFAULT_SUPPORT_BANK_REPORT))
    parser.add_argument("--support-bank-arrays", default=str(DEFAULT_SUPPORT_BANK_ARRAYS))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-queries", type=int, default=96)
    parser.add_argument("--eval-queries", type=int, default=24)
    parser.add_argument("--candidate-pool-size", type=int, default=20)
    parser.add_argument("--mixture-size", type=int, default=3)
    parser.add_argument("--max-mixtures", type=int, default=96)
    parser.add_argument("--diverse-max-pairwise-cosine", type=float, default=0.95)
    parser.add_argument("--diverse-min-index-gap", type=int, default=30)
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument("--implication-alignment-weight", type=float, default=0.25)
    parser.add_argument("--probability-temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=940)
    args = parser.parse_args()
    train_broad_response_utility(args)


if __name__ == "__main__":
    main()
