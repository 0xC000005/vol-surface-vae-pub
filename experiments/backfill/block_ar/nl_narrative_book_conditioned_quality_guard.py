#!/usr/bin/env python
"""Narrative-book-conditioned diagnostic for quality-guard support selection.

The 924e quality guard uses one fixed portfolio-response label for every
narrative. This diagnostic keeps the same auditable support-mixture contract,
but lets the grounding sidecar choose which portfolio-risk books define the
portfolio utility. It does not call OpenAI or the generator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_portfolio_response_feature_sufficiency import (  # noqa: E402
    _query_standardized,
)
from experiments.backfill.block_ar.nl_portfolio_response_quality_guard_policy import (  # noqa: E402
    _support_prior_for_metric,
    quality_guard_scores,
    select_quality_guard_support,
)
from experiments.backfill.block_ar.nl_portfolio_response_support_reliability_policy import (  # noqa: E402
    DEFAULT_BRIDGE_ARRAYS,
    DEFAULT_HISTORY_ARRAYS,
    DEFAULT_TRAIN_CANDIDATE_BRIDGE,
    _load_json,
    _load_npz,
    _support_indices,
)
from experiments.backfill.block_ar.nl_portfolio_risk_response_label_audit import (  # noqa: E402
    PORTFOLIO_BOOKS,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (  # noqa: E402
    _portfolio_quality_guard_candidate_mixtures,
    build_mixture_memory_prior,
    candidate_support_table,
)
from experiments.backfill.block_ar.nl_portfolio_quality_guard_live_breadth_diagnostic import (  # noqa: E402
    DEFAULT_BRIDGE_REPORT,
    load_support_context,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)


DEFAULT_ALLBOOK_LABEL_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_response_allbook_labels_926a/"
    "portfolio_response_label_scenario_report.json"
)
DEFAULT_BREADTH_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_portfolio_quality_guard_live_breadth_925f_minbreadth/"
    "portfolio_quality_guard_live_breadth.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_narrative_book_conditioned_quality_guard_926a"
)

MARKET_ALIASES = {
    "CRUDE_OIL": "CRUDE",
    "OIL": "CRUDE",
    "CREDIT_SPREADS": "BBB_OAS",
    "SPREADS": "BBB_OAS",
    "TREASURY_10Y": "US10Y",
    "UST10Y": "US10Y",
    "RATES": "US10Y",
    "USDJPY": "DXY",
}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
FLAT_DIRECTION_WEIGHT = {"flat": 0.25, "mixed": 0.4, "unchanged": 0.25}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _normal_market(value: Any) -> str:
    raw = str(value or "").strip().upper().replace(" ", "_")
    return MARKET_ALIASES.get(raw, raw)


def _implication_rows(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    if "condition_only_grounding" in grounding:
        grounding = grounding.get("condition_only_grounding", {})
    rows = grounding.get("current_market_state_implications", [])
    return [row for row in rows if isinstance(row, dict)]


def narrative_book_relevance_weights(
    grounding: dict[str, Any],
    *,
    books: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Map current/recent grounded market channels to portfolio-book weights."""

    books = list(books or PORTFOLIO_BOOKS)
    scores = {str(book["name"]): 0.0 for book in books}
    for implication in _implication_rows(grounding):
        market = _normal_market(implication.get("market"))
        confidence = str(implication.get("confidence", "medium")).lower()
        direction = str(implication.get("direction", "")).lower()
        weight = CONFIDENCE_WEIGHT.get(confidence, 0.6)
        weight *= FLAT_DIRECTION_WEIGHT.get(direction, 1.0)
        for book in books:
            exposure = dict(book.get("exposures", {})).get(market)
            if exposure is not None:
                scores[str(book["name"])] += weight * abs(float(exposure))
    total = float(sum(value for value in scores.values() if value > 0.0))
    if total <= 0.0:
        fallback = {"equity_beta_carry", "dollar_liquidity", "short_volatility"}
        equal = 1.0 / float(len(fallback))
        return {name: (equal if name in fallback else 0.0) for name in scores}
    return {name: float(max(value, 0.0) / total) for name, value in scores.items()}


def _score_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in report.get("window_scores", []):
        if isinstance(row, dict) and row.get("query_id") is not None:
            out[str(row["query_id"])] = row
    return out


def _book_score(method_row: dict[str, Any], book_name: str) -> float | None:
    for book in method_row.get("portfolio_response_books", []):
        if isinstance(book, dict) and str(book.get("book")) == str(book_name):
            value = book.get("composite_path_score_z")
            if value is not None and np.isfinite(float(value)):
                return float(value)
    return None


def support_priors_by_book(
    *,
    train_candidate_bridge: dict[str, Any],
    train_label_report: dict[str, Any],
) -> dict[str, dict[int, float]]:
    """Build train-only per-support reliability priors for each portfolio book."""

    label_rows = _score_lookup(train_label_report)
    books = [str(book["name"]) for book in PORTFOLIO_BOOKS]
    support_rows_by_book: dict[str, list[list[int]]] = {book: [] for book in books}
    labels_by_book: dict[str, list[float]] = {book: [] for book in books}
    query_ids_by_book: dict[str, list[Any]] = {book: [] for book in books}
    for row in train_candidate_bridge.get("evaluation", {}).get("heldout_examples", []):
        if not isinstance(row, dict) or not row.get("query_id"):
            continue
        score = label_rows.get(str(row["query_id"]))
        method = (
            score.get("methods", {}).get("narrative_generator_topk", {})
            if isinstance(score, dict)
            else {}
        )
        supports = _support_indices(row)
        for book in books:
            value = _book_score(method, book)
            if value is None:
                continue
            support_rows_by_book[book].append(supports)
            labels_by_book[book].append(-float(value))
            query_ids_by_book[book].append(row.get("window_index"))

    priors: dict[str, dict[int, float]] = {}
    for book in books:
        labels = np.asarray(labels_by_book[book], dtype=np.float64)
        if labels.size == 0:
            priors[book] = {}
            continue
        standardized = _query_standardized(labels, query_ids_by_book[book])
        buckets: dict[int, list[float]] = {}
        for supports, value in zip(
            support_rows_by_book[book], standardized, strict=True
        ):
            for support_id in supports:
                buckets.setdefault(int(support_id), []).append(float(value))
        priors[book] = {
            key: float(np.mean(values)) for key, values in buckets.items() if values
        }
    return priors


def blend_book_priors(
    priors_by_book: dict[str, dict[int, float]],
    weights: dict[str, float],
) -> dict[int, float]:
    ids = sorted(
        {
            int(support_id)
            for book, prior in priors_by_book.items()
            if float(weights.get(book, 0.0)) > 0.0
            for support_id in prior
        }
    )
    return {
        support_id: float(
            sum(
                float(weights.get(book, 0.0)) * float(prior.get(support_id, 0.0))
                for book, prior in priors_by_book.items()
            )
        )
        for support_id in ids
    }


def build_narrative_book_guard_policy_context(
    *,
    train_candidate_bridge: dict[str, Any] | None = None,
    allbook_label_report: dict[str, Any] | None = None,
    bridge_arrays: dict[str, np.ndarray] | None = None,
    history_arrays: dict[str, np.ndarray] | None = None,
    train_candidate_bridge_path: str | Path = DEFAULT_TRAIN_CANDIDATE_BRIDGE,
    allbook_label_report_path: str | Path = DEFAULT_ALLBOOK_LABEL_REPORT,
    bridge_arrays_path: str | Path = DEFAULT_BRIDGE_ARRAYS,
    history_arrays_path: str | Path = DEFAULT_HISTORY_ARRAYS,
    probability_temperature: float = 0.25,
    min_support_weight_max_quantile: float | None = None,
) -> dict[str, Any]:
    """Build train-only response priors for narrative-book support weighting.

    The context is safe to use at inference because it contains only train-set
    support reliability priors. Narrative-specific book weights are computed
    later from the current/recent grounding sidecar.
    """

    train_candidate = (
        train_candidate_bridge
        if train_candidate_bridge is not None
        else _load_json(train_candidate_bridge_path)
    )
    allbook_labels = (
        allbook_label_report
        if allbook_label_report is not None
        else _load_json(allbook_label_report_path)
    )
    bridge_npz = (
        bridge_arrays if bridge_arrays is not None else _load_npz(bridge_arrays_path)
    )
    history_npz = (
        history_arrays if history_arrays is not None else _load_npz(history_arrays_path)
    )
    priors_by_book = support_priors_by_book(
        train_candidate_bridge=train_candidate,
        train_label_report=allbook_labels,
    )
    crps_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=allbook_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric="ensemble_crps_z",
    )
    energy_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=allbook_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric="energy_score_z",
    )
    return {
        "priors_by_book": priors_by_book,
        "crps_prior": crps_prior,
        "energy_prior": energy_prior,
        "probability_temperature": float(probability_temperature),
        "min_support_weight_max_threshold": None,
        "min_support_weight_max_quantile": (
            None
            if min_support_weight_max_quantile is None
            else float(min_support_weight_max_quantile)
        ),
        "source_artifacts": {
            "train_candidate_bridge": str(train_candidate_bridge_path),
            "allbook_label_report": str(allbook_label_report_path),
            "bridge_arrays": str(bridge_arrays_path),
            "history_arrays": str(history_arrays_path),
        },
    }


def _support_jaccard(left: list[int], right: list[int]) -> float:
    a = {int(value) for value in left}
    b = {int(value) for value in right}
    if not a and not b:
        return 1.0
    return float(len(a.intersection(b)) / max(len(a.union(b)), 1))


def _condition_memory_and_grounding(
    condition_report: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    report = _load_json(condition_report)
    cached = report.get("cached_query", {})
    arrays_path = report.get("artifact_paths", {}).get("arrays")
    if not arrays_path:
        raise ValueError(f"{condition_report}: missing condition arrays")
    arrays = _load_npz(arrays_path)
    grounding = cached.get("grounding", {})
    return np.asarray(arrays["text_memory"], dtype=np.float32).reshape(-1), grounding


def run_narrative_book_guard_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    breadth = _load_json(args.breadth_report)
    bridge_npz = _load_npz(args.bridge_arrays)
    history_npz = _load_npz(args.history_arrays)
    train_candidate = _load_json(args.train_candidate_bridge)
    allbook_labels = _load_json(args.allbook_label_report)
    support_context = load_support_context(
        SimpleNamespace(
            bridge_report=args.bridge_report,
            bridge_arrays=args.bridge_arrays,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    )
    train_indices = np.asarray(support_context["train_indices"], dtype=np.int64)
    memory_targets = np.asarray(support_context["memory_targets"], dtype=np.float32)
    history_level = np.asarray(support_context["history_level"], dtype=np.float32)
    spec_names = list(support_context["spec_names"])
    priors_by_book = support_priors_by_book(
        train_candidate_bridge=train_candidate,
        train_label_report=allbook_labels,
    )
    crps_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=allbook_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric="ensemble_crps_z",
    )
    energy_prior = _support_prior_for_metric(
        train_candidate_bridge=train_candidate,
        train_label_report=allbook_labels,
        bridge_arrays=bridge_npz,
        history_arrays=history_npz,
        metric="energy_score_z",
    )

    rows: list[dict[str, Any]] = []
    for source_row in breadth.get("rows", []):
        if not isinstance(source_row, dict):
            continue
        query_memory, grounding = _condition_memory_and_grounding(
            source_row["condition_report"]
        )
        start_idx = int(source_row["start_window_index"])
        book_weights = narrative_book_relevance_weights(grounding)
        portfolio_prior = blend_book_priors(priors_by_book, book_weights)
        candidates = candidate_support_table(
            query_memory=query_memory,
            memory_targets=memory_targets,
            history_level=history_level,
            train_indices=train_indices,
            query_window_index=int(args.query_window_index),
            query_start_state=history_level[start_idx, -1, :],
            grounding=grounding,
            spec_names=spec_names,
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            start_distance_penalty=float(args.start_distance_penalty),
            implication_alignment_weight=float(args.implication_alignment_weight),
        )
        base_prior = build_mixture_memory_prior(
            query_memory=query_memory,
            memory_targets=memory_targets,
            history_level=history_level,
            train_indices=train_indices,
            query_window_index=int(args.query_window_index),
            query_start_state=history_level[start_idx, -1, :],
            grounding=grounding,
            spec_names=spec_names,
            mode="diverse_topk_narrative_start_checked",
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            start_distance_penalty=float(args.start_distance_penalty),
            implication_alignment_weight=float(args.implication_alignment_weight),
            diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(args.diverse_min_index_gap),
        )
        candidate_mixtures = _portfolio_quality_guard_candidate_mixtures(
            candidates=candidates,
            memory_targets=memory_targets,
            top_k=int(args.top_k),
            candidate_pool_size=int(args.quality_guard_candidate_pool_size),
            mixture_size=int(args.quality_guard_mixture_size),
            max_mixtures=int(args.quality_guard_max_mixtures),
            diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
            diverse_min_index_gap=int(args.diverse_min_index_gap),
        )
        if len(candidate_mixtures) < int(args.quality_guard_min_candidate_mixtures):
            selected_indices = [int(value) for value in base_prior["window_indices"]]
            fallback = True
            selected_score = None
        else:
            selection = select_quality_guard_support(
                candidate_mixtures,
                portfolio_prior=portfolio_prior,
                crps_prior=crps_prior,
                energy_prior=energy_prior,
                probability_temperature=float(args.probability_temperature),
                min_support_weight_max_threshold=None,
            )
            fallback = bool(selection["fallback"])
            if fallback:
                selected_indices = [
                    int(value) for value in base_prior["window_indices"]
                ]
                selected_score = None
            else:
                selected_indices = [
                    int(item["window_index"])
                    for item in selection.get("selected", [])
                    if isinstance(item, dict)
                ]
                selected_score = selection["support_policy"].get("selected_score")
        rows.append(
            {
                "case_name": str(source_row.get("case_name")),
                "start_window_index": int(start_idx),
                "book_weights": book_weights,
                "top_books": sorted(
                    [
                        {"book": key, "weight": float(value)}
                        for key, value in book_weights.items()
                        if float(value) > 0.0
                    ],
                    key=lambda item: item["weight"],
                    reverse=True,
                )[:3],
                "candidate_count": int(len(candidate_mixtures)),
                "fallback": bool(fallback),
                "base_window_indices": [
                    int(value) for value in base_prior["window_indices"]
                ],
                "book_guard_window_indices": selected_indices,
                "quality_guard_window_indices": [
                    int(value)
                    for value in source_row.get("quality_guard_window_indices", [])
                ],
                "jaccard_vs_base": _support_jaccard(
                    selected_indices,
                    [int(value) for value in base_prior["window_indices"]],
                ),
                "jaccard_vs_924e": _support_jaccard(
                    selected_indices,
                    [
                        int(value)
                        for value in source_row.get("quality_guard_window_indices", [])
                    ],
                ),
                "selected_score": selected_score,
            }
        )
    active = [row for row in rows if not bool(row["fallback"])]
    report = {
        "status": "completed" if rows else "empty",
        "row_count": len(rows),
        "active_count": len(active),
        "fallback_count": len(rows) - len(active),
        "changed_vs_base_count": sum(
            1 for row in rows if float(row["jaccard_vs_base"]) < 1.0
        ),
        "changed_vs_924e_count": sum(
            1 for row in rows if float(row["jaccard_vs_924e"]) < 1.0
        ),
        "mean_jaccard_vs_base": (
            float(np.mean([row["jaccard_vs_base"] for row in rows])) if rows else None
        ),
        "mean_jaccard_vs_924e": (
            float(np.mean([row["jaccard_vs_924e"] for row in rows])) if rows else None
        ),
        "source_artifacts": {
            "breadth_report": str(args.breadth_report),
            "allbook_label_report": str(args.allbook_label_report),
            "train_candidate_bridge": str(args.train_candidate_bridge),
        },
        "rows": rows,
    }
    report["artifact_paths"] = {
        "report": str(Path(args.output_dir) / "narrative_book_quality_guard.json"),
        "markdown": str(Path(args.output_dir) / "narrative_book_quality_guard.md"),
    }
    _write_json(report["artifact_paths"]["report"], report)
    _write_text(report["artifact_paths"]["markdown"], _markdown(report))
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Narrative-Book Quality Guard Diagnostic",
        "",
        f"Status: `{report['status']}`",
        f"Rows: `{report['row_count']}`",
        f"Active: `{report['active_count']}`",
        f"Fallback: `{report['fallback_count']}`",
        f"Changed vs base: `{report['changed_vs_base_count']}`",
        f"Changed vs 924e: `{report['changed_vs_924e_count']}`",
        f"Mean Jaccard vs base: `{report['mean_jaccard_vs_base']}`",
        f"Mean Jaccard vs 924e: `{report['mean_jaccard_vs_924e']}`",
        "",
        "| Case | Start | Top Books | Fallback | Candidate Count | Jaccard vs Base | Jaccard vs 924e | Support |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        top_books = ", ".join(
            f"{item['book']}={float(item['weight']):.2f}"
            for item in row.get("top_books", [])
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_name"]),
                    str(row["start_window_index"]),
                    top_books,
                    str(row["fallback"]),
                    str(row["candidate_count"]),
                    f"{float(row['jaccard_vs_base']):.3f}",
                    f"{float(row['jaccard_vs_924e']):.3f}",
                    str(row["book_guard_window_indices"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breadth-report", type=Path, default=DEFAULT_BREADTH_REPORT)
    parser.add_argument(
        "--allbook-label-report", type=Path, default=DEFAULT_ALLBOOK_LABEL_REPORT
    )
    parser.add_argument("--bridge-report", type=Path, default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument(
        "--train-candidate-bridge", type=Path, default=DEFAULT_TRAIN_CANDIDATE_BRIDGE
    )
    parser.add_argument("--bridge-arrays", type=Path, default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument("--history-arrays", type=Path, default=DEFAULT_HISTORY_ARRAYS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-window-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--start-distance-threshold-z", type=float, default=15.0)
    parser.add_argument("--start-distance-penalty", type=float, default=0.02)
    parser.add_argument("--implication-alignment-weight", type=float, default=0.25)
    parser.add_argument("--diverse-max-pairwise-cosine", type=float, default=0.95)
    parser.add_argument("--diverse-min-index-gap", type=int, default=30)
    parser.add_argument("--quality-guard-candidate-pool-size", type=int, default=12)
    parser.add_argument("--quality-guard-mixture-size", type=int, default=3)
    parser.add_argument("--quality-guard-max-mixtures", type=int, default=64)
    parser.add_argument("--quality-guard-min-candidate-mixtures", type=int, default=4)
    parser.add_argument("--probability-temperature", type=float, default=0.25)
    args = parser.parse_args()
    report = run_narrative_book_guard_diagnostic(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "row_count": report["row_count"],
                "active_count": report["active_count"],
                "report": report["artifact_paths"]["report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
