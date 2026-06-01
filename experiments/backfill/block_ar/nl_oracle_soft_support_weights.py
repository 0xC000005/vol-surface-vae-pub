#!/usr/bin/env python
"""Build an oracle soft support-weight bridge from generator-response labels.

This is a leakage diagnostic, not a deployable policy. It answers one narrow
question: if the support mixture weights were chosen with hindsight from the
frozen generator's own historical backtest response, would weighted support
sampling improve the scenario distribution? If yes, the bottleneck is learning
those weights from narrative/start features. If no, support weighting is the
wrong lever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_learned_mixture_policy_testflight import (
    _support_indices,
)


DEFAULT_BASE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_CANDIDATE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_mixture_testflight_884a_fullheldout/"
    "mixture_label_bridge_report.json"
)
DEFAULT_SCENARIO_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_rollout_response_mixture_testflight_884a_fullheldout/scenario_eval/"
    "scenario_level_eval_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_oracle_soft_support_weights_906c_fullheldout"
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _score_lookup(scenario_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in scenario_report.get("window_scores", []):
        if isinstance(row, dict) and row.get("query_id"):
            lookup[str(row["query_id"])] = row
    return lookup


def _metric(row: dict[str, Any], method: str, metric: str) -> float | None:
    value = row.get("methods", {}).get(method, {}).get(metric)
    if value is None:
        return None
    raw = float(value)
    return raw if np.isfinite(raw) else None


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-8)
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    shifted = (raw - float(np.max(raw))) / temp
    weights = np.exp(shifted)
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(raw.shape, 1.0 / max(raw.size, 1), dtype=np.float64)
    return weights / denom


def group_candidate_rows(
    candidate_bridge: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_bridge.get("evaluation", {}).get("heldout_examples", []):
        if isinstance(row, dict) and row.get("window_index") is not None:
            groups.setdefault(int(row["window_index"]), []).append(row)
    return groups


def candidate_label_scores(
    candidates: list[dict[str, Any]],
    scenario_report: dict[str, Any],
    *,
    method: str = "narrative_generator_topk",
    metric: str = "energy_score_z",
) -> np.ndarray:
    """Return higher-is-better candidate scores from lower-is-better metrics."""

    scores = _score_lookup(scenario_report)
    values: list[float] = []
    for candidate in candidates:
        query_id = str(candidate.get("query_id", ""))
        row = scores.get(query_id)
        if row is None:
            raise ValueError(f"missing scenario score for candidate {query_id!r}")
        value = _metric(row, method, metric)
        if value is None:
            raise ValueError(f"missing finite {method}.{metric} for {query_id!r}")
        values.append(-float(value))
    return np.asarray(values, dtype=np.float64)


def standardized_candidate_probabilities(
    labels: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert higher-is-better labels into within-query probabilities."""

    raw = np.asarray(labels, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        raise ValueError("labels must be non-empty")
    centered = raw - float(np.mean(raw))
    scale = max(float(np.std(raw)), 1e-8)
    return _softmax(centered / scale, temperature=float(temperature))


def marginalize_candidate_support_weights(
    candidates: list[dict[str, Any]],
    candidate_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Marginalize candidate-mixture probabilities into support-window weights."""

    probs = np.asarray(candidate_probabilities, dtype=np.float64).reshape(-1)
    if len(candidates) != int(probs.size):
        raise ValueError("candidate count and probability count differ")
    support_rows: dict[int, dict[str, Any]] = {}
    raw_weights: dict[int, float] = {}
    for candidate, prob in zip(candidates, probs, strict=True):
        for item in candidate.get("top_train_pool", []):
            if not isinstance(item, dict) or item.get("window_index") is None:
                continue
            idx = int(item["window_index"])
            raw_weights[idx] = raw_weights.get(idx, 0.0) + float(prob)
            existing = support_rows.get(idx)
            if existing is None or float(item.get("cosine", 0.0) or 0.0) > float(
                existing.get("cosine", 0.0) or 0.0
            ):
                support_rows[idx] = json.loads(json.dumps(item))
    total = float(sum(raw_weights.values()))
    if total <= 0.0:
        raise ValueError("candidate support rows are empty")
    rows: list[dict[str, Any]] = []
    for idx, row in support_rows.items():
        updated = json.loads(json.dumps(row))
        updated["window_index"] = idx
        updated["weight"] = float(raw_weights[idx] / total)
        rows.append(updated)
    rows.sort(
        key=lambda item: (
            -float(item.get("weight", 0.0) or 0.0),
            -float(item.get("cosine", 0.0) or 0.0),
            int(item.get("window_index", 0)),
        )
    )
    return rows


def build_oracle_weighted_bridge(
    *,
    base_bridge: dict[str, Any],
    candidate_bridge: dict[str, Any],
    scenario_report: dict[str, Any],
    method: str = "narrative_generator_topk",
    metric: str = "energy_score_z",
    probability_temperature: float = 1.0,
) -> dict[str, Any]:
    """Return a bridge report whose supports carry oracle marginal weights."""

    output = json.loads(json.dumps(base_bridge))
    groups = group_candidate_rows(candidate_bridge)
    changed = 0
    diagnostics: list[dict[str, Any]] = []
    for row in output.get("evaluation", {}).get("heldout_examples", []):
        if not isinstance(row, dict) or row.get("window_index") is None:
            continue
        window_index = int(row["window_index"])
        candidates = groups.get(window_index, [])
        if not candidates:
            continue
        scores = candidate_label_scores(
            candidates,
            scenario_report,
            method=method,
            metric=metric,
        )
        probabilities = standardized_candidate_probabilities(
            scores,
            temperature=float(probability_temperature),
        )
        support_rows = marginalize_candidate_support_weights(candidates, probabilities)
        row["top_train_pool"] = support_rows
        best_pos = int(np.argmax(probabilities))
        row["support_policy"] = {
            "name": "oracle_generator_response_soft_support_weights",
            "policy_kind": "oracle_soft_listwise_upper_bound",
            "label_method": method,
            "label_metric": metric,
            "probability_temperature": float(probability_temperature),
            "candidate_count": len(candidates),
            "candidate_effective_n": float(1.0 / np.sum(probabilities * probabilities)),
            "selected_query_id": str(candidates[best_pos].get("query_id", "")),
            "selected_score": float(scores[best_pos]),
            "selected_support_window_indices": [
                int(item["window_index"]) for item in support_rows
            ],
            "selected_support_weights": [
                float(item.get("weight", 0.0) or 0.0) for item in support_rows
            ],
        }
        diagnostics.append(
            {
                "window_index": window_index,
                "window_id": str(row.get("window_id", "")),
                "candidate_count": len(candidates),
                "candidate_effective_n": float(
                    1.0 / np.sum(probabilities * probabilities)
                ),
                "candidate_max_probability": float(np.max(probabilities)),
                "support_count": len(support_rows),
                "support_effective_n": float(
                    1.0
                    / np.sum(
                        np.square(
                            np.asarray(
                                [
                                    float(item.get("weight", 0.0) or 0.0)
                                    for item in support_rows
                                ],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "support_weights": [
                    {
                        "window_index": int(item["window_index"]),
                        "weight": float(item.get("weight", 0.0) or 0.0),
                    }
                    for item in support_rows
                ],
                "top_candidate_support": _support_indices(candidates[best_pos]),
            }
        )
        changed += 1
    output["oracle_support_policy"] = {
        "name": "oracle_generator_response_soft_support_weights",
        "scope_note": (
            "Leakage diagnostic upper bound. Uses realized historical future "
            "generator-response labels to weight support mixtures. Do not use "
            "as a production policy."
        ),
        "label_method": method,
        "label_metric": metric,
        "probability_temperature": float(probability_temperature),
        "changed_rows": int(changed),
        "diagnostics": diagnostics,
    }
    return output


def summarize_oracle_bridge(report: dict[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("oracle_support_policy", {}).get("diagnostics", [])
    if not diagnostics:
        return {"changed_rows": 0}
    cand_eff = np.asarray(
        [float(row["candidate_effective_n"]) for row in diagnostics], dtype=np.float64
    )
    support_eff = np.asarray(
        [float(row["support_effective_n"]) for row in diagnostics], dtype=np.float64
    )
    max_probs = np.asarray(
        [float(row["candidate_max_probability"]) for row in diagnostics],
        dtype=np.float64,
    )
    return {
        "changed_rows": int(len(diagnostics)),
        "candidate_effective_n_mean": float(np.mean(cand_eff)),
        "candidate_effective_n_median": float(np.median(cand_eff)),
        "candidate_max_probability_mean": float(np.mean(max_probs)),
        "candidate_max_probability_median": float(np.median(max_probs)),
        "support_effective_n_mean": float(np.mean(support_eff)),
        "support_effective_n_median": float(np.median(support_eff)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bridge-report", default=DEFAULT_BASE_BRIDGE_REPORT)
    parser.add_argument(
        "--candidate-bridge-report", default=DEFAULT_CANDIDATE_BRIDGE_REPORT
    )
    parser.add_argument("--scenario-report", default=DEFAULT_SCENARIO_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", default="narrative_generator_topk")
    parser.add_argument("--metric", default="energy_score_z")
    parser.add_argument("--probability-temperature", type=float, default=1.0)
    args = parser.parse_args()

    base_bridge = _load_json(args.base_bridge_report)
    candidate_bridge = _load_json(args.candidate_bridge_report)
    scenario_report = _load_json(args.scenario_report)
    report = build_oracle_weighted_bridge(
        base_bridge=base_bridge,
        candidate_bridge=candidate_bridge,
        scenario_report=scenario_report,
        method=str(args.method),
        metric=str(args.metric),
        probability_temperature=float(args.probability_temperature),
    )
    summary = summarize_oracle_bridge(report)
    report["oracle_support_policy"]["summary"] = summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge_path = output_dir / "oracle_soft_support_bridge_report.json"
    summary_path = output_dir / "oracle_soft_support_summary.json"
    _write_json(bridge_path, report)
    _write_json(
        summary_path,
        {
            "status": "ok",
            "scope_note": report["oracle_support_policy"]["scope_note"],
            "artifact_paths": {
                "bridge_report": str(bridge_path),
                "summary": str(summary_path),
            },
            "summary": summary,
        },
    )
    print(json.dumps({"bridge_report": str(bridge_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
