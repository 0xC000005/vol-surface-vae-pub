#!/usr/bin/env python
"""Build a deterministic window-selection manifest for narrative labeling.

This script does not call OpenAI. It selects a representative set of joint39
historical windows to label later: weekly/report-style anchors, eventful market
states, calm controls, and temporal train/validation/test splits with embargo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    _load_joint39_block,
    _spec_names,
    build_window_metadata,
    key_market_summary,
)


MARKET_TAGS = {
    "IV_SURFACE": "iv",
    "IV_SKEW": "iv",
    "SPX": "equity",
    "US2Y": "rates",
    "US10Y": "rates",
    "BBB_OAS": "credit",
    "AAA_OAS": "credit",
    "USDJPY": "safe_haven_fx",
    "DXY": "safe_haven_fx",
    "GOLD": "safe_haven_fx",
    "CRUDE_OIL": "commodity",
    "VIX": "vol",
}
MAGNITUDE_WEIGHTS = {
    "flat": 0.0,
    "small": 1.0,
    "medium": 2.0,
    "large": 3.0,
}
TAG_WEIGHTS = {
    "equity": 1.4,
    "vol": 1.4,
    "credit": 1.3,
    "iv": 1.2,
    "rates": 1.1,
    "safe_haven_fx": 1.0,
    "commodity": 1.0,
}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _fact_delta(fact: dict[str, Any]) -> float:
    evidence = str(fact.get("evidence", ""))
    if "=" not in evidence:
        return 0.0
    try:
        return float(evidence.rsplit("=", 1)[1])
    except ValueError:
        return 0.0


def _market_delta_scales(facts_by_window: list[list[dict[str, Any]]]) -> dict[str, float]:
    by_market: dict[str, list[float]] = {}
    for facts in facts_by_window:
        for fact in facts:
            market = str(fact.get("market", ""))
            by_market.setdefault(market, []).append(abs(_fact_delta(fact)))
    scales: dict[str, float] = {}
    for market, values in by_market.items():
        arr = np.asarray(values, dtype=np.float32)
        positive = arr[arr > 0]
        if positive.size == 0:
            scales[market] = 1.0
            continue
        scale = float(np.quantile(positive, 0.75))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.max(positive))
        scales[market] = max(scale, 1e-8)
    return scales


def score_market_facts(
    facts: list[dict[str, Any]],
    *,
    delta_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score market-state salience from observed joint39 market facts."""

    score = 0.0
    tags: set[str] = set()
    signature: set[str] = set()
    contributions: list[dict[str, Any]] = []
    for fact in facts:
        market = str(fact.get("market", ""))
        direction = str(fact.get("direction", "flat"))
        magnitude = str(fact.get("magnitude", "flat"))
        if direction == "flat":
            continue
        tag = MARKET_TAGS.get(market, "other")
        tags.add(tag)
        signature.add(f"{market}:{direction}:{magnitude}")
        if delta_scales:
            relative_strength = min(abs(_fact_delta(fact)) / delta_scales.get(market, 1.0), 3.0)
        else:
            relative_strength = MAGNITUDE_WEIGHTS.get(magnitude, 1.0)
        contribution = relative_strength * TAG_WEIGHTS.get(tag, 1.0)
        score += contribution
        contributions.append(
            {
                "market": market,
                "direction": direction,
                "magnitude": magnitude,
                "tag": tag,
                "relative_strength": _round_float(relative_strength),
                "score": _round_float(contribution),
            }
        )
    by_market = {str(fact.get("market", "")): str(fact.get("direction", "")) for fact in facts}
    if (
        by_market.get("SPX") == "down"
        and by_market.get("VIX") == "up"
        and by_market.get("BBB_OAS") == "wider"
    ):
        score += 2.0
        tags.add("risk_off_cluster")
    if by_market.get("SPX") == "up" and by_market.get("VIX") == "down":
        score += 1.2
        tags.add("risk_on_cluster")
    return {
        "score": _round_float(score),
        "tags": sorted(tags),
        "signature": sorted(signature),
        "contributions": sorted(
            contributions,
            key=lambda row: (-float(row["score"]), str(row["market"])),
        ),
    }


def build_window_records(
    history_raw: np.ndarray,
    spec_names: list[str],
    window_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create per-window selection records from raw joint39 histories."""

    histories = np.asarray(history_raw, dtype=np.float32)
    if histories.ndim != 3:
        raise ValueError("history_raw must have shape [n_windows, history_len, n_cells]")
    metadata = window_metadata or []
    records: list[dict[str, Any]] = []
    facts_by_window = [key_market_summary(histories[idx], spec_names) for idx in range(int(histories.shape[0]))]
    delta_scales = _market_delta_scales(facts_by_window)
    for idx, facts in enumerate(facts_by_window):
        meta = metadata[idx] if idx < len(metadata) else {}
        salience = score_market_facts(facts, delta_scales=delta_scales)
        window_id = str(meta.get("window_id", f"joint39_val_{idx:04d}"))
        records.append(
            {
                "window_index": int(idx),
                "window_id": window_id,
                "source_index": int(meta.get("source_index", idx)),
                "calendar_start_date": meta.get("calendar_start_date"),
                "calendar_end_date": meta.get("calendar_end_date"),
                "forecast_start_date": meta.get("forecast_start_date"),
                "forecast_end_date": meta.get("forecast_end_date"),
                "salience_score": salience["score"],
                "tags": salience["tags"],
                "signature": salience["signature"],
                "top_contributions": salience["contributions"][:8],
                "selection_reasons": [],
            }
        )
    return records


def _signature_similarity(left: list[str], right: list[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _has_min_gap(record: dict[str, Any], selected: list[dict[str, Any]], min_gap: int) -> bool:
    idx = int(record["window_index"])
    return all(abs(idx - int(row["window_index"])) >= int(min_gap) for row in selected)


def _is_diverse(
    record: dict[str, Any],
    same_bucket: list[dict[str, Any]],
    diversity_threshold: float,
) -> bool:
    for row in same_bucket:
        if _signature_similarity(record.get("signature", []), row.get("signature", [])) > float(diversity_threshold):
            return False
    return True


def _add_reason(selected: dict[int, dict[str, Any]], record: dict[str, Any], reason: str) -> None:
    idx = int(record["window_index"])
    if idx not in selected:
        selected[idx] = {**record, "selection_reasons": []}
    reasons = selected[idx]["selection_reasons"]
    if reason not in reasons:
        reasons.append(reason)


def _weekly_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monday_rows = [
        row
        for row in records
        if (_parse_date(row.get("calendar_end_date")) or _parse_date(row.get("forecast_start_date")))
        and (_parse_date(row.get("calendar_end_date")) or _parse_date(row.get("forecast_start_date"))).weekday() == 0
    ]
    fallback_rows = [row for pos, row in enumerate(records) if pos % 5 == 0]
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for row in monday_rows + fallback_rows + records:
        idx = int(row["window_index"])
        if idx in seen:
            continue
        seen.add(idx)
        out.append(row)
    return out


def select_manifest_windows(
    records: list[dict[str, Any]],
    *,
    weekly_count: int,
    eventful_count: int,
    calm_count: int,
    min_gap: int = 5,
    diversity_threshold: float = 0.80,
) -> list[dict[str, Any]]:
    """Select representative windows and annotate why each was selected."""

    selected: dict[int, dict[str, Any]] = {}
    weekly_selected: list[dict[str, Any]] = []
    for row in _weekly_candidates(records):
        if len(weekly_selected) >= max(0, int(weekly_count)):
            break
        if not _has_min_gap(row, weekly_selected, int(min_gap)):
            continue
        _add_reason(selected, row, "weekly_anchor")
        weekly_selected.append(row)

    eventful_selected: list[dict[str, Any]] = []
    eventful_candidates = sorted(
        records,
        key=lambda row: (-float(row["salience_score"]), int(row["window_index"])),
    )
    for row in eventful_candidates:
        if len(eventful_selected) >= max(0, int(eventful_count)):
            break
        if not _is_diverse(row, eventful_selected, float(diversity_threshold)):
            continue
        _add_reason(selected, row, "eventful")
        eventful_selected.append(row)

    calm_selected: list[dict[str, Any]] = []
    calm_candidates = sorted(
        records,
        key=lambda row: (float(row["salience_score"]), int(row["window_index"])),
    )
    for row in calm_candidates:
        if len(calm_selected) >= max(0, int(calm_count)):
            break
        if not _has_min_gap(row, calm_selected, int(min_gap)):
            continue
        if not _is_diverse(row, calm_selected, float(diversity_threshold)):
            continue
        _add_reason(selected, row, "calm")
        calm_selected.append(row)

    return [selected[idx] for idx in sorted(selected)]


def split_selected_records(
    selected_records: list[dict[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    embargo: int,
) -> dict[str, list[dict[str, Any]]]:
    """Temporal split with boundary embargo applied to the earlier side."""

    rows = sorted(selected_records, key=lambda row: int(row["window_index"]))
    n = len(rows)
    if n == 0:
        return {"train": [], "validation": [], "test": [], "excluded_embargo": []}
    train_end = max(1, min(n, int(round(n * float(train_fraction)))))
    validation_n = max(1, int(round(n * float(validation_fraction)))) if n - train_end > 1 else 0
    validation_end = min(n, train_end + validation_n)
    train = rows[:train_end]
    validation = rows[train_end:validation_end]
    test = rows[validation_end:]
    excluded: list[dict[str, Any]] = []

    def _apply_boundary(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not left or not right:
            return left
        right_first = int(right[0]["window_index"])
        kept: list[dict[str, Any]] = []
        for row in left:
            if 0 <= right_first - int(row["window_index"]) <= int(embargo):
                excluded.append({**row, "exclusion_reason": "temporal_embargo"})
            else:
                kept.append(row)
        return kept

    train = _apply_boundary(train, validation)
    validation = _apply_boundary(validation, test)
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "excluded_embargo": excluded,
    }


def build_manifest(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = select_manifest_windows(
        records,
        weekly_count=int(args.weekly_count),
        eventful_count=int(args.eventful_count),
        calm_count=int(args.calm_count),
        min_gap=int(args.min_gap),
        diversity_threshold=float(args.diversity_threshold),
    )
    splits = split_selected_records(
        selected,
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        embargo=int(args.embargo),
    )
    counts = {
        "available_windows": len(records),
        "selected_windows": len(selected),
        "train_windows": len(splits["train"]),
        "validation_windows": len(splits["validation"]),
        "test_windows": len(splits["test"]),
        "excluded_embargo_windows": len(splits["excluded_embargo"]),
    }
    return {
        "status": "ok",
        "scope_note": (
            "Deterministic offline manifest for later OpenAI narrative labeling. "
            "Selection uses observed joint39 market-state salience, weekly/report "
            "anchors, calm controls, diversity filtering, and temporal embargo. "
            "This manifest does not contain generated narrative labels."
        ),
        "config": {
            "weekly_count": int(args.weekly_count),
            "eventful_count": int(args.eventful_count),
            "calm_count": int(args.calm_count),
            "min_gap": int(args.min_gap),
            "diversity_threshold": float(args.diversity_threshold),
            "train_fraction": float(args.train_fraction),
            "validation_fraction": float(args.validation_fraction),
            "embargo": int(args.embargo),
        },
        "counts": counts,
        "selected_window_ids": [str(row["window_id"]) for row in selected],
        "splits": splits,
    }


def run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import load_model

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _model, payload = load_model(args.checkpoint, device)
    (
        _history_level,
        _history_norm,
        _center,
        _scale,
        _drift_feature,
        history_raw,
        specs,
        block,
    ) = _load_joint39_block(args, payload)
    metadata = build_window_metadata(
        block,
        history_len=int(payload["config"]["history_len"]),
        future_len=int(payload["config"]["future_len"]),
    )
    records = build_window_records(history_raw, _spec_names(specs), metadata)
    manifest = build_manifest(records, args)
    manifest["checkpoint"] = str(args.checkpoint)
    manifest["data_window_count"] = int(history_raw.shape[0])
    _write_json(args.output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weekly-count", type=int, default=80)
    parser.add_argument("--eventful-count", type=int, default=100)
    parser.add_argument("--calm-count", type=int, default=40)
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument("--diversity-threshold", type=float, default=0.80)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--embargo", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state_scope", choices=["joint38"], default="joint38")
    parser.add_argument("--eval_split", choices=["val", "train", "train_tail"], default="val")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="log_level")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    args = parser.parse_args()
    manifest = run_manifest(args)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
