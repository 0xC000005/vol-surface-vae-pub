#!/usr/bin/env python
"""Test a hybrid narrative-plus-direction bridge representation.

This is an offline TestFlight. It does not call OpenAI or run the scenario
generator. The question is narrower: do explicit directional features help the
existing text-to-memory bridge separate hard negatives while preserving the full
narrative embedding channel?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_condition_bridge_evaluation import (  # noqa: E402
    build_bridge_examples,
    evaluate_condition_bridge,
    load_pipeline_artifacts,
    select_train_test_windows_from_report,
    summarize_bridge_metrics,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    train_narrative_adapter,
)
from experiments.backfill.block_ar.nl_text_conditioning import (
    normalize_rows,
)  # noqa: E402


MARKET_ORDER = (
    "IV_SURFACE",
    "IV_SKEW",
    "SPX",
    "US2Y",
    "US10Y",
    "BBB_OAS",
    "AAA_OAS",
    "USDJPY",
    "DXY",
    "GOLD",
    "CRUDE_OIL",
    "VIX",
)
MAGNITUDE_VALUE = {
    "FLAT": 0.0,
    "SMALL": 0.33,
    "MEDIUM": 0.66,
    "LARGE": 1.0,
}
DIRECTION_SIGN = {
    "UP": 1.0,
    "WIDER": 1.0,
    "DOWN": -1.0,
    "TIGHTER": -1.0,
    "FLAT": 0.0,
}
TOKEN_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*):\s*"
    r"(UP|DOWN|WIDER|TIGHTER|FLAT)"
    r"(?:\s+(FLAT|SMALL|MEDIUM|LARGE))?",
    re.IGNORECASE,
)


def _round(value: Any, digits: int = 12) -> float | None:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(raw):
        return None
    return round(raw, digits)


def _feature_names(market_order: tuple[str, ...] = MARKET_ORDER) -> list[str]:
    return [
        name
        for market in market_order
        for name in (f"{market}_signed", f"{market}_present")
    ]


def _set_market(
    values: np.ndarray,
    market_to_col: dict[str, int],
    market: str,
    signed_value: float,
) -> None:
    market_key = str(market).upper()
    if market_key not in market_to_col:
        return
    col = market_to_col[market_key]
    if abs(float(signed_value)) >= abs(float(values[col])):
        values[col] = float(signed_value)
        values[col + 1] = 1.0


def direction_features_from_text(
    text: str,
    *,
    market_order: tuple[str, ...] = MARKET_ORDER,
) -> tuple[np.ndarray, list[str]]:
    """Extract signed market-direction tokens from one example text."""

    vector = np.zeros(len(market_order) * 2, dtype=np.float32)
    market_to_col = {market: idx * 2 for idx, market in enumerate(market_order)}
    for match in TOKEN_RE.finditer(str(text).upper()):
        market, direction, magnitude = match.groups()
        if market not in market_to_col:
            continue
        sign = DIRECTION_SIGN.get(str(direction).upper(), 0.0)
        mag = MAGNITUDE_VALUE.get(str(magnitude or "LARGE").upper(), 1.0)
        _set_market(vector, market_to_col, market, sign * mag)
    return vector, _feature_names(market_order)


def _safe_normalize_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected 2-D array, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    output = np.zeros_like(array, dtype=np.float32)
    mask = norms[:, 0] > eps
    if np.any(mask):
        output[mask] = array[mask] / norms[mask]
    return output


def _has_direction(vector: np.ndarray) -> bool:
    signed = np.asarray(vector, dtype=np.float32)[0::2]
    return bool(np.any(np.abs(signed) > 1e-8))


def _lexical_direction_features(
    text: str,
    *,
    market_order: tuple[str, ...] = MARKET_ORDER,
) -> np.ndarray:
    """Limited fallback for artificial negative captions without machine tokens."""

    vector = np.zeros(len(market_order) * 2, dtype=np.float32)
    market_to_col = {market: idx * 2 for idx, market in enumerate(market_order)}
    lower = str(text).lower()
    if any(
        term in lower
        for term in (
            "equities rally",
            "equity rally",
            "risk appetite improves",
            "equities recover",
        )
    ):
        _set_market(vector, market_to_col, "SPX", 1.0)
    if any(
        term in lower
        for term in ("equities fall", "equity selloff", "de-rate", "under pressure")
    ):
        _set_market(vector, market_to_col, "SPX", -1.0)
    if any(
        term in lower
        for term in (
            "volatility falls",
            "volatility compress",
            "volatility declines",
            "volatility lower",
        )
    ):
        _set_market(vector, market_to_col, "VIX", -1.0)
        _set_market(vector, market_to_col, "IV_SURFACE", -1.0)
    if any(
        term in lower
        for term in (
            "volatility jumps",
            "volatility spike",
            "volatility climbs",
            "volatility is high",
        )
    ):
        _set_market(vector, market_to_col, "VIX", 1.0)
        _set_market(vector, market_to_col, "IV_SURFACE", 1.0)
    if any(term in lower for term in ("spreads tighten", "credit spreads tighten")):
        _set_market(vector, market_to_col, "BBB_OAS", -1.0)
        _set_market(vector, market_to_col, "AAA_OAS", -1.0)
    if any(
        term in lower
        for term in ("spreads widen", "credit spreads widen", "credit risk rising")
    ):
        _set_market(vector, market_to_col, "BBB_OAS", 1.0)
        _set_market(vector, market_to_col, "AAA_OAS", 1.0)
    if any(term in lower for term in ("rates rising", "rates higher", "higher yields")):
        _set_market(vector, market_to_col, "US2Y", 1.0)
        _set_market(vector, market_to_col, "US10Y", 1.0)
    if any(
        term in lower
        for term in (
            "rates lower",
            "yields fall",
            "yields lower",
            "flight-to-quality rate rally",
        )
    ):
        _set_market(vector, market_to_col, "US2Y", -1.0)
        _set_market(vector, market_to_col, "US10Y", -1.0)
    return vector


def direction_features_for_examples(
    examples: list[dict[str, Any]],
    *,
    market_order: tuple[str, ...] = MARKET_ORDER,
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic direction features for bridge examples.

    Anchors/positives normally carry a MARKET_IMPLICATIONS line. Artificial
    contrastive negatives sometimes lack machine tokens, so this fills obvious
    opposite/magnitude controls from the anchor's observed direction vector.
    """

    names = _feature_names(market_order)
    rows = [
        direction_features_from_text(
            str(example.get("text", "")), market_order=market_order
        )[0]
        for example in examples
    ]
    by_group: dict[str, list[int]] = {}
    for idx, example in enumerate(examples):
        by_group.setdefault(str(example.get("window_id", "")), []).append(idx)
    for indices in by_group.values():
        anchor = next(
            (idx for idx in indices if str(examples[idx].get("role")) == "anchor"),
            None,
        )
        anchor_vec = rows[anchor].copy() if anchor is not None else None
        for idx in indices:
            if _has_direction(rows[idx]):
                continue
            fallback = _lexical_direction_features(
                str(examples[idx].get("text", "")),
                market_order=market_order,
            )
            if _has_direction(fallback):
                rows[idx] = fallback
                continue
            if anchor_vec is None:
                continue
            role = str(examples[idx].get("role", ""))
            kind = str(examples[idx].get("kind", ""))
            if role == "negative" and kind == "opposite":
                filled = anchor_vec.copy()
                filled[0::2] *= -1.0
                rows[idx] = filled
            elif role == "negative" and kind == "magnitude":
                filled = anchor_vec.copy()
                filled[0::2] = np.sign(filled[0::2]) * 0.33
                rows[idx] = filled
    return np.vstack(rows).astype(np.float32), names


def build_hybrid_feature_matrix(
    text_embeddings: np.ndarray,
    direction_features: np.ndarray,
) -> np.ndarray:
    """Concatenate full narrative and direction channels with equal row norm.

    If a row has no direction features, the text channel keeps the full norm
    instead of being penalized for missing machine-direction evidence.
    """

    text = normalize_rows(np.asarray(text_embeddings, dtype=np.float32))
    direction = _safe_normalize_rows(np.asarray(direction_features, dtype=np.float32))
    dir_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    has_dir = dir_norm[:, 0] > 1e-12
    text_block = text.copy()
    direction_block = direction.copy()
    scale = np.float32(2**-0.5)
    text_block[has_dir] *= scale
    direction_block[has_dir] *= scale
    return np.concatenate([text_block, direction_block], axis=1).astype(np.float32)


def _filter_train_examples(
    examples: list[dict[str, Any]], train_indices: list[int]
) -> list[int]:
    train = set(int(idx) for idx in train_indices)
    rows = [
        row_idx
        for row_idx, example in enumerate(examples)
        if int(example["window_index"]) in train
        and str(example.get("role")) in {"anchor", "positive", "negative"}
    ]
    if not rows:
        raise ValueError("no training examples selected")
    return rows


def _train_variant(
    *,
    name: str,
    features: np.ndarray,
    examples: list[dict[str, Any]],
    memory_targets: np.ndarray,
    split: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    train_rows = _filter_train_examples(examples, split["train_indices"])
    train_features = features[np.asarray(train_rows, dtype=np.int64)]
    train_examples = [examples[idx] for idx in train_rows]
    target_indices = np.asarray(
        [
            -1 if example["target_index"] is None else int(example["target_index"])
            for example in train_examples
        ],
        dtype=np.int64,
    )
    roles = [str(example["role"]) for example in train_examples]
    groups = [str(example["window_id"]) for example in train_examples]
    train_result = train_narrative_adapter(
        train_features,
        memory_targets,
        target_indices,
        roles,
        groups,
        condition_dim=int(memory_targets.shape[1]),
        hidden_dim=args.hidden_dim,
        steps=int(args.adapter_steps),
        lr=float(args.adapter_lr),
        contrastive_weight=float(args.contrastive_weight),
        contrastive_margin=float(args.contrastive_margin),
        seed=int(args.seed),
        device=device,
    )
    adapter = train_result["adapter"]
    adapter.eval()
    with torch.no_grad():
        all_conditions = (
            adapter(torch.from_numpy(normalize_rows(features)).float().to(device))
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    evaluation = evaluate_condition_bridge(
        examples,
        all_conditions,
        memory_targets,
        train_indices=split["train_indices"],
        test_indices=split["test_indices"],
        top_k=int(args.top_k),
    )
    return {
        "variant": name,
        "feature_dim": int(features.shape[1]),
        "train_example_count": len(train_examples),
        "adapter_training": {
            "loss_first": _round(train_result["loss_first"]),
            "loss_last": _round(train_result["loss_last"]),
            "steps": int(args.adapter_steps),
        },
        "summary": summarize_bridge_metrics(evaluation),
        "evaluation": evaluation,
        "condition_vectors": all_conditions,
    }


def _metric_delta(
    variant: dict[str, Any], baseline: dict[str, Any], metric: str
) -> float | None:
    left = variant.get("summary", {}).get(metric)
    right = baseline.get("summary", {}).get(metric)
    if left is None or right is None:
        return None
    return _round(float(left) - float(right))


def run_hybrid_direction_bridge(args: argparse.Namespace) -> dict[str, Any]:
    report, arrays = load_pipeline_artifacts(args.input_report, args.input_npz)
    examples = build_bridge_examples(report)
    text_embeddings = np.asarray(arrays["text_embeddings"], dtype=np.float32)
    memory_targets = np.asarray(arrays["memory_targets"], dtype=np.float32)
    if text_embeddings.shape[0] != len(examples):
        raise ValueError("text_embeddings rows do not match rebuilt examples")
    split = select_train_test_windows_from_report(
        report,
        int(memory_targets.shape[0]),
        train_windows=int(args.train_windows),
        test_windows=int(args.test_windows),
    )
    direction_features, direction_names = direction_features_for_examples(examples)
    hybrid_features = build_hybrid_feature_matrix(text_embeddings, direction_features)
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    variants: dict[str, dict[str, Any]] = {}
    for name, features in (
        ("text_only", normalize_rows(text_embeddings)),
        ("hybrid_equal_norm", hybrid_features),
    ):
        result = _train_variant(
            name=name,
            features=features,
            examples=examples,
            memory_targets=memory_targets,
            split=split,
            args=args,
            device=device,
        )
        condition_vectors = result.pop("condition_vectors")
        variants[name] = result
        np.savez_compressed(
            Path(args.output_dir) / f"{name}_bridge_arrays.npz",
            condition_vectors=condition_vectors,
            memory_targets=memory_targets,
            text_embeddings=normalize_rows(text_embeddings),
            direction_features=direction_features,
            train_indices=np.asarray(split["train_indices"], dtype=np.int64),
            test_indices=np.asarray(split["test_indices"], dtype=np.int64),
        )

    baseline = variants["text_only"]
    hybrid = variants["hybrid_equal_norm"]
    comparison_metrics = [
        "heldout_mean_target_cosine",
        "heldout_median_target_cosine",
        "heldout_hard_negative_mean_margin",
        "heldout_hard_negative_mean_gap",
        "heldout_recall_at_3_test_pool",
        "heldout_mean_top_train_cosine",
    ]
    comparison = {
        metric: _metric_delta(hybrid, baseline, metric) for metric in comparison_metrics
    }
    target_delta = comparison.get("heldout_mean_target_cosine")
    margin_delta = comparison.get("heldout_hard_negative_mean_margin")
    gap_delta = comparison.get("heldout_hard_negative_mean_gap")
    falsified = bool(
        (target_delta is not None and target_delta < -0.02)
        or (
            (margin_delta is None or margin_delta <= 0.0)
            and (gap_delta is None or gap_delta <= 0.0)
        )
    )
    direction_nonzero = np.linalg.norm(direction_features[:, 0::2], axis=1) > 1e-8
    output = {
        "status": "ok",
        "scope_note": (
            "Offline TestFlight for hybrid full-narrative embedding plus explicit "
            "direction features. No OpenAI calls and no scenario rollout."
        ),
        "input_report": str(args.input_report),
        "input_npz": str(args.input_npz),
        "embedding_backend": report.get("embedding_backend"),
        "embedding_model": report.get("embedding_model"),
        "device": device,
        "split": split,
        "feature_contract": {
            "text_channel": "full saved narrative embedding",
            "direction_channel": "explicit market-direction tokens plus artificial-negative fallback",
            "fusion": "equal row norm when direction evidence exists; text-only norm when absent",
            "direction_feature_names": direction_names,
            "direction_nonzero_rows": int(np.sum(direction_nonzero)),
            "direction_zero_rows": int(
                len(direction_nonzero) - np.sum(direction_nonzero)
            ),
        },
        "hypothesis": (
            "A small explicit directional side channel can improve hard-negative "
            "separation for risk-on/reflation cases without replacing the full "
            "narrative embedding."
        ),
        "falsifier": (
            "Falsified if mean target cosine falls by more than 0.02 or both "
            "hard-negative margin and gap fail to improve versus text-only."
        ),
        "summary": {
            "variant_count": len(variants),
            "hybrid_delta_vs_text_only": comparison,
            "falsified": falsified,
            "decision": (
                "diagnostic_continue" if not falsified else "do_not_promote_hybrid"
            ),
        },
        "variants": variants,
        "artifact_paths": {
            "report": str(
                Path(args.output_dir) / "hybrid_direction_bridge_report.json"
            ),
            "markdown": str(
                Path(args.output_dir) / "hybrid_direction_bridge_report.md"
            ),
            "text_only_arrays": str(
                Path(args.output_dir) / "text_only_bridge_arrays.npz"
            ),
            "hybrid_arrays": str(
                Path(args.output_dir) / "hybrid_equal_norm_bridge_arrays.npz"
            ),
        },
    }
    return output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    baseline = report["variants"]["text_only"]["summary"]
    hybrid = report["variants"]["hybrid_equal_norm"]["summary"]
    delta = report["summary"]["hybrid_delta_vs_text_only"]
    lines = [
        "# Hybrid Direction Feature Bridge TestFlight",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['summary']['decision']}`",
        f"- Falsified: `{report['summary']['falsified']}`",
        f"- Device: `{report['device']}`",
        f"- Direction rows: `{report['feature_contract']['direction_nonzero_rows']}` nonzero, `{report['feature_contract']['direction_zero_rows']}` zero",
        "",
        "| Metric | Text Only | Hybrid | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in report["summary"]["hybrid_delta_vs_text_only"]:
        lines.append(
            f"| {metric} | {baseline.get(metric)} | {hybrid.get(metric)} | {delta.get(metric)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-windows", type=int, default=40)
    parser.add_argument("--test-windows", type=int, default=10)
    parser.add_argument("--adapter-steps", type=int, default=700)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-margin", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=878)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_hybrid_direction_bridge(args)
    json_path = output_dir / "hybrid_direction_bridge_report.json"
    markdown_path = output_dir / "hybrid_direction_bridge_report.md"
    _write_json(json_path, report)
    _write_markdown(markdown_path, report)
    print(
        json.dumps(
            {
                "report": str(json_path),
                "decision": report["summary"]["decision"],
                "falsified": report["summary"]["falsified"],
                "delta": report["summary"]["hybrid_delta_vs_text_only"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
