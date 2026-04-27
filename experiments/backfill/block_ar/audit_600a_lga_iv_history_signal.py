#!/usr/bin/env python
"""600a: audit local-geometry IV-history features for 510a failure prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (  # noqa: E402
    build_rollout_windows,
    make_serializable,
)
from experiments.backfill.block_ar.audit_599a_factor_signal_for_iv_failures import (  # noqa: E402
    build_failure_targets,
    build_history_summary_features,
    ridge_oos_score,
    sample_native_model,
)


def softmax_neg_squared_distance(distances: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over negative squared distances, row-wise."""
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("distances must have shape [N,H]")
    temp = max(float(temperature), 1e-8)
    logits = -(d * d) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def build_lga_history_features(
    history: np.ndarray,
    *,
    prefix: str,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Deterministic local-geometry-attention-style features from history paths.

    This is an audit proxy for LGA, not a learned attention layer: the last state is
    the query, history states are keys/values, and the diagonal local metric is the
    inverse history-channel scale inside each window.
    """
    x = np.asarray(history, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("history must have shape [N,H,C]")
    query = x[:, -1]
    keys = x[:, :-1] if x.shape[1] > 1 else x[:, -1:]
    scale = x.std(axis=1) + float(eps)
    normalized = (keys - query[:, None, :]) / scale[:, None, :]
    distances = np.sqrt(np.mean(normalized * normalized, axis=2))
    weights = softmax_neg_squared_distance(distances, temperature=temperature)
    context = np.einsum("nh,nhc->nc", weights, keys)
    residual = query - context
    deltas = np.zeros_like(x)
    deltas[:, 1:] = x[:, 1:] - x[:, :-1]
    key_deltas = deltas[:, : keys.shape[1]]
    weighted_delta = np.einsum("nh,nhc->nc", weights, key_deltas)
    weighted_abs_delta = np.einsum("nh,nhc->nc", weights, np.abs(key_deltas))
    weighted_sq_resid = np.einsum("nh,nhc->nc", weights, (keys - query[:, None, :]) ** 2)

    entropy = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1, keepdims=True)
    effective_neighbors = 1.0 / np.maximum((weights * weights).sum(axis=1, keepdims=True), 1e-12)
    min_dist = distances.min(axis=1, keepdims=True)
    mean_dist = np.einsum("nh,nh->n", weights, distances)[:, None]
    ages = np.arange(keys.shape[1] - 1, -1, -1, dtype=np.float64)
    weighted_age = (weights * ages[None, :]).sum(axis=1, keepdims=True)
    last_weight = weights[:, -1:]
    blocks = [
        ("context", context),
        ("residual", residual),
        ("weighted_delta", weighted_delta),
        ("weighted_abs_delta", weighted_abs_delta),
        ("weighted_sq_resid", weighted_sq_resid),
        ("entropy", entropy),
        ("effective_neighbors", effective_neighbors),
        ("min_dist", min_dist),
        ("mean_dist", mean_dist),
        ("weighted_age", weighted_age),
        ("last_weight", last_weight),
    ]
    features = np.concatenate([values for _name, values in blocks], axis=1)
    names: list[str] = []
    for block_name, values in blocks:
        if values.shape[1] == 1 and block_name in {
            "entropy",
            "effective_neighbors",
            "min_dist",
            "mean_dist",
            "weighted_age",
            "last_weight",
        }:
            names.append(f"{prefix}_{block_name}")
        else:
            names.extend(f"{prefix}_{block_name}_{idx}" for idx in range(values.shape[1]))
    return features.astype(np.float64), names


def _target_summary(targets: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in targets.items()
    }


def _score_feature_sets(
    feature_sets: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    train_frac: float,
    ridge_alpha: float,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    lift_summary: dict[str, dict[str, float]] = {}
    for target_name, target in targets.items():
        target_scores = {
            name: ridge_oos_score(features, target, train_frac=train_frac, alpha=ridge_alpha)
            for name, features in feature_sets.items()
        }
        scores[target_name] = target_scores
        baseline = target_scores["iv_summary"]
        lift_summary[target_name] = {}
        for name, score in target_scores.items():
            if name == "iv_summary":
                continue
            lift_summary[target_name][f"{name}_minus_iv_summary_r2"] = float(score["r2"] - baseline["r2"])
            lift_summary[target_name][f"{name}_minus_iv_summary_auc"] = (
                float(score["auc"] - baseline["auc"])
                if np.isfinite(score["auc"]) and np.isfinite(baseline["auc"])
                else float("nan")
            )
    return scores, lift_summary


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# 600a LGA IV-History Signal Audit",
        "",
        f"- checkpoint: `{results['config']['checkpoint']}`",
        f"- windows: `{results['config']['n_windows']}`",
        f"- samples: `{results['config']['samples']}`",
        f"- temperature: `{results['config']['lga_temperature']}`",
        "",
        "## Lift vs IV Summary Baseline",
    ]
    for target, lift in results["lift_summary"].items():
        joined = ", ".join(f"`{key}`={value:.4f}" for key, value in lift.items())
        lines.append(f"- `{target}`: {joined}")
    lines.extend(["", "## Scores"])
    for target, scores in results["scores"].items():
        lines.append(f"### {target}")
        for name, score in scores.items():
            lines.append(
                f"- `{name}`: R2 `{score['r2']:.4f}`, corr `{score['corr']:.4f}`, "
                f"AUC `{score['auc']:.4f}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--lga_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    samples = sample_native_model(
        checkpoint=args.checkpoint,
        batch=batch,
        samples=args.samples,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        device=device,
        seed=args.seed,
    )
    iv_history = batch.history_01.detach().cpu().numpy().reshape(batch.history_01.shape[0], args.history_len, -1)
    iv_summary_features, iv_summary_names = build_history_summary_features(iv_history, "iv")
    lga_features, lga_names = build_lga_history_features(
        iv_history,
        prefix="iv_lga",
        temperature=args.lga_temperature,
    )
    summary_plus_lga = np.concatenate([iv_summary_features, lga_features], axis=1)
    feature_sets = {
        "iv_summary": iv_summary_features,
        "lga": lga_features,
        "iv_summary_lga": summary_plus_lga,
    }
    targets = build_failure_targets(samples, batch.future_01.detach().cpu().numpy())
    scores, lift_summary = _score_feature_sets(
        feature_sets,
        targets,
        train_frac=args.train_frac,
        ridge_alpha=args.ridge_alpha,
    )
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "n_windows": int(batch.history_01.shape[0]),
            "samples": int(args.samples),
            "train_frac": float(args.train_frac),
            "ridge_alpha": float(args.ridge_alpha),
            "lga_temperature": float(args.lga_temperature),
            "seed": int(args.seed),
            "n_iv_summary_features": int(iv_summary_features.shape[1]),
            "n_lga_features": int(lga_features.shape[1]),
            "iv_summary_names_head": iv_summary_names[:8],
            "lga_names_head": lga_names[:8],
        },
        "target_summary": _target_summary(targets),
        "scores": scores,
        "lift_summary": lift_summary,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), results)
    print(json.dumps(make_serializable({"target_summary": results["target_summary"], "lift_summary": lift_summary}), indent=2))


if __name__ == "__main__":
    main()
