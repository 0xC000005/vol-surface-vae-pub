from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    return arr.reshape(arr.shape[0], -1)


def _safe_norm(values: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.linalg.norm(values, axis=axis).clip(min=1e-12)


def latent_prediction_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _as_2d(predicted, "predicted")
    tgt = _as_2d(target, "target")
    if pred.shape != tgt.shape:
        raise ValueError(f"predicted and target must match, got {pred.shape} and {tgt.shape}")

    err = pred - tgt
    cosine = (pred * tgt).sum(axis=1) / (_safe_norm(pred) * _safe_norm(tgt))
    return {
        "mse": float(np.mean(err * err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "cosine_mean": float(np.mean(cosine)),
        "cosine_min": float(np.min(cosine)),
    }


def retrieval_metrics(
    predicted: np.ndarray,
    candidates: np.ndarray,
    *,
    true_indices: np.ndarray | None = None,
    top_k: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    pred = _as_2d(predicted, "predicted")
    cand = _as_2d(candidates, "candidates")
    if pred.shape[1] != cand.shape[1]:
        raise ValueError(
            f"predicted and candidates must share latent dim, got {pred.shape[1]} and {cand.shape[1]}"
        )
    if true_indices is None:
        if pred.shape[0] > cand.shape[0]:
            raise ValueError("default true_indices require at least as many candidates as predictions")
        true_indices = np.arange(pred.shape[0], dtype=np.int64)
    true_indices = np.asarray(true_indices, dtype=np.int64)
    if true_indices.shape != (pred.shape[0],):
        raise ValueError("true_indices must have one entry per prediction")

    pred_norm = pred / _safe_norm(pred)[:, None]
    cand_norm = cand / _safe_norm(cand)[:, None]
    scores = pred_norm @ cand_norm.T
    order = np.argsort(-scores, axis=1)

    ranks = np.empty(pred.shape[0], dtype=np.int64)
    for row, true_idx in enumerate(true_indices):
        matches = np.where(order[row] == true_idx)[0]
        if len(matches) == 0:
            raise ValueError(f"true index {int(true_idx)} is outside candidates")
        ranks[row] = int(matches[0]) + 1

    out = {
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }
    for k in top_k:
        kk = int(k)
        if kk <= 0:
            raise ValueError("top_k entries must be positive")
        out[f"top{kk}"] = float(np.mean(ranks <= kk))
    return out


def representation_health_metrics(embeddings: np.ndarray) -> dict[str, float | list[float]]:
    z = _as_2d(embeddings, "embeddings")
    if z.shape[0] < 2:
        raise ValueError("Need at least two embeddings for covariance metrics")

    centered = z - z.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    variance = np.diag(cov).clip(min=0.0)
    eigvals = np.linalg.eigvalsh(cov).clip(min=0.0)[::-1]
    total = float(eigvals.sum())
    if total <= 1e-12:
        effective_rank = 0.0
        participation_ratio = 0.0
    else:
        probs = eigvals / total
        probs = probs[probs > 1e-12]
        effective_rank = float(np.exp(-np.sum(probs * np.log(probs))))
        participation_ratio = float((eigvals.sum() ** 2) / np.sum(eigvals * eigvals).clip(min=1e-12))

    std = np.sqrt(variance + 1e-12)
    corr = cov / np.outer(std, std)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]

    return {
        "variance_min": float(variance.min()),
        "variance_mean": float(variance.mean()),
        "variance_max": float(variance.max()),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "offdiag_abs_mean": float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0,
        "offdiag_abs_max": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "singular_values": np.sqrt(eigvals).tolist(),
    }
