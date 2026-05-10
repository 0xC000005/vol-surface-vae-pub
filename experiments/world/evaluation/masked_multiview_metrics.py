from __future__ import annotations

import numpy as np

from experiments.world.evaluation.masked_multiview_data import MaskedMultiviewBatch
from experiments.world.evaluation.part1_metrics import (
    latent_prediction_metrics,
    representation_health_metrics,
    retrieval_metrics,
)


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    return arr.reshape(arr.shape[0], -1)


def barlow_cross_correlation_metrics(
    view_a: np.ndarray,
    view_b: np.ndarray,
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    a = _as_2d(view_a, "view_a")
    b = _as_2d(view_b, "view_b")
    if a.shape != b.shape:
        raise ValueError(f"view_a and view_b must match, got {a.shape} and {b.shape}")
    if a.shape[0] < 2:
        raise ValueError("Need at least two rows for cross-correlation metrics")

    a = (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + eps)
    b = (b - b.mean(axis=0, keepdims=True)) / (b.std(axis=0, keepdims=True) + eps)
    corr = (a.T @ b) / a.shape[0]
    diag = np.diag(corr)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "diag_mean": float(np.mean(diag)),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "diag_loss": float(np.mean((diag - 1.0) ** 2)),
        "offdiag_abs_mean": float(np.mean(np.abs(offdiag))) if offdiag.size else 0.0,
        "offdiag_abs_max": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "offdiag_loss": float(np.mean(offdiag * offdiag)) if offdiag.size else 0.0,
    }


def same_state_multiview_metrics(
    view_a: np.ndarray,
    view_b: np.ndarray,
) -> dict[str, dict[str, float]]:
    a = _as_2d(view_a, "view_a")
    b = _as_2d(view_b, "view_b")
    if a.shape != b.shape:
        raise ValueError(f"view_a and view_b must match, got {a.shape} and {b.shape}")
    return {
        "alignment": latent_prediction_metrics(a, b),
        "retrieval": retrieval_metrics(a, b, top_k=(1, 5, 10)),
        "barlow": barlow_cross_correlation_metrics(a, b),
        "view_a_health": representation_health_metrics(a),
        "view_b_health": representation_health_metrics(b),
    }


def _rate(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=bool)
    return float(np.mean(arr)) if arr.size else 0.0


def mask_visibility_summary(batch: MaskedMultiviewBatch) -> dict[str, object]:
    out: dict[str, object] = {
        "overall": {
            "observed_rate": _rate(batch.observed_mask),
            "view_a_visible_rate": _rate(batch.synthetic_mask_a),
            "view_b_visible_rate": _rate(batch.synthetic_mask_b),
            "view_a_observed_visible_rate": _rate(batch.observed_mask & batch.synthetic_mask_a),
            "view_b_observed_visible_rate": _rate(batch.observed_mask & batch.synthetic_mask_b),
        },
        "by_geometry": {},
        "by_family": {},
        "mask_family_a": {
            str(k): int(np.sum(batch.mask_family_a == k))
            for k in sorted(set(batch.mask_family_a.tolist()))
        },
        "mask_family_b": {
            str(k): int(np.sum(batch.mask_family_b == k))
            for k in sorted(set(batch.mask_family_b.tolist()))
        },
    }
    for field_name, values, key in (
        ("by_geometry", batch.token_metadata.geometry_id, "geometry"),
        ("by_family", batch.token_metadata.factor_family, "family"),
    ):
        rows: dict[str, dict[str, float | int]] = {}
        for label in sorted(set(values.tolist())):
            token_mask = values == label
            rows[str(label)] = {
                key: str(label),
                "n_tokens": int(np.sum(token_mask)),
                "observed_rate": _rate(batch.observed_mask[:, :, token_mask]),
                "view_a_visible_rate": _rate(batch.synthetic_mask_a[:, :, token_mask]),
                "view_b_visible_rate": _rate(batch.synthetic_mask_b[:, :, token_mask]),
                "view_a_observed_visible_rate": _rate(
                    (batch.observed_mask & batch.synthetic_mask_a)[:, :, token_mask]
                ),
                "view_b_observed_visible_rate": _rate(
                    (batch.observed_mask & batch.synthetic_mask_b)[:, :, token_mask]
                ),
            }
        out[field_name] = rows
    return out


def flattened_time_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected shape (B, T, C), got {arr.shape}")
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
