from __future__ import annotations

import numpy as np


def _samples_to_bstc(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 5:
        return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], -1)
    if arr.ndim == 4:
        return arr
    raise ValueError("samples must have shape (B, S, T, C) or (B, S, T, H, W)")


def _truth_to_btc(truth: np.ndarray) -> np.ndarray:
    arr = np.asarray(truth, dtype=np.float64)
    if arr.ndim == 4:
        return arr.reshape(arr.shape[0], arr.shape[1], -1)
    if arr.ndim == 3:
        return arr
    raise ValueError("truth must have shape (B, T, C) or (B, T, H, W)")


def _safe_corrcoef(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    n_cols = rows.shape[1]
    if n_cols == 1:
        return np.ones((1, 1), dtype=np.float64)
    if rows.shape[0] < 2:
        return np.eye(n_cols, dtype=np.float64)
    corr = np.corrcoef(rows.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _effective_rank(corr: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(corr).clip(min=0.0)
    total = float(eigvals.sum())
    if total <= 1e-12:
        return 0.0
    probs = eigvals / total
    probs = probs[probs > 1e-12]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def _pc1_alignment(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 1:
        return 1.0
    avec = np.linalg.eigh(a)[1][:, -1]
    bvec = np.linalg.eigh(b)[1][:, -1]
    return abs(float(np.dot(avec, bvec)))


def _mean_pairwise_distance(samples: np.ndarray) -> float:
    flat = samples.reshape(samples.shape[0], samples.shape[1], -1)
    distances = []
    for context_samples in flat:
        if context_samples.shape[0] < 2:
            continue
        diffs = context_samples[:, None, :] - context_samples[None, :, :]
        tri = np.triu_indices(context_samples.shape[0], k=1)
        distances.append(np.linalg.norm(diffs[tri], axis=1))
    if not distances:
        return 0.0
    return float(np.concatenate(distances).mean())


def compact_path_sample_metrics(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    samp = _samples_to_bstc(samples)
    gt = _truth_to_btc(truth)
    if samp.shape[0] != gt.shape[0] or samp.shape[2:] != gt.shape[1:]:
        raise ValueError(f"samples and truth shapes are incompatible: {samp.shape} vs {gt.shape}")
    if samp.shape[1] < 1:
        raise ValueError("Need at least one sample per context")

    q05 = np.quantile(samp, 0.05, axis=1)
    q95 = np.quantile(samp, 0.95, axis=1)
    coverage_90 = float(((gt >= q05) & (gt <= q95)).mean())

    sample_var = float(np.var(samp, axis=1).mean())
    truth_var = float(np.var(gt))
    variance_ratio = sample_var / max(truth_var, 1e-12)

    gen_changes = np.diff(samp, axis=2).reshape(-1, samp.shape[-1])
    gt_changes = np.diff(gt, axis=1).reshape(-1, gt.shape[-1])
    gen_corr = _safe_corrcoef(gen_changes)
    gt_corr = _safe_corrcoef(gt_changes)

    return {
        "coverage_90": coverage_90,
        "variance_ratio": float(variance_ratio),
        "pairwise_distance_mean": _mean_pairwise_distance(samp),
        "corr_frobenius": float(np.linalg.norm(gen_corr - gt_corr, ord="fro")),
        "effective_rank": _effective_rank(gen_corr),
        "gt_effective_rank": _effective_rank(gt_corr),
        "pc1_alignment": _pc1_alignment(gen_corr, gt_corr),
        "sample_std_mean": float(np.std(samp, axis=1).mean()),
    }
