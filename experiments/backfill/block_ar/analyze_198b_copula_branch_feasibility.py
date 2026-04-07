#!/usr/bin/env python
"""
Pretest for the attentional-copula branch.

Goal:
  Quantify how much a stronger cross-cell copula alone can change the strongest
  current one-shot anchor (`183c`). This is a ceiling-style test: reorder 183c
  samples with a train-estimated Gaussian copula while preserving per-cell
  marginals, then check what actually moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.analyze_183c_best_mechanism import load_model
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_ci_coverage_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_regime_coverage_tests,
    run_time_series_tests,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv


def estimate_gaussian_copula(surfaces: np.ndarray, train_end: int) -> np.ndarray:
    train = surfaces[:train_end]
    changes = np.diff(train, axis=0).reshape(-1, 25)
    corr = np.corrcoef(changes, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return np.linalg.cholesky(corr)


def apply_ecc_reorder(samples: np.ndarray, L: np.ndarray, seed: int = 42) -> np.ndarray:
    n, k, t, h, w = samples.shape
    c = h * w
    flat = samples.reshape(n, k, t, c).transpose(0, 2, 1, 3).reshape(n * t, k, c)
    sorted_idx = np.argsort(flat, axis=1)
    sorted_vals = np.take_along_axis(flat, sorted_idx, axis=1)

    rng = np.random.RandomState(seed)
    z = rng.randn(n * t, k, c)
    templates = z @ L.T
    template_ranks = np.argsort(np.argsort(templates, axis=1), axis=1)
    reordered = np.take_along_axis(sorted_vals, template_ranks, axis=1)
    reordered = reordered.reshape(n, t, k, c).transpose(0, 2, 1, 3).reshape(n, k, t, h, w)
    return reordered


def turb_calm_ratio_from_samples(samples: np.ndarray, history_norm: torch.Tensor) -> float:
    vov, q20, q80 = regime_masks_from_history(history_norm)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    lower = np.quantile(samples, 0.05, axis=1)
    upper = np.quantile(samples, 0.95, axis=1)
    width = (upper - lower).mean(axis=(1, 2, 3))
    calm = float(width[calm_mask].mean()) if calm_mask.any() else float("nan")
    turb = float(width[turb_mask].mean()) if turb_mask.any() else float("nan")
    return float(turb / max(calm, 1e-12))


@torch.no_grad()
def sample_183c(model, history_norm: torch.Tensor, device: torch.device, n_samples: int, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, history_norm.shape[0], batch_size):
        end = min(start + batch_size, history_norm.shape[0])
        hist_b = history_norm[start:end].to(device)
        samples = model.sample_batched(hist_b, n_samples=n_samples)
        out.append(samples.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="198b copula branch feasibility pretest")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--train_end", type=int, default=4040)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--test_windows", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/198_design/198b_copula_branch_feasibility",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    L = estimate_gaussian_copula(data["surface"], train_end=args.train_end)
    history_norm, future_norm = build_test_subset(
        args.data_path, args.history_len, args.future_len, args.test_start, args.test_windows
    )
    model, _ = load_model(args.ckpt, str(device))
    samples = sample_183c(model, history_norm, device=device, n_samples=args.n_samples, batch_size=args.batch_size)
    reordered = apply_ecc_reorder(samples, L, seed=42)

    gt = denormalize_iv(future_norm).cpu().numpy()
    hist = denormalize_iv(history_norm).cpu().numpy()

    width_before = np.quantile(samples, 0.95, axis=1) - np.quantile(samples, 0.05, axis=1)
    width_after = np.quantile(reordered, 0.95, axis=1) - np.quantile(reordered, 0.05, axis=1)

    summary = {
        "config": {
            "ckpt": args.ckpt,
            "test_windows": int(history_norm.shape[0]),
            "n_samples": args.n_samples,
        },
        "invariance_checks": {
            "max_abs_width_change": float(np.max(np.abs(width_before - width_after))),
            "mean_abs_width_change": float(np.mean(np.abs(width_before - width_after))),
            "turb_calm_ratio_before": turb_calm_ratio_from_samples(samples, history_norm),
            "turb_calm_ratio_after": turb_calm_ratio_from_samples(reordered, history_norm),
        },
        "suite_like_metrics": {
            "before": {
                "coverage": run_ci_coverage_tests(samples, gt),
                "time_series": run_time_series_tests(samples, gt),
                "regime_coverage": run_regime_coverage_tests(samples, gt, hist),
                "distributional": run_distributional_fidelity_tests(samples, gt, hist),
                "cross_cell_correlation": run_cross_cell_correlation_tests(samples, gt),
            },
            "after": {
                "coverage": run_ci_coverage_tests(reordered, gt),
                "time_series": run_time_series_tests(reordered, gt),
                "regime_coverage": run_regime_coverage_tests(reordered, gt, hist),
                "distributional": run_distributional_fidelity_tests(reordered, gt, hist),
                "cross_cell_correlation": run_cross_cell_correlation_tests(reordered, gt),
            },
        },
        "interpretation": {
            "read": (
                "If copula reordering changes S9 but leaves width-based and coverage-based "
                "statistics essentially unchanged, a copula-only branch is unlikely to break "
                "the shared S3/S4/S7 frontier by itself."
            )
        },
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
