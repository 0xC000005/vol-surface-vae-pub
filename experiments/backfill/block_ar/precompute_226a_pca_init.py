#!/usr/bin/env python
"""
Precompute PCA initialization targets for 226a factor-decoupled flow.

Computes factor loadings and idiosyncratic scales from training data
in asinh-transformed innovation space (matching 212ai's representation).

Output: models/backfill/226a_pca_init.npz with:
  - lambda_init: (25, 6) top-6 PCA loadings scaled by sqrt(eigenvalues)
  - d_init: (25,) per-cell residual std after removing top-6 factors
  - explained_variance_ratio: (25,) per-component variance explained
  - mean_v: (25,) mean of v vectors (for reference)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
)
from experiments.backfill.block_ar.train_212x_h1_minimal_direct_stochastic_delta_local_scale import (
    build_local_scale_history_features,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute PCA init for 226a")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--ewma_alpha", type=float, default=0.20)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--factor_rank", type=int, default=6)
    parser.add_argument("--output", type=str, default="models/backfill/226a_pca_init.npz")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    # Build training windows (same split as 212ai)
    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    print(f"Training windows: {train_hist.shape[0]}")

    # Compute asinh-transformed innovations for all training windows
    all_v = []
    batch_size = 256
    for start in range(0, train_hist.shape[0], batch_size):
        end = min(start + batch_size, train_hist.shape[0])
        hist_batch = train_hist[start:end]
        target_batch = train_target[start:end]

        # Get local scale
        _feat, local_scale = build_local_scale_history_features(
            history_01=hist_batch,
            ewma_alpha=args.ewma_alpha,
            scale_floor=args.scale_floor,
            include_scale_feature=True,
        )

        # Compute delta and asinh-transform
        prev = hist_batch[:, -1].reshape(-1, 25)
        target_delta = target_batch - prev
        v = torch.asinh(target_delta / local_scale.clamp_min(args.scale_floor))
        all_v.append(v.cpu().numpy())

    v_all = np.concatenate(all_v, axis=0)  # (N_train, 25)
    print(f"v_all shape: {v_all.shape}, mean: {v_all.mean():.4f}, std: {v_all.std():.4f}")

    # PCA via SVD
    mean_v = v_all.mean(axis=0)
    v_centered = v_all - mean_v
    U, S, Vt = np.linalg.svd(v_centered, full_matrices=False)

    # Explained variance
    eigenvalues = S ** 2 / (v_all.shape[0] - 1)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    print(f"\nExplained variance by component:")
    cumulative = 0.0
    for i in range(min(10, len(eigenvalues))):
        cumulative += explained_variance_ratio[i]
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} (cumulative: {cumulative:.4f})")

    # Top-r loadings: columns of V scaled by sqrt(eigenvalue)
    r = args.factor_rank
    lambda_init = Vt[:r].T * np.sqrt(eigenvalues[:r])  # (25, r)
    print(f"\nFactor loadings shape: {lambda_init.shape}")
    print(f"Top-{r} cumulative variance: {explained_variance_ratio[:r].sum():.4f}")

    # Residual after removing top-r factors
    projection = v_centered @ Vt[:r].T @ Vt[:r]  # (N, 25)
    residual = v_centered - projection
    d_init = residual.std(axis=0)  # (25,)
    print(f"\nIdiosyncratic std per cell:")
    print(f"  mean: {d_init.mean():.4f}, min: {d_init.min():.4f}, max: {d_init.max():.4f}")

    # Verify: factor variance + idiosyncratic variance ≈ total variance
    total_var = v_centered.var(axis=0).mean()
    factor_var = projection.var(axis=0).mean()
    idio_var = residual.var(axis=0).mean()
    print(f"\nVariance decomposition:")
    print(f"  Total: {total_var:.4f}")
    print(f"  Factor: {factor_var:.4f} ({factor_var/total_var:.1%})")
    print(f"  Idiosyncratic: {idio_var:.4f} ({idio_var/total_var:.1%})")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        lambda_init=lambda_init.astype(np.float32),
        d_init=d_init.astype(np.float32),
        explained_variance_ratio=explained_variance_ratio.astype(np.float32),
        mean_v=mean_v.astype(np.float32),
        factor_rank=r,
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
