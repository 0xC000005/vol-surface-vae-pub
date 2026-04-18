#!/usr/bin/env python
"""
Precompute frozen PCA matrices for 233a coarse features (1-3 in Section 2.3):
  1. Mean level PCA: D-dim -> 4-dim
  2. Dispersion PCA: D-dim -> 4-dim
  3. Realized squared change PCA: D-dim -> 4-dim

Also computes q90_train — the quantile threshold for the jump indicator.

Saves to models/backfill/coarse_pca_233a.npz.
Training-split only, one-time run before any 233a training.
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

DATA_PATH = Path("data/vol_surface_with_ret.npz")
OUTPUT = Path("models/backfill/coarse_pca_233a.npz")
COARSE_WINDOW = 10
TRAIN_FRACTION = 0.8

def main():
    d = np.load(DATA_PATH)
    surface = d["surface"].reshape(-1, 25)   # (N, 25)
    N = len(surface)
    train_end = int(N * TRAIN_FRACTION)
    surface_train = surface[:train_end]

    # Build sliding windows of size COARSE_WINDOW
    windows = np.lib.stride_tricks.sliding_window_view(
        surface_train, (COARSE_WINDOW, 25)
    ).squeeze(axis=1)   # (N - W + 1, W, 25)

    # Feature 1: mean level
    mean_level = windows.mean(axis=1)              # (M, 25)

    # Feature 2: dispersion
    dispersion = windows.std(axis=1)               # (M, 25)

    # Feature 3: realized squared change
    d_win = np.diff(windows, axis=1)               # (M, W-1, 25)
    rsc = (d_win ** 2).mean(axis=1)                # (M, 25)

    # Fit PCA on each; reduce to 4 components
    pca_mean = PCA(n_components=4).fit(mean_level)
    pca_disp = PCA(n_components=4).fit(dispersion)
    pca_rsc  = PCA(n_components=4).fit(rsc)

    # q90 threshold on training-split ||Δx||_2
    all_deltas = np.diff(surface_train, axis=0)    # (N-1, 25)
    norms = np.linalg.norm(all_deltas, axis=-1)    # (N-1,)
    q90 = float(np.quantile(norms, 0.90))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        pca_mean_components=pca_mean.components_.astype(np.float32),
        pca_mean_mean=pca_mean.mean_.astype(np.float32),
        pca_disp_components=pca_disp.components_.astype(np.float32),
        pca_disp_mean=pca_disp.mean_.astype(np.float32),
        pca_rsc_components=pca_rsc.components_.astype(np.float32),
        pca_rsc_mean=pca_rsc.mean_.astype(np.float32),
        q90_train=np.array(q90, dtype=np.float32),
        coarse_window=np.array(COARSE_WINDOW, dtype=np.int32),
        train_fraction=np.array(TRAIN_FRACTION, dtype=np.float32),
    )
    print(f"Saved PCA artifact to {OUTPUT}")
    print(f"  q90_train = {q90:.6f}")

if __name__ == "__main__":
    main()
