#!/usr/bin/env python
"""
Regime Clustering for Volatility Surface Trajectories (Option J).

This utility clusters 30-day future trajectories into market regimes
(e.g., calm, crisis, spike, trending) based on trajectory features.

The regime labels are used for hierarchical sampling in DDPM:
1. Classifier predicts regime from history
2. Diffusion generates trajectory conditioned on sampled regime

Usage:
    python experiments/backfill/diffusion_poc/regime_clustering.py \
        --data data/vol_surface_with_ret.npz \
        --n_regimes 5 \
        --history_len 30 \
        --future_len 30 \
        --output data/regime_labels.npz
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def extract_trajectory_features(
    surfaces: np.ndarray,
    history_len: int = 30,
    future_len: int = 30,
) -> np.ndarray:
    """Extract features from each future trajectory for regime clustering.

    For each valid sequence starting position, we extract features from
    the FUTURE portion (what we're trying to predict). This allows the
    classifier to learn "given this history, expect this regime".

    Args:
        surfaces: (N, 5, 5) array of volatility surfaces
        history_len: Number of history frames (for indexing)
        future_len: Number of future frames to extract features from

    Returns:
        features: (N - history_len - future_len + 1, 5) trajectory features
            - mean: Average IV level across trajectory
            - std: Volatility of IV (intra-trajectory variation)
            - max_drawdown: Largest drop in mean IV
            - skew: Asymmetry of IV changes
            - range: High - low of mean IV
    """
    total_len = history_len + future_len
    n_sequences = len(surfaces) - total_len + 1

    if n_sequences <= 0:
        raise ValueError(f"Not enough data: need {total_len} frames, have {len(surfaces)}")

    features = np.zeros((n_sequences, 5), dtype=np.float32)

    for i in range(n_sequences):
        # Extract future portion (what we're predicting)
        future_start = i + history_len
        future_end = future_start + future_len
        future = surfaces[future_start:future_end]  # (future_len, 5, 5)

        # Compute mean IV per day (average across moneyness/tenor grid)
        daily_mean_iv = future.mean(axis=(1, 2))  # (future_len,)

        # Feature 1: Mean IV level
        features[i, 0] = daily_mean_iv.mean()

        # Feature 2: Volatility of IV (std of daily means)
        features[i, 1] = daily_mean_iv.std()

        # Feature 3: Max drawdown (largest drop from peak)
        cummax = np.maximum.accumulate(daily_mean_iv)
        drawdowns = cummax - daily_mean_iv
        features[i, 2] = drawdowns.max()

        # Feature 4: Skewness of daily changes
        daily_changes = np.diff(daily_mean_iv)
        if daily_changes.std() > 1e-8:
            features[i, 3] = ((daily_changes - daily_changes.mean()) ** 3).mean() / (daily_changes.std() ** 3)
        else:
            features[i, 3] = 0.0

        # Feature 5: Range (high - low)
        features[i, 4] = daily_mean_iv.max() - daily_mean_iv.min()

    return features


def cluster_regimes(
    features: np.ndarray,
    n_regimes: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans, StandardScaler]:
    """Cluster trajectory features into market regimes using K-means.

    Args:
        features: (N, 5) trajectory features
        n_regimes: Number of regime clusters
        random_state: Random seed for reproducibility

    Returns:
        labels: (N,) regime indices (0 to n_regimes-1)
        clusterer: Fitted KMeans model
        scaler: Fitted StandardScaler for feature normalization
    """
    # Standardize features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering
    clusterer = KMeans(
        n_clusters=n_regimes,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = clusterer.fit_predict(features_scaled)

    return labels, clusterer, scaler


def analyze_regimes(
    features: np.ndarray,
    labels: np.ndarray,
    n_regimes: int,
) -> None:
    """Print regime analysis for interpretation.

    Args:
        features: (N, 5) trajectory features
        labels: (N,) regime labels
        n_regimes: Number of regimes
    """
    feature_names = ['Mean IV', 'IV Volatility', 'Max Drawdown', 'Skewness', 'Range']

    print("\n" + "=" * 60)
    print("Regime Analysis")
    print("=" * 60)

    for regime in range(n_regimes):
        mask = labels == regime
        count = mask.sum()
        pct = 100.0 * count / len(labels)
        regime_features = features[mask]

        print(f"\nRegime {regime}: {count} samples ({pct:.1f}%)")
        print("-" * 40)
        for j, name in enumerate(feature_names):
            mean = regime_features[:, j].mean()
            std = regime_features[:, j].std()
            print(f"  {name:15s}: {mean:7.4f} +/- {std:.4f}")

    # Suggest regime names based on characteristics
    print("\n" + "-" * 60)
    print("Suggested Regime Interpretations:")
    print("-" * 60)

    regime_stats = []
    for regime in range(n_regimes):
        mask = labels == regime
        regime_features = features[mask]
        regime_stats.append({
            'regime': regime,
            'mean_iv': regime_features[:, 0].mean(),
            'iv_vol': regime_features[:, 1].mean(),
            'max_dd': regime_features[:, 2].mean(),
            'skew': regime_features[:, 3].mean(),
        })

    # Sort by mean IV level
    regime_stats.sort(key=lambda x: x['mean_iv'])

    for i, stats in enumerate(regime_stats):
        regime = stats['regime']
        if stats['iv_vol'] > np.median([s['iv_vol'] for s in regime_stats]):
            if stats['max_dd'] > np.median([s['max_dd'] for s in regime_stats]):
                name = "CRISIS/SPIKE"
            else:
                name = "HIGH_VOLATILITY"
        elif stats['mean_iv'] < np.median([s['mean_iv'] for s in regime_stats]):
            name = "CALM/LOW_VOL"
        else:
            name = "NORMAL/TRENDING"

        print(f"  Regime {regime}: {name} (IV={stats['mean_iv']:.3f}, Vol={stats['iv_vol']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate regime labels for volatility surface trajectories"
    )
    parser.add_argument(
        "--data", type=str, default="data/vol_surface_with_ret.npz",
        help="Path to volatility surface data"
    )
    parser.add_argument(
        "--n_regimes", type=int, default=5,
        help="Number of regime clusters (default: 5)"
    )
    parser.add_argument(
        "--history_len", type=int, default=30,
        help="History length in days (default: 30)"
    )
    parser.add_argument(
        "--future_len", type=int, default=30,
        help="Future length in days (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default="data/regime_labels.npz",
        help="Output path for regime labels"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for clustering (default: 42)"
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    surfaces = data['surface']
    print(f"Loaded {len(surfaces)} surfaces with shape {surfaces.shape}")

    # Extract features
    print(f"\nExtracting trajectory features (history={args.history_len}, future={args.future_len})...")
    features = extract_trajectory_features(
        surfaces,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    print(f"Extracted features for {len(features)} trajectories")

    # Cluster into regimes
    print(f"\nClustering into {args.n_regimes} regimes...")
    labels, clusterer, scaler = cluster_regimes(
        features,
        n_regimes=args.n_regimes,
        random_state=args.seed,
    )

    # Analyze regimes
    analyze_regimes(features, labels, args.n_regimes)

    # Save results
    print(f"\nSaving regime labels to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        labels=labels,
        features=features,
        centroids=clusterer.cluster_centers_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        history_len=args.history_len,
        future_len=args.future_len,
        n_regimes=args.n_regimes,
    )

    print(f"\nSaved:")
    print(f"  - labels: ({len(labels)},) regime indices")
    print(f"  - features: {features.shape} trajectory features")
    print(f"  - centroids: {clusterer.cluster_centers_.shape} cluster centers")
    print(f"  - history_len: {args.history_len}")
    print(f"  - future_len: {args.future_len}")
    print(f"  - n_regimes: {args.n_regimes}")

    print("\nDone!")


if __name__ == "__main__":
    main()
