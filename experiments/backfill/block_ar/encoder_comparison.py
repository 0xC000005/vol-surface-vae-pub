"""Analysis A: DDPM vs MSE Encoder Representation Comparison.

WHY does the DDPM-pretrained encoder work (5/8) while MSE-pretrained, random,
and unfrozen encoders all fail (2/8)? What is structurally DIFFERENT about the
DDPM encoder's condition space?

Investigations:
1. Eigenspectrum comparison (PCA eigenvalue decay)
2. Effective rank comparison (participation ratio)
3. Linear probe comparison (coarse and fine feature R^2)
4. Per-cell reconstruction probe (25-cell R^2)
5. Condition vector statistics (mean, std, norm, inter-dim correlation)
6. CKA similarity (Centered Kernel Alignment)
7. Regime discrimination (k-means clustering, ARI)

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/encoder_comparison.py --device cuda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, denormalize_iv, normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

OUTPUT_DIR = Path("results/investigations/encoder_comparison")


# ──────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────

def load_ddpm_encoder(device):
    """Load encoder from DDPM pretrained model (block_ar_vol_scaled_30ep)."""
    ckpt = torch.load(
        "models/backfill/block_ar_vol_scaled_30ep/best_model.pt",
        map_location="cpu", weights_only=False,
    )
    # The DDPM model has encoder embedded; extract weights
    enc_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1,
    )
    encoder = GRUEncoder(enc_config)
    # Extract encoder.* keys and strip prefix
    sd = ckpt["model_state_dict"]
    enc_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd)
    encoder.eval().to(device)
    return encoder


def load_mse_encoder(device, path="models/backfill/gru_encoder_mse/encoder.pt"):
    """Load encoder from pretrain_encoder.py checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ec = ckpt["encoder_config"]
    enc_config = EncoderConfig(
        input_dim=ec["input_dim"],
        gru_hidden_dim=ec["gru_hidden_dim"],
        bottleneck_dim=ec["bottleneck_dim"],
        dropout=ec.get("dropout", 0.1),
    )
    encoder = GRUEncoder(enc_config)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval().to(device)
    return encoder


def load_random_encoder(device, seed=42):
    """Create encoder with random init."""
    torch.manual_seed(seed)
    enc_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1,
    )
    encoder = GRUEncoder(enc_config)
    encoder.eval().to(device)
    return encoder


# ──────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────

def compute_condition_vectors(encoder, history_tensor, batch_size=64):
    """Get condition vectors for all windows. Returns numpy (N, D)."""
    vecs = []
    N = history_tensor.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = history_tensor[i:i+batch_size]
            c = encoder(batch)
            vecs.append(c.cpu().numpy())
    return np.concatenate(vecs, axis=0)


def classify_regime(hist_np):
    """Classify windows into 3 regimes: 0=calm, 1=normal, 2=turb."""
    mean_iv = hist_np.mean(axis=(-1, -2))  # (N, T)
    daily_chg = np.diff(mean_iv, axis=1)
    vov = daily_chg.std(axis=1)
    q33, q67 = np.percentile(vov, [33, 67])
    labels = np.ones(len(vov), dtype=int)  # 1 = normal
    labels[vov <= q33] = 0  # calm
    labels[vov >= q67] = 2  # turb
    return labels


def compute_features(hist_np, fut_np):
    """Compute various features from history/future for linear probes.

    Returns dict of {feature_name: (N,) or (N, D) array}.
    """
    # Denormalize (input is in [-1, 1])
    hist_iv = (hist_np + 1.0) / 2.0  # (N, 30, 5, 5)
    fut_iv = (fut_np + 1.0) / 2.0

    N = hist_iv.shape[0]
    features = {}

    # 1. Mean IV level (scalar)
    features["mean_iv"] = hist_iv[:, -1].mean(axis=(-1, -2))  # last frame mean

    # 2. Vol-of-vol (scalar) — std of daily mean-IV changes
    mean_ts = hist_iv.mean(axis=(-1, -2))  # (N, 30)
    daily_chg = np.diff(mean_ts, axis=1)
    features["vol_of_vol"] = daily_chg.std(axis=1)

    # 3. Skew — difference between OTM put and OTM call (moneyness dim)
    # row 0 = deep ITM put / deep OTM call, row 4 = deep OTM put / deep ITM call
    # skew = mean(col4) - mean(col0)  (high moneyness vs low moneyness)
    last_frame = hist_iv[:, -1]  # (N, 5, 5)
    features["skew"] = last_frame[:, :, 0].mean(axis=1) - last_frame[:, :, 4].mean(axis=1)

    # 4. Term slope — difference between long tenor and short tenor
    features["term_slope"] = last_frame[:, 4, :].mean(axis=1) - last_frame[:, 0, :].mean(axis=1)

    # 5. Realized vol 5d — std of last 5 days mean IV changes
    if hist_iv.shape[1] >= 6:
        short_chg = np.diff(mean_ts[:, -6:], axis=1)
        features["realized_vol_5d"] = short_chg.std(axis=1)
    else:
        features["realized_vol_5d"] = daily_chg[:, -5:].std(axis=1)

    # 6. Future IV change (1-step ahead) — mean IV of frame 0 minus last history frame
    features["future_iv_change"] = fut_iv[:, 0].mean(axis=(-1, -2)) - hist_iv[:, -1].mean(axis=(-1, -2))

    # 7. Future vol (30-step std of mean-IV changes)
    fut_mean_ts = fut_iv.mean(axis=(-1, -2))
    fut_daily_chg = np.diff(fut_mean_ts, axis=1)
    features["future_vol"] = fut_daily_chg.std(axis=1)

    # 8. Mean-reversion signal — how far current level is from 30d average
    features["mr_signal"] = hist_iv[:, -1].mean(axis=(-1, -2)) - mean_ts.mean(axis=1)

    # 9. ATM level (center cell)
    features["atm_level"] = last_frame[:, 2, 2]

    # 10. Curvature (butterfly) — center minus wings average
    features["curvature"] = last_frame[:, 2, 2] - 0.5 * (last_frame[:, 2, 0] + last_frame[:, 2, 4])

    return features


# ──────────────────────────────────────────────────────────────────────
# Investigation 1 & 2: Eigenspectrum and Effective Rank
# ──────────────────────────────────────────────────────────────────────

def eigenspectrum_analysis(cond_vecs_dict, output_dir):
    """PCA eigenvalue decay and effective rank for each encoder."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 1 & 2: EIGENSPECTRUM & EFFECTIVE RANK")
    print("=" * 70)

    results = {}
    for name, c in cond_vecs_dict.items():
        c_centered = c - c.mean(axis=0)
        cov = np.cov(c_centered.T)  # (D, D)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # descending
        eigenvalues = np.maximum(eigenvalues, 0)  # numerical stability

        # Normalize
        total_var = eigenvalues.sum()
        ev_norm = eigenvalues / (total_var + 1e-10)
        cumvar = np.cumsum(ev_norm)

        # Participation ratio (effective rank)
        pr = (eigenvalues.sum() ** 2) / (np.sum(eigenvalues ** 2) + 1e-10)

        # Number of components for 90%, 95%, 99% variance
        n90 = int(np.searchsorted(cumvar, 0.90)) + 1
        n95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n99 = int(np.searchsorted(cumvar, 0.99)) + 1

        # 1% threshold rank
        threshold_rank = int((eigenvalues > 0.01 * eigenvalues[0]).sum())

        # Top eigenvalue dominance
        top1_pct = ev_norm[0] * 100
        top5_pct = ev_norm[:5].sum() * 100
        top10_pct = ev_norm[:10].sum() * 100

        results[name] = {
            "participation_ratio": float(pr),
            "threshold_rank_1pct": threshold_rank,
            "n_components_90pct": n90,
            "n_components_95pct": n95,
            "n_components_99pct": n99,
            "top1_var_pct": float(top1_pct),
            "top5_var_pct": float(top5_pct),
            "top10_var_pct": float(top10_pct),
            "eigenvalues_top20": eigenvalues[:20].tolist(),
            "cumvar_top20": cumvar[:20].tolist(),
        }

        print(f"\n  {name}:")
        print(f"    Participation ratio (effective rank): {pr:.2f}")
        print(f"    1% threshold rank: {threshold_rank}")
        print(f"    Components for 90%/95%/99% var: {n90} / {n95} / {n99}")
        print(f"    Top-1/5/10 var %: {top1_pct:.1f}% / {top5_pct:.1f}% / {top10_pct:.1f}%")
        print(f"    Top 10 eigenvalues: {eigenvalues[:10].round(4).tolist()}")

    # Save eigenvalue curves for plotting
    np.savez(
        output_dir / "eigenspectrum.npz",
        **{f"{name}_eigenvalues": results[name]["eigenvalues_top20"] for name in results},
    )

    return results


# ──────────────────────────────────────────────────────────────────────
# Investigation 3: Linear Probe Comparison
# ──────────────────────────────────────────────────────────────────────

def linear_probe_analysis(cond_vecs_dict, train_cond_vecs_dict,
                          test_features, train_features, output_dir):
    """Ridge regression from condition -> various features. Compare R^2."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: LINEAR PROBE COMPARISON (Condition -> Features)")
    print("=" * 70)

    feature_names = list(test_features.keys())
    results = {}

    for name in cond_vecs_dict:
        c_train = train_cond_vecs_dict[name]
        c_test = cond_vecs_dict[name]
        results[name] = {}

        for feat_name in feature_names:
            y_train = train_features[feat_name]
            y_test = test_features[feat_name]

            # Ridge regression (alpha=1.0 to avoid overfitting)
            ridge = Ridge(alpha=1.0)
            ridge.fit(c_train, y_train)
            y_pred_train = ridge.predict(c_train)
            y_pred_test = ridge.predict(c_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            results[name][feat_name] = {
                "r2_train": float(r2_train),
                "r2_test": float(r2_test),
            }

    # Print comparison table
    print(f"\n{'Feature':<20s}", end="")
    for name in cond_vecs_dict:
        print(f"  {name:>12s}", end="")
    print()
    print("-" * (20 + 14 * len(cond_vecs_dict)))

    for feat_name in feature_names:
        print(f"{feat_name:<20s}", end="")
        for name in cond_vecs_dict:
            r2 = results[name][feat_name]["r2_test"]
            print(f"  {r2:12.4f}", end="")
        print()

    return results


# ──────────────────────────────────────────────────────────────────────
# Investigation 4: Per-Cell Reconstruction Probe
# ──────────────────────────────────────────────────────────────────────

def percell_probe_analysis(cond_vecs_dict, train_cond_vecs_dict,
                           test_future, train_future, output_dir):
    """Ridge regression from condition -> each of 25 cells of next frame.

    If MSE encoder predicts individual cells better, the decoder doesn't
    need noise for prediction — explaining diversity collapse.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION 4: PER-CELL RECONSTRUCTION PROBE")
    print("=" * 70)
    print("  (R^2: condition -> next-frame cell value)")

    # Target: first future frame, denormalized, each cell
    train_target = ((train_future[:, 0] + 1.0) / 2.0).reshape(-1, 25)  # (N, 25)
    test_target = ((test_future[:, 0] + 1.0) / 2.0).reshape(-1, 25)

    labels_k = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
    labels_t = ["1M", "3M", "6M", "1Y", "2Y"]

    results = {}

    for name in cond_vecs_dict:
        c_train = train_cond_vecs_dict[name]
        c_test = cond_vecs_dict[name]

        r2_cells = np.zeros(25)
        for cell_idx in range(25):
            ridge = Ridge(alpha=1.0)
            ridge.fit(c_train, train_target[:, cell_idx])
            pred = ridge.predict(c_test)
            r2_cells[cell_idx] = r2_score(test_target[:, cell_idx], pred)

        r2_grid = r2_cells.reshape(5, 5)
        results[name] = {
            "r2_per_cell": r2_cells.tolist(),
            "r2_mean": float(r2_cells.mean()),
            "r2_min": float(r2_cells.min()),
            "r2_max": float(r2_cells.max()),
            "r2_grid": r2_grid.tolist(),
        }

    # Print grids
    for name in cond_vecs_dict:
        r2_grid = np.array(results[name]["r2_grid"])
        print(f"\n  {name} (mean R^2 = {results[name]['r2_mean']:.4f}):")
        print(f"  {'':>8s}", end="")
        for k in labels_k:
            print(f"  {k:>8s}", end="")
        print()
        for i, t in enumerate(labels_t):
            print(f"  {t:>8s}", end="")
            for j in range(5):
                print(f"  {r2_grid[i, j]:8.4f}", end="")
            print()

    # Print delta (DDPM - MSE)
    if "DDPM" in results and "MSE" in results:
        ddpm_grid = np.array(results["DDPM"]["r2_grid"])
        mse_grid = np.array(results["MSE"]["r2_grid"])
        delta = ddpm_grid - mse_grid
        print(f"\n  DELTA (DDPM - MSE):")
        print(f"  {'':>8s}", end="")
        for k in labels_k:
            print(f"  {k:>8s}", end="")
        print()
        for i, t in enumerate(labels_t):
            print(f"  {t:>8s}", end="")
            for j in range(5):
                print(f"  {delta[i, j]:+8.4f}", end="")
            print()
        print(f"  Mean delta: {delta.mean():+.4f}")
        print(f"  DDPM mean R^2: {results['DDPM']['r2_mean']:.4f}")
        print(f"  MSE  mean R^2: {results['MSE']['r2_mean']:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Investigation 5: Condition Vector Statistics
# ──────────────────────────────────────────────────────────────────────

def condition_stats_analysis(cond_vecs_dict, output_dir):
    """Compare mean, std, norm, inter-dim correlation for both encoders."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 5: CONDITION VECTOR STATISTICS")
    print("=" * 70)

    results = {}

    print(f"\n{'Metric':<25s}", end="")
    for name in cond_vecs_dict:
        print(f"  {name:>12s}", end="")
    print()
    print("-" * (25 + 14 * len(cond_vecs_dict)))

    metrics = [
        "mean_abs", "global_std", "mean_norm_L2", "std_norm_L2",
        "max_abs", "dim_std_range", "active_dims_10pct",
        "active_dims_1pct", "mean_interdim_corr", "max_interdim_corr",
        "condition_entropy",
    ]

    for metric in metrics:
        print(f"{metric:<25s}", end="")
        for name, c in cond_vecs_dict.items():
            if name not in results:
                results[name] = {}

            if metric == "mean_abs":
                val = float(np.abs(c).mean())
            elif metric == "global_std":
                val = float(c.std())
            elif metric == "mean_norm_L2":
                norms = np.linalg.norm(c, axis=1)
                val = float(norms.mean())
            elif metric == "std_norm_L2":
                norms = np.linalg.norm(c, axis=1)
                val = float(norms.std())
            elif metric == "max_abs":
                val = float(np.abs(c).max())
            elif metric == "dim_std_range":
                dim_stds = c.std(axis=0)
                val = float(dim_stds.max() / (dim_stds.min() + 1e-10))
            elif metric == "active_dims_10pct":
                dim_stds = c.std(axis=0)
                val = int((dim_stds > 0.1 * dim_stds.max()).sum())
            elif metric == "active_dims_1pct":
                dim_stds = c.std(axis=0)
                val = int((dim_stds > 0.01 * dim_stds.max()).sum())
            elif metric == "mean_interdim_corr":
                corr_matrix = np.corrcoef(c.T)  # (D, D)
                np.fill_diagonal(corr_matrix, 0)
                val = float(np.abs(corr_matrix).mean())
            elif metric == "max_interdim_corr":
                corr_matrix = np.corrcoef(c.T)
                np.fill_diagonal(corr_matrix, 0)
                val = float(np.abs(corr_matrix).max())
            elif metric == "condition_entropy":
                # Entropy of the eigenvalue distribution (spectral entropy)
                c_centered = c - c.mean(axis=0)
                cov = np.cov(c_centered.T)
                ev = np.linalg.eigvalsh(cov)[::-1]
                ev = np.maximum(ev, 1e-10)
                p = ev / ev.sum()
                val = float(-np.sum(p * np.log(p + 1e-10)))

            results[name][metric] = val
            if isinstance(val, int):
                print(f"  {val:12d}", end="")
            else:
                print(f"  {val:12.4f}", end="")
        print()

    return results


# ──────────────────────────────────────────────────────────────────────
# Investigation 6: CKA Similarity
# ──────────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """Compute Linear CKA (Kornblith et al., 2019).

    X: (N, D1), Y: (N, D2). Both centered.
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X.T @ X  # (D1, D1)
    YtY = Y.T @ Y  # (D2, D2)
    YtX = Y.T @ X  # (D2, D1)

    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def cka_analysis(cond_vecs_dict, output_dir):
    """CKA similarity between all encoder pairs."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 6: CKA SIMILARITY")
    print("=" * 70)

    names = list(cond_vecs_dict.keys())
    n = len(names)
    cka_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = linear_cka(cond_vecs_dict[names[i]], cond_vecs_dict[names[j]])

    print(f"\n{'':>12s}", end="")
    for name in names:
        print(f"  {name:>12s}", end="")
    print()

    for i, name_i in enumerate(names):
        print(f"{name_i:>12s}", end="")
        for j in range(n):
            print(f"  {cka_matrix[i, j]:12.4f}", end="")
        print()

    results = {
        "names": names,
        "cka_matrix": cka_matrix.tolist(),
    }
    return results


# ──────────────────────────────────────────────────────────────────────
# Investigation 7: Regime Discrimination
# ──────────────────────────────────────────────────────────────────────

def regime_discrimination_analysis(cond_vecs_dict, regime_labels, output_dir):
    """k-means clustering (k=3) on conditions, compare to true regime labels."""
    print("\n" + "=" * 70)
    print("INVESTIGATION 7: REGIME DISCRIMINATION")
    print("=" * 70)

    results = {}

    print(f"\n{'Encoder':<15s}  {'ARI':>8s}  {'Purity':>8s}  {'Silhouette':>10s}  {'Inertia':>10s}")
    print("-" * 65)

    for name, c in cond_vecs_dict.items():
        # Standardize
        scaler = StandardScaler()
        c_scaled = scaler.fit_transform(c)

        # k-means with k=3
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        pred_labels = km.fit_predict(c_scaled)

        # Adjusted Rand Index
        ari = adjusted_rand_score(regime_labels, pred_labels)

        # Purity
        from collections import Counter
        purity = 0
        for cluster_id in range(3):
            mask = pred_labels == cluster_id
            if mask.sum() == 0:
                continue
            most_common = Counter(regime_labels[mask]).most_common(1)[0][1]
            purity += most_common
        purity /= len(regime_labels)

        # Silhouette score (subsample for speed)
        from sklearn.metrics import silhouette_score
        n_sub = min(2000, len(c_scaled))
        idx = np.random.RandomState(42).choice(len(c_scaled), n_sub, replace=False)
        sil = silhouette_score(c_scaled[idx], pred_labels[idx])

        inertia = km.inertia_

        results[name] = {
            "ari": float(ari),
            "purity": float(purity),
            "silhouette": float(sil),
            "inertia": float(inertia),
        }

        print(f"{name:<15s}  {ari:8.4f}  {purity:8.4f}  {sil:10.4f}  {inertia:10.1f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# BONUS: Condition information sufficiency
# ──────────────────────────────────────────────────────────────────────

def information_sufficiency_analysis(cond_vecs_dict, train_cond_vecs_dict,
                                     test_future, train_future, output_dir):
    """For each encoder, how well can condition predict the MEAN of next 30 frames
    vs the VARIANCE of next 30 frames?

    Key insight: if condition encodes "too much" about mean trajectory,
    the decoder has nothing left for noise to contribute.
    """
    print("\n" + "=" * 70)
    print("BONUS: INFORMATION SUFFICIENCY — Mean vs Variance Prediction")
    print("=" * 70)

    # Targets: mean and std of future 30 frames
    train_fut_iv = (train_future + 1.0) / 2.0  # (N, 30, 5, 5)
    test_fut_iv = (test_future + 1.0) / 2.0

    # Mean trajectory (average IV over 30 future frames, per cell)
    train_mean_traj = train_fut_iv.mean(axis=1).reshape(-1, 25)  # (N, 25)
    test_mean_traj = test_fut_iv.mean(axis=1).reshape(-1, 25)

    # Std trajectory (std of IV over 30 future frames, per cell)
    train_std_traj = train_fut_iv.std(axis=1).reshape(-1, 25)
    test_std_traj = test_fut_iv.std(axis=1).reshape(-1, 25)

    # Cumulative change (how much IV moves over 30 frames)
    train_cum_change = (train_fut_iv[:, -1] - train_fut_iv[:, 0]).reshape(-1, 25)
    test_cum_change = (test_fut_iv[:, -1] - test_fut_iv[:, 0]).reshape(-1, 25)

    results = {}

    for name in cond_vecs_dict:
        c_train = train_cond_vecs_dict[name]
        c_test = cond_vecs_dict[name]

        metrics = {}
        for target_name, y_train, y_test in [
            ("mean_traj", train_mean_traj, test_mean_traj),
            ("std_traj", train_std_traj, test_std_traj),
            ("cum_change", train_cum_change, test_cum_change),
        ]:
            ridge = Ridge(alpha=1.0)
            ridge.fit(c_train, y_train)
            pred = ridge.predict(c_test)
            r2_per_cell = np.array([
                r2_score(y_test[:, i], pred[:, i]) for i in range(25)
            ])
            metrics[target_name] = {
                "r2_mean": float(r2_per_cell.mean()),
                "r2_min": float(r2_per_cell.min()),
                "r2_max": float(r2_per_cell.max()),
            }

        results[name] = metrics

    print(f"\n{'Encoder':<15s}  {'Mean R^2':>10s}  {'Std R^2':>10s}  {'CumChg R^2':>12s}")
    print("-" * 55)
    for name in cond_vecs_dict:
        m = results[name]
        print(f"{name:<15s}  {m['mean_traj']['r2_mean']:10.4f}  "
              f"{m['std_traj']['r2_mean']:10.4f}  "
              f"{m['cum_change']['r2_mean']:12.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# BONUS 2: Condition noise sensitivity
# ──────────────────────────────────────────────────────────────────────

def noise_sensitivity_analysis(cond_vecs_dict, output_dir):
    """How stable are the condition vectors under perturbation?

    Measure how much adding noise to history changes the condition vector.
    DDPM encoder (trained across noise levels 0-1000) should be MORE robust
    to input perturbations, while MSE encoder (trained at noise=0) may be
    more sensitive — every small change in input changes the condition.
    """
    print("\n" + "=" * 70)
    print("BONUS 2: INPUT PERTURBATION SENSITIVITY")
    print("=" * 70)
    print("  (Skipped — requires re-running encoder on perturbed inputs)")
    print("  (See condition_stats for related diagnostics)")

    return {}


# ──────────────────────────────────────────────────────────────────────
# BONUS 3: Per-dimension informativeness
# ──────────────────────────────────────────────────────────────────────

def dimension_informativeness(cond_vecs_dict, train_cond_vecs_dict,
                               test_features, train_features, output_dir):
    """Which dimensions carry information? For each dim, compute
    correlation with each feature. Compare which dims are 'alive'."""
    print("\n" + "=" * 70)
    print("BONUS 3: PER-DIMENSION INFORMATIVENESS")
    print("=" * 70)

    results = {}
    key_features = ["mean_iv", "vol_of_vol", "skew", "term_slope"]

    for name, c in cond_vecs_dict.items():
        dim_scores = np.zeros(c.shape[1])
        for feat_name in key_features:
            y = test_features[feat_name]
            for d in range(c.shape[1]):
                corr = np.corrcoef(c[:, d], y)[0, 1]
                dim_scores[d] = max(dim_scores[d], abs(corr))

        # Sort by score
        top_dims = np.argsort(dim_scores)[::-1]
        n_informative = (dim_scores > 0.3).sum()
        n_highly_informative = (dim_scores > 0.5).sum()

        results[name] = {
            "n_informative_r03": int(n_informative),
            "n_informative_r05": int(n_highly_informative),
            "top10_dims": top_dims[:10].tolist(),
            "top10_scores": dim_scores[top_dims[:10]].tolist(),
        }

        print(f"\n  {name}:")
        print(f"    Dims with |r| > 0.3: {n_informative}")
        print(f"    Dims with |r| > 0.5: {n_highly_informative}")
        print(f"    Top 10 dims: {top_dims[:10].tolist()}")
        print(f"    Top 10 scores: {dim_scores[top_dims[:10]].round(3).tolist()}")

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DDPM vs MSE Encoder Comparison")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANALYSIS A: DDPM vs MSE ENCODER REPRESENTATION COMPARISON")
    print("=" * 70)

    # ── Load encoders ──
    print("\nLoading encoders...")
    encoders = {}

    print("  Loading DDPM encoder...")
    encoders["DDPM"] = load_ddpm_encoder(device)

    print("  Loading MSE encoder...")
    encoders["MSE"] = load_mse_encoder(device)

    # Try MSE 15ep variant
    try:
        encoders["MSE_15ep"] = load_mse_encoder(
            device, "models/backfill/gru_encoder_mse_15ep/encoder.pt"
        )
        print("  Loaded MSE_15ep encoder")
    except Exception as e:
        print(f"  MSE_15ep not found: {e}")

    print("  Creating Random encoder...")
    encoders["Random"] = load_random_encoder(device)

    # ── Load data ──
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Test set: 4540:5822
    test_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_history, all_future = [], []
    for batch in test_loader:
        all_history.append(batch["history"])
        all_future.append(batch["future"])
    test_history = torch.cat(all_history, dim=0).to(device)
    test_future_t = torch.cat(all_future, dim=0)
    test_future_np = test_future_t.numpy()
    N_test = test_history.shape[0]
    print(f"  Test windows: {N_test}")

    # Training set: 0:4040
    train_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    train_history_list, train_future_list = [], []
    for batch in train_loader:
        train_history_list.append(batch["history"])
        train_future_list.append(batch["future"])
    train_history = torch.cat(train_history_list, dim=0).to(device)
    train_future_t = torch.cat(train_future_list, dim=0)
    train_future_np = train_future_t.numpy()
    N_train = train_history.shape[0]
    print(f"  Train windows: {N_train}")

    # ── Compute condition vectors ──
    print("\nComputing condition vectors...")
    test_cond = {}
    train_cond = {}
    for name, enc in encoders.items():
        print(f"  {name}...")
        test_cond[name] = compute_condition_vectors(enc, test_history)
        train_cond[name] = compute_condition_vectors(enc, train_history)

    # ── Compute features for probes ──
    print("\nComputing features...")
    test_hist_np = test_history.cpu().numpy()
    train_hist_np = train_history.cpu().numpy()
    test_features = compute_features(test_hist_np, test_future_np)
    train_features = compute_features(train_hist_np, train_future_np)

    # ── Regime labels ──
    hist_iv = (test_hist_np + 1.0) / 2.0
    regime_labels = classify_regime(hist_iv)
    print(f"  Regime split: calm={np.sum(regime_labels==0)}, "
          f"normal={np.sum(regime_labels==1)}, turb={np.sum(regime_labels==2)}")

    # ══════════════════════════════════════════════════════════════════
    # Run all investigations
    # ══════════════════════════════════════════════════════════════════

    all_results = {}

    all_results["eigenspectrum"] = eigenspectrum_analysis(test_cond, OUTPUT_DIR)
    all_results["linear_probe"] = linear_probe_analysis(
        test_cond, train_cond, test_features, train_features, OUTPUT_DIR,
    )
    all_results["percell_probe"] = percell_probe_analysis(
        test_cond, train_cond, test_future_np, train_future_np, OUTPUT_DIR,
    )
    all_results["condition_stats"] = condition_stats_analysis(test_cond, OUTPUT_DIR)
    all_results["cka"] = cka_analysis(test_cond, OUTPUT_DIR)
    all_results["regime"] = regime_discrimination_analysis(test_cond, regime_labels, OUTPUT_DIR)
    all_results["info_sufficiency"] = information_sufficiency_analysis(
        test_cond, train_cond, test_future_np, train_future_np, OUTPUT_DIR,
    )
    all_results["dim_informativeness"] = dimension_informativeness(
        test_cond, train_cond, test_features, train_features, OUTPUT_DIR,
    )

    # ══════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SYNTHESIS: WHY DOES THE DDPM ENCODER WORK?")
    print("=" * 70)

    ddpm_eff_rank = all_results["eigenspectrum"]["DDPM"]["participation_ratio"]
    mse_eff_rank = all_results["eigenspectrum"]["MSE"]["participation_ratio"]
    print(f"\n  1. Effective rank: DDPM={ddpm_eff_rank:.2f}, MSE={mse_eff_rank:.2f}")
    if ddpm_eff_rank < mse_eff_rank:
        print(f"     -> DDPM is MORE compressed ({ddpm_eff_rank:.1f} < {mse_eff_rank:.1f})")
    else:
        print(f"     -> DDPM is LESS compressed ({ddpm_eff_rank:.1f} > {mse_eff_rank:.1f})")

    ddpm_top5 = all_results["eigenspectrum"]["DDPM"]["top5_var_pct"]
    mse_top5 = all_results["eigenspectrum"]["MSE"]["top5_var_pct"]
    print(f"\n  2. Top-5 eigenvalue variance: DDPM={ddpm_top5:.1f}%, MSE={mse_top5:.1f}%")
    if ddpm_top5 > mse_top5:
        print("     -> DDPM has steeper eigenvalue decay (more hierarchical)")
    else:
        print("     -> MSE has steeper decay (unexpected)")

    ddpm_cell_r2 = all_results["percell_probe"]["DDPM"]["r2_mean"]
    mse_cell_r2 = all_results["percell_probe"]["MSE"]["r2_mean"]
    print(f"\n  3. Per-cell next-frame R^2: DDPM={ddpm_cell_r2:.4f}, MSE={mse_cell_r2:.4f}")
    if mse_cell_r2 > ddpm_cell_r2:
        print("     -> MSE encoder gives BETTER per-cell prediction")
        print("     -> CONFIRMS: MSE condition is 'too informative' — decoder doesn't need noise")
    else:
        print("     -> Unexpected: DDPM gives better per-cell prediction")

    ddpm_mean_r2 = all_results["info_sufficiency"]["DDPM"]["mean_traj"]["r2_mean"]
    mse_mean_r2 = all_results["info_sufficiency"]["MSE"]["mean_traj"]["r2_mean"]
    ddpm_std_r2 = all_results["info_sufficiency"]["DDPM"]["std_traj"]["r2_mean"]
    mse_std_r2 = all_results["info_sufficiency"]["MSE"]["std_traj"]["r2_mean"]
    print(f"\n  4. Mean trajectory R^2: DDPM={ddpm_mean_r2:.4f}, MSE={mse_mean_r2:.4f}")
    print(f"     Std trajectory R^2:  DDPM={ddpm_std_r2:.4f}, MSE={mse_std_r2:.4f}")

    if "DDPM" in all_results["cka"] and "MSE" in all_results["cka"]:
        names = all_results["cka"]["names"]
        cka_mat = all_results["cka"]["cka_matrix"]
        ddpm_idx = names.index("DDPM")
        mse_idx = names.index("MSE")
        cka_val = cka_mat[ddpm_idx][mse_idx]
        print(f"\n  5. CKA(DDPM, MSE) = {cka_val:.4f}")
        if cka_val > 0.8:
            print("     -> Representations are HIGHLY similar (>0.8)")
        elif cka_val > 0.5:
            print("     -> Representations are MODERATELY similar (0.5-0.8)")
        else:
            print("     -> Representations are QUITE DIFFERENT (<0.5)")

    ddpm_ari = all_results["regime"]["DDPM"]["ari"]
    mse_ari = all_results["regime"]["MSE"]["ari"]
    print(f"\n  6. Regime ARI: DDPM={ddpm_ari:.4f}, MSE={mse_ari:.4f}")

    # Key conclusion
    print("\n" + "-" * 70)
    print("KEY CONCLUSION:")
    if mse_cell_r2 > ddpm_cell_r2 + 0.01:
        print("  The MSE encoder encodes TOO MUCH per-cell detail into the condition vector.")
        print("  This gives the decoder a near-exact prediction target, making noise irrelevant.")
        print("  The DDPM encoder provides a COARSER summary (regime, level, shape) that")
        print("  requires the decoder to use noise for per-cell prediction uncertainty.")
    elif abs(mse_cell_r2 - ddpm_cell_r2) < 0.01:
        print("  Per-cell prediction accuracy is SIMILAR — the difference must lie in")
        print("  the GEOMETRY of the representation space, not raw information content.")
        print("  Check eigenspectrum and inter-dim correlation differences.")
    else:
        print("  DDPM encoder has HIGHER per-cell R^2 — the conventional 'too informative'")
        print("  hypothesis is WRONG. Check other structural differences.")
    print("-" * 70)

    # ── Save all results ──
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = OUTPUT_DIR / "encoder_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save condition vectors for further analysis
    np.savez(
        OUTPUT_DIR / "condition_vectors.npz",
        **{f"{name}_test": v for name, v in test_cond.items()},
        **{f"{name}_train": v for name, v in train_cond.items()},
        regime_labels=regime_labels,
    )
    print(f"Condition vectors saved to {OUTPUT_DIR / 'condition_vectors.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
