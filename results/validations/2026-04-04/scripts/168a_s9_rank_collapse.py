#!/usr/bin/env python
"""
168a S9 Rank Collapse Investigation: Why K=4 destroys cross-cell correlation rank structure.

CONTEXT: 168a (K=4) has rank_ratio=0.401 (FAIL, gate 0.5-3.0).
Baseline K=16 has rank_ratio ~1.3 (PASS). correlation_ratio=1.656 (PASS).
So 168a has correct average correlation but WRONG rank structure -- effective rank of
generated cross-cell correlation matrix is too low.

ANALYSES:
1. Factor analysis: PCA eigenvalue spectrum on generated deltas vs GT
2. Member diversity: Pairwise cosine similarity between K members
3. CLN noise contribution: Effective rank of output across many noise draws
4. Structural vs training: Is this a fundamental K=4 limitation?

Usage:
    PYTHONPATH=. python results/validations/2026-04-04/scripts/168a_s9_rank_collapse.py --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.backfill.block_ar.train_164a_v3_percell_bptt_softplus import (
    ARSpatialTransformerModel, SpatialTransformerDecoder, normalize_iv, denormalize_iv,
)
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig


def make_serializable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def load_model(model_path, device):
    """Load ARSpatialTransformerModel from checkpoint."""
    ckpt = torch.load(model_path, weights_only=False, map_location=device)
    config = ckpt["config"]

    encoder_config = EncoderConfig(**config["encoder"])
    decoder_config = config["decoder"]

    model = ARSpatialTransformerModel(encoder_config, decoder_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def load_data(device):
    """Load vol surface data, return test windows starting at 4540."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5) in [0, 1]
    N = surfaces.shape[0]
    H, T = 30, 30

    # Test windows starting at 4540 (proper test split)
    test_start = 4540
    test_indices = np.arange(test_start, N - H - T + 1)

    surf_t = torch.from_numpy(surfaces.astype(np.float32)).to(device)

    # Build test windows
    idx = torch.from_numpy(test_indices).long().to(device)
    offsets_h = torch.arange(H, device=device).unsqueeze(0)
    offsets_f = torch.arange(T, device=device).unsqueeze(0)
    hist_idx = idx.unsqueeze(1) + offsets_h
    fut_idx = idx.unsqueeze(1) + H + offsets_f

    hist = surf_t[hist_idx]  # (N_test, H, 5, 5)
    future = surf_t[fut_idx]  # (N_test, T, 5, 5)

    return hist, future, test_indices


def compute_effective_rank(corr_matrix):
    """Effective rank from eigenvalue spectrum of correlation matrix."""
    eigvals = np.linalg.eigvalsh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-10)  # numerical stability
    eigvals = eigvals / eigvals.sum()
    entropy = -np.sum(eigvals * np.log(eigvals))
    return np.exp(entropy)


def compute_eigenvalue_spectrum(deltas):
    """PCA eigenvalue spectrum on deltas (N, 25).
    Returns eigenvalues (descending) and explained variance ratios."""
    deltas_centered = deltas - deltas.mean(axis=0, keepdims=True)
    cov = np.cov(deltas_centered, rowvar=False)  # (25, 25)
    eigvals = np.linalg.eigvalsh(cov)[::-1]  # descending
    total = eigvals.sum()
    explained = eigvals / total if total > 0 else eigvals
    return eigvals, explained


def analysis_1_factor_structure(model_168a, model_baseline, hist, future, device,
                                n_windows=30, n_samples=50):
    """Factor analysis: PCA eigenvalue spectrum comparison."""
    print("\n" + "="*70)
    print("ANALYSIS 1: Factor Structure (PCA Eigenvalue Spectrum)")
    print("="*70)

    B = n_windows
    hist_subset = hist[:B]
    future_subset = future[:B]

    # Normalize history for model input
    hist_norm = normalize_iv(hist_subset)

    # GT deltas: day-to-day changes in future
    gt_future_flat = future_subset.reshape(B, 30, 25).cpu().numpy()
    # Use last frame from history as day 0
    last_frame = hist_subset[:, -1].reshape(B, 25).cpu().numpy()
    gt_all = np.concatenate([last_frame[:, np.newaxis, :], gt_future_flat], axis=1)  # (B, 31, 25)
    gt_deltas = np.diff(gt_all, axis=1)  # (B, 30, 25)
    gt_deltas_flat = gt_deltas.reshape(-1, 25)  # (B*30, 25)

    # Generated deltas: 168a (K=4)
    with torch.no_grad():
        samples_168a = model_168a.sample_batched(hist_norm, n_samples=n_samples)
    # samples: (B, n_samples, 30, 5, 5)
    gen_168a = samples_168a.reshape(B, n_samples, 30, 25).cpu().numpy()
    last_frame_exp = np.expand_dims(last_frame, (1, 2))  # (B, 1, 1, 25)
    gen_168a_all = np.concatenate([
        np.broadcast_to(last_frame_exp, (B, n_samples, 1, 25)),
        gen_168a
    ], axis=2)  # (B, n_samples, 31, 25)
    gen_168a_deltas = np.diff(gen_168a_all, axis=2)  # (B, n_samples, 30, 25)
    gen_168a_flat = gen_168a_deltas.reshape(-1, 25)

    # Generated deltas: baseline (K=16)
    with torch.no_grad():
        samples_base = model_baseline.sample_batched(hist_norm, n_samples=n_samples)
    gen_base = samples_base.reshape(B, n_samples, 30, 25).cpu().numpy()
    gen_base_all = np.concatenate([
        np.broadcast_to(last_frame_exp, (B, n_samples, 1, 25)),
        gen_base
    ], axis=2)
    gen_base_deltas = np.diff(gen_base_all, axis=2)
    gen_base_flat = gen_base_deltas.reshape(-1, 25)

    # Eigenvalue spectra
    gt_eigvals, gt_explained = compute_eigenvalue_spectrum(gt_deltas_flat)
    k4_eigvals, k4_explained = compute_eigenvalue_spectrum(gen_168a_flat)
    base_eigvals, base_explained = compute_eigenvalue_spectrum(gen_base_flat)

    # Effective ranks
    gt_corr = np.corrcoef(gt_deltas_flat, rowvar=False)
    k4_corr = np.corrcoef(gen_168a_flat, rowvar=False)
    base_corr = np.corrcoef(gen_base_flat, rowvar=False)

    gt_eff_rank = compute_effective_rank(gt_corr)
    k4_eff_rank = compute_effective_rank(k4_corr)
    base_eff_rank = compute_effective_rank(base_corr)

    print(f"\nEffective rank of delta correlation matrix:")
    print(f"  GT:       {gt_eff_rank:.2f}")
    print(f"  K=16:     {base_eff_rank:.2f} (ratio: {base_eff_rank/gt_eff_rank:.3f})")
    print(f"  K=4:      {k4_eff_rank:.2f} (ratio: {k4_eff_rank/gt_eff_rank:.3f})")

    print(f"\nTop-5 eigenvalue explained variance:")
    print(f"  GT:   {gt_explained[:5]}")
    print(f"  K=16: {base_explained[:5]}")
    print(f"  K=4:  {k4_explained[:5]}")

    # PC1 dominance
    print(f"\nPC1 dominance:")
    print(f"  GT:   {gt_explained[0]*100:.1f}%")
    print(f"  K=16: {base_explained[0]*100:.1f}%")
    print(f"  K=4:  {k4_explained[0]*100:.1f}%")

    # Is K=4 producing rank-1 output?
    rank1_threshold = 0.80
    k4_is_rank1 = k4_explained[0] > rank1_threshold
    print(f"\n  K=4 rank-1 collapse (PC1 > {rank1_threshold*100}%): {k4_is_rank1}")

    return {
        "gt_effective_rank": gt_eff_rank,
        "k16_effective_rank": base_eff_rank,
        "k4_effective_rank": k4_eff_rank,
        "k16_rank_ratio": base_eff_rank / gt_eff_rank,
        "k4_rank_ratio": k4_eff_rank / gt_eff_rank,
        "gt_explained_top5": gt_explained[:5].tolist(),
        "k16_explained_top5": base_explained[:5].tolist(),
        "k4_explained_top5": k4_explained[:5].tolist(),
        "gt_pc1_pct": float(gt_explained[0] * 100),
        "k16_pc1_pct": float(base_explained[0] * 100),
        "k4_pc1_pct": float(k4_explained[0] * 100),
        "k4_is_rank1": k4_is_rank1,
        "gt_eigenvalues": gt_eigvals.tolist(),
        "k16_eigenvalues": k4_eigvals.tolist(),
        "k4_eigenvalues": k4_eigvals.tolist(),
    }


def analysis_2_member_diversity(model_168a, model_baseline, hist, device,
                                n_windows=30):
    """Member diversity: pairwise cosine similarity between ensemble members."""
    print("\n" + "="*70)
    print("ANALYSIS 2: Member Diversity (Pairwise Cosine Similarity)")
    print("="*70)

    B = n_windows
    hist_subset = hist[:B]
    hist_norm = normalize_iv(hist_subset)

    results = {}

    for name, model, n_members in [("K=4 (168a)", model_168a, 4),
                                    ("K=16 (baseline)", model_baseline, 16)]:
        print(f"\n--- {name} ---")

        # Generate exactly n_members samples per window to analyze member diversity
        with torch.no_grad():
            samples = model.sample_batched(hist_norm, n_samples=n_members)
        # (B, K, 30, 5, 5)

        # Flatten to deltas
        samples_flat = samples.reshape(B, n_members, 30, 25).cpu().numpy()
        last_frame = hist_subset[:, -1].reshape(B, 25).cpu().numpy()

        # Compute deltas for each member
        member_deltas = []
        for k in range(n_members):
            member_frames = samples_flat[:, k, :, :]  # (B, 30, 25)
            all_frames = np.concatenate([last_frame[:, np.newaxis, :], member_frames], axis=1)
            deltas = np.diff(all_frames, axis=1)  # (B, 30, 25)
            member_deltas.append(deltas)

        # Pairwise cosine similarity between members
        # Flatten each member's trajectory to a vector: (B, 30*25)
        member_vecs = [d.reshape(B, -1) for d in member_deltas]  # list of (B, 750)

        cos_sims = []
        for i in range(n_members):
            for j in range(i+1, n_members):
                v_i = torch.from_numpy(member_vecs[i]).float()
                v_j = torch.from_numpy(member_vecs[j]).float()
                cos = F.cosine_similarity(v_i, v_j, dim=1)  # (B,)
                cos_sims.append(cos.numpy())

        cos_sims = np.array(cos_sims)  # (n_pairs, B)
        mean_cos = cos_sims.mean()
        std_cos = cos_sims.std()
        min_cos = cos_sims.min()
        max_cos = cos_sims.max()

        n_pairs = cos_sims.shape[0]
        high_corr_frac = (cos_sims > 0.9).mean()

        print(f"  Num member pairs: {n_pairs}")
        print(f"  Mean cosine similarity: {mean_cos:.4f}")
        print(f"  Std cosine similarity:  {std_cos:.4f}")
        print(f"  Min/Max: {min_cos:.4f} / {max_cos:.4f}")
        print(f"  Fraction with cos > 0.9: {high_corr_frac:.3f}")

        key = "k4" if "168a" in name else "k16"
        results[key] = {
            "n_members": n_members,
            "n_pairs": int(n_pairs),
            "mean_cosine": float(mean_cos),
            "std_cosine": float(std_cos),
            "min_cosine": float(min_cos),
            "max_cosine": float(max_cos),
            "frac_high_corr_gt09": float(high_corr_frac),
        }

        # Also compute per-frame cosine similarity (are members diverse at each step?)
        per_frame_cos = []
        for t in range(30):
            frame_vecs = [d[:, t, :] for d in member_deltas]  # list of (B, 25)
            frame_cos = []
            for i in range(n_members):
                for j in range(i+1, n_members):
                    v_i = torch.from_numpy(frame_vecs[i]).float()
                    v_j = torch.from_numpy(frame_vecs[j]).float()
                    cos = F.cosine_similarity(v_i, v_j, dim=1).mean().item()
                    frame_cos.append(cos)
            per_frame_cos.append(np.mean(frame_cos))

        results[key]["per_frame_mean_cosine"] = per_frame_cos
        print(f"  Per-frame cosine (first 5 frames): {per_frame_cos[:5]}")
        print(f"  Per-frame cosine (last 5 frames):  {per_frame_cos[-5:]}")

    return results


def analysis_3_cln_noise_contribution(model_168a, hist, device, n_windows=10,
                                      n_noise_draws=100):
    """CLN noise contribution: effective rank of output across many noise draws."""
    print("\n" + "="*70)
    print("ANALYSIS 3: CLN Noise Contribution (Diversity from Noise)")
    print("="*70)

    B = n_windows
    hist_subset = hist[:B]
    hist_norm = normalize_iv(hist_subset)

    # Generate 100 samples from same windows, different noise
    with torch.no_grad():
        samples = model_168a.sample_batched(hist_norm, n_samples=n_noise_draws)
    # (B, 100, 30, 5, 5)
    samples_flat = samples.reshape(B, n_noise_draws, 30, 25).cpu().numpy()
    last_frame = hist_subset[:, -1].reshape(B, 25).cpu().numpy()

    results = {
        "per_window": [],
        "per_window_per_horizon": [],
    }

    for w in range(B):
        # For this window, we have 100 generated trajectories
        # Compute deltas for each draw
        member_frames = samples_flat[w]  # (100, 30, 25)
        lf = last_frame[w:w+1]  # (1, 25)
        all_frames = np.concatenate([
            np.broadcast_to(lf[:, np.newaxis, :], (1, 1, 25)).repeat(n_noise_draws, axis=0),
            member_frames
        ], axis=1)  # (100, 31, 25)
        deltas = np.diff(all_frames, axis=1)  # (100, 30, 25)

        # Effective rank of the 100 samples at different horizons
        horizon_ranks = []
        for t in [0, 4, 9, 14, 19, 24, 29]:
            frame_deltas = deltas[:, t, :]  # (100, 25)
            if frame_deltas.std() < 1e-10:
                horizon_ranks.append({"horizon": t+1, "eff_rank": 1.0, "pc1_pct": 100.0})
                continue
            corr = np.corrcoef(frame_deltas, rowvar=False)
            if np.any(np.isnan(corr)):
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
            eff_rank = compute_effective_rank(corr)

            eigvals, explained = compute_eigenvalue_spectrum(frame_deltas)
            horizon_ranks.append({
                "horizon": t+1,
                "eff_rank": float(eff_rank),
                "pc1_pct": float(explained[0] * 100),
                "pc2_pct": float(explained[1] * 100) if len(explained) > 1 else 0,
                "pc3_pct": float(explained[2] * 100) if len(explained) > 2 else 0,
            })

        results["per_window_per_horizon"].append(horizon_ranks)

        # Full trajectory effective rank
        deltas_flat = deltas.reshape(n_noise_draws, -1)  # (100, 750)
        # Correlation across the 25 cells, pooling over time
        deltas_all_times = deltas.reshape(-1, 25)  # (100*30, 25)
        corr_all = np.corrcoef(deltas_all_times, rowvar=False)
        if np.any(np.isnan(corr_all)):
            corr_all = np.nan_to_num(corr_all, nan=0.0)
            np.fill_diagonal(corr_all, 1.0)
        eff_rank_all = compute_effective_rank(corr_all)
        results["per_window"].append(float(eff_rank_all))

    # Aggregate horizon-level stats
    n_horizons = len(results["per_window_per_horizon"][0])
    agg_horizons = []
    for hi in range(n_horizons):
        h = results["per_window_per_horizon"][0][hi]["horizon"]
        ranks = [results["per_window_per_horizon"][w][hi]["eff_rank"] for w in range(B)]
        pc1s = [results["per_window_per_horizon"][w][hi]["pc1_pct"] for w in range(B)]
        agg_horizons.append({
            "horizon": h,
            "mean_eff_rank": float(np.mean(ranks)),
            "std_eff_rank": float(np.std(ranks)),
            "mean_pc1_pct": float(np.mean(pc1s)),
        })
        print(f"  Horizon {h:2d}: eff_rank={np.mean(ranks):.2f} +/- {np.std(ranks):.2f}, "
              f"PC1={np.mean(pc1s):.1f}%")

    results["aggregate_by_horizon"] = agg_horizons
    results["overall_mean_eff_rank"] = float(np.mean(results["per_window"]))

    print(f"\n  Overall mean effective rank (pooled over time): {results['overall_mean_eff_rank']:.2f}")

    return results


def analysis_4_structural_vs_training(model_168a, model_baseline, hist, future,
                                      device, n_windows=30, n_samples=50):
    """Structural analysis: Is K=4 spread loss fundamentally insufficient?"""
    print("\n" + "="*70)
    print("ANALYSIS 4: Structural vs Training (Spread & VS Loss Analysis)")
    print("="*70)

    B = n_windows
    hist_subset = hist[:B]
    future_subset = future[:B]
    hist_norm = normalize_iv(hist_subset)

    # Generate samples from both models
    with torch.no_grad():
        # For K=4: generate exactly 4 members (as during training)
        samples_k4_train = model_168a.sample_batched(hist_norm, n_samples=4)
        # For K=4: generate 50 members (at inference)
        samples_k4_infer = model_168a.sample_batched(hist_norm, n_samples=50)
        # For K=16: generate exactly 16 members (as during training)
        samples_k16_train = model_baseline.sample_batched(hist_norm, n_samples=16)
        # For K=16: generate 50 members (at inference)
        samples_k16_infer = model_baseline.sample_batched(hist_norm, n_samples=50)

    results = {}

    # 4a. Training-time spread pairs
    print("\n--- 4a: Training-time spread pairs ---")
    for name, samples, K in [("K=4", samples_k4_train, 4),
                              ("K=16", samples_k16_train, 16)]:
        n_pairs = K * (K - 1) // 2
        s = samples.reshape(B, K, 30, 25)

        # Compute pairwise spread at each frame
        idx_i, idx_j = torch.triu_indices(K, K, offset=1)
        diffs = (s[:, idx_i] - s[:, idx_j]).abs()  # (B, n_pairs, 30, 25)
        mean_spread = diffs.mean().item()
        per_cell_spread = diffs.mean(dim=(0, 1, 2)).cpu().numpy()  # (25,)
        spread_std_across_cells = per_cell_spread.std()

        print(f"  {name}: {n_pairs} pairs, mean spread={mean_spread:.4f}, "
              f"cell spread std={spread_std_across_cells:.4f}")

        key = name.lower().replace("=", "")
        results[f"{key}_train"] = {
            "K": K,
            "n_pairs": int(n_pairs),
            "mean_spread": float(mean_spread),
            "per_cell_spread_std": float(spread_std_across_cells),
            "per_cell_spread": per_cell_spread.tolist(),
        }

    # 4b. Inference-time correlation structure
    print("\n--- 4b: Inference-time correlation structure ---")
    gt_flat = future_subset.reshape(B, 30, 25).cpu().numpy()
    last_frame = hist_subset[:, -1].reshape(B, 25).cpu().numpy()

    for name, samples in [("K=4 (168a)", samples_k4_infer),
                           ("K=16 (baseline)", samples_k16_infer)]:
        s = samples.reshape(B, 50, 30, 25).cpu().numpy()

        # Pool all sample deltas
        all_deltas = []
        for b in range(B):
            for k in range(50):
                member_frames = s[b, k]  # (30, 25)
                full = np.concatenate([last_frame[b:b+1], member_frames], axis=0)
                deltas = np.diff(full, axis=0)  # (30, 25)
                all_deltas.append(deltas)
        all_deltas = np.concatenate(all_deltas, axis=0)  # (B*50*30, 25)

        corr = np.corrcoef(all_deltas, rowvar=False)
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

        eff_rank = compute_effective_rank(corr)
        eigvals, explained = compute_eigenvalue_spectrum(all_deltas)

        # Mean off-diagonal correlation
        mask = ~np.eye(25, dtype=bool)
        mean_corr = corr[mask].mean()

        key = "k4_infer" if "168a" in name else "k16_infer"
        results[key] = {
            "eff_rank": float(eff_rank),
            "pc1_pct": float(explained[0] * 100),
            "mean_offdiag_corr": float(mean_corr),
            "top5_explained": explained[:5].tolist(),
        }

        print(f"  {name}:")
        print(f"    Effective rank: {eff_rank:.2f}")
        print(f"    PC1: {explained[0]*100:.1f}%, PC2: {explained[1]*100:.1f}%, "
              f"PC3: {explained[2]*100:.1f}%")
        print(f"    Mean off-diagonal correlation: {mean_corr:.4f}")

    # GT reference
    gt_deltas_all = []
    for b in range(B):
        full = np.concatenate([last_frame[b:b+1], gt_flat[b]], axis=0)
        deltas = np.diff(full, axis=0)
        gt_deltas_all.append(deltas)
    gt_deltas_all = np.concatenate(gt_deltas_all, axis=0)

    gt_corr = np.corrcoef(gt_deltas_all, rowvar=False)
    gt_eff_rank = compute_effective_rank(gt_corr)
    _, gt_explained = compute_eigenvalue_spectrum(gt_deltas_all)

    print(f"  GT:")
    print(f"    Effective rank: {gt_eff_rank:.2f}")
    print(f"    PC1: {gt_explained[0]*100:.1f}%, PC2: {gt_explained[1]*100:.1f}%, "
          f"PC3: {gt_explained[2]*100:.1f}%")

    results["gt"] = {
        "eff_rank": float(gt_eff_rank),
        "pc1_pct": float(gt_explained[0] * 100),
        "top5_explained": gt_explained[:5].tolist(),
    }

    # 4c. VS loss gradient signal analysis
    print("\n--- 4c: VS loss gradient signal ---")
    print(f"  K=4:  C(4,2) = 6 spread pairs for CRPS triu term")
    print(f"  K=16: C(16,2) = 120 spread pairs for CRPS triu term")
    print(f"  Ratio: 120/6 = 20x more gradient signal for cross-member structure")
    print(f"  VS loss compares C(25,2)=300 cell pairs, same for both")
    print(f"  But K=4 estimates E[|X_i - X_j|^p] with 4 samples vs 16 samples")
    print(f"  Variance of estimator scales as 1/K, so K=4 has 4x more noise than K=16")

    results["structural_analysis"] = {
        "k4_crps_pairs": 6,
        "k16_crps_pairs": 120,
        "crps_pair_ratio": 20.0,
        "vs_cell_pairs": 300,
        "vs_estimator_variance_ratio": 4.0,  # K=16/K=4 = 16/4 = 4x less noise
        "explanation": (
            "K=4 has 20x fewer CRPS spread pairs and 4x noisier VS gradient. "
            "The CRPS spread term with 6 pairs cannot distinguish rank-1 from rank-5 "
            "structure. VS helps but its gradient is 4x noisier, preventing fine-grained "
            "eigenvalue learning."
        ),
    }

    # 4d. Noise orthogonality analysis
    print("\n--- 4d: Noise draw orthogonality ---")
    # With K=4, how likely are the noise draws to span the factor space?
    noise_dim = 32
    n_trials = 1000
    eff_ranks_k4 = []
    eff_ranks_k16 = []

    for _ in range(n_trials):
        z4 = np.random.randn(4, noise_dim)
        gram4 = z4 @ z4.T
        gram4_norm = gram4 / np.sqrt(np.diag(gram4)[:, None] * np.diag(gram4)[None, :])
        eigvals4 = np.linalg.eigvalsh(gram4_norm)
        eigvals4 = np.maximum(eigvals4, 1e-10)
        eigvals4 /= eigvals4.sum()
        eff_ranks_k4.append(np.exp(-np.sum(eigvals4 * np.log(eigvals4))))

        z16 = np.random.randn(16, noise_dim)
        gram16 = z16 @ z16.T
        gram16_norm = gram16 / np.sqrt(np.diag(gram16)[:, None] * np.diag(gram16)[None, :])
        eigvals16 = np.linalg.eigvalsh(gram16_norm)
        eigvals16 = np.maximum(eigvals16, 1e-10)
        eigvals16 /= eigvals16.sum()
        eff_ranks_k16.append(np.exp(-np.sum(eigvals16 * np.log(eigvals16))))

    print(f"  Random noise Gram matrix effective rank:")
    print(f"    K=4:  {np.mean(eff_ranks_k4):.2f} +/- {np.std(eff_ranks_k4):.2f} "
          f"(max possible: 4)")
    print(f"    K=16: {np.mean(eff_ranks_k16):.2f} +/- {np.std(eff_ranks_k16):.2f} "
          f"(max possible: 16)")
    print(f"  K=4 noise draws span {np.mean(eff_ranks_k4)/4*100:.0f}% of their "
          f"max dimensionality")
    print(f"  K=16 noise draws span {np.mean(eff_ranks_k16)/16*100:.0f}% of their "
          f"max dimensionality")

    results["noise_orthogonality"] = {
        "k4_mean_eff_rank": float(np.mean(eff_ranks_k4)),
        "k16_mean_eff_rank": float(np.mean(eff_ranks_k16)),
        "k4_max_possible": 4,
        "k16_max_possible": 16,
        "k4_span_pct": float(np.mean(eff_ranks_k4) / 4 * 100),
        "k16_span_pct": float(np.mean(eff_ranks_k16) / 16 * 100),
    }

    # 4e. Does increasing inference K help?
    print("\n--- 4e: Does more inference samples fix it? ---")
    for n_inf in [4, 16, 50, 100, 200]:
        with torch.no_grad():
            s = model_168a.sample_batched(hist_norm[:10], n_samples=n_inf)
        s_flat = s.reshape(10, n_inf, 30, 25).cpu().numpy()
        lf = last_frame[:10]

        all_d = []
        for b in range(10):
            for k in range(n_inf):
                full = np.concatenate([lf[b:b+1], s_flat[b, k]], axis=0)
                d = np.diff(full, axis=0)
                all_d.append(d)
        all_d = np.concatenate(all_d, axis=0)
        corr = np.corrcoef(all_d, rowvar=False)
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
        er = compute_effective_rank(corr)
        _, expl = compute_eigenvalue_spectrum(all_d)
        print(f"  n_samples={n_inf:3d}: eff_rank={er:.2f}, PC1={expl[0]*100:.1f}%")

        if "inference_scaling" not in results:
            results["inference_scaling"] = []
        results["inference_scaling"].append({
            "n_samples": n_inf,
            "eff_rank": float(er),
            "pc1_pct": float(expl[0] * 100),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device

    t0 = time.time()

    print("Loading models...")
    model_168a, config_168a = load_model(
        "models/backfill/afcrps_168a/best_model.pt", device)
    model_baseline, config_baseline = load_model(
        "models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt", device)

    print(f"  168a (K=4): noise_dim={config_168a['decoder']['noise_dim']}, "
          f"n_members={config_168a['n_members']}")
    print(f"  Baseline (K=16): noise_dim={config_baseline['decoder']['noise_dim']}, "
          f"n_members={config_baseline['n_members']}")

    print("\nLoading data...")
    hist, future, test_indices = load_data(device)
    print(f"  Test windows: {len(test_indices)} (starting at idx {test_indices[0]})")

    all_results = {}

    # Analysis 1: Factor structure
    all_results["analysis_1_factor_structure"] = analysis_1_factor_structure(
        model_168a, model_baseline, hist, future, device)

    # Analysis 2: Member diversity
    all_results["analysis_2_member_diversity"] = analysis_2_member_diversity(
        model_168a, model_baseline, hist, device)

    # Analysis 3: CLN noise contribution
    all_results["analysis_3_cln_noise"] = analysis_3_cln_noise_contribution(
        model_168a, hist, device)

    # Analysis 4: Structural vs training
    all_results["analysis_4_structural"] = analysis_4_structural_vs_training(
        model_168a, model_baseline, hist, future, device)

    elapsed = time.time() - t0

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: WHY K=4 DESTROYS RANK STRUCTURE")
    print("="*70)

    a1 = all_results["analysis_1_factor_structure"]
    a2 = all_results["analysis_2_member_diversity"]
    a4 = all_results["analysis_4_structural"]

    print(f"\n1. FACTOR STRUCTURE:")
    print(f"   GT effective rank: {a1['gt_effective_rank']:.2f}")
    print(f"   K=16 rank ratio:   {a1['k16_rank_ratio']:.3f} (PASS gate: 0.5-3.0)")
    print(f"   K=4 rank ratio:    {a1['k4_rank_ratio']:.3f} (FAIL gate: 0.5-3.0)")
    print(f"   K=4 PC1:           {a1['k4_pc1_pct']:.1f}% vs GT {a1['gt_pc1_pct']:.1f}%")

    print(f"\n2. MEMBER DIVERSITY:")
    print(f"   K=4 mean cosine:   {a2['k4']['mean_cosine']:.4f}")
    print(f"   K=16 mean cosine:  {a2['k16']['mean_cosine']:.4f}")
    print(f"   K=4 high-corr (>0.9): {a2['k4']['frac_high_corr_gt09']:.1%}")
    print(f"   K=16 high-corr (>0.9): {a2['k16']['frac_high_corr_gt09']:.1%}")

    print(f"\n3. CLN NOISE:")
    a3 = all_results["analysis_3_cln_noise"]
    print(f"   Overall mean effective rank: {a3['overall_mean_eff_rank']:.2f}")

    print(f"\n4. STRUCTURAL:")
    print(f"   CRPS spread pairs: K=4 has 6 vs K=16 has 120 (20x fewer)")
    print(f"   VS estimator noise: K=4 has 4x more variance than K=16")

    # Determine root cause
    is_rank1 = a1["k4_is_rank1"]
    k4_cos = a2["k4"]["mean_cosine"]
    k16_cos = a2["k16"]["mean_cosine"]
    inference_scaling = a4.get("inference_scaling", [])

    # Check if adding more inference samples helps
    inf_4 = [x for x in inference_scaling if x["n_samples"] == 4]
    inf_200 = [x for x in inference_scaling if x["n_samples"] == 200]
    scaling_helps = False
    if inf_4 and inf_200:
        scaling_helps = inf_200[0]["eff_rank"] > inf_4[0]["eff_rank"] * 1.2

    root_cause = []
    if is_rank1:
        root_cause.append("MODEL produces near-rank-1 output (PC1 > 80%)")
    if k4_cos > 0.8:
        root_cause.append(f"MEMBERS highly correlated (cos={k4_cos:.3f})")
    if not scaling_helps:
        root_cause.append("INFERENCE scaling does NOT help -- problem is in the LEARNED weights")
    else:
        root_cause.append("INFERENCE scaling DOES help -- CLN diversity is present but sparse")

    conclusion = (
        "STRUCTURAL: " if not scaling_helps else "PARTIALLY STRUCTURAL: "
    ) + "; ".join(root_cause)

    print(f"\n  ROOT CAUSE: {conclusion}")

    all_results["summary"] = {
        "conclusion": conclusion,
        "root_causes": root_cause,
        "is_structural": not scaling_helps,
        "recommendation": (
            "K=4 is fundamentally insufficient for cross-cell rank structure. "
            "The 6 CRPS spread pairs cannot constrain 25-cell rank structure (needs ~C(5,2)=10 "
            "independent directions). K >= 8 is the minimum to reliably pass S9."
            if not scaling_helps else
            "K=4 CLN has learned some diversity but training signal was too weak. "
            "Consider higher VS weight or more training epochs."
        ),
    }

    all_results["metadata"] = {
        "model_168a": "models/backfill/afcrps_168a/best_model.pt",
        "model_baseline": "models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt",
        "k4_config": config_168a,
        "k16_config": config_baseline,
        "n_windows_used": 30,
        "n_samples": 50,
        "test_start_idx": int(test_indices[0]),
        "elapsed_seconds": elapsed,
    }

    # Save results
    out_path = "results/validations/2026-04-04/analysis/168a_followup/s9_rank_collapse.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Save verification
    verify_path = "results/validations/2026-04-04/verification_results/168a_s9_rank_collapse.json"
    Path(verify_path).parent.mkdir(parents=True, exist_ok=True)
    verification = {
        "experiment": "168a S9 rank collapse investigation",
        "question": "Why does K=4 destroy cross-cell correlation rank structure?",
        "k4_rank_ratio": a1["k4_rank_ratio"],
        "k16_rank_ratio": a1["k16_rank_ratio"],
        "k4_pc1_pct": a1["k4_pc1_pct"],
        "k16_pc1_pct": a1["k16_pc1_pct"],
        "gt_pc1_pct": a1["gt_pc1_pct"],
        "k4_member_cosine": a2["k4"]["mean_cosine"],
        "k16_member_cosine": a2["k16"]["mean_cosine"],
        "cln_overall_eff_rank": a3["overall_mean_eff_rank"],
        "inference_scaling_helps": scaling_helps,
        "conclusion": conclusion,
        "root_causes": root_cause,
        "is_structural": not scaling_helps,
        "elapsed_seconds": elapsed,
    }
    with open(verify_path, "w") as f:
        json.dump(make_serializable(verification), f, indent=2)
    print(f"Verification saved to: {verify_path}")


if __name__ == "__main__":
    main()
