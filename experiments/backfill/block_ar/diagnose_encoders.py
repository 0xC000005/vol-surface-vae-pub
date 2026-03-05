"""Diagnose why DDPM encoder uniquely enables sample diversity.

Compare four frozen encoders: DDPM, MSE 100ep, MSE 15ep+dropout, random.
Tests: condition vector stats, regime separability, next-frame predictability,
noise-driven vs condition-driven variance.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_encoders.py --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR, SinglePassConfig, denormalize_iv, normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

OUTPUT_DIR = "results/block_ar/encoder_diagnosis"


def load_encoder_from_afcrps(model_path, device, no_ema=True):
    """Load encoder from a trained afCRPS model checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = SinglePassConfig(**cfg)
    model = SinglePassBlockAR(cfg)
    key = "ema_state_dict" if not no_ema and "ema_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.eval().to(device)
    return model.encoder, model


def load_encoder_from_pretrain(encoder_path, device):
    """Load encoder from pretrain_encoder.py checkpoint."""
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
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
    """Create encoder with random init (matching 90i)."""
    torch.manual_seed(seed)
    enc_config = EncoderConfig(
        input_dim=25, gru_hidden_dim=64, bottleneck_dim=128, dropout=0.1,
    )
    encoder = GRUEncoder(enc_config)
    encoder.eval().to(device)
    return encoder


def classify_regime(hist_np):
    """Classify windows as turbulent (True) or calm (False)."""
    mean_iv = hist_np.mean(axis=(-1, -2))  # (N, T)
    daily_chg = np.diff(mean_iv, axis=1)
    vov = daily_chg.std(axis=1)
    return vov > np.median(vov)


def effective_rank(cond_vecs):
    """Number of singular values above 1% of max."""
    U, S, Vh = np.linalg.svd(cond_vecs - cond_vecs.mean(axis=0), full_matrices=False)
    threshold = 0.01 * S[0]
    return int((S > threshold).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load encoders ──
    print("Loading encoders...")
    encoders = {}

    # DDPM encoder (from 90d)
    enc_ddpm, model_90d = load_encoder_from_afcrps(
        "models/backfill/afcrps_90d/best_model.pt", device
    )
    encoders["DDPM"] = enc_ddpm

    # MSE 100ep encoder
    encoders["MSE_100ep"] = load_encoder_from_pretrain(
        "models/backfill/gru_encoder_mse/encoder.pt", device
    )

    # MSE 15ep + dropout 0.3
    encoders["MSE_15ep_drop"] = load_encoder_from_pretrain(
        "models/backfill/gru_encoder_mse_15ep/encoder.pt", device
    )

    # Random encoder
    encoders["Random"] = load_random_encoder(device)

    # Contrastive encoder (MSE + SupCon)
    try:
        encoders["Contrastive"] = load_encoder_from_pretrain(
            "models/backfill/gru_encoder_contrastive/encoder.pt", device
        )
    except FileNotFoundError:
        print("  Contrastive encoder not found, skipping")

    # Noise-conditioned encoder
    try:
        encoders["NoiseCond"] = load_encoder_from_pretrain(
            "models/backfill/gru_encoder_noise_cond/encoder.pt", device
        )
    except FileNotFoundError:
        print("  NoiseCond encoder not found, skipping")

    # ── Load test data ──
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Collect test windows
    all_history, all_future = [], []
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        all_history.append(batch["history"])
        all_future.append(batch["future"])
    history_t = torch.cat(all_history, dim=0).to(device)  # (N, 30, 5, 5)
    future_t = torch.cat(all_future, dim=0).to(device)
    N = history_t.shape[0]
    print(f"Test windows: {N}")

    # Regime labels
    hist_np = denormalize_iv(history_t).cpu().numpy()
    regime = classify_regime(hist_np)
    n_turb = regime.sum()
    n_calm = (~regime).sum()
    print(f"Regime split: {n_turb} turb, {n_calm} calm\n")

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Condition vector statistics
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("1. CONDITION VECTOR STATISTICS")
    print("=" * 70)

    cond_vecs = {}
    for name, enc in encoders.items():
        with torch.no_grad():
            c = enc(history_t).cpu().numpy()  # (N, 128)
        cond_vecs[name] = c

    print(f"\n{'Encoder':<18s}  {'Mean':>8s}  {'Std':>8s}  {'EffRank':>8s}  {'Max|c|':>8s}  {'Sparsity':>8s}")
    print("-" * 70)
    for name, c in cond_vecs.items():
        mean_abs = np.abs(c).mean()
        std_all = c.std()
        erank = effective_rank(c)
        max_abs = np.abs(c).max()
        # Sparsity: fraction of dims with std < 10% of max dim std
        dim_stds = c.std(axis=0)
        sparsity = (dim_stds < 0.1 * dim_stds.max()).mean()
        print(f"{name:<18s}  {mean_abs:8.4f}  {std_all:8.4f}  {erank:8d}  {max_abs:8.4f}  {sparsity:8.3f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Calm vs turb separability
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("2. CALM vs TURB SEPARABILITY")
    print("=" * 70)

    print(f"\n{'Encoder':<18s}  {'L2 dist':>8s}  {'Norm L2':>8s}  {'Cosine':>8s}  {'t-stat max':>10s}")
    print("-" * 70)
    for name, c in cond_vecs.items():
        calm_mean = c[~regime].mean(axis=0)
        turb_mean = c[regime].mean(axis=0)
        l2 = np.linalg.norm(turb_mean - calm_mean)
        overall_std = c.std()
        norm_l2 = l2 / (overall_std * np.sqrt(128))  # normalized
        # Cosine similarity between centroids
        cos = np.dot(calm_mean, turb_mean) / (np.linalg.norm(calm_mean) * np.linalg.norm(turb_mean) + 1e-8)
        # Per-dim t-statistic (regime discrimination power)
        calm_c, turb_c = c[~regime], c[regime]
        pooled_std = np.sqrt((calm_c.var(0) + turb_c.var(0)) / 2 + 1e-8)
        t_stats = np.abs(turb_c.mean(0) - calm_c.mean(0)) / (pooled_std / np.sqrt(min(n_turb, n_calm)))
        print(f"{name:<18s}  {l2:8.4f}  {norm_l2:8.4f}  {cos:8.4f}  {t_stats.max():10.2f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Condition → next frame predictability
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("3. CONDITION → NEXT FRAME PREDICTABILITY")
    print("=" * 70)
    print("  (Lower MSE = condition encodes more about next frame)")

    # Load training data for fitting the linear probe
    train_ds = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    train_history, train_future = [], []
    for batch in train_loader:
        train_history.append(batch["history"])
        train_future.append(batch["future"])
    train_hist_t = torch.cat(train_history, dim=0).to(device)
    train_fut_t = torch.cat(train_future, dim=0).to(device)
    # Target: first future frame (denormalized)
    train_target = denormalize_iv(train_fut_t[:, 0]).reshape(-1, 25).cpu().numpy()
    test_target = denormalize_iv(future_t[:, 0]).reshape(-1, 25).cpu().numpy()

    print(f"\n{'Encoder':<18s}  {'Train MSE':>10s}  {'Test MSE':>10s}  {'Ratio':>8s}")
    print("-" * 70)

    for name, enc in encoders.items():
        # Get condition vectors for train + test
        with torch.no_grad():
            train_c = enc(train_hist_t).cpu().numpy()
            test_c = cond_vecs[name]

        # Fit linear probe: c → next frame (25 dim)
        # Closed-form OLS: w = (X^T X)^{-1} X^T y
        X = np.column_stack([train_c, np.ones(len(train_c))])  # (N, 129) with bias
        w = np.linalg.lstsq(X, train_target, rcond=None)[0]  # (129, 25)

        train_pred = X @ w
        train_mse = ((train_pred - train_target) ** 2).mean()

        X_test = np.column_stack([test_c, np.ones(len(test_c))])
        test_pred = X_test @ w
        test_mse = ((test_pred - test_target) ** 2).mean()
        ratio = test_mse / (train_mse + 1e-8)

        print(f"{name:<18s}  {train_mse:10.6f}  {test_mse:10.6f}  {ratio:8.3f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Noise-driven vs condition-driven variance
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("4. NOISE-DRIVEN vs CONDITION-DRIVEN VARIANCE (90d model)")
    print("=" * 70)
    print("  Generating 50 samples for 50 windows...")

    # Use first 50 test windows
    n_windows = min(50, N)
    hist_subset = history_t[:n_windows]

    with torch.no_grad():
        samples_90d = model_90d.sample(hist_subset, n_samples=50)  # (50, 50, 30, 5, 5)
    samples_np = samples_90d.cpu().numpy()  # (N_w, S, T, 5, 5)

    # For each cell, compute:
    # - noise_var: mean over windows of var over samples (within-window diversity)
    # - cond_var: var over windows of mean over samples (between-window signal)
    # Collapse time dimension by averaging over all horizons

    sample_means = samples_np.mean(axis=1)  # (N_w, T, 5, 5) — mean traj per window
    sample_vars = samples_np.var(axis=1)    # (N_w, T, 5, 5) — within-window var

    noise_var = sample_vars.mean(axis=(0, 1))   # (5, 5) — avg noise-driven var
    cond_var = sample_means.var(axis=0).mean(axis=0)  # (5, 5) — condition-driven var

    total_var = noise_var + cond_var
    noise_frac = noise_var / (total_var + 1e-10)

    labels_k = ["K=0.70", "K=0.85", "K=1.00", "K=1.15", "K=1.30"]
    labels_t = ["1M", "3M", "6M", "1Y", "2Y"]

    print(f"\nNoise-driven variance fraction (higher = more diversity from z_t):")
    print(f"{'':>8s}", end="")
    for k in labels_k:
        print(f"  {k:>8s}", end="")
    print()
    for i, t in enumerate(labels_t):
        print(f"{t:>8s}", end="")
        for j in range(5):
            print(f"  {noise_frac[i, j]:8.4f}", end="")
        print()
    print(f"\n  Mean noise fraction: {noise_frac.mean():.4f}")
    print(f"  Noise var range: [{noise_var.min():.6f}, {noise_var.max():.6f}]")
    print(f"  Cond  var range: [{cond_var.min():.6f}, {cond_var.max():.6f}]")

    # Also compute for 90f if available
    try:
        _, model_90f = load_encoder_from_afcrps(
            "models/backfill/afcrps_90f/best_model.pt", device
        )
        with torch.no_grad():
            samples_90f = model_90f.sample(hist_subset, n_samples=50)
        s_np = samples_90f.cpu().numpy()

        s_means = s_np.mean(axis=1)
        s_vars = s_np.var(axis=1)
        nv_f = s_vars.mean(axis=(0, 1))
        cv_f = s_means.var(axis=0).mean(axis=0)
        nf_f = nv_f / (nv_f + cv_f + 1e-10)

        print(f"\n90f (MSE encoder) noise-driven variance fraction:")
        print(f"{'':>8s}", end="")
        for k in labels_k:
            print(f"  {k:>8s}", end="")
        print()
        for i, t in enumerate(labels_t):
            print(f"{t:>8s}", end="")
            for j in range(5):
                print(f"  {nf_f[i, j]:8.4f}", end="")
            print()
        print(f"\n  Mean noise fraction: {nf_f.mean():.4f}")

        print(f"\n  90d vs 90f noise fraction: {noise_frac.mean():.4f} vs {nf_f.mean():.4f} "
              f"({'90d higher' if noise_frac.mean() > nf_f.mean() else '90f higher'})")
    except Exception as e:
        print(f"\n  90f model not available: {e}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print("""
Key hypothesis: DDPM encoder has HIGHER next-frame prediction residual (Test 3)
AND comparable regime separability (Test 2). This would mean it provides useful
regime signal without giving away the exact answer — the sweet spot for afCRPS.
""")


if __name__ == "__main__":
    main()
