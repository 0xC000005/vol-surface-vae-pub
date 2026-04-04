#!/usr/bin/env python
"""
167a Noise Effective Rank Evolution Analysis

Tracks noise effective rank, PC1 dominance, L_norm, L_std, and pathway spread
across all intermediate checkpoints (ep10-ep80) of the 167a factorized decoder model.

Evaluates 100 noise samples on 30 test windows (test split starts at index 4540).
For each checkpoint, measures:
  1. PCA on output deltas -> effective rank, PC1 dominance
  2. L_norm (average Frobenius norm across test windows)
  3. L_std (std of L Frobenius norms across windows)
  4. Spread from CLN-only pathway (eps=0, standard noise z through CLN)
  5. Spread from factor-only pathway (z=0, only L@eps noise)

Usage:
    PYTHONPATH=. python results/validations/2026-04-04/scripts/167a_rank_evolution.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/max/Documents/vol-surface-vae-pub")
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from experiments.backfill.block_ar.train_167a_factorized import (
    ARFactorizedTransformerModel,
    normalize_iv,
    denormalize_iv,
    reflecting_boundary,
)


def load_model(checkpoint_path, device):
    """Load 167a model from checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt["config"] if "config" in ckpt else ckpt
    encoder_cfg = EncoderConfig(**cfg["encoder"])
    decoder_cfg = cfg["decoder"]
    n_factors = cfg.get("n_factors", 5)
    model = ARFactorizedTransformerModel(encoder_cfg, decoder_cfg, n_factors=n_factors)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, n_factors


def compute_effective_rank(singular_values):
    """Compute effective rank from singular values using Shannon entropy."""
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 1.0
    p = sv / sv.sum()
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def analyze_checkpoint(model, surf_tensor, test_indices, device,
                       n_noise_samples=100, n_windows=30, n_factors=5):
    """Analyze a single checkpoint for rank evolution metrics."""
    H, T, C = 30, 30, 25
    noise_dim = model.decoder.noise_dim

    # Select windows
    n_win = min(n_windows, len(test_indices))
    selected = test_indices[:n_win]

    all_deltas_full = []       # deltas from full model (base + L@eps)
    all_deltas_cln_only = []   # deltas with eps=0 (CLN noise only)
    all_deltas_factor_only = []  # deltas with z=0 (factor noise only)
    all_L_norms = []
    all_L_matrices = []

    with torch.no_grad():
        for idx in selected:
            # Prepare history
            hist = surf_tensor[idx:idx + H].unsqueeze(0).to(device)  # (1, 30, 5, 5)
            hist_norm = normalize_iv(hist)
            last_frame = denormalize_iv(hist_norm[:, -1]).reshape(1, 25)

            # Encode
            hist_flat = hist_norm.reshape(1, H, C)
            gru_outputs, h_last = model.encoder.gru(hist_flat)
            attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=1)
            h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
            cond = model.encoder.bottleneck(h_pooled)  # (1, 128)

            # Expand for n_noise_samples
            cond_exp = cond.expand(n_noise_samples, -1)
            last_exp = last_frame.expand(n_noise_samples, -1)

            # --- Full model forward (single step for analysis) ---
            z_full = torch.randn(n_noise_samples, noise_dim, device=device)
            delta_base_full, L_full = model.decoder(cond_exp, last_exp, z_full)
            eps_full = torch.randn(n_noise_samples, n_factors, device=device)
            delta_full = delta_base_full + torch.einsum("bcr,br->bc", L_full, eps_full)
            all_deltas_full.append(delta_full.cpu())

            # --- CLN-only: eps = 0 (no factor noise, only CLN noise through z) ---
            z_cln = torch.randn(n_noise_samples, noise_dim, device=device)
            delta_base_cln, L_cln = model.decoder(cond_exp, last_exp, z_cln)
            eps_zero = torch.zeros(n_noise_samples, n_factors, device=device)
            delta_cln_only = delta_base_cln + torch.einsum("bcr,br->bc", L_cln, eps_zero)
            all_deltas_cln_only.append(delta_cln_only.cpu())

            # --- Factor-only: z = 0 (no CLN noise, only L@eps) ---
            z_zero = torch.zeros(n_noise_samples, noise_dim, device=device)
            delta_base_fac, L_fac = model.decoder(cond_exp, last_exp, z_zero)
            eps_fac = torch.randn(n_noise_samples, n_factors, device=device)
            delta_factor_only = delta_base_fac + torch.einsum("bcr,br->bc", L_fac, eps_fac)
            all_deltas_factor_only.append(delta_factor_only.cpu())

            # L statistics (use full model's L)
            L_norm = torch.norm(L_full.mean(dim=0), p="fro").item()  # mean over samples
            all_L_norms.append(L_norm)
            all_L_matrices.append(L_full.mean(dim=0).cpu())  # (25, n_factors)

    # Concatenate all deltas: (n_win * n_noise_samples, 25)
    deltas_full = torch.cat(all_deltas_full, dim=0)
    deltas_cln = torch.cat(all_deltas_cln_only, dim=0)
    deltas_fac = torch.cat(all_deltas_factor_only, dim=0)

    # --- PCA on full deltas ---
    deltas_centered = deltas_full - deltas_full.mean(dim=0, keepdim=True)
    U, S, V = torch.svd(deltas_centered)
    eff_rank = compute_effective_rank(S)
    pc1_dominance = (S[0] ** 2 / (S ** 2).sum()).item()
    # Top-5 variance explained
    total_var = (S ** 2).sum()
    top5_var = (S[:5] ** 2).sum() / total_var if len(S) >= 5 else 1.0
    sv_explained = [(S[i] ** 2 / total_var).item() for i in range(min(10, len(S)))]

    # --- PCA on CLN-only deltas ---
    deltas_cln_c = deltas_cln - deltas_cln.mean(dim=0, keepdim=True)
    _, S_cln, _ = torch.svd(deltas_cln_c)
    eff_rank_cln = compute_effective_rank(S_cln)
    pc1_cln = (S_cln[0] ** 2 / (S_cln ** 2).sum()).item()

    # --- PCA on factor-only deltas ---
    deltas_fac_c = deltas_fac - deltas_fac.mean(dim=0, keepdim=True)
    _, S_fac, _ = torch.svd(deltas_fac_c)
    eff_rank_fac = compute_effective_rank(S_fac)
    pc1_fac = (S_fac[0] ** 2 / (S_fac ** 2).sum()).item()

    # --- Spread metrics ---
    # Compute spread as avg std across cells
    # Reshape back to per-window: (n_win, n_noise_samples, 25)
    def compute_spread(deltas_cat):
        d = deltas_cat.reshape(n_win, n_noise_samples, 25)
        per_cell_std = d.std(dim=1)  # (n_win, 25)
        return per_cell_std.mean().item()

    spread_full = compute_spread(deltas_full)
    spread_cln_only = compute_spread(deltas_cln)
    spread_factor_only = compute_spread(deltas_fac)

    # L statistics
    L_norm_mean = np.mean(all_L_norms)
    L_norm_std = np.std(all_L_norms)

    # L matrix PCA (to check if L itself has rank collapse)
    L_stack = torch.stack(all_L_matrices, dim=0)  # (n_win, 25, n_factors)
    L_mean_across_windows = L_stack.mean(dim=0)  # (25, n_factors)
    _, S_L, _ = torch.svd(L_mean_across_windows)
    L_eff_rank = compute_effective_rank(S_L)
    L_sv_explained = [(S_L[i] ** 2 / (S_L ** 2).sum()).item() for i in range(len(S_L))]

    return {
        # Full model PCA
        "eff_rank": round(eff_rank, 3),
        "pc1_dominance": round(pc1_dominance, 4),
        "top5_var_explained": round(top5_var.item() if isinstance(top5_var, torch.Tensor) else top5_var, 4),
        "sv_explained_top10": [round(x, 4) for x in sv_explained],
        # CLN-only PCA
        "eff_rank_cln_only": round(eff_rank_cln, 3),
        "pc1_cln_only": round(pc1_cln, 4),
        # Factor-only PCA
        "eff_rank_factor_only": round(eff_rank_fac, 3),
        "pc1_factor_only": round(pc1_fac, 4),
        # Spread
        "spread_full": round(spread_full, 6),
        "spread_cln_only": round(spread_cln_only, 6),
        "spread_factor_only": round(spread_factor_only, 6),
        "spread_ratio_factor_vs_cln": round(spread_factor_only / max(spread_cln_only, 1e-10), 4),
        # L statistics
        "L_norm_mean": round(float(L_norm_mean), 6),
        "L_norm_std": round(float(L_norm_std), 6),
        "L_eff_rank": round(L_eff_rank, 3),
        "L_sv_explained": [round(x, 4) for x in L_sv_explained],
        # Config
        "n_windows": n_win,
        "n_noise_samples": n_noise_samples,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    data = np.load("/home/max/Documents/vol-surface-vae-pub/data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    N_total = surfaces.shape[0]
    surf_tensor = torch.from_numpy(surfaces).float()

    # Test split indices (start at 4540, matching v2 test suite)
    TEST_START = 4540
    H, T = 30, 30
    test_indices = list(range(TEST_START, N_total - H - T))
    print(f"Test indices: {len(test_indices)} windows starting at {TEST_START}")

    # Checkpoints to evaluate
    ckpt_dir = Path("/home/max/Documents/vol-surface-vae-pub/models/backfill/afcrps_167a")
    epochs = [10, 20, 30, 40, 50, 60, 70, 80]

    results = {}
    trajectory = []

    for ep in epochs:
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{ep}.pt"
        if not ckpt_path.exists():
            print(f"SKIP: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Epoch {ep}")
        print(f"{'='*60}")

        t0 = time.time()
        model, n_factors = load_model(str(ckpt_path), device)
        metrics = analyze_checkpoint(
            model, surf_tensor, test_indices, device,
            n_noise_samples=100, n_windows=30, n_factors=n_factors,
        )
        elapsed = time.time() - t0

        metrics["epoch"] = ep
        metrics["elapsed_sec"] = round(elapsed, 1)
        results[f"epoch_{ep}"] = metrics

        trajectory.append({
            "epoch": ep,
            "eff_rank": metrics["eff_rank"],
            "pc1_dominance": metrics["pc1_dominance"],
            "eff_rank_cln_only": metrics["eff_rank_cln_only"],
            "eff_rank_factor_only": metrics["eff_rank_factor_only"],
            "spread_full": metrics["spread_full"],
            "spread_cln_only": metrics["spread_cln_only"],
            "spread_factor_only": metrics["spread_factor_only"],
            "spread_ratio_factor_vs_cln": metrics["spread_ratio_factor_vs_cln"],
            "L_norm_mean": metrics["L_norm_mean"],
            "L_norm_std": metrics["L_norm_std"],
            "L_eff_rank": metrics["L_eff_rank"],
        })

        print(f"  eff_rank:       {metrics['eff_rank']:.3f}")
        print(f"  PC1 dominance:  {metrics['pc1_dominance']:.4f}")
        print(f"  eff_rank CLN:   {metrics['eff_rank_cln_only']:.3f}")
        print(f"  eff_rank factor:{metrics['eff_rank_factor_only']:.3f}")
        print(f"  spread full:    {metrics['spread_full']:.6f}")
        print(f"  spread CLN:     {metrics['spread_cln_only']:.6f}")
        print(f"  spread factor:  {metrics['spread_factor_only']:.6f}")
        print(f"  spread ratio:   {metrics['spread_ratio_factor_vs_cln']:.4f}")
        print(f"  L_norm:         {metrics['L_norm_mean']:.6f} +/- {metrics['L_norm_std']:.6f}")
        print(f"  L eff_rank:     {metrics['L_eff_rank']:.3f}")
        print(f"  elapsed:        {elapsed:.1f}s")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # --- Print trajectory table ---
    print(f"\n\n{'='*120}")
    print("TRAJECTORY TABLE")
    print(f"{'='*120}")
    header = f"{'Epoch':>5} | {'EffRank':>7} | {'PC1%':>6} | {'ER_CLN':>6} | {'ER_Fac':>6} | {'Sp_Full':>10} | {'Sp_CLN':>10} | {'Sp_Fac':>10} | {'Fac/CLN':>7} | {'L_norm':>10} | {'L_std':>10} | {'L_erank':>7}"
    print(header)
    print("-" * 120)
    for row in trajectory:
        print(f"{row['epoch']:>5} | {row['eff_rank']:>7.3f} | {row['pc1_dominance']*100:>5.1f}% | {row['eff_rank_cln_only']:>6.3f} | {row['eff_rank_factor_only']:>6.3f} | {row['spread_full']:>10.6f} | {row['spread_cln_only']:>10.6f} | {row['spread_factor_only']:>10.6f} | {row['spread_ratio_factor_vs_cln']:>7.4f} | {row['L_norm_mean']:>10.6f} | {row['L_norm_std']:>10.6f} | {row['L_eff_rank']:>7.3f}")

    # --- Analysis summary ---
    if len(trajectory) >= 2:
        ep10 = trajectory[0]
        ep80 = trajectory[-1]
        print(f"\n\nKEY COMPARISONS (ep10 vs ep80):")
        print(f"  eff_rank:       {ep10['eff_rank']:.3f} -> {ep80['eff_rank']:.3f} (delta: {ep80['eff_rank'] - ep10['eff_rank']:+.3f})")
        print(f"  PC1 dominance:  {ep10['pc1_dominance']*100:.1f}% -> {ep80['pc1_dominance']*100:.1f}% (delta: {(ep80['pc1_dominance'] - ep10['pc1_dominance'])*100:+.1f}%)")
        print(f"  L_norm:         {ep10['L_norm_mean']:.6f} -> {ep80['L_norm_mean']:.6f} (ratio: {ep80['L_norm_mean']/max(ep10['L_norm_mean'],1e-10):.3f}x)")
        print(f"  spread factor:  {ep10['spread_factor_only']:.6f} -> {ep80['spread_factor_only']:.6f}")
        print(f"  spread CLN:     {ep10['spread_cln_only']:.6f} -> {ep80['spread_cln_only']:.6f}")

        # Check CRPS-destroys-factor-structure pattern
        if ep10['eff_rank'] > ep80['eff_rank']:
            print(f"\n  CONFIRMED: CRPS destroys factor structure (eff_rank {ep10['eff_rank']:.3f} -> {ep80['eff_rank']:.3f})")
        elif ep10['eff_rank'] < ep80['eff_rank']:
            print(f"\n  UNEXPECTED: eff_rank INCREASED over training ({ep10['eff_rank']:.3f} -> {ep80['eff_rank']:.3f})")
        else:
            print(f"\n  NEUTRAL: eff_rank unchanged over training")

    # --- Save results ---
    out_dir = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-04-04/analysis/167a_rank_evolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "experiment": "167a_rank_evolution",
        "description": "Noise effective rank evolution across training epochs for 167a factorized decoder",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_split_start": TEST_START,
        "n_windows": 30,
        "n_noise_samples": 100,
        "trajectory": trajectory,
        "per_epoch": results,
    }

    results_path = out_dir / "rank_trajectory.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # --- Verification results ---
    verif_dir = Path("/home/max/Documents/vol-surface-vae-pub/results/validations/2026-04-04/verification_results")
    verif_dir.mkdir(parents=True, exist_ok=True)

    verification = {
        "experiment": "167a_rank_evolution",
        "status": "COMPLETE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "epochs_evaluated": [t["epoch"] for t in trajectory],
            "eff_rank_trajectory": {t["epoch"]: t["eff_rank"] for t in trajectory},
            "L_norm_trajectory": {t["epoch"]: t["L_norm_mean"] for t in trajectory},
            "spread_factor_trajectory": {t["epoch"]: t["spread_factor_only"] for t in trajectory},
            "pc1_trajectory": {t["epoch"]: round(t["pc1_dominance"] * 100, 1) for t in trajectory},
        },
        "findings": [],
    }

    # Add findings
    if len(trajectory) >= 2:
        ep10 = trajectory[0]
        ep80 = trajectory[-1]

        # Factor path death
        L_ratio = ep80["L_norm_mean"] / max(ep10["L_norm_mean"], 1e-10)
        verification["findings"].append({
            "name": "factor_path_death",
            "L_norm_ep10": ep10["L_norm_mean"],
            "L_norm_ep80": ep80["L_norm_mean"],
            "L_norm_ratio": round(L_ratio, 4),
            "confirmed": L_ratio < 0.5,
        })

        # CRPS destroys factor structure
        verification["findings"].append({
            "name": "crps_destroys_factor_structure",
            "eff_rank_ep10": ep10["eff_rank"],
            "eff_rank_ep80": ep80["eff_rank"],
            "confirmed": ep10["eff_rank"] > ep80["eff_rank"],
        })

        # Factor-only pathway contribution
        verification["findings"].append({
            "name": "factor_pathway_contribution",
            "spread_ratio_ep10": ep10["spread_ratio_factor_vs_cln"],
            "spread_ratio_ep80": ep80["spread_ratio_factor_vs_cln"],
            "factor_became_negligible": ep80["spread_ratio_factor_vs_cln"] < 0.1,
        })

    verif_path = verif_dir / "167a_rank_evolution.json"
    with open(verif_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"Saved: {verif_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
