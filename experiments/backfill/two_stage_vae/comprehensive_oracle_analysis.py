"""
Comprehensive Oracle Analysis for Two-Stage VAE

Deep, evidence-based analysis of the Two-Stage VAE under oracle conditions
(encoder sees target). Quantifies issues with architecture, training, and output quality.

Experiments:
- Part 1: Architecture (bottleneck capacity, FiLM params, gradient flow)
- Part 2: Training (loss components, z contribution)
- Part 3: Output Quality (directional accuracy, magnitude, per-grid errors)
- Part 4: Correlation (matrix comparison, factor contribution)
- Part 5: Tails (kurtosis recovery, tail events)
- Part 6: Temporal (ACF, mean reversion)

Usage:
    python experiments/backfill/two_stage_vae/comprehensive_oracle_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from scipy.stats import kurtosis, pearsonr
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.backfill.two_stage_vae.exp_student_t_decoder import (
    CVAETwoStageStudentT,
    to_log_returns,
    create_dataloader,
)


# ============================================================================
# Configuration
# ============================================================================

PERIODS = {
    "Vol Spike (Sep 2008)": 2100,
    "Crisis Peak (Oct 2008)": 2150,
    "Recovery (Mar 2009)": 2280,
    "Debt Ceiling (Aug 2011)": 2900,
    "Calm (2017)": 4300,
}

CONTEXT_LEN = 20
HORIZON = 30


def load_model(device: str = "cuda"):
    """Load the Student-t model."""
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def load_data():
    """Load and prepare data."""
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)
    return surfaces, log_returns


# ============================================================================
# Part 1: Architecture Analysis
# ============================================================================

def test_bottleneck_capacity(model, log_returns, config, device):
    """
    Experiment A1: Measure information loss at each bottleneck.

    Compares MSE when using:
    - Full model (ctx_emb + z)
    - Context only (ctx_emb + z=0)
    - Z only (ctx_emb=0 + z)
    """
    print("\n" + "="*70)
    print("EXPERIMENT A1: Bottleneck Capacity Test")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=64, shuffle=False)

    mse_full = []
    mse_ctx_only = []
    mse_z_only = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            # Get embeddings
            ctx_emb = model.ctx_encoder({"surface": batch_data})
            z_mean, z_logvar, z = model.main_encoder({"surface": batch_data})

            # Target
            target = batch_data[:, 1:]

            # Full model
            mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full.append(((pred_full - target) ** 2).mean().item())

            # Context only (z=0)
            z_zero = torch.zeros_like(z)
            mean_ctx, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
            pred_ctx = mean_ctx[:, :-1]
            mse_ctx_only.append(((pred_ctx - target) ** 2).mean().item())

            # Z only (ctx=0)
            ctx_zero = torch.zeros_like(ctx_emb)
            mean_z, _, _, _ = model.decoder(ctx_zero, z, sample=False)
            pred_z = mean_z[:, :-1]
            mse_z_only.append(((pred_z - target) ** 2).mean().item())

    mse_full_avg = np.mean(mse_full)
    mse_ctx_avg = np.mean(mse_ctx_only)
    mse_z_avg = np.mean(mse_z_only)

    # Contribution calculations
    # z_contribution: how much does z reduce error compared to ctx-only?
    z_contribution = (mse_ctx_avg - mse_full_avg) / mse_ctx_avg * 100 if mse_ctx_avg > 0 else 0
    # ctx_contribution: how much does ctx reduce error compared to z-only?
    ctx_contribution = (mse_z_avg - mse_full_avg) / mse_z_avg * 100 if mse_z_avg > 0 else 0

    results = {
        "mse_full": mse_full_avg,
        "mse_ctx_only": mse_ctx_avg,
        "mse_z_only": mse_z_avg,
        "z_contribution_pct": z_contribution,
        "ctx_contribution_pct": ctx_contribution,
    }

    print(f"\n  MSE (full model):     {mse_full_avg:.6f}")
    print(f"  MSE (ctx only, z=0):  {mse_ctx_avg:.6f}")
    print(f"  MSE (z only, ctx=0):  {mse_z_avg:.6f}")
    print(f"\n  Z Contribution:       {z_contribution:.1f}%")
    print(f"  Ctx Contribution:     {ctx_contribution:.1f}%")

    return results


def analyze_decoder_params(model):
    """
    Experiment A2: Analyze decoder parameters including mean_net structure.

    For Student-t decoder, checks mean_net and covariance nets.
    """
    print("\n" + "="*70)
    print("EXPERIMENT A2: Decoder Parameter Analysis")
    print("="*70)

    results = {}

    # Analyze mean_net
    mean_net_params = sum(p.numel() for p in model.decoder.mean_net.parameters())
    mean_net_norms = []
    for name, param in model.decoder.mean_net.named_parameters():
        mean_net_norms.append((name, param.norm().item()))

    # Analyze factor_net (covariance)
    factor_net_params = sum(p.numel() for p in model.decoder.factor_net.parameters())
    factor_net_norms = []
    for name, param in model.decoder.factor_net.named_parameters():
        factor_net_norms.append((name, param.norm().item()))

    # Analyze log_diag_net
    log_diag_net_params = sum(p.numel() for p in model.decoder.log_diag_net.parameters())

    results = {
        "mean_net_params": mean_net_params,
        "mean_net_weight_norms": mean_net_norms,
        "factor_net_params": factor_net_params,
        "factor_net_weight_norms": factor_net_norms,
        "log_diag_net_params": log_diag_net_params,
    }

    print(f"\n  Mean Net Parameters:     {mean_net_params}")
    print(f"  Factor Net Parameters:   {factor_net_params}")
    print(f"  Log Diag Net Parameters: {log_diag_net_params}")

    print("\n  Mean Net Weight Norms:")
    for name, norm in mean_net_norms:
        print(f"    {name}: {norm:.4f}")

    return results


def analyze_gradient_flow(model, log_returns, config, device):
    """
    Experiment A3: Measure gradient magnitude at each layer.
    """
    print("\n" + "="*70)
    print("EXPERIMENT A3: Gradient Flow Analysis")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)
    batch_data = next(iter(val_loader))[0].to(device)

    model.train()  # Enable gradients
    model.zero_grad()

    # Forward pass
    batch = {"surface": batch_data}
    mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

    target = batch_data[:, 1:]
    pred = mean[:, :-1]
    loss = ((pred - target) ** 2).mean()
    loss.backward()

    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.norm().item()

    model.eval()

    # Group by component
    ctx_encoder_grads = {k: v for k, v in gradients.items() if "ctx_encoder" in k}
    main_encoder_grads = {k: v for k, v in gradients.items() if "main_encoder" in k}
    decoder_grads = {k: v for k, v in gradients.items() if "decoder" in k}

    results = {
        "ctx_encoder_grad_norm": sum(ctx_encoder_grads.values()),
        "main_encoder_grad_norm": sum(main_encoder_grads.values()),
        "decoder_grad_norm": sum(decoder_grads.values()),
        "all_gradients": gradients,
    }

    print(f"\n  Ctx Encoder Total Grad Norm:  {results['ctx_encoder_grad_norm']:.4f}")
    print(f"  Main Encoder Total Grad Norm: {results['main_encoder_grad_norm']:.4f}")
    print(f"  Decoder Total Grad Norm:      {results['decoder_grad_norm']:.4f}")

    # Check key layers
    print("\n  Key Layer Gradients:")
    key_layers = ["decoder.mean_net", "decoder.factor_net", "decoder.log_diag_net"]
    for key in key_layers:
        layer_grads = {k: v for k, v in gradients.items() if key in k}
        total = sum(layer_grads.values())
        print(f"    {key}: {total:.4f}")

    return results


# ============================================================================
# Part 2: Training Analysis
# ============================================================================

def analyze_loss_components(model, log_returns, config, device):
    """
    Experiment B1: Analyze loss component magnitudes.
    """
    print("\n" + "="*70)
    print("EXPERIMENT B1: Loss Component Analysis")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=64, shuffle=False)
    kl_weight = config.get("kl_weight", 0.001)

    mse_losses = []
    kl_losses = []
    nll_losses = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # MSE loss
            mse = ((pred - target) ** 2).mean().item()
            mse_losses.append(mse)

            # KL loss
            kl = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean().item()
            kl_losses.append(kl)

            # Student-t NLL
            nll = model.decoder.compute_student_t_nll(pred, target, factor, log_diag).item()
            nll_losses.append(nll)

    results = {
        "mse_mean": np.mean(mse_losses),
        "kl_mean": np.mean(kl_losses),
        "weighted_kl_mean": kl_weight * np.mean(kl_losses),
        "nll_mean": np.mean(nll_losses),
        "kl_to_mse_ratio": np.mean(kl_losses) / np.mean(mse_losses) if np.mean(mse_losses) > 0 else 0,
        "weighted_kl_to_mse_ratio": kl_weight * np.mean(kl_losses) / np.mean(mse_losses) if np.mean(mse_losses) > 0 else 0,
    }

    print(f"\n  MSE Loss (mean):          {results['mse_mean']:.6f}")
    print(f"  KL Loss (mean):           {results['kl_mean']:.4f}")
    print(f"  Weighted KL (×{kl_weight}):     {results['weighted_kl_mean']:.6f}")
    print(f"  Student-t NLL (mean):     {results['nll_mean']:.4f}")
    print(f"\n  KL / MSE Ratio:           {results['kl_to_mse_ratio']:.2f}")
    print(f"  Weighted KL / MSE Ratio:  {results['weighted_kl_to_mse_ratio']:.4f}")

    if results['weighted_kl_to_mse_ratio'] < 0.01:
        print("\n  WARNING: Weighted KL << MSE, KL has minimal training effect!")

    return results


# ============================================================================
# Part 3: Output Quality Analysis
# ============================================================================

def analyze_directional_accuracy(model, surfaces, log_returns, config, device):
    """
    Experiment C1: Analyze directional accuracy and magnitude.

    IMPORTANT: Training semantics are mean[t] predicts x[t+1] (next step).
    So pred[h] (prediction at position context_len + h) predicts gt_returns[h+1].
    We compare pred[h] with gt_returns[h+1], NOT gt_returns[h].
    """
    print("\n" + "="*70)
    print("EXPERIMENT C1: Directional Accuracy Analysis")
    print("="*70)
    print("  NOTE: Using correct alignment (pred[h] vs gt[h+1])")
    print("        Training: mean[t] predicts x[t+1]")

    results = {}

    for period_name, start_idx in PERIODS.items():
        print(f"\n  Processing {period_name}...")

        # Get context and generate predictions
        # Need HORIZON+1 GT values since pred[h] compares to gt[h+1]
        context = log_returns[start_idx:start_idx + CONTEXT_LEN]
        gt_returns = log_returns[start_idx + CONTEXT_LEN:start_idx + CONTEXT_LEN + HORIZON + 1]

        if len(gt_returns) < HORIZON + 1:
            print(f"    Skipping - insufficient data")
            continue

        # Get model predictions (mean)
        pred_means = []

        with torch.no_grad():
            for h in range(HORIZON):
                if h == 0:
                    full_seq = np.concatenate([context, gt_returns[:1]], axis=0)
                else:
                    full_seq = np.concatenate([context, gt_returns[:h+1]], axis=0)

                seq_tensor = torch.tensor(full_seq, dtype=torch.float32).unsqueeze(0).to(device)
                batch = {"surface": seq_tensor}

                # Get mean prediction
                # mean[t] is trained to predict x[t+1]
                # At position (context_len + h), predicts gt_returns[h+1]
                mean, _, _, _, _ = model(batch, return_full_sequence=True)
                pred_h = mean[0, -1, 2, 2].cpu().numpy()  # ATM
                pred_means.append(pred_h)

        pred_means = np.array(pred_means)

        # CORRECT ALIGNMENT: pred[h] predicts gt[h+1]
        gt_atm = gt_returns[1:HORIZON+1, 2, 2]  # gt[1], gt[2], ..., gt[HORIZON]

        # Direction accuracy
        gt_signs = np.sign(gt_atm)
        pred_signs = np.sign(pred_means)
        direction_accuracy = (gt_signs == pred_signs).mean()

        # Magnitude analysis
        gt_magnitude = np.abs(gt_atm).mean()
        pred_magnitude = np.abs(pred_means).mean()
        magnitude_ratio = pred_magnitude / gt_magnitude if gt_magnitude > 0 else 0

        # Cumulative direction
        gt_cumsum = gt_atm.sum()
        pred_cumsum = pred_means.sum()
        cumulative_match = np.sign(gt_cumsum) == np.sign(pred_cumsum)

        # Correlation
        corr, _ = pearsonr(gt_atm, pred_means)

        results[period_name] = {
            "direction_accuracy": float(direction_accuracy),
            "gt_magnitude_mean": float(gt_magnitude),
            "pred_magnitude_mean": float(pred_magnitude),
            "magnitude_ratio": float(magnitude_ratio),
            "gt_cumsum": float(gt_cumsum),
            "pred_cumsum": float(pred_cumsum),
            "cumulative_match": bool(cumulative_match),
            "correlation": float(corr),
        }

        print(f"    Direction Accuracy: {direction_accuracy*100:.1f}%")
        print(f"    Magnitude Ratio:    {magnitude_ratio*100:.1f}%")
        print(f"    GT Cumsum:          {gt_cumsum*100:.2f}%")
        print(f"    Pred Cumsum:        {pred_cumsum*100:.2f}%")
        print(f"    Correlation:        {corr:.3f}")

    # Overall statistics
    avg_direction = np.mean([r["direction_accuracy"] for r in results.values()])
    avg_magnitude_ratio = np.mean([r["magnitude_ratio"] for r in results.values()])
    cumulative_matches = sum(1 for r in results.values() if r["cumulative_match"])

    results["summary"] = {
        "avg_direction_accuracy": avg_direction,
        "avg_magnitude_ratio": avg_magnitude_ratio,
        "cumulative_matches": f"{cumulative_matches}/{len(PERIODS)}",
    }

    print(f"\n  SUMMARY:")
    print(f"    Avg Direction Accuracy: {avg_direction*100:.1f}%")
    print(f"    Avg Magnitude Ratio:    {avg_magnitude_ratio*100:.1f}%")
    print(f"    Cumulative Matches:     {cumulative_matches}/{len(PERIODS)}")

    return results


def test_conditional_mean(model, log_returns, config, device):
    """
    Experiment C2: Test if prediction magnitude correlates with z magnitude.
    """
    print("\n" + "="*70)
    print("EXPERIMENT C2: Conditional Mean Test")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=64, shuffle=False)

    z_magnitudes = []
    pred_magnitudes = []
    gt_magnitudes = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            # Get z
            z_mean, z_logvar, z = model.main_encoder({"surface": batch_data})
            ctx_emb = model.ctx_encoder({"surface": batch_data})

            # Get prediction
            mean, _, _, _ = model.decoder(ctx_emb, z, sample=False)

            # Target
            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # Compute magnitudes per sample
            for i in range(batch_data.shape[0]):
                z_mag = z[i].norm().item()
                pred_mag = pred[i].abs().mean().item()
                gt_mag = target[i].abs().mean().item()

                z_magnitudes.append(z_mag)
                pred_magnitudes.append(pred_mag)
                gt_magnitudes.append(gt_mag)

    # Correlations
    z_pred_corr, _ = pearsonr(z_magnitudes, pred_magnitudes)
    z_gt_corr, _ = pearsonr(z_magnitudes, gt_magnitudes)
    pred_gt_corr, _ = pearsonr(pred_magnitudes, gt_magnitudes)

    results = {
        "z_pred_correlation": z_pred_corr,
        "z_gt_correlation": z_gt_corr,
        "pred_gt_correlation": pred_gt_corr,
        "z_magnitude_mean": np.mean(z_magnitudes),
        "z_magnitude_std": np.std(z_magnitudes),
        "pred_magnitude_mean": np.mean(pred_magnitudes),
        "gt_magnitude_mean": np.mean(gt_magnitudes),
    }

    print(f"\n  Correlation(|z|, |pred|):  {z_pred_corr:.3f}")
    print(f"  Correlation(|z|, |GT|):    {z_gt_corr:.3f}")
    print(f"  Correlation(|pred|, |GT|): {pred_gt_corr:.3f}")
    print(f"\n  |z| mean:    {np.mean(z_magnitudes):.3f} ± {np.std(z_magnitudes):.3f}")
    print(f"  |pred| mean: {np.mean(pred_magnitudes):.6f}")
    print(f"  |GT| mean:   {np.mean(gt_magnitudes):.6f}")

    if abs(z_pred_corr) < 0.3:
        print("\n  WARNING: Low correlation between z and prediction magnitude!")
        print("           Model may be ignoring z for mean prediction.")

    return results


def analyze_per_grid_errors(model, log_returns, config, device):
    """
    Experiment C3: Per-grid error analysis.
    """
    print("\n" + "="*70)
    print("EXPERIMENT C3: Per-Grid Error Analysis")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=64, shuffle=False)

    grid_mse = np.zeros((5, 5))
    grid_bias = np.zeros((5, 5))
    grid_direction_acc = np.zeros((5, 5))
    n_samples = 0

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, _, _, _, _ = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:].cpu().numpy()
            pred = mean[:, :-1].cpu().numpy()

            for i in range(5):
                for j in range(5):
                    error = pred[:, :, i, j] - target[:, :, i, j]
                    grid_mse[i, j] += (error ** 2).sum()
                    grid_bias[i, j] += error.sum()
                    grid_direction_acc[i, j] += (np.sign(pred[:, :, i, j]) == np.sign(target[:, :, i, j])).sum()

            n_samples += target.size

    grid_mse /= (n_samples / 25)
    grid_bias /= (n_samples / 25)
    grid_direction_acc /= (n_samples / 25)

    # Normalize to ATM
    atm_mse = grid_mse[2, 2]
    mse_ratio = grid_mse / atm_mse

    results = {
        "grid_mse": grid_mse.tolist(),
        "grid_bias": grid_bias.tolist(),
        "grid_direction_acc": grid_direction_acc.tolist(),
        "mse_ratio_to_atm": mse_ratio.tolist(),
        "atm_mse": float(atm_mse),
        "max_mse": float(grid_mse.max()),
        "min_mse": float(grid_mse.min()),
        "max_mse_ratio": float(mse_ratio.max()),
    }

    print("\n  MSE per Grid Point (×1e6):")
    print(np.round(grid_mse * 1e6, 2))

    print("\n  MSE Ratio to ATM:")
    print(np.round(mse_ratio, 1))

    print("\n  Direction Accuracy per Grid Point:")
    print(np.round(grid_direction_acc * 100, 1))

    print(f"\n  ATM MSE:      {atm_mse:.6f}")
    print(f"  Max MSE:      {grid_mse.max():.6f} (at corner)")
    print(f"  Max/ATM Ratio: {mse_ratio.max():.1f}×")

    return results


# ============================================================================
# Part 4: Correlation and Covariance Analysis
# ============================================================================

def compare_correlation_matrices(model, log_returns, config, device, n_samples=100):
    """
    Experiment D1: Compare sample vs GT correlation matrices.
    """
    print("\n" + "="*70)
    print("EXPERIMENT D1: Correlation Matrix Comparison")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 10:  # Limit for speed
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            # Get target
            target = batch_data[:, 1:]  # (B, T, 5, 5)

            # Flatten and collect
            samples_flat = samples[:, :, -1, :, :].cpu().numpy().reshape(-1, 25)
            gt_flat = target[:, -1, :, :].cpu().numpy().reshape(-1, 25)

            all_samples.append(samples_flat)
            all_gt.append(gt_flat)

    all_samples = np.concatenate(all_samples, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    # Compute correlation matrices
    sample_corr = np.corrcoef(all_samples.T)
    gt_corr = np.corrcoef(all_gt.T)

    # Metrics
    frobenius_diff = np.linalg.norm(sample_corr - gt_corr, 'fro')

    # Correlation of correlations (how well do off-diagonal correlations match?)
    sample_triu = sample_corr[np.triu_indices(25, k=1)]
    gt_triu = gt_corr[np.triu_indices(25, k=1)]
    corr_of_corr, _ = pearsonr(sample_triu, gt_triu)

    # Mean absolute correlation
    sample_abs_corr = np.abs(sample_corr[np.triu_indices(25, k=1)]).mean()
    gt_abs_corr = np.abs(gt_corr[np.triu_indices(25, k=1)]).mean()
    correlation_preserved = sample_abs_corr / gt_abs_corr if gt_abs_corr > 0 else 0

    results = {
        "frobenius_diff": float(frobenius_diff),
        "correlation_of_correlations": float(corr_of_corr),
        "sample_mean_abs_corr": float(sample_abs_corr),
        "gt_mean_abs_corr": float(gt_abs_corr),
        "correlation_preserved_pct": float(correlation_preserved * 100),
        "sample_corr_matrix": sample_corr.tolist(),
        "gt_corr_matrix": gt_corr.tolist(),
    }

    print(f"\n  Frobenius Norm Difference:     {frobenius_diff:.3f}")
    print(f"  Correlation of Correlations:   {corr_of_corr:.3f}")
    print(f"  Sample Mean |Correlation|:     {sample_abs_corr:.3f}")
    print(f"  GT Mean |Correlation|:         {gt_abs_corr:.3f}")
    print(f"  Correlation Preserved:         {correlation_preserved*100:.1f}%")

    return results


def analyze_factor_contribution(model, log_returns, config, device):
    """
    Experiment D2: Analyze factor F contribution to covariance.
    """
    print("\n" + "="*70)
    print("EXPERIMENT D2: Factor Contribution Analysis")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=64, shuffle=False)

    all_factors = []
    all_log_diags = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            all_factors.append(factor.cpu().numpy())
            all_log_diags.append(log_diag.cpu().numpy())

    # Average factor and log_diag across batches
    factors = np.concatenate(all_factors, axis=0)  # (N, 25, rank)
    log_diags = np.concatenate(all_log_diags, axis=0)  # (N, 25)

    avg_factor = factors.mean(axis=0)  # (25, rank)
    avg_log_diag = log_diags.mean(axis=0)  # (25,)
    avg_diag = np.exp(avg_log_diag)  # (25,)

    # Covariance: Σ = FF^T + D
    FFT = avg_factor @ avg_factor.T  # (25, 25)

    # Factor contribution to total variance
    factor_var = np.diag(FFT)
    total_var = factor_var + avg_diag
    factor_contribution = factor_var / total_var

    # Off-diagonal analysis
    off_diag_mask = ~np.eye(25, dtype=bool)
    off_diag_FFT = FFT[off_diag_mask]
    off_diag_magnitude = np.abs(off_diag_FFT).mean()

    results = {
        "factor_contribution_mean": float(factor_contribution.mean()),
        "factor_contribution_per_grid": factor_contribution.reshape(5, 5).tolist(),
        "off_diag_magnitude": float(off_diag_magnitude),
        "avg_diag_variance": float(avg_diag.mean()),
        "avg_factor_variance": float(factor_var.mean()),
        "rank": avg_factor.shape[1],
    }

    print(f"\n  Factor Rank:                  {avg_factor.shape[1]}")
    print(f"  Avg Factor Contribution:      {factor_contribution.mean()*100:.1f}%")
    print(f"  Off-diagonal Magnitude:       {off_diag_magnitude:.6f}")
    print(f"  Avg Diagonal Variance:        {avg_diag.mean():.6f}")
    print(f"  Avg Factor Variance:          {factor_var.mean():.6f}")

    print("\n  Factor Contribution per Grid (%):")
    print(np.round(factor_contribution.reshape(5, 5) * 100, 1))

    return results


# ============================================================================
# Part 5: Tail and Distribution Analysis
# ============================================================================

def analyze_kurtosis_per_grid(model, log_returns, config, device, n_samples=200):
    """
    Experiment E1: Kurtosis per grid point.
    """
    print("\n" + "="*70)
    print("EXPERIMENT E1: Kurtosis Per Grid Analysis")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:  # Limit for speed
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            target = batch_data[:, 1:]

            # Take last timestep
            samples_last = samples[:, :, -1, :, :].cpu().numpy()  # (n_samples, B, 5, 5)
            gt_last = target[:, -1, :, :].cpu().numpy()  # (B, 5, 5)

            all_samples.append(samples_last.reshape(-1, 5, 5))
            all_gt.append(gt_last)

    all_samples = np.concatenate(all_samples, axis=0)  # (N, 5, 5)
    all_gt = np.concatenate(all_gt, axis=0)  # (M, 5, 5)

    gt_kurtosis = np.zeros((5, 5))
    model_kurtosis = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_kurtosis[i, j] = kurtosis(all_gt[:, i, j], fisher=True)
            model_kurtosis[i, j] = kurtosis(all_samples[:, i, j], fisher=True)

    # Recovery ratio
    recovery = np.abs(model_kurtosis) / np.abs(gt_kurtosis + 1e-8)
    recovery = np.clip(recovery, 0, 2)  # Cap at 200% for display

    results = {
        "gt_kurtosis": gt_kurtosis.tolist(),
        "model_kurtosis": model_kurtosis.tolist(),
        "recovery_ratio": recovery.tolist(),
        "mean_recovery": float(recovery.mean()),
        "atm_gt_kurtosis": float(gt_kurtosis[2, 2]),
        "atm_model_kurtosis": float(model_kurtosis[2, 2]),
    }

    print("\n  GT Kurtosis (Fisher):")
    print(np.round(gt_kurtosis, 1))

    print("\n  Model Kurtosis (Fisher):")
    print(np.round(model_kurtosis, 2))

    print("\n  Recovery Ratio (%):")
    print(np.round(recovery * 100, 1))

    print(f"\n  Mean Recovery:      {recovery.mean()*100:.1f}%")
    print(f"  ATM GT Kurtosis:    {gt_kurtosis[2, 2]:.1f}")
    print(f"  ATM Model Kurtosis: {model_kurtosis[2, 2]:.2f}")

    return results


def analyze_tail_events(model, log_returns, config, device, n_samples=200):
    """
    Experiment E2: Tail event capture at various sigma levels.
    """
    print("\n" + "="*70)
    print("EXPERIMENT E2: Tail Event Analysis")
    print("="*70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)
    thresholds = [1, 2, 3, 4]

    # Collect all data
    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            target = batch_data[:, 1:]

            samples_last = samples[:, :, -1, :, :].cpu().numpy().flatten()
            gt_last = target[:, -1, :, :].cpu().numpy().flatten()

            all_samples.extend(samples_last)
            all_gt.extend(gt_last)

    all_samples = np.array(all_samples)
    all_gt = np.array(all_gt)

    gt_std = all_gt.std()
    sample_std = all_samples.std()

    results = {"gt_std": float(gt_std), "sample_std": float(sample_std)}

    print(f"\n  GT Std:     {gt_std:.6f}")
    print(f"  Sample Std: {sample_std:.6f}")
    print(f"  Std Ratio:  {sample_std/gt_std:.2f}")

    print("\n  Tail Event Rates:")
    print("  " + "-"*50)
    print(f"  {'Threshold':<12} {'GT Rate':<12} {'Model Rate':<12} {'Ratio':<12}")
    print("  " + "-"*50)

    for thresh in thresholds:
        gt_tail_rate = (np.abs(all_gt) > thresh * gt_std).mean()
        model_tail_rate = (np.abs(all_samples) > thresh * sample_std).mean()
        ratio = model_tail_rate / gt_tail_rate if gt_tail_rate > 0 else 0

        results[f"{thresh}sigma"] = {
            "gt_tail_rate": float(gt_tail_rate),
            "model_tail_rate": float(model_tail_rate),
            "ratio": float(ratio),
        }

        print(f"  {thresh}σ{'':<10} {gt_tail_rate*100:.3f}%{'':<6} {model_tail_rate*100:.3f}%{'':<6} {ratio:.2f}×")

    return results


# ============================================================================
# Part 6: Temporal Dynamics Analysis
# ============================================================================

def analyze_autocorrelation(model, log_returns, config, device, n_samples=50):
    """
    Experiment F1: Autocorrelation comparison.
    """
    print("\n" + "="*70)
    print("EXPERIMENT F1: Autocorrelation Analysis")
    print("="*70)

    # Use longer sequences for ACF
    seq_len = 50
    val_loader = create_dataloader(log_returns, seq_len, batch_size=16, shuffle=False)
    max_lag = 10

    gt_acfs = []
    model_acfs = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 10:
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            target = batch_data[:, 1:, 2, 2].cpu().numpy()  # ATM only, (B, T)
            samples_atm = samples[:, :, :-1, 2, 2].cpu().numpy()  # (n_samples, B, T)

            # Compute ACF for each sequence
            for seq in target:
                if len(seq) > max_lag:
                    acf = np.correlate(seq - seq.mean(), seq - seq.mean(), mode='full')
                    acf = acf[len(acf)//2:len(acf)//2 + max_lag + 1]
                    acf = acf / acf[0]
                    gt_acfs.append(acf)

            for s in range(min(10, n_samples)):  # Limit samples
                for seq in samples_atm[s]:
                    if len(seq) > max_lag:
                        acf = np.correlate(seq - seq.mean(), seq - seq.mean(), mode='full')
                        acf = acf[len(acf)//2:len(acf)//2 + max_lag + 1]
                        acf = acf / (acf[0] + 1e-8)
                        model_acfs.append(acf)

    if len(gt_acfs) > 0 and len(model_acfs) > 0:
        gt_acf_mean = np.mean(gt_acfs, axis=0)
        model_acf_mean = np.mean(model_acfs, axis=0)

        # ACF preservation ratio
        acf_ratio = model_acf_mean / (gt_acf_mean + 1e-8)

        results = {
            "gt_acf": gt_acf_mean.tolist(),
            "model_acf": model_acf_mean.tolist(),
            "acf_ratio": acf_ratio.tolist(),
            "lag1_gt": float(gt_acf_mean[1]),
            "lag1_model": float(model_acf_mean[1]),
            "lag1_preservation": float(model_acf_mean[1] / (gt_acf_mean[1] + 1e-8)),
        }

        print("\n  Autocorrelation Function (ATM):")
        print(f"  {'Lag':<6} {'GT ACF':<12} {'Model ACF':<12} {'Ratio':<12}")
        print("  " + "-"*42)
        for lag in range(min(6, len(gt_acf_mean))):
            print(f"  {lag:<6} {gt_acf_mean[lag]:.4f}{'':<6} {model_acf_mean[lag]:.4f}{'':<6} {acf_ratio[lag]:.2f}")

        print(f"\n  Lag-1 ACF Preservation: {results['lag1_preservation']*100:.1f}%")
    else:
        results = {"error": "Insufficient data for ACF calculation"}
        print("  Insufficient data for ACF calculation")

    return results


def analyze_mean_reversion(model, log_returns, config, device, n_samples=50):
    """
    Experiment F2: Mean reversion speed analysis via AR(1) fitting.
    """
    print("\n" + "="*70)
    print("EXPERIMENT F2: Mean Reversion Analysis")
    print("="*70)

    # Use longer sequences
    seq_len = 30
    val_loader = create_dataloader(log_returns, seq_len, batch_size=16, shuffle=False)

    gt_phis = []
    model_phis = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 10:
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)

            target = batch_data[:, 1:, 2, 2].cpu().numpy()  # ATM only
            samples_atm = samples[:, :, :-1, 2, 2].cpu().numpy()

            # Fit AR(1): x_t = c + phi * x_{t-1} + epsilon
            # Using simple OLS: phi = cov(x_t, x_{t-1}) / var(x_{t-1})
            for seq in target:
                if len(seq) > 5:
                    x_t = seq[1:]
                    x_t1 = seq[:-1]
                    if np.var(x_t1) > 1e-10:
                        phi = np.cov(x_t, x_t1)[0, 1] / np.var(x_t1)
                        gt_phis.append(phi)

            for s in range(min(10, n_samples)):
                for seq in samples_atm[s]:
                    if len(seq) > 5:
                        x_t = seq[1:]
                        x_t1 = seq[:-1]
                        if np.var(x_t1) > 1e-10:
                            phi = np.cov(x_t, x_t1)[0, 1] / np.var(x_t1)
                            model_phis.append(phi)

    if len(gt_phis) > 0 and len(model_phis) > 0:
        gt_phi_mean = np.mean(gt_phis)
        model_phi_mean = np.mean(model_phis)

        gt_speed = 1 - gt_phi_mean
        model_speed = 1 - model_phi_mean

        results = {
            "gt_phi": float(gt_phi_mean),
            "model_phi": float(model_phi_mean),
            "gt_mean_reversion_speed": float(gt_speed),
            "model_mean_reversion_speed": float(model_speed),
            "speed_ratio": float(model_speed / gt_speed) if gt_speed != 0 else 0,
        }

        print(f"\n  GT AR(1) φ:                 {gt_phi_mean:.4f}")
        print(f"  Model AR(1) φ:              {model_phi_mean:.4f}")
        print(f"\n  GT Mean Reversion Speed:    {gt_speed:.4f}")
        print(f"  Model Mean Reversion Speed: {model_speed:.4f}")
        print(f"  Speed Ratio:                {results['speed_ratio']:.2f}")

        if model_speed > gt_speed * 1.5:
            print("\n  WARNING: Model shows excessive mean reversion!")
    else:
        results = {"error": "Insufficient data for AR(1) fitting"}
        print("  Insufficient data for AR(1) fitting")

    return results


# ============================================================================
# Main Analysis Runner
# ============================================================================

def generate_report(results, output_path):
    """Generate markdown report from results."""
    report = []
    report.append("# Comprehensive Oracle Analysis Report")
    report.append(f"\nGenerated from Two-Stage VAE with Student-t Decoder\n")
    report.append("="*70 + "\n")

    # Summary table
    report.append("## Summary of Key Findings\n")
    report.append("| Category | Metric | Value | Target | Status |")
    report.append("|----------|--------|-------|--------|--------|")

    # Architecture
    z_contrib = results.get("bottleneck", {}).get("z_contribution_pct", 0)
    report.append(f"| Architecture | Z Contribution | {z_contrib:.1f}% | >50% | {'✓' if z_contrib > 50 else '✗'} |")

    # Output Quality
    dir_acc = results.get("directional", {}).get("summary", {}).get("avg_direction_accuracy", 0) * 100
    mag_ratio = results.get("directional", {}).get("summary", {}).get("avg_magnitude_ratio", 0) * 100
    report.append(f"| Output | Direction Accuracy | {dir_acc:.1f}% | >60% | {'✓' if dir_acc > 60 else '✗'} |")
    report.append(f"| Output | Magnitude Ratio | {mag_ratio:.1f}% | >50% | {'✓' if mag_ratio > 50 else '✗'} |")

    # Correlation
    corr_preserved = results.get("correlation", {}).get("correlation_preserved_pct", 0)
    report.append(f"| Correlation | Preserved | {corr_preserved:.1f}% | >50% | {'✓' if corr_preserved > 50 else '✗'} |")

    # Tails
    kurtosis_recovery = results.get("kurtosis", {}).get("mean_recovery", 0) * 100
    report.append(f"| Tails | Kurtosis Recovery | {kurtosis_recovery:.1f}% | >50% | {'✓' if kurtosis_recovery > 50 else '✗'} |")

    # Temporal
    acf_pres = results.get("acf", {}).get("lag1_preservation", 0) * 100
    report.append(f"| Temporal | ACF Preservation | {acf_pres:.1f}% | >50% | {'✓' if acf_pres > 50 else '✗'} |")

    report.append("\n")

    # Detailed sections
    for section, data in results.items():
        report.append(f"\n## {section.replace('_', ' ').title()}\n")
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, (list, dict)):
                    report.append(f"- **{key}**: {value}")
        report.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to {output_path}")


def run_comprehensive_analysis():
    """Run all diagnostic experiments."""
    print("="*70)
    print("COMPREHENSIVE ORACLE ANALYSIS FOR TWO-STAGE VAE")
    print("="*70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model and data
    print("\nLoading model and data...")
    model, config = load_model(device)
    surfaces, log_returns = load_data()
    print(f"Data shape: {log_returns.shape}")

    results = {}

    # Part 1: Architecture Analysis
    print("\n" + "#"*70)
    print("# PART 1: ARCHITECTURE ANALYSIS")
    print("#"*70)

    results["bottleneck"] = test_bottleneck_capacity(model, log_returns, config, device)
    results["decoder_params"] = analyze_decoder_params(model)
    results["gradient_flow"] = analyze_gradient_flow(model, log_returns, config, device)

    # Part 2: Training Analysis
    print("\n" + "#"*70)
    print("# PART 2: TRAINING ANALYSIS")
    print("#"*70)

    results["loss_components"] = analyze_loss_components(model, log_returns, config, device)

    # Part 3: Output Quality Analysis
    print("\n" + "#"*70)
    print("# PART 3: OUTPUT QUALITY ANALYSIS")
    print("#"*70)

    results["directional"] = analyze_directional_accuracy(model, surfaces, log_returns, config, device)
    results["conditional_mean"] = test_conditional_mean(model, log_returns, config, device)
    results["per_grid"] = analyze_per_grid_errors(model, log_returns, config, device)

    # Part 4: Correlation Analysis
    print("\n" + "#"*70)
    print("# PART 4: CORRELATION ANALYSIS")
    print("#"*70)

    results["correlation"] = compare_correlation_matrices(model, log_returns, config, device)
    results["factor"] = analyze_factor_contribution(model, log_returns, config, device)

    # Part 5: Tail Analysis
    print("\n" + "#"*70)
    print("# PART 5: TAIL AND DISTRIBUTION ANALYSIS")
    print("#"*70)

    results["kurtosis"] = analyze_kurtosis_per_grid(model, log_returns, config, device)
    results["tail_events"] = analyze_tail_events(model, log_returns, config, device)

    # Part 6: Temporal Analysis
    print("\n" + "#"*70)
    print("# PART 6: TEMPORAL DYNAMICS ANALYSIS")
    print("#"*70)

    results["acf"] = analyze_autocorrelation(model, log_returns, config, device)
    results["mean_reversion"] = analyze_mean_reversion(model, log_returns, config, device)

    # Save results
    output_dir = Path("results/two_stage_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "comprehensive_analysis_results.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    results_serializable = convert_to_serializable(results)
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Generate report
    report_path = output_dir / "comprehensive_analysis_report.md"
    generate_report(results_serializable, report_path)

    # Final Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - FINAL SUMMARY")
    print("="*70)

    print("\n  KEY METRICS:")
    print(f"    Z Contribution:        {results['bottleneck']['z_contribution_pct']:.1f}%")
    print(f"    Direction Accuracy:    {results['directional']['summary']['avg_direction_accuracy']*100:.1f}%")
    print(f"    Magnitude Ratio:       {results['directional']['summary']['avg_magnitude_ratio']*100:.1f}%")
    print(f"    Correlation Preserved: {results['correlation']['correlation_preserved_pct']:.1f}%")
    print(f"    Kurtosis Recovery:     {results['kurtosis']['mean_recovery']*100:.1f}%")

    return results


if __name__ == "__main__":
    results = run_comprehensive_analysis()
