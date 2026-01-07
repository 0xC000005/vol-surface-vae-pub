"""
Validate Two-Stage CVAE Heteroscedastic - Forecast Mode (Prior Sampling).

This tests REALISTIC forecasting where z is NOT conditioned on target:
1. Context-only encoding (no target knowledge)
2. Prior sampling for z during forecast horizon
3. Monte Carlo samples to estimate CI

This is the proper test of whether CIs are calibrated for deployment.

Usage:
    python experiments/backfill/prior_encoder_ablation/validate_two_stage_forecast.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageHeteroscedastic


def to_log_returns(surfaces):
    """Transform IV surfaces to log-returns."""
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def transform_back_samples(samples_log_return, initial_log_surface):
    """Transform log-return samples back to IV levels."""
    cumsum = np.cumsum(samples_log_return, axis=1)
    log_iv_samples = initial_log_surface + cumsum
    return np.exp(log_iv_samples)


def forecast_with_prior(model, context, horizon, n_samples, device):
    """
    Generate forecast samples using prior sampling for z.

    Args:
        model: The CVAETwoStageHeteroscedastic model
        context: (B, context_len, 5, 5) context sequence
        horizon: Number of steps to forecast
        n_samples: Number of Monte Carlo samples
        device: torch device

    Returns:
        samples: (n_samples, horizon, 5, 5) forecast samples
        pred_stds: (n_samples, horizon, 5, 5) predicted standard deviations
    """
    model.eval()
    B, C, H, W = context.shape
    latent_dim = model.config["latent_dim"]

    samples_all = []
    stds_all = []

    with torch.no_grad():
        for _ in range(n_samples):
            # Encode context only
            ctx_emb = model.ctx_encoder({"surface": context})  # (B, C, ctx_dim)

            # Create a dummy extended embedding for the decoder
            # The decoder expects (B, T, ctx_dim) where T = C + horizon
            last_ctx_emb = ctx_emb[:, -1:, :]  # (B, 1, ctx_dim)
            extended_ctx_emb = torch.cat([
                ctx_emb,
                last_ctx_emb.repeat(1, horizon, 1)
            ], dim=1)  # (B, C+horizon, ctx_dim)

            # Sample z from prior for horizon portion
            # For context: use encoder mean (deterministic)
            # For horizon: sample from N(0, 1)
            z_context_mean, _, _ = model.encoder({"surface": context})  # (B, C, latent_dim)

            z_horizon = torch.randn(B, horizon, latent_dim, device=device)

            z = torch.cat([z_context_mean, z_horizon], dim=1)  # (B, C+horizon, latent_dim)

            # Decode with heteroscedastic decoder
            decoded_mean, decoded_logvar = model.decoder(extended_ctx_emb, z)  # (B, C+horizon, 5, 5)

            # Extract horizon portion
            forecast_mean = decoded_mean[:, C:].cpu().numpy()  # (B, horizon, 5, 5)
            forecast_logvar = decoded_logvar[:, C:].cpu().numpy()  # (B, horizon, 5, 5)

            # Sample from predicted distribution
            forecast_std = np.exp(0.5 * forecast_logvar)
            sample = forecast_mean + forecast_std * np.random.randn(*forecast_mean.shape)

            samples_all.append(sample[0])  # (horizon, 5, 5)
            stds_all.append(forecast_std[0])  # (horizon, 5, 5)

    return np.array(samples_all), np.array(stds_all)  # (n_samples, horizon, 5, 5)


def main():
    print("=" * 70)
    print("Two-Stage CVAE Heteroscedastic - FORECAST Validation")
    print("=" * 70)
    print()
    print("Testing with PRIOR SAMPLING (realistic forecasting scenario)")
    print("z is NOT conditioned on target - true uncertainty estimation")
    print()

    # Load checkpoint
    checkpoint_path = Path("models/backfill/two_stage/two_stage_log_return_nll_best.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    config = checkpoint["model_config"]

    # Build model
    model = CVAETwoStageHeteroscedastic(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = config["device"]
    print(f"  Device: {device}")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    print(f"  Total surfaces: {len(surfaces)}")

    # Transform to log-returns
    log_returns, log_surfaces = to_log_returns(surfaces)
    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)

    gt_std = log_returns.std()
    print(f"  Ground truth log-return std: {gt_std:.4f}")

    # Parameters
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    n_samples = 1000

    # Create validation sequences
    all_sequences = create_sequences(log_returns_tensor, seq_len)
    n_train = int(len(all_sequences) * 0.8)
    val_sequences = all_sequences[n_train:]
    print(f"\nValidation sequences: {len(val_sequences)}")

    # Sample subset for efficiency
    n_eval = min(200, len(val_sequences))
    eval_indices = np.random.choice(len(val_sequences), n_eval, replace=False)
    eval_sequences = val_sequences[eval_indices]

    print(f"Evaluating {n_eval} sequences with {n_samples} samples each...")
    print("(This tests PRIOR sampling - realistic forecasting)")

    # Storage for results
    all_violations_log = []
    all_violations_iv = []
    all_pred_std = []
    all_sample_std = []

    # Evaluate each sequence
    for seq_idx, seq in enumerate(tqdm(eval_sequences, desc="Forecasting")):
        seq_full = seq.numpy()  # (seq_len, 5, 5)
        context = seq[:context_len].unsqueeze(0).to(device)  # (1, context_len, 5, 5)
        target_log = seq_full[context_len:]  # (horizon, 5, 5)

        # Generate forecast samples with prior sampling
        samples_log, pred_stds = forecast_with_prior(
            model, context, horizon, n_samples, device
        )  # (n_samples, horizon, 5, 5)

        # Statistics
        all_pred_std.append(pred_stds.mean())
        all_sample_std.append(samples_log.std(axis=0).mean())

        # CI Violations in log-return space
        ci_lower = np.percentile(samples_log, 5, axis=0)
        ci_upper = np.percentile(samples_log, 95, axis=0)
        violations_log = (target_log < ci_lower) | (target_log > ci_upper)
        all_violations_log.append(violations_log)

        # CI Violations in IV space
        initial_log_iv = log_surfaces[n_train + eval_indices[seq_idx] + context_len - 1]
        samples_iv = transform_back_samples(samples_log, initial_log_iv)
        target_iv = transform_back_samples(target_log[np.newaxis], initial_log_iv)[0]

        ci_lower_iv = np.percentile(samples_iv, 5, axis=0)
        ci_upper_iv = np.percentile(samples_iv, 95, axis=0)
        violations_iv = (target_iv < ci_lower_iv) | (target_iv > ci_upper_iv)
        all_violations_iv.append(violations_iv)

    # Aggregate results
    all_violations_log = np.array(all_violations_log)
    all_violations_iv = np.array(all_violations_iv)

    # --- Results ---
    print("\n" + "=" * 70)
    print("FORECAST VALIDATION RESULTS (Prior Sampling)")
    print("=" * 70)

    # Overall statistics
    print("\n1. UNCERTAINTY CALIBRATION")
    print("-" * 40)
    mean_pred_std = np.mean(all_pred_std)
    mean_sample_std = np.mean(all_sample_std)
    print(f"  Ground truth log-return std:  {gt_std:.4f}")
    print(f"  Model predicted std (mean):   {mean_pred_std:.4f}")
    print(f"  Monte Carlo sample std (mean):{mean_sample_std:.4f}")
    print(f"  Pred / GT ratio:              {mean_pred_std/gt_std*100:.1f}%")
    print(f"  Sample / GT ratio:            {mean_sample_std/gt_std*100:.1f}%")

    # CI violations in log-return space
    print("\n2. CI VIOLATIONS IN LOG-RETURN SPACE")
    print("-" * 40)
    print("Target: 10% violations for 90% CI")
    print()
    print("Per-Horizon Violations:")
    print(f"  {'Horizon':<10} {'Violations':<12} {'Status'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10}")

    for h in range(horizon):
        violation_rate = all_violations_log[:, h].mean() * 100
        status = "✓" if 8 <= violation_rate <= 12 else "⚠" if 5 <= violation_rate <= 15 else "✗"
        print(f"  H={h+1:<7} {violation_rate:>6.1f}%      {status}")

    overall_log = all_violations_log.mean() * 100
    print(f"\n  Overall:   {overall_log:.1f}%")

    # CI violations in IV space
    print("\n3. CI VIOLATIONS IN IV SPACE (after transform-back)")
    print("-" * 40)
    print("Target: 10% violations for 90% CI")
    print()
    print("Per-Horizon Violations:")
    print(f"  {'Horizon':<10} {'Violations':<12} {'Status'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10}")

    for h in range(horizon):
        violation_rate = all_violations_iv[:, h].mean() * 100
        status = "✓" if 8 <= violation_rate <= 12 else "⚠" if 5 <= violation_rate <= 15 else "✗"
        print(f"  H={h+1:<7} {violation_rate:>6.1f}%      {status}")

    overall_iv = all_violations_iv.mean() * 100
    print(f"\n  Overall:   {overall_iv:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Log-return space violations: {overall_log:.1f}% (target: 10%)")
    print(f"  IV space violations:         {overall_iv:.1f}% (target: 10%)")
    print(f"  Predicted std ratio:         {mean_pred_std/gt_std*100:.1f}% of GT")

    if 8 <= overall_log <= 12:
        print("\n  ✓ SUCCESS: CI coverage is well-calibrated (~10% violations)!")
    elif overall_log < 8:
        print(f"\n  ⚠ Over-conservative: CIs too wide ({overall_log:.1f}% < 8% target)")
    else:
        print(f"\n  ⚠ Under-conservative: CIs too narrow ({overall_log:.1f}% > 12% target)")

    # Save results
    results_path = Path("models/backfill/two_stage/validation_results_forecast.npz")
    np.savez(results_path,
             violations_log=all_violations_log,
             violations_iv=all_violations_iv,
             pred_std=np.array(all_pred_std),
             sample_std=np.array(all_sample_std),
             gt_std=gt_std)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
