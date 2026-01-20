#!/usr/bin/env python
"""
Diagnose Posterior Collapse in Causal 3D VAE.

This script verifies if the poor CI coverage (33% vs 90%) is due to posterior collapse.

Hypothesis: The KL weight (1e-6) is too small, causing the encoder to output
near-zero variance (logvar → -30), making all samples collapse to the mean.

Tests:
1. Posterior Statistics: Check logvar values (should be ~0, not -30)
2. Sample Diversity: Check variance across 100 samples (should be >0)
3. Decoder Sensitivity: Check if decoder responds to z perturbations
4. KL Contribution: Check actual KL values

Usage:
    python experiments/backfill/two_stage_vae/diagnose_causal_3d_posterior.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.causal_3d_vae import create_vol_surface_vae_small, AutoencoderCausal3D


def load_model(model_path: str, device: str = "cuda") -> AutoencoderCausal3D:
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {})

    model = create_vol_surface_vae_small(
        latent_channels=model_config.get("latent_channels", 4),
        temporal_compression=model_config.get("temporal_compression", 2),
        block_channels=model_config.get("block_channels", (8, 16, 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def diagnose_posterior_statistics(model, data_tensor, device):
    """
    Test 1: Check posterior variance (logvar) statistics.

    If collapsed: logvar ≈ -30, std ≈ 1e-13
    If healthy: logvar ≈ 0, std ≈ 1
    """
    print("\n" + "=" * 60)
    print("TEST 1: POSTERIOR STATISTICS")
    print("=" * 60)

    all_logvar = []
    all_std = []
    all_mean = []

    with torch.no_grad():
        for i in tqdm(range(min(50, len(data_tensor))), desc="Encoding sequences"):
            sequence = data_tensor[i:i+1].to(device)
            posterior = model.encode(sequence, return_dict=True)

            all_logvar.append(posterior.logvar.cpu().numpy())
            all_std.append(posterior.std.cpu().numpy())
            all_mean.append(posterior.mean.cpu().numpy())

    all_logvar = np.concatenate(all_logvar)
    all_std = np.concatenate(all_std)
    all_mean = np.concatenate(all_mean)

    print(f"\nLogvar Statistics:")
    print(f"  Mean: {all_logvar.mean():.4f}")
    print(f"  Std:  {all_logvar.std():.4f}")
    print(f"  Min:  {all_logvar.min():.4f}")
    print(f"  Max:  {all_logvar.max():.4f}")

    print(f"\nPosterior Std Statistics:")
    print(f"  Mean: {all_std.mean():.6f}")
    print(f"  Min:  {all_std.min():.6f}")
    print(f"  Max:  {all_std.max():.6f}")

    # Diagnosis
    if all_logvar.mean() < -10:
        print(f"\n[COLLAPSED] logvar mean ({all_logvar.mean():.2f}) << 0")
        print(f"  → Posterior variance is essentially zero")
        print(f"  → All samples will be identical")
    elif all_logvar.mean() < -2:
        print(f"\n[WARNING] logvar mean ({all_logvar.mean():.2f}) is quite negative")
        print(f"  → Posterior variance is smaller than prior")
    else:
        print(f"\n[HEALTHY] logvar mean ({all_logvar.mean():.2f}) is reasonable")

    return {
        "logvar_mean": float(all_logvar.mean()),
        "logvar_std": float(all_logvar.std()),
        "logvar_min": float(all_logvar.min()),
        "logvar_max": float(all_logvar.max()),
        "posterior_std_mean": float(all_std.mean()),
    }


def diagnose_sample_diversity(model, data_tensor, device, num_samples=100):
    """
    Test 2: Check if multiple samples from the same posterior are diverse.

    If collapsed: all samples are identical (std ≈ 0)
    If healthy: samples vary (std > 0)
    """
    print("\n" + "=" * 60)
    print("TEST 2: SAMPLE DIVERSITY")
    print("=" * 60)

    # Use a single sequence
    sequence = data_tensor[0:1].to(device)

    with torch.no_grad():
        posterior = model.encode(sequence, return_dict=True)

        # Sample multiple times
        z_samples = []
        output_samples = []

        for _ in tqdm(range(num_samples), desc="Sampling"):
            z = posterior.sample()
            z_samples.append(z.cpu().numpy())

            # Decode
            output = model.decode(z * model.scaling_factor)
            output_samples.append(output.cpu().numpy())

    z_samples = np.array(z_samples)  # (num_samples, ...)
    output_samples = np.array(output_samples)  # (num_samples, ...)

    # Compute variance across samples
    z_std = z_samples.std(axis=0)
    output_std = output_samples.std(axis=0)

    print(f"\nLatent z diversity (std across {num_samples} samples):")
    print(f"  Mean std: {z_std.mean():.8f}")
    print(f"  Max std:  {z_std.max():.8f}")

    print(f"\nOutput diversity (std across {num_samples} samples):")
    print(f"  Mean std: {output_std.mean():.8f}")
    print(f"  Max std:  {output_std.max():.8f}")

    # Diagnosis
    if z_std.mean() < 1e-6:
        print(f"\n[COLLAPSED] z std ({z_std.mean():.2e}) ≈ 0")
        print(f"  → All 100 z samples are IDENTICAL")
        print(f"  → No variance in latent space")
    else:
        print(f"\n[HEALTHY] z std ({z_std.mean():.4f}) > 0")

    if output_std.mean() < 1e-6:
        print(f"\n[COLLAPSED] output std ({output_std.mean():.2e}) ≈ 0")
        print(f"  → All 100 output samples are IDENTICAL")
        print(f"  → CIs will have zero width → 0% coverage")
    else:
        print(f"\n[HEALTHY] output std ({output_std.mean():.6f}) > 0")

    return {
        "z_std_mean": float(z_std.mean()),
        "output_std_mean": float(output_std.mean()),
    }


def diagnose_decoder_sensitivity(model, data_tensor, device):
    """
    Test 3: Check if decoder responds to z perturbations.

    If decoder ignores z: perturbing z has no effect
    If decoder uses z: perturbing z changes output
    """
    print("\n" + "=" * 60)
    print("TEST 3: DECODER SENSITIVITY TO Z")
    print("=" * 60)

    sequence = data_tensor[0:1].to(device)

    with torch.no_grad():
        posterior = model.encode(sequence, return_dict=True)
        z_mean = posterior.mean * model.scaling_factor

        # Decode with mean z
        output_mean = model.decode(z_mean)

        # Perturbation levels
        perturbation_levels = [0.01, 0.1, 0.5, 1.0, 2.0]

        print(f"\nDecoder response to z perturbations:")
        print(f"{'Perturbation σ':<15} {'Output MAE':<15} {'Relative Change':<15}")
        print("-" * 45)

        for sigma in perturbation_levels:
            # Add Gaussian noise to z
            noise = torch.randn_like(z_mean) * sigma
            z_perturbed = z_mean + noise

            output_perturbed = model.decode(z_perturbed)

            # Measure change
            mae = (output_perturbed - output_mean).abs().mean().item()
            rel_change = mae / (output_mean.abs().mean().item() + 1e-8)

            print(f"{sigma:<15.2f} {mae:<15.6f} {rel_change:<15.4f}")

    # Also test: does decoder differ from ignoring z entirely?
    print(f"\nTest: Decode with z=0 vs z=mean:")
    with torch.no_grad():
        z_zero = torch.zeros_like(z_mean)
        output_zero = model.decode(z_zero)

        diff_zero_vs_mean = (output_mean - output_zero).abs().mean().item()
        print(f"  MAE(z=mean, z=0): {diff_zero_vs_mean:.6f}")

        if diff_zero_vs_mean < 1e-4:
            print(f"\n[WARNING] Decoder ignores z (z=0 ≈ z=mean)")
        else:
            print(f"\n[HEALTHY] Decoder uses z (z=0 ≠ z=mean)")


def diagnose_kl_contribution(model, data_tensor, device):
    """
    Test 4: Check KL divergence values.

    If KL is very small: posterior ≈ prior (N(0,1))
    """
    print("\n" + "=" * 60)
    print("TEST 4: KL DIVERGENCE")
    print("=" * 60)

    all_kl = []

    with torch.no_grad():
        for i in tqdm(range(min(50, len(data_tensor))), desc="Computing KL"):
            sequence = data_tensor[i:i+1].to(device)
            posterior = model.encode(sequence, return_dict=True)

            kl = posterior.kl()
            all_kl.append(kl.sum().item())

    all_kl = np.array(all_kl)

    print(f"\nKL Divergence Statistics:")
    print(f"  Mean: {all_kl.mean():.4f}")
    print(f"  Std:  {all_kl.std():.4f}")
    print(f"  Min:  {all_kl.min():.4f}")
    print(f"  Max:  {all_kl.max():.4f}")

    # With kl_weight=1e-6, actual loss contribution
    kl_weight = 1e-6
    print(f"\nWith kl_weight={kl_weight}:")
    print(f"  KL loss contribution: {kl_weight * all_kl.mean():.8f}")
    print(f"  (Compare to reconstruction loss ~0.003-0.01)")

    if kl_weight * all_kl.mean() < 1e-5:
        print(f"\n[ISSUE] KL contributes <1e-5 to loss → effectively ignored")


def main():
    print("=" * 60)
    print("CAUSAL 3D VAE POSTERIOR COLLAPSE DIAGNOSTIC")
    print("=" * 60)

    model_path = "models/backfill/causal_3d/best_model.pt"
    data_path = "data/vol_surface_with_ret.npz"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path, device)

    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)

    # Create sequences (context + horizon = 120)
    context_length = 60
    horizon = 60
    total_length = context_length + horizon

    # Create tensor of sequences
    sequences = []
    for i in range(0, len(surfaces) - total_length, 100):  # Sample every 100
        seq = surfaces[i:i+total_length]
        sequences.append(seq)
    sequences = np.array(sequences)
    print(f"Created {len(sequences)} sequences of length {total_length}")

    # Convert to tensor: (B, 1, T, H, W)
    data_tensor = torch.from_numpy(sequences).unsqueeze(1)
    print(f"Data tensor shape: {data_tensor.shape}")

    # Run diagnostics
    results = {}

    # Test 1: Posterior statistics
    results["posterior"] = diagnose_posterior_statistics(model, data_tensor, device)

    # Test 2: Sample diversity
    results["diversity"] = diagnose_sample_diversity(model, data_tensor, device)

    # Test 3: Decoder sensitivity
    diagnose_decoder_sensitivity(model, data_tensor, device)

    # Test 4: KL contribution
    diagnose_kl_contribution(model, data_tensor, device)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    collapsed = False

    if results["posterior"]["logvar_mean"] < -10:
        print(f"[COLLAPSED] Logvar mean = {results['posterior']['logvar_mean']:.2f} (should be ~0)")
        collapsed = True

    if results["diversity"]["z_std_mean"] < 1e-6:
        print(f"[COLLAPSED] z sample std = {results['diversity']['z_std_mean']:.2e} (should be ~1)")
        collapsed = True

    if results["diversity"]["output_std_mean"] < 1e-6:
        print(f"[COLLAPSED] Output sample std = {results['diversity']['output_std_mean']:.2e} (should be >0)")
        collapsed = True

    if collapsed:
        print("\n" + "=" * 60)
        print("CONCLUSION: POSTERIOR COLLAPSE CONFIRMED")
        print("=" * 60)
        print("\nRoot cause: KL weight (1e-6) is too small")
        print("The encoder outputs logvar → -30, causing std → 0")
        print("All samples collapse to the mean → no variance → poor coverage")
        print("\nRecommended fixes:")
        print("  1. Increase KL weight to 1e-3 or higher")
        print("  2. Tighten logvar clamp from [-30, 20] to [-5, 5]")
        print("  3. Add KL annealing (start low, increase during training)")
    else:
        print("\n" + "=" * 60)
        print("CONCLUSION: POSTERIOR IS HEALTHY")
        print("=" * 60)
        print("Look for other causes of poor coverage")


if __name__ == "__main__":
    main()
