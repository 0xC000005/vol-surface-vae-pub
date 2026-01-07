"""
Validate Two-Stage CVAE with Full Covariance Decoder.

This script validates the trained CVAETwoStageFullCovariance model by:
1. Generating samples under posterior mode (z conditioned on target)
2. Computing empirical cross-grid correlation from samples
3. Comparing to diagonal decoder and ground truth correlations

Key Metrics:
- Cross-grid correlation: ATM↔OTM, ATM↔ITM
- Correlation matrix Frobenius error
- RMSE (mean prediction accuracy)
- Visual: Correlation heatmaps

Usage:
    python experiments/backfill/two_stage_vae/validate_full_covariance.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageFullCovariance


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


def compute_correlation_matrix(samples):
    """
    Compute empirical correlation matrix from samples.

    Args:
        samples: (N, 25) - flattened samples

    Returns:
        corr_matrix: (25, 25) correlation matrix
    """
    return np.corrcoef(samples.T)


def main():
    print("=" * 70)
    print("Validating Full Covariance Decoder")
    print("=" * 70)
    print()
    print("Testing if full covariance decoder preserves cross-grid correlations.")
    print()

    # Load checkpoint
    checkpoint_path = Path("models/backfill/two_stage/two_stage_full_covariance_best.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run training first: python experiments/backfill/two_stage_vae/train_two_stage_full_covariance.py")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint["model_config"]

    # Build model
    model = CVAETwoStageFullCovariance(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = config["device"]
    print(f"  Model loaded, device: {device}")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    # Transform to log-returns
    log_returns, log_surfaces = to_log_returns(surfaces)
    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)

    # Create sequences
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    all_sequences = create_sequences(log_returns_tensor, seq_len)

    # Use validation set
    n_train = int(len(all_sequences) * 0.8)
    val_sequences = all_sequences[n_train:]
    print(f"  Validation sequences: {len(val_sequences)}")

    # Sample a subset for analysis
    n_eval = min(500, len(val_sequences))
    eval_sequences = val_sequences[:n_eval]
    print(f"  Evaluation sequences: {n_eval}")

    # =========================================================================
    # 1. Compute Ground Truth Correlation
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. Ground Truth Correlations")
    print("=" * 50)

    # Get all horizon log-returns from GT
    gt_horizon = eval_sequences[:, context_len:].numpy()  # (N, H, 5, 5)
    gt_flat = gt_horizon.reshape(-1, 25)  # (N*H, 25)

    gt_corr = compute_correlation_matrix(gt_flat)
    gt_atm_otm = gt_corr[12, 0]   # ATM (2,2)=12 vs OTM-short (0,0)=0
    gt_atm_itm = gt_corr[12, 24]  # ATM (2,2)=12 vs ITM-long (4,4)=24
    gt_mean_corr = (np.abs(gt_corr).sum() - 25) / (25*24)  # Mean abs corr (excl diagonal)

    print(f"  ATM↔OTM-short correlation: {gt_atm_otm:.3f}")
    print(f"  ATM↔ITM-long correlation:  {gt_atm_itm:.3f}")
    print(f"  Mean |correlation|:        {gt_mean_corr:.3f}")

    # =========================================================================
    # 2. Generate Samples from Full Covariance Model
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. Full Covariance Model Sampling")
    print("=" * 50)

    n_samples_per_seq = 100  # Number of samples per sequence
    all_samples = []

    print(f"  Generating {n_samples_per_seq} samples per sequence...")

    with torch.no_grad():
        for i in tqdm(range(n_eval), desc="Sampling"):
            seq = eval_sequences[i:i+1].to(device)  # (1, T, 5, 5)

            # Forward to get mean and L
            mean, L, z_mean, z_logvar, z = model.forward({"surface": seq}, return_full_sequence=True)

            # Get horizon predictions
            mean_h = mean[:, context_len:]  # (1, H, 5, 5)
            L_h = L[:, context_len:]        # (1, H, 25, 25)

            # Generate multiple samples
            for _ in range(n_samples_per_seq):
                sample = model.sample_from_decoder(mean_h, L_h)  # (1, H, 5, 5)
                all_samples.append(sample.cpu().numpy())

    all_samples = np.array(all_samples)  # (N*n_samples, H, 5, 5)
    all_samples = all_samples.reshape(-1, 25)  # Flatten for correlation
    print(f"  Total samples: {len(all_samples)}")

    # Compute VAE correlation
    vae_corr = compute_correlation_matrix(all_samples)
    vae_atm_otm = vae_corr[12, 0]
    vae_atm_itm = vae_corr[12, 24]
    vae_mean_corr = (np.abs(vae_corr).sum() - 25) / (25*24)

    print(f"\n  VAE Full Covariance Results:")
    print(f"    ATM↔OTM-short correlation: {vae_atm_otm:.3f} (GT: {gt_atm_otm:.3f})")
    print(f"    ATM↔ITM-long correlation:  {vae_atm_itm:.3f} (GT: {gt_atm_itm:.3f})")
    print(f"    Mean |correlation|:        {vae_mean_corr:.3f} (GT: {gt_mean_corr:.3f})")

    # Correlation matrix error
    corr_error = np.sqrt(((vae_corr - gt_corr)**2).sum())
    print(f"    Correlation Frobenius error: {corr_error:.2f}")

    # =========================================================================
    # 3. Compare with Diagonal Baseline (if available)
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. Comparison Summary")
    print("=" * 50)

    # Reference values from TWO_STAGE_VAE_CRITIQUE.md
    diag_atm_otm = 0.02
    diag_atm_itm = 0.03
    diag_corr_error = 13.38

    print(f"\n  Correlation Comparison:")
    print(f"  {'Metric':<30} {'Diagonal':<12} {'Full Cov':<12} {'GT':<12} {'Improvement':<12}")
    print(f"  {'-'*78}")
    print(f"  {'ATM↔OTM correlation':<30} {diag_atm_otm:<12.3f} {vae_atm_otm:<12.3f} {gt_atm_otm:<12.3f} {(vae_atm_otm-diag_atm_otm)/(gt_atm_otm-diag_atm_otm)*100:+.1f}%")
    print(f"  {'ATM↔ITM correlation':<30} {diag_atm_itm:<12.3f} {vae_atm_itm:<12.3f} {gt_atm_itm:<12.3f} {(vae_atm_itm-diag_atm_itm)/(gt_atm_itm-diag_atm_itm)*100:+.1f}%")
    print(f"  {'Corr matrix Frobenius error':<30} {diag_corr_error:<12.2f} {corr_error:<12.2f} {'0':<12} {(diag_corr_error-corr_error)/diag_corr_error*100:+.1f}%")

    # =========================================================================
    # 4. Visualizations
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. Generating Visualizations")
    print("=" * 50)

    viz_dir = Path("models/backfill/two_stage/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Plot correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # GT correlation
    im0 = axes[0].imshow(gt_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Ground Truth Correlation')
    axes[0].set_xlabel('Grid Point')
    axes[0].set_ylabel('Grid Point')
    plt.colorbar(im0, ax=axes[0])

    # VAE Full Cov correlation
    im1 = axes[1].imshow(vae_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('VAE Full Covariance Correlation')
    axes[1].set_xlabel('Grid Point')
    axes[1].set_ylabel('Grid Point')
    plt.colorbar(im1, ax=axes[1])

    # Difference
    diff = vae_corr - gt_corr
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title(f'Difference (Frobenius: {corr_error:.2f})')
    axes[2].set_xlabel('Grid Point')
    axes[2].set_ylabel('Grid Point')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    corr_path = viz_dir / "full_covariance_correlation_comparison.png"
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"  Saved: {corr_path}")

    # =========================================================================
    # 5. Check Cholesky Structure
    # =========================================================================
    print("\n" + "=" * 50)
    print("5. Cholesky Factor Analysis")
    print("=" * 50)

    # Get L from a few samples
    with torch.no_grad():
        seq = eval_sequences[0:1].to(device)
        mean, L, _, _, _ = model.forward({"surface": seq}, return_full_sequence=True)
        L_h = L[:, context_len:]  # (1, H, 25, 25)

    # Analyze L structure
    L_sample = L_h[0, 0].cpu().numpy()  # First horizon step
    L_diag = np.diag(L_sample)
    L_offdiag = L_sample[np.tril_indices(25, k=-1)]

    print(f"  Diagonal elements:")
    print(f"    Mean: {L_diag.mean():.4f}")
    print(f"    Std:  {L_diag.std():.4f}")
    print(f"    Min:  {L_diag.min():.4f}")
    print(f"    Max:  {L_diag.max():.4f}")
    print(f"\n  Off-diagonal elements:")
    print(f"    Mean: {L_offdiag.mean():.4f}")
    print(f"    Std:  {L_offdiag.std():.4f}")
    print(f"    Mean |value|: {np.abs(L_offdiag).mean():.4f}")

    # Compute implied covariance
    Sigma = L_sample @ L_sample.T
    implied_corr = Sigma / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))

    print(f"\n  Implied correlation from L @ L.T:")
    print(f"    ATM↔OTM: {implied_corr[12, 0]:.3f}")
    print(f"    ATM↔ITM: {implied_corr[12, 24]:.3f}")

    # Plot L structure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(L_sample, cmap='RdBu_r')
    axes[0].set_title('Cholesky Factor L')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(implied_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Implied Correlation (L @ L.T)')
    axes[1].set_xlabel('Grid Point')
    axes[1].set_ylabel('Grid Point')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    chol_path = viz_dir / "full_covariance_cholesky_structure.png"
    plt.savefig(chol_path, dpi=150)
    plt.close()
    print(f"  Saved: {chol_path}")

    # =========================================================================
    # 6. Save Results
    # =========================================================================
    print("\n" + "=" * 50)
    print("6. Saving Results")
    print("=" * 50)

    results_path = Path("models/backfill/two_stage/full_covariance_validation.npz")
    np.savez(results_path,
             gt_corr=gt_corr,
             vae_corr=vae_corr,
             corr_error=corr_error,
             gt_atm_otm=gt_atm_otm,
             gt_atm_itm=gt_atm_itm,
             vae_atm_otm=vae_atm_otm,
             vae_atm_itm=vae_atm_itm,
             L_sample=L_sample,
             implied_corr=implied_corr)
    print(f"  Saved: {results_path}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    success = vae_atm_otm > 0.15 and vae_atm_itm > 0.25  # Reasonable thresholds

    if success:
        print("\n✓ FULL COVARIANCE DECODER SUCCESSFULLY PRESERVES CORRELATIONS!")
        print(f"  ATM↔OTM: {diag_atm_otm:.2f} → {vae_atm_otm:.2f} (target: {gt_atm_otm:.2f})")
        print(f"  ATM↔ITM: {diag_atm_itm:.2f} → {vae_atm_itm:.2f} (target: {gt_atm_itm:.2f})")
    else:
        print("\n✗ CORRELATION IMPROVEMENT INSUFFICIENT")
        print(f"  ATM↔OTM: {vae_atm_otm:.2f} (target: >{gt_atm_otm*0.5:.2f})")
        print(f"  ATM↔ITM: {vae_atm_itm:.2f} (target: >{gt_atm_itm*0.5:.2f})")
        print("\n  Possible issues:")
        print("    - Training not converged")
        print("    - NLL weight too low")
        print("    - Need more decoder capacity")


if __name__ == "__main__":
    main()
