"""
Validate Two-Stage CVAE with Multivariate Student-t Decoder.

This script validates the trained CVAETwoStageStudentT model by:
1. Generating samples under posterior mode (z conditioned on target)
2. Computing empirical kurtosis from samples (main metric for fat tails)
3. Computing empirical cross-grid correlation from samples
4. Comparing to baseline (Full Cov) and ground truth statistics

Key Metrics (from TWO_STAGE_VAE_CRITIQUE.md):
- Kurtosis (ATM, H=15): GT 21.3, Full Cov 1.4, Target ~8-15
- Cross-grid correlation: GT 0.21-0.42, Full Cov 0.18, Target 0.20-0.40
- Skewness (ATM): GT +1.6, Full Cov -0.26 (asymmetry still not fixed)

Usage:
    python experiments/backfill/two_stage_vae/validate_student_t.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageStudentT


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
    print("Validating Student-t Decoder")
    print("=" * 70)
    print()
    print("Testing if Student-t decoder produces fat tails AND correlations.")
    print()

    # Load checkpoint (use fixed-nu version which has heterogeneous nu from GT kurtosis)
    checkpoint_path = Path("models/backfill/two_stage/two_stage_student_t_fixed_nu_best.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Run training first: python experiments/backfill/two_stage_vae/train_two_stage_student_t.py")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint["model_config"]
    learned_nu = checkpoint.get("nu", None)

    # Build model
    model = CVAETwoStageStudentT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = config["device"]
    print(f"  Model loaded, device: {device}")
    if learned_nu:
        print(f"  Learned nu from training: {learned_nu:.2f}")

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
    # 1. Compute Ground Truth Statistics
    # =========================================================================
    print("\n" + "=" * 50)
    print("1. Ground Truth Statistics")
    print("=" * 50)

    # Get all horizon log-returns from GT
    gt_horizon = eval_sequences[:, context_len:].numpy()  # (N, H, 5, 5)
    gt_flat = gt_horizon.reshape(-1, 25)  # (N*H, 25)

    # Kurtosis (excess kurtosis, Gaussian = 0)
    gt_kurtosis_all = stats.kurtosis(gt_flat, axis=0).reshape(5, 5)
    gt_kurtosis_atm = gt_kurtosis_all[2, 2]  # ATM point
    gt_kurtosis_mean = gt_kurtosis_all.mean()

    # Skewness
    gt_skewness_all = stats.skew(gt_flat, axis=0).reshape(5, 5)
    gt_skewness_atm = gt_skewness_all[2, 2]

    # Correlation
    gt_corr = compute_correlation_matrix(gt_flat)
    gt_atm_otm = gt_corr[12, 0]   # ATM (2,2)=12 vs OTM-short (0,0)=0
    gt_atm_itm = gt_corr[12, 24]  # ATM (2,2)=12 vs ITM-long (4,4)=24
    gt_mean_corr = (np.abs(gt_corr).sum() - 25) / (25*24)

    print(f"  Kurtosis (ATM, excess): {gt_kurtosis_atm:.2f}")
    print(f"  Kurtosis (mean across grid): {gt_kurtosis_mean:.2f}")
    print(f"  Skewness (ATM): {gt_skewness_atm:.2f}")
    print(f"  ATM↔OTM-short correlation: {gt_atm_otm:.3f}")
    print(f"  ATM↔ITM-long correlation:  {gt_atm_itm:.3f}")
    print(f"  Mean |correlation|:        {gt_mean_corr:.3f}")

    # =========================================================================
    # 2. Generate Samples from Student-t Model
    # =========================================================================
    print("\n" + "=" * 50)
    print("2. Student-t Model Sampling")
    print("=" * 50)

    n_samples_per_seq = 100
    all_samples = []

    print(f"  Generating {n_samples_per_seq} samples per sequence...")

    with torch.no_grad():
        for i in tqdm(range(n_eval), desc="Sampling"):
            seq = eval_sequences[i:i+1].to(device)  # (1, T, 5, 5)

            # Forward to get mean, L, and nu
            mean, L, nu, z_mean, z_logvar, z = model.forward({"surface": seq}, return_full_sequence=True)

            # Get horizon predictions
            mean_h = mean[:, context_len:]  # (1, H, 5, 5)
            L_h = L[:, context_len:]        # (1, H, 25, 25)

            # Generate multiple samples using Student-t
            for _ in range(n_samples_per_seq):
                sample = model.sample_from_decoder(mean_h, L_h, nu)  # (1, H, 5, 5)
                all_samples.append(sample.cpu().numpy())

    all_samples = np.array(all_samples)  # (N*n_samples, H, 5, 5)
    all_samples_flat = all_samples.reshape(-1, 25)  # Flatten for statistics
    print(f"  Total samples: {len(all_samples_flat)}")

    # Per-grid-point nu (25 values)
    nu_grid = nu.cpu().numpy().reshape(5, 5)
    print(f"  Per-grid-point nu:")
    print(f"    Mean: {nu.mean().item():.2f}")
    print(f"    Min:  {nu.min().item():.2f}")
    print(f"    Max:  {nu.max().item():.2f}")
    print(f"    Std:  {nu.std().item():.4f}")

    # Compute VAE kurtosis
    vae_kurtosis_all = stats.kurtosis(all_samples_flat, axis=0).reshape(5, 5)
    vae_kurtosis_atm = vae_kurtosis_all[2, 2]
    vae_kurtosis_mean = vae_kurtosis_all.mean()

    # Compute VAE skewness
    vae_skewness_all = stats.skew(all_samples_flat, axis=0).reshape(5, 5)
    vae_skewness_atm = vae_skewness_all[2, 2]

    # Compute VAE correlation
    vae_corr = compute_correlation_matrix(all_samples_flat)
    vae_atm_otm = vae_corr[12, 0]
    vae_atm_itm = vae_corr[12, 24]
    vae_mean_corr = (np.abs(vae_corr).sum() - 25) / (25*24)
    corr_error = np.sqrt(((vae_corr - gt_corr)**2).sum())

    print(f"\n  VAE Student-t Results:")
    print(f"    Kurtosis (ATM, excess): {vae_kurtosis_atm:.2f} (GT: {gt_kurtosis_atm:.2f})")
    print(f"    Kurtosis (mean):        {vae_kurtosis_mean:.2f} (GT: {gt_kurtosis_mean:.2f})")
    print(f"    Skewness (ATM):         {vae_skewness_atm:.2f} (GT: {gt_skewness_atm:.2f})")
    print(f"    ATM↔OTM-short correlation: {vae_atm_otm:.3f} (GT: {gt_atm_otm:.3f})")
    print(f"    ATM↔ITM-long correlation:  {vae_atm_itm:.3f} (GT: {gt_atm_itm:.3f})")
    print(f"    Mean |correlation|:        {vae_mean_corr:.3f} (GT: {gt_mean_corr:.3f})")
    print(f"    Correlation Frobenius error: {corr_error:.2f}")

    # =========================================================================
    # 3. Comparison with Full Covariance Baseline
    # =========================================================================
    print("\n" + "=" * 50)
    print("3. Comparison Summary")
    print("=" * 50)

    # Reference values from Full Covariance validation
    full_cov_kurtosis = 1.35  # From previous validation
    full_cov_atm_otm = 0.184
    full_cov_atm_itm = 0.174
    full_cov_corr_error = 6.22

    print(f"\n  Kurtosis Comparison (ATM, excess):")
    print(f"  {'Model':<20} {'Kurtosis':<12} {'% of GT':<12}")
    print(f"  {'-'*44}")
    print(f"  {'Gaussian (ref)':<20} {'0.00':<12} 0.0%")
    print(f"  {'Full Covariance':<20} {full_cov_kurtosis:<12.2f} {full_cov_kurtosis/gt_kurtosis_atm*100:.1f}%")
    print(f"  {'Student-t':<20} {vae_kurtosis_atm:<12.2f} {vae_kurtosis_atm/gt_kurtosis_atm*100:.1f}%")
    print(f"  {'Ground Truth':<20} {gt_kurtosis_atm:<12.2f} 100.0%")

    print(f"\n  Correlation Comparison:")
    print(f"  {'Metric':<30} {'Full Cov':<12} {'Student-t':<12} {'GT':<12} {'Improvement':<12}")
    print(f"  {'-'*78}")

    atm_otm_improvement = (vae_atm_otm - full_cov_atm_otm) / (gt_atm_otm - full_cov_atm_otm) * 100 if gt_atm_otm != full_cov_atm_otm else 0
    atm_itm_improvement = (vae_atm_itm - full_cov_atm_itm) / (gt_atm_itm - full_cov_atm_itm) * 100 if gt_atm_itm != full_cov_atm_itm else 0
    corr_improvement = (full_cov_corr_error - corr_error) / full_cov_corr_error * 100

    print(f"  {'ATM↔OTM correlation':<30} {full_cov_atm_otm:<12.3f} {vae_atm_otm:<12.3f} {gt_atm_otm:<12.3f} {atm_otm_improvement:+.1f}%")
    print(f"  {'ATM↔ITM correlation':<30} {full_cov_atm_itm:<12.3f} {vae_atm_itm:<12.3f} {gt_atm_itm:<12.3f} {atm_itm_improvement:+.1f}%")
    print(f"  {'Corr matrix Frobenius error':<30} {full_cov_corr_error:<12.2f} {corr_error:<12.2f} {'0':<12} {corr_improvement:+.1f}%")

    # =========================================================================
    # 4. Per-Grid-Point nu vs Kurtosis Analysis
    # =========================================================================
    print("\n" + "=" * 50)
    print("4. Per-Grid-Point nu vs Kurtosis Analysis")
    print("=" * 50)

    # Compute theoretical kurtosis from per-grid-point nu
    # For nu > 4: excess kurtosis = 6 / (nu - 4)
    theoretical_kurt_grid = np.where(nu_grid > 4, 6 / (nu_grid - 4), np.inf)

    print(f"\n  Learned nu grid (5x5):")
    print(np.array2string(nu_grid, precision=2, suppress_small=True))

    print(f"\n  Theoretical kurtosis from nu (for nu > 4):")
    print(np.array2string(theoretical_kurt_grid, precision=1, suppress_small=True))

    print(f"\n  Empirical VAE kurtosis (5x5):")
    print(np.array2string(vae_kurtosis_all, precision=1, suppress_small=True))

    print(f"\n  Ground Truth kurtosis (5x5):")
    print(np.array2string(gt_kurtosis_all, precision=1, suppress_small=True))

    # Check if nu learned heterogeneity
    print(f"\n  Nu Heterogeneity Analysis:")
    print(f"    nu range: {nu_grid.min():.2f} - {nu_grid.max():.2f}")
    print(f"    nu std:   {nu_grid.std():.4f}")
    print(f"    GT kurtosis range: {gt_kurtosis_all.min():.1f} - {gt_kurtosis_all.max():.1f}")

    # Correlation between nu and GT kurtosis (should be negative: low nu -> high kurtosis)
    nu_flat = nu_grid.flatten()
    gt_kurt_flat = gt_kurtosis_all.flatten()
    # Filter out infinite theoretical kurtosis
    valid_mask = nu_flat > 4
    if valid_mask.sum() > 0:
        corr_nu_vs_gt = np.corrcoef(nu_flat[valid_mask], gt_kurt_flat[valid_mask])[0, 1]
        print(f"    Correlation(nu, GT_kurtosis): {corr_nu_vs_gt:.3f} (expect negative)")
    else:
        print(f"    All nu <= 4, cannot compute correlation")

    # =========================================================================
    # 5. Tail Probability Analysis
    # =========================================================================
    print("\n" + "=" * 50)
    print("5. Tail Probability Analysis")
    print("=" * 50)

    gt_std = gt_flat.std()

    for threshold in [2.0, 2.5, 3.0]:
        gt_extreme = (np.abs(gt_flat) > threshold * gt_std).mean()
        vae_extreme = (np.abs(all_samples_flat) > threshold * gt_std).mean()
        gaussian_extreme = 2 * (1 - stats.norm.cdf(threshold))  # Two-tailed

        print(f"  {threshold}σ threshold:")
        print(f"    Gaussian (theoretical): {gaussian_extreme*100:.3f}%")
        print(f"    Ground Truth:           {gt_extreme*100:.3f}%")
        print(f"    Student-t VAE:          {vae_extreme*100:.3f}%")

    # =========================================================================
    # 6. Visualizations
    # =========================================================================
    print("\n" + "=" * 50)
    print("6. Generating Visualizations")
    print("=" * 50)

    viz_dir = Path("models/backfill/two_stage/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Kurtosis heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # GT kurtosis
    im0 = axes[0].imshow(gt_kurtosis_all, cmap='YlOrRd', vmin=0, vmax=50)
    axes[0].set_title(f'Ground Truth Kurtosis\n(mean: {gt_kurtosis_mean:.1f})')
    axes[0].set_xlabel('Tenor')
    axes[0].set_ylabel('Moneyness')
    plt.colorbar(im0, ax=axes[0])

    # VAE kurtosis
    im1 = axes[1].imshow(vae_kurtosis_all, cmap='YlOrRd', vmin=0, vmax=50)
    axes[1].set_title(f'Student-t VAE Kurtosis\n(mean: {vae_kurtosis_mean:.1f})')
    axes[1].set_xlabel('Tenor')
    axes[1].set_ylabel('Moneyness')
    plt.colorbar(im1, ax=axes[1])

    # Ratio
    ratio = vae_kurtosis_all / np.maximum(gt_kurtosis_all, 0.1)
    im2 = axes[2].imshow(ratio, cmap='RdYlGn', vmin=0, vmax=2)
    axes[2].set_title(f'VAE/GT Kurtosis Ratio\n(1.0 = perfect match)')
    axes[2].set_xlabel('Tenor')
    axes[2].set_ylabel('Moneyness')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    kurt_path = viz_dir / "student_t_kurtosis_comparison.png"
    plt.savefig(kurt_path, dpi=150)
    plt.close()
    print(f"  Saved: {kurt_path}")

    # Plot 2: Correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # GT correlation
    im0 = axes[0].imshow(gt_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Ground Truth Correlation')
    axes[0].set_xlabel('Grid Point')
    axes[0].set_ylabel('Grid Point')
    plt.colorbar(im0, ax=axes[0])

    # VAE correlation
    im1 = axes[1].imshow(vae_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Student-t VAE Correlation')
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
    corr_path = viz_dir / "student_t_correlation_comparison.png"
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"  Saved: {corr_path}")

    # Plot 3: Sample distribution (Q-Q plot style)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ATM histogram comparison
    ax = axes[0]
    gt_atm = gt_flat[:, 12]
    vae_atm = all_samples_flat[:, 12]

    bins = np.linspace(-0.15, 0.15, 50)
    ax.hist(gt_atm, bins=bins, alpha=0.5, density=True, label='Ground Truth')
    ax.hist(vae_atm, bins=bins, alpha=0.5, density=True, label='Student-t VAE')
    ax.set_xlabel('Log-Return (ATM)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison (ATM)')
    ax.legend()
    ax.set_xlim(-0.15, 0.15)

    # Q-Q plot
    ax = axes[1]
    sorted_gt = np.sort(gt_atm)
    sorted_vae = np.sort(vae_atm)
    # Resample to same size for comparison
    if len(sorted_vae) > len(sorted_gt):
        indices = np.linspace(0, len(sorted_vae)-1, len(sorted_gt)).astype(int)
        sorted_vae = sorted_vae[indices]
    else:
        indices = np.linspace(0, len(sorted_gt)-1, len(sorted_vae)).astype(int)
        sorted_gt = sorted_gt[indices]

    ax.scatter(sorted_gt, sorted_vae, alpha=0.1, s=1)
    lim = max(abs(sorted_gt.min()), abs(sorted_gt.max()), abs(sorted_vae.min()), abs(sorted_vae.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', label='y=x')
    ax.set_xlabel('Ground Truth Quantiles')
    ax.set_ylabel('Student-t VAE Quantiles')
    ax.set_title('Q-Q Plot (ATM)')
    ax.legend()

    plt.tight_layout()
    dist_path = viz_dir / "student_t_distribution_comparison.png"
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"  Saved: {dist_path}")

    # =========================================================================
    # 7. Save Results
    # =========================================================================
    print("\n" + "=" * 50)
    print("7. Saving Results")
    print("=" * 50)

    results_path = Path("models/backfill/two_stage/student_t_validation.npz")
    np.savez(results_path,
             gt_corr=gt_corr,
             vae_corr=vae_corr,
             gt_kurtosis=gt_kurtosis_all,
             vae_kurtosis=vae_kurtosis_all,
             gt_skewness=gt_skewness_all,
             vae_skewness=vae_skewness_all,
             corr_error=corr_error,
             # Per-grid-point nu (25 values as 5x5 grid)
             learned_nu_grid=nu_grid,
             learned_nu_mean=nu_grid.mean(),
             learned_nu_min=nu_grid.min(),
             learned_nu_max=nu_grid.max(),
             learned_nu_std=nu_grid.std(),
             theoretical_kurtosis_grid=theoretical_kurt_grid,
             gt_atm_otm=gt_atm_otm,
             gt_atm_itm=gt_atm_itm,
             vae_atm_otm=vae_atm_otm,
             vae_atm_itm=vae_atm_itm,
             gt_kurtosis_atm=gt_kurtosis_atm,
             vae_kurtosis_atm=vae_kurtosis_atm)
    print(f"  Saved: {results_path}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Assess success based on kurtosis and correlation improvements
    kurtosis_improved = vae_kurtosis_atm > full_cov_kurtosis * 2
    corr_improved = corr_error < full_cov_corr_error

    if kurtosis_improved and corr_improved:
        print("\n✓ STUDENT-T DECODER SUCCESSFULLY IMPROVES BOTH METRICS!")
        print(f"  Kurtosis (ATM): {full_cov_kurtosis:.2f} → {vae_kurtosis_atm:.2f} (GT: {gt_kurtosis_atm:.2f})")
        print(f"  Frobenius error: {full_cov_corr_error:.2f} → {corr_error:.2f}")
        print(f"  Learned nu: mean={nu_grid.mean():.2f}, range=[{nu_grid.min():.2f}, {nu_grid.max():.2f}]")
    elif kurtosis_improved:
        print("\n✓ KURTOSIS IMPROVED, but correlation needs work")
        print(f"  Kurtosis: {full_cov_kurtosis:.2f} → {vae_kurtosis_atm:.2f} (GT: {gt_kurtosis_atm:.2f})")
        print(f"  Try: Increase NLL weight further or train longer")
    elif corr_improved:
        print("\n✓ CORRELATION IMPROVED, but kurtosis needs work")
        print(f"  Frobenius error: {full_cov_corr_error:.2f} → {corr_error:.2f}")
        print(f"  nu may have collapsed to high value (approaching Gaussian)")
        print(f"  Try: Lower nu_max or add kurtosis regularization")
    else:
        print("\n✗ IMPROVEMENTS INSUFFICIENT")
        print(f"  Kurtosis: {vae_kurtosis_atm:.2f} (target: >{full_cov_kurtosis*2:.2f})")
        print(f"  Frobenius: {corr_error:.2f} (target: <{full_cov_corr_error:.2f})")

    # Note about skewness
    print("\n  Note: Skewness (asymmetry) NOT addressed by Student-t")
    print(f"  GT skewness: {gt_skewness_atm:.2f}, VAE: {vae_skewness_atm:.2f}")
    print("  Fix requires: Skew-Normal or Skew-t decoder")


if __name__ == "__main__":
    main()
