"""
Spatial Feature Preservation During Autoregressive Chaining

Goal: Verify that spatial features (volatility smile, term structure, cross-grid
correlations) are preserved before and after autoregressive chaining.

Current chaining experiments only measure ATM (2,2) metrics - this script
validates the full 5x5 grid structure.

Metrics computed per hop:
1. Volatility smile: Curvature and amplitude across moneyness
2. Term structure: Slope and curvature across maturities
3. Cross-grid correlation: 25x25 correlation matrix (Frobenius norm vs GT)
4. Per-grid kurtosis: Kurtosis at all 25 grid points

Usage:
    python experiments/backfill/two_stage_vae/exp_spatial_preservation_chaining.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP


# =============================================================================
# Spatial Metrics Functions
# =============================================================================

def compute_smile_metrics(surfaces):
    """
    Compute volatility smile metrics across samples.

    Smile is across moneyness (columns): 0.70, 0.85, 1.00, 1.15, 1.30

    Args:
        surfaces: (n_samples, H, 5, 5) or (H, 5, 5) log-returns

    Returns:
        curvature: Butterfly spread = wings - 2*ATM
        amplitude: Wings average - ATM
    """
    if surfaces.ndim == 4:
        mean_surf = surfaces.mean(axis=0)  # (H, 5, 5)
    else:
        mean_surf = surfaces  # (H, 5, 5)

    # Curvature per maturity row: OTM_put + OTM_call - 2*ATM
    # Columns: 0=0.70 (OTM put), 2=1.00 (ATM), 4=1.30 (OTM call)
    curvature = mean_surf[:, :, 0] + mean_surf[:, :, 4] - 2 * mean_surf[:, :, 2]

    # Amplitude: mean(wings) - ATM
    wings = (mean_surf[:, :, 0] + mean_surf[:, :, 4]) / 2
    atm = mean_surf[:, :, 2]
    amplitude = wings - atm

    return curvature.mean(), amplitude.mean()


def compute_term_structure_metrics(surfaces):
    """
    Compute term structure metrics across maturities.

    Term structure is across maturity (rows): 1M, 3M, 6M, 1Y, 2Y

    Args:
        surfaces: (n_samples, H, 5, 5) or (H, 5, 5) log-returns

    Returns:
        slope: Long tenor (2Y) - Short tenor (1M)
        curvature: (1M + 2Y) - 2*6M
    """
    if surfaces.ndim == 4:
        mean_surf = surfaces.mean(axis=0)  # (H, 5, 5)
    else:
        mean_surf = surfaces

    # Slope: 2Y (row 4) - 1M (row 0), averaged across moneyness
    slope = (mean_surf[:, 4, :] - mean_surf[:, 0, :]).mean()

    # Curvature: (1M + 2Y) - 2*6M
    curvature = (mean_surf[:, 0, :] + mean_surf[:, 4, :] - 2 * mean_surf[:, 2, :]).mean()

    return slope, curvature


def compute_cross_grid_correlation(samples):
    """
    Compute 25x25 correlation matrix from samples.

    Args:
        samples: (n_samples, H, 5, 5) log-returns

    Returns:
        corr: (25, 25) correlation matrix
    """
    if samples.ndim == 3:
        samples = samples[np.newaxis, ...]  # (1, H, 5, 5)

    B, H = samples.shape[:2]
    flat = samples.reshape(B * H, 25)  # (B*H, 25)

    # Handle case with too few samples
    if flat.shape[0] < 2:
        return np.eye(25)

    corr = np.corrcoef(flat.T)  # (25, 25)
    return corr


def compute_per_grid_kurtosis(samples):
    """
    Compute kurtosis at each grid point.

    Args:
        samples: (n_samples, H, 5, 5) log-returns

    Returns:
        kurt_grid: (5, 5) kurtosis values
    """
    if samples.ndim == 3:
        samples = samples[np.newaxis, ...]

    kurt_grid = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            cell_samples = samples[:, :, i, j].flatten()
            if len(cell_samples) > 3:
                kurt_grid[i, j] = kurtosis(cell_samples, fisher=True)
            else:
                kurt_grid[i, j] = 0.0

    return kurt_grid


def compute_smile_sign_match(gen_surfaces, gt_surfaces):
    """
    Compute how often generated smile has same sign as GT.

    Args:
        gen_surfaces: (n_samples, H, 5, 5)
        gt_surfaces: (H, 5, 5)

    Returns:
        sign_match_rate: Percentage of samples with matching curvature sign
    """
    if gen_surfaces.ndim == 3:
        gen_surfaces = gen_surfaces[np.newaxis, ...]

    matches = 0
    total = 0

    for s in range(gen_surfaces.shape[0]):
        for h in range(gen_surfaces.shape[1]):
            # Curvature at each maturity
            for m in range(5):  # maturity rows
                gen_curv = gen_surfaces[s, h, m, 0] + gen_surfaces[s, h, m, 4] - 2 * gen_surfaces[s, h, m, 2]
                gt_curv = gt_surfaces[h, m, 0] + gt_surfaces[h, m, 4] - 2 * gt_surfaces[h, m, 2]

                if np.sign(gen_curv) == np.sign(gt_curv):
                    matches += 1
                total += 1

    return (matches / total * 100) if total > 0 else 0.0


# =============================================================================
# Data Loading
# =============================================================================

def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_vae(device: str = "cuda"):
    """Load the Student-t VAE model."""
    vae_paths = [
        "models/backfill/two_stage/student_t/student_t_best.pt",
        "models/backfill/two_stage/student_t_acf/student_t_acf_lambda0.02.pt",
    ]

    for vae_path in vae_paths:
        if Path(vae_path).exists():
            print(f"Loading VAE from {vae_path}")
            vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
            config = vae_ckpt["model_config"]
            config["device"] = device

            vae = CVAETwoStageStudentTMLP(config)
            vae.load_state_dict(vae_ckpt["model_state_dict"])
            vae = vae.to(device)
            vae.eval()
            return vae, config

    raise FileNotFoundError("No VAE model found")


# =============================================================================
# Generation Functions
# =============================================================================

def generate_single_hop(vae, context_log_returns, horizon, n_samples, device):
    """
    Generate one hop of predictions using Oracle mode.

    Args:
        vae: Trained VAE model
        context_log_returns: (context_len, 5, 5) log-return context
        horizon: Number of days to generate
        n_samples: Number of samples per day

    Returns:
        generated: (n_samples, horizon, 5, 5) generated log-returns
    """
    context_tensor = torch.tensor(context_log_returns, dtype=torch.float32).unsqueeze(0).to(device)

    generated = np.zeros((n_samples, horizon, 5, 5))

    with torch.no_grad():
        for h in range(horizon):
            if h == 0:
                batch = {"surface": context_tensor}
            else:
                gen_so_far = torch.tensor(
                    generated[:, :h].mean(axis=0),
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                seq = torch.cat([context_tensor, gen_so_far], dim=1)
                batch = {"surface": seq}

            samples = vae.sample(batch, n_samples=n_samples)
            generated[:, h] = samples[:, 0, -1].cpu().numpy()

    return generated


def run_chained_generation(
    vae, initial_context, gt_log_returns,
    n_hops=4, horizon=30, overlap=15, n_samples=50, device="cuda"
):
    """
    Run chained generation and collect spatial metrics per hop.
    """
    context_len = initial_context.shape[0]

    print(f"\nGenerating {n_hops} hops × {horizon} days with {overlap}-day overlap")

    current_context = initial_context.copy()
    real_days_in_context = context_len

    # Compute GT spatial metrics (from full horizon range)
    total_days = n_hops * (horizon - overlap) + overlap
    gt_full = gt_log_returns[:total_days]

    gt_smile_curv, gt_smile_amp = compute_smile_metrics(gt_full)
    gt_term_slope, gt_term_curv = compute_term_structure_metrics(gt_full)
    gt_corr = compute_cross_grid_correlation(gt_full[np.newaxis, ...])
    gt_kurt = compute_per_grid_kurtosis(gt_full[np.newaxis, ...])

    print(f"\nGT Spatial Metrics (full {total_days} days):")
    print(f"  Smile curvature: {gt_smile_curv:.6f}")
    print(f"  Smile amplitude: {gt_smile_amp:.6f}")
    print(f"  Term slope: {gt_term_slope:.6f}")
    print(f"  Term curvature: {gt_term_curv:.6f}")
    print(f"  Mean kurtosis: {gt_kurt.mean():.2f}")

    results = []

    for hop in range(n_hops):
        print(f"\n--- Hop {hop + 1}/{n_hops} ---")
        print(f"  Context: {real_days_in_context} real + {context_len - real_days_in_context} generated")

        # Generate this hop
        generated = generate_single_hop(vae, current_context, horizon, n_samples, device)

        # Get GT for this hop
        start_idx = hop * (horizon - overlap)
        end_idx = start_idx + horizon
        gt_hop = gt_log_returns[start_idx:end_idx]

        # Compute spatial metrics for generated samples
        gen_smile_curv, gen_smile_amp = compute_smile_metrics(generated)
        gen_term_slope, gen_term_curv = compute_term_structure_metrics(generated)
        gen_corr = compute_cross_grid_correlation(generated)
        gen_kurt = compute_per_grid_kurtosis(generated)

        # Compute GT metrics for this specific hop
        hop_gt_smile_curv, hop_gt_smile_amp = compute_smile_metrics(gt_hop)
        hop_gt_term_slope, hop_gt_term_curv = compute_term_structure_metrics(gt_hop)
        hop_gt_corr = compute_cross_grid_correlation(gt_hop[np.newaxis, ...])
        hop_gt_kurt = compute_per_grid_kurtosis(gt_hop[np.newaxis, ...])

        # Compute errors
        smile_curv_error = abs(gen_smile_curv - hop_gt_smile_curv) / (abs(hop_gt_smile_curv) + 1e-8) * 100
        smile_amp_error = abs(gen_smile_amp - hop_gt_smile_amp) / (abs(hop_gt_smile_amp) + 1e-8) * 100
        term_slope_error = abs(gen_term_slope - hop_gt_term_slope) / (abs(hop_gt_term_slope) + 1e-8) * 100
        term_curv_error = abs(gen_term_curv - hop_gt_term_curv) / (abs(hop_gt_term_curv) + 1e-8) * 100

        # Correlation Frobenius norm
        corr_frobenius = np.linalg.norm(gen_corr - hop_gt_corr, 'fro')

        # Kurtosis MAE across grid
        kurt_mae = np.abs(gen_kurt - hop_gt_kurt).mean()

        # Smile sign match rate
        sign_match = compute_smile_sign_match(generated, gt_hop)

        # RMSE for reference
        mean_pred = generated.mean(axis=0)
        rmse = np.sqrt(np.mean((mean_pred - gt_hop) ** 2))

        result = {
            'hop': hop + 1,
            'real_days': real_days_in_context,
            # Smile metrics
            'gen_smile_curv': gen_smile_curv,
            'gt_smile_curv': hop_gt_smile_curv,
            'smile_curv_error_%': smile_curv_error,
            'gen_smile_amp': gen_smile_amp,
            'gt_smile_amp': hop_gt_smile_amp,
            'smile_amp_error_%': smile_amp_error,
            'smile_sign_match_%': sign_match,
            # Term structure metrics
            'gen_term_slope': gen_term_slope,
            'gt_term_slope': hop_gt_term_slope,
            'term_slope_error_%': term_slope_error,
            'gen_term_curv': gen_term_curv,
            'gt_term_curv': hop_gt_term_curv,
            'term_curv_error_%': term_curv_error,
            # Correlation
            'corr_frobenius': corr_frobenius,
            # Kurtosis
            'gen_kurt_mean': gen_kurt.mean(),
            'gt_kurt_mean': hop_gt_kurt.mean(),
            'kurt_mae': kurt_mae,
            # Overall
            'rmse': rmse,
        }

        results.append(result)

        print(f"  Smile curv error: {smile_curv_error:.1f}%")
        print(f"  Smile amp error: {smile_amp_error:.1f}%")
        print(f"  Smile sign match: {sign_match:.1f}%")
        print(f"  Term slope error: {term_slope_error:.1f}%")
        print(f"  Corr Frobenius: {corr_frobenius:.3f}")
        print(f"  Kurtosis MAE: {kurt_mae:.2f}")
        print(f"  RMSE: {rmse:.4f}")

        # Update context for next hop
        if hop < n_hops - 1:
            mean_generated = generated.mean(axis=0)
            new_context = np.concatenate([
                current_context[overlap:],
                mean_generated[:overlap]
            ], axis=0)

            current_context = new_context
            real_days_in_context = max(0, real_days_in_context - (horizon - overlap))

    return pd.DataFrame(results), gt_corr, gt_kurt


# =============================================================================
# Visualization
# =============================================================================

def plot_spatial_preservation(df, gt_corr, gt_kurt, save_dir):
    """Create visualization plots for spatial preservation analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    hops = df['hop'].values

    # Plot 1: Smile curvature error
    ax1 = axes[0, 0]
    ax1.plot(hops, df['smile_curv_error_%'], 'b-o', markersize=8)
    ax1.set_xlabel('Hop')
    ax1.set_ylabel('Error (%)')
    ax1.set_title('Smile Curvature Error')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Smile amplitude error
    ax2 = axes[0, 1]
    ax2.plot(hops, df['smile_amp_error_%'], 'r-o', markersize=8)
    ax2.set_xlabel('Hop')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Smile Amplitude Error')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Term structure errors
    ax3 = axes[0, 2]
    ax3.plot(hops, df['term_slope_error_%'], 'g-o', markersize=8, label='Slope')
    ax3.plot(hops, df['term_curv_error_%'], 'm-s', markersize=8, label='Curvature')
    ax3.set_xlabel('Hop')
    ax3.set_ylabel('Error (%)')
    ax3.set_title('Term Structure Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Correlation Frobenius norm
    ax4 = axes[1, 0]
    ax4.plot(hops, df['corr_frobenius'], 'c-o', markersize=8)
    ax4.set_xlabel('Hop')
    ax4.set_ylabel('Frobenius Norm')
    ax4.set_title('Cross-Grid Correlation Error')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Kurtosis MAE
    ax5 = axes[1, 1]
    ax5.plot(hops, df['kurt_mae'], 'orange', marker='o', markersize=8)
    ax5.set_xlabel('Hop')
    ax5.set_ylabel('MAE')
    ax5.set_title('Per-Grid Kurtosis MAE')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Real days in context
    ax6 = axes[1, 2]
    ax6.bar(hops, df['real_days'], color='green', alpha=0.7)
    ax6.set_xlabel('Hop')
    ax6.set_ylabel('Days')
    ax6.set_title('Real Data in Context')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "spatial_preservation_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'spatial_preservation_summary.png'}")
    plt.close()

    # Correlation heatmap (GT)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(gt_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Ground Truth Cross-Grid Correlation (25x25)')
    ax.set_xlabel('Grid Point')
    ax.set_ylabel('Grid Point')
    plt.colorbar(im, ax=ax)
    plt.savefig(save_dir / "gt_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'gt_correlation_heatmap.png'}")
    plt.close()

    # Kurtosis heatmap (GT)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(gt_kurt, cmap='viridis')
    ax.set_title('Ground Truth Per-Grid Kurtosis (5x5)')
    ax.set_xlabel('Moneyness (0.70 → 1.30)')
    ax.set_ylabel('Maturity (1M → 2Y)')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['0.70', '0.85', '1.00', '1.15', '1.30'])
    ax.set_yticks(range(5))
    ax.set_yticklabels(['1M', '3M', '6M', '1Y', '2Y'])
    plt.colorbar(im, ax=ax, label='Excess Kurtosis')

    # Add values
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{gt_kurt[i, j]:.1f}', ha='center', va='center', color='white', fontsize=9)

    plt.savefig(save_dir / "gt_kurtosis_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'gt_kurtosis_heatmap.png'}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run spatial preservation analysis during chaining."""
    print("=" * 70)
    print("Spatial Feature Preservation During Autoregressive Chaining")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    vae, config = load_vae(device)

    # Configuration
    context_len = config.get("context_len", 30)
    horizon = 30
    overlap = 15
    n_hops = 4
    n_samples = 50

    print(f"\nConfiguration:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon per hop: {horizon}")
    print(f"  Overlap: {overlap} days")
    print(f"  Number of hops: {n_hops}")
    print(f"  Samples: {n_samples}")

    # Pick starting point in validation set
    train_end = int(len(log_returns) * 0.7)
    start_idx = train_end + 100

    initial_context = log_returns[start_idx:start_idx + context_len]
    gt_log_returns = log_returns[start_idx + context_len:]

    print(f"\nStarting from index {start_idx}")

    # Run experiment
    df, gt_corr, gt_kurt = run_chained_generation(
        vae, initial_context, gt_log_returns,
        n_hops=n_hops, horizon=horizon, overlap=overlap,
        n_samples=n_samples, device=device
    )

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: SPATIAL FEATURE PRESERVATION")
    print("=" * 70)

    print("\n" + df.to_string(index=False))

    # Key comparison: Hop 1 vs Hop 3+ (before vs after)
    print("\n" + "-" * 70)
    print("BEFORE (Hop 1, 30 real days) vs AFTER (Hop 3+, 0 real days)")
    print("-" * 70)

    hop1 = df[df['hop'] == 1].iloc[0]
    hop3 = df[df['hop'] == 3].iloc[0] if len(df) >= 3 else None

    metrics = ['smile_curv_error_%', 'smile_amp_error_%', 'term_slope_error_%',
               'corr_frobenius', 'kurt_mae', 'rmse']

    print(f"\n{'Metric':<25} | {'Hop 1':>12} | {'Hop 3':>12} | {'Degradation':>12}")
    print("-" * 65)

    for m in metrics:
        v1 = hop1[m]
        v3 = hop3[m] if hop3 is not None else None
        if v3 is not None:
            if '%' in m:
                deg = f"{v3 - v1:+.1f}pp"
            else:
                deg = f"{(v3/v1 - 1)*100:+.1f}%"
            print(f"{m:<25} | {v1:>12.2f} | {v3:>12.2f} | {deg:>12}")
        else:
            print(f"{m:<25} | {v1:>12.2f} | {'N/A':>12} | {'N/A':>12}")

    # Save results
    save_dir = Path("results/two_stage_vae/spatial_preservation")
    save_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_dir / "spatial_metrics_per_hop.csv", index=False)
    print(f"\nResults saved to: {save_dir / 'spatial_metrics_per_hop.csv'}")

    # Generate plots
    plot_spatial_preservation(df, gt_corr, gt_kurt, save_dir)

    # Save GT data
    np.savez(
        save_dir / "gt_spatial_data.npz",
        gt_corr=gt_corr,
        gt_kurt=gt_kurt,
    )

    return df


if __name__ == "__main__":
    main()
