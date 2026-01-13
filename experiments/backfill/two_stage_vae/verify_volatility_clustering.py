"""
Verify Volatility Clustering in GT vs Model

Check if volatility clustering (ACF of squared returns) actually exists in:
1. Ground Truth data
2. Model-generated paths (Oracle mode)
3. Model-generated paths (Prior mode)

Volatility clustering = positive autocorrelation in squared returns
- If GT has it, model should try to match it
- If GT doesn't have it, we shouldn't optimize for it

Usage:
    python experiments/backfill/two_stage_vae/verify_volatility_clustering.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictorCov


def compute_acf(series, max_lag=10):
    """Compute autocorrelation function for multiple lags."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-10:
        return np.zeros(max_lag)

    acf = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            acf.append(0)
        else:
            cov = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
            acf.append(cov / var)
    return np.array(acf)


def compute_acf_squared(series, max_lag=10):
    """Compute ACF of squared series (volatility clustering measure)."""
    return compute_acf(series ** 2, max_lag)


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_models(device):
    """Load VAE and predictor."""
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = str(device)

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load AR(1) predictor
    ar1_path = "models/backfill/two_stage/prior_network_ar1/prior_network_ar1_best.pt"
    ar1_ckpt = torch.load(ar1_path, map_location=device, weights_only=False)

    ar1_config = config.copy()
    ar1_config["init_rho"] = 0.8
    ar1_config["init_sigma"] = 1.0
    ar1_predictor = LatentPredictorCov(ar1_config)
    ar1_predictor.load_state_dict(ar1_ckpt["predictor_state_dict"])
    ar1_predictor = ar1_predictor.to(device)
    ar1_predictor.eval()

    return vae, ar1_predictor, config


def analyze_gt_volatility_clustering(log_returns, max_lag=10):
    """Analyze volatility clustering in ground truth."""
    print("\n" + "=" * 70)
    print("Ground Truth Volatility Clustering Analysis")
    print("=" * 70)

    # Full time series ACF of squared returns
    atm_returns = log_returns[:, 2, 2]

    print(f"\nFull time series (N={len(atm_returns)}):")
    acf_sq = compute_acf_squared(atm_returns, max_lag)
    print(f"  ACF of squared returns (lag 1-{max_lag}):")
    for lag, acf in enumerate(acf_sq, 1):
        print(f"    Lag {lag:2d}: {acf:+.4f}")

    # Also compute for raw returns (should be near zero for efficient markets)
    acf_raw = compute_acf(atm_returns, max_lag)
    print(f"\n  ACF of raw returns (lag 1-{max_lag}):")
    for lag, acf in enumerate(acf_raw, 1):
        print(f"    Lag {lag:2d}: {acf:+.4f}")

    # Window-based analysis (30-day windows like model generates)
    print(f"\n30-day window analysis:")
    window_acfs_sq = []
    window_acfs_raw = []

    for i in range(0, len(atm_returns) - 30, 10):
        window = atm_returns[i:i+30]
        acf_sq_1 = compute_acf_squared(window, 1)[0]
        acf_raw_1 = compute_acf(window, 1)[0]
        if not np.isnan(acf_sq_1):
            window_acfs_sq.append(acf_sq_1)
        if not np.isnan(acf_raw_1):
            window_acfs_raw.append(acf_raw_1)

    print(f"  N windows: {len(window_acfs_sq)}")
    print(f"  ACF(1) of squared returns:")
    print(f"    Mean: {np.mean(window_acfs_sq):+.4f}")
    print(f"    Std:  {np.std(window_acfs_sq):.4f}")
    print(f"    Min:  {np.min(window_acfs_sq):+.4f}")
    print(f"    Max:  {np.max(window_acfs_sq):+.4f}")

    print(f"  ACF(1) of raw returns:")
    print(f"    Mean: {np.mean(window_acfs_raw):+.4f}")
    print(f"    Std:  {np.std(window_acfs_raw):.4f}")

    return {
        "full_acf_sq": acf_sq,
        "full_acf_raw": acf_raw,
        "window_acf_sq_mean": np.mean(window_acfs_sq),
        "window_acf_sq_std": np.std(window_acfs_sq),
        "window_acf_raw_mean": np.mean(window_acfs_raw),
        "window_acfs_sq": window_acfs_sq,
        "window_acfs_raw": window_acfs_raw,
    }


def generate_oracle_samples(vae, log_returns, n_windows=200, horizon=30, n_samples=20, device="cuda"):
    """Generate samples in oracle mode (z from posterior)."""
    all_acfs_sq = []
    all_acfs_raw = []

    N = len(log_returns)
    context_len = 20

    for i in range(0, min(n_windows * 10, N - context_len - horizon), 10):
        # Full sequence for oracle
        seq = log_returns[i:i + context_len + horizon]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Oracle: get z from posterior (sees full sequence)
            ctx_emb = vae.ctx_encoder({"surface": seq_tensor})
            z_mean, z_logvar, _ = vae.main_encoder({"surface": seq_tensor})
            z_std = torch.exp(0.5 * z_logvar)

            for _ in range(n_samples):
                # Sample z from posterior
                eps = torch.randn_like(z_mean)
                z = z_mean + z_std * eps

                # Decode
                _, samples, _, _ = vae.decoder(ctx_emb, z, sample=True)

                # Extract horizon portion (after context)
                sample_returns = samples[0, context_len:, 2, 2].cpu().numpy()

                if len(sample_returns) >= 5:
                    acf_sq = compute_acf_squared(sample_returns, 1)[0]
                    acf_raw = compute_acf(sample_returns, 1)[0]
                    if not np.isnan(acf_sq):
                        all_acfs_sq.append(acf_sq)
                    if not np.isnan(acf_raw):
                        all_acfs_raw.append(acf_raw)

    return {
        "acfs_sq": all_acfs_sq,
        "acfs_raw": all_acfs_raw,
        "mean_acf_sq": np.mean(all_acfs_sq) if all_acfs_sq else 0,
        "mean_acf_raw": np.mean(all_acfs_raw) if all_acfs_raw else 0,
    }


def generate_prior_samples(vae, predictor, log_returns, n_windows=200, horizon=30,
                           n_samples=20, device="cuda"):
    """Generate samples in prior mode (z from predictor)."""
    all_acfs_sq = []
    all_acfs_raw = []

    N = len(log_returns)
    context_len = 20

    for i in range(0, min(n_windows * 10, N - context_len - horizon), 10):
        context = log_returns[i:i + context_len]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get z prediction from prior network
            z_mean, _ = predictor(context_tensor, horizon=horizon)
            z_samples = predictor.sample_z(z_mean, n_samples=n_samples)

            for s in range(n_samples):
                z = z_samples[s]

                # Create full sequence context embedding (zeros for future)
                full_seq = torch.zeros(1, context_len + horizon, 5, 5, device=device)
                full_seq[0, :context_len] = context_tensor[0]
                ctx_emb = vae.ctx_encoder({"surface": full_seq})

                # Decode
                _, samples, _, _ = vae.decoder(ctx_emb, z, sample=True)

                # Extract generated portion
                sample_returns = samples[0, :, 2, 2].cpu().numpy()

                if len(sample_returns) >= 5:
                    acf_sq = compute_acf_squared(sample_returns, 1)[0]
                    acf_raw = compute_acf(sample_returns, 1)[0]
                    if not np.isnan(acf_sq):
                        all_acfs_sq.append(acf_sq)
                    if not np.isnan(acf_raw):
                        all_acfs_raw.append(acf_raw)

    return {
        "acfs_sq": all_acfs_sq,
        "acfs_raw": all_acfs_raw,
        "mean_acf_sq": np.mean(all_acfs_sq) if all_acfs_sq else 0,
        "mean_acf_raw": np.mean(all_acfs_raw) if all_acfs_raw else 0,
    }


def main():
    print("=" * 70)
    print("Volatility Clustering Verification")
    print("=" * 70)
    print("\nVolatility clustering = positive ACF of squared returns")
    print("If squared returns are autocorrelated, large moves follow large moves")

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Analyze GT
    gt_results = analyze_gt_volatility_clustering(log_returns, max_lag=10)

    # Load models
    print("\nLoading models...")
    vae, predictor, config = load_models(device)

    # Generate Oracle samples
    print("\n" + "=" * 70)
    print("Oracle Mode (z from posterior)")
    print("=" * 70)
    oracle_results = generate_oracle_samples(vae, log_returns, n_windows=100,
                                             n_samples=10, device=device)
    print(f"\n  N samples: {len(oracle_results['acfs_sq'])}")
    print(f"  ACF(1) of squared returns:")
    print(f"    Mean: {oracle_results['mean_acf_sq']:+.4f}")
    print(f"    Std:  {np.std(oracle_results['acfs_sq']):.4f}")
    print(f"  ACF(1) of raw returns:")
    print(f"    Mean: {oracle_results['mean_acf_raw']:+.4f}")

    # Generate Prior samples
    print("\n" + "=" * 70)
    print("Prior Mode (z from predictor)")
    print("=" * 70)
    prior_results = generate_prior_samples(vae, predictor, log_returns, n_windows=100,
                                           n_samples=10, device=device)
    print(f"\n  N samples: {len(prior_results['acfs_sq'])}")
    print(f"  ACF(1) of squared returns:")
    print(f"    Mean: {prior_results['mean_acf_sq']:+.4f}")
    print(f"    Std:  {np.std(prior_results['acfs_sq']):.4f}")
    print(f"  ACF(1) of raw returns:")
    print(f"    Mean: {prior_results['mean_acf_raw']:+.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Volatility Clustering Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<40} {'GT':<12} {'Oracle':<12} {'Prior':<12}")
    print("-" * 76)
    print(f"{'ACF(1) of SQUARED returns (vol cluster)':<40} "
          f"{gt_results['window_acf_sq_mean']:+.4f}{'':<6} "
          f"{oracle_results['mean_acf_sq']:+.4f}{'':<6} "
          f"{prior_results['mean_acf_sq']:+.4f}")
    print(f"{'ACF(1) of RAW returns (mean reversion)':<40} "
          f"{gt_results['window_acf_raw_mean']:+.4f}{'':<6} "
          f"{oracle_results['mean_acf_raw']:+.4f}{'':<6} "
          f"{prior_results['mean_acf_raw']:+.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    gt_vol_cluster = gt_results['window_acf_sq_mean']

    if gt_vol_cluster > 0.1:
        print(f"\n  GT has STRONG volatility clustering (ACF_sq = {gt_vol_cluster:.3f})")
        print("  → Large moves tend to follow large moves")
        print("  → Model SHOULD try to capture this")
    elif gt_vol_cluster > 0.02:
        print(f"\n  GT has WEAK volatility clustering (ACF_sq = {gt_vol_cluster:.3f})")
        print("  → Some tendency for large moves to cluster")
        print("  → Model could optionally try to capture this")
    else:
        print(f"\n  GT has NO volatility clustering (ACF_sq = {gt_vol_cluster:.3f})")
        print("  → Squared returns are essentially independent")
        print("  → NO NEED to optimize for volatility clustering!")

    # Statistical test
    print("\n  Statistical significance (H0: ACF_sq = 0):")
    t_stat, p_value = stats.ttest_1samp(gt_results['window_acfs_sq'], 0)
    print(f"    t-statistic: {t_stat:.2f}")
    print(f"    p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"    → Significant at p<0.05: GT volatility clustering EXISTS")
    else:
        print(f"    → NOT significant: GT volatility clustering may NOT exist")

    # Save plot
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ACF of squared returns comparison
    ax = axes[0]
    labels = ['GT', 'Oracle', 'Prior']
    means = [gt_results['window_acf_sq_mean'], oracle_results['mean_acf_sq'], prior_results['mean_acf_sq']]
    stds = [gt_results['window_acf_sq_std'], np.std(oracle_results['acfs_sq']), np.std(prior_results['acfs_sq'])]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['blue', 'green', 'orange'], alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('ACF(1) of Squared Returns')
    ax.set_title('Volatility Clustering: ACF of Squared Returns')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', va='bottom', fontsize=10)

    # ACF of raw returns comparison
    ax = axes[1]
    means_raw = [gt_results['window_acf_raw_mean'], oracle_results['mean_acf_raw'], prior_results['mean_acf_raw']]
    stds_raw = [np.std(gt_results['window_acfs_raw']), np.std(oracle_results['acfs_raw']), np.std(prior_results['acfs_raw'])]

    bars = ax.bar(x, means_raw, yerr=stds_raw, capsize=5, color=['blue', 'green', 'orange'], alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('ACF(1) of Raw Returns')
    ax.set_title('Mean Reversion: ACF of Raw Returns')
    ax.grid(True, alpha=0.3)

    for i, (m, s) in enumerate(zip(means_raw, stds_raw)):
        ax.text(i, m + s + 0.02 if m > 0 else m - s - 0.05, f'{m:.3f}', ha='center',
                va='bottom' if m > 0 else 'top', fontsize=10)

    plt.suptitle('Volatility Clustering Verification: GT vs Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "volatility_clustering_verification.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {save_dir / 'volatility_clustering_verification.png'}")
    plt.close()

    return {
        "gt": gt_results,
        "oracle": oracle_results,
        "prior": prior_results,
    }


if __name__ == "__main__":
    results = main()
