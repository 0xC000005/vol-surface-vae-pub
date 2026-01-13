"""
Verify Correlated Decoder Noise

Test if correlating the decoder's Student-t noise (not just z) improves ACF.

The standard decoder samples:
- eps_rank ~ N(0,1) independently per timestep
- eps_diag ~ N(0,1) independently per timestep

This script tests correlating these using AR(1) structure.

Usage:
    python experiments/backfill/two_stage_vae/verify_correlated_decoder_noise.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor, LatentPredictorCov


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def build_ar1_cholesky(T: int, rho: float, device: str = "cuda"):
    """Build Cholesky factor of AR(1) covariance matrix."""
    idx = torch.arange(T, device=device, dtype=torch.float32)
    diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    Sigma = rho ** diff
    Sigma = Sigma + 1e-6 * torch.eye(T, device=device)
    return torch.linalg.cholesky(Sigma)


def decode_with_correlated_noise(decoder, z, rho_noise: float = 0.8, device: str = "cuda"):
    """
    Decode z with correlated Student-t noise.

    Instead of sampling eps_rank and eps_diag independently per timestep,
    we correlate them across time using AR(1) structure.
    """
    B, T, _ = z.shape

    # Get mean and covariance parameters (no sampling)
    z_flat = z.view(B * T, -1)
    mean = decoder.mean_net(z_flat).view(B, T, 5, 5)

    z_pooled = z.mean(dim=1)
    factor_flat = decoder.factor_net(z_pooled)
    factor = factor_flat.view(B, 25, decoder.rank)

    log_diag = decoder.log_diag_net(z_pooled)
    log_diag = torch.clamp(log_diag, min=-10, max=2)

    # Build AR(1) Cholesky factor for time correlation
    L = build_ar1_cholesky(T, rho_noise, device)  # (T, T)

    # Sample correlated noise for rank component
    eps_rank_iid = torch.randn(B, T, decoder.rank, device=device)
    # Apply correlation: eps_rank[b, t, r] = sum_j L[t,j] * eps_rank_iid[b, j, r]
    eps_rank = torch.einsum('ij,bjr->bir', L, eps_rank_iid)

    # Sample correlated noise for diagonal component
    eps_diag_iid = torch.randn(B, T, 25, device=device)
    eps_diag = torch.einsum('ij,bjk->bik', L, eps_diag_iid)

    # Create Gaussian samples
    correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)
    diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
    independent_gauss = diag_std * eps_diag
    total_gauss = correlated_gauss + independent_gauss

    # Convert to Student-t by dividing by sqrt(chi2/nu)
    chi2_samples = torch.zeros(B, T, 25, device=device)
    for i in range(25):
        nu_i = decoder.nu[i].item()
        alpha = nu_i / 2
        beta = nu_i / 2
        gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
        chi2_samples[:, :, i] = gamma_samples

    student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
    total_t = total_gauss * student_t_factor

    samples = mean + total_t.view(B, T, 5, 5)

    return mean, samples


def load_models(device: str = "cuda"):
    """Load VAE and predictors."""
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load AR(1) predictor (for correlated z sampling)
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


def generate_samples(vae, predictor, log_returns, context_len=20, horizon=30,
                     n_windows=100, n_samples=10, rho_z=0.8, rho_noise=0.8, device="cuda"):
    """Generate samples with correlated z and correlated decoder noise."""
    N = len(log_returns)
    all_returns = []

    window_starts = np.linspace(100, N - context_len - horizon - 100, n_windows).astype(int)

    for start_idx in window_starts:
        context = log_returns[start_idx:start_idx + context_len]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get z mean from predictor
            z_mean, _ = predictor(context_tensor, horizon=horizon)

            # Sample z with AR(1) correlation
            z_samples = predictor.sample_z(z_mean, n_samples=n_samples)

            for s in range(n_samples):
                z = z_samples[s]

                # Decode with correlated noise
                _, samples = decode_with_correlated_noise(
                    vae.decoder, z, rho_noise=rho_noise, device=device
                )

                returns = samples[0, :, 2, 2].cpu().numpy()
                all_returns.append(returns)

    return np.array(all_returns)


def main():
    print("=" * 70)
    print("Verifying Correlated Decoder Noise")
    print("=" * 70)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vae, predictor, config = load_models(device)

    context_len = 20
    horizon = 30
    n_windows = 100
    n_samples = 20

    # Compute GT statistics
    print("\nComputing Ground Truth statistics...")
    gt_returns = []
    for i in range(100, len(log_returns) - context_len - horizon - 100, 10):
        ret = log_returns[i + context_len:i + context_len + horizon, 2, 2]
        gt_returns.append(ret)
    gt_returns = np.array(gt_returns)

    gt_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in gt_returns]
    gt_acf = np.mean([a for a in gt_acfs if not np.isnan(a)])
    gt_cumsum = np.array([np.cumsum(r) for r in gt_returns])
    gt_kurt = stats.kurtosis(gt_cumsum[:, -1], fisher=True)

    print(f"  GT ACF(1): {gt_acf:.4f}")
    print(f"  GT Kurtosis (h=30): {gt_kurt:.2f}")

    # Test different rho_noise values
    rho_values = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
    results = {}

    for rho_noise in rho_values:
        print(f"\nGenerating with rho_noise={rho_noise}...")

        samples = generate_samples(
            vae, predictor, log_returns,
            context_len=context_len, horizon=horizon,
            n_windows=n_windows, n_samples=n_samples,
            rho_z=0.8, rho_noise=rho_noise, device=device
        )

        acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in samples]
        acf = np.mean([a for a in acfs if not np.isnan(a)])
        cumsum = np.array([np.cumsum(r) for r in samples])
        kurt = stats.kurtosis(cumsum[:, -1], fisher=True)
        std = np.std(cumsum[:, -1])

        results[rho_noise] = {"acf": acf, "kurt": kurt, "std": std}
        print(f"  ACF(1): {acf:.4f}, Kurtosis: {kurt:.2f}, Std: {std:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Effect of Decoder Noise Correlation (rho_noise)")
    print("=" * 70)

    print(f"\n{'rho_noise':<12} {'ACF(1)':<12} {'ACF Gap':<12} {'Kurtosis':<12} {'Kurt Gap':<12}")
    print("-" * 60)
    print(f"{'GT':<12} {gt_acf:<12.4f} {'-':<12} {gt_kurt:<12.2f} {'-':<12}")

    for rho_noise, r in results.items():
        acf_gap = abs(gt_acf - r["acf"])
        kurt_gap = abs(gt_kurt - r["kurt"])
        print(f"{rho_noise:<12.1f} {r['acf']:<12.4f} {acf_gap:<12.4f} {r['kurt']:<12.2f} {kurt_gap:<12.2f}")

    # Find best rho_noise for ACF
    best_rho_acf = min(results.keys(), key=lambda r: abs(gt_acf - results[r]["acf"]))
    best_rho_kurt = min(results.keys(), key=lambda r: abs(gt_kurt - results[r]["kurt"]))

    print(f"\nBest rho_noise for ACF: {best_rho_acf} (gap: {abs(gt_acf - results[best_rho_acf]['acf']):.4f})")
    print(f"Best rho_noise for Kurtosis: {best_rho_kurt} (gap: {abs(gt_kurt - results[best_rho_kurt]['kurt']):.2f})")

    # Save plot
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rhos = list(results.keys())
    acfs = [results[r]["acf"] for r in rhos]
    kurts = [results[r]["kurt"] for r in rhos]
    stds = [results[r]["std"] for r in rhos]

    axes[0].plot(rhos, acfs, 'b-o', linewidth=2, markersize=8, label='Model')
    axes[0].axhline(gt_acf, color='r', linestyle='--', linewidth=2, label=f'GT ({gt_acf:.3f})')
    axes[0].set_xlabel('rho_noise')
    axes[0].set_ylabel('ACF(1)')
    axes[0].set_title('ACF vs Decoder Noise Correlation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rhos, kurts, 'g-o', linewidth=2, markersize=8, label='Model')
    axes[1].axhline(gt_kurt, color='r', linestyle='--', linewidth=2, label=f'GT ({gt_kurt:.1f})')
    axes[1].set_xlabel('rho_noise')
    axes[1].set_ylabel('Kurtosis (h=30)')
    axes[1].set_title('Kurtosis vs Decoder Noise Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rhos, stds, 'm-o', linewidth=2, markersize=8, label='Model')
    axes[2].axhline(np.std(gt_cumsum[:, -1]), color='r', linestyle='--', linewidth=2,
                    label=f'GT ({np.std(gt_cumsum[:, -1]):.3f})')
    axes[2].set_xlabel('rho_noise')
    axes[2].set_ylabel('Std (h=30)')
    axes[2].set_title('Std vs Decoder Noise Correlation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Effect of Correlated Decoder Noise on Output Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "correlated_decoder_noise.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_dir / 'correlated_decoder_noise.png'}")
    plt.close()

    return results


if __name__ == "__main__":
    results = main()
