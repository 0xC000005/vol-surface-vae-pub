"""
Test Negative Correlation for Mean-Reverting ACF

The GT has negative ACF (mean-reversion), but positive rho creates positive ACF.
Test if negative rho_noise creates negative ACF matching GT.

Usage:
    python experiments/backfill/two_stage_vae/verify_negative_correlation.py
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


def to_log_returns(surfaces: np.ndarray):
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def build_mean_reverting_cholesky(T: int, rho: float, device: str = "cuda"):
    """
    Build Cholesky factor for mean-reverting (negative) correlation.

    For mean-reverting process: Sigma[i,j] = sigma^2 * (-rho)^|i-j| for |rho| < 1
    This creates alternating positive/negative correlations.

    Actually, a cleaner approach is to use the AR(1) with negative rho directly:
    Sigma[i,j] = rho^|i-j| where rho can be negative.

    For rho < 0: adjacent timesteps are negatively correlated (mean-reverting)
    """
    idx = torch.arange(T, device=device, dtype=torch.float32)
    diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

    # For negative rho, rho^|i-j| creates alternating signs
    # rho^1 < 0 (adjacent negative)
    # rho^2 > 0 (next-adjacent positive)
    # etc.
    Sigma = torch.pow(torch.tensor(rho, device=device), diff)

    # Add jitter for numerical stability
    Sigma = Sigma + 1e-4 * torch.eye(T, device=device)

    # Ensure positive definiteness (needed for Cholesky)
    try:
        L = torch.linalg.cholesky(Sigma)
    except RuntimeError:
        # If not positive definite, use eigenvalue clipping
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        eigvals = torch.clamp(eigvals, min=1e-4)
        Sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        L = torch.linalg.cholesky(Sigma)

    return L


def decode_with_correlated_noise(decoder, z, rho_noise: float = 0.0, device: str = "cuda"):
    """Decode z with correlated Student-t noise."""
    B, T, _ = z.shape

    z_flat = z.view(B * T, -1)
    mean = decoder.mean_net(z_flat).view(B, T, 5, 5)

    z_pooled = z.mean(dim=1)
    factor_flat = decoder.factor_net(z_pooled)
    factor = factor_flat.view(B, 25, decoder.rank)

    log_diag = decoder.log_diag_net(z_pooled)
    log_diag = torch.clamp(log_diag, min=-10, max=2)

    if abs(rho_noise) < 0.01:
        # No correlation - use standard independent sampling
        eps_rank = torch.randn(B, T, decoder.rank, device=device)
        eps_diag = torch.randn(B, T, 25, device=device)
    else:
        # Build correlation matrix Cholesky
        L = build_mean_reverting_cholesky(T, rho_noise, device)

        eps_rank_iid = torch.randn(B, T, decoder.rank, device=device)
        eps_rank = torch.einsum('ij,bjr->bir', L, eps_rank_iid)

        eps_diag_iid = torch.randn(B, T, 25, device=device)
        eps_diag = torch.einsum('ij,bjk->bik', L, eps_diag_iid)

    correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)
    diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
    independent_gauss = diag_std * eps_diag
    total_gauss = correlated_gauss + independent_gauss

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
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

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
                     n_windows=100, n_samples=10, rho_noise=0.0, device="cuda"):
    N = len(log_returns)
    all_returns = []

    window_starts = np.linspace(100, N - context_len - horizon - 100, n_windows).astype(int)

    for start_idx in window_starts:
        context = log_returns[start_idx:start_idx + context_len]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            z_mean, _ = predictor(context_tensor, horizon=horizon)
            z_samples = predictor.sample_z(z_mean, n_samples=n_samples)

            for s in range(n_samples):
                z = z_samples[s]
                _, samples = decode_with_correlated_noise(
                    vae.decoder, z, rho_noise=rho_noise, device=device
                )
                returns = samples[0, :, 2, 2].cpu().numpy()
                all_returns.append(returns)

    return np.array(all_returns)


def main():
    print("=" * 70)
    print("Testing Negative Correlation for Mean-Reverting ACF")
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
    gt_std = np.std(gt_cumsum[:, -1])

    print(f"  GT ACF(1): {gt_acf:.4f}")
    print(f"  GT Kurtosis (h=30): {gt_kurt:.2f}")
    print(f"  GT Std (h=30): {gt_std:.4f}")

    # Test different rho_noise values including negative
    rho_values = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    results = {}

    for rho_noise in rho_values:
        print(f"\nGenerating with rho_noise={rho_noise}...")

        try:
            samples = generate_samples(
                vae, predictor, log_returns,
                context_len=context_len, horizon=horizon,
                n_windows=n_windows, n_samples=n_samples,
                rho_noise=rho_noise, device=device
            )

            acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in samples]
            acf = np.mean([a for a in acfs if not np.isnan(a)])
            cumsum = np.array([np.cumsum(r) for r in samples])
            kurt = stats.kurtosis(cumsum[:, -1], fisher=True)
            std = np.std(cumsum[:, -1])

            results[rho_noise] = {"acf": acf, "kurt": kurt, "std": std}
            print(f"  ACF(1): {acf:.4f}, Kurtosis: {kurt:.2f}, Std: {std:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[rho_noise] = {"acf": np.nan, "kurt": np.nan, "std": np.nan}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Effect of Negative Noise Correlation")
    print("=" * 70)

    print(f"\n{'rho_noise':<12} {'ACF(1)':<12} {'ACF Gap':<12} {'Kurtosis':<12} {'Kurt Gap':<12} {'Std':<12}")
    print("-" * 72)
    print(f"{'GT':<12} {gt_acf:<12.4f} {'-':<12} {gt_kurt:<12.2f} {'-':<12} {gt_std:<12.4f}")

    for rho_noise in sorted(results.keys()):
        r = results[rho_noise]
        if not np.isnan(r["acf"]):
            acf_gap = abs(gt_acf - r["acf"])
            kurt_gap = abs(gt_kurt - r["kurt"])
            print(f"{rho_noise:<12.1f} {r['acf']:<12.4f} {acf_gap:<12.4f} {r['kurt']:<12.2f} {kurt_gap:<12.2f} {r['std']:<12.4f}")

    # Find best rho_noise for ACF
    valid_results = {k: v for k, v in results.items() if not np.isnan(v["acf"])}
    if valid_results:
        best_rho_acf = min(valid_results.keys(), key=lambda r: abs(gt_acf - valid_results[r]["acf"]))
        print(f"\nBest rho_noise for ACF: {best_rho_acf} (gap: {abs(gt_acf - valid_results[best_rho_acf]['acf']):.4f})")

    # Save plot
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rhos = sorted([r for r in results.keys() if not np.isnan(results[r]["acf"])])
    acfs = [results[r]["acf"] for r in rhos]
    kurts = [results[r]["kurt"] for r in rhos]
    stds = [results[r]["std"] for r in rhos]

    axes[0].plot(rhos, acfs, 'b-o', linewidth=2, markersize=8, label='Model')
    axes[0].axhline(gt_acf, color='r', linestyle='--', linewidth=2, label=f'GT ({gt_acf:.3f})')
    axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('rho_noise')
    axes[0].set_ylabel('ACF(1)')
    axes[0].set_title('ACF vs Noise Correlation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rhos, kurts, 'g-o', linewidth=2, markersize=8, label='Model')
    axes[1].axhline(gt_kurt, color='r', linestyle='--', linewidth=2, label=f'GT ({gt_kurt:.1f})')
    axes[1].set_xlabel('rho_noise')
    axes[1].set_ylabel('Kurtosis (h=30)')
    axes[1].set_title('Kurtosis vs Noise Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rhos, stds, 'm-o', linewidth=2, markersize=8, label='Model')
    axes[2].axhline(gt_std, color='r', linestyle='--', linewidth=2, label=f'GT ({gt_std:.3f})')
    axes[2].set_xlabel('rho_noise')
    axes[2].set_ylabel('Std (h=30)')
    axes[2].set_title('Std vs Noise Correlation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Effect of Negative Noise Correlation (Mean-Reversion)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "negative_correlation.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_dir / 'negative_correlation.png'}")
    plt.close()

    return results


if __name__ == "__main__":
    results = main()
