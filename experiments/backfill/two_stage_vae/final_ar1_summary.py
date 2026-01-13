"""
Final AR(1) Correlated Noise Summary

Compare baseline vs optimal correlated noise configuration.

Key Finding: rho_noise = -0.2 achieves 87.6% ACF recovery (vs 1.8% baseline)

Usage:
    python experiments/backfill/two_stage_vae/final_ar1_summary.py
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
    """Build Cholesky factor for correlation structure."""
    idx = torch.arange(T, device=device, dtype=torch.float32)
    diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    Sigma = torch.pow(torch.tensor(rho, device=device), diff)
    Sigma = Sigma + 1e-4 * torch.eye(T, device=device)

    try:
        L = torch.linalg.cholesky(Sigma)
    except RuntimeError:
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        eigvals = torch.clamp(eigvals, min=1e-4)
        Sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        L = torch.linalg.cholesky(Sigma)
    return L


def decode_with_correlated_noise(decoder, z, rho_noise: float, device: str = "cuda"):
    """Decode z with correlated noise."""
    B, T, _ = z.shape

    z_flat = z.view(B * T, -1)
    mean = decoder.mean_net(z_flat).view(B, T, 5, 5)

    z_pooled = z.mean(dim=1)
    factor_flat = decoder.factor_net(z_pooled)
    factor = factor_flat.view(B, 25, decoder.rank)

    log_diag = decoder.log_diag_net(z_pooled)
    log_diag = torch.clamp(log_diag, min=-10, max=2)

    if abs(rho_noise) < 0.01:
        eps_rank = torch.randn(B, T, decoder.rank, device=device)
        eps_diag = torch.randn(B, T, 25, device=device)
    else:
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
                     n_windows=200, n_samples=20, rho_noise=0.0, device="cuda"):
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


def compute_metrics(returns):
    """Compute comprehensive metrics from returns."""
    acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in returns]
    acf = np.mean([a for a in acfs if not np.isnan(a)])

    cumsum = np.array([np.cumsum(r) for r in returns])
    kurt = stats.kurtosis(cumsum[:, -1], fisher=True)
    std = np.std(cumsum[:, -1])
    mean = np.mean(cumsum[:, -1])

    return {"acf": acf, "kurt": kurt, "std": std, "mean": mean}


def main():
    print("=" * 70)
    print("FINAL SUMMARY: AR(1) Correlated Noise for Mean-Reverting ACF")
    print("=" * 70)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vae, predictor, config = load_models(device)

    context_len = 20
    horizon = 30
    n_windows = 200
    n_samples = 20

    # Compute GT statistics
    print("\n" + "=" * 70)
    print("GROUND TRUTH STATISTICS")
    print("=" * 70)

    gt_returns = []
    for i in range(100, len(log_returns) - context_len - horizon - 100, 5):
        ret = log_returns[i + context_len:i + context_len + horizon, 2, 2]
        gt_returns.append(ret)
    gt_returns = np.array(gt_returns)
    gt_metrics = compute_metrics(gt_returns)

    print(f"  ACF(1):    {gt_metrics['acf']:.4f}")
    print(f"  Kurtosis:  {gt_metrics['kurt']:.2f}")
    print(f"  Std:       {gt_metrics['std']:.4f}")

    # Generate baseline (rho_noise = 0)
    print("\n" + "=" * 70)
    print("BASELINE (rho_noise = 0.0)")
    print("=" * 70)

    baseline_returns = generate_samples(
        vae, predictor, log_returns,
        context_len=context_len, horizon=horizon,
        n_windows=n_windows, n_samples=n_samples,
        rho_noise=0.0, device=device
    )
    baseline_metrics = compute_metrics(baseline_returns)

    print(f"  ACF(1):    {baseline_metrics['acf']:.4f} (gap: {abs(gt_metrics['acf'] - baseline_metrics['acf']):.4f})")
    print(f"  Kurtosis:  {baseline_metrics['kurt']:.2f} (gap: {abs(gt_metrics['kurt'] - baseline_metrics['kurt']):.2f})")
    print(f"  Std:       {baseline_metrics['std']:.4f} (gap: {abs(gt_metrics['std'] - baseline_metrics['std']):.4f})")

    # Generate optimal (rho_noise = -0.2)
    print("\n" + "=" * 70)
    print("OPTIMAL (rho_noise = -0.2)")
    print("=" * 70)

    optimal_returns = generate_samples(
        vae, predictor, log_returns,
        context_len=context_len, horizon=horizon,
        n_windows=n_windows, n_samples=n_samples,
        rho_noise=-0.2, device=device
    )
    optimal_metrics = compute_metrics(optimal_returns)

    print(f"  ACF(1):    {optimal_metrics['acf']:.4f} (gap: {abs(gt_metrics['acf'] - optimal_metrics['acf']):.4f})")
    print(f"  Kurtosis:  {optimal_metrics['kurt']:.2f} (gap: {abs(gt_metrics['kurt'] - optimal_metrics['kurt']):.2f})")
    print(f"  Std:       {optimal_metrics['std']:.4f} (gap: {abs(gt_metrics['std'] - optimal_metrics['std']):.4f})")

    # Calculate improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)

    acf_baseline_gap = abs(gt_metrics['acf'] - baseline_metrics['acf'])
    acf_optimal_gap = abs(gt_metrics['acf'] - optimal_metrics['acf'])
    acf_improvement = (acf_baseline_gap - acf_optimal_gap) / acf_baseline_gap * 100

    kurt_baseline_gap = abs(gt_metrics['kurt'] - baseline_metrics['kurt'])
    kurt_optimal_gap = abs(gt_metrics['kurt'] - optimal_metrics['kurt'])
    kurt_improvement = (kurt_baseline_gap - kurt_optimal_gap) / kurt_baseline_gap * 100 if kurt_baseline_gap > 0 else 0

    print(f"\n{'Metric':<20} {'Baseline Gap':<15} {'Optimal Gap':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'ACF(1)':<20} {acf_baseline_gap:<15.4f} {acf_optimal_gap:<15.4f} {acf_improvement:+.1f}%")
    print(f"{'Kurtosis':<20} {kurt_baseline_gap:<15.2f} {kurt_optimal_gap:<15.2f} {kurt_improvement:+.1f}%")

    # ACF recovery percentage
    acf_recovery = (1 - acf_optimal_gap / abs(gt_metrics['acf'])) * 100
    print(f"\n** ACF Recovery: {acf_recovery:.1f}% **")

    # Save comprehensive plot
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Distribution comparisons
    gt_cumsum = np.array([np.cumsum(r) for r in gt_returns])
    baseline_cumsum = np.array([np.cumsum(r) for r in baseline_returns])
    optimal_cumsum = np.array([np.cumsum(r) for r in optimal_returns])

    # Cumulative return distribution at h=30
    axes[0, 0].hist(gt_cumsum[:, -1], bins=50, alpha=0.5, label=f'GT (k={gt_metrics["kurt"]:.1f})', density=True)
    axes[0, 0].hist(baseline_cumsum[:, -1], bins=50, alpha=0.5, label=f'Baseline (k={baseline_metrics["kurt"]:.1f})', density=True)
    axes[0, 0].hist(optimal_cumsum[:, -1], bins=50, alpha=0.5, label=f'Optimal (k={optimal_metrics["kurt"]:.1f})', density=True)
    axes[0, 0].set_xlabel('Cumulative Return at h=30')
    axes[0, 0].set_title('Distribution Comparison')
    axes[0, 0].legend(fontsize=8)

    # Q-Q plot
    n_points = 100
    quantiles = np.linspace(0.01, 0.99, n_points)
    gt_q = np.quantile(gt_cumsum[:, -1], quantiles)
    baseline_q = np.quantile(baseline_cumsum[:, -1], quantiles)
    optimal_q = np.quantile(optimal_cumsum[:, -1], quantiles)

    axes[0, 1].scatter(gt_q, baseline_q, alpha=0.5, s=20, label='Baseline vs GT')
    axes[0, 1].scatter(gt_q, optimal_q, alpha=0.5, s=20, label='Optimal vs GT')
    min_val = min(gt_q.min(), baseline_q.min(), optimal_q.min())
    max_val = max(gt_q.max(), baseline_q.max(), optimal_q.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    axes[0, 1].set_xlabel('GT Quantiles')
    axes[0, 1].set_ylabel('Model Quantiles')
    axes[0, 1].set_title('Q-Q Plot (h=30)')
    axes[0, 1].legend(fontsize=8)

    # Per-sequence ACF histogram
    gt_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in gt_returns if len(r) > 1]
    baseline_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in baseline_returns if len(r) > 1]
    optimal_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in optimal_returns if len(r) > 1]

    gt_acfs = [a for a in gt_acfs if not np.isnan(a)]
    baseline_acfs = [a for a in baseline_acfs if not np.isnan(a)]
    optimal_acfs = [a for a in optimal_acfs if not np.isnan(a)]

    axes[0, 2].hist(gt_acfs, bins=30, alpha=0.5, label=f'GT (mean={np.mean(gt_acfs):.3f})', density=True)
    axes[0, 2].hist(baseline_acfs, bins=30, alpha=0.5, label=f'Baseline (mean={np.mean(baseline_acfs):.3f})', density=True)
    axes[0, 2].hist(optimal_acfs, bins=30, alpha=0.5, label=f'Optimal (mean={np.mean(optimal_acfs):.3f})', density=True)
    axes[0, 2].axvline(gt_metrics['acf'], color='blue', linestyle='--', linewidth=2)
    axes[0, 2].axvline(baseline_metrics['acf'], color='orange', linestyle='--', linewidth=2)
    axes[0, 2].axvline(optimal_metrics['acf'], color='green', linestyle='--', linewidth=2)
    axes[0, 2].set_xlabel('ACF(1)')
    axes[0, 2].set_title('Per-Sequence ACF Distribution')
    axes[0, 2].legend(fontsize=8)

    # Row 2: Trajectory comparisons
    np.random.seed(42)
    sample_idx = np.random.choice(len(gt_returns), 10, replace=False)

    for i in sample_idx:
        if i < len(gt_cumsum):
            axes[1, 0].plot(gt_cumsum[i], alpha=0.3, color='blue')
    axes[1, 0].set_xlabel('Horizon')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].set_title(f'GT Trajectories (ACF={gt_metrics["acf"]:.3f})')

    for i in sample_idx:
        if i < len(baseline_cumsum):
            axes[1, 1].plot(baseline_cumsum[i], alpha=0.3, color='orange')
    axes[1, 1].set_xlabel('Horizon')
    axes[1, 1].set_title(f'Baseline Trajectories (ACF={baseline_metrics["acf"]:.3f})')

    for i in sample_idx:
        if i < len(optimal_cumsum):
            axes[1, 2].plot(optimal_cumsum[i], alpha=0.3, color='green')
    axes[1, 2].set_xlabel('Horizon')
    axes[1, 2].set_title(f'Optimal Trajectories (ACF={optimal_metrics["acf"]:.3f})')

    plt.suptitle(
        f'AR(1) Correlated Noise: ACF Recovery = {acf_recovery:.1f}%\n'
        f'Optimal rho_noise = -0.2',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_dir / "final_summary.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_dir / 'final_summary.png'}")
    plt.close()

    # Save results
    results = {
        "gt_metrics": gt_metrics,
        "baseline_metrics": baseline_metrics,
        "optimal_metrics": optimal_metrics,
        "acf_improvement_pct": acf_improvement,
        "acf_recovery_pct": acf_recovery,
        "optimal_rho_noise": -0.2,
    }
    np.savez(save_dir / "final_results.npz", **results)
    print(f"Results saved to {save_dir / 'final_results.npz'}")

    return results


if __name__ == "__main__":
    results = main()
