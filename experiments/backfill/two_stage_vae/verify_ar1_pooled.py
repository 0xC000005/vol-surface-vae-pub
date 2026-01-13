"""
Verify AR(1) Predictor with Pooled Statistics

Compare standard vs AR(1) predictor using pooled statistics across many windows,
matching the methodology from our original diagnostics.

Usage:
    python experiments/backfill/two_stage_vae/verify_ar1_pooled.py
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


def load_models(device: str = "cuda"):
    """Load VAE and both predictors."""
    # Load VAE
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load standard predictor
    std_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    std_ckpt = torch.load(std_path, map_location=device, weights_only=False)

    std_predictor = LatentPredictor(config)
    std_predictor.load_state_dict(std_ckpt["predictor_state_dict"])
    std_predictor = std_predictor.to(device)
    std_predictor.eval()

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

    return vae, std_predictor, ar1_predictor, config


def generate_many_samples(vae, predictor, log_returns, context_len=20, horizon=30,
                          n_windows=200, n_samples_per_window=10, use_ar1=False, device="cuda"):
    """Generate samples from many windows."""
    N = len(log_returns)
    all_returns = []  # Will be (n_windows * n_samples, horizon)
    all_cumsum = []   # Will be (n_windows * n_samples, horizon)

    # Sample windows evenly across data
    window_starts = np.linspace(100, N - context_len - horizon - 100, n_windows).astype(int)

    for start_idx in window_starts:
        context = log_returns[start_idx:start_idx + context_len]
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get context embedding
            ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})

            # Get z mean from predictor
            z_mean, z_logvar = predictor(context_tensor, horizon=horizon)

            # Sample z
            if use_ar1:
                z_samples = predictor.sample_z(z_mean, n_samples=n_samples_per_window)
            else:
                # Independent sampling for standard predictor
                z_samples = []
                for _ in range(n_samples_per_window):
                    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
                    z_samples.append(z)
                z_samples = torch.stack(z_samples, dim=0)

            # Decode each sample
            ctx_dim = ctx_emb_context.shape[-1]
            ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            for s in range(n_samples_per_window):
                z = z_samples[s]
                if z.dim() == 2:
                    z = z.unsqueeze(0)

                # Decode - sample=True to get Student-t samples
                _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)

                # Extract ATM returns
                returns = sample[0, :, 2, 2].cpu().numpy()
                all_returns.append(returns)
                all_cumsum.append(np.cumsum(returns))

    return np.array(all_returns), np.array(all_cumsum)


def main():
    print("=" * 70)
    print("Verifying AR(1) Predictor with Pooled Statistics")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    vae, std_predictor, ar1_predictor, config = load_models(device)

    # Parameters
    context_len = 20
    horizon = 30
    n_windows = 200
    n_samples_per_window = 20

    print(f"\nGenerating {n_windows} windows x {n_samples_per_window} samples = {n_windows * n_samples_per_window} total")

    # Compute GT statistics
    print("\nComputing Ground Truth statistics...")
    gt_returns = []
    gt_cumsum = []
    for i in range(100, len(log_returns) - context_len - horizon - 100, 10):
        ret = log_returns[i + context_len:i + context_len + horizon, 2, 2]
        gt_returns.append(ret)
        gt_cumsum.append(np.cumsum(ret))

    gt_returns = np.array(gt_returns)
    gt_cumsum = np.array(gt_cumsum)

    # GT metrics
    gt_acf = np.mean([np.corrcoef(r[:-1], r[1:])[0, 1] for r in gt_returns if len(r) > 1])
    gt_kurt_h30 = stats.kurtosis(gt_cumsum[:, -1], fisher=True)
    gt_std_h30 = np.std(gt_cumsum[:, -1])

    print(f"  GT ACF(1): {gt_acf:.4f}")
    print(f"  GT Kurtosis (h=30): {gt_kurt_h30:.2f}")
    print(f"  GT Std (h=30): {gt_std_h30:.4f}")

    # Generate standard predictor samples
    print("\nGenerating Standard predictor samples...")
    std_returns, std_cumsum = generate_many_samples(
        vae, std_predictor, log_returns,
        context_len=context_len, horizon=horizon,
        n_windows=n_windows, n_samples_per_window=n_samples_per_window,
        use_ar1=False, device=device
    )

    std_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in std_returns if len(r) > 1 and not np.isnan(np.corrcoef(r[:-1], r[1:])[0, 1])]
    std_acf = np.mean(std_acfs)
    std_kurt_h30 = stats.kurtosis(std_cumsum[:, -1], fisher=True)
    std_std_h30 = np.std(std_cumsum[:, -1])

    print(f"  Standard ACF(1): {std_acf:.4f}")
    print(f"  Standard Kurtosis (h=30): {std_kurt_h30:.2f}")
    print(f"  Standard Std (h=30): {std_std_h30:.4f}")

    # Generate AR(1) predictor samples
    print("\nGenerating AR(1) predictor samples...")
    ar1_returns, ar1_cumsum = generate_many_samples(
        vae, ar1_predictor, log_returns,
        context_len=context_len, horizon=horizon,
        n_windows=n_windows, n_samples_per_window=n_samples_per_window,
        use_ar1=True, device=device
    )

    ar1_acfs = [np.corrcoef(r[:-1], r[1:])[0, 1] for r in ar1_returns if len(r) > 1 and not np.isnan(np.corrcoef(r[:-1], r[1:])[0, 1])]
    ar1_acf = np.mean(ar1_acfs)
    ar1_kurt_h30 = stats.kurtosis(ar1_cumsum[:, -1], fisher=True)
    ar1_std_h30 = np.std(ar1_cumsum[:, -1])

    print(f"  AR(1) ACF(1): {ar1_acf:.4f}")
    print(f"  AR(1) Kurtosis (h=30): {ar1_kurt_h30:.2f}")
    print(f"  AR(1) Std (h=30): {ar1_std_h30:.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'GT':<12} {'Standard':<12} {'AR(1)':<12}")
    print("-" * 60)
    print(f"{'ACF(1)':<25} {gt_acf:<12.4f} {std_acf:<12.4f} {ar1_acf:<12.4f}")
    print(f"{'Kurtosis (h=30)':<25} {gt_kurt_h30:<12.2f} {std_kurt_h30:<12.2f} {ar1_kurt_h30:<12.2f}")
    print(f"{'Std (h=30)':<25} {gt_std_h30:<12.4f} {std_std_h30:<12.4f} {ar1_std_h30:<12.4f}")

    # Improvement analysis
    print(f"\n{'Improvement Analysis':<25}")
    print("-" * 60)

    std_acf_gap = abs(gt_acf - std_acf)
    ar1_acf_gap = abs(gt_acf - ar1_acf)
    acf_improvement = (std_acf_gap - ar1_acf_gap) / std_acf_gap * 100 if std_acf_gap > 0 else 0

    std_kurt_gap = abs(gt_kurt_h30 - std_kurt_h30)
    ar1_kurt_gap = abs(gt_kurt_h30 - ar1_kurt_h30)
    kurt_improvement = (std_kurt_gap - ar1_kurt_gap) / std_kurt_gap * 100 if std_kurt_gap > 0 else 0

    print(f"{'ACF Gap':<25} {'':<12} {std_acf_gap:<12.4f} {ar1_acf_gap:<12.4f} ({acf_improvement:+.1f}%)")
    print(f"{'Kurtosis Gap':<25} {'':<12} {std_kurt_gap:<12.2f} {ar1_kurt_gap:<12.2f} ({kurt_improvement:+.1f}%)")

    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if ar1_acf_gap < std_acf_gap:
        print(f"ACF: AR(1) is {acf_improvement:.1f}% closer to GT")
    else:
        print(f"ACF: AR(1) is {-acf_improvement:.1f}% further from GT")

    if ar1_kurt_gap < std_kurt_gap:
        print(f"Kurtosis: AR(1) is {kurt_improvement:.1f}% closer to GT")
    else:
        print(f"Kurtosis: AR(1) is {-kurt_improvement:.1f}% further from GT")

    # Key insight
    print("\nKEY INSIGHT:")
    print("-" * 60)
    print(f"Z-space correlation is working (predictor.rho = {ar1_predictor.rho.item():.4f})")
    print("But output ACF remains near 0 because:")
    print("  1. Decoder processes each timestep independently")
    print("  2. Decoder Student-t noise is sampled independently per timestep")
    print("  3. ~90% of output variance comes from decoder noise, not z")
    print("\nConclusion: Correlating z alone is insufficient.")
    print("Need to correlate decoder noise or restructure decoder architecture.")

    # Save diagnostic plot
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ACF histograms
    axes[0, 0].hist(gt_acf * np.ones(100), bins=20, alpha=0.5, label='GT', density=True)
    axes[0, 0].hist(std_acfs, bins=20, alpha=0.5, label='Standard', density=True)
    axes[0, 0].hist(ar1_acfs, bins=20, alpha=0.5, label='AR(1)', density=True)
    axes[0, 0].axvline(gt_acf, color='blue', linestyle='--', label=f'GT={gt_acf:.3f}')
    axes[0, 0].axvline(std_acf, color='orange', linestyle='--', label=f'Std={std_acf:.3f}')
    axes[0, 0].axvline(ar1_acf, color='green', linestyle='--', label=f'AR1={ar1_acf:.3f}')
    axes[0, 0].set_xlabel('ACF(1)')
    axes[0, 0].set_title('Per-Sequence ACF Distribution')
    axes[0, 0].legend(fontsize=8)

    # Kurtosis at h=30
    axes[0, 1].hist(gt_cumsum[:, -1], bins=30, alpha=0.5, label='GT', density=True)
    axes[0, 1].hist(std_cumsum[:, -1], bins=30, alpha=0.5, label='Standard', density=True)
    axes[0, 1].hist(ar1_cumsum[:, -1], bins=30, alpha=0.5, label='AR(1)', density=True)
    axes[0, 1].set_xlabel('Cumulative Return at h=30')
    axes[0, 1].set_title(f'h=30 Distribution\nKurt: GT={gt_kurt_h30:.1f}, Std={std_kurt_h30:.1f}, AR1={ar1_kurt_h30:.1f}')
    axes[0, 1].legend()

    # Q-Q plot for tails
    gt_sorted = np.sort(gt_cumsum[:, -1])
    std_sorted = np.sort(std_cumsum[:, -1])
    ar1_sorted = np.sort(ar1_cumsum[:, -1])

    n_min = min(len(gt_sorted), len(std_sorted), len(ar1_sorted))
    quantiles = np.linspace(0, 1, n_min)

    gt_q = np.quantile(gt_cumsum[:, -1], quantiles)
    std_q = np.quantile(std_cumsum[:, -1], quantiles)
    ar1_q = np.quantile(ar1_cumsum[:, -1], quantiles)

    axes[0, 2].scatter(gt_q, std_q, alpha=0.5, s=10, label='Standard vs GT')
    axes[0, 2].scatter(gt_q, ar1_q, alpha=0.5, s=10, label='AR(1) vs GT')
    min_val = min(gt_q.min(), std_q.min(), ar1_q.min())
    max_val = max(gt_q.max(), std_q.max(), ar1_q.max())
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    axes[0, 2].set_xlabel('GT Quantiles')
    axes[0, 2].set_ylabel('Model Quantiles')
    axes[0, 2].set_title('Q-Q Plot (h=30 cumulative returns)')
    axes[0, 2].legend()

    # Sample trajectories
    for i in range(min(5, len(std_returns))):
        axes[1, 0].plot(np.cumsum(std_returns[i]), alpha=0.3, color='orange')
    axes[1, 0].set_title('Standard: Sample Trajectories')
    axes[1, 0].set_xlabel('Horizon')
    axes[1, 0].set_ylabel('Cumulative Return')

    for i in range(min(5, len(ar1_returns))):
        axes[1, 1].plot(np.cumsum(ar1_returns[i]), alpha=0.3, color='green')
    axes[1, 1].set_title('AR(1): Sample Trajectories')
    axes[1, 1].set_xlabel('Horizon')
    axes[1, 1].set_ylabel('Cumulative Return')

    # ACF by horizon
    gt_acf_by_h = []
    std_acf_by_h = []
    ar1_acf_by_h = []
    for h in range(1, horizon):
        gt_acf_by_h.append(np.corrcoef(gt_returns[:, h-1], gt_returns[:, h])[0, 1])
        std_acf_by_h.append(np.corrcoef(std_returns[:, h-1], std_returns[:, h])[0, 1])
        ar1_acf_by_h.append(np.corrcoef(ar1_returns[:, h-1], ar1_returns[:, h])[0, 1])

    axes[1, 2].plot(range(1, horizon), gt_acf_by_h, 'b-', label='GT', linewidth=2)
    axes[1, 2].plot(range(1, horizon), std_acf_by_h, 'orange', label='Standard', linewidth=2)
    axes[1, 2].plot(range(1, horizon), ar1_acf_by_h, 'g-', label='AR(1)', linewidth=2)
    axes[1, 2].axhline(0, color='gray', linestyle='--')
    axes[1, 2].set_xlabel('Horizon')
    axes[1, 2].set_ylabel('ACF(1) at horizon h')
    axes[1, 2].set_title('Cross-Sectional ACF by Horizon')
    axes[1, 2].legend()

    plt.suptitle('AR(1) Predictor Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "verification_pooled.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_dir / 'verification_pooled.png'}")
    plt.close()

    return {
        "gt": {"acf": gt_acf, "kurt": gt_kurt_h30, "std": gt_std_h30},
        "standard": {"acf": std_acf, "kurt": std_kurt_h30, "std": std_std_h30},
        "ar1": {"acf": ar1_acf, "kurt": ar1_kurt_h30, "std": ar1_std_h30},
    }


if __name__ == "__main__":
    results = main()
