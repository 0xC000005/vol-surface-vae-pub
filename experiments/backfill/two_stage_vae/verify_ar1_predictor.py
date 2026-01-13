"""
Verify AR(1) Predictor: ACF, Kurtosis, and CI Calibration

Compare the standard predictor (independent z sampling) vs AR(1) predictor
(correlated z sampling) on key metrics:
1. Autocorrelation (ACF) - should improve from -0.02 toward GT -0.22
2. Cumulative return kurtosis - should improve from 0.55 toward GT 6.99
3. CI calibration - should improve

Usage:
    python experiments/backfill/two_stage_vae/verify_ar1_predictor.py
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

    print(f"VAE loaded: latent_dim={config['latent_dim']}")
    print(f"Standard predictor loaded")
    print(f"AR(1) predictor loaded: rho={ar1_predictor.rho.item():.4f}, sigma={ar1_predictor.sigma.item():.4f}")

    return vae, std_predictor, ar1_predictor, config


def generate_trajectories_standard(vae, predictor, log_returns, surfaces, start_idx,
                                    context_len=20, horizon=30, n_samples=100, device="cuda"):
    """Generate trajectories using standard predictor (independent z sampling)."""
    context = log_returns[start_idx:start_idx + context_len]
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]
    gt_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    sample_returns = np.zeros((n_samples, horizon))

    with torch.no_grad():
        # Get context embedding
        ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})

        # Get z mean and logvar from predictor
        z_mean, z_logvar = predictor(context_tensor, horizon=horizon)

        for s in range(n_samples):
            # Sample z INDEPENDENTLY for each timestep (standard approach)
            z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

            # Build ctx_emb for full sequence
            ctx_dim = ctx_emb_context.shape[-1]
            ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            # Decode
            _, sample, _, _ = vae.decoder(ctx_emb, torch.cat([z_mean[:, :0], z], dim=1), sample=True)

            # Extract returns for horizon positions
            for h in range(horizon):
                sample_returns[s, h] = sample[0, h, 2, 2].cpu().numpy()

    return gt_returns, sample_returns, initial_iv


def generate_trajectories_ar1(vae, predictor, log_returns, surfaces, start_idx,
                               context_len=20, horizon=30, n_samples=100, device="cuda"):
    """Generate trajectories using AR(1) predictor (correlated z sampling)."""
    context = log_returns[start_idx:start_idx + context_len]
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]
    gt_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    sample_returns = np.zeros((n_samples, horizon))

    with torch.no_grad():
        # Get context embedding
        ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})

        # Get z mean from predictor
        z_mean, _ = predictor(context_tensor, horizon=horizon)

        # Sample z with AR(1) correlation
        z_samples = predictor.sample_z(z_mean, n_samples=n_samples)  # (n_samples, 1, H, D)

        for s in range(n_samples):
            z = z_samples[s]  # (1, H, D)

            # Build ctx_emb for full sequence
            ctx_dim = ctx_emb_context.shape[-1]
            ctx_emb_future = torch.zeros(1, horizon, ctx_dim, device=device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            # Decode
            _, sample, _, _ = vae.decoder(ctx_emb, torch.cat([z_mean[:, :0], z], dim=1), sample=True)

            # Extract returns for horizon positions
            for h in range(horizon):
                sample_returns[s, h] = sample[0, h, 2, 2].cpu().numpy()

    return gt_returns, sample_returns, initial_iv


def compute_metrics(gt_returns, sample_returns):
    """Compute ACF, kurtosis, and other metrics."""
    horizon = len(gt_returns)
    n_samples = sample_returns.shape[0]

    # Cumulative returns for GT
    gt_cumsum = np.cumsum(gt_returns)

    # Cumulative returns for samples
    sample_cumsum = np.cumsum(sample_returns, axis=1)

    # ACF of returns (lag-1)
    gt_acf = np.corrcoef(gt_returns[:-1], gt_returns[1:])[0, 1] if len(gt_returns) > 1 else 0

    sample_acfs = []
    for s in range(n_samples):
        if len(sample_returns[s]) > 1:
            acf = np.corrcoef(sample_returns[s, :-1], sample_returns[s, 1:])[0, 1]
            if not np.isnan(acf):
                sample_acfs.append(acf)
    model_acf = np.mean(sample_acfs) if sample_acfs else 0

    # Kurtosis of cumulative returns at horizon
    gt_kurt = stats.kurtosis(gt_cumsum, fisher=True) if len(gt_cumsum) > 3 else 0
    model_kurt = stats.kurtosis(sample_cumsum[:, -1], fisher=True) if n_samples > 3 else 0

    # Std of cumulative returns at horizon
    gt_std = np.std(gt_cumsum)
    model_std = np.std(sample_cumsum[:, -1])

    return {
        "gt_acf": gt_acf,
        "model_acf": model_acf,
        "gt_kurt": gt_kurt,
        "model_kurt": model_kurt,
        "gt_std": gt_std,
        "model_std": model_std,
    }


def main():
    print("=" * 70)
    print("Verifying AR(1) Predictor: ACF, Kurtosis, CI Calibration")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load models
    vae, std_predictor, ar1_predictor, config = load_models(device)

    # Test on multiple starting points
    test_indices = [2100, 2150, 2280, 2900, 3500, 4000]  # Different market regimes
    context_len = 20
    horizon = 30
    n_samples = 100

    print("\n" + "=" * 70)
    print("GENERATING TRAJECTORIES")
    print("=" * 70)

    std_metrics_all = []
    ar1_metrics_all = []

    for idx in test_indices:
        print(f"\nStarting index {idx}...")

        try:
            # Generate with standard predictor
            gt_ret, std_ret, _ = generate_trajectories_standard(
                vae, std_predictor, log_returns, surfaces, idx,
                context_len=context_len, horizon=horizon, n_samples=n_samples, device=device
            )
            std_metrics = compute_metrics(gt_ret, std_ret)
            std_metrics_all.append(std_metrics)

            # Generate with AR(1) predictor
            _, ar1_ret, _ = generate_trajectories_ar1(
                vae, ar1_predictor, log_returns, surfaces, idx,
                context_len=context_len, horizon=horizon, n_samples=n_samples, device=device
            )
            ar1_metrics = compute_metrics(gt_ret, ar1_ret)
            ar1_metrics_all.append(ar1_metrics)

            print(f"  Standard ACF: {std_metrics['model_acf']:.4f} | AR(1) ACF: {ar1_metrics['model_acf']:.4f} | GT ACF: {std_metrics['gt_acf']:.4f}")
            print(f"  Standard Kurt: {std_metrics['model_kurt']:.2f} | AR(1) Kurt: {ar1_metrics['model_kurt']:.2f} | GT Kurt: {std_metrics['gt_kurt']:.2f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    std_acf = np.mean([m["model_acf"] for m in std_metrics_all])
    ar1_acf = np.mean([m["model_acf"] for m in ar1_metrics_all])
    gt_acf = np.mean([m["gt_acf"] for m in std_metrics_all])

    std_kurt = np.mean([m["model_kurt"] for m in std_metrics_all])
    ar1_kurt = np.mean([m["model_kurt"] for m in ar1_metrics_all])
    gt_kurt = np.mean([m["gt_kurt"] for m in std_metrics_all])

    std_std = np.mean([m["model_std"] for m in std_metrics_all])
    ar1_std = np.mean([m["model_std"] for m in ar1_metrics_all])
    gt_std = np.mean([m["gt_std"] for m in std_metrics_all])

    print(f"\n{'Metric':<20} {'GT':<12} {'Standard':<12} {'AR(1)':<12} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'ACF(1)':<20} {gt_acf:<12.4f} {std_acf:<12.4f} {ar1_acf:<12.4f} {(ar1_acf - std_acf):.4f}")
    print(f"{'Kurtosis (h=30)':<20} {gt_kurt:<12.2f} {std_kurt:<12.2f} {ar1_kurt:<12.2f} {(ar1_kurt - std_kurt):.2f}")
    print(f"{'Std (h=30)':<20} {gt_std:<12.4f} {std_std:<12.4f} {ar1_std:<12.4f} {(ar1_std - std_std):.4f}")

    # ACF gap analysis
    std_acf_gap = abs(gt_acf - std_acf)
    ar1_acf_gap = abs(gt_acf - ar1_acf)
    acf_improvement = (std_acf_gap - ar1_acf_gap) / std_acf_gap * 100 if std_acf_gap > 0 else 0

    print(f"\n{'ACF Gap to GT':<20} {'-':<12} {std_acf_gap:<12.4f} {ar1_acf_gap:<12.4f} {acf_improvement:.1f}% closer")

    # Kurtosis gap analysis
    std_kurt_gap = abs(gt_kurt - std_kurt)
    ar1_kurt_gap = abs(gt_kurt - ar1_kurt)
    kurt_improvement = (std_kurt_gap - ar1_kurt_gap) / std_kurt_gap * 100 if std_kurt_gap > 0 else 0

    print(f"{'Kurtosis Gap to GT':<20} {'-':<12} {std_kurt_gap:<12.2f} {ar1_kurt_gap:<12.2f} {kurt_improvement:.1f}% closer")

    # Additional diagnostic: check z autocorrelation
    print("\n" + "=" * 70)
    print("Z AUTOCORRELATION DIAGNOSTIC")
    print("=" * 70)

    # Sample z from AR(1) predictor and check ACF
    context_tensor = torch.tensor(
        log_returns[2100:2120], dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        z_mean, _ = ar1_predictor(context_tensor, horizon=30)
        z_samples = ar1_predictor.sample_z(z_mean, n_samples=1000)  # (1000, 1, 30, 8)

    z_samples_np = z_samples[:, 0, :, 0].cpu().numpy()  # (1000, 30) first latent dim

    # Compute ACF of z samples
    z_acfs = []
    for s in range(1000):
        acf = np.corrcoef(z_samples_np[s, :-1], z_samples_np[s, 1:])[0, 1]
        if not np.isnan(acf):
            z_acfs.append(acf)

    print(f"Z sample ACF(1): {np.mean(z_acfs):.4f} (expected ~{ar1_predictor.rho.item():.4f} based on rho)")

    # Save results
    save_dir = Path("models/backfill/two_stage/prior_network_ar1")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "standard_metrics": std_metrics_all,
        "ar1_metrics": ar1_metrics_all,
        "aggregate": {
            "gt_acf": gt_acf,
            "std_acf": std_acf,
            "ar1_acf": ar1_acf,
            "gt_kurt": gt_kurt,
            "std_kurt": std_kurt,
            "ar1_kurt": ar1_kurt,
            "acf_improvement_pct": acf_improvement,
            "kurt_improvement_pct": kurt_improvement,
        }
    }

    np.savez(save_dir / "verification_results.npz", **results)
    print(f"\nResults saved to {save_dir / 'verification_results.npz'}")

    return results


if __name__ == "__main__":
    results = main()
