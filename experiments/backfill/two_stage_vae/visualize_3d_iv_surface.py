"""
3D Implied Volatility Surface Visualization During Chaining

Shows 3D volatility surfaces in IV space (not log-returns) to visualize
the smile and term structure during autoregressive chaining.

Usage:
    python experiments/backfill/two_stage_vae/visualize_3d_iv_surface.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP


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


def to_log_returns(surfaces: np.ndarray):
    """Convert IV surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def log_returns_to_iv(log_returns, starting_iv):
    """
    Convert log-returns back to IV levels.

    Args:
        log_returns: (T, 5, 5) or (n_samples, T, 5, 5) log-returns
        starting_iv: (5, 5) starting IV surface

    Returns:
        iv_surfaces: IV surfaces
    """
    if log_returns.ndim == 3:
        # Single trajectory
        iv = np.zeros((log_returns.shape[0] + 1, 5, 5))
        iv[0] = starting_iv
        for t in range(log_returns.shape[0]):
            iv[t + 1] = iv[t] * np.exp(log_returns[t])
        return iv[1:]  # Return predicted IVs (not starting)
    else:
        # Multiple samples
        n_samples, T = log_returns.shape[:2]
        iv = np.zeros((n_samples, T + 1, 5, 5))
        iv[:, 0] = starting_iv
        for t in range(T):
            iv[:, t + 1] = iv[:, t] * np.exp(log_returns[:, t])
        return iv[:, 1:]


def plot_3d_surface(ax, surface, title, moneyness, maturities, zlim=None):
    """Plot a single 3D surface."""
    X, Y = np.meshgrid(moneyness, maturities)

    surf = ax.plot_surface(X, Y, surface, cmap='viridis', alpha=0.8,
                           linewidth=0.5, antialiased=True, edgecolor='gray')

    ax.set_xlabel('Moneyness', fontsize=10)
    ax.set_ylabel('Maturity (months)', fontsize=10)
    ax.set_zlabel('IV', fontsize=10)
    ax.set_title(title, fontsize=12)

    if zlim:
        ax.set_zlim(zlim)

    return surf


def generate_and_plot_3d(vae, surfaces, start_idx, context_len, horizon, n_samples, device, save_dir):
    """Generate surfaces and create 3D plots."""

    # Get log returns
    log_returns, log_surfaces = to_log_returns(surfaces)

    # Context and GT
    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]  # Last context day IV
    gt_iv = surfaces[start_idx + context_len:start_idx + context_len + horizon]

    # Generate predictions
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    print("Generating samples...")
    generated_log_returns = np.zeros((n_samples, horizon, 5, 5))

    with torch.no_grad():
        for h in range(horizon):
            if h == 0:
                batch = {"surface": context_tensor}
            else:
                gen_so_far = torch.tensor(
                    generated_log_returns[:, :h].mean(axis=0),
                    dtype=torch.float32
                ).unsqueeze(0).to(device)
                seq = torch.cat([context_tensor, gen_so_far], dim=1)
                batch = {"surface": seq}

            samples = vae.sample(batch, n_samples=n_samples)
            generated_log_returns[:, h] = samples[:, 0, -1].cpu().numpy()

    # Convert to IV space
    generated_iv = log_returns_to_iv(generated_log_returns, starting_iv)
    gen_mean_iv = generated_iv.mean(axis=0)

    # Grid coordinates
    moneyness = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    maturities = np.array([1, 3, 6, 12, 24])  # months

    # Common z limits
    all_ivs = np.concatenate([gt_iv.flatten(), gen_mean_iv.flatten()])
    zlim = (all_ivs.min() * 0.95, all_ivs.max() * 1.05)

    # Plot different days
    days_to_plot = [0, 14, 29]

    for day in days_to_plot:
        fig = plt.figure(figsize=(16, 6))

        # GT surface
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d_surface(ax1, gt_iv[day], f'Ground Truth (Day {day})',
                       moneyness, maturities, zlim)
        ax1.view_init(elev=25, azim=45)

        # Generated mean
        ax2 = fig.add_subplot(132, projection='3d')
        plot_3d_surface(ax2, gen_mean_iv[day], f'Generated Mean (Day {day})',
                       moneyness, maturities, zlim)
        ax2.view_init(elev=25, azim=45)

        # Difference
        ax3 = fig.add_subplot(133, projection='3d')
        diff = gen_mean_iv[day] - gt_iv[day]
        diff_zlim = (-0.05, 0.05)
        X, Y = np.meshgrid(moneyness, maturities)
        colors = np.where(diff > 0, 'red', 'blue')
        ax3.plot_surface(X, Y, diff, cmap='RdBu_r', alpha=0.8,
                        linewidth=0.5, antialiased=True)
        ax3.set_xlabel('Moneyness', fontsize=10)
        ax3.set_ylabel('Maturity (months)', fontsize=10)
        ax3.set_zlabel('IV Difference', fontsize=10)
        ax3.set_title(f'Difference (Gen - GT)', fontsize=12)
        ax3.set_zlim(diff_zlim)
        ax3.view_init(elev=25, azim=45)

        plt.tight_layout()
        plt.savefig(save_dir / f'3d_surface_day{day}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 3d_surface_day{day}.png")
        plt.close()

    # Create evolution plot (single figure with multiple surfaces)
    fig = plt.figure(figsize=(20, 10))

    for idx, day in enumerate([0, 7, 14, 21, 29]):
        # GT
        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
        plot_3d_surface(ax, gt_iv[day], f'GT Day {day}', moneyness, maturities, zlim)
        ax.view_init(elev=20, azim=45)

        # Generated
        ax = fig.add_subplot(2, 5, idx + 6, projection='3d')
        plot_3d_surface(ax, gen_mean_iv[day], f'Gen Day {day}', moneyness, maturities, zlim)
        ax.view_init(elev=20, azim=45)

    plt.suptitle('IV Surface Evolution: GT (top) vs Generated (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / '3d_surface_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: 3d_surface_evolution.png")
    plt.close()

    # Single sample trajectory
    fig = plt.figure(figsize=(20, 5))
    sample_iv = generated_iv[0]  # First sample

    for idx, day in enumerate([0, 7, 14, 21, 29]):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        plot_3d_surface(ax, sample_iv[day], f'Sample Day {day}', moneyness, maturities, zlim)
        ax.view_init(elev=20, azim=45)

    plt.suptitle('Single Sample IV Surface Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / '3d_single_sample_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: 3d_single_sample_evolution.png")
    plt.close()

    return gt_iv, gen_mean_iv, generated_iv


def main():
    print("=" * 70)
    print("3D IV Surface Visualization")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    print(f"\nData: {len(surfaces)} days of IV surfaces")
    print(f"IV range: {surfaces.min():.3f} to {surfaces.max():.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    vae, config = load_vae(device)

    # Configuration
    context_len = config.get("context_len", 30)
    horizon = 30
    n_samples = 10

    # Pick starting point
    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100

    print(f"\nContext length: {context_len}")
    print(f"Horizon: {horizon}")
    print(f"Starting from index: {start_idx}")

    # Create output directory
    save_dir = Path("results/two_stage_vae/3d_iv_surfaces")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate and plot
    gt_iv, gen_mean_iv, generated_iv = generate_and_plot_3d(
        vae, surfaces, start_idx, context_len, horizon, n_samples, device, save_dir
    )

    print(f"\nResults saved to: {save_dir}")

    # Print some statistics
    print("\n" + "=" * 70)
    print("IV Statistics")
    print("=" * 70)
    print(f"\nGT IV range: {gt_iv.min():.3f} - {gt_iv.max():.3f}")
    print(f"Gen IV range: {gen_mean_iv.min():.3f} - {gen_mean_iv.max():.3f}")
    print(f"Mean absolute difference: {np.abs(gen_mean_iv - gt_iv).mean():.4f}")


if __name__ == "__main__":
    main()
