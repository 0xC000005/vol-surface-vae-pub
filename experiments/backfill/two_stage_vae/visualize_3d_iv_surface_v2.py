"""
3D Implied Volatility Surface Visualization (v2 - with clipping)

Shows 3D volatility surfaces in IV space with proper handling of outliers.

Usage:
    python experiments/backfill/two_stage_vae/visualize_3d_iv_surface_v2.py
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
    ]

    for vae_path in vae_paths:
        if Path(vae_path).exists():
            vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
            config = vae_ckpt["model_config"]
            config["device"] = device
            vae = CVAETwoStageStudentTMLP(config)
            vae.load_state_dict(vae_ckpt["model_state_dict"])
            vae = vae.to(device)
            vae.eval()
            return vae, config

    raise FileNotFoundError("No VAE model found")


def to_log_returns(surfaces):
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns


def plot_3d_iv_surface(ax, surface, title, moneyness, maturities):
    """Plot a single 3D IV surface."""
    X, Y = np.meshgrid(moneyness, maturities)

    surf = ax.plot_surface(X, Y, surface, cmap='viridis', alpha=0.9,
                           linewidth=0.3, antialiased=True, edgecolor='darkgray')

    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Maturity (mo)')
    ax.set_zlabel('IV')
    ax.set_title(title)

    return surf


def main():
    print("=" * 70)
    print("3D IV Surface Visualization (Proper Scale)")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # IV surfaces
    log_returns = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, config = load_vae(device)

    context_len = 30
    horizon = 30
    n_samples = 20

    # Pick a point in validation set
    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100

    # Get context and starting IV
    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]
    gt_surfaces = surfaces[start_idx + context_len:start_idx + context_len + horizon]

    print(f"\nStarting IV (ATM): {starting_iv[2,2]:.3f}")
    print(f"GT IV range: {gt_surfaces.min():.3f} - {gt_surfaces.max():.3f}")

    # Generate step by step, converting to IV at each step
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    # Generate samples and track IV evolution
    gen_iv_samples = np.zeros((n_samples, horizon, 5, 5))

    print("\nGenerating samples...")
    with torch.no_grad():
        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()

            for h in range(horizon):
                ctx_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).to(device)
                batch = {"surface": ctx_tensor}

                samples = vae.sample(batch, n_samples=1)
                log_ret = samples[0, 0, -1].cpu().numpy()

                # Clip log returns to prevent explosion
                log_ret = np.clip(log_ret, -0.5, 0.5)

                # Convert to IV
                new_iv = current_iv * np.exp(log_ret)

                # Clip IV to reasonable range
                new_iv = np.clip(new_iv, 0.01, 2.0)

                gen_iv_samples[s, h] = new_iv
                current_iv = new_iv

                # Update context with the new log return
                current_context = np.concatenate([current_context[1:], log_ret[np.newaxis]], axis=0)

    gen_mean = gen_iv_samples.mean(axis=0)

    print(f"Generated IV range: {gen_iv_samples.min():.3f} - {gen_iv_samples.max():.3f}")

    # Grid coordinates
    moneyness = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    maturities = np.array([1, 3, 6, 12, 24])

    # Create output directory
    save_dir = Path("results/two_stage_vae/3d_iv_surfaces_v2")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Common z-limits based on reasonable IV range
    zlim = (0, max(gt_surfaces.max(), gen_mean.max()) * 1.1)

    # Plot comparison for specific days
    for day in [0, 14, 29]:
        fig = plt.figure(figsize=(14, 5))

        # GT
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d_iv_surface(ax1, gt_surfaces[day], f'Ground Truth (Day {day})', moneyness, maturities)
        ax1.set_zlim(zlim)
        ax1.view_init(elev=25, azim=45)

        # Generated Mean
        ax2 = fig.add_subplot(132, projection='3d')
        plot_3d_iv_surface(ax2, gen_mean[day], f'Generated Mean (Day {day})', moneyness, maturities)
        ax2.set_zlim(zlim)
        ax2.view_init(elev=25, azim=45)

        # Overlay view (wireframe)
        ax3 = fig.add_subplot(133, projection='3d')
        X, Y = np.meshgrid(moneyness, maturities)
        ax3.plot_wireframe(X, Y, gt_surfaces[day], color='blue', alpha=0.6, label='GT')
        ax3.plot_wireframe(X, Y, gen_mean[day], color='red', alpha=0.6, label='Generated')
        ax3.set_xlabel('Moneyness')
        ax3.set_ylabel('Maturity (mo)')
        ax3.set_zlabel('IV')
        ax3.set_title(f'Overlay (Day {day})')
        ax3.set_zlim(zlim)
        ax3.view_init(elev=25, azim=45)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(save_dir / f'3d_iv_day{day}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 3d_iv_day{day}.png")
        plt.close()

    # Evolution comparison
    fig = plt.figure(figsize=(20, 8))
    days = [0, 7, 14, 21, 29]

    for idx, day in enumerate(days):
        # GT row
        ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
        plot_3d_iv_surface(ax, gt_surfaces[day], f'GT Day {day}', moneyness, maturities)
        ax.set_zlim(zlim)
        ax.view_init(elev=20, azim=45)

        # Gen row
        ax = fig.add_subplot(2, 5, idx + 6, projection='3d')
        plot_3d_iv_surface(ax, gen_mean[day], f'Gen Day {day}', moneyness, maturities)
        ax.set_zlim(zlim)
        ax.view_init(elev=20, azim=45)

    plt.suptitle('IV Surface Evolution: Ground Truth (top) vs Generated Mean (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / '3d_iv_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: 3d_iv_evolution.png")
    plt.close()

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
