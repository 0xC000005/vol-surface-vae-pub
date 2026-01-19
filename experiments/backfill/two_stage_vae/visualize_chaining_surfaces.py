"""
Visualize Surfaces During Autoregressive Chaining

Creates visual plots of actual 5×5 volatility surfaces at each hop
to examine if generated surfaces look realistic.

Usage:
    python experiments/backfill/two_stage_vae/visualize_chaining_surfaces.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP


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


def generate_chained_with_surfaces(
    vae, initial_context, n_hops=4, horizon=30, overlap=15, n_samples=10, device="cuda"
):
    """
    Generate chained sequence and return all surfaces for visualization.

    Returns:
        all_generated: list of (n_samples, horizon, 5, 5) per hop
        contexts: list of (context_len, 5, 5) context used for each hop
    """
    context_len = initial_context.shape[0]

    current_context = initial_context.copy()
    all_generated = []
    contexts = [current_context.copy()]

    for hop in range(n_hops):
        print(f"Generating hop {hop + 1}/{n_hops}...")

        context_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).to(device)
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

        all_generated.append(generated)

        # Update context for next hop
        if hop < n_hops - 1:
            mean_generated = generated.mean(axis=0)
            new_context = np.concatenate([
                current_context[overlap:],
                mean_generated[:overlap]
            ], axis=0)
            current_context = new_context
            contexts.append(current_context.copy())

    return all_generated, contexts


def plot_surface_comparison(gt_surfaces, gen_surfaces, hop, day, save_dir):
    """
    Plot side-by-side comparison of GT vs Generated surface for a specific day.

    Args:
        gt_surfaces: (horizon, 5, 5) ground truth
        gen_surfaces: (n_samples, horizon, 5, 5) generated
        hop: hop number
        day: day within hop
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    gt = gt_surfaces[day]
    gen_mean = gen_surfaces[:, day].mean(axis=0)
    gen_std = gen_surfaces[:, day].std(axis=0)

    # Common colorbar range
    vmin = min(gt.min(), gen_mean.min())
    vmax = max(gt.max(), gen_mean.max())

    # GT surface
    im1 = axes[0].imshow(gt, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth (Hop {hop}, Day {day})')
    axes[0].set_xlabel('Moneyness')
    axes[0].set_ylabel('Maturity')
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['0.70', '0.85', '1.00', '1.15', '1.30'], fontsize=8)
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(['1M', '3M', '6M', '1Y', '2Y'], fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='Log-return')

    # Add values
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f'{gt[i,j]:.3f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(gt[i,j]) < 0.02 else 'white')

    # Generated mean
    im2 = axes[1].imshow(gen_mean, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Generated Mean (Hop {hop}, Day {day})')
    axes[1].set_xlabel('Moneyness')
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(['0.70', '0.85', '1.00', '1.15', '1.30'], fontsize=8)
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(['1M', '3M', '6M', '1Y', '2Y'], fontsize=8)
    plt.colorbar(im2, ax=axes[1], label='Log-return')

    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, f'{gen_mean[i,j]:.3f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(gen_mean[i,j]) < 0.02 else 'white')

    # Difference
    diff = gen_mean - gt
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.02, vmax=0.02)
    axes[2].set_title(f'Difference (Gen - GT)')
    axes[2].set_xlabel('Moneyness')
    axes[2].set_xticks(range(5))
    axes[2].set_xticklabels(['0.70', '0.85', '1.00', '1.15', '1.30'], fontsize=8)
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(['1M', '3M', '6M', '1Y', '2Y'], fontsize=8)
    plt.colorbar(im3, ax=axes[2], label='Difference')

    for i in range(5):
        for j in range(5):
            axes[2].text(j, i, f'{diff[i,j]:.3f}', ha='center', va='center',
                        fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(save_dir / f'surface_hop{hop}_day{day}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_smile_comparison(gt_surfaces, gen_surfaces, hop, save_dir):
    """
    Plot volatility smile (moneyness dimension) at 6M maturity for multiple days.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    moneyness = [0.70, 0.85, 1.00, 1.15, 1.30]
    days_to_plot = [0, 5, 10, 15, 20, 25]

    for idx, day in enumerate(days_to_plot):
        if day >= gt_surfaces.shape[0]:
            continue

        ax = axes[idx]

        # GT smile at 6M maturity (row 2)
        gt_smile = gt_surfaces[day, 2, :]
        ax.plot(moneyness, gt_smile, 'ko-', linewidth=2, markersize=8, label='GT')

        # Generated samples
        for s in range(min(5, gen_surfaces.shape[0])):
            gen_smile = gen_surfaces[s, day, 2, :]
            ax.plot(moneyness, gen_smile, 'b-', alpha=0.3, linewidth=1)

        # Generated mean
        gen_mean_smile = gen_surfaces[:, day, 2, :].mean(axis=0)
        ax.plot(moneyness, gen_mean_smile, 'b--', linewidth=2, label='Gen Mean')

        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Log-return')
        ax.set_title(f'Hop {hop}, Day {day} - 6M Smile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / f'smile_hop{hop}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_term_structure_comparison(gt_surfaces, gen_surfaces, hop, save_dir):
    """
    Plot term structure (maturity dimension) at ATM for multiple days.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    maturities = ['1M', '3M', '6M', '1Y', '2Y']
    mat_idx = [0, 1, 2, 3, 4]
    days_to_plot = [0, 5, 10, 15, 20, 25]

    for idx, day in enumerate(days_to_plot):
        if day >= gt_surfaces.shape[0]:
            continue

        ax = axes[idx]

        # GT term structure at ATM (column 2)
        gt_term = gt_surfaces[day, :, 2]
        ax.plot(mat_idx, gt_term, 'ko-', linewidth=2, markersize=8, label='GT')

        # Generated samples
        for s in range(min(5, gen_surfaces.shape[0])):
            gen_term = gen_surfaces[s, day, :, 2]
            ax.plot(mat_idx, gen_term, 'r-', alpha=0.3, linewidth=1)

        # Generated mean
        gen_mean_term = gen_surfaces[:, day, :, 2].mean(axis=0)
        ax.plot(mat_idx, gen_mean_term, 'r--', linewidth=2, label='Gen Mean')

        ax.set_xticks(mat_idx)
        ax.set_xticklabels(maturities)
        ax.set_xlabel('Maturity')
        ax.set_ylabel('Log-return')
        ax.set_title(f'Hop {hop}, Day {day} - ATM Term Structure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / f'term_structure_hop{hop}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_surface_evolution(all_generated, save_dir):
    """
    Plot how a single sample's surface evolves across all hops.
    """
    n_hops = len(all_generated)

    fig, axes = plt.subplots(n_hops, 4, figsize=(16, 4 * n_hops))

    sample_idx = 0  # Use first sample
    days_to_show = [0, 10, 20, 29]  # Days within each hop

    for hop_idx, generated in enumerate(all_generated):
        for day_idx, day in enumerate(days_to_show):
            ax = axes[hop_idx, day_idx]

            surface = generated[sample_idx, day]
            im = ax.imshow(surface, cmap='RdBu_r', vmin=-0.05, vmax=0.05)

            ax.set_title(f'Hop {hop_idx + 1}, Day {day}')
            if day_idx == 0:
                ax.set_ylabel('Maturity')
            if hop_idx == n_hops - 1:
                ax.set_xlabel('Moneyness')

            ax.set_xticks(range(5))
            ax.set_xticklabels(['0.70', '0.85', '1.00', '1.15', '1.30'], fontsize=7, rotation=45)
            ax.set_yticks(range(5))
            ax.set_yticklabels(['1M', '3M', '6M', '1Y', '2Y'], fontsize=7)

            plt.colorbar(im, ax=ax)

    plt.suptitle('Surface Evolution Across Hops (Single Sample)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'surface_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Generate and visualize surfaces during chaining."""
    print("=" * 70)
    print("Visualize Surfaces During Autoregressive Chaining")
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
    n_samples = 10  # Fewer samples for visualization

    # Pick starting point
    train_end = int(len(log_returns) * 0.7)
    start_idx = train_end + 100

    initial_context = log_returns[start_idx:start_idx + context_len]
    gt_log_returns = log_returns[start_idx + context_len:]

    print(f"\nGenerating {n_hops} hops with {n_samples} samples each...")

    # Generate
    all_generated, contexts = generate_chained_with_surfaces(
        vae, initial_context, n_hops=n_hops, horizon=horizon,
        overlap=overlap, n_samples=n_samples, device=device
    )

    # Create output directory
    save_dir = Path("results/two_stage_vae/chaining_surfaces")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    for hop in range(n_hops):
        print(f"  Hop {hop + 1}...")

        start_day = hop * (horizon - overlap)
        end_day = start_day + horizon
        gt_hop = gt_log_returns[start_day:end_day]
        gen_hop = all_generated[hop]

        # Surface comparisons for day 0, 15, 29
        for day in [0, 15, 29]:
            plot_surface_comparison(gt_hop, gen_hop, hop + 1, day, save_dir)

        # Smile comparison
        plot_smile_comparison(gt_hop, gen_hop, hop + 1, save_dir)

        # Term structure comparison
        plot_term_structure_comparison(gt_hop, gen_hop, hop + 1, save_dir)

    # Surface evolution across hops
    plot_surface_evolution(all_generated, save_dir)

    print(f"\nVisualizationsaved to: {save_dir}")
    print("\nGenerated files:")
    for f in sorted(save_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
