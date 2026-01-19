"""
3D IV Surface Visualization with Stable Autoregressive Chaining

Implements multiple sampling strategies to prevent explosion during chaining:
1. Mean-only: Use predicted mean, no sampling (deterministic, stable)
2. Truncated Student-t: Sample but reject extreme values (stochastic, stable)
3. Temperature-scaled: Reduce scale parameter during sampling (stochastic, stable)

Compares against:
- Ground truth evolution
- Original sampling (may explode)

Usage:
    python experiments/backfill/two_stage_vae/visualize_3d_iv_stable_chaining.py
    python experiments/backfill/two_stage_vae/visualize_3d_iv_stable_chaining.py --method truncated
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


def sample_truncated_student_t(model, batch, n_samples=1, truncate_bounds=(-0.5, 0.5), max_attempts=100):
    """
    Sample from Student-t with truncation (rejection sampling).

    Args:
        model: VAE model
        batch: Input batch with context
        n_samples: Number of samples to generate
        truncate_bounds: (lower, upper) bounds for log-returns
        max_attempts: Max rejection sampling attempts

    Returns:
        samples: (n_samples, T, 5, 5) tensor of samples within bounds
    """
    device = batch["surface"].device
    surface = batch["surface"]
    B, T = surface.shape[:2]

    ctx_emb = model.ctx_encoder({"surface": surface})
    z_mean, z_logvar, _ = model.main_encoder({"surface": surface})
    z_std = torch.exp(0.5 * z_logvar)

    valid_samples = []

    for _ in range(n_samples):
        # Try to get a valid sample
        for attempt in range(max_attempts):
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z
            _, sample, _, _ = model.decoder(ctx_emb, z, sample=True)

            # Check bounds on the last timestep (prediction)
            last_sample = sample[:, -1]  # (B, 5, 5)

            if (last_sample >= truncate_bounds[0]).all() and (last_sample <= truncate_bounds[1]).all():
                valid_samples.append(sample)
                break
        else:
            # If all attempts failed, use clipped mean
            mean, _, _, _ = model.decoder(ctx_emb, z_mean, sample=False)
            valid_samples.append(mean)

    return torch.stack(valid_samples, dim=0)


def sample_temperature_scaled(model, batch, n_samples=1, temperature=0.5):
    """
    Sample with reduced temperature/scale.

    Args:
        model: VAE model
        batch: Input batch with context
        n_samples: Number of samples
        temperature: Scale factor for sampling (< 1.0 = narrower distribution)

    Returns:
        samples: (n_samples, T, 5, 5) tensor
    """
    device = batch["surface"].device
    surface = batch["surface"]
    B, T = surface.shape[:2]

    ctx_emb = model.ctx_encoder({"surface": surface})
    z_mean, z_logvar, _ = model.main_encoder({"surface": surface})
    z_std = torch.exp(0.5 * z_logvar)

    samples = []

    for _ in range(n_samples):
        # Sample z with reduced variance
        eps_z = torch.randn_like(z_std) * temperature
        z = z_mean + z_std * eps_z

        # Get mean and add scaled noise
        mean, _, factor, log_diag = model.decoder(ctx_emb, z, sample=False)

        # Manual scaled sampling
        rank = factor.shape[-1]
        eps_rank = torch.randn(B, T, rank, device=device) * temperature
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)

        eps_diag = torch.randn(B, T, 25, device=device) * temperature
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag

        total_gauss = correlated_gauss + independent_gauss

        sample = mean + total_gauss.view(B, T, 5, 5)
        samples.append(sample)

    return torch.stack(samples, dim=0)


def get_mean_only(model, batch):
    """Get mean prediction without sampling (deterministic)."""
    surface = batch["surface"]
    ctx_emb = model.ctx_encoder({"surface": surface})
    z_mean, _, _ = model.main_encoder({"surface": surface})
    mean, _, _, _ = model.decoder(ctx_emb, z_mean, sample=False)
    return mean


def autoregressive_chain(model, context, starting_iv, horizon=30, method="mean_only",
                         n_samples=1, truncate_bounds=(-0.5, 0.5), temperature=0.5):
    """
    Generate autoregressive chain with specified stability method.

    Args:
        model: VAE model
        context: Initial context (C, 5, 5) log-returns
        starting_iv: Starting IV surface (5, 5)
        horizon: Number of steps to chain
        method: "mean_only", "truncated", "temperature", or "original"
        n_samples: Number of samples (for stochastic methods)
        truncate_bounds: Bounds for truncated sampling
        temperature: Scale factor for temperature sampling

    Returns:
        iv_trajectories: (n_samples, horizon, 5, 5) IV surfaces
    """
    device = next(model.parameters()).device

    all_trajectories = np.zeros((n_samples, horizon, 5, 5))

    for s in range(n_samples):
        current_iv = starting_iv.copy()
        current_context = context.copy()

        for h in range(horizon):
            ctx_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).to(device)
            batch = {"surface": ctx_tensor}

            with torch.no_grad():
                if method == "mean_only":
                    mean = get_mean_only(model, batch)
                    log_ret = mean[0, -1].cpu().numpy()

                elif method == "truncated":
                    samples = sample_truncated_student_t(model, batch, n_samples=1,
                                                         truncate_bounds=truncate_bounds)
                    log_ret = samples[0, 0, -1].cpu().numpy()

                elif method == "temperature":
                    samples = sample_temperature_scaled(model, batch, n_samples=1,
                                                        temperature=temperature)
                    log_ret = samples[0, 0, -1].cpu().numpy()

                elif method == "original":
                    samples = model.sample(batch, n_samples=1)
                    log_ret = samples[0, 0, -1].cpu().numpy()

                else:
                    raise ValueError(f"Unknown method: {method}")

            # Convert to IV
            new_iv = current_iv * np.exp(log_ret)
            all_trajectories[s, h] = new_iv

            # Update for next step
            current_iv = new_iv
            current_context = np.concatenate([current_context[1:], log_ret[np.newaxis]], axis=0)

    return all_trajectories


def plot_3d_iv_surface(ax, surface, title, moneyness, maturities, zlim=None):
    """Plot a single 3D IV surface."""
    X, Y = np.meshgrid(moneyness, maturities)

    surf = ax.plot_surface(X, Y, surface, cmap='viridis', alpha=0.9,
                           linewidth=0.3, antialiased=True, edgecolor='darkgray')

    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Maturity (mo)')
    ax.set_zlabel('IV')
    ax.set_title(title)

    if zlim:
        ax.set_zlim(zlim)

    return surf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="mean_only",
                        choices=["mean_only", "truncated", "temperature", "original", "all"],
                        help="Chaining method to use")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples for stochastic methods")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for temperature method")
    parser.add_argument("--truncate_lower", type=float, default=-0.3, help="Lower truncation bound")
    parser.add_argument("--truncate_upper", type=float, default=0.3, help="Upper truncation bound")
    args = parser.parse_args()

    print("=" * 70)
    print("3D IV Surface Visualization with Stable Autoregressive Chaining")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, config = load_vae(device)

    context_len = 30
    horizon = 30

    # Pick a point in validation set
    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100

    # Get context and ground truth
    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]
    gt_surfaces = surfaces[start_idx + context_len:start_idx + context_len + horizon]

    print(f"\nStarting IV (ATM): {starting_iv[2,2]:.4f}")
    print(f"GT IV range: {gt_surfaces.min():.4f} - {gt_surfaces.max():.4f}")

    # Grid coordinates
    moneyness = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    maturities = np.array([1, 3, 6, 12, 24])

    # Create output directory
    save_dir = Path("results/two_stage_vae/3d_iv_stable_chaining")
    save_dir.mkdir(parents=True, exist_ok=True)

    truncate_bounds = (args.truncate_lower, args.truncate_upper)

    methods_to_run = ["mean_only", "truncated", "temperature", "original"] if args.method == "all" else [args.method]

    results = {}

    for method in methods_to_run:
        print(f"\n{'='*70}")
        print(f"Method: {method}")
        print(f"{'='*70}")

        # Generate trajectories
        if method == "mean_only":
            n_samples = 1  # Deterministic
        else:
            n_samples = args.n_samples

        try:
            trajectories = autoregressive_chain(
                vae, context, starting_iv, horizon=horizon,
                method=method, n_samples=n_samples,
                truncate_bounds=truncate_bounds,
                temperature=args.temperature
            )

            results[method] = {
                "trajectories": trajectories,
                "mean": trajectories.mean(axis=0),
                "max_iv": trajectories.max(),
                "min_iv": trajectories.min(),
                "exploded": trajectories.max() > 5.0  # IV > 500% is explosion
            }

            print(f"  Generated IV range: {trajectories.min():.4f} - {trajectories.max():.4f}")
            print(f"  Exploded: {results[method]['exploded']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[method] = {"error": str(e), "exploded": True}

    # Create comparison plots
    zlim = (0, max(gt_surfaces.max(), max(r.get("mean", gt_surfaces).max()
                                          for r in results.values() if "mean" in r)) * 1.1)
    zlim = (0, min(zlim[1], 2.0))  # Cap at 200% IV for visualization

    # Plot comparison for Day 29
    fig = plt.figure(figsize=(20, 5))

    # GT
    ax1 = fig.add_subplot(141, projection='3d')
    plot_3d_iv_surface(ax1, gt_surfaces[-1], 'Ground Truth (Day 29)', moneyness, maturities, zlim)
    ax1.view_init(elev=25, azim=45)

    # Compare methods
    plot_idx = 2
    for method in methods_to_run[:3]:  # Max 3 methods
        if method in results and "mean" in results[method]:
            ax = fig.add_subplot(1, 4, plot_idx, projection='3d')
            title = f'{method.replace("_", " ").title()} (Day 29)'
            if results[method]["exploded"]:
                title += " [CLIPPED]"
            mean_surface = np.clip(results[method]["mean"][-1], 0, zlim[1])
            plot_3d_iv_surface(ax, mean_surface, title, moneyness, maturities, zlim)
            ax.view_init(elev=25, azim=45)
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_dir / 'method_comparison_day29.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: method_comparison_day29.png")
    plt.close()

    # Evolution plot for best stable method
    best_method = "mean_only"
    for m in ["mean_only", "truncated", "temperature"]:
        if m in results and not results[m].get("exploded", True):
            best_method = m
            break

    if best_method in results and "mean" in results[best_method]:
        gen_mean = results[best_method]["mean"]

        fig = plt.figure(figsize=(20, 8))
        days = [0, 7, 14, 21, 29]

        for idx, day in enumerate(days):
            # GT row
            ax = fig.add_subplot(2, 5, idx + 1, projection='3d')
            plot_3d_iv_surface(ax, gt_surfaces[day], f'GT Day {day}', moneyness, maturities, zlim)
            ax.view_init(elev=20, azim=45)

            # Gen row
            ax = fig.add_subplot(2, 5, idx + 6, projection='3d')
            plot_3d_iv_surface(ax, gen_mean[day], f'{best_method} Day {day}', moneyness, maturities, zlim)
            ax.view_init(elev=20, azim=45)

        plt.suptitle(f'IV Surface Evolution: Ground Truth (top) vs {best_method.replace("_", " ").title()} (bottom)',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'3d_iv_evolution_{best_method}.png', dpi=150, bbox_inches='tight')
        print(f"Saved: 3d_iv_evolution_{best_method}.png")
        plt.close()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Stable Chaining Methods Comparison")
    print("=" * 70)
    print(f"\n{'Method':<15} | {'Max IV':>10} | {'Min IV':>10} | {'Exploded':>10}")
    print("-" * 55)
    print(f"{'Ground Truth':<15} | {gt_surfaces.max():>10.4f} | {gt_surfaces.min():>10.4f} | {'No':>10}")
    for method, r in results.items():
        if "max_iv" in r:
            print(f"{method:<15} | {r['max_iv']:>10.4f} | {r['min_iv']:>10.4f} | {'YES' if r['exploded'] else 'No':>10}")
        else:
            print(f"{method:<15} | {'ERROR':>10} | {'ERROR':>10} | {'YES':>10}")

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
