"""
Autoregressive Chaining Fan Chart

Generate 50-sample fan chart showing chained trajectories vs GT.
4 hops × 30 days with 15-day overlap = 75 days total.

Usage:
    python experiments/backfill/two_stage_vae/plot_chaining_fan_chart.py
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
    """Load the Student-t VAE."""
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


def generate_chained_fan_chart(
    vae, log_returns, surfaces, start_idx,
    context_len=30, horizon_per_hop=30, overlap=15, n_hops=4, n_samples=50, device="cuda"
):
    """
    Generate chained trajectories and return IV fan chart data.

    Returns:
        gt_iv: (total_days,) ground truth ATM IV
        sample_iv: (n_samples, total_days) sampled ATM IV trajectories
        initial_iv: Starting IV level
    """
    # Total unique days generated
    total_days = n_hops * (horizon_per_hop - overlap) + overlap
    print(f"Generating {total_days} days via {n_hops} hops")

    # Get initial context and starting IV
    initial_context = log_returns[start_idx:start_idx + context_len]
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]

    # GT trajectory
    gt_iv = surfaces[start_idx + context_len:start_idx + context_len + total_days, 2, 2]

    # Storage for sample trajectories
    sample_iv = np.zeros((n_samples, total_days))

    # Current context for each sample (will diverge as we chain)
    current_contexts = np.tile(initial_context, (n_samples, 1, 1, 1))  # (n_samples, ctx_len, 5, 5)

    # Track cumulative IV for each sample
    current_iv = np.full(n_samples, initial_iv)

    # Day index in output
    day_idx = 0

    with torch.no_grad():
        for hop in range(n_hops):
            print(f"  Hop {hop + 1}/{n_hops}...")

            # Days to generate this hop (accounting for overlap)
            if hop == 0:
                days_this_hop = horizon_per_hop
            else:
                days_this_hop = horizon_per_hop - overlap

            for h in range(days_this_hop):
                # For each sample, generate one step
                for s in range(n_samples):
                    # Build sequence for this sample
                    ctx = current_contexts[s]  # (ctx_len, 5, 5)
                    ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)
                    batch = {"surface": ctx_tensor}

                    # Sample one prediction
                    samples = vae.sample(batch, n_samples=1)
                    log_return = samples[0, 0, -1, 2, 2].cpu().numpy()

                    # Update IV
                    current_iv[s] = current_iv[s] * np.exp(log_return)
                    sample_iv[s, day_idx] = current_iv[s]

                    # Update context: shift and append
                    new_surface = samples[0, 0, -1].cpu().numpy()  # (5, 5)
                    current_contexts[s] = np.concatenate([
                        current_contexts[s, 1:],  # Drop first day
                        new_surface[np.newaxis]   # Add new prediction
                    ], axis=0)

                day_idx += 1

                if day_idx % 10 == 0:
                    print(f"    Day {day_idx}/{total_days}")

    return gt_iv, sample_iv, initial_iv


def plot_fan_chart(gt_iv, sample_iv, initial_iv, title="Autoregressive Chaining Fan Chart", save_path=None):
    """Plot fan chart with percentile bands."""
    fig, ax = plt.subplots(figsize=(14, 7))

    total_days = len(gt_iv)
    days = np.arange(total_days)

    # Compute percentiles
    p5 = np.percentile(sample_iv, 5, axis=0)
    p25 = np.percentile(sample_iv, 25, axis=0)
    p50 = np.percentile(sample_iv, 50, axis=0)
    p75 = np.percentile(sample_iv, 75, axis=0)
    p95 = np.percentile(sample_iv, 95, axis=0)

    # Plot percentile bands
    ax.fill_between(days, p5, p95, alpha=0.2, color='blue', label='5-95% CI')
    ax.fill_between(days, p25, p75, alpha=0.3, color='blue', label='25-75% CI')

    # Plot median
    ax.plot(days, p50, 'b-', linewidth=2, label='Median')

    # Plot GT
    ax.plot(days, gt_iv, 'r-', linewidth=2, label='Ground Truth')

    # Plot a few sample trajectories
    for i in range(min(10, sample_iv.shape[0])):
        ax.plot(days, sample_iv[i], 'b-', alpha=0.1, linewidth=0.5)

    # Mark hop boundaries
    hop_boundaries = [30, 45, 60]  # Days where new hops start (after overlap)
    for hb in hop_boundaries:
        if hb < total_days:
            ax.axvline(hb, color='gray', linestyle='--', alpha=0.5)

    ax.axhline(initial_iv, color='green', linestyle=':', alpha=0.5, label=f'Initial IV ({initial_iv:.3f})')

    ax.set_xlabel('Days from Start', fontsize=12)
    ax.set_ylabel('ATM Implied Volatility', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add text annotations for hops
    ax.text(15, ax.get_ylim()[1] * 0.95, 'Hop 1', ha='center', fontsize=10, color='gray')
    if total_days > 30:
        ax.text(37, ax.get_ylim()[1] * 0.95, 'Hop 2', ha='center', fontsize=10, color='gray')
    if total_days > 45:
        ax.text(52, ax.get_ylim()[1] * 0.95, 'Hop 3', ha='center', fontsize=10, color='gray')
    if total_days > 60:
        ax.text(67, ax.get_ylim()[1] * 0.95, 'Hop 4', ha='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

    return fig


def main():
    """Generate and plot chained fan chart."""
    print("=" * 70)
    print("Autoregressive Chaining Fan Chart")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"Data: {len(surfaces)} days")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    vae, config = load_vae(device)

    # Configuration
    context_len = 30
    horizon_per_hop = 30
    overlap = 15
    n_hops = 4
    n_samples = 50

    total_days = n_hops * (horizon_per_hop - overlap) + overlap
    print(f"\nConfiguration:")
    print(f"  Context: {context_len} days")
    print(f"  Horizon per hop: {horizon_per_hop} days")
    print(f"  Overlap: {overlap} days")
    print(f"  Hops: {n_hops}")
    print(f"  Total days: {total_days}")
    print(f"  Samples: {n_samples}")

    # Pick starting point - try crisis period (Sep 2008)
    # Day ~2100 is around Sep 2008 crisis
    start_idx = 2100  # Crisis period

    print(f"\nStarting from day {start_idx} (Crisis period - Sep 2008)")

    # Generate
    gt_iv, sample_iv, initial_iv = generate_chained_fan_chart(
        vae, log_returns, surfaces, start_idx,
        context_len=context_len,
        horizon_per_hop=horizon_per_hop,
        overlap=overlap,
        n_hops=n_hops,
        n_samples=n_samples,
        device=device
    )

    print(f"\nGenerated {sample_iv.shape[1]} days with {sample_iv.shape[0]} samples")

    # Compute CI violations
    p5 = np.percentile(sample_iv, 5, axis=0)
    p95 = np.percentile(sample_iv, 95, axis=0)
    violations = np.mean((gt_iv < p5) | (gt_iv > p95)) * 100
    print(f"90% CI violations: {violations:.1f}%")

    # Plot
    save_dir = Path("results/two_stage_vae/autoregressive_chaining")
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_fan_chart(
        gt_iv, sample_iv, initial_iv,
        title=f"Autoregressive Chaining: {n_hops} Hops × {horizon_per_hop} Days (50 samples)",
        save_path=save_dir / "chaining_fan_chart.png"
    )

    # Save data
    np.savez(
        save_dir / "fan_chart_data.npz",
        gt_iv=gt_iv,
        sample_iv=sample_iv,
        initial_iv=initial_iv,
        config={
            "context_len": context_len,
            "horizon_per_hop": horizon_per_hop,
            "overlap": overlap,
            "n_hops": n_hops,
            "n_samples": n_samples,
            "start_idx": start_idx,
        }
    )

    return gt_iv, sample_iv


if __name__ == "__main__":
    main()
