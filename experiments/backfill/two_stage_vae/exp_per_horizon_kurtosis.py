"""
Per-Horizon Kurtosis Analysis

Goal: Determine if kurtosis degradation at longer horizons is due to:
1. CLT effect from aggregation (flattening all timesteps)
2. Actual per-step degradation with horizon distance

Current evaluation flattens ALL timesteps:
    samples_atm = samples[:, :, :, 2, 2].flatten()  # All H steps together
    kurtosis(samples_atm)  # Single number

This script measures kurtosis at EACH horizon step separately:
    for h in range(horizon):
        kurt_h = kurtosis(samples[:, :, h, 2, 2].flatten())

Usage:
    python experiments/backfill/two_stage_vae/exp_per_horizon_kurtosis.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from config.two_stage_config import TWO_STAGE_CONFIG


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, horizon, batch_size, shuffle=True):
    """Create dataloader with specified horizon."""
    seq_len = context_len + horizon
    sequences = []

    for i in range(len(log_returns) - seq_len):
        seq = log_returns[i:i + seq_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_per_horizon_kurtosis(model, val_loader, config, n_samples=50):
    """
    Compute kurtosis at each horizon position separately.

    Returns dict with:
        - per_horizon_kurt: kurtosis at each h=0,1,...,H-1
        - per_horizon_gt_kurt: GT kurtosis at each h
        - aggregated_kurt: kurtosis of flattened samples (old method)
    """
    device = config["device"]
    horizon = config["horizon"]
    context_len = config["context_len"]
    model.eval()

    # Collect all samples: list of (n_samples, B, T, 5, 5)
    all_samples = []
    all_gt = []

    print(f"Collecting samples (n_samples={n_samples})...")
    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5) where T = context_len + horizon + 1

            all_samples.append(samples.cpu().numpy())
            all_gt.append(batch_data.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(val_loader)}")

    # Concatenate across batches
    # all_samples: (n_samples, total_B, T, 5, 5)
    all_samples = np.concatenate(all_samples, axis=1)
    all_gt = np.concatenate(all_gt, axis=0)

    print(f"Samples shape: {all_samples.shape}")
    print(f"GT shape: {all_gt.shape}")

    # Extract horizon positions (after context)
    # Model output T = context_len + horizon + 1 (because of return_full_sequence)
    # Horizon positions are [context_len:context_len+horizon]
    T = all_samples.shape[2]
    print(f"Total sequence length T={T}, context_len={context_len}, horizon={horizon}")

    # Compute per-horizon kurtosis
    per_horizon_kurt = []
    per_horizon_gt_kurt = []

    print("\nPer-Horizon Kurtosis Analysis (ATM grid point [2,2]):")
    print("-" * 60)
    print(f"{'Horizon':>8} | {'Model Kurt':>12} | {'GT Kurt':>12} | {'Recovery':>10}")
    print("-" * 60)

    for h in range(horizon):
        # Position in sequence: context_len + h
        pos = context_len + h

        if pos >= T:
            print(f"Warning: position {pos} >= T={T}, skipping")
            break

        # Samples at this horizon position
        samples_at_h = all_samples[:, :, pos, 2, 2].flatten()  # (n_samples * total_B,)
        gt_at_h = all_gt[:, pos, 2, 2].flatten()  # (total_B,)

        # Compute kurtosis
        kurt_h = kurtosis(samples_at_h, fisher=True)
        gt_kurt_h = kurtosis(gt_at_h, fisher=True)

        recovery = (kurt_h / gt_kurt_h * 100) if abs(gt_kurt_h) > 0.1 else 0

        per_horizon_kurt.append(kurt_h)
        per_horizon_gt_kurt.append(gt_kurt_h)

        print(f"{h:>8} | {kurt_h:>12.2f} | {gt_kurt_h:>12.2f} | {recovery:>9.1f}%")

    print("-" * 60)

    # Aggregated kurtosis (old method)
    all_samples_horizon = all_samples[:, :, context_len:context_len+horizon, 2, 2].flatten()
    all_gt_horizon = all_gt[:, context_len:context_len+horizon, 2, 2].flatten()

    aggregated_kurt = kurtosis(all_samples_horizon, fisher=True)
    aggregated_gt_kurt = kurtosis(all_gt_horizon, fisher=True)

    print(f"\nAggregated (all {horizon} steps flattened):")
    print(f"  Model: {aggregated_kurt:.2f}")
    print(f"  GT:    {aggregated_gt_kurt:.2f}")
    print(f"  Recovery: {aggregated_kurt/aggregated_gt_kurt*100:.1f}%")

    return {
        "per_horizon_kurt": np.array(per_horizon_kurt),
        "per_horizon_gt_kurt": np.array(per_horizon_gt_kurt),
        "aggregated_kurt": aggregated_kurt,
        "aggregated_gt_kurt": aggregated_gt_kurt,
    }


def plot_per_horizon_kurtosis(results, horizon, save_path=None):
    """Plot kurtosis vs horizon position."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    h_range = np.arange(len(results["per_horizon_kurt"]))

    # Plot 1: Absolute kurtosis
    ax1 = axes[0]
    ax1.plot(h_range, results["per_horizon_kurt"], 'b-o', label='Model', markersize=4)
    ax1.plot(h_range, results["per_horizon_gt_kurt"], 'g-s', label='GT', markersize=4)
    ax1.axhline(results["aggregated_kurt"], color='b', linestyle='--', alpha=0.5,
                label=f'Model Aggregated ({results["aggregated_kurt"]:.1f})')
    ax1.axhline(results["aggregated_gt_kurt"], color='g', linestyle='--', alpha=0.5,
                label=f'GT Aggregated ({results["aggregated_gt_kurt"]:.1f})')
    ax1.set_xlabel('Horizon Position (h)')
    ax1.set_ylabel('Excess Kurtosis')
    ax1.set_title('Kurtosis vs Horizon Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Recovery %
    ax2 = axes[1]
    recovery = results["per_horizon_kurt"] / (results["per_horizon_gt_kurt"] + 1e-8) * 100
    ax2.plot(h_range, recovery, 'r-o', markersize=4)
    ax2.axhline(100, color='k', linestyle='--', alpha=0.5, label='Perfect Recovery')
    ax2.axhline(results["aggregated_kurt"]/results["aggregated_gt_kurt"]*100,
                color='b', linestyle='--', alpha=0.5,
                label=f'Aggregated Recovery ({results["aggregated_kurt"]/results["aggregated_gt_kurt"]*100:.1f}%)')
    ax2.set_xlabel('Horizon Position (h)')
    ax2.set_ylabel('Kurtosis Recovery (%)')
    ax2.set_title('Kurtosis Recovery vs Horizon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(200, recovery.max() * 1.1)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    """Run per-horizon kurtosis analysis."""
    print("=" * 70)
    print("Per-Horizon Kurtosis Analysis")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)
    val_data = log_returns[train_end:val_end]

    print(f"Val: {len(val_data)} days")

    # Configuration - test with H=30
    context_len = 30
    horizon = 30
    batch_size = 32

    print(f"\nConfiguration:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  Batch size: {batch_size}")

    # Create dataloader
    val_loader = create_dataloader(val_data, context_len, horizon, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Model config
    config = TWO_STAGE_CONFIG.copy()
    config["device"] = device
    config["horizon"] = horizon
    config["max_horizon"] = horizon
    config["context_len"] = context_len

    # Create and train a fresh model (or load existing)
    model_path = Path("models/backfill/two_stage/student_t_acf")

    # Check for existing model
    existing_models = list(model_path.glob("*.pt")) if model_path.exists() else []

    if existing_models:
        print(f"\nLoading existing model from {existing_models[0]}...")
        checkpoint = torch.load(existing_models[0], map_location=device)
        model = CVAETwoStageStudentTMLP(checkpoint.get("model_config", config))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("\nNo existing model found. Training new model...")
        # Train a quick model for testing
        from experiments.backfill.two_stage_vae.exp_student_t_acf import train_student_t_with_acf

        train_data = log_returns[:train_end]
        train_loader = create_dataloader(train_data, context_len, horizon, batch_size, shuffle=True)

        model = CVAETwoStageStudentTMLP(config)
        model = train_student_t_with_acf(
            model, train_loader, val_loader, config,
            epochs=50, lambda_acf=0.0, phase1_ratio=1.0
        )

    model = model.to(device)
    model.eval()

    # Run analysis
    print("\n" + "=" * 70)
    print("Running Per-Horizon Kurtosis Analysis")
    print("=" * 70)

    results = evaluate_per_horizon_kurtosis(model, val_loader, config, n_samples=30)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_per_h_kurt = np.mean(results["per_horizon_kurt"])
    mean_per_h_gt = np.mean(results["per_horizon_gt_kurt"])
    mean_recovery = mean_per_h_kurt / mean_per_h_gt * 100

    print(f"Mean per-horizon kurtosis: {mean_per_h_kurt:.2f} (model) vs {mean_per_h_gt:.2f} (GT)")
    print(f"Mean per-horizon recovery: {mean_recovery:.1f}%")
    print(f"Aggregated recovery: {results['aggregated_kurt']/results['aggregated_gt_kurt']*100:.1f}%")

    # Check if kurtosis degrades with h
    first_10_mean = np.mean(results["per_horizon_kurt"][:10])
    last_10_mean = np.mean(results["per_horizon_kurt"][-10:])

    print(f"\nTrend analysis:")
    print(f"  First 10 steps mean kurtosis: {first_10_mean:.2f}")
    print(f"  Last 10 steps mean kurtosis: {last_10_mean:.2f}")
    print(f"  Degradation: {(last_10_mean - first_10_mean) / first_10_mean * 100:.1f}%")

    # Plot
    save_dir = Path("results/two_stage_vae/per_horizon_kurtosis")
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_per_horizon_kurtosis(results, horizon, save_path=save_dir / "kurtosis_vs_horizon.png")

    # Save results
    np.savez(
        save_dir / "per_horizon_kurtosis_results.npz",
        **results,
        horizon=horizon,
        context_len=context_len,
    )
    print(f"\nResults saved to: {save_dir}")

    return results


if __name__ == "__main__":
    main()
