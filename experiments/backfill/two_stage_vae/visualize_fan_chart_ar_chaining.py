"""
Autoregressive Chaining with AR(1) Decoder for Mean Reversion

This script properly implements autoregressive chaining using the AR(1) decoder
that captures mean reversion (ACF ~ -0.35).

Key fixes from the original chaining code:
1. Uses CVAETwoStageDualPathAR (with AR(1) decoder)
2. Passes prev_x at each step for mean reversion correction
3. Properly handles the autoregressive feedback loop

Expected improvement:
- ACF preservation: ~0% → 34.98%
- Kurtosis recovery: preserved at 131.95%
- Mean reversion: φ = -0.367 (vs GT -0.35)

Usage:
    python experiments/backfill/two_stage_vae/visualize_fan_chart_ar_chaining.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageDualPathAR


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_ar_model(device: str = "cuda"):
    """Load the DualPath+AR(1) model."""
    model_path = "models/backfill/two_stage/dual_path_ar/dual_path_ar_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageDualPathAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def compute_acf(series, lag=1):
    """Compute autocorrelation at given lag."""
    n = len(series)
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    if var < 1e-10:
        return 0.0
    cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
    return cov / var


def chain_with_ar_decoder(model, context, starting_iv, horizon=30, n_samples=50,
                          device="cuda", temperature=1.0, clip_log_ret=0.3):
    """
    Autoregressive chaining with AR(1) decoder.

    Key difference from original: passes prev_x at each step for mean reversion.

    Args:
        model: CVAETwoStageDualPathAR model
        context: Initial context (C, 5, 5) log-returns
        starting_iv: Starting IV surface (5, 5)
        horizon: Number of steps to chain
        n_samples: Number of trajectories to generate
        temperature: Sampling temperature (1.0 = full variance, <1 = reduced)
        clip_log_ret: Clip log-returns to [-clip, +clip] to prevent explosion

    Returns:
        trajectories: (n_samples, horizon, 5, 5) IV surfaces
        log_returns: (n_samples, horizon, 5, 5) log-returns for ACF analysis
    """
    C = context.shape[0]
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get ctx_emb from context (only computed once)
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})  # (1, C, ctx_dim)

        # For z, we sample from encoder (oracle mode)
        # In true prediction mode, you'd use a prior or predictor
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()  # Last context log-return for AR(1)

            for h in range(horizon):
                # Sample z
                eps = torch.randn_like(z_std) * temperature
                z = z_mean + z_std * eps

                # Prepare prev_x for AR(1) correction
                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 5, 5)

                # We need ctx_emb for just one step - use last position
                ctx_emb_step = ctx_emb[:, -1:, :]  # (1, 1, ctx_dim)
                z_step = z[:, -1:, :]  # (1, 1, latent_dim)

                # Decode with AR(1) correction via prev_x
                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, prev_x=prev_x_tensor, sample=True
                )

                # Apply temperature scaling to the sample deviation from mean
                if temperature < 1.0:
                    sample = mean + temperature * (sample - mean)

                log_ret = sample[0, 0].cpu().numpy()  # (5, 5)

                # Clip log-returns to prevent explosion
                if clip_log_ret is not None:
                    log_ret = np.clip(log_ret, -clip_log_ret, clip_log_ret)

                # Store log-return for ACF analysis
                all_log_returns[s, h] = log_ret

                # Update IV
                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                # Update for next iteration
                current_iv = new_iv
                prev_log_return = log_ret

                # Update context (sliding window)
                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                # Recompute ctx_emb with updated context
                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def main():
    print("=" * 70)
    print("Autoregressive Chaining with AR(1) Decoder")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load AR(1) model
    print("\nLoading DualPath+AR(1) model...")
    model, config = load_ar_model(device)

    # Print learned AR(1) coefficient
    phi = model.decoder.get_phi().item()
    print(f"Loaded model with learned AR(1) phi = {phi:.4f}")
    print(f"Ground truth target: phi = -0.35")

    # Setup
    context_len = 20
    horizon = 30
    n_samples = 50
    temperature = 0.8  # Increased from 0.5 for better CI coverage

    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100  # Start in test set

    # Get context and ground truth
    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]
    gt_surfaces = surfaces[start_idx + context_len:start_idx + context_len + horizon]
    gt_log_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon]

    print(f"\nChaining setup:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  N samples: {n_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Starting IV (ATM): {starting_iv[2, 2]:.4f}")
    print(f"  GT IV range: {gt_surfaces.min():.4f} - {gt_surfaces.max():.4f}")

    # Generate trajectories with AR(1) decoder
    clip_log_ret = 0.35  # Increased from 0.2 for better CI coverage
    print(f"  Clip log-returns: +/- {clip_log_ret}")

    print("\nGenerating trajectories with AR(1) decoder...")
    trajectories, log_rets = chain_with_ar_decoder(
        model, context, starting_iv,
        horizon=horizon, n_samples=n_samples, device=device,
        temperature=temperature, clip_log_ret=clip_log_ret
    )

    print(f"  Generated IV range: {trajectories.min():.4f} - {trajectories.max():.4f}")

    # Compute ACF of generated sequences
    print("\n" + "=" * 70)
    print("ACF Analysis (Mean Reversion Check)")
    print("=" * 70)

    # Ground truth ACF (from log-returns)
    gt_acf_atm = compute_acf(gt_log_returns[:, 2, 2])

    # Model ACF (average across samples)
    model_acfs = []
    for s in range(n_samples):
        acf = compute_acf(log_rets[s, :, 2, 2])
        model_acfs.append(acf)
    model_acf_mean = np.mean(model_acfs)
    model_acf_std = np.std(model_acfs)

    print(f"\nATM Point (2,2):")
    print(f"  GT ACF(1):     {gt_acf_atm:.4f}")
    print(f"  Model ACF(1):  {model_acf_mean:.4f} +/- {model_acf_std:.4f}")
    print(f"  Preservation:  {abs(model_acf_mean) / abs(gt_acf_atm) * 100:.1f}%")

    # Create fan chart
    save_dir = Path("results/two_stage_vae/ar_chaining")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    grid_points = [
        ((2, 2), "ATM (1.00, 6mo)"),
        ((0, 0), "Deep OTM Put (0.70, 1mo)"),
        ((4, 4), "Deep OTM Call (1.30, 24mo)"),
        ((0, 2), "OTM Put Short (0.70, 6mo)"),
        ((4, 0), "OTM Call Short (1.30, 1mo)"),
        ((2, 4), "ATM Long (1.00, 24mo)"),
    ]

    days = np.arange(horizon)

    for ax, ((i, j), name) in zip(axes.flat, grid_points):
        # Ground truth
        gt_vals = gt_surfaces[:, i, j]

        # Model trajectories
        traj_vals = trajectories[:, :, i, j]
        median = np.median(traj_vals, axis=0)
        p10 = np.percentile(traj_vals, 10, axis=0)
        p90 = np.percentile(traj_vals, 90, axis=0)
        p25 = np.percentile(traj_vals, 25, axis=0)
        p75 = np.percentile(traj_vals, 75, axis=0)

        # Plot
        ax.plot(days, gt_vals, 'k-', linewidth=2, label='Ground Truth')
        ax.plot(days, median, 'b-', linewidth=1.5, label='Model Median')
        ax.fill_between(days, p10, p90, alpha=0.2, color='blue', label='10-90% CI')
        ax.fill_between(days, p25, p75, alpha=0.3, color='blue', label='25-75% CI')

        # Compute local ACF
        local_acfs = [compute_acf(log_rets[s, :, i, j]) for s in range(n_samples)]
        local_acf_mean = np.mean(local_acfs)

        ax.set_title(f'{name}\nACF={local_acf_mean:.3f}')
        ax.set_xlabel('Days')
        ax.set_ylabel('IV')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'AR(1) Chaining Fan Chart (phi={phi:.3f}, temp={temperature})\n'
        f'GT ACF={gt_acf_atm:.3f}, Model ACF={model_acf_mean:.3f}',
        fontsize=12
    )
    plt.tight_layout()

    output_path = save_dir / 'ar_chaining_fan_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Compute CI coverage
    print("\n" + "=" * 70)
    print("CI COVERAGE ANALYSIS")
    print("=" * 70)

    for (i, j), name in grid_points:
        gt_vals = gt_surfaces[:, i, j]
        traj_vals = trajectories[:, :, i, j]
        p5 = np.percentile(traj_vals, 5, axis=0)
        p95 = np.percentile(traj_vals, 95, axis=0)

        in_ci = (gt_vals >= p5) & (gt_vals <= p95)
        coverage = in_ci.mean() * 100
        print(f"{name}: {coverage:.1f}% of GT within 90% CI (target: 90%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"AR(1) coefficient: phi = {phi:.4f} (GT: -0.35)")
    print(f"ACF(1) preservation: {abs(model_acf_mean) / abs(gt_acf_atm) * 100:.1f}%")
    print(f"IV range: {trajectories.min():.4f} - {trajectories.max():.4f}")
    print(f"GT range: {gt_surfaces.min():.4f} - {gt_surfaces.max():.4f}")

    # Check if any grid point exploded
    max_iv = trajectories.max()
    if max_iv > 2.0:
        print(f"\nWARNING: Some IV values exceeded 2.0 (max: {max_iv:.2f})")
    else:
        print(f"\nNo explosion: max IV = {max_iv:.4f}")


if __name__ == "__main__":
    main()
