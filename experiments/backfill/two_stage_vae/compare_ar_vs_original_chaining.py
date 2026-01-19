"""
Compare AR(1) vs Original Chaining

Side-by-side comparison of:
1. Original chaining (StudentTMLPDecoder, no AR, no prev_x)
2. AR(1) chaining (StudentTDualPathARDecoder with prev_x)

Shows the improvement in ACF preservation and fan chart realism.

Usage:
    python experiments/backfill/two_stage_vae/compare_ar_vs_original_chaining.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP, CVAETwoStageDualPathAR


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def compute_acf(series, lag=1):
    """Compute autocorrelation at given lag."""
    n = len(series)
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    if var < 1e-10:
        return 0.0
    cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
    return cov / var


def load_original_model(device: str = "cuda"):
    """Load the original StudentTMLP model (no AR)."""
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageStudentTMLP(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def load_ar_model(device: str = "cuda"):
    """Load the AR(1) model."""
    model_path = "models/backfill/two_stage/dual_path_ar/dual_path_ar_best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = CVAETwoStageDualPathAR(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def chain_original(model, context, starting_iv, horizon=30, n_samples=50,
                   device="cuda", clip_log_ret=0.2):
    """Original chaining (no AR, no prev_x)."""
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()

            for h in range(horizon):
                # Sample z
                eps = torch.randn_like(z_std) * 0.5  # Temperature = 0.5
                z = z_mean + z_std * eps

                # Original: use forward output[-1] (reconstruction)
                ctx_tensor_h = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)

                # Get ctx_emb (even though MLP decoder ignores it)
                if hasattr(model, 'ctx_encoder') and model.ctx_encoder is not None:
                    ctx_emb = model.ctx_encoder({"surface": ctx_tensor_h})
                else:
                    ctx_emb = None

                z_mean_h, z_logvar_h, _ = model.main_encoder({"surface": ctx_tensor_h})
                z_std_h = torch.exp(0.5 * z_logvar_h)
                eps = torch.randn_like(z_std_h) * 0.5
                z_h = z_mean_h + z_std_h * eps

                # Decode (no prev_x)
                mean, sample, _, _ = model.decoder(ctx_emb, z_h, sample=True)
                log_ret = sample[0, -1].cpu().numpy()  # Last position (reconstruction)

                # Clip
                if clip_log_ret is not None:
                    log_ret = np.clip(log_ret, -clip_log_ret, clip_log_ret)

                all_log_returns[s, h] = log_ret
                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                current_iv = new_iv
                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

    return all_trajectories, all_log_returns


def chain_ar(model, context, starting_iv, horizon=30, n_samples=50,
             device="cuda", clip_log_ret=0.2):
    """AR(1) chaining with prev_x."""
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()

            for h in range(horizon):
                # Sample z
                eps = torch.randn_like(z_std) * 0.5  # Temperature = 0.5
                z = z_mean + z_std * eps

                # AR(1): pass prev_x
                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, prev_x=prev_x_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()

                # Clip
                if clip_log_ret is not None:
                    log_ret = np.clip(log_ret, -clip_log_ret, clip_log_ret)

                all_log_returns[s, h] = log_ret
                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                current_iv = new_iv
                prev_log_return = log_ret
                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                # Update context encoder
                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def main():
    print("=" * 70)
    print("Comparison: Original vs AR(1) Chaining")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load both models
    print("\nLoading models...")
    original_model, _ = load_original_model(device)
    ar_model, _ = load_ar_model(device)

    phi = ar_model.decoder.get_phi().item()
    print(f"AR(1) model phi = {phi:.4f}")

    # Setup
    context_len = 20
    horizon = 30
    n_samples = 50
    clip_log_ret = 0.2

    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100

    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]
    gt_surfaces = surfaces[start_idx + context_len:start_idx + context_len + horizon]
    gt_log_returns = log_returns[start_idx + context_len:start_idx + context_len + horizon]

    # Generate with both methods
    print("\nGenerating with Original method (no AR)...")
    orig_traj, orig_log_rets = chain_original(
        original_model, context, starting_iv,
        horizon=horizon, n_samples=n_samples, device=device,
        clip_log_ret=clip_log_ret
    )

    print("Generating with AR(1) method...")
    ar_traj, ar_log_rets = chain_ar(
        ar_model, context, starting_iv,
        horizon=horizon, n_samples=n_samples, device=device,
        clip_log_ret=clip_log_ret
    )

    # Compute ACFs
    gt_acf = compute_acf(gt_log_returns[:, 2, 2])

    orig_acfs = [compute_acf(orig_log_rets[s, :, 2, 2]) for s in range(n_samples)]
    orig_acf_mean = np.mean(orig_acfs)

    ar_acfs = [compute_acf(ar_log_rets[s, :, 2, 2]) for s in range(n_samples)]
    ar_acf_mean = np.mean(ar_acfs)

    print("\n" + "=" * 70)
    print("ACF COMPARISON (ATM Point)")
    print("=" * 70)
    print(f"Ground Truth ACF(1): {gt_acf:.4f}")
    print(f"Original ACF(1):     {orig_acf_mean:.4f} ({abs(orig_acf_mean)/abs(gt_acf)*100:.1f}% preservation)")
    print(f"AR(1) ACF(1):        {ar_acf_mean:.4f} ({abs(ar_acf_mean)/abs(gt_acf)*100:.1f}% preservation)")
    print(f"\nImprovement: {abs(ar_acf_mean)/abs(orig_acf_mean)*100:.1f}x better ACF preservation")

    # Create comparison plot
    save_dir = Path("results/two_stage_vae/ar_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    days = np.arange(horizon)
    grid_point = (2, 2)  # ATM
    i, j = grid_point

    # Top left: Original fan chart
    ax = axes[0, 0]
    gt_vals = gt_surfaces[:, i, j]
    traj_vals = orig_traj[:, :, i, j]
    median = np.median(traj_vals, axis=0)
    p10 = np.percentile(traj_vals, 10, axis=0)
    p90 = np.percentile(traj_vals, 90, axis=0)

    ax.plot(days, gt_vals, 'k-', linewidth=2, label='GT')
    ax.plot(days, median, 'r-', linewidth=1.5, label='Median')
    ax.fill_between(days, p10, p90, alpha=0.3, color='red', label='10-90% CI')
    ax.set_title(f'Original (No AR)\nACF={orig_acf_mean:.3f}')
    ax.set_xlabel('Days')
    ax.set_ylabel('IV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: AR(1) fan chart
    ax = axes[0, 1]
    traj_vals = ar_traj[:, :, i, j]
    median = np.median(traj_vals, axis=0)
    p10 = np.percentile(traj_vals, 10, axis=0)
    p90 = np.percentile(traj_vals, 90, axis=0)

    ax.plot(days, gt_vals, 'k-', linewidth=2, label='GT')
    ax.plot(days, median, 'b-', linewidth=1.5, label='Median')
    ax.fill_between(days, p10, p90, alpha=0.3, color='blue', label='10-90% CI')
    ax.set_title(f'AR(1) Chaining\nACF={ar_acf_mean:.3f}')
    ax.set_xlabel('Days')
    ax.set_ylabel('IV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: ACF distribution comparison
    ax = axes[1, 0]
    ax.hist(orig_acfs, bins=20, alpha=0.5, color='red', label='Original')
    ax.hist(ar_acfs, bins=20, alpha=0.5, color='blue', label='AR(1)')
    ax.axvline(gt_acf, color='black', linestyle='--', linewidth=2, label=f'GT={gt_acf:.3f}')
    ax.set_xlabel('ACF(1)')
    ax.set_ylabel('Count')
    ax.set_title('ACF Distribution Across Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    COMPARISON SUMMARY (ATM Point)
    ==============================

    Ground Truth ACF(1): {gt_acf:.4f}

    Original Method (No AR):
    - ACF(1) mean: {orig_acf_mean:.4f}
    - Preservation: {abs(orig_acf_mean)/abs(gt_acf)*100:.1f}%
    - IV range: {orig_traj.min():.4f} - {orig_traj.max():.4f}

    AR(1) Method:
    - ACF(1) mean: {ar_acf_mean:.4f}
    - Preservation: {abs(ar_acf_mean)/abs(gt_acf)*100:.1f}%
    - IV range: {ar_traj.min():.4f} - {ar_traj.max():.4f}
    - phi = {phi:.4f}

    IMPROVEMENT:
    - ACF preservation: {abs(ar_acf_mean)/abs(orig_acf_mean):.1f}x better
    - Mean reversion: YES (phi={phi:.3f})
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Original vs AR(1) Chaining Comparison', fontsize=14)
    plt.tight_layout()

    output_path = save_dir / 'original_vs_ar_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"AR(1) decoder achieves {abs(ar_acf_mean)/abs(gt_acf)*100:.1f}% ACF preservation")
    print(f"vs Original's {abs(orig_acf_mean)/abs(gt_acf)*100:.1f}% - a {abs(ar_acf_mean)/abs(orig_acf_mean):.1f}x improvement")


if __name__ == "__main__":
    main()
