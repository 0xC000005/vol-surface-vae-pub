"""
Compare Oracle vs Prior Predictor: Conditional Fan Charts

Side-by-side comparison of ATM IV 30-day trajectories with 50 samples.
Shows 4 market periods, with Oracle mode (left) and Predictor mode (right).

Usage:
    python experiments/backfill/two_stage_vae/compare_oracle_vs_predictor_fan_chart.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def load_models(device: str = "cuda"):
    """Load the Student-t VAE and trained predictor."""
    # Load VAE
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    config = vae_ckpt["model_config"]
    config["device"] = device

    vae = CVAETwoStageStudentTMLP(config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load predictor
    pred_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    pred_ckpt = torch.load(pred_path, map_location=device, weights_only=False)

    predictor = LatentPredictor(config)
    predictor.load_state_dict(pred_ckpt["predictor_state_dict"])
    predictor = predictor.to(device)
    predictor.eval()

    return vae, predictor, config


def get_period_indices():
    """Get indices for 4 selective periods."""
    return {
        "Vol Spike (Sep 2008)": 2100,
        "Crisis Peak (Oct 2008)": 2150,
        "Recovery (Mar 2009)": 2280,
        "Debt Ceiling (Aug 2011)": 2900,
    }


def generate_trajectories_oracle(vae, log_returns, surfaces, start_idx,
                                  context_len=20, horizon=30, n_samples=50, device="cuda"):
    """
    Generate trajectories using Oracle mode (z from posterior).

    Oracle sees full sequence including target - upper bound performance.
    """
    # Build full sequence (context + horizon)
    full_seq = log_returns[start_idx:start_idx + context_len + horizon]

    # Get initial IV and GT trajectory
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]
    gt_iv = surfaces[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    sample_iv = np.zeros((n_samples, horizon))

    with torch.no_grad():
        for h in range(horizon):
            # For each horizon step, use full sequence up to that point
            seq_len = context_len + h + 1
            seq = log_returns[start_idx:start_idx + seq_len]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            batch = {"surface": seq_tensor}

            # Oracle: sample from posterior (encoder sees full sequence)
            samples = vae.sample(batch, n_samples=n_samples)
            returns_h = samples[:, 0, -1, 2, 2].cpu().numpy()  # (n_samples,)

            # Accumulate into IV levels
            if h == 0:
                sample_iv[:, h] = initial_iv * np.exp(returns_h)
            else:
                sample_iv[:, h] = sample_iv[:, h-1] * np.exp(returns_h)

    return gt_iv, sample_iv, initial_iv


def generate_trajectories_predictor(vae, predictor, log_returns, surfaces, start_idx,
                                     context_len=20, horizon=30, n_samples=50, device="cuda"):
    """
    Generate trajectories using Prior Predictor mode.

    Context positions use encoder, future positions use trained predictor.
    """
    # Get context only
    context = log_returns[start_idx:start_idx + context_len]

    # Get initial IV and GT trajectory
    initial_iv = surfaces[start_idx + context_len - 1, 2, 2]
    gt_iv = surfaces[start_idx + context_len:start_idx + context_len + horizon, 2, 2]

    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    # Get ctx_emb for full sequence (context + horizon)
    # For context: use actual context
    # For future: repeat last context embedding (simplified)

    sample_iv = np.zeros((n_samples, horizon))
    latent_dim = vae.config["latent_dim"]

    with torch.no_grad():
        # Get context embedding
        ctx_emb_context = vae.ctx_encoder({"surface": context_tensor})  # (1, C, ctx_dim)

        # Get z for context from encoder
        z_ctx_mean, z_ctx_logvar, _ = vae.main_encoder({"surface": context_tensor})

        # Get predicted z for future from predictor
        z_future_mean, z_future_logvar = predictor(context_tensor, horizon=horizon)

        for s in range(n_samples):
            # Sample z for context
            z_ctx = z_ctx_mean + torch.exp(0.5 * z_ctx_logvar) * torch.randn_like(z_ctx_mean)

            # Sample z for each future step from predictor
            z_future = z_future_mean + torch.exp(0.5 * z_future_logvar) * torch.randn_like(z_future_mean)

            # Generate trajectory step by step
            for h in range(horizon):
                # Build sequence up to this horizon
                seq_len = context_len + h + 1

                # ctx_emb: context + zeros for future (simplified)
                ctx_dim = ctx_emb_context.shape[-1]
                ctx_emb_future = torch.zeros(1, h + 1, ctx_dim, device=device)
                ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

                # z: context + future samples
                z = torch.cat([z_ctx, z_future[:, :h+1]], dim=1)

                # Decode
                _, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)
                return_h = sample[0, -1, 2, 2].cpu().numpy()

                # Accumulate
                if h == 0:
                    sample_iv[s, h] = initial_iv * np.exp(return_h)
                else:
                    sample_iv[s, h] = sample_iv[s, h-1] * np.exp(return_h)

    return gt_iv, sample_iv, initial_iv


def plot_single_panel(ax, gt_iv, sample_iv, initial_iv, title, color, show_ylabel=True):
    """Plot a single fan chart panel."""
    horizon = len(gt_iv)
    days = np.arange(1, horizon + 1)

    # Plot sample paths (spaghetti)
    for s in range(sample_iv.shape[0]):
        ax.plot(days, sample_iv[s], color=color, alpha=0.15, linewidth=0.5)

    # Compute and plot CI bands
    p05 = np.percentile(sample_iv, 5, axis=0)
    p50 = np.percentile(sample_iv, 50, axis=0)
    p95 = np.percentile(sample_iv, 95, axis=0)

    ax.fill_between(days, p05, p95, color=color, alpha=0.2, label='90% CI')
    ax.plot(days, p50, color=color, linewidth=2, label='Median')

    # Plot ground truth
    ax.plot(days, gt_iv, 'k-', linewidth=2.5, label='Ground Truth')

    # Mark initial IV
    ax.axhline(y=initial_iv, color='gray', linestyle='--', alpha=0.5)

    # Check CI violations
    violations = (gt_iv < p05) | (gt_iv > p95)
    violation_days = days[violations]
    violation_ivs = gt_iv[violations]

    if len(violation_days) > 0:
        ax.scatter(violation_days, violation_ivs, c='red', s=40, zorder=5,
                  marker='x', linewidths=2)

    # Formatting
    ax.set_xlabel('Horizon (days)')
    if show_ylabel:
        ax.set_ylabel('ATM IV')
    violation_pct = violations.mean() * 100
    ax.set_title(f'{title}\n(Violations: {violation_pct:.0f}%)', fontsize=10)
    ax.grid(True, alpha=0.3)

    return violation_pct


def main():
    print("=" * 70)
    print("Oracle vs Prior Predictor: Fan Chart Comparison")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("\nLoading models...")
    vae, predictor, config = load_models(device)
    print("Models loaded.")

    # Get period indices
    periods = get_period_indices()

    context_len = 20
    horizon = 30
    n_samples = 50

    # Create figure: 4 rows (periods) x 2 columns (oracle, predictor)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    colors = {
        "Vol Spike (Sep 2008)": "#e74c3c",
        "Crisis Peak (Oct 2008)": "#c0392b",
        "Recovery (Mar 2009)": "#27ae60",
        "Debt Ceiling (Aug 2011)": "#9b59b6",
    }

    oracle_violations = []
    predictor_violations = []

    for row, (period_name, start_idx) in enumerate(periods.items()):
        print(f"\nGenerating for {period_name}...")
        color = colors[period_name]

        try:
            # Generate Oracle trajectories
            gt_iv_oracle, sample_iv_oracle, initial_iv = generate_trajectories_oracle(
                vae, log_returns, surfaces, start_idx,
                context_len=context_len, horizon=horizon,
                n_samples=n_samples, device=device
            )

            # Generate Predictor trajectories
            gt_iv_pred, sample_iv_pred, _ = generate_trajectories_predictor(
                vae, predictor, log_returns, surfaces, start_idx,
                context_len=context_len, horizon=horizon,
                n_samples=n_samples, device=device
            )

            # Plot Oracle (left column)
            viol_oracle = plot_single_panel(
                axes[row, 0], gt_iv_oracle, sample_iv_oracle, initial_iv,
                f"ORACLE - {period_name}", "blue", show_ylabel=True
            )
            oracle_violations.append(viol_oracle)

            # Plot Predictor (right column)
            viol_pred = plot_single_panel(
                axes[row, 1], gt_iv_pred, sample_iv_pred, initial_iv,
                f"PREDICTOR - {period_name}", "green", show_ylabel=False
            )
            predictor_violations.append(viol_pred)

            print(f"  Oracle violations: {viol_oracle:.0f}%")
            print(f"  Predictor violations: {viol_pred:.0f}%")

        except Exception as e:
            print(f"  Error: {e}")
            for col in range(2):
                axes[row, col].text(0.5, 0.5, f"Data not available\n{e}",
                                   transform=axes[row, col].transAxes, ha='center', va='center')

    # Add column headers
    axes[0, 0].annotate('ORACLE (z from posterior)', xy=(0.5, 1.15), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=14, fontweight='bold', color='blue')
    axes[0, 1].annotate('PRIOR PREDICTOR (z from trained net)', xy=(0.5, 1.15), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=14, fontweight='bold', color='green')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_oracle = np.mean(oracle_violations)
    avg_pred = np.mean(predictor_violations)
    print(f"Average CI violations:")
    print(f"  Oracle:    {avg_oracle:.1f}%")
    print(f"  Predictor: {avg_pred:.1f}%")
    print(f"  Target:    10%")

    # Add summary to figure
    fig.suptitle(
        f"Oracle vs Prior Predictor | ATM IV 30-day Trajectories | 50 Samples\n"
        f"Avg Violations: Oracle={avg_oracle:.1f}%, Predictor={avg_pred:.1f}%",
        fontsize=14, fontweight='bold', y=0.99
    )

    # Save
    output_path = "models/backfill/two_stage/prior_network/oracle_vs_predictor_fan_chart.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    plt.close()

    return oracle_violations, predictor_violations


if __name__ == "__main__":
    oracle_viol, pred_viol = main()
