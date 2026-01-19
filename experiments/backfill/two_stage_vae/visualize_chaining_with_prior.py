"""
Autoregressive chaining using the Prior Network (LatentPredictor).

The key insight is that the original chaining script used the VAE's main_encoder,
which sees the current input (z[t] includes x[t]). This makes it RECONSTRUCTION,
not PREDICTION.

For actual prediction, we need to use the Prior Network which:
1. Takes only context (no target information)
2. Predicts z for future positions
3. Decoder then uses (ctx_emb, z_predicted) to generate predictions

This script compares:
- Oracle: VAE main_encoder (sees target, reconstruction)
- Prior Network: Uses predictor (context only, true prediction)

Usage:
    python experiments/backfill/two_stage_vae/visualize_chaining_with_prior.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.predictors import LatentPredictor


def load_models(device: str = "cuda"):
    """Load the VAE and Prior Network."""
    # Load VAE
    vae_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae_config = vae_ckpt["model_config"]
    vae_config["device"] = device
    vae = CVAETwoStageStudentTMLP(vae_config)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Load Prior Network - use the saved config directly
    prior_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    prior_ckpt = torch.load(prior_path, map_location=device, weights_only=False)
    prior_config = prior_ckpt["config"].copy()
    prior_config["device"] = device

    # Override hidden_size to match saved model (LSTM weights show hidden_size=8)
    # LSTM weight_ih has shape [4*hidden_size, input_size] = [32, 50]
    # So hidden_size = 32/4 = 8
    prior_config["hidden_size"] = 8

    prior = LatentPredictor(prior_config)
    prior.load_state_dict(prior_ckpt["predictor_state_dict"])
    prior = prior.to(device)
    prior.eval()

    prior_context_len = prior_ckpt.get("context_len", 20)

    print(f"Loaded VAE from {vae_path}")
    print(f"Loaded Prior Network from {prior_path}")
    print(f"  Prior context_len: {prior_context_len}")

    return vae, prior, vae_config, prior_context_len


def to_log_returns(surfaces):
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns


def predict_with_prior(vae, prior, context, device):
    """
    Make a 1-step prediction using the Prior Network.

    Args:
        vae: Student-t VAE (frozen)
        prior: LatentPredictor (for z prediction)
        context: (C, 5, 5) numpy array of log-returns

    Returns:
        mean_pred: (5, 5) predicted log-return
        sample_pred: (5, 5) sampled log-return
    """
    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get ctx_emb from VAE context encoder
        ctx_emb = vae.ctx_encoder({"surface": ctx_tensor})  # (1, C, ctx_dim)

        # Get z prediction from Prior Network (1-step ahead)
        z_pred, z_logvar = prior(ctx_tensor, horizon=1)  # (1, 1, latent_dim)

        # Extend ctx_emb for the prediction position
        # ctx_emb[-1] = f(x_0:C-1), which is what we want for predicting position C
        ctx_emb_pred = ctx_emb[:, -1:, :]  # (1, 1, ctx_dim)

        # Decode with predicted z
        mean, sample, factor, log_diag = vae.decoder(ctx_emb_pred, z_pred, sample=True)

    return mean[0, 0].cpu().numpy(), sample[0, 0].cpu().numpy()


def predict_with_oracle(vae, context, device):
    """
    Make a 1-step prediction using the VAE main_encoder (oracle - sees target).

    This is NOT true prediction because z[t] sees x[t].
    Included for comparison only.
    """
    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = vae.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, z = vae.main_encoder({"surface": ctx_tensor})
        mean, sample, _, _ = vae.decoder(ctx_emb, z, sample=True)

    # Output[-1] is reconstruction of input[-1], NOT prediction
    return mean[0, -1].cpu().numpy(), sample[0, -1].cpu().numpy()


def chain_with_prior(vae, prior, context, starting_iv, horizon=30, n_samples=50, device="cuda"):
    """
    Autoregressive chaining using Prior Network for true prediction.

    Args:
        vae: Student-t VAE
        prior: LatentPredictor
        context: Initial context (C, 5, 5) log-returns
        starting_iv: Starting IV surface (5, 5)
        horizon: Number of steps to chain
        n_samples: Number of trajectories to generate

    Returns:
        trajectories: (n_samples, horizon, 5, 5) IV surfaces
    """
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))

    for s in range(n_samples):
        current_iv = starting_iv.copy()
        current_context = context.copy()

        for h in range(horizon):
            # Predict using prior network
            mean_pred, sample_pred = predict_with_prior(vae, prior, current_context, device)

            # Use mean for deterministic trajectory, sample for stochastic
            if s == 0:
                log_ret = mean_pred  # First trajectory is deterministic
            else:
                log_ret = sample_pred  # Others are stochastic

            # Convert to IV
            new_iv = current_iv * np.exp(log_ret)
            all_trajectories[s, h] = new_iv

            # Update context for next step
            current_iv = new_iv
            current_context = np.concatenate([current_context[1:], log_ret[np.newaxis]], axis=0)

    return all_trajectories


def test_single_step_prediction(vae, prior, log_returns, device, n_test=200, context_len=20):
    """Test single-step prediction quality of Prior Network vs Oracle."""
    train_end = int(len(log_returns) * 0.7)

    prior_preds = []
    oracle_preds = []
    actual_values = []

    print("\nTesting single-step prediction quality...")

    for i in range(n_test):
        idx = train_end + i
        context = log_returns[idx:idx + context_len]
        actual_next = log_returns[idx + context_len]

        # Prior network prediction
        prior_mean, _ = predict_with_prior(vae, prior, context, device)

        # Oracle prediction (for comparison)
        oracle_mean, _ = predict_with_oracle(vae, context, device)

        prior_preds.append(prior_mean[2, 2])  # ATM
        oracle_preds.append(oracle_mean[2, 2])
        actual_values.append(actual_next[2, 2])

    prior_preds = np.array(prior_preds)
    oracle_preds = np.array(oracle_preds)
    actual_values = np.array(actual_values)

    # Compute correlations
    prior_corr = np.corrcoef(prior_preds, actual_values)[0, 1]
    oracle_corr = np.corrcoef(oracle_preds, actual_values)[0, 1]

    print("\nSingle-Step Prediction Correlation (ATM):")
    print(f"  Prior Network: {prior_corr:.4f}")
    print(f"  Oracle (VAE):  {oracle_corr:.4f}")
    print(f"  Actual std:    {actual_values.std():.6f}")
    print(f"  Prior std:     {prior_preds.std():.6f}")
    print(f"  Oracle std:    {oracle_preds.std():.6f}")

    return prior_corr, oracle_corr


def main():
    print("=" * 70)
    print("Autoregressive Chaining with Prior Network")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    vae, prior, config, prior_context_len = load_models(device)

    # Test single-step prediction (use prior's context length)
    prior_corr, oracle_corr = test_single_step_prediction(
        vae, prior, log_returns, device, context_len=prior_context_len
    )

    # Setup for chaining
    context_len = prior_context_len  # Use prior network's context length
    horizon = 30
    n_samples = 50

    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100

    # Get context and ground truth
    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]
    gt_surfaces = surfaces[start_idx + context_len:start_idx + context_len + horizon]

    print(f"\nChaining setup:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  Starting IV (ATM): {starting_iv[2, 2]:.4f}")
    print(f"  GT IV range: {gt_surfaces.min():.4f} - {gt_surfaces.max():.4f}")

    # Generate trajectories
    print("\nGenerating trajectories with Prior Network...")
    trajectories = chain_with_prior(
        vae, prior, context, starting_iv,
        horizon=horizon, n_samples=n_samples, device=device
    )

    print(f"  Generated IV range: {trajectories.min():.4f} - {trajectories.max():.4f}")
    print(f"  Exploded: {trajectories.max() > 5.0}")

    # Create fan chart
    save_dir = Path("results/two_stage_vae/prior_network_chaining")
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

        # Prior network trajectories
        traj_vals = trajectories[:, :, i, j]
        median = np.median(traj_vals, axis=0)
        p10 = np.percentile(traj_vals, 10, axis=0)
        p90 = np.percentile(traj_vals, 90, axis=0)

        ax.plot(days, gt_vals, 'k-', linewidth=2, label='Ground Truth')
        ax.plot(days, median, 'b-', linewidth=1.5, label='Prior Network Median')
        ax.fill_between(days, p10, p90, alpha=0.3, color='blue', label='10-90% CI')

        ax.set_title(name)
        ax.set_xlabel('Days')
        ax.set_ylabel('IV')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'30-Day Chaining with Prior Network\n'
                 f'Single-step corr: Prior={prior_corr:.3f}, Oracle={oracle_corr:.3f}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / 'prior_network_fan_chart.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'prior_network_fan_chart.png'}")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Single-step prediction correlation (Prior Network): {prior_corr:.4f}")
    print(f"Single-step prediction correlation (Oracle VAE):    {oracle_corr:.4f}")
    print(f"Generated IV range: {trajectories.min():.4f} - {trajectories.max():.4f}")
    print(f"GT IV range:        {gt_surfaces.min():.4f} - {gt_surfaces.max():.4f}")


if __name__ == "__main__":
    main()
