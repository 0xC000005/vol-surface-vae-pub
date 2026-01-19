"""
Diagnose Prior Network prediction quality.

Key questions:
1. What is the prior network actually predicting?
2. Why does it have no correlation with actual next step?
3. What was it trained to predict?
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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

    # Load Prior Network
    prior_path = "models/backfill/two_stage/prior_network/prior_network_best.pt"
    prior_ckpt = torch.load(prior_path, map_location=device, weights_only=False)
    prior_config = prior_ckpt["config"].copy()
    prior_config["device"] = device
    prior_config["hidden_size"] = 8  # From saved model

    prior = LatentPredictor(prior_config)
    prior.load_state_dict(prior_ckpt["predictor_state_dict"])
    prior = prior.to(device)
    prior.eval()

    prior_context_len = prior_ckpt.get("context_len", 20)

    return vae, prior, vae_config, prior_context_len


def main():
    print("=" * 70)
    print("DIAGNOSIS: Prior Network Prediction Quality")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, prior, config, context_len = load_models(device)

    print(f"\nPrior Network context_len: {context_len}")
    print(f"VAE latent_dim: {config['latent_dim']}")

    # Test data
    train_end = int(len(log_returns) * 0.7)
    n_test = 100

    # Collect z values from different sources
    z_encoder_list = []  # z from VAE encoder (oracle)
    z_prior_list = []    # z from prior network (prediction)
    actual_next_list = []  # Actual next step

    print("\n" + "=" * 70)
    print("Collecting z values from encoder and prior network...")
    print("=" * 70)

    for i in range(n_test):
        idx = train_end + i
        context = log_returns[idx:idx + context_len]  # (C, 5, 5)
        target = log_returns[idx + context_len]  # Next step (ground truth)

        # Full sequence for encoder
        full_seq = log_returns[idx:idx + context_len + 1]  # (C+1, 5, 5)

        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
        full_tensor = torch.tensor(full_seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # z from VAE encoder (sees full sequence including target)
            z_encoder, _, _ = vae.main_encoder({"surface": full_tensor})
            z_encoder_next = z_encoder[0, context_len].cpu().numpy()  # z at prediction position

            # z from prior network (only sees context)
            z_prior, z_logvar = prior(ctx_tensor, horizon=1)
            z_prior = z_prior[0, 0].cpu().numpy()  # First (only) prediction

        z_encoder_list.append(z_encoder_next)
        z_prior_list.append(z_prior)
        actual_next_list.append(target[2, 2])  # ATM

    z_encoder_arr = np.array(z_encoder_list)  # (n_test, latent_dim)
    z_prior_arr = np.array(z_prior_list)  # (n_test, latent_dim)

    # Check correlation between encoder z and prior z
    print("\nCorrelation between z_encoder and z_prior per dimension:")
    for d in range(z_encoder_arr.shape[1]):
        corr = np.corrcoef(z_encoder_arr[:, d], z_prior_arr[:, d])[0, 1]
        print(f"  dim {d}: {corr:.4f}")

    print("\nOverall z statistics:")
    print(f"  z_encoder mean: {z_encoder_arr.mean():.4f}, std: {z_encoder_arr.std():.4f}")
    print(f"  z_prior mean:   {z_prior_arr.mean():.4f}, std: {z_prior_arr.std():.4f}")

    # Check if prior is predicting useful information
    print("\n" + "=" * 70)
    print("What does prior z predict?")
    print("=" * 70)

    # Decode prior z and compare to actual
    decoded_means = []
    for i in range(min(50, n_test)):
        idx = train_end + i
        context = log_returns[idx:idx + context_len]

        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get ctx_emb
            ctx_emb = vae.ctx_encoder({"surface": ctx_tensor})
            ctx_emb_last = ctx_emb[:, -1:, :]  # (1, 1, ctx_dim)

            # Get z from prior
            z_prior, _ = prior(ctx_tensor, horizon=1)

            # Decode
            mean, _, _, _ = vae.decoder(ctx_emb_last, z_prior, sample=False)

        decoded_means.append(mean[0, 0, 2, 2].cpu().item())

    decoded_means = np.array(decoded_means)
    actual_next = np.array(actual_next_list[:50])

    corr_decoded = np.corrcoef(decoded_means, actual_next)[0, 1]

    print(f"\nDecoded mean from prior z vs actual next step (ATM):")
    print(f"  Correlation: {corr_decoded:.4f}")
    print(f"  Decoded mean std: {decoded_means.std():.6f}")
    print(f"  Actual std:       {actual_next.std():.6f}")

    # KEY INSIGHT: What was the prior trained to predict?
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    print("""
The prior network was trained to minimize:
    MSE(z_prior, z_encoder)

where z_encoder comes from the VAE's main_encoder which sees the FULL sequence.

This means:
1. The prior network tries to predict what z_encoder would output
2. z_encoder has access to the target (non-causal)
3. z_encoder encodes information about x[t] at position t

So the prior network is trying to predict the latent encoding of a sequence
that INCLUDES the target. But without seeing the target, the prior can only
output the "average" z, which is near zero.

The prior network CAN'T predict the actual next step because:
1. It was never trained to do that
2. It was trained to predict z_encoder, which has target information
3. Without target information, the best prior can do is output E[z|context] ≈ 0

This is a fundamental architecture issue: the VAE was designed for reconstruction,
not prediction. Adding a prior network doesn't fix this - it just predicts
what the encoder would output IF it had seen the target.

SOLUTION: Train a model that is specifically designed for PREDICTION:
1. Causal architecture where z[t] doesn't see x[t]
2. Loss function that optimizes predicting x[t+1] from context
3. Or use a different model class entirely (e.g., autoregressive transformer)
""")


if __name__ == "__main__":
    main()
