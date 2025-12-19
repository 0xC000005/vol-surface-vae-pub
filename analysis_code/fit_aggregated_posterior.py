"""
Fit Gaussian Mixture Model to empirical aggregated posterior.

This script:
1. Loads the trained VAE model
2. Runs encoder on all training sequences
3. Fits a GMM to the aggregated latent means
4. Saves parameters for use in prior sampling

**Root Cause**: The VAE shows systematic negative bias in Prior mode because
the aggregated posterior q(z) ≠ N(0,1). The decoder was trained with latents
from q(z|context, target), but at inference we sample from N(0,1).

**Solution**: Fit a GMM to the empirical aggregated posterior and sample from
it instead of N(0,1), matching the distribution the decoder was trained on.

Usage:
    PYTHONPATH=. python analysis_code/fit_aggregated_posterior.py

Output:
    models/backfill/context60_experiment/fitted_prior_gmm.pt
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from pathlib import Path
from vae.cvae_with_mem_randomized import CVAEMemRand


# Key parameters
MODEL_PATH = "models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v2_best.pt"
DATA_PATH = "data/vol_surface_with_ret.npz"
OUTPUT_PATH = "models/backfill/context60_experiment/fitted_prior_gmm.pt"
N_COMPONENTS = 5  # GMM components
CONTEXT_LEN = 60


def load_model(model_path: str):
    """Load trained VAE model."""
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, weights_only=False)

    model = CVAEMemRand(model_data["model_config"])
    model.load_weights(dict_to_load=model_data)
    model.eval()

    # Ensure model is in double precision to match data
    model = model.double()

    print(f"  Model loaded successfully")
    print(f"  Latent dim: {model.config['latent_dim']}")
    print(f"  Context len: {CONTEXT_LEN}")

    return model


def fit_aggregated_posterior():
    """
    Fit GMM to empirical aggregated posterior q(z) = E_x[q(z|x)].

    Returns:
        dict: Fitted GMM parameters
    """
    print("=" * 80)
    print("FITTING AGGREGATED POSTERIOR")
    print("=" * 80)
    print()

    # Load model
    model = load_model(MODEL_PATH)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    surfaces = data['surface']  # (N_days, 5, 5)
    ex_data = np.stack([data['ret'], data['skews'], data['slopes']], axis=1)  # (N_days, 3)
    print(f"  Loaded {len(surfaces)} days of data")

    # Collect latent means from encoder
    print(f"\nEncoding sequences to collect latent means...")
    all_z_means = []

    with torch.no_grad():
        for i in range(CONTEXT_LEN, len(surfaces) - 90):
            # Get context + one step ahead
            seq_surface = surfaces[i-CONTEXT_LEN:i+1]  # (C+1, 5, 5)
            seq_ex = ex_data[i-CONTEXT_LEN:i+1]  # (C+1, 3)

            x = {
                "surface": torch.tensor(seq_surface).unsqueeze(0).double().to(model.device),
                "ex_feats": torch.tensor(seq_ex).unsqueeze(0).double().to(model.device)
            }
            z_mean, z_logvar, z = model.encoder(x)

            # Store latent for final timestep (most relevant for future prediction)
            all_z_means.append(z_mean[0, -1, :].cpu().numpy())

            if (i - CONTEXT_LEN + 1) % 500 == 0:
                print(f"  Processed {i - CONTEXT_LEN + 1}/{len(surfaces) - CONTEXT_LEN - 90} sequences...")

    all_z_means = np.array(all_z_means)  # (N, latent_dim)
    print(f"\n✓ Collected {len(all_z_means)} latent vectors")
    print(f"\nEmpirical posterior statistics:")
    print(f"  Mean: {all_z_means.mean(axis=0)}")
    print(f"  Std:  {all_z_means.std(axis=0)}")
    print(f"  Min:  {all_z_means.min(axis=0)}")
    print(f"  Max:  {all_z_means.max(axis=0)}")

    # Check if close to N(0,1)
    mean_norm = np.linalg.norm(all_z_means.mean(axis=0))
    std_mean = all_z_means.std(axis=0).mean()
    print(f"\nDistance from N(0,1):")
    print(f"  ||mean||: {mean_norm:.4f} (ideal: 0.0)")
    print(f"  avg(std): {std_mean:.4f} (ideal: 1.0)")

    if mean_norm < 0.1 and abs(std_mean - 1.0) < 0.1:
        print("  ⚠️  Posterior is close to N(0,1). GMM may not provide much benefit.")
    else:
        print("  ✓ Posterior differs from N(0,1). GMM fitting is beneficial.")

    # Fit GMM
    print(f"\nFitting GMM with {N_COMPONENTS} components...")
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type='full',
        random_state=42,
        max_iter=200,
        verbose=1
    )
    gmm.fit(all_z_means)

    print(f"\n✓ GMM converged: {gmm.converged_}")
    print(f"  n_iter: {gmm.n_iter_}")
    print(f"  Log-likelihood: {gmm.score(all_z_means):.2f}")

    # Save parameters
    print(f"\nSaving fitted prior to {OUTPUT_PATH}...")
    params = {
        'means': torch.tensor(gmm.means_, dtype=torch.float64),
        'covariances': torch.tensor(gmm.covariances_, dtype=torch.float64),
        'weights': torch.tensor(gmm.weights_, dtype=torch.float64),
        'n_components': N_COMPONENTS,
        'latent_dim': all_z_means.shape[1],
        'training_stats': {
            'empirical_mean': all_z_means.mean(axis=0).tolist(),
            'empirical_std': all_z_means.std(axis=0).tolist(),
            'n_samples': int(len(all_z_means)),
            'mean_norm': float(mean_norm),
            'std_mean': float(std_mean)
        }
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, OUTPUT_PATH)

    print(f"✓ Saved fitted prior parameters")
    print(f"  File size: {Path(OUTPUT_PATH).stat().st_size / 1024:.1f} KB")

    # Print GMM component info
    print(f"\nGMM Component Weights:")
    for i, w in enumerate(gmm.weights_):
        print(f"  Component {i+1}: {w:.4f} ({w*100:.1f}%)")

    print()
    print("=" * 80)
    print("FITTING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Use --prior_mode fitted when generating predictions")
    print("  2. Compare fanning patterns: standard vs fitted")
    print("  3. Validate marginal distribution matching")

    return params


if __name__ == "__main__":
    fit_aggregated_posterior()
