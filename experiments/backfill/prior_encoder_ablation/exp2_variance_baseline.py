"""
Experiment 2: Diagnostic Variance Baseline

Hypothesis: Prior is "too certain" - measure this directly.

Method:
1. Load trained V4 model
2. For 1000 test contexts:
   - Sample z ~ p(z|context) from prior (100 samples each)
   - Decode to surfaces
   - Compute Var(surface|context) at each grid point
3. Compute E[Var(X|C)] / Var(X)

Expected: Establishes baseline (0.44%) for improvement measurement.

Time: ~15 minutes, NO TRAINING
"""

import torch
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def sample_from_prior(model, context, num_samples=100):
    """
    Sample from prior p(z|context) and decode to surfaces.

    Args:
        model: Trained CVAEFullCovPrior
        context: (B, C, 5, 5) context surfaces
        num_samples: Number of samples to generate per context

    Returns:
        surfaces: (B, num_samples, 1, 5, 5) generated surfaces
    """
    model.eval()
    B, C, H, W = context.shape

    with torch.no_grad():
        # Get context encoding
        ctx_input = {"surface": context}
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]  # (B, latent_dim)

        # Get prior distribution
        horizon = 1  # Generate 1-day-ahead
        mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon)
        # mu_p: (B, H, latent_dim), Sigma_p: (H, H)

        # Sample from prior
        # z ~ N(mu_p, Sigma_p)
        all_samples = []

        for _ in range(num_samples):
            # Cholesky decomposition
            L = torch.linalg.cholesky(Sigma_p + 1e-6 * torch.eye(Sigma_p.shape[0], device=Sigma_p.device))

            # Sample epsilon ~ N(0, I)
            epsilon = torch.randn(B, horizon, model.latent_dim, device=model.device)

            # Transform: z = mu + L @ epsilon
            # L: (H, H), epsilon: (B, H, D)
            # Need to expand L for batch dimension
            L_expanded = L.unsqueeze(0).expand(B, -1, -1)  # (B, H, H)

            # Reshape for batch matrix multiplication
            epsilon_flat = epsilon.view(B, horizon, model.latent_dim)
            z_sample = mu_p + torch.bmm(L_expanded, epsilon_flat.transpose(1, 2)).transpose(1, 2)
            # z_sample: (B, H, D)

            # Decode
            z_input = z_sample.reshape(B * horizon, model.latent_dim)
            decoded = model.decoder(z_input)  # (B*H, 5, 5)
            decoded = decoded.reshape(B, horizon, H, W)  # (B, H, 5, 5)

            all_samples.append(decoded)

        # Stack samples: (B, num_samples, H, 5, 5)
        surfaces = torch.stack(all_samples, dim=1)

    return surfaces


def compute_conditional_variance(model, val_surface, num_contexts=1000, num_samples=100):
    """
    Compute E[Var(X|C)] / Var(X) for conditional variance ratio.

    Args:
        model: Trained model
        val_surface: Validation surface data
        num_contexts: Number of test contexts to evaluate
        num_samples: Number of samples per context

    Returns:
        dict with variance statistics
    """
    C = model.context_len
    device = model.device

    print(f"Computing conditional variance on {num_contexts} contexts...")
    print(f"Generating {num_samples} samples per context...")
    print()

    # Collect all generated surfaces
    all_samples_per_context = []
    all_ground_truth = []

    for i in tqdm(range(num_contexts), desc="Sampling contexts"):
        if i + C + 1 > len(val_surface):
            break

        # Get context
        context = val_surface[i:i+C].unsqueeze(0).to(device)  # (1, C, 5, 5)
        ground_truth = val_surface[i+C].unsqueeze(0).to(device)  # (1, 5, 5)

        # Sample from prior
        samples = sample_from_prior(model, context, num_samples=num_samples)
        # samples: (1, num_samples, 1, 5, 5)

        samples = samples.squeeze(0).squeeze(1)  # (num_samples, 5, 5)

        all_samples_per_context.append(samples.cpu())
        all_ground_truth.append(ground_truth.cpu())

    # Convert to numpy
    all_samples = torch.stack(all_samples_per_context).numpy()  # (num_contexts, num_samples, 5, 5)
    all_ground_truth = torch.stack(all_ground_truth).squeeze(1).numpy()  # (num_contexts, 5, 5)

    print()
    print(f"Generated samples shape: {all_samples.shape}")
    print(f"Ground truth shape: {all_ground_truth.shape}")
    print()

    # Compute within-context variance: Var(X|C)
    # For each context, compute variance across samples at each grid point
    var_given_context = np.var(all_samples, axis=1)  # (num_contexts, 5, 5)

    # Average across contexts: E[Var(X|C)]
    expected_conditional_var = np.mean(var_given_context)

    # Compute total variance: Var(X)
    # Flatten all ground truth surfaces
    all_gt_flat = all_ground_truth.reshape(-1, 5, 5)
    total_var = np.var(all_gt_flat, axis=0)  # (5, 5)
    mean_total_var = np.mean(total_var)

    # Conditional variance ratio
    ratio = expected_conditional_var / mean_total_var

    # Compute per-grid-point statistics
    mean_var_per_grid = np.mean(var_given_context, axis=0)  # (5, 5)
    ratio_per_grid = mean_var_per_grid / total_var  # (5, 5)

    results = {
        'expected_conditional_var': expected_conditional_var,
        'mean_total_var': mean_total_var,
        'conditional_var_ratio': ratio,
        'var_given_context': var_given_context,
        'mean_var_per_grid': mean_var_per_grid,
        'total_var_per_grid': total_var,
        'ratio_per_grid': ratio_per_grid,
        'all_samples': all_samples,
        'all_ground_truth': all_ground_truth
    }

    return results


def main():
    print("=" * 80)
    print("EXPERIMENT 2: Diagnostic Variance Baseline")
    print("=" * 80)
    print()

    # Load configuration
    config = BackfillContext60ConfigV4FullCov

    # Find the trained model
    checkpoint_path = Path("models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_phase1_ep99.pt")

    if not checkpoint_path.exists():
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        print("Please ensure the model has been trained.")
        return

    print(f"Loading model from: {checkpoint_path}")
    model_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Strip _orig_mod. prefix from compiled models
    if "model" in model_data and any(k.startswith("_orig_mod.") for k in model_data["model"].keys()):
        model_data["model"] = {k.replace("_orig_mod.", ""): v for k, v in model_data["model"].items()}

    # Initialize model
    model = CVAEFullCovPrior(model_data["model_config"])
    model.load_weights(dict_to_load=model_data)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device = device
    print(f"Model loaded on device: {device}")
    print()

    # Load data
    print("Loading data...")
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)

    surface = torch.tensor(data["surface"], dtype=torch.float32)

    # Use validation set
    train_end = config.train_end_idx
    val_surface = surface[train_end:train_end+2000]

    print(f"Validation data shape: {val_surface.shape}")
    print()

    # Compute conditional variance
    results = compute_conditional_variance(
        model,
        val_surface,
        num_contexts=1000,
        num_samples=100
    )

    print("=" * 80)
    print("RESULTS: Conditional Variance Analysis")
    print("=" * 80)
    print()

    print(f"E[Var(X|C)]:              {results['expected_conditional_var']:.8f}")
    print(f"Var(X):                   {results['mean_total_var']:.8f}")
    print(f"Conditional Var Ratio:    {results['conditional_var_ratio']:.4%}")
    print()

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    ratio_pct = results['conditional_var_ratio'] * 100

    if ratio_pct < 1.0:
        print(f"⚠ VERY LOW CONDITIONAL VARIANCE: {ratio_pct:.2f}%")
        print("  Prior is 'too certain' - same context produces nearly identical outputs")
        print("  Target: >2-5% for reasonable uncertainty")
        print("  → Architecture B should increase conditional variance")
    elif ratio_pct < 5.0:
        print(f"⚠ LOW CONDITIONAL VARIANCE: {ratio_pct:.2f}%")
        print("  Prior provides some diversity but still conservative")
        print("  Target: >5% for good uncertainty quantification")
    else:
        print(f"✓ HEALTHY CONDITIONAL VARIANCE: {ratio_pct:.2f}%")
        print("  Prior provides sufficient diversity across contexts")

    print()

    # Per-grid analysis
    print("=" * 80)
    print("PER-GRID CONDITIONAL VARIANCE RATIO")
    print("=" * 80)
    print()
    print("Ratio = E[Var(X|C)] / Var(X) at each grid point:")
    print()

    ratio_grid = results['ratio_per_grid']
    for i in range(5):
        row_str = " ".join([f"{ratio_grid[i, j]:.4f}" for j in range(5)])
        print(f"  Row {i}: {row_str}")

    print()

    min_ratio = np.min(ratio_grid)
    max_ratio = np.max(ratio_grid)
    print(f"Min ratio: {min_ratio:.4f} ({min_ratio*100:.2f}%)")
    print(f"Max ratio: {max_ratio:.4f} ({max_ratio*100:.2f}%)")
    print()

    # Save results
    output_dir = Path("results/prior_encoder_ablation/variance_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "variance_baseline_results.npz"
    np.savez(
        output_file,
        expected_conditional_var=results['expected_conditional_var'],
        mean_total_var=results['mean_total_var'],
        conditional_var_ratio=results['conditional_var_ratio'],
        mean_var_per_grid=results['mean_var_per_grid'],
        total_var_per_grid=results['total_var_per_grid'],
        ratio_per_grid=results['ratio_per_grid'],
        # Save aggregated stats only, not all samples (too large)
    )

    print(f"Results saved to: {output_file}")
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()
