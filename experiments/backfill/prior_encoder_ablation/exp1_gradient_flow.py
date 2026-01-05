"""
Experiment 1: Gradient Flow Analysis

Hypothesis: Context encoder receives strong KL gradients, causing over-specialization.

Method:
1. Load trained V4 model
2. Forward pass on batch
3. Compute gradients separately for reconstruction loss and KL loss
4. Measure |grad_KL| vs |grad_recon| to context encoder

Decision Point:
- If |grad_KL| >> |grad_recon|: Confirms confounded training
- If balanced: Gradient imbalance is not the issue

Time: ~5 minutes, NO TRAINING
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.full_covariance_prior import kl_divergence_full_covariance
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def compute_gradient_magnitudes(model, x, context_len):
    """
    Compute gradient magnitudes from reconstruction loss and KL loss separately.

    Returns:
        dict with gradient statistics for context encoder parameters
    """
    model.train()  # Must be in training mode to compute gradients for RNN layers

    # Move data to device
    surface = x["surface"].to(model.device)
    x_device = {"surface": surface}
    if "ex_feats" in x:
        x_device["ex_feats"] = x["ex_feats"].to(model.device)

    # Get context length and horizon
    B, T, H, W = surface.shape
    C = context_len
    horizon = T - C

    # =========================================================================
    # PART 1: Reconstruction Loss Gradients
    # =========================================================================
    model.zero_grad()

    # Forward pass
    output = model.forward(x_device)
    # Model returns tuple, extract first 3 values we need
    if isinstance(output, dict):
        z_mean = output["z_mean"]
        z_log_var = output["z_log_var"]
        recon = output["recon"]
    else:
        # Unpack flexibly - may return (recon, z_mean, z_log_var) or more values
        recon = output[0]
        z_mean = output[1]
        z_log_var = output[2]

    # Compute reconstruction loss only
    target = surface[:, C:, :, :]  # (B, H, 5, 5)
    recon_loss = torch.nn.functional.mse_loss(recon, target, reduction='mean')

    # Backward pass for reconstruction loss
    recon_loss.backward(retain_graph=True)

    # Collect gradients for context encoder
    recon_grads = {}
    for name, param in model.ctx_encoder.named_parameters():
        if param.grad is not None:
            recon_grads[name] = param.grad.clone().detach()
        else:
            recon_grads[name] = torch.zeros_like(param)

    # =========================================================================
    # PART 2: KL Loss Gradients
    # =========================================================================
    model.zero_grad()

    # Get prior distribution
    ctx_surface = surface[:, :C, :, :]
    ctx_input = {"surface": ctx_surface}
    if "ex_feats" in x_device:
        ctx_input["ex_feats"] = x_device["ex_feats"][:, :C, :]

    # Context encoding for prior
    ctx_out = model.ctx_encoder(ctx_input)
    context_summary = ctx_out[:, -1, :]  # (B, latent_dim) - last timestep

    # Prior network
    mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon)

    # Compute KL divergence (only for the horizon timesteps, not context)
    # z_mean and z_log_var are (B, T, D), but prior is only for horizon (B, H, D)
    # Extract only the horizon part: z_mean[:, C:, :]
    z_mean_horizon = z_mean[:, C:, :]  # (B, H, D)
    z_log_var_horizon = z_log_var[:, C:, :]  # (B, H, D)
    kl_loss = kl_divergence_full_covariance(z_mean_horizon, z_log_var_horizon, mu_p, Sigma_p)

    # Backward pass for KL loss
    kl_loss.backward()

    # Collect gradients for context encoder
    kl_grads = {}
    for name, param in model.ctx_encoder.named_parameters():
        if param.grad is not None:
            kl_grads[name] = param.grad.clone().detach()
        else:
            kl_grads[name] = torch.zeros_like(param)

    # =========================================================================
    # PART 3: Compute Statistics
    # =========================================================================

    results = {
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'param_gradients': {}
    }

    # Compute per-parameter gradient magnitudes
    for name in recon_grads.keys():
        recon_grad_norm = torch.norm(recon_grads[name]).item()
        kl_grad_norm = torch.norm(kl_grads[name]).item()
        ratio = kl_grad_norm / (recon_grad_norm + 1e-10)

        results['param_gradients'][name] = {
            'recon_grad_norm': recon_grad_norm,
            'kl_grad_norm': kl_grad_norm,
            'ratio_kl_to_recon': ratio
        }

    return results


def main():
    print("=" * 80)
    print("EXPERIMENT 1: Gradient Flow Analysis")
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
    # Construct ex_data from components
    ex_data = np.stack([data["ret"], data["skews"], data["slopes"]], axis=1)
    ex_data = torch.tensor(ex_data, dtype=torch.float32)

    # Use validation set (same indices as in training script)
    train_start = config.train_start_idx
    train_end = config.train_end_idx

    val_surface = surface[train_end:train_end+1000]
    val_ex_data = ex_data[train_end:train_end+1000]

    print(f"Validation data shape: {val_surface.shape}")
    print()

    # Test on multiple batches to get stable statistics
    # Note: Model trained with horizon=90, need full sequence length
    num_batches = 10
    batch_size = 16
    # Get model horizon from config
    model_horizon = model_data["model_config"].get("horizon", 1)
    seq_len = config.context_len + model_horizon  # Context + horizon (no +1 needed)

    print(f"Testing gradient flow on {num_batches} batches...")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print()

    all_results = []

    for batch_idx in range(num_batches):
        # Sample batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        if end_idx + seq_len > len(val_surface):
            break

        batch_surface = torch.stack([
            val_surface[i:i+seq_len] for i in range(start_idx, end_idx)
        ])
        batch_ex = torch.stack([
            val_ex_data[i:i+seq_len] for i in range(start_idx, end_idx)
        ])

        x = {
            "surface": batch_surface,
            "ex_feats": batch_ex
        }

        # Compute gradients
        with torch.set_grad_enabled(True):
            results = compute_gradient_magnitudes(model, x, config.context_len)

        all_results.append(results)

        print(f"Batch {batch_idx+1}/{num_batches}: recon_loss={results['recon_loss']:.6f}, kl_loss={results['kl_loss']:.3f}")

    print()
    print("=" * 80)
    print("RESULTS: Gradient Flow Analysis")
    print("=" * 80)
    print()

    # Aggregate statistics across batches
    # For each parameter, compute mean gradient norms and ratios

    param_names = list(all_results[0]['param_gradients'].keys())

    print(f"Context Encoder Parameters: {len(param_names)}")
    print()

    # Compute mean statistics
    aggregated = {}
    for name in param_names:
        recon_norms = [r['param_gradients'][name]['recon_grad_norm'] for r in all_results]
        kl_norms = [r['param_gradients'][name]['kl_grad_norm'] for r in all_results]
        ratios = [r['param_gradients'][name]['ratio_kl_to_recon'] for r in all_results]

        aggregated[name] = {
            'recon_grad_norm': np.mean(recon_norms),
            'kl_grad_norm': np.mean(kl_norms),
            'ratio_kl_to_recon': np.mean(ratios)
        }

    # Print per-layer summary
    print("Per-Layer Gradient Magnitudes:")
    print("-" * 80)
    print(f"{'Parameter':<50} {'Recon Grad':<15} {'KL Grad':<15} {'Ratio (KL/Recon)':<15}")
    print("-" * 80)

    for name, stats in aggregated.items():
        print(f"{name:<50} {stats['recon_grad_norm']:<15.6e} {stats['kl_grad_norm']:<15.6e} {stats['ratio_kl_to_recon']:<15.2f}")

    print()

    # Compute overall statistics
    total_recon_grad = sum(stats['recon_grad_norm'] for stats in aggregated.values())
    total_kl_grad = sum(stats['kl_grad_norm'] for stats in aggregated.values())
    overall_ratio = total_kl_grad / (total_recon_grad + 1e-10)

    print("=" * 80)
    print("OVERALL GRADIENT FLOW STATISTICS")
    print("=" * 80)
    print(f"Total Recon Gradient Norm: {total_recon_grad:.6e}")
    print(f"Total KL Gradient Norm:    {total_kl_grad:.6e}")
    print(f"Ratio (KL / Recon):        {overall_ratio:.2f}")
    print()

    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if overall_ratio > 2.0:
        print("✓ GRADIENT CONFOUNDING CONFIRMED")
        print(f"  KL gradients are {overall_ratio:.1f}× larger than reconstruction gradients")
        print("  Context encoder is being pulled by KL loss to over-specialize")
        print("  → Architecture B (Prior Encoder) should help by separating gradients")
    elif overall_ratio > 0.5:
        print("⚠ GRADIENT BALANCE DETECTED")
        print(f"  KL and reconstruction gradients are comparable (ratio={overall_ratio:.2f})")
        print("  Gradient confounding may not be the primary issue")
        print("  → Other factors may contribute to low conditional variance")
    else:
        print("⚠ RECONSTRUCTION DOMINATES")
        print(f"  Reconstruction gradients are {1/overall_ratio:.1f}× larger than KL gradients")
        print("  KL loss may be too weak (kl_weight too small?)")
        print("  → Consider increasing kl_weight")

    print()
    print("=" * 80)

    # Save results
    output_dir = Path("results/prior_encoder_ablation/gradient_flow")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "gradient_flow_results.npz"
    np.savez(
        output_file,
        param_names=np.array(list(aggregated.keys())),
        recon_grad_norms=np.array([aggregated[name]['recon_grad_norm'] for name in aggregated.keys()]),
        kl_grad_norms=np.array([aggregated[name]['kl_grad_norm'] for name in aggregated.keys()]),
        ratios=np.array([aggregated[name]['ratio_kl_to_recon'] for name in aggregated.keys()]),
        total_recon_grad=total_recon_grad,
        total_kl_grad=total_kl_grad,
        overall_ratio=overall_ratio
    )

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
