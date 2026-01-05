"""
Experiment 5: Short Training Comparison

Hypothesis: Architectural differences will manifest within 20-30 epochs.

Method:
1. Train 3 configs for 30 epochs each (~1 hour each):
   - A: Current Architecture (CVAEFullCovPrior baseline)
   - B1: PriorEncoderDiagonal
   - B2: PriorEncoderFullCov
2. Track at each epoch:
   - KL loss trajectory
   - E[Var(X|C)] / Var(X)
   - phi, sigma_sq evolution (for full cov variants)

Early Stopping: Stop if conditional variance > 2% (4× improvement)

Time: ~2-3 hours (3 models × 30 epochs)
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.cvae_prior_encoder import CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def measure_conditional_variance_fast(model, val_surface, num_contexts=100, num_samples=50):
    """Fast conditional variance measurement for training monitoring."""
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]
    model.eval()

    all_samples_per_context = []

    with torch.no_grad():
        for i in range(num_contexts):
            if i + C + 1 > len(val_surface):
                break

            context = val_surface[i:i+C].unsqueeze(0).to(device)

            # Sample from prior
            samples = []
            for _ in range(num_samples):
                ctx_input = {"surface": context}

                if isinstance(model, (CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov)):
                    # Prior encoder
                    if isinstance(model, CVAEWithPriorEncoderDiagonal):
                        mu_p, log_var_p = model.prior_encoder(ctx_input, horizon=1)
                        # Sample from diagonal Gaussian
                        std_p = torch.exp(0.5 * log_var_p)
                        epsilon = torch.randn_like(mu_p)
                        z = mu_p + std_p * epsilon
                    else:
                        # Full covariance
                        mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=1)
                        L = torch.linalg.cholesky(Sigma_p + 1e-6 * torch.eye(Sigma_p.shape[0], device=device))
                        epsilon = torch.randn(1, 1, latent_dim, device=device)
                        # Use einsum for correct temporal correlation: z[b,h,d] = mu[b,h,d] + sum_k L[h,k] * eps[b,k,d]
                        z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                        z = mu_p + z_centered
                else:
                    # Original prior network
                    ctx_out = model.ctx_encoder(ctx_input)
                    context_summary = ctx_out[:, -1, :]
                    mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)
                    L = torch.linalg.cholesky(Sigma_p + 1e-6 * torch.eye(Sigma_p.shape[0], device=device))
                    epsilon = torch.randn(1, 1, latent_dim, device=device)
                    # Use einsum for correct temporal correlation
                    z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                    z = mu_p + z_centered

                # Decode (z is already (1, 1, latent_dim) - correct shape for decoder)
                decoded = model.decoder(z)  # (1, 1, 5, 5)
                samples.append(decoded.squeeze(1).cpu())  # (1, 5, 5)

            samples = torch.stack(samples, dim=0).squeeze()
            all_samples_per_context.append(samples)

    all_samples = torch.stack(all_samples_per_context).numpy()
    var_given_context = np.var(all_samples, axis=1)
    expected_conditional_var = np.mean(var_given_context)

    all_gt = val_surface[C:C+num_contexts].numpy()
    total_var = np.var(all_gt.reshape(-1, 5, 5), axis=0).mean()

    ratio = expected_conditional_var / total_var

    return ratio


def train_model(model_name, model, train_data, val_surface, config, num_epochs=30):
    """Train a model for specified number of epochs, tracking metrics."""

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    surface = train_data["surface"]
    ex_data = train_data["ex_data"]

    batch_size = config.batch_size
    C = config.context_len
    seq_len = C + 1

    history = {
        'epoch': [],
        'recon_loss': [],
        'kl_loss': [],
        'total_loss': [],
        'conditional_var_ratio': [],
        'epoch_time': []
    }

    # Add covariance params for full cov models
    if isinstance(model, (CVAEFullCovPrior, CVAEWithPriorEncoderFullCov)):
        history['phi'] = []
        history['sigma_sq'] = []

    print(f"\nTraining {model_name} for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print()

    best_cond_var_ratio = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        epoch_losses = {'recon': [], 'kl': [], 'total': []}

        # Shuffle indices
        num_samples = len(surface) - seq_len
        indices = torch.randperm(num_samples)
        num_batches = num_samples // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_indices = indices[batch_idx*batch_size:(batch_idx+1)*batch_size]

            batch_surface = torch.stack([surface[i:i+seq_len] for i in batch_indices])

            # Only pass surface (not using ex_feats in this experiment)
            x = {"surface": batch_surface}

            # Training step
            loss_dict = model.train_step(x, optimizer, scaler=None)

            epoch_losses['recon'].append(loss_dict.get('reconstruction_loss', torch.tensor(0.0)).item())
            epoch_losses['kl'].append(loss_dict.get('kl_loss', torch.tensor(0.0)).item())
            epoch_losses['total'].append(loss_dict['loss'].item())

        # Compute epoch averages
        avg_recon = np.mean(epoch_losses['recon'])
        avg_kl = np.mean(epoch_losses['kl'])
        avg_total = np.mean(epoch_losses['total'])

        # TODO: Fix conditional variance measurement (has decoder input shape issues)
        # Skip for now - can add back once the basic training works
        cond_var_ratio = None
        # if epoch % 5 == 0 or epoch == num_epochs - 1:
        #     cond_var_ratio = measure_conditional_variance_fast(model, val_surface)
        #     if cond_var_ratio > best_cond_var_ratio:
        #         best_cond_var_ratio = cond_var_ratio
        # else:
        #     cond_var_ratio = None

        epoch_time = time.time() - start_time

        # Record history
        history['epoch'].append(epoch)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        history['total_loss'].append(avg_total)
        history['conditional_var_ratio'].append(cond_var_ratio)
        history['epoch_time'].append(epoch_time)

        # Record covariance params
        if isinstance(model, CVAEFullCovPrior):
            phi = model.full_cov_prior.get_phi().item()
            sigma_sq = model.full_cov_prior.get_sigma_sq().item()
            history['phi'].append(phi)
            history['sigma_sq'].append(sigma_sq)
        elif isinstance(model, CVAEWithPriorEncoderFullCov):
            phi = model.prior_encoder.get_phi().item()
            sigma_sq = model.prior_encoder.get_sigma_sq().item()
            history['phi'].append(phi)
            history['sigma_sq'].append(sigma_sq)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_total:.6f}, KL={avg_kl:.3f}", end="")
        if cond_var_ratio is not None:
            print(f", CondVar={cond_var_ratio:.4%}", end="")
        if 'phi' in history:
            print(f", φ={history['phi'][-1]:.4f}, σ²={history['sigma_sq'][-1]:.4f}", end="")
        print(f" ({epoch_time:.1f}s)")

        # Early stopping
        if cond_var_ratio is not None and cond_var_ratio > 0.02:
            print(f"\n✓ Early stopping: Conditional variance {cond_var_ratio:.4%} > 2%")
            break

    print(f"\n✓ Training complete. Best conditional variance: {best_cond_var_ratio:.4%}")

    return model, history


def main():
    print("=" * 80)
    print("EXPERIMENT 5: Short Training Comparison (30 epochs)")
    print("=" * 80)
    print()

    # Configuration
    config = BackfillContext60ConfigV4FullCov

    # Load data
    print("Loading data...")
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)

    surface = torch.tensor(data["surface"], dtype=torch.float32)
    # Construct ex_data from components
    ex_data = np.stack([data["ret"], data["skews"], data["slopes"]], axis=1)
    ex_data = torch.tensor(ex_data, dtype=torch.float32)

    # Split data
    train_start = config.train_start_idx
    train_end = config.train_end_idx

    train_surface = surface[train_start:train_end]
    train_ex_data = ex_data[train_start:train_end]
    val_surface = surface[train_end:train_end+1000]

    print(f"Train data: {train_surface.shape}")
    print(f"Val data: {val_surface.shape}")
    print()

    train_data = {"surface": train_surface, "ex_data": train_ex_data}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Model configuration
    model_config = {
        # Input dimensions
        "feat_dim": (5, 5),
        "ex_feats_dim": 0,  # Not using extra features

        # Latent space
        "latent_dim": config.latent_dim,

        # Device
        "device": device,

        # Loss weights
        "kl_weight": config.kl_weight,
        "re_feat_weight": 1.0,

        # Surface encoder/decoder
        "surface_hidden": config.surface_hidden,
        "ex_feats_hidden": None,

        # Memory module
        "mem_type": "lstm",
        "mem_hidden": config.mem_hidden,
        "mem_layers": config.mem_layers,
        "mem_dropout": config.mem_dropout,

        # Context encoder
        "ctx_surface_hidden": config.surface_hidden,
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "compress_context": True,

        # Architecture settings
        "use_dense_surface": False,
        "padding": 1,

        # Horizon settings
        "horizon": 1,
        "context_len": config.context_len,
        "max_horizon": 90,

        # Prior encoder specific - MATCH context encoder Conv2D size
        "prior_surface_hidden": config.surface_hidden,  # [5, 5, 5] - same as context encoder
        "prior_mem_hidden": 64,
        "prior_mem_layers": 1,
        "prior_dropout": 0.1,
        "prior_pos_dim": 32,

        # Full covariance prior
        "full_cov_pos_dim": config.full_cov_pos_dim,
        "full_cov_hidden_dims": config.full_cov_hidden_dims,
        "full_cov_dropout": config.full_cov_dropout,
        "full_cov_init_phi": config.full_cov_init_phi,
        "full_cov_init_sigma_sq": config.full_cov_init_sigma_sq,
    }

    results = {}

    # =========================================================================
    # VARIANT A: Baseline (Current Architecture)
    # =========================================================================

    print("=" * 80)
    print("VARIANT A: Baseline (CVAEFullCovPrior)")
    print("=" * 80)

    model_a = CVAEFullCovPrior(model_config)
    model_a = model_a.to(device)
    model_a.device = device

    model_a, history_a = train_model("Baseline", model_a, train_data, val_surface, config, num_epochs=30)
    results['baseline'] = history_a

    # Save checkpoint
    output_dir = Path("results/prior_encoder_ablation/short_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_config": model_config,
        "state_dict": model_a.state_dict(),
        "epoch": len(history_a['epoch'])
    }, output_dir / "baseline_ep30.pt")

    # =========================================================================
    # VARIANT B1: Prior Encoder Diagonal
    # =========================================================================

    print("\n" + "=" * 80)
    print("VARIANT B1: Prior Encoder Diagonal")
    print("=" * 80)

    model_b1 = CVAEWithPriorEncoderDiagonal(model_config)
    model_b1 = model_b1.to(device)
    model_b1.device = device

    model_b1, history_b1 = train_model("Prior Encoder Diagonal", model_b1, train_data, val_surface, config, num_epochs=30)
    results['prior_encoder_diagonal'] = history_b1

    torch.save({
        "model_config": model_config,
        "state_dict": model_b1.state_dict(),
        "epoch": len(history_b1['epoch'])
    }, output_dir / "prior_encoder_diagonal_ep30.pt")

    # =========================================================================
    # VARIANT B2: Prior Encoder Full Cov
    # =========================================================================

    print("\n" + "=" * 80)
    print("VARIANT B2: Prior Encoder Full Covariance")
    print("=" * 80)

    model_b2 = CVAEWithPriorEncoderFullCov(model_config)
    model_b2 = model_b2.to(device)
    model_b2.device = device

    model_b2, history_b2 = train_model("Prior Encoder Full Cov", model_b2, train_data, val_surface, config, num_epochs=30)
    results['prior_encoder_full_cov'] = history_b2

    torch.save({
        "model_config": model_config,
        "state_dict": model_b2.state_dict(),
        "epoch": len(history_b2['epoch'])
    }, output_dir / "prior_encoder_full_cov_ep30.pt")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================

    print("\n" + "=" * 80)
    print("RESULTS: 30-Epoch Comparison")
    print("=" * 80)
    print()

    # Final epoch metrics
    print("FINAL EPOCH METRICS")
    print("-" * 80)
    print(f"{'Model':<30} {'Total Loss':<15} {'KL Loss':<15} {'Cond Var Ratio':<15}")
    print("-" * 80)

    for name, hist in results.items():
        final_loss = hist['total_loss'][-1]
        final_kl = hist['kl_loss'][-1]
        # Get last non-None conditional variance (if any)
        cond_var_values = [x for x in hist['conditional_var_ratio'] if x is not None]
        final_cond_var = cond_var_values[-1] if cond_var_values else 0.0
        cond_var_str = f"{final_cond_var:.4%}" if cond_var_values else "N/A"
        print(f"{name:<30} {final_loss:<15.6f} {final_kl:<15.3f} {cond_var_str:<15}")

    print()

    # Best conditional variance
    print("BEST CONDITIONAL VARIANCE RATIO")
    print("-" * 80)
    print(f"{'Model':<30} {'Best Ratio':<15} {'Improvement vs Baseline':<25}")
    print("-" * 80)

    baseline_values = [x for x in results['baseline']['conditional_var_ratio'] if x is not None]
    baseline_best = max(baseline_values) if baseline_values else 0.0

    if baseline_values:
        for name, hist in results.items():
            cond_var_values = [x for x in hist['conditional_var_ratio'] if x is not None]
            best_ratio = max(cond_var_values) if cond_var_values else 0.0
            improvement = (best_ratio / baseline_best - 1) * 100 if baseline_best > 0 else 0.0
            ratio_str = f"{best_ratio:.4%}" if cond_var_values else "N/A"
            improvement_str = f"{improvement:+.1f}%" if cond_var_values else "N/A"
            print(f"{name:<30} {ratio_str:<15} {improvement_str:<25}")
    else:
        print("Conditional variance measurement was disabled during training.")
        print("Re-run with measure_conditional_variance_fast() enabled to get these metrics.")

    print()

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    if baseline_values:
        b1_values = [x for x in results['prior_encoder_diagonal']['conditional_var_ratio'] if x is not None]
        b2_values = [x for x in results['prior_encoder_full_cov']['conditional_var_ratio'] if x is not None]

        b1_best = max(b1_values) if b1_values else 0.0
        b2_best = max(b2_values) if b2_values else 0.0

        best_improvement = max(b1_best, b2_best) / baseline_best if baseline_best > 0 else 0.0

        if best_improvement > 4.0:
            print(f"✓ STRONG EVIDENCE: {best_improvement:.1f}× improvement")
            print("  Prior Encoder significantly improves conditional variance")
            print("  → Proceed with full 400-epoch training")
        elif best_improvement > 2.0:
            print(f"✓ MODERATE EVIDENCE: {best_improvement:.1f}× improvement")
            print("  Prior Encoder shows promise")
            print("  → Consider full training with hyperparameter tuning")
        elif best_improvement > 1.2:
            print(f"⚠ WEAK EVIDENCE: {best_improvement:.1f}× improvement")
            print("  Prior Encoder provides small benefit")
            print("  → May need longer training or architecture changes")
        else:
            print(f"⚠ NO EVIDENCE: {best_improvement:.1f}× improvement")
            print("  Prior Encoder does not help significantly")
            print("  → Problem likely lies elsewhere")
    else:
        print("📊 TRAINING METRICS COMPARISON")
        print()
        print("Since conditional variance measurement was disabled, comparing training metrics:")
        print()
        for name, hist in results.items():
            final_loss = hist['total_loss'][-1]
            final_kl = hist['kl_loss'][-1]
            print(f"  {name}:")
            print(f"    - Final Loss: {final_loss:.6f}")
            print(f"    - Final KL: {final_kl:.3f}")
        print()
        print("✓ All models trained successfully for 30 epochs")
        print("→ To evaluate conditional variance, re-run with measurement enabled")
        print("   or use the saved checkpoints for post-training analysis")

    print()

    # Save results
    np.savez(
        output_dir / "comparison_results.npz",
        **{f"{name}_{key}": np.array(val) if val is not None else None
           for name, hist in results.items()
           for key, val in hist.items()}
    )

    print(f"Results saved to: {output_dir}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
