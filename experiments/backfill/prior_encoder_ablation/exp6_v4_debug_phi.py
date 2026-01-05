"""
Experiment 6 v4: Debug Why φ (phi) Is Not Learning

Goal: Understand why φ stayed at exactly 0.5000 for 200 epochs in exp6_v3

Key debugging:
1. Log gradients for raw_phi and log_sigma_sq after backward()
2. Verify the KL computation includes φ contribution
3. Check if gradients flow through Cholesky decomposition

Only runs B2 (Prior Encoder Full Cov) for faster iteration.
Expected runtime: ~45 minutes (50 epochs)
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, TensorDataset

# Enable TF32 for faster CUDA training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_prior_encoder import CVAEWithPriorEncoderFullCov
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def measure_conditional_variance_fast(model, val_surface, num_contexts=50, num_samples=30):
    """Measure E[Var(X|C)] / Var(X) (Problem 1 metric)."""
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    model.eval()

    all_samples_per_context = []

    with torch.no_grad():
        for i in range(num_contexts):
            if i + C + 1 > len(val_surface):
                break

            context = val_surface[i:i+C].unsqueeze(0).to(device)

            samples = []
            for _ in range(num_samples):
                ctx_zeros_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
                ctx_input = {"surface": context, "ex_feats": ctx_zeros_feats}

                if hasattr(model, 'prior_encoder'):
                    mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=1)
                    eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                    eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                    L = torch.linalg.cholesky(Sigma_p + eps * eye)
                    epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                    z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                    z = mu_p + z_centered

                if model.config.get("compress_context", True):
                    ctx_embedding_dim = latent_dim
                else:
                    ctx_embedding_dim = model.config["mem_hidden"]

                ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
                decoder_input = torch.cat([z, ctx_zeros], dim=-1)
                decoded = model.decoder(decoder_input)

                if isinstance(decoded, tuple):
                    decoded = decoded[0]

                if decoded.dim() == 4:
                    decoded = decoded.squeeze(1)

                samples.append(decoded.cpu())

            samples = torch.stack(samples, dim=0).squeeze()
            all_samples_per_context.append(samples)

    all_samples = torch.stack(all_samples_per_context).numpy()
    var_given_context = np.var(all_samples, axis=1)
    expected_conditional_var = np.mean(var_given_context)

    all_gt = val_surface[C:C+num_contexts].numpy()
    total_var = np.var(all_gt.reshape(-1, 5, 5), axis=0).mean()

    ratio = expected_conditional_var / total_var

    return ratio


def train_model_with_gradient_debug(model_name, model, train_loader, val_surface, config, num_epochs=50):
    """Train with detailed gradient logging to debug φ learning."""

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        'epoch': [],
        'total_loss': [],
        'kl_loss': [],
        'conditional_var_ratio': [],
        'phi': [],
        'sigma_sq': [],
        'phi_grad': [],       # NEW: Track gradients
        'sigma_sq_grad': [],  # NEW: Track gradients
        'kl_loss_value': []   # NEW: Actual KL loss value
    }

    print(f"\nTraining {model_name} for {num_epochs} epochs with GRADIENT DEBUGGING...")
    print(f"Batch size: {config.batch_size}")
    print()

    best_cond_var_ratio = 0.0
    kl_anneal_epochs = 50

    for epoch in range(num_epochs):
        start_time = time.time()

        # KL Annealing
        if epoch < kl_anneal_epochs:
            effective_kl_weight = config.kl_weight * ((epoch + 1) / kl_anneal_epochs)
            model.kl_weight = effective_kl_weight
        else:
            model.kl_weight = config.kl_weight
            effective_kl_weight = config.kl_weight

        model.train()
        epoch_losses = {'kl': [], 'total': []}

        # Track gradients for one batch per epoch
        batch_phi_grads = []
        batch_sigma_grads = []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            batch_surface, batch_ex = batch
            batch_surface = batch_surface.to(model.device)
            batch_ex = batch_ex.to(model.device)

            x = {
                "surface": batch_surface,
                "ex_feats": batch_ex
            }

            # Manual forward/backward to capture gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass - manually replicate train_step logic
            surface = x["surface"]
            if len(surface.shape) == 3:
                surface = surface.unsqueeze(0)
            T = surface.shape[1]
            C = T - model.horizon
            surface_real = surface[:, C:, :, :].to(model.device)

            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :].to(model.device)

            x_device = {
                "surface": surface.to(model.device),
                "ex_feats": ex_feats.to(model.device)
            }

            # Forward pass through main model
            surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = model.forward(x_device)

            # Reconstruction loss
            re_surface = torch.nn.functional.mse_loss(surface_reconstruction, surface_real)
            reconstruction_error = re_surface

            # Prior encoder processes RAW context
            ctx_input = {
                "surface": x_device["surface"][:, :C, :, :],
                "ex_feats": x_device["ex_feats"][:, :C, :]
            }

            # Get prior parameters from prior encoder
            from vae.cvae_prior_encoder import kl_divergence_full_covariance
            mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=model.horizon)

            # Compute KL(diagonal posterior || full cov prior)
            kl_loss = kl_divergence_full_covariance(
                z_mean[:, C:, :],
                z_log_var[:, C:, :],
                mu_p,
                Sigma_p
            )

            total_loss = reconstruction_error + model.kl_weight * kl_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # === GRADIENT DEBUGGING ===
            # Check gradients for phi and sigma_sq BEFORE optimizer step
            # NOTE: The parameter is log_phi (logit of phi), not raw_phi
            if hasattr(model, 'prior_encoder') and hasattr(model.prior_encoder, 'log_phi'):
                phi_grad = model.prior_encoder.log_phi.grad
                sigma_grad = model.prior_encoder.log_sigma_sq.grad

                if phi_grad is not None:
                    batch_phi_grads.append(phi_grad.item())
                else:
                    batch_phi_grads.append(None)

                if sigma_grad is not None:
                    batch_sigma_grads.append(sigma_grad.item())
                else:
                    batch_sigma_grads.append(None)

            optimizer.step()

            epoch_losses['kl'].append(kl_loss.item())
            epoch_losses['total'].append(total_loss.item())

        avg_kl = np.mean(epoch_losses['kl'])
        avg_total = np.mean(epoch_losses['total'])

        # Compute average gradients for this epoch
        valid_phi_grads = [g for g in batch_phi_grads if g is not None]
        valid_sigma_grads = [g for g in batch_sigma_grads if g is not None]

        avg_phi_grad = np.mean(valid_phi_grads) if valid_phi_grads else None
        avg_sigma_grad = np.mean(valid_sigma_grads) if valid_sigma_grads else None

        # Get current parameter values
        phi = model.prior_encoder.get_phi().item()
        sigma_sq = model.prior_encoder.get_sigma_sq().item()

        # Measure P1 every 10 epochs
        cond_var_ratio = None
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            cond_var_ratio = measure_conditional_variance_fast(model, val_surface)
            if cond_var_ratio > best_cond_var_ratio:
                best_cond_var_ratio = cond_var_ratio

        epoch_time = time.time() - start_time

        # Record history
        history['epoch'].append(epoch)
        history['total_loss'].append(avg_total)
        history['kl_loss'].append(avg_kl)
        history['conditional_var_ratio'].append(cond_var_ratio)
        history['phi'].append(phi)
        history['sigma_sq'].append(sigma_sq)
        history['phi_grad'].append(avg_phi_grad)
        history['sigma_sq_grad'].append(avg_sigma_grad)
        history['kl_loss_value'].append(avg_kl)

        # Print progress with gradient info
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_total:.6f}, KL={avg_kl:.3f}")
        print(f"  PARAMS: φ={phi:.6f}, σ²={sigma_sq:.6f}")

        if avg_phi_grad is not None:
            print(f"  GRADS:  φ_grad={avg_phi_grad:.2e}, σ²_grad={avg_sigma_grad:.2e}")

            # Highlight if gradients are zero or very small
            if abs(avg_phi_grad) < 1e-10:
                print(f"  ⚠️  WARNING: φ gradient is essentially ZERO!")
            elif abs(avg_phi_grad) < 1e-6:
                print(f"  ⚠️  WARNING: φ gradient is very small ({avg_phi_grad:.2e})")
        else:
            print(f"  ⚠️  ERROR: φ gradient is None (not in computation graph!)")

        if cond_var_ratio is not None:
            print(f"  P1={cond_var_ratio:.4%}")

        print(f"  ({epoch_time:.1f}s)")
        print()

    print(f"\n✓ Training complete. Best P1: {best_cond_var_ratio:.4%}")

    return model, history


def analyze_gradient_history(history):
    """Analyze gradient patterns to diagnose why φ isn't learning."""

    print("\n" + "=" * 80)
    print("GRADIENT ANALYSIS")
    print("=" * 80)

    phi_grads = [g for g in history['phi_grad'] if g is not None]
    sigma_grads = [g for g in history['sigma_sq_grad'] if g is not None]

    if not phi_grads:
        print("\n❌ CRITICAL: No φ gradients recorded!")
        print("   φ is NOT connected to the computation graph.")
        print("   → Check if φ is used in the forward pass and KL computation.")
        return

    print("\nφ (phi) Gradient Statistics:")
    print(f"  Mean:    {np.mean(phi_grads):.2e}")
    print(f"  Std:     {np.std(phi_grads):.2e}")
    print(f"  Min:     {np.min(phi_grads):.2e}")
    print(f"  Max:     {np.max(phi_grads):.2e}")
    print(f"  # Zero:  {sum(abs(g) < 1e-15 for g in phi_grads)} / {len(phi_grads)}")

    print("\nσ² (sigma_sq) Gradient Statistics:")
    print(f"  Mean:    {np.mean(sigma_grads):.2e}")
    print(f"  Std:     {np.std(sigma_grads):.2e}")
    print(f"  Min:     {np.min(sigma_grads):.2e}")
    print(f"  Max:     {np.max(sigma_grads):.2e}")
    print(f"  # Zero:  {sum(abs(g) < 1e-15 for g in sigma_grads)} / {len(sigma_grads)}")

    # Parameter changes
    phi_values = history['phi']
    sigma_values = history['sigma_sq']

    print("\nParameter Changes:")
    print(f"  φ:   {phi_values[0]:.6f} → {phi_values[-1]:.6f} (Δ={phi_values[-1]-phi_values[0]:.6f})")
    print(f"  σ²:  {sigma_values[0]:.6f} → {sigma_values[-1]:.6f} (Δ={sigma_values[-1]-sigma_values[0]:.6f})")

    # Diagnosis
    print("\n" + "-" * 80)
    print("DIAGNOSIS:")

    if all(abs(g) < 1e-10 for g in phi_grads):
        print("❌ φ gradients are essentially zero throughout training.")
        print("   Possible causes:")
        print("   1. KL weight is too small (current: 1e-5)")
        print("   2. φ is at a local minimum")
        print("   3. Gradient is disconnected through Cholesky")
        print()
        print("   → TRY: Increase KL weight to 1e-4 or 1e-3")

    elif abs(phi_values[-1] - phi_values[0]) < 0.001:
        print("❌ φ has non-zero gradients but didn't move.")
        print("   Possible causes:")
        print("   1. Learning rate is too small for φ")
        print("   2. Gradients cancel out over batches")
        print()
        print("   → TRY: Separate learning rate for prior parameters")

    else:
        print("✓ φ is learning (changed during training)")

    if np.mean(np.abs(sigma_grads)) > np.mean(np.abs(phi_grads)) * 100:
        print("\n📊 σ² gradients are ~100x larger than φ gradients")
        print("   This explains why σ² moves but φ doesn't")


def main():
    print("=" * 80)
    print("EXPERIMENT 6 v4: Debug Why φ (phi) Is Not Learning")
    print("=" * 80)
    print()
    print("Goal: Understand why φ stayed at 0.5000 for 200 epochs in exp6_v3")
    print()

    config = BackfillContext60ConfigV4FullCov

    # Load data
    print("Loading data...")
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)

    surface = torch.tensor(data["surface"], dtype=torch.float32)
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

    # Create DataLoader
    C = config.context_len
    seq_len = C + 1

    train_sequences = []
    train_ex_sequences = []
    for i in range(len(train_surface) - seq_len):
        train_sequences.append(train_surface[i:i+seq_len])
        train_ex_sequences.append(train_ex_data[i:i+seq_len])

    train_sequences = torch.stack(train_sequences)
    train_ex_sequences = torch.stack(train_ex_sequences)

    dataset = TensorDataset(train_sequences, train_ex_sequences)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Model configuration
    model_config = {
        "feat_dim": (5, 5),
        "ex_feats_dim": 3,
        "ex_feats_hidden": [16, 8],
        "latent_dim": config.latent_dim,
        "device": device,
        "kl_weight": config.kl_weight,
        "re_feat_weight": 1.0,
        "surface_hidden": config.surface_hidden,
        "mem_type": "lstm",
        "mem_hidden": config.mem_hidden,
        "mem_layers": config.mem_layers,
        "mem_dropout": config.mem_dropout,
        "ctx_surface_hidden": config.surface_hidden,
        "ctx_ex_feats_hidden": [16, 8],
        "interaction_layers": 2,
        "compress_context": True,
        "use_dense_surface": False,
        "padding": 1,
        "ex_feats_loss_type": "l2",
        "ex_loss_on_ret_only": True,
        "horizon": 1,
        "context_len": config.context_len,
        "max_horizon": 90,
        "prior_surface_hidden": config.surface_hidden,
        "prior_mem_hidden": 64,
        "prior_mem_layers": 1,
        "prior_dropout": 0.1,
        "prior_pos_dim": 32,
        "full_cov_pos_dim": config.full_cov_pos_dim,
        "full_cov_hidden_dims": config.full_cov_hidden_dims,
        "full_cov_dropout": config.full_cov_dropout,
        "full_cov_init_phi": config.full_cov_init_phi,
        "full_cov_init_sigma_sq": config.full_cov_init_sigma_sq,
    }

    # =========================================================================
    # Only run B2 (Prior Encoder Full Cov) for debugging
    # =========================================================================

    print("=" * 80)
    print("Running B2: Prior Encoder Full Cov (with gradient debugging)")
    print("=" * 80)

    model = CVAEWithPriorEncoderFullCov(model_config)
    model = model.to(device)

    # DON'T compile - we need to access gradients
    # model = torch.compile(model, mode="default", dynamic=True)

    model.device = device
    model.config = model_config

    # Print initial state
    print(f"\nInitial φ: {model.prior_encoder.get_phi().item():.6f}")
    print(f"Initial σ²: {model.prior_encoder.get_sigma_sq().item():.6f}")
    print(f"KL weight: {config.kl_weight}")
    print()

    model, history = train_model_with_gradient_debug(
        "Prior Encoder Full Cov",
        model,
        train_loader,
        val_surface,
        config,
        num_epochs=50
    )

    # Analyze gradients
    analyze_gradient_history(history)

    # Save results
    output_dir = Path("results/prior_encoder_ablation/exp6_v4_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "gradient_debug_history.npz",
        **{k: np.array(v) if isinstance(v, list) else v for k, v in history.items()}
    )

    torch.save({
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "history": history
    }, output_dir / "model_checkpoint.pt")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
