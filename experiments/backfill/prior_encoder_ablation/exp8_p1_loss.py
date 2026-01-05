"""
Experiment 8: P1 Loss Regularization to Increase Decoder Gain

BACKGROUND FROM exp7:
- Decoder gain = 1.2e-7 (near zero!)
- Even 5x prior sigma produces only 0.000003 output variance
- Prior modifications cannot fix this - bottleneck is the DECODER

ROOT CAUSE:
- VAE training: Decoder learned to minimize reconstruction loss
- Result: Decoder suppresses z variation to be "precise"
- z encodes "which specific outcome" not "uncertainty"

SOLUTION (exp8):
Add P1 loss that DIRECTLY maximizes output variance when z varies:

    p1_loss = -log(output_variance + eps)

This forces the decoder to have higher gain from z to output.

EXPECTED OUTCOMES:
- Decoder gain should increase from 1.2e-7 to meaningful levels
- P1 should improve as output variance increases
- P2 should improve as z→output mapping becomes stronger

SUCCESS CRITERIA:
- P1 > 0.5% (100x improvement over current 0.006%)
- Decoder gain > 1e-4 (1000x improvement)
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
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

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov

# =========================================================================
# KEY PARAMETERS
# =========================================================================
TRAINING_HORIZON = 5        # Multi-horizon for phi gradient
KL_WEIGHT = 0.00001         # Standard KL weight (not aggressive)
P1_WEIGHT = 0.1             # Weight for P1 loss (output variance maximization)
P1_NUM_SAMPLES = 10         # Number of z samples for P1 loss computation
P1_TARGET_VAR = 0.0002      # Target output variance (0.02 * total_var ~= 0.0002)


def compute_p1_loss(model, context_batch, num_samples=10, target_var=0.0002):
    """
    Compute P1 loss: Maximize output variance when z varies.

    For each context in batch:
    1. Sample num_samples z values from prior
    2. Decode each z
    3. Compute variance of outputs
    4. Loss = -log(variance + eps) to maximize variance

    Returns:
        p1_loss: Scalar loss to minimize (negative log variance)
        output_var: Mean output variance across batch (for monitoring)
    """
    device = model.device
    B, C = context_batch["surface"].shape[:2]
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    all_output_vars = []

    for b in range(B):
        # Get single context
        ctx_surface = context_batch["surface"][b:b+1]  # (1, C, 5, 5)
        ctx_feats = context_batch["ex_feats"][b:b+1]   # (1, C, 3)
        ctx_input = {"surface": ctx_surface, "ex_feats": ctx_feats}

        # Get prior parameters
        ctx_out = model.ctx_encoder(ctx_input)
        context_summary = ctx_out[:, -1, :]  # (1, latent_dim)
        mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

        # Sample multiple z values
        eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
        eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
        L = torch.linalg.cholesky(Sigma_p + eps * eye)

        samples = []
        for _ in range(num_samples):
            epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
            z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
            z = mu_p + z_centered

            # Decode
            if model.config.get("compress_context", True):
                ctx_embedding_dim = latent_dim
            else:
                ctx_embedding_dim = model.config["mem_hidden"]

            ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
            decoder_input = torch.cat([z, ctx_zeros], dim=-1)
            decoded = model.decoder(decoder_input)

            if isinstance(decoded, tuple):
                decoded = decoded[0]

            samples.append(decoded.squeeze())  # (5, 5)

        # Compute variance across samples
        samples_tensor = torch.stack(samples)  # (num_samples, 5, 5)
        output_var = samples_tensor.var(dim=0).mean()
        all_output_vars.append(output_var)

    # Mean variance across batch
    mean_output_var = torch.stack(all_output_vars).mean()

    # P1 loss: -log(var + eps) to maximize variance
    # We can also use a target-based loss: MSE(var, target_var) or hinge loss
    # Using log loss is more stable for very small variances
    p1_loss = -torch.log(mean_output_var + 1e-8)

    # Alternative: Hinge loss if we want to target a specific variance
    # p1_loss = F.relu(target_var - mean_output_var)

    return p1_loss, mean_output_var.item()


def train_step_with_p1_loss(model, batch, optimizer, p1_weight=0.1, p1_num_samples=10):
    """
    Training step with P1 loss regularization.

    Total loss = reconstruction + kl_weight * KL + p1_weight * P1_loss
    """
    from torch.amp import autocast
    import torch.nn as nn

    model.train()
    optimizer.zero_grad(set_to_none=True)

    surface = batch["surface"]
    B = surface.shape[0]
    T = surface.shape[1]
    H = model.horizon
    C = T - H

    surface_real = surface[:, C:, :, :].to(model.device)

    ex_feats = batch["ex_feats"]
    ex_feats_real = ex_feats[:, C:, :].to(model.device)

    # Move batch to device
    batch_device = {
        "surface": surface.to(model.device),
        "ex_feats": ex_feats.to(model.device)
    }

    # Standard VAE forward pass with mixed precision
    with autocast('cuda', dtype=torch.bfloat16):
        surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = model.forward(batch_device)

        # Reconstruction loss
        re_surface = nn.functional.mse_loss(surface_reconstruction, surface_real)
        if model.config["ex_loss_on_ret_only"]:
            ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
            ex_feats_real_loss = ex_feats_real[:, :, :1]
        else:
            ex_feats_real_loss = ex_feats_real
        re_ex_feats = nn.functional.mse_loss(ex_feats_reconstruction, ex_feats_real_loss)
        reconstruction_error = re_surface + model.config["re_feat_weight"] * re_ex_feats

        # KL loss with full covariance prior
        from vae.full_covariance_prior import kl_divergence_full_covariance

        ctx_surface = batch_device["surface"][:, :C, :, :]
        ctx_encoder_input = {"surface": ctx_surface, "ex_feats": batch_device["ex_feats"][:, :C, :]}
        ctx_embedding = model.ctx_encoder(ctx_encoder_input)
        context_summary = ctx_embedding[:, -1, :]
        mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=H)

        kl_loss = kl_divergence_full_covariance(
            z_mean[:, C:, :],
            z_log_var[:, C:, :],
            mu_p,
            Sigma_p
        )

        base_loss = reconstruction_error + model.kl_weight * kl_loss

        # ============================================================================
        # P1 LOSS: Maximize output variance when z varies
        # ============================================================================
        latent_dim = model.config["latent_dim"]
        dtype = next(model.parameters()).dtype

        # Sample from prior for a subset of batch (to save memory)
        sample_batch_size = min(4, B)
        all_output_vars = []

        for b in range(sample_batch_size):
            # Get prior for this context
            ctx_sum = context_summary[b:b+1]  # (1, latent_dim)
            mu_p_single, Sigma_p_single = model.full_cov_prior.get_prior_params(ctx_sum, horizon=1)

            # Sample multiple z values
            eps = 1e-4
            eye = torch.eye(Sigma_p_single.shape[0], device=model.device, dtype=dtype)
            L = torch.linalg.cholesky(Sigma_p_single + eps * eye)

            samples = []
            for _ in range(p1_num_samples):
                epsilon = torch.randn(1, 1, latent_dim, device=model.device, dtype=dtype)
                z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                z_sample = mu_p_single + z_centered

                # Decode
                if model.config.get("compress_context", True):
                    ctx_embedding_dim = latent_dim
                else:
                    ctx_embedding_dim = model.config["mem_hidden"]

                ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=model.device, dtype=dtype)
                decoder_input = torch.cat([z_sample, ctx_zeros], dim=-1)
                decoded = model.decoder(decoder_input)

                if isinstance(decoded, tuple):
                    decoded = decoded[0]

                samples.append(decoded.squeeze())

            # Compute variance across samples
            samples_tensor = torch.stack(samples)
            output_var = samples_tensor.var(dim=0).mean()
            all_output_vars.append(output_var)

        mean_output_var = torch.stack(all_output_vars).mean()

        # P1 loss: -log(var + eps) to maximize variance
        p1_loss = -torch.log(mean_output_var + 1e-8)

        total_loss = base_loss + p1_weight * p1_loss

    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'loss': total_loss,
        'reconstruction_loss': re_surface,
        'kl_loss': kl_loss,
        'p1_loss': p1_loss,
        'output_var': mean_output_var.item()
    }


def measure_decoder_gain(model, val_surface, num_contexts=50, num_samples=50):
    """
    Measure decoder gain: output_variance / z_variance

    This is the key metric we want to improve.
    """
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    model.eval()

    scales = [1.0, 2.0, 5.0]
    results = {scale: {'z_var': [], 'out_var': []} for scale in scales}

    with torch.no_grad():
        for i in range(num_contexts):
            if i + C + 1 > len(val_surface):
                break

            context = val_surface[i:i+C].unsqueeze(0).to(device).to(dtype)
            ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
            ctx_input = {"surface": context, "ex_feats": ctx_feats}

            # Get prior parameters
            ctx_out = model.ctx_encoder(ctx_input)
            context_summary = ctx_out[:, -1, :]
            mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

            # Get sigma from covariance diagonal
            sigma_p = torch.sqrt(torch.diag(Sigma_p)).unsqueeze(0).unsqueeze(0)

            for scale in scales:
                samples = []
                z_samples = []

                for _ in range(num_samples):
                    epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                    z = mu_p + scale * sigma_p * epsilon
                    z_samples.append(z.cpu())

                    # Decode
                    if model.config.get("compress_context", True):
                        ctx_embedding_dim = latent_dim
                    else:
                        ctx_embedding_dim = model.config["mem_hidden"]

                    ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
                    decoder_input = torch.cat([z, ctx_zeros], dim=-1)
                    decoded = model.decoder(decoder_input)

                    if isinstance(decoded, tuple):
                        decoded = decoded[0]

                    samples.append(decoded.cpu())

                samples_tensor = torch.stack(samples).squeeze()
                z_tensor = torch.stack(z_samples).squeeze()

                output_var = samples_tensor.var(dim=0).mean().item()
                z_var = z_tensor.var(dim=0).mean().item()

                results[scale]['z_var'].append(z_var)
                results[scale]['out_var'].append(output_var)

    # Compute mean gain at scale=1.0
    mean_z_var = np.mean(results[1.0]['z_var'])
    mean_out_var = np.mean(results[1.0]['out_var'])
    gain = mean_out_var / mean_z_var if mean_z_var > 0 else 0

    return gain, mean_z_var, mean_out_var, results


def measure_conditional_variance(model, val_surface, num_contexts=100, num_samples=50):
    """Measure P1 metric: E[Var(X|C)] / Var(X)"""
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

            context = val_surface[i:i+C].unsqueeze(0).to(device).to(dtype)
            ctx_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
            ctx_input = {"surface": context, "ex_feats": ctx_feats}

            samples = []
            for _ in range(num_samples):
                ctx_out = model.ctx_encoder(ctx_input)
                context_summary = ctx_out[:, -1, :]
                mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

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


def train_model(model, train_loader, val_surface, config, num_epochs=200,
                p1_weight=0.1, p1_num_samples=10):
    """Train model with P1 loss regularization."""

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        'epoch': [],
        'recon_loss': [],
        'kl_loss': [],
        'p1_loss': [],
        'total_loss': [],
        'output_var': [],
        'decoder_gain': [],
        'conditional_var_ratio': [],
        'epoch_time': [],
        'phi': [],
        'sigma_sq': [],
    }

    print(f"\nTraining with P1 loss regularization...")
    print(f"P1 weight: {p1_weight}")
    print(f"P1 samples per context: {p1_num_samples}")
    print()

    # Initial measurements
    print("Initial decoder gain measurement...")
    gain, z_var, out_var, _ = measure_decoder_gain(model, val_surface, num_contexts=30, num_samples=30)
    print(f"  Initial decoder gain: {gain:.2e}")
    print(f"  Z variance: {z_var:.6f}, Output variance: {out_var:.6f}")
    print()

    best_gain = gain
    best_p1 = 0.0

    kl_anneal_epochs = 50
    target_kl_weight = config.kl_weight

    for epoch in range(num_epochs):
        start_time = time.time()

        # KL annealing
        if epoch < kl_anneal_epochs:
            effective_kl_weight = target_kl_weight * ((epoch + 1) / kl_anneal_epochs)
            model.kl_weight = effective_kl_weight
        else:
            model.kl_weight = target_kl_weight

        model.train()
        epoch_losses = {'recon': [], 'kl': [], 'p1': [], 'total': [], 'out_var': []}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_surface, batch_ex = batch
            batch_surface = batch_surface.to(model.device)
            batch_ex = batch_ex.to(model.device)

            x = {
                "surface": batch_surface,
                "ex_feats": batch_ex
            }

            loss_dict = train_step_with_p1_loss(
                model, x, optimizer,
                p1_weight=p1_weight,
                p1_num_samples=p1_num_samples
            )

            epoch_losses['recon'].append(loss_dict['reconstruction_loss'].item())
            epoch_losses['kl'].append(loss_dict['kl_loss'].item())
            epoch_losses['p1'].append(loss_dict['p1_loss'].item())
            epoch_losses['total'].append(loss_dict['loss'].item())
            epoch_losses['out_var'].append(loss_dict['output_var'])

        avg_recon = np.mean(epoch_losses['recon'])
        avg_kl = np.mean(epoch_losses['kl'])
        avg_p1 = np.mean(epoch_losses['p1'])
        avg_total = np.mean(epoch_losses['total'])
        avg_out_var = np.mean(epoch_losses['out_var'])

        # Measure metrics every 20 epochs
        gain = None
        p1_ratio = None

        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f"\n  Measuring decoder gain and P1...")
            gain, z_var, out_var, _ = measure_decoder_gain(model, val_surface, num_contexts=50, num_samples=30)
            p1_ratio = measure_conditional_variance(model, val_surface, num_contexts=50, num_samples=30)

            if gain > best_gain:
                best_gain = gain
            if p1_ratio > best_p1:
                best_p1 = p1_ratio

        epoch_time = time.time() - start_time

        # Get variance parameters
        phi = model.full_cov_prior.get_phi().item()
        sigma_sq = model.full_cov_prior.get_sigma_sq().item()

        # Record history
        history['epoch'].append(epoch)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        history['p1_loss'].append(avg_p1)
        history['total_loss'].append(avg_total)
        history['output_var'].append(avg_out_var)
        history['decoder_gain'].append(gain)
        history['conditional_var_ratio'].append(p1_ratio)
        history['epoch_time'].append(epoch_time)
        history['phi'].append(phi)
        history['sigma_sq'].append(sigma_sq)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_total:.4f}, Recon={avg_recon:.6f}, KL={avg_kl:.3f}, P1_loss={avg_p1:.2f}", end="")
        print(f", OutVar={avg_out_var:.2e}, phi={phi:.4f}", end="")
        if gain is not None:
            print(f", Gain={gain:.2e}, P1={p1_ratio:.4%}", end="")
        print(f" ({epoch_time:.1f}s)")

        # Early stopping on success
        if p1_ratio is not None and p1_ratio > 0.02:
            print(f"\n[EARLY STOP] P1 = {p1_ratio:.4%} > 2% target!")
            break

    print(f"\n[COMPLETE] Best decoder gain: {best_gain:.2e}, Best P1: {best_p1:.4%}")

    return model, history


def main():
    print("=" * 80)
    print("EXPERIMENT 8: P1 Loss Regularization")
    print("=" * 80)
    print()
    print("ROOT CAUSE (from exp7):")
    print("  Decoder gain = 1.2e-7 (near zero!)")
    print("  Prior modifications cannot fix this - bottleneck is the DECODER")
    print()
    print("SOLUTION:")
    print("  Add P1 loss: -log(output_variance + eps)")
    print("  This forces decoder to increase gain from z to output")
    print()
    print(f"PARAMETERS:")
    print(f"  P1_WEIGHT = {P1_WEIGHT}")
    print(f"  P1_NUM_SAMPLES = {P1_NUM_SAMPLES}")
    print(f"  TRAINING_HORIZON = {TRAINING_HORIZON}")
    print(f"  KL_WEIGHT = {KL_WEIGHT}")
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

    # Create sequences
    C = config.context_len
    H = TRAINING_HORIZON
    seq_len = C + H

    print(f"Sequence length: {seq_len} (context={C} + horizon={H})")
    print()

    train_sequences = []
    train_ex_sequences = []
    for i in range(len(train_surface) - seq_len):
        train_sequences.append(train_surface[i:i+seq_len])
        train_ex_sequences.append(train_ex_data[i:i+seq_len])

    train_sequences = torch.stack(train_sequences)
    train_ex_sequences = torch.stack(train_ex_sequences)

    print(f"Training sequences: {train_sequences.shape}")

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

    # Model config
    model_config = {
        "feat_dim": (5, 5),
        "ex_feats_dim": 3,
        "ex_feats_hidden": [16, 8],
        "latent_dim": config.latent_dim,
        "device": device,
        "kl_weight": KL_WEIGHT,
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
        "horizon": H,
        "context_len": config.context_len,
        "max_horizon": 90,
        "full_cov_pos_dim": config.full_cov_pos_dim,
        "full_cov_hidden_dims": config.full_cov_hidden_dims,
        "full_cov_dropout": config.full_cov_dropout,
        "full_cov_init_phi": config.full_cov_init_phi,
        "full_cov_init_sigma_sq": config.full_cov_init_sigma_sq,
    }

    output_dir = Path("results/prior_encoder_ablation/exp8_p1_loss")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # TRAIN MODEL WITH P1 LOSS
    # =========================================================================

    print("=" * 80)
    print("Training CVAEFullCovPrior with P1 Loss Regularization")
    print("=" * 80)

    model = CVAEFullCovPrior(model_config)
    model = model.to(device)

    print("Compiling model with torch.compile()...")
    model = torch.compile(model, mode="default", dynamic=True)

    model.device = device
    model.config = model_config

    model, history = train_model(
        model, train_loader, val_surface, config,
        num_epochs=200,
        p1_weight=P1_WEIGHT,
        p1_num_samples=P1_NUM_SAMPLES
    )

    # Save model
    torch.save({
        "model_config": model_config,
        "state_dict": model.state_dict(),
        "epoch": len(history['epoch']),
        "p1_weight": P1_WEIGHT,
        "p1_num_samples": P1_NUM_SAMPLES
    }, output_dir / "model_ep200.pt")

    # Save history
    np.savez(output_dir / "history.npz", **history)

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Measure final decoder gain
    print("\nFinal decoder gain measurement...")
    gain, z_var, out_var, results = measure_decoder_gain(model, val_surface, num_contexts=100, num_samples=50)

    print(f"\nDecoder Gain Results:")
    print("-" * 60)
    print(f"| Scale | Z Variance | Output Variance | Gain |")
    print(f"|-------|------------|-----------------|------|")

    for scale in [1.0, 2.0, 5.0]:
        z_v = np.mean(results[scale]['z_var'])
        out_v = np.mean(results[scale]['out_var'])
        g = out_v / z_v if z_v > 0 else 0
        print(f"| {scale}x    | {z_v:.6f}   | {out_v:.6f}        | {g:.2e} |")

    print()
    print(f"Effective decoder gain (1x scale): {gain:.2e}")

    # Measure final P1
    print("\nFinal P1 measurement...")
    p1_ratio = measure_conditional_variance(model, val_surface, num_contexts=100, num_samples=50)
    print(f"P1 (conditional variance ratio): {p1_ratio:.4%}")

    # Get final variance parameters
    phi = model.full_cov_prior.get_phi().item()
    sigma_sq = model.full_cov_prior.get_sigma_sq().item()

    print(f"\nVariance parameters:")
    print(f"  phi = {phi:.4f}")
    print(f"  sigma_sq = {sigma_sq:.4f}")

    # =========================================================================
    # COMPARISON WITH BASELINE
    # =========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE (exp6/exp7)")
    print("=" * 80)
    print()
    print(f"| Metric                | Baseline (exp7) | With P1 Loss | Improvement |")
    print(f"|-----------------------|-----------------|--------------|-------------|")
    print(f"| Decoder Gain          | 1.2e-7          | {gain:.2e}    | {gain/1.2e-7:.0f}x         |")
    print(f"| P1 (cond var ratio)   | 0.006%          | {p1_ratio:.4%}    | {p1_ratio/0.00006:.1f}x         |")
    print(f"| Output Var (1x scale) | 0.000000        | {out_var:.6f}   | -           |")
    print()

    # Success check
    print("=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)
    print()

    gain_success = "[OK]" if gain > 1e-4 else "[X]"
    p1_success = "[OK]" if p1_ratio > 0.005 else "[X]"

    print(f"{gain_success} Decoder gain > 1e-4: {gain:.2e}")
    print(f"{p1_success} P1 > 0.5%: {p1_ratio:.4%}")
    print()

    if gain > 1e-4 and p1_ratio > 0.005:
        print("SUCCESS! P1 loss regularization improved decoder gain.")
    else:
        print("PARTIAL SUCCESS - Further tuning of P1 weight may be needed.")

    print(f"\nResults saved to: {output_dir}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
