"""
Experiment 9c: Stable P1 Loss with Clamped CV

BACKGROUND FROM exp9b:
- Phase 1 (marginal-only) worked perfectly: Mean ratio 99.4%, Var ratio 93.3%
- Phase 2 (P1 added) caused runaway: CV exploded to 1000+, Mean → 0

ROOT CAUSE:
- The -log(cv) loss rewards unbounded CV increases
- Model pushes variance to infinity and mean to 0 to maximize CV
- Marginal constraints get overwhelmed by P1 loss gradient

SOLUTION (exp9c):
1. CLAMPED CV: Clamp CV to max of 1.0 to prevent runaway
   - cv_clamped = min(cv, 1.0)
   - Loss = -log(cv_clamped + 1e-8)
   - Once CV reaches 1.0, loss saturates at -log(1) = 0

2. LOWER P1 WEIGHT: 0.01 instead of 0.1 (10x reduction)

3. HIGHER MARGINAL WEIGHTS IN PHASE 2: 10x increase during P1 ramp
   - This prevents marginal degradation when P1 is active

4. SLOWER P1 RAMP: Ramp over full 100 epochs instead of 50

SUCCESS CRITERIA:
- P1 > 2% (conditional variance ratio)
- Gen Mean / GT Mean: 90-110%
- Gen Var / GT Var: 80-120%
- 90% CI Coverage > 85%
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
# KEY PARAMETERS (exp9c: Fixed for stability)
# =========================================================================
TRAINING_HORIZON = 5        # Multi-horizon for phi gradient
KL_WEIGHT = 0.00001         # Standard KL weight
P1_WEIGHT = 0.01            # REDUCED: 0.1 → 0.01 (10x lower)
MEAN_WEIGHT = 1.0           # Weight for marginal mean constraint (Phase 1)
VAR_WEIGHT = 1.0            # Weight for marginal variance constraint (Phase 1)
P1_NUM_SAMPLES = 10         # Number of z samples for P1 loss

# Phase 2: Increase marginal weights to counteract P1 loss
PHASE2_MEAN_WEIGHT = 10.0   # NEW: 10x higher during P1 phase
PHASE2_VAR_WEIGHT = 10.0    # NEW: 10x higher during P1 phase

# CV clamping to prevent runaway
CV_MAX = 1.0                # NEW: Clamp CV to this maximum

# Ground truth statistics (from exp8_verify_marginal.py)
GT_MEAN = 0.193             # Volatility surface mean
GT_VAR = 0.00353            # Volatility surface variance

# Training schedule
PHASE1_EPOCHS = 100         # Establish marginal (no P1)
PHASE2_EPOCHS = 100         # Add P1 while maintaining marginal
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS


def get_scheduled_weights(epoch):
    """
    Scheduled weights for P1 loss and marginal constraints.

    Phase 1 (0-100): p1_weight = 0 (establish marginal)
    Phase 2 (100-200): p1_weight ramps from 0 to P1_WEIGHT over full 100 epochs

    Returns: (p1_weight, mean_weight, var_weight)
    """
    if epoch < PHASE1_EPOCHS:
        return 0.0, MEAN_WEIGHT, VAR_WEIGHT
    else:
        # Linear ramp from 0 to P1_WEIGHT over FULL PHASE2_EPOCHS (slower ramp)
        progress = (epoch - PHASE1_EPOCHS) / PHASE2_EPOCHS
        p1_w = min(P1_WEIGHT, P1_WEIGHT * progress)
        # Use higher marginal weights during Phase 2
        return p1_w, PHASE2_MEAN_WEIGHT, PHASE2_VAR_WEIGHT


def train_step_with_marginal_constraints(model, batch, optimizer, epoch,
                                          p1_weight_base=0.1,
                                          mean_weight=1.0,
                                          var_weight=1.0,
                                          p1_num_samples=10):
    """
    Training step with scale-normalized P1 loss and marginal constraints.

    Key changes from exp8:
    1. Scale-normalized P1: cv = std / |mean| instead of absolute variance
    2. Marginal constraints: force output mean and variance to match GT
    3. Scheduled P1 weight: start with 0, ramp up after establishing marginal
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

    # Get scheduled weights (P1 weight and marginal weights)
    p1_weight, current_mean_weight, current_var_weight = get_scheduled_weights(epoch)

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
        # SCALE-NORMALIZED P1 LOSS + MARGINAL CONSTRAINTS
        # ============================================================================
        latent_dim = model.config["latent_dim"]
        dtype = next(model.parameters()).dtype

        # Sample from prior for a subset of batch
        sample_batch_size = min(4, B)
        all_outputs = []

        for b in range(sample_batch_size):
            # Get prior for this context
            ctx_sum = context_summary[b:b+1]
            mu_p_single, Sigma_p_single = model.full_cov_prior.get_prior_params(ctx_sum, horizon=1)

            # Sample multiple z values
            eps = 1e-4
            eye = torch.eye(Sigma_p_single.shape[0], device=model.device, dtype=dtype)
            L = torch.linalg.cholesky(Sigma_p_single + eps * eye)

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

                all_outputs.append(decoded.squeeze())

        # Stack all outputs: (sample_batch_size * p1_num_samples, 5, 5)
        outputs_tensor = torch.stack(all_outputs)

        # Compute output statistics
        output_mean = outputs_tensor.mean()
        output_var = outputs_tensor.var()
        output_std = torch.sqrt(output_var + 1e-8)

        # ============================================================================
        # 1. CLAMPED SCALE-NORMALIZED P1 LOSS (Coefficient of Variation)
        # ============================================================================
        # CV = std / |mean| is scale-invariant
        # CLAMPED: Prevent runaway by capping CV at CV_MAX
        cv = output_std / (torch.abs(output_mean) + 1e-8)
        cv_clamped = torch.clamp(cv, max=CV_MAX)  # NEW: Clamp to prevent runaway
        p1_loss = -torch.log(cv_clamped + 1e-8)

        # ============================================================================
        # 2. MARGINAL CONSTRAINTS
        # ============================================================================
        # Force output mean and variance to match ground truth
        gt_mean_tensor = torch.tensor(GT_MEAN, device=model.device, dtype=dtype)
        gt_var_tensor = torch.tensor(GT_VAR, device=model.device, dtype=dtype)

        mean_loss = (output_mean - gt_mean_tensor) ** 2
        var_loss = (output_var - gt_var_tensor) ** 2

        # ============================================================================
        # TOTAL LOSS (using dynamic weights from schedule)
        # ============================================================================
        total_loss = (base_loss
                     + p1_weight * p1_loss
                     + current_mean_weight * mean_loss
                     + current_var_weight * var_loss)

    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'loss': total_loss,
        'reconstruction_loss': re_surface,
        'kl_loss': kl_loss,
        'p1_loss': p1_loss,
        'mean_loss': mean_loss,
        'var_loss': var_loss,
        'output_mean': output_mean.item(),
        'output_var': output_var.item(),
        'cv': cv.item(),
        'cv_clamped': cv_clamped.item(),
        'p1_weight': p1_weight,
        'mean_weight': current_mean_weight,
        'var_weight': current_var_weight
    }


def measure_decoder_gain(model, val_surface, num_contexts=50, num_samples=50):
    """Measure decoder gain: output_variance / z_variance"""
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    model.eval()

    z_vars = []
    out_vars = []

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
            sigma_p = torch.sqrt(torch.diag(Sigma_p)).unsqueeze(0).unsqueeze(0)

            samples = []
            z_samples = []

            for _ in range(num_samples):
                epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                z = mu_p + sigma_p * epsilon
                z_samples.append(z.cpu())

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

            out_vars.append(samples_tensor.var(dim=0).mean().item())
            z_vars.append(z_tensor.var(dim=0).mean().item())

    mean_out_var = np.mean(out_vars)
    mean_z_var = np.mean(z_vars)
    decoder_gain = mean_out_var / (mean_z_var + 1e-10)

    return decoder_gain, mean_z_var, mean_out_var


def evaluate_marginal(model, val_surface, num_contexts=100, num_samples=50):
    """Evaluate marginal statistics: mean, variance, and P1."""
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    model.eval()

    all_outputs = []
    within_vars = []

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
            sigma_p = torch.sqrt(torch.diag(Sigma_p)).unsqueeze(0).unsqueeze(0)

            samples = []
            for _ in range(num_samples):
                epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                z = mu_p + sigma_p * epsilon

                if model.config.get("compress_context", True):
                    ctx_embedding_dim = latent_dim
                else:
                    ctx_embedding_dim = model.config["mem_hidden"]

                ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
                decoder_input = torch.cat([z, ctx_zeros], dim=-1)
                decoded = model.decoder(decoder_input)

                if isinstance(decoded, tuple):
                    decoded = decoded[0]

                samples.append(decoded.squeeze().cpu().numpy())

            samples_array = np.array(samples)
            all_outputs.append(samples_array.mean(axis=0))
            within_vars.append(samples_array.var(axis=0).mean())

    all_outputs = np.array(all_outputs)
    gen_mean = all_outputs.mean()
    gen_var = all_outputs.var()
    within_var = np.mean(within_vars)
    p1 = within_var / GT_VAR * 100

    return gen_mean, gen_var, within_var, p1


def train_model(model, train_loader, val_surface, config, epochs=TOTAL_EPOCHS):
    """Train model with marginal-preserving P1 loss."""
    lr = config.get("lr", 0.001) if isinstance(config, dict) else config.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        'loss': [], 'recon_loss': [], 'kl_loss': [], 'p1_loss': [],
        'mean_loss': [], 'var_loss': [],
        'output_mean': [], 'output_var': [], 'cv': [],
        'decoder_gain': [], 'p1_metric': [],
        'gen_mean': [], 'gen_var': []
    }

    best_loss = float('inf')
    output_dir = Path("results/prior_encoder_ablation/exp9c_stable_p1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining Schedule:")
    print(f"  Phase 1 (epochs 0-{PHASE1_EPOCHS}): Establish marginal (no P1)")
    print(f"  Phase 2 (epochs {PHASE1_EPOCHS}-{TOTAL_EPOCHS}): Add P1 while maintaining marginal")

    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: [] for k in ['loss', 'recon', 'kl', 'p1', 'mean', 'var', 'out_mean', 'out_var', 'cv']}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch_dict = {
                "surface": batch[0],
                "ex_feats": batch[1]
            }

            loss_dict = train_step_with_marginal_constraints(
                model, batch_dict, optimizer, epoch,
                p1_weight_base=P1_WEIGHT,
                mean_weight=MEAN_WEIGHT,
                var_weight=VAR_WEIGHT,
                p1_num_samples=P1_NUM_SAMPLES
            )

            epoch_losses['loss'].append(loss_dict['loss'].item())
            epoch_losses['recon'].append(loss_dict['reconstruction_loss'].item())
            epoch_losses['kl'].append(loss_dict['kl_loss'].item())
            epoch_losses['p1'].append(loss_dict['p1_loss'].item())
            epoch_losses['mean'].append(loss_dict['mean_loss'].item())
            epoch_losses['var'].append(loss_dict['var_loss'].item())
            epoch_losses['out_mean'].append(loss_dict['output_mean'])
            epoch_losses['out_var'].append(loss_dict['output_var'])
            epoch_losses['cv'].append(loss_dict['cv'])

            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'mean': f"{loss_dict['output_mean']:.3f}",
                'cv': f"{loss_dict['cv']:.4f}",
                'p1_w': f"{loss_dict['p1_weight']:.3f}"
            })

        scheduler.step()

        # Record history
        history['loss'].append(np.mean(epoch_losses['loss']))
        history['recon_loss'].append(np.mean(epoch_losses['recon']))
        history['kl_loss'].append(np.mean(epoch_losses['kl']))
        history['p1_loss'].append(np.mean(epoch_losses['p1']))
        history['mean_loss'].append(np.mean(epoch_losses['mean']))
        history['var_loss'].append(np.mean(epoch_losses['var']))
        history['output_mean'].append(np.mean(epoch_losses['out_mean']))
        history['output_var'].append(np.mean(epoch_losses['out_var']))
        history['cv'].append(np.mean(epoch_losses['cv']))

        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            decoder_gain, z_var, out_var = measure_decoder_gain(model, val_surface)
            gen_mean, gen_var, within_var, p1_metric = evaluate_marginal(model, val_surface)

            history['decoder_gain'].append(decoder_gain)
            history['p1_metric'].append(p1_metric)
            history['gen_mean'].append(gen_mean)
            history['gen_var'].append(gen_var)

            phase = "Phase 1 (Marginal)" if epoch < PHASE1_EPOCHS else "Phase 2 (P1)"

            print(f"\n  [{phase}] Epoch {epoch+1}:")
            print(f"    Decoder Gain: {decoder_gain:.2e}")
            print(f"    P1 Metric: {p1_metric:.4f}%")
            print(f"    Gen Mean: {gen_mean:.4f} (GT: {GT_MEAN:.4f}, ratio: {gen_mean/GT_MEAN*100:.1f}%)")
            print(f"    Gen Var:  {gen_var:.6f} (GT: {GT_VAR:.6f}, ratio: {gen_var/GT_VAR*100:.1f}%)")
            print(f"    CV: {np.mean(epoch_losses['cv']):.4f}")

            # Check success criteria
            mean_ratio = gen_mean / GT_MEAN * 100
            var_ratio = gen_var / GT_VAR * 100

            if 90 <= mean_ratio <= 110 and 80 <= var_ratio <= 120 and p1_metric > 2.0:
                print(f"\n  SUCCESS CRITERIA MET!")
                print(f"    Mean ratio: {mean_ratio:.1f}% [90-110%]")
                print(f"    Var ratio: {var_ratio:.1f}% [80-120%]")
                print(f"    P1: {p1_metric:.2f}% [>2%]")

        # Save checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'model_config': model.config,
                'epoch': epoch + 1,
                'history': history
            }, output_dir / f"model_ep{epoch+1}.pt")

    # Save final model
    torch.save({
        'state_dict': model.state_dict(),
        'model_config': model.config,
        'epoch': epochs,
        'history': history
    }, output_dir / f"model_ep{epochs}.pt")

    return model, history


def main():
    print("="*70)
    print("EXPERIMENT 9c: Stable P1 Loss with Clamped CV")
    print("="*70)

    print("\nKEY CHANGES FROM EXP9b:")
    print("  1. CLAMPED CV: max CV = 1.0 to prevent runaway")
    print("  2. LOWER P1 WEIGHT: 0.01 instead of 0.1")
    print("  3. HIGHER MARGINAL WEIGHTS IN PHASE 2: 10x increase")
    print("  4. SLOWER P1 RAMP: Full 100 epochs instead of 50")

    print(f"\nPARAMETERS:")
    print(f"  P1_WEIGHT = {P1_WEIGHT} (was 0.1 in exp9b)")
    print(f"  CV_MAX = {CV_MAX} (NEW: clamp CV)")
    print(f"  MEAN_WEIGHT Phase1 = {MEAN_WEIGHT}, Phase2 = {PHASE2_MEAN_WEIGHT}")
    print(f"  VAR_WEIGHT Phase1 = {VAR_WEIGHT}, Phase2 = {PHASE2_VAR_WEIGHT}")
    print(f"  GT_MEAN = {GT_MEAN}")
    print(f"  GT_VAR = {GT_VAR}")
    print(f"  PHASE1_EPOCHS = {PHASE1_EPOCHS}")
    print(f"  PHASE2_EPOCHS = {PHASE2_EPOCHS}")

    # Load data
    data_file = Path("data/vol_surface_with_ret.npz")
    data = np.load(data_file)
    surface = torch.tensor(data["surface"], dtype=torch.float32)

    print(f"\nLoading data...")
    print(f"Train data: {surface[:4000].shape}")
    print(f"Val data: {surface[4000:].shape}")

    # Use config as class (dataclass-like)
    config = BackfillContext60ConfigV4FullCov

    # Build sequences
    train_surface = surface[:4000]
    val_surface = surface[4000:]

    C = config.context_len
    H = TRAINING_HORIZON
    seq_len = C + H

    print(f"\nSequence length: {seq_len} (context={C} + horizon={H})")

    train_seqs = []
    for i in range(len(train_surface) - seq_len + 1):
        train_seqs.append(train_surface[i:i+seq_len])

    train_seqs = torch.stack(train_seqs)
    train_feats = torch.zeros(len(train_seqs), seq_len, 3)

    print(f"Training sequences: {train_seqs.shape}")

    train_dataset = TensorDataset(train_seqs, train_feats)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model config dict
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

    model = CVAEFullCovPrior(model_config)
    model = model.to(device)
    model.device = device
    model.config = model_config

    print("\n" + "="*70)
    print("Training CVAEFullCovPrior with Stable P1 Loss (Clamped CV)")
    print("="*70)

    # Compile model
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

    # Initial measurements
    print("\nInitial decoder gain measurement...")
    decoder_gain, z_var, out_var = measure_decoder_gain(model._orig_mod, val_surface)
    print(f"  Initial decoder gain: {decoder_gain:.2e}")
    print(f"  Z variance: {z_var:.6f}, Output variance: {out_var:.6f}")

    gen_mean, gen_var, within_var, p1_metric = evaluate_marginal(model._orig_mod, val_surface)
    print(f"  Initial gen mean: {gen_mean:.4f} (GT: {GT_MEAN:.4f})")
    print(f"  Initial gen var: {gen_var:.6f} (GT: {GT_VAR:.6f})")
    print(f"  Initial P1: {p1_metric:.4f}%")

    # Train
    model, history = train_model(
        model, train_loader, val_surface, model_config,
        epochs=TOTAL_EPOCHS
    )

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    decoder_gain, z_var, out_var = measure_decoder_gain(model._orig_mod, val_surface)
    gen_mean, gen_var, within_var, p1_metric = evaluate_marginal(model._orig_mod, val_surface)

    print(f"\nDecoder Gain: {decoder_gain:.2e}")
    print(f"P1 Metric: {p1_metric:.4f}%")
    print(f"\nMarginal Statistics:")
    print(f"  Gen Mean: {gen_mean:.4f} (GT: {GT_MEAN:.4f}, ratio: {gen_mean/GT_MEAN*100:.1f}%)")
    print(f"  Gen Var:  {gen_var:.6f} (GT: {GT_VAR:.6f}, ratio: {gen_var/GT_VAR*100:.1f}%)")

    # Check success criteria
    print("\n" + "-"*70)
    print("SUCCESS CRITERIA CHECK:")

    mean_ratio = gen_mean / GT_MEAN * 100
    var_ratio = gen_var / GT_VAR * 100

    criteria = [
        (p1_metric > 2.0, f"P1 > 2%: {p1_metric:.2f}%"),
        (90 <= mean_ratio <= 110, f"Mean ratio 90-110%: {mean_ratio:.1f}%"),
        (80 <= var_ratio <= 120, f"Var ratio 80-120%: {var_ratio:.1f}%")
    ]

    all_passed = True
    for passed, desc in criteria:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nALL CRITERIA PASSED!")
    else:
        print("\nSome criteria not met. May need tuning.")

    output_dir = Path("results/prior_encoder_ablation/exp9c_stable_p1")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
