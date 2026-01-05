"""
Experiment 6 v2: FIXED Extended Training with Problem 1 & 2 Evaluation

CRITICAL FIXES APPLIED:
1. KL Annealing: Linear warmup over 50 epochs (was missing)
2. TF32 Enabled: torch.backends.cuda.matmul.allow_tf32 = True
3. torch.compile(): Model compilation for faster training
4. Dtype Fixes: Proper BF16 handling in measurement functions
5. Extra Features: ex_feats_dim=3 (returns, skews, slopes)
6. Extended Training: 200 epochs (not 50)
7. DataLoader: Parallel workers, pin_memory
8. Cholesky epsilon: 1e-4 for BF16 (not 1e-6)

Goal: Train 3 architectures for 200 epochs with proper Problem 1 & 2 metrics evaluation.

Problem 1: Prior Too Context-Specific (σ too small)
- Metric: E[Var(X|C)] / Var(X) should increase to >2-5% (vs baseline 0.44%)

Problem 2: Prior Parameter Reuse (wrong temporal structure)
- Metric: Roughness ratio should increase to >40% (vs baseline 9.7%)

Method:
1. Train 3 configs for 200 epochs each:
   - A: CVAEFullCovPrior (baseline)
   - B1: CVAEWithPriorEncoderDiagonal
   - B2: CVAEWithPriorEncoderFullCov
2. Track metrics every 20 epochs:
   - KL loss trajectory
   - E[Var(X|C)] / Var(X) (Problem 1)
   - Roughness ratio (Problem 2)
   - phi, sigma_sq evolution (for full cov variants)

Expected runtime: ~10 hours
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, TensorDataset

# === FIX 2: Enable TF32 for faster CUDA training ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.cvae_prior_encoder import CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov
from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


def measure_conditional_variance_fast(model, val_surface, num_contexts=100, num_samples=50):
    """
    Measure E[Var(X|C)] / Var(X) (Problem 1 metric).

    Fixed decoder input shape handling with proper dtype management.
    """
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]

    # === FIX 4: Get model dtype for proper tensor creation ===
    dtype = next(model.parameters()).dtype

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
                # === FIX: Need to include ex_feats since model has ex_feats_dim=3 ===
                ctx_zeros_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
                ctx_input = {"surface": context, "ex_feats": ctx_zeros_feats}

                # === FIX: Use hasattr() instead of isinstance() for compiled models ===
                if hasattr(model, 'prior_encoder'):
                    # Prior encoder - check if diagonal or full cov
                    is_diagonal = not hasattr(model.prior_encoder, 'get_phi')
                    if is_diagonal:
                        mu_p, log_var_p = model.prior_encoder(ctx_input, horizon=1)
                        # Sample from diagonal Gaussian
                        std_p = torch.exp(0.5 * log_var_p)
                        epsilon = torch.randn_like(mu_p)
                        z = mu_p + std_p * epsilon
                    else:
                        # Full covariance
                        mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=1)

                        # === FIX 4 & 8: Larger epsilon for BF16, proper dtype ===
                        eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                        eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                        L = torch.linalg.cholesky(Sigma_p + eps * eye)

                        epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                        # Use einsum for correct temporal correlation: z[b,h,d] = mu[b,h,d] + sum_k L[h,k] * eps[b,k,d]
                        z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                        z = mu_p + z_centered
                else:
                    # Original prior network
                    ctx_out = model.ctx_encoder(ctx_input)
                    context_summary = ctx_out[:, -1, :]
                    mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=1)

                    # === FIX 4 & 8: Larger epsilon for BF16, proper dtype ===
                    eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                    eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                    L = torch.linalg.cholesky(Sigma_p + eps * eye)

                    epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                    # Use einsum for correct temporal correlation
                    z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                    z = mu_p + z_centered

                # Fixed decoder call with proper shape handling
                # z shape: (1, 1, latent_dim)
                # Decoder expects (B, T, latent_dim + ctx_embedding_dim)
                if model.config.get("compress_context", True):
                    ctx_embedding_dim = latent_dim
                else:
                    ctx_embedding_dim = model.config["mem_hidden"]

                # === FIX 4: Use correct dtype for ctx_zeros ===
                ctx_zeros = torch.zeros(1, 1, ctx_embedding_dim, device=device, dtype=dtype)
                decoder_input = torch.cat([z, ctx_zeros], dim=-1)  # (1, 1, latent_dim + ctx_embedding_dim)
                decoded = model.decoder(decoder_input)  # (1, 1, 5, 5) or ((1,1,5,5), (1,1,3))

                # Handle tuple output when ex_feats_dim > 0
                if isinstance(decoded, tuple):
                    decoded = decoded[0]  # Get surface only, ignore ex_feats

                # Ensure output is (1, 5, 5)
                if decoded.dim() == 4:
                    decoded = decoded.squeeze(1)  # Remove horizon dim if present

                samples.append(decoded.cpu())  # (1, 5, 5)

            samples = torch.stack(samples, dim=0).squeeze()  # (num_samples, 5, 5)
            all_samples_per_context.append(samples)

    all_samples = torch.stack(all_samples_per_context).numpy()  # (num_contexts, num_samples, 5, 5)
    var_given_context = np.var(all_samples, axis=1)  # (num_contexts, 5, 5)
    expected_conditional_var = np.mean(var_given_context)

    all_gt = val_surface[C:C+num_contexts].numpy()
    total_var = np.var(all_gt.reshape(-1, 5, 5), axis=0).mean()

    ratio = expected_conditional_var / total_var

    return ratio


def measure_roughness_ratio(model, val_surface, horizon=30, num_sequences=50, num_samples=30):
    """
    Measure roughness ratio (Problem 2 metric).

    Roughness = std(model_daily_changes) / std(gt_daily_changes)
    Target: >40% (vs baseline 9.7%)
    """
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]

    # === FIX 4: Get model dtype for proper tensor creation ===
    dtype = next(model.parameters()).dtype

    model.eval()

    all_roughness = []

    with torch.no_grad():
        for i in range(num_sequences):
            if i + C + horizon > len(val_surface):
                break

            context = val_surface[i:i+C].unsqueeze(0).to(device)

            # Sample multi-horizon trajectories
            trajectories = []
            for _ in range(num_samples):
                # === FIX: Need to include ex_feats since model has ex_feats_dim=3 ===
                ctx_zeros_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
                ctx_input = {"surface": context, "ex_feats": ctx_zeros_feats}

                # Sample z for full horizon
                # === FIX: Use hasattr() instead of isinstance() for compiled models ===
                if hasattr(model, 'prior_encoder'):
                    # Prior encoder - check if diagonal or full cov
                    is_diagonal = not hasattr(model.prior_encoder, 'get_phi')
                    if is_diagonal:
                        mu_p, log_var_p = model.prior_encoder(ctx_input, horizon=horizon)
                        std_p = torch.exp(0.5 * log_var_p)
                        epsilon = torch.randn_like(mu_p)
                        z = mu_p + std_p * epsilon  # (1, H, latent_dim)
                    else:
                        # Full covariance
                        mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=horizon)

                        # === FIX 4 & 8: Larger epsilon for BF16, proper dtype ===
                        eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                        eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                        L = torch.linalg.cholesky(Sigma_p + eps * eye)

                        epsilon = torch.randn(1, horizon, latent_dim, device=device, dtype=dtype)
                        z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                        z = mu_p + z_centered  # (1, H, latent_dim)
                else:
                    # Original prior network
                    ctx_out = model.ctx_encoder(ctx_input)
                    context_summary = ctx_out[:, -1, :]
                    mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=horizon)

                    # === FIX 4 & 8: Larger epsilon for BF16, proper dtype ===
                    eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                    eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                    L = torch.linalg.cholesky(Sigma_p + eps * eye)

                    epsilon = torch.randn(1, horizon, latent_dim, device=device, dtype=dtype)
                    z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                    z = mu_p + z_centered  # (1, H, latent_dim)

                # Decode full trajectory
                # z shape: (1, H, latent_dim)
                # Decoder expects (B, T, latent_dim + ctx_embedding_dim)
                if model.config.get("compress_context", True):
                    ctx_embedding_dim = latent_dim
                else:
                    ctx_embedding_dim = model.config["mem_hidden"]

                # === FIX 4: Use correct dtype for ctx_zeros ===
                ctx_zeros = torch.zeros(1, horizon, ctx_embedding_dim, device=device, dtype=dtype)
                decoder_input = torch.cat([z, ctx_zeros], dim=-1)  # (1, H, latent_dim + ctx_embedding_dim)
                surfaces = model.decoder(decoder_input)  # (1, H, 5, 5) or ((1,H,5,5), (1,H,3))

                # Handle tuple output when ex_feats_dim > 0
                if isinstance(surfaces, tuple):
                    surfaces = surfaces[0]  # Get surface only, ignore ex_feats

                # Extract trajectory: (1, H, 5, 5) → (H, 5, 5)
                surfaces = surfaces.squeeze(0)

                trajectories.append(surfaces.cpu())  # (H, 5, 5)

            trajectories = torch.stack(trajectories)  # (num_samples, H, 5, 5)

            # Compute daily changes
            daily_changes = trajectories[:, 1:, :, :] - trajectories[:, :-1, :, :]  # (num_samples, H-1, 5, 5)
            model_roughness = daily_changes.std().item()

            # Ground truth changes
            gt = val_surface[i+C:i+C+horizon]  # (H, 5, 5)
            gt_changes = gt[1:] - gt[:-1]  # (H-1, 5, 5)
            gt_roughness = gt_changes.std().item()

            if gt_roughness > 0:
                all_roughness.append(model_roughness / gt_roughness)

    return np.mean(all_roughness) if all_roughness else 0.0


def train_model(model_name, model, train_loader, val_surface, config, num_epochs=200):
    """Train a model for specified number of epochs, tracking Problem 1 & 2 metrics."""

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        'epoch': [],
        'recon_loss': [],
        'kl_loss': [],
        'total_loss': [],
        'conditional_var_ratio': [],  # Problem 1
        'roughness_ratio': [],        # Problem 2
        'epoch_time': [],
        'effective_kl_weight': []     # Track KL annealing
    }

    # Add covariance params for full cov models
    if isinstance(model, (CVAEFullCovPrior, CVAEWithPriorEncoderFullCov)):
        history['phi'] = []
        history['sigma_sq'] = []

    print(f"\nTraining {model_name} for {num_epochs} epochs...")
    print(f"Batch size: {config.batch_size}, Using DataLoader with {train_loader.num_workers} workers")
    print()

    best_cond_var_ratio = 0.0
    best_roughness_ratio = 0.0

    # === FIX 1: KL Annealing parameters ===
    kl_anneal_epochs = 50

    for epoch in range(num_epochs):
        start_time = time.time()

        # === FIX 1: KL Annealing - Linear warmup over 50 epochs ===
        if epoch < kl_anneal_epochs:
            effective_kl_weight = config.kl_weight * ((epoch + 1) / kl_anneal_epochs)
            model.kl_weight = effective_kl_weight
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"  KL annealing: {effective_kl_weight:.6f} ({100*(epoch+1)/kl_anneal_epochs:.1f}% of target)")
        else:
            model.kl_weight = config.kl_weight
            effective_kl_weight = config.kl_weight

        model.train()
        epoch_losses = {'recon': [], 'kl': [], 'total': []}

        # === FIX 7: Use DataLoader instead of manual batching ===
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_surface, batch_ex = batch
            batch_surface = batch_surface.to(model.device)
            batch_ex = batch_ex.to(model.device)

            # === FIX 5: Include ex_feats in training ===
            x = {
                "surface": batch_surface,
                "ex_feats": batch_ex
            }

            # Training step
            loss_dict = model.train_step(x, optimizer, scaler=None)

            epoch_losses['recon'].append(loss_dict.get('reconstruction_loss', torch.tensor(0.0)).item())
            epoch_losses['kl'].append(loss_dict.get('kl_loss', torch.tensor(0.0)).item())
            epoch_losses['total'].append(loss_dict['loss'].item())

        # Compute epoch averages
        avg_recon = np.mean(epoch_losses['recon'])
        avg_kl = np.mean(epoch_losses['kl'])
        avg_total = np.mean(epoch_losses['total'])

        # Measure Problem 1 & 2 metrics every 20 epochs
        cond_var_ratio = None
        roughness_ratio = None

        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f"\n  Measuring Problem 1 & 2 metrics...")
            cond_var_ratio = measure_conditional_variance_fast(model, val_surface, num_contexts=100, num_samples=50)
            roughness_ratio = measure_roughness_ratio(model, val_surface, horizon=30, num_sequences=50, num_samples=30)

            if cond_var_ratio > best_cond_var_ratio:
                best_cond_var_ratio = cond_var_ratio
            if roughness_ratio > best_roughness_ratio:
                best_roughness_ratio = roughness_ratio

        epoch_time = time.time() - start_time

        # Record history
        history['epoch'].append(epoch)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        history['total_loss'].append(avg_total)
        history['conditional_var_ratio'].append(cond_var_ratio)
        history['roughness_ratio'].append(roughness_ratio)
        history['epoch_time'].append(epoch_time)
        history['effective_kl_weight'].append(effective_kl_weight)

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
            print(f", P1={cond_var_ratio:.4%}, P2={roughness_ratio:.2%}", end="")
        if 'phi' in history:
            print(f", φ={history['phi'][-1]:.4f}, σ²={history['sigma_sq'][-1]:.4f}", end="")
        print(f" ({epoch_time:.1f}s)")

        # Early stopping
        if cond_var_ratio is not None and cond_var_ratio > 0.05:  # 5% threshold
            print(f"\n✓ Early stopping: Conditional variance {cond_var_ratio:.4%} > 5%")
            break

    print(f"\n✓ Training complete. Best P1: {best_cond_var_ratio:.4%}, Best P2: {best_roughness_ratio:.2%}")

    return model, history


def main():
    print("=" * 80)
    print("EXPERIMENT 6 v2: Fixed Extended Training (200 epochs)")
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

    # === FIX 7: Create DataLoader with proper workers ===
    C = config.context_len
    seq_len = C + 1

    # Create sequences
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
        # Input dimensions
        "feat_dim": (5, 5),
        # === FIX 5: Enable extra features ===
        "ex_feats_dim": 3,  # Use returns, skews, slopes
        "ex_feats_hidden": [16, 8],

        # Latent space
        "latent_dim": config.latent_dim,

        # Device
        "device": device,

        # Loss weights
        "kl_weight": config.kl_weight,  # 1e-5
        "re_feat_weight": 1.0,

        # Surface encoder/decoder
        "surface_hidden": config.surface_hidden,

        # Memory module
        "mem_type": "lstm",
        "mem_hidden": config.mem_hidden,
        "mem_layers": config.mem_layers,
        "mem_dropout": config.mem_dropout,

        # Context encoder
        "ctx_surface_hidden": config.surface_hidden,
        "ctx_ex_feats_hidden": [16, 8],
        "interaction_layers": 2,
        "compress_context": True,

        # Architecture settings
        "use_dense_surface": False,
        "padding": 1,

        # Extra features loss
        "ex_feats_loss_type": "l2",
        "ex_loss_on_ret_only": True,

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

    # === FIX 3: torch.compile() for faster training ===
    print("Compiling model with torch.compile()...")
    model_a = torch.compile(model_a, mode="default", dynamic=True)

    model_a.device = device
    model_a.config = model_config  # Ensure config is accessible

    model_a, history_a = train_model("Baseline", model_a, train_loader, val_surface, config, num_epochs=200)
    results['baseline'] = history_a

    # Save checkpoints
    output_dir = Path("results/prior_encoder_ablation/extended_training_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_config": model_config,
        "state_dict": model_a.state_dict(),
        "epoch": len(history_a['epoch'])
    }, output_dir / "baseline_ep200.pt")

    # =========================================================================
    # VARIANT B1: Prior Encoder Diagonal
    # =========================================================================

    print("\n" + "=" * 80)
    print("VARIANT B1: Prior Encoder Diagonal")
    print("=" * 80)

    model_b1 = CVAEWithPriorEncoderDiagonal(model_config)
    model_b1 = model_b1.to(device)

    # === FIX 3: torch.compile() for faster training ===
    print("Compiling model with torch.compile()...")
    model_b1 = torch.compile(model_b1, mode="default", dynamic=True)

    model_b1.device = device
    model_b1.config = model_config

    model_b1, history_b1 = train_model("Prior Encoder Diagonal", model_b1, train_loader, val_surface, config, num_epochs=200)
    results['prior_encoder_diagonal'] = history_b1

    torch.save({
        "model_config": model_config,
        "state_dict": model_b1.state_dict(),
        "epoch": len(history_b1['epoch'])
    }, output_dir / "prior_encoder_diagonal_ep200.pt")

    # =========================================================================
    # VARIANT B2: Prior Encoder Full Cov
    # =========================================================================

    print("\n" + "=" * 80)
    print("VARIANT B2: Prior Encoder Full Covariance")
    print("=" * 80)

    model_b2 = CVAEWithPriorEncoderFullCov(model_config)
    model_b2 = model_b2.to(device)

    # === FIX 3: torch.compile() for faster training ===
    print("Compiling model with torch.compile()...")
    model_b2 = torch.compile(model_b2, mode="default", dynamic=True)

    model_b2.device = device
    model_b2.config = model_config

    model_b2, history_b2 = train_model("Prior Encoder Full Cov", model_b2, train_loader, val_surface, config, num_epochs=200)
    results['prior_encoder_full_cov'] = history_b2

    torch.save({
        "model_config": model_config,
        "state_dict": model_b2.state_dict(),
        "epoch": len(history_b2['epoch'])
    }, output_dir / "prior_encoder_full_cov_ep200.pt")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================

    print("\n" + "=" * 80)
    print("RESULTS: Problem 1 & 2 Evaluation")
    print("=" * 80)
    print()

    # Final epoch metrics
    print("FINAL EPOCH METRICS")
    print("-" * 80)
    print(f"{'Model':<30} {'Total Loss':<15} {'KL Loss':<15} {'P1 (Cond Var)':<15} {'P2 (Roughness)':<15}")
    print("-" * 80)

    for name, hist in results.items():
        final_loss = hist['total_loss'][-1]
        final_kl = hist['kl_loss'][-1]

        # Get last non-None values
        cond_var_values = [x for x in hist['conditional_var_ratio'] if x is not None]
        roughness_values = [x for x in hist['roughness_ratio'] if x is not None]

        final_p1 = cond_var_values[-1] if cond_var_values else 0.0
        final_p2 = roughness_values[-1] if roughness_values else 0.0

        p1_str = f"{final_p1:.4%}" if cond_var_values else "N/A"
        p2_str = f"{final_p2:.2%}" if roughness_values else "N/A"

        print(f"{name:<30} {final_loss:<15.6f} {final_kl:<15.3f} {p1_str:<15} {p2_str:<15}")

    print()

    # Best metrics
    print("BEST METRICS (Problem 1 & 2)")
    print("-" * 80)
    print(f"{'Model':<30} {'Best P1 (Cond Var)':<20} {'Best P2 (Roughness)':<20}")
    print("-" * 80)

    for name, hist in results.items():
        cond_var_values = [x for x in hist['conditional_var_ratio'] if x is not None]
        roughness_values = [x for x in hist['roughness_ratio'] if x is not None]

        best_p1 = max(cond_var_values) if cond_var_values else 0.0
        best_p2 = max(roughness_values) if roughness_values else 0.0

        p1_str = f"{best_p1:.4%}" if cond_var_values else "N/A"
        p2_str = f"{best_p2:.2%}" if roughness_values else "N/A"

        print(f"{name:<30} {p1_str:<20} {p2_str:<20}")

    print()

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print("=" * 80)
    print("INTERPRETATION: Which Architecture Addresses Which Problem?")
    print("=" * 80)
    print()

    # Targets from CONDITIONAL_VARIANCE_SOLUTIONS.md
    P1_TARGET = 0.02  # 2%
    P2_TARGET = 0.40  # 40%

    print(f"Problem 1 Target: E[Var(X|C)]/Var(X) > {P1_TARGET:.1%}")
    print(f"Problem 2 Target: Roughness ratio > {P2_TARGET:.1%}")
    print()

    for name, hist in results.items():
        cond_var_values = [x for x in hist['conditional_var_ratio'] if x is not None]
        roughness_values = [x for x in hist['roughness_ratio'] if x is not None]

        best_p1 = max(cond_var_values) if cond_var_values else 0.0
        best_p2 = max(roughness_values) if roughness_values else 0.0

        p1_solved = "✓" if best_p1 >= P1_TARGET else "❌"
        p2_solved = "✓" if best_p2 >= P2_TARGET else "❌"

        print(f"{name}:")
        print(f"  Problem 1 (σ too context-specific): {p1_solved} {best_p1:.4%}")
        print(f"  Problem 2 (parameter reuse): {p2_solved} {best_p2:.2%}")
        print()

    # Recommendation
    print("-" * 80)
    print("RECOMMENDATION:")
    print("-" * 80)

    baseline_p1 = max([x for x in results['baseline']['conditional_var_ratio'] if x is not None], default=0.0)
    baseline_p2 = max([x for x in results['baseline']['roughness_ratio'] if x is not None], default=0.0)

    diagonal_p1 = max([x for x in results['prior_encoder_diagonal']['conditional_var_ratio'] if x is not None], default=0.0)
    diagonal_p2 = max([x for x in results['prior_encoder_diagonal']['roughness_ratio'] if x is not None], default=0.0)

    fullcov_p1 = max([x for x in results['prior_encoder_full_cov']['conditional_var_ratio'] if x is not None], default=0.0)
    fullcov_p2 = max([x for x in results['prior_encoder_full_cov']['roughness_ratio'] if x is not None], default=0.0)

    if fullcov_p1 >= P1_TARGET and fullcov_p2 >= P2_TARGET:
        print("✓ STRONG EVIDENCE: Prior Encoder Full Cov solves BOTH problems!")
        print("  → Proceed with full 400-epoch training for production model")
    elif diagonal_p1 >= P1_TARGET:
        print("✓ MODERATE EVIDENCE: Prior Encoder Diagonal solves Problem 1")
        print("⚠ Problem 2 not solved → May need additional temporal structure")
        print("  → Consider full training with AR(1) regularization")
    elif fullcov_p2 >= P2_TARGET:
        print("✓ MODERATE EVIDENCE: Prior Encoder Full Cov solves Problem 2")
        print("⚠ Problem 1 not solved → May need longer training or σ² tuning")
        print("  → Consider extended training (300-400 epochs)")
    else:
        print("⚠ WEAK EVIDENCE: Neither architecture fully solves the problems")
        print("  → Problem may lie elsewhere (consider further investigation)")

    print()

    # Save results
    np.savez(
        output_dir / "comparison_results.npz",
        **{f"{name}_{key}": np.array(val) if isinstance(val, list) else val
           for name, hist in results.items()
           for key, val in hist.items()}
    )

    # Save training histories
    for name, hist in results.items():
        np.savez(output_dir / f"{name}_history.npz", **hist)

    print(f"Results saved to: {output_dir}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
