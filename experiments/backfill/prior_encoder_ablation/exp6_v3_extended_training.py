"""
Experiment 6 v3: Extended Training with FIXED Metrics

FIXES FROM v2:
1. Fixed roughness calculation: per-sample std then average (was: global std)
2. Fixed isinstance() → hasattr() for compiled model logging
3. Added variance logging for Diagonal model
4. Added variance floor to all models (0.01 minimum)

Goal: Fair three-way comparison to determine which architecture solves P1/P2:
- A: Prior Network (single ctx → global σ²)
- B1: Prior Encoder Diagonal (full ctx → per-timestep σ²)
- B2: Prior Encoder Full Cov (full ctx → global σ² with AR(1))

Test question: Which architecture allows maintaining unconditional marginal
while NOT collapsing conditional marginal?

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

# Enable TF32 for faster CUDA training
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
    """
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

                # === FIX: Use hasattr() instead of isinstance() for compiled models ===
                if hasattr(model, 'prior_encoder'):
                    is_diagonal = not hasattr(model.prior_encoder, 'get_phi')
                    if is_diagonal:
                        mu_p, log_var_p = model.prior_encoder(ctx_input, horizon=1)
                        std_p = torch.exp(0.5 * log_var_p)
                        epsilon = torch.randn_like(mu_p)
                        z = mu_p + std_p * epsilon
                    else:
                        mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=1)
                        eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                        eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                        L = torch.linalg.cholesky(Sigma_p + eps * eye)
                        epsilon = torch.randn(1, 1, latent_dim, device=device, dtype=dtype)
                        z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                        z = mu_p + z_centered
                else:
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


def measure_roughness_ratio(model, val_surface, horizon=30, num_sequences=50, num_samples=30):
    """
    Measure roughness ratio (Problem 2 metric).

    === FIX v3: Compute per-sample roughness, then average ===
    (v2 computed global std which included inter-sample variance)
    """
    C = model.config["context_len"]
    device = model.device
    latent_dim = model.config["latent_dim"]
    dtype = next(model.parameters()).dtype

    model.eval()

    all_roughness = []

    with torch.no_grad():
        for i in range(num_sequences):
            if i + C + horizon > len(val_surface):
                break

            context = val_surface[i:i+C].unsqueeze(0).to(device)

            trajectories = []
            for _ in range(num_samples):
                ctx_zeros_feats = torch.zeros(1, C, 3, device=device, dtype=dtype)
                ctx_input = {"surface": context, "ex_feats": ctx_zeros_feats}

                # === FIX: Use hasattr() instead of isinstance() for compiled models ===
                if hasattr(model, 'prior_encoder'):
                    is_diagonal = not hasattr(model.prior_encoder, 'get_phi')
                    if is_diagonal:
                        mu_p, log_var_p = model.prior_encoder(ctx_input, horizon=horizon)
                        std_p = torch.exp(0.5 * log_var_p)
                        epsilon = torch.randn_like(mu_p)
                        z = mu_p + std_p * epsilon
                    else:
                        mu_p, Sigma_p = model.prior_encoder(ctx_input, horizon=horizon)
                        eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                        eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                        L = torch.linalg.cholesky(Sigma_p + eps * eye)
                        epsilon = torch.randn(1, horizon, latent_dim, device=device, dtype=dtype)
                        z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                        z = mu_p + z_centered
                else:
                    ctx_out = model.ctx_encoder(ctx_input)
                    context_summary = ctx_out[:, -1, :]
                    mu_p, Sigma_p = model.full_cov_prior.get_prior_params(context_summary, horizon=horizon)
                    eps = 1e-4 if dtype == torch.bfloat16 else 1e-6
                    eye = torch.eye(Sigma_p.shape[0], device=device, dtype=dtype)
                    L = torch.linalg.cholesky(Sigma_p + eps * eye)
                    epsilon = torch.randn(1, horizon, latent_dim, device=device, dtype=dtype)
                    z_centered = torch.einsum('hk,bkd->bhd', L, epsilon)
                    z = mu_p + z_centered

                if model.config.get("compress_context", True):
                    ctx_embedding_dim = latent_dim
                else:
                    ctx_embedding_dim = model.config["mem_hidden"]

                ctx_zeros = torch.zeros(1, horizon, ctx_embedding_dim, device=device, dtype=dtype)
                decoder_input = torch.cat([z, ctx_zeros], dim=-1)
                surfaces = model.decoder(decoder_input)

                if isinstance(surfaces, tuple):
                    surfaces = surfaces[0]

                surfaces = surfaces.squeeze(0)
                trajectories.append(surfaces.cpu())

            trajectories = torch.stack(trajectories)  # (num_samples, H, 5, 5)

            # Compute daily changes
            daily_changes = trajectories[:, 1:, :, :] - trajectories[:, :-1, :, :]  # (num_samples, H-1, 5, 5)

            # === FIX v3: Per-sample roughness, then average ===
            # (v2 used daily_changes.std() which included inter-sample variance)
            per_sample_roughness = daily_changes.std(dim=(1, 2, 3))  # (num_samples,) - std over time+grid per sample
            model_roughness = per_sample_roughness.mean().item()  # average across samples

            # Ground truth changes
            gt = val_surface[i+C:i+C+horizon]
            gt_changes = gt[1:] - gt[:-1]
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
        'conditional_var_ratio': [],
        'roughness_ratio': [],
        'epoch_time': [],
        'effective_kl_weight': [],
        'phi': [],        # For full cov models
        'sigma_sq': [],   # For full cov models
        'avg_var': []     # For diagonal model
    }

    print(f"\nTraining {model_name} for {num_epochs} epochs...")
    print(f"Batch size: {config.batch_size}, Using DataLoader with {train_loader.num_workers} workers")
    print()

    best_cond_var_ratio = 0.0
    best_roughness_ratio = 0.0

    kl_anneal_epochs = 50

    for epoch in range(num_epochs):
        start_time = time.time()

        # KL Annealing
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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_surface, batch_ex = batch
            batch_surface = batch_surface.to(model.device)
            batch_ex = batch_ex.to(model.device)

            x = {
                "surface": batch_surface,
                "ex_feats": batch_ex
            }

            loss_dict = model.train_step(x, optimizer, scaler=None)

            epoch_losses['recon'].append(loss_dict.get('reconstruction_loss', torch.tensor(0.0)).item())
            epoch_losses['kl'].append(loss_dict.get('kl_loss', torch.tensor(0.0)).item())
            epoch_losses['total'].append(loss_dict['loss'].item())

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

        # === FIX v3: Use hasattr() for compiled models, log all variance types ===
        phi = None
        sigma_sq = None
        avg_var = None

        if hasattr(model, 'full_cov_prior'):
            # Baseline model
            phi = model.full_cov_prior.get_phi().item()
            sigma_sq = model.full_cov_prior.get_sigma_sq().item()
        elif hasattr(model, 'prior_encoder'):
            if hasattr(model.prior_encoder, 'get_phi'):
                # Full cov prior encoder
                phi = model.prior_encoder.get_phi().item()
                sigma_sq = model.prior_encoder.get_sigma_sq().item()
            else:
                # Diagonal prior encoder - compute average variance from a sample
                # Note: variance is computed per forward pass, so we sample once
                model.eval()
                with torch.no_grad():
                    ctx = val_surface[:model.config["context_len"]].unsqueeze(0).to(model.device)
                    ctx_feats = torch.zeros(1, model.config["context_len"], 3, device=model.device)
                    ctx_input = {"surface": ctx, "ex_feats": ctx_feats}
                    _, log_var_p = model.prior_encoder(ctx_input, horizon=1)
                    avg_var = torch.exp(log_var_p).mean().item()
                model.train()

        history['phi'].append(phi)
        history['sigma_sq'].append(sigma_sq)
        history['avg_var'].append(avg_var)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_total:.6f}, KL={avg_kl:.3f}", end="")
        if cond_var_ratio is not None:
            print(f", P1={cond_var_ratio:.4%}, P2={roughness_ratio:.2%}", end="")
        if phi is not None:
            print(f", φ={phi:.4f}, σ²={sigma_sq:.4f}", end="")
        elif avg_var is not None:
            print(f", avg_var={avg_var:.4f}", end="")
        print(f" ({epoch_time:.1f}s)")

        # Early stopping
        if cond_var_ratio is not None and cond_var_ratio > 0.05:
            print(f"\n✓ Early stopping: Conditional variance {cond_var_ratio:.4%} > 5%")
            break

    print(f"\n✓ Training complete. Best P1: {best_cond_var_ratio:.4%}, Best P2: {best_roughness_ratio:.2%}")

    return model, history


def main():
    print("=" * 80)
    print("EXPERIMENT 6 v3: Fixed Extended Training (200 epochs)")
    print("=" * 80)
    print()
    print("FIXES FROM v2:")
    print("  1. Roughness: per-sample std then average (was: global std)")
    print("  2. Logging: hasattr() for compiled models (was: isinstance())")
    print("  3. Variance logging for Diagonal model")
    print("  4. Variance floor in all models (min=0.01)")
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

    results = {}

    # =========================================================================
    # VARIANT A: Baseline (Prior Network)
    # =========================================================================

    print("=" * 80)
    print("VARIANT A: Baseline (Prior Network - single ctx → global σ²)")
    print("=" * 80)

    model_a = CVAEFullCovPrior(model_config)
    model_a = model_a.to(device)

    print("Compiling model with torch.compile()...")
    model_a = torch.compile(model_a, mode="default", dynamic=True)

    model_a.device = device
    model_a.config = model_config

    model_a, history_a = train_model("Baseline", model_a, train_loader, val_surface, config, num_epochs=200)
    results['baseline'] = history_a

    # Save checkpoint
    output_dir = Path("results/prior_encoder_ablation/extended_training_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_config": model_config,
        "state_dict": model_a.state_dict(),
        "epoch": len(history_a['epoch'])
    }, output_dir / "baseline_ep200.pt")

    # =========================================================================
    # VARIANT B1: Prior Encoder Diagonal (context-dependent variance)
    # =========================================================================

    print("\n" + "=" * 80)
    print("VARIANT B1: Prior Encoder Diagonal (full ctx → per-timestep σ²)")
    print("=" * 80)

    model_b1 = CVAEWithPriorEncoderDiagonal(model_config)
    model_b1 = model_b1.to(device)

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
    # VARIANT B2: Prior Encoder Full Cov (global variance)
    # =========================================================================

    print("\n" + "=" * 80)
    print("VARIANT B2: Prior Encoder Full Cov (full ctx → global σ² AR(1))")
    print("=" * 80)

    model_b2 = CVAEWithPriorEncoderFullCov(model_config)
    model_b2 = model_b2.to(device)

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
    print("RESULTS: Problem 1 & 2 Evaluation (v3 - Fixed Metrics)")
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

    # Variance parameters
    print("VARIANCE PARAMETERS (End of Training)")
    print("-" * 80)
    print(f"{'Model':<30} {'φ (phi)':<15} {'σ² (sigma_sq)':<15} {'avg_var':<15}")
    print("-" * 80)

    for name, hist in results.items():
        phi_values = [x for x in hist['phi'] if x is not None]
        sigma_sq_values = [x for x in hist['sigma_sq'] if x is not None]
        avg_var_values = [x for x in hist['avg_var'] if x is not None]

        phi_str = f"{phi_values[-1]:.4f}" if phi_values else "N/A"
        sigma_sq_str = f"{sigma_sq_values[-1]:.4f}" if sigma_sq_values else "N/A"
        avg_var_str = f"{avg_var_values[-1]:.4f}" if avg_var_values else "N/A"

        print(f"{name:<30} {phi_str:<15} {sigma_sq_str:<15} {avg_var_str:<15}")

    print()

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print("=" * 80)
    print("INTERPRETATION: Three-Way Architecture Comparison")
    print("=" * 80)
    print()

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
        print(f"  Problem 1 (conditional variance): {p1_solved} {best_p1:.4%}")
        print(f"  Problem 2 (temporal structure): {p2_solved} {best_p2:.2%}")
        print()

    print("-" * 80)
    print("KEY QUESTION: Does context-dependent variance (B1) collapse, or can it work?")
    print("-" * 80)

    diagonal_p1 = max([x for x in results['prior_encoder_diagonal']['conditional_var_ratio'] if x is not None], default=0.0)
    fullcov_p1 = max([x for x in results['prior_encoder_full_cov']['conditional_var_ratio'] if x is not None], default=0.0)

    if diagonal_p1 >= P1_TARGET:
        print("→ Context-dependent variance (B1) WORKS with variance floor")
    elif fullcov_p1 >= P1_TARGET and diagonal_p1 < P1_TARGET:
        print("→ Global variance (B2) is NECESSARY to prevent collapse")
    else:
        print("→ Neither architecture solves P1 - may need additional investigation")

    print()

    # Save results
    np.savez(
        output_dir / "comparison_results.npz",
        **{f"{name}_{key}": np.array(val) if isinstance(val, list) else val
           for name, hist in results.items()
           for key, val in hist.items()}
    )

    for name, hist in results.items():
        np.savez(output_dir / f"{name}_history.npz", **hist)

    print(f"Results saved to: {output_dir}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
