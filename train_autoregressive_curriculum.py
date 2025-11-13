"""
True Autoregressive Training with Curriculum Learning

Following the Bitter Lesson: Train exactly how you'll deploy.
No hand-crafted statistical losses - let properties emerge naturally.

Curriculum:
- Phase 1 (epochs 0-200): Teacher forcing (single-step)
- Phase 2 (epochs 201-350): Short autoregressive (3-step rollout)
- Phase 3 (epochs 351-500): Long autoregressive (10-step rollout)
"""

import torch
import numpy as np
from tqdm import tqdm
from config.backfill_config import BackfillConfig
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from torch.utils.data import DataLoader
import os


def train_autoregressive_step(model, batch, optimizer, ar_steps=3):
    """
    Train with autoregressive feedback.

    Args:
        model: CVAEMemRand instance
        batch: dict with "surface" (B, T, 5, 5) where T >= context_len + ar_steps
        optimizer: PyTorch optimizer
        ar_steps: Number of autoregressive steps (3, 10, or 30)

    Returns:
        dict with loss metrics
    """
    optimizer.zero_grad()

    surface = batch["surface"].to(model.device)
    B, T, H, W = surface.shape
    C = model.config.get("context_len", 20)

    # Validate sequence length
    if T < C + ar_steps:
        raise ValueError(f"Sequence too short: need {C + ar_steps}, got {T}")

    # Initialize with real context
    context = {"surface": surface[:, :C, :, :]}
    if "ex_feats" in batch:
        context["ex_feats"] = batch["ex_feats"][:, :C, :].to(model.device)

    total_loss = 0
    total_recon = 0
    total_kl = 0

    # Autoregressive rollout
    for step in range(ar_steps):
        # Forward pass
        if "ex_feats" in context:
            # Temporarily set horizon=1 for single-step prediction
            original_horizon = model.horizon
            model.horizon = 1

            surf_recon, ex_recon, z_mean, z_log_var, z = model(context)
            model.horizon = original_horizon
        else:
            original_horizon = model.horizon
            model.horizon = 1
            surf_recon, z_mean, z_log_var, z = model(context)
            model.horizon = original_horizon

        # Ground truth for this step
        target_surface = surface[:, C+step:C+step+1, :, :]

        # Reconstruction loss (quantile regression)
        recon_loss = model.quantile_loss_fn(surf_recon, target_surface)

        # Handle ex_feats if present
        if "ex_feats" in context:
            target_ex_feats = batch["ex_feats"][:, C+step:C+step+1, :].to(model.device)
            if model.config["ex_loss_on_ret_only"]:
                ex_recon = ex_recon[:, :, :1]
                target_ex_feats = target_ex_feats[:, :, :1]
            ex_loss = model.ex_feats_loss_fn(ex_recon, target_ex_feats)
            recon_loss = recon_loss + model.config["re_feat_weight"] * ex_loss

        # KL divergence
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        # Step loss
        step_loss = recon_loss + model.config["kl_weight"] * kl_loss
        total_loss += step_loss
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        # KEY: Feed prediction back as context for next step
        # Use p50 (median) as point estimate
        pred_surface = surf_recon[:, 0, 1, :, :]  # (B, H, W) - median quantile

        # Update context: drop oldest, append prediction
        new_context_surface = torch.cat([
            context["surface"][:, 1:, :, :],  # Drop oldest
            pred_surface.unsqueeze(1)          # Append prediction
        ], dim=1)

        context = {"surface": new_context_surface}

        # Handle ex_feats
        if "ex_feats" in batch:
            if "ex_feats" in locals() and ex_recon is not None:
                # Use predicted ex_feats if available
                pred_ex_feats = ex_recon[:, 0, :]  # (B, ex_dim)
                new_context_ex_feats = torch.cat([
                    context.get("ex_feats", batch["ex_feats"][:, :C, :].to(model.device))[:, 1:, :],
                    pred_ex_feats.unsqueeze(1)
                ], dim=1)
            else:
                # Fallback: repeat last value
                old_ex_feats = batch["ex_feats"][:, :C+step, :].to(model.device)
                new_context_ex_feats = torch.cat([
                    old_ex_feats[:, 1:, :],
                    old_ex_feats[:, -1:, :]
                ], dim=1)
            context["ex_feats"] = new_context_ex_feats

    # Average loss over steps
    total_loss = total_loss / ar_steps
    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "reconstruction_loss": total_recon / ar_steps,
        "kl_loss": total_kl / ar_steps,
        "ar_steps": ar_steps
    }


def train_curriculum(model, train_loader, valid_loader, epochs=500, lr=1e-5,
                     model_dir="models_backfill", file_name="backfill_ar.pt"):
    """
    Curriculum training with increasing autoregressive steps.

    Phase 1 (0-200): Teacher forcing (horizon=1)
    Phase 2 (201-350): Short AR (3 steps)
    Phase 3 (351-500): Long AR (10 steps)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    print("=" * 80)
    print("CURRICULUM AUTOREGRESSIVE TRAINING")
    print("=" * 80)
    print(f"Phase 1 (epochs 1-200): Teacher forcing")
    print(f"Phase 2 (epochs 201-350): Autoregressive (3 steps)")
    print(f"Phase 3 (epochs 351-500): Autoregressive (10 steps)")
    print("=" * 80)
    print()

    for epoch in range(epochs):
        # Determine training phase
        if epoch < 200:
            phase = 1
            ar_steps = 1
            phase_name = "Teacher Forcing"
        elif epoch < 350:
            phase = 2
            ar_steps = 3
            phase_name = "Short AR (3 steps)"
        else:
            phase = 3
            ar_steps = 10
            phase_name = "Long AR (10 steps)"

        print(f"\nPhase {phase} - {phase_name} (epoch {epoch+1}/{epochs})")

        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        pbar = tqdm(train_loader, desc=f"train-{epoch}")
        for batch in pbar:
            if ar_steps == 1:
                # Use existing train_step for teacher forcing
                metrics = model.train_step(batch, optimizer)
            else:
                # Use autoregressive training
                metrics = train_autoregressive_step(model, batch, optimizer, ar_steps)

            train_loss += metrics["loss"]
            train_recon += metrics["reconstruction_loss"]
            train_kl += metrics["kl_loss"]

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "recon": f"{metrics['reconstruction_loss']:.4f}"
            })

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        # Validation (always use teacher forcing for consistency)
        model.eval()
        valid_loss = 0
        valid_recon = 0
        valid_kl = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid-{epoch}"):
                metrics = model.test_step(batch)
                valid_loss += metrics["loss"]
                valid_recon += metrics["reconstruction_loss"]
                valid_kl += metrics["kl_loss"]

        valid_loss /= len(valid_loader)
        valid_recon /= len(valid_loader)
        valid_kl /= len(valid_loader)

        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Valid={valid_loss:.4f}, AR_steps={ar_steps}")

        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "model_config": model.config,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "phase": phase,
                "ar_steps": ar_steps
            }, f"{model_dir}/{file_name}")
            print(f"  ✓ Saved best model (loss={best_loss:.4f})")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {model_dir}/{file_name}")


if __name__ == "__main__":
    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    ex_data = data["ex_data"]

    # Training indices (4000 days)
    train_start, train_end = BackfillConfig.get_train_indices()
    train_surface = surfaces[train_start:train_end]
    train_ex_data = ex_data[train_start:train_end]

    # Split train/valid
    split_idx = int(0.8 * len(train_surface))

    # Create datasets
    train_dataset = VolSurfaceDataSetRand(
        train_surface[:split_idx],
        train_ex_data[:split_idx],
        min_seq_len=50,
        max_seq_len=60
    )

    valid_dataset = VolSurfaceDataSetRand(
        train_surface[split_idx:],
        train_ex_data[split_idx:],
        min_seq_len=50,
        max_seq_len=60
    )

    # Create data loaders
    train_sampler = CustomBatchSampler(train_dataset, batch_size=64)
    valid_sampler = CustomBatchSampler(valid_dataset, batch_size=64)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler)

    # Create model
    model_config = {
        "feat_dim": (5, 5),
        "ex_feats_dim": 3,
        "latent_dim": BackfillConfig.latent_dim,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "kl_weight": BackfillConfig.kl_weight,
        "re_feat_weight": 1.0,
        "surface_hidden": BackfillConfig.surface_hidden,
        "ex_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": BackfillConfig.mem_hidden,
        "mem_layers": BackfillConfig.mem_layers,
        "mem_dropout": BackfillConfig.mem_dropout,
        "ctx_surface_hidden": BackfillConfig.surface_hidden,
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "compress_context": True,
        "use_dense_surface": False,
        "use_quantile_regression": BackfillConfig.use_quantile_regression,
        "num_quantiles": BackfillConfig.num_quantiles,
        "quantiles": BackfillConfig.quantiles,
        "quantile_loss_weights": BackfillConfig.quantile_loss_weights,
        "ex_loss_on_ret_only": True,
        "ex_feats_loss_type": "l2",
        "horizon": 1,
        "context_len": BackfillConfig.context_len,
    }

    model = CVAEMemRand(model_config)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Train with curriculum
    train_curriculum(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=500,
        lr=BackfillConfig.learning_rate,
        model_dir="models_backfill",
        file_name="backfill_autoregressive.pt"
    )
