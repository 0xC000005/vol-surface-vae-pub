"""
Three-Stage Training: Pretrain + Fine-tune + Backfill
Following the Bitter Lesson WITH practical constraints

Stage 1: Pretrain on abundant recent data (scale)
Stage 2: Fine-tune on limited historical data (adaptation)
Stage 3: Generate for target period (backfill)

This combines:
- Bitter Lesson: Scale matters (pretrain on large corpus)
- Reality: Limited data in target regime (fine-tune)
- Transfer learning: General patterns transfer across regimes
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
    Train with autoregressive feedback (same as train_autoregressive_curriculum.py)
    """
    optimizer.zero_grad()

    surface = batch["surface"].to(model.device)
    B, T, H, W = surface.shape
    C = model.config.get("context_len", 20)

    if T < C + ar_steps:
        raise ValueError(f"Sequence too short: need {C + ar_steps}, got {T}")

    context = {"surface": surface[:, :C, :, :]}
    if "ex_feats" in batch:
        context["ex_feats"] = batch["ex_feats"][:, :C, :].to(model.device)

    total_loss = 0
    total_recon = 0
    total_kl = 0

    for step in range(ar_steps):
        original_horizon = model.horizon
        model.horizon = 1

        if "ex_feats" in context:
            surf_recon, ex_recon, z_mean, z_log_var, z = model(context)
        else:
            surf_recon, z_mean, z_log_var, z = model(context)

        model.horizon = original_horizon

        target_surface = surface[:, C+step:C+step+1, :, :]
        recon_loss = model.quantile_loss_fn(surf_recon, target_surface)

        if "ex_feats" in context:
            target_ex_feats = batch["ex_feats"][:, C+step:C+step+1, :].to(model.device)
            if model.config["ex_loss_on_ret_only"]:
                ex_recon = ex_recon[:, :, :1]
                target_ex_feats = target_ex_feats[:, :, :1]
            ex_loss = model.ex_feats_loss_fn(ex_recon, target_ex_feats)
            recon_loss = recon_loss + model.config["re_feat_weight"] * ex_loss

        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        step_loss = recon_loss + model.config["kl_weight"] * kl_loss
        total_loss += step_loss
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        pred_surface = surf_recon[:, 0, 1, :, :]
        new_context_surface = torch.cat([
            context["surface"][:, 1:, :, :],
            pred_surface.unsqueeze(1)
        ], dim=1)

        context = {"surface": new_context_surface}

        if "ex_feats" in batch:
            if 'ex_recon' in locals() and ex_recon is not None:
                pred_ex_feats = ex_recon[:, 0, :]
                new_context_ex_feats = torch.cat([
                    context.get("ex_feats", batch["ex_feats"][:, :C, :].to(model.device))[:, 1:, :],
                    pred_ex_feats.unsqueeze(1)
                ], dim=1)
            else:
                old_ex_feats = batch["ex_feats"][:, :C+step, :].to(model.device)
                new_context_ex_feats = torch.cat([
                    old_ex_feats[:, 1:, :],
                    old_ex_feats[:, -1:, :]
                ], dim=1)
            context["ex_feats"] = new_context_ex_feats

    total_loss = total_loss / ar_steps
    total_loss.backward()
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "reconstruction_loss": total_recon / ar_steps,
        "kl_loss": total_kl / ar_steps,
        "ar_steps": ar_steps
    }


def stage1_pretrain(data_range, epochs=500, model_dir="models_pretrain"):
    """
    Stage 1: Pretrain on abundant recent data

    Args:
        data_range: (start_idx, end_idx) for training data (e.g., (3000, 5000))
        epochs: Number of training epochs

    Returns:
        Trained model
    """
    print("=" * 80)
    print("STAGE 1: PRETRAINING ON ABUNDANT DATA")
    print("=" * 80)
    print(f"Data range: {data_range[0]}-{data_range[1]} ({data_range[1]-data_range[0]} days)")
    print(f"Epochs: {epochs}")
    print("Goal: Learn general volatility surface dynamics")
    print("=" * 80)
    print()

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    ex_data = data["ex_data"]

    train_surface = surfaces[data_range[0]:data_range[1]]
    train_ex_data = ex_data[data_range[0]:data_range[1]]

    # Split train/valid
    split_idx = int(0.8 * len(train_surface))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_loss = float('inf')

    for epoch in range(epochs):
        # Curriculum: phase based on epoch
        if epoch < 200:
            ar_steps = 1
            phase = "Teacher Forcing"
        elif epoch < 350:
            ar_steps = 3
            phase = "Short AR (3 steps)"
        else:
            ar_steps = 10
            phase = "Long AR (10 steps)"

        print(f"\n{phase} (epoch {epoch+1}/{epochs})")

        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"pretrain-{epoch}"):
            if ar_steps == 1:
                metrics = model.train_step(batch, optimizer)
            else:
                metrics = train_autoregressive_step(model, batch, optimizer, ar_steps)
            train_loss += metrics["loss"]

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                metrics = model.test_step(batch)
                valid_loss += metrics["loss"]

        valid_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Valid={valid_loss:.4f}")

        # Save best
        if valid_loss < best_loss:
            best_loss = valid_loss
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "model_config": model.config,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "stage": "pretrain",
                "data_range": data_range
            }, f"{model_dir}/pretrained.pt")
            print(f"  ✓ Saved (loss={best_loss:.4f})")

    print("\n" + "=" * 80)
    print(f"STAGE 1 COMPLETE: Pretrained model saved")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 80)

    return model


def stage2_finetune(pretrained_model_path, data_range, epochs=100, model_dir="models_finetune"):
    """
    Stage 2: Fine-tune on limited historical data

    Args:
        pretrained_model_path: Path to pretrained model
        data_range: (start_idx, end_idx) for limited historical data
        epochs: Number of fine-tuning epochs (fewer than pretraining!)

    Returns:
        Fine-tuned model
    """
    print("=" * 80)
    print("STAGE 2: FINE-TUNING ON LIMITED HISTORICAL DATA")
    print("=" * 80)
    print(f"Data range: {data_range[0]}-{data_range[1]} ({data_range[1]-data_range[0]} days)")
    print(f"Epochs: {epochs}")
    print("Goal: Adapt to target regime with limited data")
    print("=" * 80)
    print()

    # Load pretrained model
    print(f"Loading pretrained model from {pretrained_model_path}...")
    model_data = torch.load(pretrained_model_path, weights_only=False)
    model = CVAEMemRand(model_data["model_config"])
    model.load_state_dict(model_data["model_state_dict"])
    print(f"✓ Loaded pretrained model (loss={model_data['best_loss']:.4f})")
    print()

    # Load limited historical data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    ex_data = data["ex_data"]

    train_surface = surfaces[data_range[0]:data_range[1]]
    train_ex_data = ex_data[data_range[0]:data_range[1]]

    split_idx = int(0.8 * len(train_surface))

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

    train_sampler = CustomBatchSampler(train_dataset, batch_size=64)
    valid_sampler = CustomBatchSampler(valid_dataset, batch_size=64)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler)

    # Fine-tune with lower learning rate (don't forget pretrained knowledge!)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # 10× lower LR
    best_loss = float('inf')

    for epoch in range(epochs):
        # Always use autoregressive training in fine-tuning
        ar_steps = 10

        print(f"\nFine-tuning epoch {epoch+1}/{epochs} (AR steps={ar_steps})")

        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"finetune-{epoch}"):
            metrics = train_autoregressive_step(model, batch, optimizer, ar_steps)
            train_loss += metrics["loss"]

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                metrics = model.test_step(batch)
                valid_loss += metrics["loss"]

        valid_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Valid={valid_loss:.4f}")

        # Save best
        if valid_loss < best_loss:
            best_loss = valid_loss
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                "model_config": model.config,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "stage": "finetune",
                "data_range": data_range,
                "pretrained_from": pretrained_model_path
            }, f"{model_dir}/finetuned.pt")
            print(f"  ✓ Saved (loss={best_loss:.4f})")

    print("\n" + "=" * 80)
    print(f"STAGE 2 COMPLETE: Fine-tuned model saved")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 80)

    return model


if __name__ == "__main__":
    print("=" * 80)
    print("THREE-STAGE TRAINING: PRETRAIN + FINE-TUNE + BACKFILL")
    print("=" * 80)
    print()
    print("Example scenario:")
    print("  Stage 1: Pretrain on 2010-2023 (abundant data)")
    print("  Stage 2: Fine-tune on 2000-2007 (limited data)")
    print("  Stage 3: Generate for 2008-2010 (backfill target)")
    print()
    print("This combines:")
    print("  - Bitter Lesson: Scale on pretraining")
    print("  - Reality: Limited data in target regime")
    print("  - Transfer: General patterns transfer across regimes")
    print("=" * 80)
    print()

    # Example usage (adjust indices to your data)

    # Stage 1: Pretrain on recent abundant data
    # Assuming indices 3000-5000 represent 2010-2023
    pretrained_model = stage1_pretrain(
        data_range=(3000, 5000),  # ~2000 days of recent data
        epochs=500,
        model_dir="models_pretrain"
    )

    # Stage 2: Fine-tune on limited historical data
    # Assuming indices 1000-2000 represent 2000-2007
    finetuned_model = stage2_finetune(
        pretrained_model_path="models_pretrain/pretrained.pt",
        data_range=(1000, 2000),  # ~1000 days of limited historical data
        epochs=100,  # Fewer epochs for fine-tuning
        model_dir="models_finetune"
    )

    print("\n" + "=" * 80)
    print("READY FOR STAGE 3: BACKFILL GENERATION")
    print("=" * 80)
    print()
    print("Next step:")
    print("  python generate_backfill_sequences.py \\")
    print("    --model models_finetune/finetuned.pt \\")
    print("    --target_period 2008-2010")
    print("=" * 80)
