"""
Train Cumulative-Aware VAE Options (1-4)

This script trains the 4 different approaches to prevent IV explosion during chaining:
1. Option 1: Cumulative log-return conditioning (StudentTCumulativeDecoder)
2. Option 2: Current log-IV level conditioning (StudentTLevelDecoder)
3. Option 3: Sequence-level training loss (trajectory loss)
4. Option 4: Multi-task decoder (return + level heads)

The key insight is that AR(1) in log-return space doesn't prevent level drift.
These options let the VAE see cumulative/level information to learn level-dependent dynamics.

Usage:
    # Train Option 1 (cumulative)
    python experiments/backfill/two_stage_vae/train_cumulative_options.py --option 1

    # Train Option 2 (level)
    python experiments/backfill/two_stage_vae/train_cumulative_options.py --option 2

    # Train Option 3 (sequence loss)
    python experiments/backfill/two_stage_vae/train_cumulative_options.py --option 3

    # Train Option 4 (multi-task)
    python experiments/backfill/two_stage_vae/train_cumulative_options.py --option 4

    # Train all options
    python experiments/backfill/two_stage_vae/train_cumulative_options.py --option all
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import (
    CVAETwoStageCumulative,
    CVAETwoStageLevel,
    CVAETwoStageMultiTask,
    CVAETwoStageDualPathAR,  # Base AR model for Option 3
)


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_sequences(log_returns, log_surfaces, seq_len=30):
    """Create overlapping sequences for training."""
    n_samples = len(log_returns) - seq_len + 1

    sequences = np.zeros((n_samples, seq_len, 5, 5), dtype=np.float32)
    level_sequences = np.zeros((n_samples, seq_len, 5, 5), dtype=np.float32)

    for i in range(n_samples):
        sequences[i] = log_returns[i:i + seq_len]
        # Level at each position (log-IV after applying returns up to that point)
        level_sequences[i] = log_surfaces[i + 1:i + seq_len + 1]

    return sequences, level_sequences


def train_option1(config, train_loader, val_loader, device, epochs=100, save_dir=None):
    """Train Option 1: Cumulative log-return conditioning."""
    print("\n" + "=" * 70)
    print("Training Option 1: Cumulative Log-Return Conditioning")
    print("=" * 70)

    model = CVAETwoStageCumulative(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_data in train_loader:
            surface = batch_data[0].to(device)
            batch = {"surface": surface}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch, kl_weight=0.001)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                surface = batch_data[0].to(device)
                batch = {"surface": surface}
                loss_dict = model.compute_loss(batch, kl_weight=0.001)
                val_losses.append(loss_dict["loss"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        phi = model.get_ar_phi().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, phi={phi:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                save_path = save_dir / "cumulative_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, save_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return model


def train_option2(config, train_loader, val_loader, level_train, level_val, device, epochs=100, save_dir=None):
    """Train Option 2: Current log-IV level conditioning."""
    print("\n" + "=" * 70)
    print("Training Option 2: Current Log-IV Level Conditioning")
    print("=" * 70)

    model = CVAETwoStageLevel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    # Create level data loaders
    level_train_loader = DataLoader(TensorDataset(level_train), batch_size=64, shuffle=False)
    level_val_loader = DataLoader(TensorDataset(level_val), batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for (batch_data,), (level_data,) in zip(train_loader, level_train_loader):
            surface = batch_data.to(device)
            log_surfaces = level_data.to(device)
            batch = {"surface": surface}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch, log_surfaces=log_surfaces, kl_weight=0.001)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,), (level_data,) in zip(val_loader, level_val_loader):
                surface = batch_data.to(device)
                log_surfaces = level_data.to(device)
                batch = {"surface": surface}
                loss_dict = model.compute_loss(batch, log_surfaces=log_surfaces, kl_weight=0.001)
                val_losses.append(loss_dict["loss"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        phi = model.get_ar_phi().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, phi={phi:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                save_path = save_dir / "level_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, save_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return model


def train_option3(config, train_data, val_data, device, epochs=100, chain_steps=5, save_dir=None):
    """
    Train Option 3: Sequence-level training loss.

    Instead of per-step reconstruction, we chain predictions and compute
    loss on the entire trajectory. This forces the model to learn stable chaining.

    Note: train_data/val_data are already sequences of shape (N_seqs, seq_len, 5, 5).
    We use the first context_len steps as context and the next chain_steps as target.
    """
    print("\n" + "=" * 70)
    print("Training Option 3: Sequence-Level Training Loss")
    print(f"  Chain steps: {chain_steps}")
    print("=" * 70)

    # Use base AR model
    model = CVAETwoStageDualPathAR(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    context_len = 20

    def create_chain_batches(data, batch_size=32):
        """
        Create batches suitable for chain training.

        data: (N_seqs, seq_len, 5, 5) - already windowed sequences
        We use [:, :context_len] as context and [:, context_len:context_len+chain_steps] as target.
        """
        # Filter sequences that are long enough
        seq_len = data.shape[1]
        if seq_len < context_len + chain_steps:
            raise ValueError(f"Sequence length {seq_len} too short for context_len={context_len} + chain_steps={chain_steps}")

        n_seqs = len(data)
        indices = np.random.permutation(n_seqs)

        batches = []
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]

            # Extract context and target from each sequence
            batch_context = data[batch_idx, :context_len]  # (B, context_len, 5, 5)
            batch_target = data[batch_idx, context_len:context_len + chain_steps]  # (B, chain_steps, 5, 5)

            batch_context = torch.tensor(batch_context, dtype=torch.float32)
            batch_target = torch.tensor(batch_target, dtype=torch.float32)
            batches.append((batch_context, batch_target))

        return batches

    for epoch in range(epochs):
        model.train()
        train_losses = []

        train_batches = create_chain_batches(train_data, batch_size=32)

        for context, target_trajectory in train_batches:
            context = context.to(device)  # (B, C, 5, 5)
            target_trajectory = target_trajectory.to(device)  # (B, chain_steps, 5, 5)
            B = context.shape[0]

            optimizer.zero_grad()

            # Chain predictions
            pred_trajectory = []
            current_context = context.clone()
            prev_log_return = context[:, -1].clone()  # (B, 5, 5)

            for step in range(chain_steps):
                # Encode current context
                ctx_emb = model.ctx_encoder({"surface": current_context})
                z_mean, z_logvar, z = model.main_encoder({"surface": current_context})

                # Prepare prev_x for AR(1)
                prev_x = prev_log_return.unsqueeze(1)  # (B, 1, 5, 5)

                # Predict next step (use mean, not sample, for gradient flow)
                mean, _, _, _ = model.decoder(
                    ctx_emb[:, -1:], z[:, -1:], prev_x=prev_x, sample=False
                )
                pred = mean[:, 0]  # (B, 5, 5)
                pred_trajectory.append(pred)

                # Update context with prediction (autoregressive)
                current_context = torch.cat([current_context[:, 1:], pred.unsqueeze(1)], dim=1)
                prev_log_return = pred

            pred_trajectory = torch.stack(pred_trajectory, dim=1)  # (B, chain_steps, 5, 5)

            # Loss on TRAJECTORY, not individual steps
            trajectory_loss = F.mse_loss(pred_trajectory, target_trajectory)

            # Add KL regularization
            kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            loss = trajectory_loss + 0.001 * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(trajectory_loss.item())

        # Validation
        model.eval()
        val_losses = []

        val_batches = create_chain_batches(val_data, batch_size=32)

        with torch.no_grad():
            for context, target_trajectory in val_batches:
                context = context.to(device)
                target_trajectory = target_trajectory.to(device)
                B = context.shape[0]

                pred_trajectory = []
                current_context = context.clone()
                prev_log_return = context[:, -1].clone()

                for step in range(chain_steps):
                    ctx_emb = model.ctx_encoder({"surface": current_context})
                    z_mean, z_logvar, z = model.main_encoder({"surface": current_context})

                    prev_x = prev_log_return.unsqueeze(1)
                    mean, _, _, _ = model.decoder(
                        ctx_emb[:, -1:], z[:, -1:], prev_x=prev_x, sample=False
                    )
                    pred = mean[:, 0]
                    pred_trajectory.append(pred)

                    current_context = torch.cat([current_context[:, 1:], pred.unsqueeze(1)], dim=1)
                    prev_log_return = pred

                pred_trajectory = torch.stack(pred_trajectory, dim=1)
                trajectory_loss = F.mse_loss(pred_trajectory, target_trajectory)
                val_losses.append(trajectory_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        phi = model.get_ar_phi().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_traj_loss={train_loss:.6f}, val_traj_loss={val_loss:.6f}, phi={phi:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                save_path = save_dir / "sequence_loss_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "chain_steps": chain_steps,
                }, save_path)

    print(f"\nBest validation trajectory loss: {best_val_loss:.6f}")
    return model


def train_option4(config, train_loader, val_loader, level_train, level_val, device, epochs=100, save_dir=None):
    """Train Option 4: Multi-task decoder (return + level)."""
    print("\n" + "=" * 70)
    print("Training Option 4: Multi-Task Decoder (Return + Level)")
    print("=" * 70)

    model = CVAETwoStageMultiTask(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    # Create level data loaders
    level_train_loader = DataLoader(TensorDataset(level_train), batch_size=64, shuffle=False)
    level_val_loader = DataLoader(TensorDataset(level_val), batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        level_mses = []

        for (batch_data,), (level_data,) in zip(train_loader, level_train_loader):
            surface = batch_data.to(device)
            log_surfaces = level_data.to(device)
            batch = {"surface": surface}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch, log_surfaces=log_surfaces, kl_weight=0.001)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            level_mses.append(loss_dict["level_mse"].item())

        # Validation
        model.eval()
        val_losses = []
        val_level_mses = []
        with torch.no_grad():
            for (batch_data,), (level_data,) in zip(val_loader, level_val_loader):
                surface = batch_data.to(device)
                log_surfaces = level_data.to(device)
                batch = {"surface": surface}
                loss_dict = model.compute_loss(batch, log_surfaces=log_surfaces, kl_weight=0.001)
                val_losses.append(loss_dict["loss"].item())
                val_level_mses.append(loss_dict["level_mse"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        level_mse = np.mean(level_mses)
        val_level_mse = np.mean(val_level_mses)
        phi = model.get_ar_phi().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"level_mse={level_mse:.6f}, phi={phi:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                save_path = save_dir / "multitask_best.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, save_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train cumulative-aware VAE options")
    parser.add_argument("--option", type=str, default="all",
                        help="Which option to train: 1, 2, 3, 4, or all")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--chain_steps", type=int, default=5,
                        help="Number of chain steps for Option 3")
    args = parser.parse_args()

    print("=" * 70)
    print("Cumulative-Aware VAE Training")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)

    # Create sequences
    seq_len = 30
    sequences, level_sequences = create_sequences(log_returns, log_surfaces, seq_len=seq_len)

    # Train/val split
    train_end = int(len(sequences) * 0.7)
    train_data = sequences[:train_end]
    val_data = sequences[train_end:]
    level_train = level_sequences[:train_end]
    level_val = level_sequences[train_end:]

    print(f"\nData shapes:")
    print(f"  Train sequences: {train_data.shape}")
    print(f"  Val sequences: {val_data.shape}")
    print(f"  Train levels: {level_train.shape}")
    print(f"  Val levels: {level_val.shape}")

    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)
    level_train_tensor = torch.tensor(level_train, dtype=torch.float32)
    level_val_tensor = torch.tensor(level_val, dtype=torch.float32)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Model config
    config = {
        "feat_dim": (5, 5),
        "latent_dim": 16,
        "ctx_embedding_dim": 3,
        "context_len": 20,
        "cov_rank": 4,
        "learn_ar_phi": True,
        "target_ar_phi": -0.35,
        "lambda_level": 0.1,  # For Option 4
        # Level reversion for Option 1 (prevent explosion during chaining)
        "learn_level_reversion": False,
        "level_reversion_strength": 0.3,  # Tested: 0.3 prevents explosion
        "device": device,
        # Encoder config
        "surface_hidden": [2, 4, 2],
        "mem_type": "lstm",
        "mem_hidden": 8,
        "mem_layers": 1,
        "mem_dropout": 0.2,
        # Ctx encoder
        "ctx_surface_hidden": [2, 4, 2],
        "ctx_mem_type": "lstm",
        "ctx_mem_hidden": 8,
        "ctx_mem_layers": 1,
        "ctx_mem_dropout": 0.2,
        "ctx_compress": True,
    }

    # Create save directory
    save_dir = Path("models/backfill/two_stage/cumulative_options")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train requested options
    options = [args.option] if args.option != "all" else ["1", "2", "3", "4"]

    for opt in options:
        if opt == "1":
            train_option1(config, train_loader, val_loader, device,
                         epochs=args.epochs, save_dir=save_dir)
        elif opt == "2":
            train_option2(config, train_loader, val_loader,
                         level_train_tensor, level_val_tensor, device,
                         epochs=args.epochs, save_dir=save_dir)
        elif opt == "3":
            train_option3(config, train_data, val_data, device,
                         epochs=args.epochs, chain_steps=args.chain_steps,
                         save_dir=save_dir)
        elif opt == "4":
            train_option4(config, train_loader, val_loader,
                         level_train_tensor, level_val_tensor, device,
                         epochs=args.epochs, save_dir=save_dir)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Models saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
