"""
Train Two-Stage CVAE with Full Covariance Decoder on Log-Returns.

This script trains the CVAETwoStageFullCovariance model which uses:
1. Log-return transformation (stationary time series)
2. Full covariance decoder (outputs mean + Cholesky factor L)
3. Combined MSE + Multivariate NLL loss

The full covariance decoder enables correlated sampling across grid points,
addressing Issue #3: Cross-grid correlation destroyed (0.02 vs GT 0.32).

Key Differences from train_two_stage_log_return_nll.py:
- Uses CVAETwoStageFullCovariance instead of CVAETwoStageHeteroscedastic
- Decoder outputs 25x25 Cholesky factor (325 params) instead of 25 variances
- Sampling: x = μ + L @ ε (correlated) vs x_i = μ_i + σ_i * ε_i (independent)
- Multivariate Gaussian NLL instead of per-pixel Gaussian NLL

Usage:
    python experiments/backfill/two_stage_vae/train_two_stage_full_covariance.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageFullCovariance
from config.two_stage_config import TwoStageConfig


def to_log_returns(surfaces):
    """
    Transform IV surfaces to log-returns.

    Args:
        surfaces: (N, 5, 5) array of IV surfaces

    Returns:
        log_returns: (N-1, 5, 5) array of log-returns
        log_surfaces: (N, 5, 5) array of log(IV) for reconstruction
    """
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def train_epoch(model, train_sequences, optimizer, batch_size, device):
    """Train for one epoch with full covariance loss."""
    model.train()
    metrics = {
        "loss": 0, "mse_loss": 0, "nll_loss": 0,
        "kl_loss": 0, "L_diag_mean": 0, "L_offdiag_mean": 0
    }
    n_batches = 0

    # Shuffle sequences
    indices = torch.randperm(len(train_sequences))

    pbar = tqdm(range(0, len(indices), batch_size), desc="Training", leave=False)
    for i in pbar:
        batch_idx = indices[i:i+batch_size]
        batch = train_sequences[batch_idx].to(device)

        losses = model.train_step_autoencoder({"surface": batch}, optimizer)

        for key in metrics:
            if key in losses:
                val = losses[key]
                metrics[key] += val.item() if torch.is_tensor(val) else val
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "mse": f"{losses['mse_loss'].item():.6f}",
            "nll": f"{losses['nll_loss'].item():.4f}",
            "L_diag": f"{losses['L_diag_mean'].item():.3f}",
            "L_off": f"{losses['L_offdiag_mean'].item():.4f}"
        })

    return {k: v / n_batches for k, v in metrics.items()}


def validate(model, val_sequences, batch_size, device):
    """Validate model with full covariance metrics."""
    model.eval()
    metrics = {
        "loss": 0, "mse_loss": 0, "nll_loss": 0,
        "kl_loss": 0, "L_diag_mean": 0
    }
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_sequences), batch_size):
            batch = val_sequences[i:i+batch_size].to(device)
            losses = model.test_step({"surface": batch})

            for key in metrics:
                if key in losses:
                    val = losses[key]
                    metrics[key] += val.item() if torch.is_tensor(val) else val
            n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


def main():
    print("=" * 70)
    print("Two-Stage CVAE Full Covariance Log-Return Training")
    print("=" * 70)
    print()
    print("Training on LOG-RETURNS with FULL COVARIANCE decoder.")
    print("This fixes:")
    print("  1. Horizon variance growth (via log-return stationarity)")
    print("  2. Cross-grid correlation (via Cholesky covariance)")
    print()
    print("Issue being addressed: #3 Cross-grid correlation destroyed")
    print("  Before (diagonal): correlation = 0.02")
    print("  Target (GT):       correlation = 0.32")
    print()

    # Get base config and modify for full covariance
    config = TwoStageConfig.get_model_config()

    # Full covariance-specific config
    config["full_covariance"] = True
    config["mse_weight"] = 1.0           # Weight for mean accuracy
    config["nll_weight"] = 0.1           # Weight for covariance calibration
    config["cholesky_diag_floor"] = 1e-3 # Min diagonal for stability
    config["cholesky_diag_init"] = -2.0  # softplus(-2) ≈ 0.13
    config["decoder_mem_hidden"] = 32    # Need capacity for 325 Cholesky outputs

    # Training parameters
    batch_size = 128  # Smaller batch due to larger model
    n_epochs = 100
    learning_rate = 1e-4

    # Sequence parameters
    context_len = config["context_len"]  # 30
    horizon = config["horizon"]  # 30
    seq_len = context_len + horizon  # 60

    print(f"Training Parameters:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Context Length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  Sequence Length: {seq_len}")
    print()
    print(f"Full Covariance Config:")
    print(f"  MSE Weight: {config['mse_weight']}")
    print(f"  NLL Weight: {config['nll_weight']}")
    print(f"  Cholesky Diag Floor: {config['cholesky_diag_floor']}")
    print(f"  Decoder LSTM Hidden: {config['decoder_mem_hidden']}")
    print(f"  KL Weight: {config['kl_weight']}")
    print(f"  Cholesky parameters: 325 (25×26/2)")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]  # (N, 5, 5)
    print(f"  Total surfaces: {len(surfaces)}")
    print(f"  IV range: [{surfaces.min():.4f}, {surfaces.max():.4f}]")

    # Transform to log-returns
    print("\nTransforming to log-returns...")
    log_returns, log_surfaces = to_log_returns(surfaces)
    log_return_var = log_returns.var()
    log_return_std = log_returns.std()
    print(f"  Log-returns shape: {log_returns.shape}")
    print(f"  Log-return range: [{log_returns.min():.4f}, {log_returns.max():.4f}]")
    print(f"  Log-return mean: {log_returns.mean():.6f}")
    print(f"  Log-return std: {log_return_std:.4f}")
    print(f"  Log-return var: {log_return_var:.4f}")

    # Compute GT cross-grid correlation for reference
    log_returns_flat = log_returns.reshape(-1, 25)
    gt_corr_matrix = np.corrcoef(log_returns_flat.T)
    gt_atm_otm_corr = gt_corr_matrix[12, 0]  # ATM (2,2) vs OTM-short (0,0)
    gt_atm_itm_corr = gt_corr_matrix[12, 24]  # ATM (2,2) vs ITM-long (4,4)
    print(f"  GT ATM-OTM correlation: {gt_atm_otm_corr:.3f}")
    print(f"  GT ATM-ITM correlation: {gt_atm_itm_corr:.3f}")

    # Convert to tensor
    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)

    # Create sequences
    print(f"\nCreating sequences (length={seq_len})...")
    all_sequences = create_sequences(log_returns_tensor, seq_len)
    print(f"  Total sequences: {len(all_sequences)}")

    # Train/val split (80/20)
    n_train = int(len(all_sequences) * 0.8)
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:]
    print(f"  Train sequences: {len(train_sequences)}")
    print(f"  Val sequences: {len(val_sequences)}")

    # Build model
    print("\nBuilding full covariance model...")
    device = config["device"]
    model = CVAETwoStageFullCovariance(config)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Count decoder-specific parameters
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    cholesky_params = sum(p.numel() for p in model.decoder.cholesky_head.parameters())
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Cholesky head parameters: {cholesky_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Checkpoint path
    checkpoint_dir = Path(TwoStageConfig.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "two_stage_full_covariance_best.pt"

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_val_loss = float('inf')
    history = {"train": [], "val": []}

    for epoch in range(n_epochs):
        # Train
        train_metrics = train_epoch(model, train_sequences, optimizer, batch_size, device)
        history["train"].append(train_metrics)

        # Validate
        val_metrics = validate(model, val_sequences, batch_size, device)
        history["val"].append(val_metrics)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train: loss={train_metrics['loss']:.4f} mse={train_metrics['mse_loss']:.6f} "
              f"nll={train_metrics['nll_loss']:.4f} L_diag={train_metrics['L_diag_mean']:.3f} "
              f"L_off={train_metrics['L_offdiag_mean']:.4f} | "
              f"Val: loss={val_metrics['loss']:.4f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "train_loss": train_metrics["loss"],
                "history": history,
                # Metadata
                "use_log_returns": True,
                "full_covariance": True,
                "log_return_stats": {
                    "mean": float(log_returns.mean()),
                    "std": float(log_return_std),
                    "var": float(log_return_var),
                    "min": float(log_returns.min()),
                    "max": float(log_returns.max()),
                },
                "gt_correlations": {
                    "atm_otm": float(gt_atm_otm_corr),
                    "atm_itm": float(gt_atm_itm_corr),
                },
            }, checkpoint_path)
            print(f"         → Saved best model (val_loss: {val_metrics['loss']:.6f})")

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Final L_diag: {val_metrics['L_diag_mean']:.3f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print()
    print("Next steps:")
    print("  1. Run validation: python experiments/backfill/two_stage_vae/validate_full_covariance.py")
    print("  2. Check cross-grid correlation improved from 0.02 to ~0.30")
    print("  3. Verify sampling produces correlated grid points")

    # Save training history
    history_path = checkpoint_dir / "two_stage_full_covariance_history.npz"
    np.savez(history_path,
             train_loss=[h["loss"] for h in history["train"]],
             train_mse=[h["mse_loss"] for h in history["train"]],
             train_nll=[h["nll_loss"] for h in history["train"]],
             train_L_diag=[h["L_diag_mean"] for h in history["train"]],
             train_L_offdiag=[h["L_offdiag_mean"] for h in history["train"]],
             val_loss=[h["loss"] for h in history["val"]],
             val_mse=[h["mse_loss"] for h in history["val"]],
             val_nll=[h["nll_loss"] for h in history["val"]])
    print(f"  History: {history_path}")


if __name__ == "__main__":
    main()
