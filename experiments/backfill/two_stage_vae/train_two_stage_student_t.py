"""
Train Two-Stage CVAE with Multivariate Student-t Decoder on Log-Returns.

This script trains the CVAETwoStageStudentT model which uses:
1. Log-return transformation (stationary time series)
2. Full Cholesky covariance (correlated sampling)
3. Per-grid-point Student-t distribution (25 learnable nu parameters)
4. Combined MSE + Multivariate Student-t NLL loss

Key Features:
- Per-grid-point nu: 25 learnable degrees-of-freedom parameters (one per grid point)
- Allows heterogeneous tail heaviness across the volatility surface
- GT kurtosis varies from 3 to 180 across grid - single nu cannot capture this
- Sampling: x = μ + L @ ε / √u where u_i ~ Gamma(ν_i/2, ν_i/2)
- Student-t NLL encourages both correlation AND per-point fat tail learning

Issues Addressed:
1. Fat tails missing (kurtosis 1.4 vs GT 21.3)
2. Heterogeneous kurtosis across grid (GT varies 3-180)
3. Cross-grid correlation weakly learned (0.18 vs GT 0.21-0.42)

Usage:
    python experiments/backfill/two_stage_vae/train_two_stage_student_t.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageStudentT
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


def compute_gt_excess_kurtosis_per_grid(data):
    """
    Compute excess kurtosis (Fisher's definition) for each of 25 grid points.

    This is used for theoretical kurtosis loss supervision:
    - For Student-t: excess_kurtosis = 6 / (nu - 4)
    - We supervise nu to match GT kurtosis via this formula

    Args:
        data: (N, 5, 5) array of log-returns

    Returns:
        excess_kurtosis: (25,) array of excess kurtosis per grid point
    """
    from scipy import stats
    data_flat = data.reshape(-1, 25)  # (N, 25)
    excess_kurtosis = np.array([
        stats.kurtosis(data_flat[:, i]) for i in range(25)
    ])
    return excess_kurtosis


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def train_epoch(model, train_sequences, optimizer, batch_size, device):
    """Train for one epoch with Student-t loss."""
    model.train()
    metrics = {
        "loss": 0, "mse_loss": 0, "nll_loss": 0, "kurt_loss": 0,
        "kl_loss": 0, "nu_mean": 0, "nu_min": 0, "nu_max": 0, "nu_std": 0,
        "L_diag_mean": 0, "L_offdiag_mean": 0
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

        kurt_loss_val = losses.get('kurt_loss', torch.tensor(0.0))
        kurt_loss_float = kurt_loss_val.item() if torch.is_tensor(kurt_loss_val) else kurt_loss_val
        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "mse": f"{losses['mse_loss'].item():.6f}",
            "nll": f"{losses['nll_loss'].item():.4f}",
            "kurt": f"{kurt_loss_float:.2f}",
            "nu": f"{losses['nu_mean']:.2f}[{losses['nu_min']:.1f}-{losses['nu_max']:.1f}]",
            "L_diag": f"{losses['L_diag_mean'].item():.3f}",
            "L_off": f"{losses['L_offdiag_mean'].item():.4f}"
        })

    return {k: v / n_batches for k, v in metrics.items()}


def validate(model, val_sequences, batch_size, device):
    """Validate model with Student-t metrics."""
    model.eval()
    metrics = {
        "loss": 0, "mse_loss": 0, "nll_loss": 0,
        "kl_loss": 0, "nu_mean": 0, "nu_min": 0, "nu_max": 0, "nu_std": 0, "L_diag_mean": 0
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
    print("Two-Stage CVAE Student-t Decoder Training (Per-Grid-Point nu)")
    print("=" * 70)
    print()
    print("Training on LOG-RETURNS with PER-GRID-POINT STUDENT-T decoder.")
    print("Key feature: 25 learnable nu parameters (one per grid point)")
    print()
    print("This fixes:")
    print("  1. Fat tails missing (via per-point degrees-of-freedom nu)")
    print("  2. Heterogeneous kurtosis (GT varies 3-180 across grid)")
    print("  3. Cross-grid correlation weak (via higher NLL weight)")
    print()
    print("Issues being addressed:")
    print("  #1: Fat tails missing - kurtosis 4.9 vs GT 14.4 (ATM)")
    print("  #2: Heterogeneous kurtosis - GT varies 3-180, single nu gives uniform ~5-6")
    print("  #3: Cross-grid correlation - 0.26 vs GT 0.21 (already good)")
    print()

    # Get base config and modify for Student-t
    config = TwoStageConfig.get_model_config()

    # Student-t specific config
    config["student_t"] = True
    config["nu_floor"] = 2.1            # nu > 2 for finite variance
    config["nu_max"] = 100.0            # Prevent collapse to Gaussian
    config["nu_init"] = 5.0             # Initial nu, kurtosis ~ 9
    config["mse_weight"] = 1.0          # Weight for mean accuracy
    config["nll_weight"] = 1.0          # HIGHER than Full Cov (0.1) for correlation
    config["kurtosis_loss_weight"] = 0.0  # Disable kurtosis loss when using fixed nu
    config["cholesky_diag_floor"] = 1e-3
    config["cholesky_diag_init"] = -2.0
    config["decoder_mem_hidden"] = 32   # Need capacity for Cholesky outputs

    # Fix nu from GT kurtosis (literature-recommended approach)
    # Learning nu via gradient descent is fundamentally difficult
    config["learn_nu"] = False          # Fixed nu, not learned

    # Training parameters
    batch_size = 128
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
    print(f"Per-Grid-Point Student-t Config:")
    print(f"  nu parameters: 25 (one per grid point)")
    print(f"  Initial nu (all): {config['nu_init']} (kurtosis ~ 9 for nu > 4)")
    print(f"  nu_floor: {config['nu_floor']} (nu > 2 for finite variance)")
    print(f"  nu_max: {config['nu_max']} (prevents Gaussian collapse)")
    print(f"  MSE Weight: {config['mse_weight']}")
    print(f"  NLL Weight: {config['nll_weight']} (10x higher than Full Cov!)")
    print(f"  KL Weight: {config['kl_weight']}")
    print(f"  Cholesky parameters: 325 (25x26/2)")
    print(f"  Total nu + Cholesky params: 25 + 325 = 350")

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

    # Compute GT statistics for reference
    from scipy import stats
    log_returns_flat = log_returns.reshape(-1, 25)
    gt_kurtosis = stats.kurtosis(log_returns_flat[:, 12])  # ATM (2,2) = index 12
    gt_corr_matrix = np.corrcoef(log_returns_flat.T)
    gt_atm_otm_corr = gt_corr_matrix[12, 0]  # ATM (2,2) vs OTM-short (0,0)
    gt_atm_itm_corr = gt_corr_matrix[12, 24]  # ATM (2,2) vs ITM-long (4,4)
    print(f"  GT Kurtosis (ATM): {gt_kurtosis:.2f}")
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

    # Compute GT excess kurtosis for theoretical kurtosis loss
    print("\nComputing GT excess kurtosis per grid point...")
    gt_excess_kurtosis = compute_gt_excess_kurtosis_per_grid(log_returns)
    gt_excess_kurtosis_grid = gt_excess_kurtosis.reshape(5, 5)
    print(f"  GT excess kurtosis (5x5 grid):")
    print(f"    min={gt_excess_kurtosis.min():.2f}, max={gt_excess_kurtosis.max():.2f}, "
          f"mean={gt_excess_kurtosis.mean():.2f}")
    print(f"    ATM [2,2]: {gt_excess_kurtosis_grid[2, 2]:.2f}")
    print(f"    Corners: [{gt_excess_kurtosis_grid[0,0]:.1f}, {gt_excess_kurtosis_grid[0,4]:.1f}, "
          f"{gt_excess_kurtosis_grid[4,0]:.1f}, {gt_excess_kurtosis_grid[4,4]:.1f}]")

    # Build model
    print("\nBuilding Student-t model...")
    device = config["device"]
    model = CVAETwoStageStudentT(config)

    # Fix nu from GT kurtosis via method of moments (literature-recommended approach)
    # This directly sets nu = 4 + 6/excess_kurtosis (heterogeneous per grid point)
    model.fix_nu_from_kurtosis(gt_excess_kurtosis)
    print(f"  learn_nu: {config['learn_nu']} (nu {'IS' if config['learn_nu'] else 'is NOT'} trainable)")
    print(f"  Kurtosis loss weight: {config['kurtosis_loss_weight']}")

    print(f"  Device: {device}")
    print(f"  Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Count decoder-specific parameters
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    cholesky_params = sum(p.numel() for p in model.decoder.cholesky_head.parameters())
    nu_params = model.decoder.nu_raw.numel()
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Cholesky head parameters: {cholesky_params:,}")
    print(f"  nu parameters: {nu_params} (per-grid-point, FIXED from GT kurtosis)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Checkpoint path (use fixed_nu suffix to distinguish from learned-nu version)
    checkpoint_dir = Path(TwoStageConfig.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "two_stage_student_t_fixed_nu_best.pt"

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

        # Print progress with per-grid-point nu statistics and kurtosis loss
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train: loss={train_metrics['loss']:.4f} mse={train_metrics['mse_loss']:.6f} "
              f"nll={train_metrics['nll_loss']:.4f} kurt={train_metrics.get('kurt_loss', 0):.2f} "
              f"nu={train_metrics['nu_mean']:.2f}[{train_metrics['nu_min']:.1f}-{train_metrics['nu_max']:.1f}] "
              f"L_diag={train_metrics['L_diag_mean']:.3f} L_off={train_metrics['L_offdiag_mean']:.4f} | "
              f"Val: loss={val_metrics['loss']:.4f} nu={val_metrics['nu_mean']:.2f}[{val_metrics['nu_min']:.1f}-{val_metrics['nu_max']:.1f}]")

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
                # Model-specific
                "use_log_returns": True,
                "student_t": True,
                "per_grid_nu": True,  # NEW: per-grid-point nu (25 params)
                "nu_mean": val_metrics["nu_mean"],
                "nu_min": val_metrics["nu_min"],
                "nu_max": val_metrics["nu_max"],
                "nu_std": val_metrics["nu_std"],
                "log_return_stats": {
                    "mean": float(log_returns.mean()),
                    "std": float(log_return_std),
                    "var": float(log_return_var),
                    "min": float(log_returns.min()),
                    "max": float(log_returns.max()),
                },
                "gt_statistics": {
                    "kurtosis_atm": float(gt_kurtosis),
                    "atm_otm_corr": float(gt_atm_otm_corr),
                    "atm_itm_corr": float(gt_atm_itm_corr),
                },
                "gt_excess_kurtosis": gt_excess_kurtosis,  # (25,) for kurtosis loss supervision
                "kurtosis_loss_weight": config["kurtosis_loss_weight"],
            }, checkpoint_path)
            print(f"         → Saved best model (val_loss: {val_metrics['loss']:.6f}, nu_mean: {val_metrics['nu_mean']:.2f}[{val_metrics['nu_min']:.1f}-{val_metrics['nu_max']:.1f}])")

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Final Per-Grid nu: mean={val_metrics['nu_mean']:.2f}, min={val_metrics['nu_min']:.2f}, max={val_metrics['nu_max']:.2f}")
    print(f"  Final L_diag: {val_metrics['L_diag_mean']:.3f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print()
    print("Per-grid-point nu allows heterogeneous kurtosis:")
    print("  nu ~ 2.1 (floor): kurtosis ~ 30+ (very heavy tails)")
    print("  nu ~ 5: kurtosis ~ 9 (heavy tails)")
    print("  nu ~ 10: kurtosis ~ 6 (moderate tails)")
    print("  nu ~ 30+: approaching Gaussian (kurtosis ~ 3)")
    print()
    print("Next steps:")
    print("  1. Run validation: python experiments/backfill/two_stage_vae/validate_student_t.py")
    print("  2. Check per-grid-point kurtosis matches GT heterogeneity (3-180)")
    print("  3. Check nu_min < 5 for high-kurtosis grid points")

    # Save training history with per-grid nu stats
    history_path = checkpoint_dir / "two_stage_student_t_fixed_nu_history.npz"
    np.savez(history_path,
             train_loss=[h["loss"] for h in history["train"]],
             train_mse=[h["mse_loss"] for h in history["train"]],
             train_nll=[h["nll_loss"] for h in history["train"]],
             train_nu_mean=[h["nu_mean"] for h in history["train"]],
             train_nu_min=[h["nu_min"] for h in history["train"]],
             train_nu_max=[h["nu_max"] for h in history["train"]],
             train_nu_std=[h["nu_std"] for h in history["train"]],
             train_L_diag=[h["L_diag_mean"] for h in history["train"]],
             train_L_offdiag=[h["L_offdiag_mean"] for h in history["train"]],
             val_loss=[h["loss"] for h in history["val"]],
             val_mse=[h["mse_loss"] for h in history["val"]],
             val_nll=[h["nll_loss"] for h in history["val"]],
             val_nu_mean=[h["nu_mean"] for h in history["val"]],
             val_nu_min=[h["nu_min"] for h in history["val"]],
             val_nu_max=[h["nu_max"] for h in history["val"]],
             val_nu_std=[h["nu_std"] for h in history["val"]])
    print(f"  History: {history_path}")


if __name__ == "__main__":
    main()
