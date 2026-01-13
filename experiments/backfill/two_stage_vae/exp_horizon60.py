"""
Horizon=60 Training for Volatility Clustering

Goal: Train with longer horizon (60 days) to force the model to learn
volatility regime persistence for longer autoregressive generation.

Background:
- Volatility clustering is a long-memory phenomenon (full series ACF_sq = +0.39)
- 30-day windows have weak clustering (ACF_sq = +0.054)
- Current model has no clustering (ACF_sq = -0.03)
- Gap is small for 30-day generation, but compounds for longer sequences

Solution: Train with horizon=60 to force learning of longer dependencies.

Usage:
    python experiments/backfill/two_stage_vae/exp_horizon60.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentTMLP
from vae.losses import differentiable_acf
from config.two_stage_config import TWO_STAGE_CONFIG


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, horizon, batch_size, shuffle=True):
    """
    Create dataloader for training with specified horizon.

    Each sequence has: context_len + horizon days
    (1 extra for predicting 1-day-ahead targets)
    """
    seq_len = context_len + horizon
    sequences = []

    for i in range(len(log_returns) - seq_len):
        seq = log_returns[i:i + seq_len + 1]  # +1 for target of last position
        sequences.append(seq)

    if len(sequences) == 0:
        raise ValueError(f"No sequences possible with context_len={context_len}, "
                        f"horizon={horizon}, data_len={len(log_returns)}")

    print(f"  Created {len(sequences)} sequences of length {seq_len + 1}")

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_acf(x: np.ndarray, lag: int = 1) -> float:
    """Compute ACF at given lag."""
    x = x.flatten()
    n = len(x)
    if n <= lag:
        return 0.0
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return 0.0
    cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
    return cov / var


def compute_acf_squared(series: np.ndarray, lag: int = 1) -> float:
    """Compute ACF of squared series (volatility clustering measure)."""
    return compute_acf(series ** 2, lag)


def evaluate_volatility_clustering(model, val_loader, config, n_samples=30):
    """Evaluate volatility clustering via ACF of squared returns."""
    device = config["device"]
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            # Get ATM predictions (grid point 2,2)
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy()  # (n_samples, B, T)
            gt_atm = batch_data[:, :, 2, 2].cpu().numpy()  # (B, T)

            all_samples.append(samples_atm)
            all_gt.append(gt_atm)

    # Compute ACF of squared returns
    sample_acf_sq = []
    for samples_batch in all_samples:
        for s in range(samples_batch.shape[0]):
            for b in range(samples_batch.shape[1]):
                if samples_batch.shape[2] > 5:
                    acf = compute_acf_squared(samples_batch[s, b, :], lag=1)
                    sample_acf_sq.append(acf)

    gt_acf_sq = []
    for gt_batch in all_gt:
        for b in range(gt_batch.shape[0]):
            if gt_batch.shape[1] > 5:
                acf = compute_acf_squared(gt_batch[b, :], lag=1)
                gt_acf_sq.append(acf)

    model_acf = np.mean(sample_acf_sq) if sample_acf_sq else 0
    gt_acf = np.mean(gt_acf_sq) if gt_acf_sq else 0

    return {
        "gt_acf_squared": float(gt_acf),
        "model_acf_squared": float(model_acf),
        "clustering_ratio": float(model_acf / gt_acf) if abs(gt_acf) > 1e-8 else 0,
    }


def evaluate_acf_preservation(model, val_loader, config, n_samples=30):
    """Evaluate ACF preservation of model samples vs ground truth."""
    device = config["device"]
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy()
            gt_atm = batch_data[:, :, 2, 2].cpu().numpy()

            all_samples.append(samples_atm)
            all_gt.append(gt_atm)

    sample_acfs = []
    for samples_batch in all_samples:
        for s in range(samples_batch.shape[0]):
            for b in range(samples_batch.shape[1]):
                if samples_batch.shape[2] > 1:
                    acf = compute_acf(samples_batch[s, b, :], lag=1)
                    sample_acfs.append(acf)

    gt_acfs = []
    for gt_batch in all_gt:
        for b in range(gt_batch.shape[0]):
            if gt_batch.shape[1] > 1:
                acf = compute_acf(gt_batch[b, :], lag=1)
                gt_acfs.append(acf)

    sample_acf_mean = np.mean(sample_acfs) if sample_acfs else 0
    gt_acf_mean = np.mean(gt_acfs) if gt_acfs else 0

    return {
        "gt_acf_lag1": float(gt_acf_mean),
        "model_acf_lag1": float(sample_acf_mean),
        "acf_preservation": float(sample_acf_mean / gt_acf_mean) if abs(gt_acf_mean) > 1e-8 else 0,
    }


def evaluate_kurtosis(model, val_loader, config, n_samples=30):
    """Evaluate kurtosis recovery."""
    device = config["device"]
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy().flatten()
            gt_atm = batch_data[:, :, 2, 2].cpu().numpy().flatten()

            all_samples.extend(samples_atm)
            all_gt.extend(gt_atm)

    sample_kurt = kurtosis(all_samples, fisher=True)
    gt_kurt = kurtosis(all_gt, fisher=True)

    return {
        "gt_kurtosis": float(gt_kurt),
        "model_kurtosis": float(sample_kurt),
        "kurtosis_recovery": float(sample_kurt / gt_kurt * 100) if abs(gt_kurt) > 1e-8 else 0,
    }


def train_horizon60(
    model, train_loader, val_loader, config,
    epochs=100, lambda_acf=0.0
):
    """
    Train Student-t model with horizon=60.

    Unlike exp_student_t_acf.py, this focuses on longer horizon training
    without explicit ACF loss - the model should learn temporal structure
    naturally from having to predict 60 days ahead.
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    best_loss = float('inf')
    best_state = None

    print(f"\n{'='*70}")
    print(f"Training Student-t with Horizon=60")
    print(f"Epochs: {epochs}, λ_acf: {lambda_acf}")
    print(f"{'='*70}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_nlls = []
        train_acfs = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Forward pass
            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            # Target and prediction (shifted by 1)
            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # NLL loss (Student-t)
            nll_loss = model.decoder.compute_student_t_nll(pred, target, factor, log_diag)

            # KL loss
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            # Optional ACF loss
            if lambda_acf > 0:
                pred_atm = pred[:, :, 2, 2]
                target_atm = target[:, :, 2, 2]

                batch_acf_losses = []
                for b in range(pred_atm.shape[0]):
                    if pred_atm.shape[1] > 1:
                        pred_acf = differentiable_acf(pred_atm[b], lag=1, dim=0)
                        target_acf = differentiable_acf(target_atm[b], lag=1, dim=0)
                        acf_diff = (pred_acf - target_acf) ** 2
                        batch_acf_losses.append(acf_diff)

                if batch_acf_losses:
                    acf_loss = torch.stack(batch_acf_losses).mean()
                else:
                    acf_loss = torch.tensor(0.0, device=device)
            else:
                acf_loss = torch.tensor(0.0, device=device)

            # Combined loss
            total_loss = nll_loss + kl_weight * kl_loss + lambda_acf * acf_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())
            train_nlls.append(nll_loss.item())
            train_acfs.append(acf_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}

                mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)
                target = batch_data[:, 1:]
                pred = mean[:, :-1]

                nll_loss = model.decoder.compute_student_t_nll(pred, target, factor, log_diag)
                val_losses.append(nll_loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"NLL={np.mean(train_nlls):.4f}, "
                  f"Val={val_loss:.4f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

            # Mid-training evaluation of volatility clustering
            if (epoch + 1) % 50 == 0:
                clustering = evaluate_volatility_clustering(model, val_loader, config, n_samples=10)
                print(f"    Vol. Clustering: GT={clustering['gt_acf_squared']:.3f}, "
                      f"Model={clustering['model_acf_squared']:.3f}, "
                      f"Ratio={clustering['clustering_ratio']*100:.1f}%")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    """Train and evaluate horizon=60 model."""
    print("=" * 70)
    print("Horizon=60 Training for Volatility Clustering")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)

    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]

    print(f"Train: {len(train_data)} days")
    print(f"Val: {len(val_data)} days")

    # Configuration
    context_len = 30
    horizon = 60  # KEY CHANGE: 60 instead of 30
    batch_size = 32  # Reduced for memory

    print(f"\nConfiguration:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  Sequence length: {context_len + horizon + 1} = {context_len + horizon + 1} days")
    print(f"  Batch size: {batch_size}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(train_data, context_len, horizon, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, context_len, horizon, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Model config - update horizon
    config = TWO_STAGE_CONFIG.copy()
    config["device"] = device
    config["horizon"] = horizon
    config["max_horizon"] = horizon
    config["context_len"] = context_len

    # Create model
    print("\nCreating model...")
    model = CVAETwoStageStudentTMLP(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    model = train_horizon60(
        model, train_loader, val_loader, config,
        epochs=100, lambda_acf=0.0
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Volatility clustering (main metric)
    clustering = evaluate_volatility_clustering(model, val_loader, config, n_samples=50)
    print(f"\nVolatility Clustering (ACF of squared returns):")
    print(f"  GT (60-day windows):    {clustering['gt_acf_squared']:.4f}")
    print(f"  Model:                  {clustering['model_acf_squared']:.4f}")
    print(f"  Clustering Ratio:       {clustering['clustering_ratio']*100:.1f}%")

    # ACF preservation
    acf_results = evaluate_acf_preservation(model, val_loader, config, n_samples=50)
    print(f"\nACF Preservation (lag-1):")
    print(f"  GT:          {acf_results['gt_acf_lag1']:.4f}")
    print(f"  Model:       {acf_results['model_acf_lag1']:.4f}")
    print(f"  Preservation: {acf_results['acf_preservation']*100:.1f}%")

    # Kurtosis
    kurt_results = evaluate_kurtosis(model, val_loader, config, n_samples=50)
    print(f"\nKurtosis Recovery:")
    print(f"  GT:       {kurt_results['gt_kurtosis']:.2f}")
    print(f"  Model:    {kurt_results['model_kurtosis']:.2f}")
    print(f"  Recovery: {kurt_results['kurtosis_recovery']:.1f}%")

    # Save model
    save_path = Path("models/backfill/two_stage/horizon60")
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_path / "student_t_horizon60.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "metrics": {
            "clustering": clustering,
            "acf": acf_results,
            "kurtosis": kurt_results,
        },
    }, checkpoint_path)

    print(f"\nModel saved to: {checkpoint_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Horizon:             {horizon} days")
    print(f"Vol. Clustering:     {clustering['clustering_ratio']*100:.1f}% of GT")
    print(f"ACF Preservation:    {acf_results['acf_preservation']*100:.1f}%")
    print(f"Kurtosis Recovery:   {kurt_results['kurtosis_recovery']:.1f}%")

    return model, config


if __name__ == "__main__":
    main()
