"""
LSTM + Additive Z Pathway Decoder Training Experiment

This experiment trains the CVAETwoStageLSTMDualPath model that uses:
- LSTM pathway for temporal dynamics (volatility clustering, path coherence)
- Additive z pathway for stochastic innovation (fat tails, uncertainty)

    mean = LSTM_out(ctx_emb) + MLP_out(z)

Key insight from research (vae/cvae_two_stage.py lines 3034-3036):
- LSTM+FiLM: 3.8% z contribution, ~60% ctx contribution (washes out z)
- Pure MLP: 39.5% z contribution, 0% ctx contribution (ignores context)
- This LSTM+Additive approach: Forces both pathways to contribute

Expected improvements over MLP decoder:
- LSTM contribution: >20% (was 0% for MLP)
- Z contribution: >25% (was 39.5% for MLP)
- ACF of squared returns: Improved toward GT (volatility clustering)
- Kurtosis recovery: Preserved (>100%)

Training phases:
1. Phase A: Joint training with MSE loss
2. Phase B: Variance training with Student-t NLL

Usage:
    python experiments/backfill/two_stage_vae/exp_lstm_dual_path.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TWO_STAGE_CONFIG
from vae.cvae_two_stage import CVAETwoStageLSTMDualPath


CONTEXT_LEN = 20


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, batch_size, shuffle=True):
    """Create dataloader for training."""
    sequences = []
    for i in range(len(log_returns) - context_len):
        seq = log_returns[i:i + context_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_acf(series, lag=1):
    """Compute autocorrelation at given lag."""
    n = len(series)
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    if var < 1e-10:
        return 0.0
    cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
    return cov / var


def compute_acf_squared_returns(returns, lag=1):
    """Compute ACF of squared returns (volatility clustering measure)."""
    squared = returns ** 2
    return compute_acf(squared, lag)


def evaluate_pathway_contributions(model, val_loader, device):
    """
    Evaluate contribution of each pathway using MSE analysis.

    Returns:
    - mse_full: MSE with both pathways
    - mse_lstm_only: MSE with z=0 (LSTM pathway only)
    - mse_z_only: MSE with LSTM output zeroed (z pathway only)
    - z_contribution: % reduction in MSE from adding z
    - lstm_contribution: % reduction in MSE from adding LSTM
    """
    model.eval()

    mse_full_list = []
    mse_lstm_only_list = []
    mse_z_only_list = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Get embeddings
            ctx_emb = model.ctx_encoder(batch)
            z_mean, z_logvar, z = model.main_encoder(batch)

            target = batch_data[:, 1:]
            B, T, _ = z.shape

            # Full model (LSTM + z)
            mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full = F.mse_loss(pred_full, target).item()
            mse_full_list.append(mse_full)

            # LSTM only (z=0)
            z_zero = torch.zeros_like(z)
            mean_lstm, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
            pred_lstm = mean_lstm[:, :-1]
            mse_lstm_only = F.mse_loss(pred_lstm, target).item()
            mse_lstm_only_list.append(mse_lstm_only)

            # Z only (zeroed LSTM hidden state via zero ctx_emb)
            # Note: This zeros the LSTM input, which effectively gives baseline LSTM output
            ctx_zero = torch.zeros_like(ctx_emb)
            mean_z, _, _, _ = model.decoder(ctx_zero, z, sample=False)
            pred_z = mean_z[:, :-1]
            mse_z_only = F.mse_loss(pred_z, target).item()
            mse_z_only_list.append(mse_z_only)

    mse_full = np.mean(mse_full_list)
    mse_lstm_only = np.mean(mse_lstm_only_list)
    mse_z_only = np.mean(mse_z_only_list)

    # Contributions
    z_contribution = (mse_lstm_only - mse_full) / mse_lstm_only * 100 if mse_lstm_only > 0 else 0
    lstm_contribution = (mse_z_only - mse_full) / mse_z_only * 100 if mse_z_only > 0 else 0

    return {
        "mse_full": mse_full,
        "mse_lstm_only": mse_lstm_only,
        "mse_z_only": mse_z_only,
        "z_contribution_pct": z_contribution,
        "lstm_contribution_pct": lstm_contribution,
    }


def evaluate_kurtosis(model, val_loader, device, n_samples=200):
    """Evaluate kurtosis of generated samples vs GT."""
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:  # Limit for speed
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples // 10)  # (n_samples, B, T, 5, 5)

            # Extract ATM point
            samples_atm = samples[:, :, 1:, 2, 2].cpu().numpy()  # (n_samples, B, T-1)
            gt_atm = batch_data[:, 1:, 2, 2].cpu().numpy()  # (B, T-1)

            all_samples.append(samples_atm.flatten())
            all_gt.append(gt_atm.flatten())

    all_samples = np.concatenate(all_samples)
    all_gt = np.concatenate(all_gt)

    sample_kurt = kurtosis(all_samples, fisher=True)
    gt_kurt = kurtosis(all_gt, fisher=True)

    return {
        "sample_kurtosis": sample_kurt,
        "gt_kurtosis": gt_kurt,
        "kurtosis_recovery_pct": (sample_kurt / gt_kurt * 100) if gt_kurt != 0 else 0,
    }


def evaluate_volatility_clustering(model, val_loader, device, n_samples=100):
    """
    Evaluate volatility clustering (ACF of squared returns).

    This is the key metric that LSTM decoder should improve.
    """
    model.eval()

    sample_acfs = []
    gt_acfs = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples // 10)

            # Compute ACF of squared returns for samples
            samples_atm = samples[:, :, 1:, 2, 2].cpu().numpy()
            for s in range(samples_atm.shape[0]):
                for b in range(samples_atm.shape[1]):
                    acf = compute_acf_squared_returns(samples_atm[s, b, :])
                    if not np.isnan(acf):
                        sample_acfs.append(acf)

            # Compute ACF of squared returns for GT
            gt_atm = batch_data[:, 1:, 2, 2].cpu().numpy()
            for b in range(gt_atm.shape[0]):
                acf = compute_acf_squared_returns(gt_atm[b, :])
                if not np.isnan(acf):
                    gt_acfs.append(acf)

    return {
        "sample_acf_sq_mean": np.mean(sample_acfs) if sample_acfs else 0,
        "gt_acf_sq_mean": np.mean(gt_acfs) if gt_acfs else 0,
        "acf_sq_gap": abs(np.mean(gt_acfs) - np.mean(sample_acfs)) if (gt_acfs and sample_acfs) else float('inf'),
    }


def train_phase_a(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Phase A: Train with MSE loss to learn mean prediction."""
    print("\n" + "=" * 60)
    print("Phase A: MSE Training")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            optimizer.zero_grad()

            # Forward pass
            ctx_emb = model.ctx_encoder(batch)
            z_mean, z_logvar, z = model.main_encoder(batch)
            mean, _, _, _ = model.decoder(ctx_emb, z, sample=False)

            # MSE loss (compare predictions to targets)
            target = batch_data[:, 1:]
            pred = mean[:, :-1]
            loss = F.mse_loss(pred, target)

            # KL loss
            kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            total_loss = loss + 0.1 * kl

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}

                ctx_emb = model.ctx_encoder(batch)
                z_mean, z_logvar, z = model.main_encoder(batch)
                mean, _, _, _ = model.decoder(ctx_emb, z, sample=False)

                target = batch_data[:, 1:]
                pred = mean[:, :-1]
                loss = F.mse_loss(pred, target)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_mse={train_loss:.6f}, val_mse={val_loss:.6f}")

    # Restore best state
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print(f"\n  Best val MSE: {best_val_loss:.6f}")

    # Check pathway contributions
    contrib = evaluate_pathway_contributions(model, val_loader, device)
    print(f"  LSTM contribution: {contrib['lstm_contribution_pct']:.1f}%")
    print(f"  Z contribution: {contrib['z_contribution_pct']:.1f}%")

    return best_val_loss, contrib


def train_phase_b(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    """Phase B: Train variance with Student-t NLL."""
    print("\n" + "=" * 60)
    print("Phase B: Student-t NLL Training")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            optimizer.zero_grad()

            loss_dict = model.compute_loss(batch, kl_weight=0.1)
            total_loss = loss_dict["loss"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss_dict["nll"].item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}

                loss_dict = model.compute_loss(batch, kl_weight=0.1)
                val_losses.append(loss_dict["nll"].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_nll={train_loss:.4f}, val_nll={val_loss:.4f}")

    # Restore best state
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print(f"\n  Best val NLL: {best_val_loss:.4f}")

    return best_val_loss


def main():
    print("=" * 70)
    print("LSTM + Additive Z Pathway Decoder Experiment")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)

    print(f"\nData: {len(log_returns)} days of log-returns")

    # Split data
    n = len(log_returns)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]
    test_data = log_returns[val_end:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create dataloaders
    batch_size = 32
    train_loader = create_dataloader(train_data, CONTEXT_LEN, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, CONTEXT_LEN, batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, CONTEXT_LEN, batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model config
    config = TWO_STAGE_CONFIG.copy()
    config["device"] = str(device)
    config["latent_dim"] = 16
    config["ctx_embedding_dim"] = 3
    config["lstm_hidden"] = 64
    config["lstm_layers"] = 2
    config["cov_rank"] = 4

    print(f"\nConfig: latent_dim={config['latent_dim']}, "
          f"lstm_hidden={config['lstm_hidden']}, lstm_layers={config['lstm_layers']}")

    # Create model
    model = CVAETwoStageLSTMDualPath(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Training
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    # Phase A: MSE
    mse_loss, contrib_a = train_phase_a(model, train_loader, val_loader, device, epochs=50)

    # Phase B: Student-t NLL
    nll_loss = train_phase_b(model, train_loader, val_loader, device, epochs=50)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    # Pathway contributions
    contrib = evaluate_pathway_contributions(model, val_loader, device)
    print(f"\n  Pathway Contributions:")
    print(f"    LSTM contribution: {contrib['lstm_contribution_pct']:.1f}%")
    print(f"    Z contribution:    {contrib['z_contribution_pct']:.1f}%")
    print(f"    MSE (full):        {contrib['mse_full']:.6f}")
    print(f"    MSE (LSTM only):   {contrib['mse_lstm_only']:.6f}")
    print(f"    MSE (Z only):      {contrib['mse_z_only']:.6f}")

    # Kurtosis
    kurt = evaluate_kurtosis(model, val_loader, device)
    print(f"\n  Kurtosis:")
    print(f"    Sample kurtosis:   {kurt['sample_kurtosis']:.2f}")
    print(f"    GT kurtosis:       {kurt['gt_kurtosis']:.2f}")
    print(f"    Recovery:          {kurt['kurtosis_recovery_pct']:.1f}%")

    # Volatility clustering
    vol_clust = evaluate_volatility_clustering(model, val_loader, device)
    print(f"\n  Volatility Clustering (ACF of squared returns):")
    print(f"    Sample ACF:        {vol_clust['sample_acf_sq_mean']:.4f}")
    print(f"    GT ACF:            {vol_clust['gt_acf_sq_mean']:.4f}")
    print(f"    Gap:               {vol_clust['acf_sq_gap']:.4f}")

    # Save model
    save_dir = Path("models/backfill/two_stage/lstm_dual_path")
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "training_results": {
            "phase_a_mse": mse_loss,
            "phase_b_nll": nll_loss,
            "contributions": contrib,
            "kurtosis": kurt,
            "volatility_clustering": vol_clust,
        }
    }, save_dir / "lstm_dual_path_best.pt")

    print(f"\n  Model saved to {save_dir / 'lstm_dual_path_best.pt'}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: LSTM vs MLP Decoder")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'MLP Decoder':<15} {'LSTM Decoder':<15} {'Target':<15}")
    print("-" * 75)
    print(f"{'LSTM/Ctx contribution':<30} {'0%':<15} {contrib['lstm_contribution_pct']:.1f}%{'':<10} {'>20%':<15}")
    print(f"{'Z contribution':<30} {'39.5%':<15} {contrib['z_contribution_pct']:.1f}%{'':<10} {'>25%':<15}")
    print(f"{'Kurtosis recovery':<30} {'135.8%':<15} {kurt['kurtosis_recovery_pct']:.1f}%{'':<10} {'>100%':<15}")
    print(f"{'ACF(sq returns)':<30} {'-0.02':<15} {vol_clust['sample_acf_sq_mean']:.4f}{'':<10} {'~0.39':<15}")

    # Pass/Fail assessment
    lstm_pass = contrib['lstm_contribution_pct'] > 20
    z_pass = contrib['z_contribution_pct'] > 20
    kurt_pass = kurt['kurtosis_recovery_pct'] > 100

    print("\n" + "=" * 70)
    print("Assessment")
    print("=" * 70)
    print(f"\n  {'Criterion':<25} {'Value':<15} {'Target':<15} {'Status':<10}")
    print("-" * 65)
    print(f"  {'LSTM contribution':<25} {contrib['lstm_contribution_pct']:.1f}%{'':<10} {'>20%':<15} {'PASS' if lstm_pass else 'FAIL':<10}")
    print(f"  {'Z contribution':<25} {contrib['z_contribution_pct']:.1f}%{'':<10} {'>20%':<15} {'PASS' if z_pass else 'FAIL':<10}")
    print(f"  {'Kurtosis recovery':<25} {kurt['kurtosis_recovery_pct']:.1f}%{'':<10} {'>100%':<15} {'PASS' if kurt_pass else 'FAIL':<10}")

    all_pass = lstm_pass and z_pass and kurt_pass
    print(f"\n  Overall: {'ALL CRITERIA PASSED' if all_pass else 'SOME CRITERIA FAILED'}")

    return {
        "contributions": contrib,
        "kurtosis": kurt,
        "volatility_clustering": vol_clust,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    results = main()
