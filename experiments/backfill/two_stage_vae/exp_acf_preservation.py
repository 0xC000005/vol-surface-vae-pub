"""
ACF Preservation Experiment

This experiment trains the CVAETwoStageDualPathAR model which combines:
1. Dual-path additive decoder (preserves ctx + z contribution)
2. AR(1) mean-reversion component (preserves temporal structure)
3. Spectral loss (matches power spectrum / ACF)
4. Student-t sampling (preserves fat tails / kurtosis)

Target metrics:
- ACF Preservation: >30% (up from 15-19%)
- Kurtosis Recovery: >100% (must not regress from 134%)
- Ctx Contribution: >5% (must not regress from 16%)
- Z Contribution: >20% (must not regress from 39%)

Usage:
    python experiments/backfill/two_stage_vae/exp_acf_preservation.py
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TWO_STAGE_CONFIG
from vae.cvae_two_stage import CVAETwoStageDualPathAR
from vae.losses import spectral_loss, acf_loss, combined_temporal_loss


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


def evaluate_bottleneck(model, val_loader, device):
    """Evaluate pathway contributions."""
    model.eval()

    mse_full = []
    mse_ctx_only = []
    mse_z_only = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            ctx_emb = model.ctx_encoder(batch)
            z_mean, z_logvar, z = model.main_encoder(batch)

            target = batch_data[:, 1:]
            B, T, _ = z.shape

            # Full model
            prev_x = batch_data.clone()
            mean_full, _, _, _ = model.decoder(ctx_emb, z, prev_x=prev_x, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full.append(((pred_full - target) ** 2).mean().item())

            # Ctx only (z=0)
            z_zero = torch.zeros_like(z)
            mean_ctx, _, _, _ = model.decoder(ctx_emb, z_zero, prev_x=prev_x, sample=False)
            pred_ctx = mean_ctx[:, :-1]
            mse_ctx_only.append(((pred_ctx - target) ** 2).mean().item())

            # Z only (ctx=0)
            ctx_zero = torch.zeros_like(ctx_emb)
            mean_z, _, _, _ = model.decoder(ctx_zero, z, prev_x=prev_x, sample=False)
            pred_z = mean_z[:, :-1]
            mse_z_only.append(((pred_z - target) ** 2).mean().item())

    mse_full_avg = np.mean(mse_full)
    mse_ctx_avg = np.mean(mse_ctx_only)
    mse_z_avg = np.mean(mse_z_only)

    z_contribution = (mse_ctx_avg - mse_full_avg) / mse_ctx_avg * 100 if mse_ctx_avg > 0 else 0
    ctx_contribution = (mse_z_avg - mse_full_avg) / mse_z_avg * 100 if mse_z_avg > 0 else 0

    return {
        "mse_full": mse_full_avg,
        "mse_ctx_only": mse_ctx_avg,
        "mse_z_only": mse_z_avg,
        "z_contribution_pct": z_contribution,
        "ctx_contribution_pct": ctx_contribution,
    }


def evaluate_kurtosis(model, val_loader, device, n_samples=200):
    """Evaluate kurtosis recovery."""
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            target = batch_data[:, 1:]

            samples_last = samples[:, :, -1, :, :].cpu().numpy()
            gt_last = target[:, -1, :, :].cpu().numpy()

            all_samples.append(samples_last.reshape(-1, 5, 5))
            all_gt.append(gt_last)

    all_samples = np.concatenate(all_samples, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    gt_kurtosis = np.zeros((5, 5))
    model_kurtosis = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_kurtosis[i, j] = kurtosis(all_gt[:, i, j], fisher=True)
            model_kurtosis[i, j] = kurtosis(all_samples[:, i, j], fisher=True)

    recovery = np.abs(model_kurtosis) / np.abs(gt_kurtosis + 1e-8)
    recovery = np.clip(recovery, 0, 2)

    return {
        "mean_recovery": float(recovery.mean()),
        "atm_gt_kurtosis": float(gt_kurtosis[2, 2]),
        "atm_model_kurtosis": float(model_kurtosis[2, 2]),
    }


def evaluate_acf(model, val_loader, device, n_samples=100):
    """Evaluate ACF preservation."""
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 10:
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            target = batch_data[:, 1:]

            # ATM point across all timesteps
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy()
            gt_atm = target[:, :, 2, 2].cpu().numpy()

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

    preservation = sample_acf_mean / gt_acf_mean if abs(gt_acf_mean) > 1e-8 else 0

    return {
        "gt_acf_lag1": float(gt_acf_mean),
        "model_acf_lag1": float(sample_acf_mean),
        "acf_preservation": float(preservation),
    }


def train_acf_model(model, train_loader, val_loader, config, epochs=100):
    """
    Train the dual-path AR(1) model with temporal losses.

    Training phases:
    1. Phase A (40%): MSE + KL (basic reconstruction)
    2. Phase B (30%): Add spectral loss (frequency matching)
    3. Phase C (30%): Add Student-t NLL (variance calibration)
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)
    spectral_weight = config.get("spectral_weight", 0.1)
    acf_weight = config.get("acf_weight", 0.05)

    phase_a_epochs = int(epochs * 0.4)
    phase_b_epochs = int(epochs * 0.3)
    phase_c_epochs = epochs - phase_a_epochs - phase_b_epochs

    # ========================================
    # PHASE A: Basic MSE + KL
    # ========================================
    print(f"\n--- Phase A: MSE + KL ({phase_a_epochs} epochs) ---")

    optimizer_a = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(phase_a_epochs):
        model.train()
        train_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            mse_loss = F.mse_loss(pred, target)
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            loss = mse_loss + kl_weight * kl_loss

            optimizer_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_a.step()

            train_losses.append(mse_loss.item())

        if (epoch + 1) % 10 == 0:
            phi = model.get_ar_phi().item()
            print(f"    Epoch {epoch+1}/{phase_a_epochs}: MSE={np.mean(train_losses):.6f}, phi={phi:.3f}")

    # ========================================
    # PHASE B: Add spectral loss
    # ========================================
    print(f"\n--- Phase B: MSE + Spectral Loss ({phase_b_epochs} epochs) ---")

    optimizer_b = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(phase_b_epochs):
        model.train()
        train_losses = []
        spectral_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            mse_loss = F.mse_loss(pred, target)
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            # Spectral loss at ATM point
            pred_atm = pred[:, :, 2, 2]
            target_atm = target[:, :, 2, 2]
            spec_loss = spectral_loss(pred_atm, target_atm, dim=1)

            loss = mse_loss + kl_weight * kl_loss + spectral_weight * spec_loss

            optimizer_b.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            train_losses.append(mse_loss.item())
            spectral_losses.append(spec_loss.item())

        if (epoch + 1) % 10 == 0:
            phi = model.get_ar_phi().item()
            acf_result = evaluate_acf(model, val_loader, device, n_samples=50)
            print(f"    Epoch {epoch+1}/{phase_b_epochs}: MSE={np.mean(train_losses):.6f}, "
                  f"Spectral={np.mean(spectral_losses):.6f}, phi={phi:.3f}, "
                  f"ACF={acf_result['acf_preservation']*100:.1f}%")

    # ========================================
    # PHASE C: Student-t NLL for variance
    # ========================================
    print(f"\n--- Phase C: Student-t NLL ({phase_c_epochs} epochs) ---")

    # Freeze mean networks, only train covariance
    for param in model.ctx_encoder.parameters():
        param.requires_grad = False
    for param in model.main_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.ctx_mean_net.parameters():
        param.requires_grad = False
    for param in model.decoder.z_residual_net.parameters():
        param.requires_grad = False

    optimizer_c = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    for epoch in range(phase_c_epochs):
        model.train()
        train_nlls = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            nll_loss = model.decoder.compute_student_t_nll(pred, target, factor, log_diag)

            optimizer_c.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_c.step()

            train_nlls.append(nll_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_c_epochs}: NLL={np.mean(train_nlls):.4f}")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    return model


def run_experiment(epochs=100, save_model=True):
    """Run ACF preservation experiment."""
    print("=" * 70)
    print("ACF PRESERVATION EXPERIMENT")
    print("=" * 70)
    print("\nArchitecture: Dual-Path + AR(1) + Spectral Loss")
    print("Goal: Improve ACF from 15-19% to >30% without regressing kurtosis")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)
    print(f"Data shape: {log_returns.shape}")

    # Config
    config = TWO_STAGE_CONFIG.copy()
    config["latent_dim"] = 8
    config["ctx_embedding_dim"] = 3
    config["cov_rank"] = 4
    config["kl_weight"] = 0.001
    config["spectral_weight"] = 0.1
    config["acf_weight"] = 0.05
    config["learn_ar_phi"] = True
    config["target_ar_phi"] = -0.35
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfig:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  spectral_weight: {config['spectral_weight']}")
    print(f"  learn_ar_phi: {config['learn_ar_phi']}")
    print(f"  target_ar_phi: {config['target_ar_phi']}")
    print(f"  device: {config['device']}")

    # Create dataloaders
    n_train = int(len(log_returns) * 0.8)
    train_loader = create_dataloader(log_returns[:n_train], CONTEXT_LEN, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], CONTEXT_LEN, batch_size=32, shuffle=False)

    # For fair comparison, also evaluate on full dataset
    full_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)

    # Create model
    model = CVAETwoStageDualPathAR(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {n_params:,} parameters")

    # Train
    model = train_acf_model(model, train_loader, val_loader, config, epochs=epochs)

    # ========================================
    # EVALUATION (using full dataset for fair comparison)
    # ========================================
    print("\n" + "=" * 70)
    print("EVALUATION (Full Dataset)")
    print("=" * 70)

    device = config["device"]

    # 1. Bottleneck
    print("\n--- Pathway Contributions ---")
    bottleneck = evaluate_bottleneck(model, full_loader, device)
    print(f"  MSE (full):       {bottleneck['mse_full']:.6f}")
    print(f"  Z Contribution:   {bottleneck['z_contribution_pct']:.1f}%")
    print(f"  Ctx Contribution: {bottleneck['ctx_contribution_pct']:.1f}%")

    # 2. Kurtosis
    print("\n--- Kurtosis Recovery ---")
    print("Generating samples (may take a minute)...")
    kurt = evaluate_kurtosis(model, full_loader, device, n_samples=200)
    print(f"  GT Kurtosis (ATM):    {kurt['atm_gt_kurtosis']:.2f}")
    print(f"  Model Kurtosis (ATM): {kurt['atm_model_kurtosis']:.2f}")
    print(f"  Mean Recovery:        {kurt['mean_recovery']*100:.1f}%")

    # 3. ACF
    print("\n--- ACF Preservation ---")
    acf_result = evaluate_acf(model, full_loader, device, n_samples=100)
    print(f"  GT ACF lag-1:    {acf_result['gt_acf_lag1']:.4f}")
    print(f"  Model ACF lag-1: {acf_result['model_acf_lag1']:.4f}")
    print(f"  Preservation:    {acf_result['acf_preservation']*100:.1f}%")

    # 4. AR(1) coefficient
    print("\n--- AR(1) Coefficient ---")
    phi = model.get_ar_phi().item()
    print(f"  Learned φ: {phi:.4f}")
    print(f"  Target φ:  {config['target_ar_phi']:.4f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        "bottleneck": bottleneck,
        "kurtosis": {
            "mean_recovery": kurt['mean_recovery'],
            "atm_gt": kurt['atm_gt_kurtosis'],
            "atm_model": kurt['atm_model_kurtosis'],
        },
        "acf": acf_result,
        "ar_phi": phi,
    }

    # Check pass criteria
    acf_pass = abs(acf_result['acf_preservation']) > 0.3
    kurt_pass = kurt['mean_recovery'] > 1.0
    ctx_pass = bottleneck['ctx_contribution_pct'] > 5
    z_pass = bottleneck['z_contribution_pct'] > 20

    print(f"\n  {'Metric':<20} {'Value':>12} {'Target':>12} {'Status':>10}")
    print("  " + "-" * 56)
    print(f"  {'ACF Preservation':<20} {acf_result['acf_preservation']*100:>11.1f}% {'>30%':>12} {'PASS' if acf_pass else 'FAIL':>10}")
    print(f"  {'Kurtosis Recovery':<20} {kurt['mean_recovery']*100:>11.1f}% {'>100%':>12} {'PASS' if kurt_pass else 'FAIL':>10}")
    print(f"  {'Ctx Contribution':<20} {bottleneck['ctx_contribution_pct']:>11.1f}% {'>5%':>12} {'PASS' if ctx_pass else 'FAIL':>10}")
    print(f"  {'Z Contribution':<20} {bottleneck['z_contribution_pct']:>11.1f}% {'>20%':>12} {'PASS' if z_pass else 'FAIL':>10}")

    results["pass_criteria"] = {
        "acf": bool(acf_pass),
        "kurtosis": bool(kurt_pass),
        "ctx_contribution": bool(ctx_pass),
        "z_contribution": bool(z_pass),
    }

    overall_pass = acf_pass and kurt_pass and ctx_pass and z_pass

    print("\n" + "=" * 70)
    if overall_pass:
        print("OVERALL: SUCCESS! All criteria met!")
    elif kurt_pass:
        print("OVERALL: PARTIAL SUCCESS - Kurtosis preserved")
        if not acf_pass:
            print(f"  - ACF needs improvement: {acf_result['acf_preservation']*100:.1f}% < 30%")
    else:
        print("OVERALL: FAIL - Kurtosis regressed")
    print("=" * 70)

    # Save model
    if save_model:
        save_dir = Path("models/backfill/two_stage/dual_path_ar")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "dual_path_ar_best.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "results": results,
        }
        torch.save(checkpoint, save_path)
        print(f"\nModel saved to: {save_path}")

        # Save results JSON
        results_path = save_dir / "dual_path_ar_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

    return model, results


if __name__ == "__main__":
    model, results = run_experiment(epochs=100, save_model=True)
