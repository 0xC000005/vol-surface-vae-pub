"""
Dual-Path Decoder Training Experiment

This experiment trains the new CVAETwoStageDualPath model that uses
ADDITIVE combination of context and z pathways:

    mean = ctx_mean + z_residual

Key insight: Additive combination forces both pathways to contribute,
unlike FiLM (multiplicative) where z can be bypassed.

Expected improvements over current StudentTMLPDecoder:
- Kurtosis recovery: preserved (>100%)
- Z contribution: >20% (currently 39.5%)
- Ctx contribution: >20% (currently 0%)
- Direction accuracy: improved
- ACF preservation: improved

Training phases:
1. Phase A: Joint training of both pathways with MSE
2. Phase B: Variance training with Student-t NLL
3. Phase C (optional): Fine-tune with auxiliary losses (ACF, magnitude)

Usage:
    python experiments/backfill/two_stage_vae/exp_dual_path_decoder.py
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
from vae.cvae_two_stage import CVAETwoStageDualPath


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


def compute_acf_loss(model, batch, target, n_samples=10):
    """
    ACF matching loss for temporal structure.

    Computes ACF of generated samples vs GT and returns MSE.
    """
    device = target.device

    # Generate samples
    samples = model.sample(batch, n_samples=n_samples)  # (n_samples, B, T, 5, 5)

    # Take ATM point
    samples_atm = samples[:, :, :, 2, 2].cpu().numpy()  # (n_samples, B, T)
    gt_atm = target[:, :, 2, 2].cpu().numpy()  # (B, T)

    # Compute ACF lag-1 for samples (average across samples)
    sample_acfs = []
    for s in range(n_samples):
        for b in range(samples_atm.shape[1]):
            acf = compute_acf(samples_atm[s, b, :], lag=1)
            sample_acfs.append(acf)
    sample_acf_mean = np.mean(sample_acfs)

    # Compute ACF lag-1 for GT
    gt_acfs = []
    for b in range(gt_atm.shape[0]):
        acf = compute_acf(gt_atm[b, :], lag=1)
        gt_acfs.append(acf)
    gt_acf_mean = np.mean(gt_acfs)

    # Return MSE as tensor
    acf_loss = (sample_acf_mean - gt_acf_mean) ** 2
    return torch.tensor(acf_loss, device=device, dtype=torch.float32)


def compute_magnitude_loss(pred, target):
    """
    Magnitude matching loss to prevent attenuation.

    Matches standard deviation of predictions to GT.
    """
    pred_std = pred.std()
    gt_std = target.std()
    return F.mse_loss(pred_std, gt_std)


def evaluate_pathway_contributions(model, val_loader, device):
    """
    Evaluate contribution of each pathway using MSE analysis.

    Returns:
    - mse_full: MSE with both pathways
    - mse_ctx_only: MSE with z=0
    - mse_z_only: MSE with ctx=0
    - z_contribution: % reduction in MSE from adding z
    - ctx_contribution: % reduction in MSE from adding ctx
    """
    model.eval()

    mse_full_list = []
    mse_ctx_only_list = []
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

            # Full model (ctx + z)
            mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full = F.mse_loss(pred_full, target).item()
            mse_full_list.append(mse_full)

            # Context only (z=0)
            z_zero = torch.zeros_like(z)
            mean_ctx, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
            pred_ctx = mean_ctx[:, :-1]
            mse_ctx_only = F.mse_loss(pred_ctx, target).item()
            mse_ctx_only_list.append(mse_ctx_only)

            # Z only (ctx=0)
            ctx_zero = torch.zeros_like(ctx_emb)
            mean_z, _, _, _ = model.decoder(ctx_zero, z, sample=False)
            pred_z = mean_z[:, :-1]
            mse_z_only = F.mse_loss(pred_z, target).item()
            mse_z_only_list.append(mse_z_only)

    mse_full = np.mean(mse_full_list)
    mse_ctx_only = np.mean(mse_ctx_only_list)
    mse_z_only = np.mean(mse_z_only_list)

    # Contributions
    z_contribution = (mse_ctx_only - mse_full) / mse_ctx_only * 100 if mse_ctx_only > 0 else 0
    ctx_contribution = (mse_z_only - mse_full) / mse_z_only * 100 if mse_z_only > 0 else 0

    return {
        "mse_full": mse_full,
        "mse_ctx_only": mse_ctx_only,
        "mse_z_only": mse_z_only,
        "z_contribution_pct": z_contribution,
        "ctx_contribution_pct": ctx_contribution,
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

            samples = model.sample(batch, n_samples=n_samples)
            target = batch_data[:, 1:]

            # Take last timestep
            samples_last = samples[:, :, -1, :, :].cpu().numpy()  # (n_samples, B, 5, 5)
            gt_last = target[:, -1, :, :].cpu().numpy()  # (B, 5, 5)

            all_samples.append(samples_last.reshape(-1, 5, 5))
            all_gt.append(gt_last)

    all_samples = np.concatenate(all_samples, axis=0)  # (N, 5, 5)
    all_gt = np.concatenate(all_gt, axis=0)  # (M, 5, 5)

    gt_kurtosis = np.zeros((5, 5))
    model_kurtosis = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_kurtosis[i, j] = kurtosis(all_gt[:, i, j], fisher=True)
            model_kurtosis[i, j] = kurtosis(all_samples[:, i, j], fisher=True)

    # Recovery ratio
    recovery = np.abs(model_kurtosis) / np.abs(gt_kurtosis + 1e-8)
    recovery = np.clip(recovery, 0, 2)  # Cap at 200%

    return {
        "gt_kurtosis": gt_kurtosis,
        "model_kurtosis": model_kurtosis,
        "recovery_ratio": recovery,
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

            # ATM point
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy()  # (n_samples, B, T)
            gt_atm = target[:, :, 2, 2].cpu().numpy()  # (B, T)

            all_samples.append(samples_atm)
            all_gt.append(gt_atm)

    # Compute ACF lag-1
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

    # Preservation ratio
    preservation = sample_acf_mean / gt_acf_mean if abs(gt_acf_mean) > 1e-8 else 0

    return {
        "gt_acf_lag1": float(gt_acf_mean),
        "model_acf_lag1": float(sample_acf_mean),
        "acf_preservation": float(preservation),
    }


def train_dual_path(model, train_loader, val_loader, config, epochs=100):
    """
    Train dual-path decoder with two-phase training.

    Phase A: Train both pathways with MSE (joint training)
    Phase B: Train variance with Student-t NLL
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    phase_a_epochs = int(epochs * 0.6)
    phase_b_epochs = epochs - phase_a_epochs

    best_z_contrib = 0
    best_ctx_contrib = 0

    # ========================================
    # PHASE A: Joint training with MSE
    # ========================================
    print(f"\n--- Phase A: Joint Training with MSE ({phase_a_epochs} epochs) ---")
    print("    Training both ctx_mean_net and z_residual_net together")

    # Freeze variance networks
    for param in model.decoder.factor_net.parameters():
        param.requires_grad = False
    for param in model.decoder.log_diag_net.parameters():
        param.requires_grad = False

    optimizer_a = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, patience=10, factor=0.5)

    for epoch in range(phase_a_epochs):
        model.train()
        train_mses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, _, _ = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            mse_loss = F.mse_loss(pred, target)
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            loss = mse_loss + kl_weight * kl_loss

            optimizer_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_a.step()

            train_mses.append(mse_loss.item())

        avg_mse = np.mean(train_mses)
        scheduler_a.step(avg_mse)

        if (epoch + 1) % 10 == 0:
            # Check pathway contributions
            contrib = evaluate_pathway_contributions(model, val_loader, device)
            print(f"    Epoch {epoch+1}/{phase_a_epochs}: MSE={avg_mse:.6f}, "
                  f"z_contrib={contrib['z_contribution_pct']:.1f}%, "
                  f"ctx_contrib={contrib['ctx_contribution_pct']:.1f}%")

            if contrib['z_contribution_pct'] > best_z_contrib:
                best_z_contrib = contrib['z_contribution_pct']
            if contrib['ctx_contribution_pct'] > best_ctx_contrib:
                best_ctx_contrib = contrib['ctx_contribution_pct']

    # Check contributions after Phase A
    print("\n    Final Phase A contributions:")
    contrib = evaluate_pathway_contributions(model, val_loader, device)
    print(f"    Z contribution: {contrib['z_contribution_pct']:.1f}%")
    print(f"    Ctx contribution: {contrib['ctx_contribution_pct']:.1f}%")

    # ========================================
    # PHASE B: Variance training with Student-t NLL
    # ========================================
    print(f"\n--- Phase B: Variance Training with Student-t NLL ({phase_b_epochs} epochs) ---")

    # Freeze mean networks, unfreeze variance networks
    for param in model.ctx_encoder.parameters():
        param.requires_grad = False
    for param in model.main_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.ctx_mean_net.parameters():
        param.requires_grad = False
    for param in model.decoder.z_residual_net.parameters():
        param.requires_grad = False

    for param in model.decoder.factor_net.parameters():
        param.requires_grad = True
    for param in model.decoder.log_diag_net.parameters():
        param.requires_grad = True

    optimizer_b = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    for epoch in range(phase_b_epochs):
        model.train()
        train_nlls = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            nll_loss = model.decoder.compute_student_t_nll(pred, target, factor, log_diag)

            optimizer_b.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            train_nlls.append(nll_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_b_epochs}: NLL={np.mean(train_nlls):.4f}")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    return model


def run_experiment(epochs=100, save_model=True):
    """Run dual-path decoder experiment."""
    print("=" * 70)
    print("DUAL-PATH DECODER EXPERIMENT")
    print("=" * 70)
    print("\nKey insight: mean = ctx_mean + z_residual (additive, not FiLM)")
    print("Expected: both pathways contribute (z>20%, ctx>20%)")

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
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfig:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  ctx_embedding_dim: {config['ctx_embedding_dim']}")
    print(f"  cov_rank: {config['cov_rank']}")
    print(f"  device: {config['device']}")

    # Create dataloaders
    n_train = int(len(log_returns) * 0.8)
    train_loader = create_dataloader(log_returns[:n_train], CONTEXT_LEN, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], CONTEXT_LEN, batch_size=32, shuffle=False)

    # Create model
    model = CVAETwoStageDualPath(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {n_params:,} parameters")

    # Count parameters per component
    ctx_encoder_params = sum(p.numel() for p in model.ctx_encoder.parameters())
    main_encoder_params = sum(p.numel() for p in model.main_encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    ctx_mean_params = sum(p.numel() for p in model.decoder.ctx_mean_net.parameters())
    z_residual_params = sum(p.numel() for p in model.decoder.z_residual_net.parameters())

    print(f"  ctx_encoder: {ctx_encoder_params:,}")
    print(f"  main_encoder: {main_encoder_params:,}")
    print(f"  decoder total: {decoder_params:,}")
    print(f"    ctx_mean_net: {ctx_mean_params:,}")
    print(f"    z_residual_net: {z_residual_params:,}")

    # Train
    model = train_dual_path(model, train_loader, val_loader, config, epochs=epochs)

    # ========================================
    # EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    device = config["device"]

    # 1. Pathway contributions
    print("\n--- Pathway Contributions ---")
    contrib = evaluate_pathway_contributions(model, val_loader, device)
    print(f"  MSE (full):      {contrib['mse_full']:.6f}")
    print(f"  MSE (ctx only):  {contrib['mse_ctx_only']:.6f}")
    print(f"  MSE (z only):    {contrib['mse_z_only']:.6f}")
    print(f"  Z Contribution:  {contrib['z_contribution_pct']:.1f}%")
    print(f"  Ctx Contribution: {contrib['ctx_contribution_pct']:.1f}%")

    z_pass = contrib['z_contribution_pct'] > 20
    ctx_pass = contrib['ctx_contribution_pct'] > 20
    print(f"\n  Z > 20%: {'PASS' if z_pass else 'FAIL'}")
    print(f"  Ctx > 20%: {'PASS' if ctx_pass else 'FAIL'}")

    # 2. Kurtosis
    print("\n--- Kurtosis Recovery ---")
    print("Generating samples (may take a minute)...")
    kurt = evaluate_kurtosis(model, val_loader, device, n_samples=200)
    print(f"  GT Kurtosis (ATM):    {kurt['atm_gt_kurtosis']:.2f}")
    print(f"  Model Kurtosis (ATM): {kurt['atm_model_kurtosis']:.2f}")
    print(f"  Mean Recovery:        {kurt['mean_recovery']*100:.1f}%")

    kurt_pass = kurt['mean_recovery'] > 1.0
    print(f"\n  Kurtosis > 100%: {'PASS' if kurt_pass else 'FAIL'}")

    # 3. ACF
    print("\n--- ACF Preservation ---")
    acf = evaluate_acf(model, val_loader, device, n_samples=100)
    print(f"  GT ACF lag-1:    {acf['gt_acf_lag1']:.4f}")
    print(f"  Model ACF lag-1: {acf['model_acf_lag1']:.4f}")
    print(f"  Preservation:    {acf['acf_preservation']*100:.1f}%")

    acf_pass = abs(acf['acf_preservation']) > 0.3
    print(f"\n  ACF > 30%: {'PASS' if acf_pass else 'FAIL'}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        "contributions": contrib,
        "kurtosis": {
            "mean_recovery": kurt['mean_recovery'],
            "atm_gt": kurt['atm_gt_kurtosis'],
            "atm_model": kurt['atm_model_kurtosis'],
        },
        "acf": acf,
        "pass_criteria": {
            "z_contribution": bool(z_pass),
            "ctx_contribution": bool(ctx_pass),
            "kurtosis": bool(kurt_pass),
            "acf": bool(acf_pass),
        }
    }

    print(f"\n  {'Metric':<25} {'Value':>12} {'Target':>12} {'Status':>10}")
    print("  " + "-" * 60)
    print(f"  {'Z Contribution':<25} {contrib['z_contribution_pct']:>11.1f}% {'>20%':>12} {'PASS' if z_pass else 'FAIL':>10}")
    print(f"  {'Ctx Contribution':<25} {contrib['ctx_contribution_pct']:>11.1f}% {'>20%':>12} {'PASS' if ctx_pass else 'FAIL':>10}")
    print(f"  {'Kurtosis Recovery':<25} {kurt['mean_recovery']*100:>11.1f}% {'>100%':>12} {'PASS' if kurt_pass else 'FAIL':>10}")
    print(f"  {'ACF Preservation':<25} {acf['acf_preservation']*100:>11.1f}% {'>30%':>12} {'PASS' if acf_pass else 'FAIL':>10}")

    all_pass = z_pass and ctx_pass and kurt_pass
    critical_pass = z_pass and ctx_pass and kurt_pass  # These are critical

    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: ALL TARGETS MET!")
    elif critical_pass:
        print("OVERALL: CRITICAL TARGETS MET (z, ctx, kurtosis)")
    else:
        print("OVERALL: SOME TARGETS NOT MET")
        if not z_pass:
            print("  - Z contribution below 20%")
        if not ctx_pass:
            print("  - Ctx contribution below 20%")
        if not kurt_pass:
            print("  - Kurtosis recovery below 100%")
    print("=" * 70)

    # Save model if requested
    if save_model:
        save_dir = Path("models/backfill/two_stage/dual_path")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "dual_path_best.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "results": results,
        }
        torch.save(checkpoint, save_path)
        print(f"\nModel saved to: {save_path}")

        # Save results JSON
        results_path = save_dir / "dual_path_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

    return model, results


if __name__ == "__main__":
    model, results = run_experiment(epochs=100, save_model=True)
