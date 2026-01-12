"""
Warm-Start Context Addition Experiment

This experiment:
1. Loads the trained StudentTMLPDecoder (which achieves 139.9% kurtosis)
2. Creates a new model with the SAME z pathway weights
3. Only trains the new context pathway on top

Key insight: Don't retrain what already works. Just add context.

Expected:
- Kurtosis recovery: preserved (>100%) since z pathway is frozen
- Ctx contribution: positive (>5%)
- ACF: potentially improved

Usage:
    python experiments/backfill/two_stage_vae/exp_warmstart_context.py
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
from vae.cvae_two_stage import (
    CVAETwoStageStudentTMLP,
    CVAETwoStageGatedResidual,
    TwoStageCtxEncoder,
    TwoStageMainEncoder,
)


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
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    if var < 1e-10:
        return 0.0
    cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
    return cov / var


def evaluate_bottleneck(model, val_loader, device):
    """
    Evaluate bottleneck capacity exactly like comprehensive_oracle_analysis.

    Returns MSE for:
    - Full model (ctx + z)
    - Ctx only (z=0)
    - Z only (ctx=0)
    """
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

            # Full model
            mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full.append(((pred_full - target) ** 2).mean().item())

            # Ctx only (z=0)
            z_zero = torch.zeros_like(z)
            mean_ctx, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
            pred_ctx = mean_ctx[:, :-1]
            mse_ctx_only.append(((pred_ctx - target) ** 2).mean().item())

            # Z only (ctx=0)
            ctx_zero = torch.zeros_like(ctx_emb)
            mean_z, _, _, _ = model.decoder(ctx_zero, z, sample=False)
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
    """Evaluate kurtosis using same methodology as comprehensive analysis."""
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:  # Same limit as comprehensive
                break

            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            samples = model.sample(batch, n_samples=n_samples)
            target = batch_data[:, 1:]

            # Take last timestep
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


def run_experiment():
    """Run warm-start context addition experiment."""
    print("=" * 70)
    print("WARM-START CONTEXT ADDITION EXPERIMENT")
    print("=" * 70)
    print("\nStrategy: Load trained z pathway, freeze it, train context only")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the trained StudentTMLPDecoder checkpoint
    checkpoint_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    print(f"\nLoading trained checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_config = checkpoint["model_config"]
    base_config["device"] = device

    # Create the base model to verify it works
    print("\n--- Verifying Base Model (StudentTMLPDecoder) ---")
    base_model = CVAETwoStageStudentTMLP(base_config)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model = base_model.to(device)
    base_model.eval()

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)
    n_train = int(len(log_returns) * 0.8)
    val_loader = create_dataloader(log_returns[n_train:], CONTEXT_LEN, batch_size=32, shuffle=False)
    train_loader = create_dataloader(log_returns[:n_train], CONTEXT_LEN, batch_size=32)

    # Verify base model performance
    base_bottleneck = evaluate_bottleneck(base_model, val_loader, device)
    print(f"  Base MSE (full):     {base_bottleneck['mse_full']:.6f}")
    print(f"  Base Z contrib:      {base_bottleneck['z_contribution_pct']:.1f}%")
    print(f"  Base Ctx contrib:    {base_bottleneck['ctx_contribution_pct']:.1f}%")

    base_kurt = evaluate_kurtosis(base_model, val_loader, device, n_samples=200)
    print(f"  Base Kurtosis:       {base_kurt['mean_recovery']*100:.1f}%")

    # Now create the gated residual model and copy weights
    print("\n--- Creating Warm-Started Gated Residual Model ---")

    # Create new model
    new_config = base_config.copy()
    new_config["gate_scale"] = 0.3
    new_model = CVAETwoStageGatedResidual(new_config)

    # Copy ctx_encoder weights
    new_model.ctx_encoder.load_state_dict(base_model.ctx_encoder.state_dict())

    # Copy main_encoder weights
    new_model.main_encoder.load_state_dict(base_model.main_encoder.state_dict())

    # Copy z pathway weights from base decoder to new decoder
    # StudentTMLPDecoder.mean_net -> StudentTGatedResidualDecoder.z_pred_net
    new_model.decoder.z_pred_net.load_state_dict(base_model.decoder.mean_net.state_dict())

    # Copy covariance weights
    new_model.decoder.factor_net.load_state_dict(base_model.decoder.factor_net.state_dict())
    new_model.decoder.log_diag_net.load_state_dict(base_model.decoder.log_diag_net.state_dict())

    # Copy nu buffer
    new_model.decoder.nu = base_model.decoder.nu.clone()

    new_model = new_model.to(device)

    # Verify warm-started model matches base
    print("\n  Verifying warm-start (should match base)...")
    new_bottleneck = evaluate_bottleneck(new_model, val_loader, device)
    print(f"  New MSE (full):      {new_bottleneck['mse_full']:.6f}")
    print(f"  New Z contrib:       {new_bottleneck['z_contribution_pct']:.1f}%")

    # The new model should match or be very close to base (gate starts near 0)
    if abs(new_bottleneck['mse_full'] - base_bottleneck['mse_full']) < 0.01:
        print("  Warm-start successful: MSE matches base model")
    else:
        print("  WARNING: MSE differs from base model!")

    # Check kurtosis of warm-started model (should match base)
    new_kurt = evaluate_kurtosis(new_model, val_loader, device, n_samples=200)
    print(f"  New Kurtosis:        {new_kurt['mean_recovery']*100:.1f}%")

    # ========================================
    # TRAIN CONTEXT PATHWAY ONLY
    # ========================================
    print("\n--- Training Context Pathway (z pathway frozen) ---")

    # Freeze everything except context correction and gate
    for param in new_model.parameters():
        param.requires_grad = False

    for param in new_model.decoder.ctx_correction_net.parameters():
        param.requires_grad = True
    for param in new_model.decoder.gate_net.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for p in new_model.parameters() if p.requires_grad],
        lr=5e-4  # Lower learning rate to be careful
    )

    epochs = 50
    for epoch in range(epochs):
        new_model.train()
        train_mses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            ctx_emb = new_model.ctx_encoder(batch)
            z_mean, z_logvar, z = new_model.main_encoder(batch)
            mean, _, _, _ = new_model.decoder(ctx_emb, z, sample=False)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            mse_loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(new_model.parameters(), 1.0)
            optimizer.step()

            train_mses.append(mse_loss.item())

        if (epoch + 1) % 10 == 0:
            bottleneck = evaluate_bottleneck(new_model, val_loader, device)
            print(f"  Epoch {epoch+1}/{epochs}: MSE={np.mean(train_mses):.6f}, "
                  f"ctx_contrib={bottleneck['ctx_contribution_pct']:.1f}%")

    # ========================================
    # FINAL EVALUATION
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_bottleneck = evaluate_bottleneck(new_model, val_loader, device)
    print(f"\n  MSE (full):       {final_bottleneck['mse_full']:.6f}")
    print(f"  MSE (ctx only):   {final_bottleneck['mse_ctx_only']:.6f}")
    print(f"  MSE (z only):     {final_bottleneck['mse_z_only']:.6f}")
    print(f"  Z Contribution:   {final_bottleneck['z_contribution_pct']:.1f}%")
    print(f"  Ctx Contribution: {final_bottleneck['ctx_contribution_pct']:.1f}%")

    final_kurt = evaluate_kurtosis(new_model, val_loader, device, n_samples=200)
    print(f"\n  Kurtosis Recovery: {final_kurt['mean_recovery']*100:.1f}%")
    print(f"  ATM GT Kurtosis:   {final_kurt['atm_gt_kurtosis']:.2f}")
    print(f"  ATM Model Kurt:    {final_kurt['atm_model_kurtosis']:.2f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON: BASE vs WARM-START + CONTEXT")
    print("=" * 70)

    print(f"\n  {'Metric':<20} {'Base':>12} {'+ Context':>12} {'Change':>12}")
    print("  " + "-" * 58)
    print(f"  {'MSE':<20} {base_bottleneck['mse_full']:>12.6f} {final_bottleneck['mse_full']:>12.6f} "
          f"{final_bottleneck['mse_full'] - base_bottleneck['mse_full']:>+12.6f}")
    print(f"  {'Z Contribution':<20} {base_bottleneck['z_contribution_pct']:>11.1f}% {final_bottleneck['z_contribution_pct']:>11.1f}% "
          f"{final_bottleneck['z_contribution_pct'] - base_bottleneck['z_contribution_pct']:>+11.1f}%")
    print(f"  {'Ctx Contribution':<20} {base_bottleneck['ctx_contribution_pct']:>11.1f}% {final_bottleneck['ctx_contribution_pct']:>11.1f}% "
          f"{final_bottleneck['ctx_contribution_pct'] - base_bottleneck['ctx_contribution_pct']:>+11.1f}%")
    print(f"  {'Kurtosis':<20} {base_kurt['mean_recovery']*100:>11.1f}% {final_kurt['mean_recovery']*100:>11.1f}% "
          f"{(final_kurt['mean_recovery'] - base_kurt['mean_recovery'])*100:>+11.1f}%")

    # Success criteria
    mse_improved = final_bottleneck['mse_full'] < base_bottleneck['mse_full']
    ctx_positive = final_bottleneck['ctx_contribution_pct'] > 0
    kurt_preserved = final_kurt['mean_recovery'] > 0.9  # Allow 10% regression

    print("\n" + "=" * 70)
    if mse_improved and ctx_positive and kurt_preserved:
        print("SUCCESS: MSE improved, context contributing, kurtosis preserved!")
    elif kurt_preserved:
        print("PARTIAL SUCCESS: Kurtosis preserved")
    else:
        print("REGRESSION: Kurtosis dropped significantly")
    print("=" * 70)

    # Save model
    save_dir = Path("models/backfill/two_stage/warmstart_context")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "warmstart_context_best.pt"

    checkpoint = {
        "model_state_dict": new_model.state_dict(),
        "model_config": new_config,
        "results": {
            "base": {
                "bottleneck": base_bottleneck,
                "kurtosis": base_kurt,
            },
            "final": {
                "bottleneck": final_bottleneck,
                "kurtosis": final_kurt,
            }
        }
    }
    torch.save(checkpoint, save_path)
    print(f"\nModel saved to: {save_path}")

    return new_model


if __name__ == "__main__":
    model = run_experiment()
