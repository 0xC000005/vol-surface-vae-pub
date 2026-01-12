"""
Gated Residual Decoder Training Experiment

This experiment trains the CVAETwoStageGatedResidual model that uses
z as the primary prediction with a gated context correction:

    mean = z_pred + gate * ctx_correction

Key insight: Z dominates (preserves kurtosis), context provides small correction.
Gate prevents context from dominating (unlike FiLM bypass).

Expected improvements over StudentTMLPDecoder:
- Kurtosis recovery: preserved (>100%) - z still dominant
- Ctx contribution: small but positive (>5%)
- ACF preservation: improved (context adds temporal signal)

Training phases:
1. Phase A: Train z pathway with MSE (like original)
2. Phase B: Add and train gated context pathway
3. Phase C: Train variance with Student-t NLL

Usage:
    python experiments/backfill/two_stage_vae/exp_gated_residual_decoder.py
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
from vae.cvae_two_stage import CVAETwoStageGatedResidual


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


def evaluate_pathway_contributions(model, val_loader, device):
    """
    Evaluate contribution of each pathway using MSE analysis.

    For gated residual: mean = z_pred + gate * ctx_correction
    - z_only: z_pred (ctx_correction = 0)
    - full: z_pred + gate * ctx_correction
    """
    model.eval()

    mse_full_list = []
    mse_z_only_list = []
    avg_gate_list = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Get embeddings
            ctx_emb = model.ctx_encoder(batch)
            z_mean, z_logvar, z = model.main_encoder(batch)

            target = batch_data[:, 1:]
            B, T, _ = z.shape

            # Full model
            mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full = F.mse_loss(pred_full, target).item()
            mse_full_list.append(mse_full)

            # Z only (compute z_pred directly without gate correction)
            z_flat = z.view(B * T, -1)
            z_pred = model.decoder.z_pred_net(z_flat).view(B, T, 5, 5)
            pred_z_only = z_pred[:, :-1]
            mse_z_only = F.mse_loss(pred_z_only, target).item()
            mse_z_only_list.append(mse_z_only)

            # Compute average gate value
            ctx_flat = ctx_emb.view(B * T, -1)
            combined = torch.cat([z_flat, ctx_flat], dim=-1)
            gate_logits = model.decoder.gate_net(combined)
            gate = torch.sigmoid(gate_logits) * model.decoder.gate_scale
            avg_gate_list.append(gate.mean().item())

    mse_full = np.mean(mse_full_list)
    mse_z_only = np.mean(mse_z_only_list)
    avg_gate = np.mean(avg_gate_list)

    # Context contribution: reduction in MSE from adding gated context
    ctx_contribution = (mse_z_only - mse_full) / mse_z_only * 100 if mse_z_only > 0 else 0

    return {
        "mse_full": mse_full,
        "mse_z_only": mse_z_only,
        "ctx_contribution_pct": ctx_contribution,
        "avg_gate": avg_gate,
    }


def evaluate_kurtosis(model, val_loader, device, n_samples=200):
    """Evaluate kurtosis of generated samples vs GT."""
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


def train_gated_residual(model, train_loader, val_loader, config, epochs=100):
    """
    Train gated residual decoder with three-phase training.

    Phase A: Train z pathway only with MSE (like original)
    Phase B: Add and train gated context pathway
    Phase C: Train variance with Student-t NLL
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    phase_a_epochs = int(epochs * 0.4)
    phase_b_epochs = int(epochs * 0.3)
    phase_c_epochs = epochs - phase_a_epochs - phase_b_epochs

    # ========================================
    # PHASE A: Train z pathway only with MSE
    # ========================================
    print(f"\n--- Phase A: Train Z Pathway Only ({phase_a_epochs} epochs) ---")
    print("    This replicates the original StudentTMLPDecoder training")

    # Freeze context and gate networks, variance networks
    for param in model.decoder.ctx_correction_net.parameters():
        param.requires_grad = False
    for param in model.decoder.gate_net.parameters():
        param.requires_grad = False
    for param in model.decoder.factor_net.parameters():
        param.requires_grad = False
    for param in model.decoder.log_diag_net.parameters():
        param.requires_grad = False

    optimizer_a = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    for epoch in range(phase_a_epochs):
        model.train()
        train_mses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Get z_pred directly (gate is frozen at 0 due to initialization)
            ctx_emb = model.ctx_encoder(batch)
            z_mean, z_logvar, z = model.main_encoder(batch)
            mean, _, _, _ = model.decoder(ctx_emb, z, sample=False)

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

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_a_epochs}: MSE={np.mean(train_mses):.6f}")

    # Check after Phase A
    print("\n    Checking after Phase A...")
    contrib = evaluate_pathway_contributions(model, val_loader, device)
    print(f"    MSE (z_only): {contrib['mse_z_only']:.6f}")
    print(f"    Avg gate: {contrib['avg_gate']:.4f}")

    # ========================================
    # PHASE B: Add and train gated context pathway
    # ========================================
    print(f"\n--- Phase B: Train Gated Context Pathway ({phase_b_epochs} epochs) ---")
    print("    Unfreezing context and gate networks")

    # Freeze z pathway and encoders, unfreeze context and gate
    for param in model.ctx_encoder.parameters():
        param.requires_grad = False
    for param in model.main_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.z_pred_net.parameters():
        param.requires_grad = False

    for param in model.decoder.ctx_correction_net.parameters():
        param.requires_grad = True
    for param in model.decoder.gate_net.parameters():
        param.requires_grad = True

    optimizer_b = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    for epoch in range(phase_b_epochs):
        model.train()
        train_mses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, _, _, _ = model.decoder(
                model.ctx_encoder(batch),
                model.main_encoder(batch)[2],
                sample=False
            )

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            mse_loss = F.mse_loss(pred, target)

            optimizer_b.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            train_mses.append(mse_loss.item())

        if (epoch + 1) % 10 == 0:
            contrib = evaluate_pathway_contributions(model, val_loader, device)
            print(f"    Epoch {epoch+1}/{phase_b_epochs}: MSE={np.mean(train_mses):.6f}, "
                  f"ctx_contrib={contrib['ctx_contribution_pct']:.1f}%, "
                  f"avg_gate={contrib['avg_gate']:.4f}")

    # ========================================
    # PHASE C: Variance training with Student-t NLL
    # ========================================
    print(f"\n--- Phase C: Variance Training with Student-t NLL ({phase_c_epochs} epochs) ---")

    # Freeze everything except variance networks
    for param in model.parameters():
        param.requires_grad = False

    for param in model.decoder.factor_net.parameters():
        param.requires_grad = True
    for param in model.decoder.log_diag_net.parameters():
        param.requires_grad = True

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
    """Run gated residual decoder experiment."""
    print("=" * 70)
    print("GATED RESIDUAL DECODER EXPERIMENT")
    print("=" * 70)
    print("\nArchitecture: mean = z_pred + gate * ctx_correction")
    print("Key insight: Z dominates (preserves kurtosis), context adds small correction")

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
    config["gate_scale"] = 0.3  # Max gate value
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfig:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  ctx_embedding_dim: {config['ctx_embedding_dim']}")
    print(f"  cov_rank: {config['cov_rank']}")
    print(f"  gate_scale: {config['gate_scale']}")
    print(f"  device: {config['device']}")

    # Create dataloaders
    n_train = int(len(log_returns) * 0.8)
    train_loader = create_dataloader(log_returns[:n_train], CONTEXT_LEN, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], CONTEXT_LEN, batch_size=32, shuffle=False)

    # Create model
    model = CVAETwoStageGatedResidual(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {n_params:,} parameters")

    # Train
    model = train_gated_residual(model, train_loader, val_loader, config, epochs=epochs)

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
    print(f"  MSE (z only):    {contrib['mse_z_only']:.6f}")
    print(f"  Ctx Contribution: {contrib['ctx_contribution_pct']:.1f}%")
    print(f"  Avg Gate Value:   {contrib['avg_gate']:.4f}")

    ctx_pass = contrib['ctx_contribution_pct'] > 5

    # 2. Kurtosis
    print("\n--- Kurtosis Recovery ---")
    print("Generating samples (may take a minute)...")
    kurt = evaluate_kurtosis(model, val_loader, device, n_samples=200)
    print(f"  GT Kurtosis (ATM):    {kurt['atm_gt_kurtosis']:.2f}")
    print(f"  Model Kurtosis (ATM): {kurt['atm_model_kurtosis']:.2f}")
    print(f"  Mean Recovery:        {kurt['mean_recovery']*100:.1f}%")

    kurt_pass = kurt['mean_recovery'] > 1.0

    # 3. ACF
    print("\n--- ACF Preservation ---")
    acf = evaluate_acf(model, val_loader, device, n_samples=100)
    print(f"  GT ACF lag-1:    {acf['gt_acf_lag1']:.4f}")
    print(f"  Model ACF lag-1: {acf['model_acf_lag1']:.4f}")
    print(f"  Preservation:    {acf['acf_preservation']*100:.1f}%")

    acf_pass = abs(acf['acf_preservation']) > 0.3

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
            "ctx_contribution": bool(ctx_pass),
            "kurtosis": bool(kurt_pass),
            "acf": bool(acf_pass),
        }
    }

    print(f"\n  {'Metric':<25} {'Value':>12} {'Target':>12} {'Status':>10}")
    print("  " + "-" * 60)
    print(f"  {'Ctx Contribution':<25} {contrib['ctx_contribution_pct']:>11.1f}% {'>5%':>12} {'PASS' if ctx_pass else 'FAIL':>10}")
    print(f"  {'Kurtosis Recovery':<25} {kurt['mean_recovery']*100:>11.1f}% {'>100%':>12} {'PASS' if kurt_pass else 'FAIL':>10}")
    print(f"  {'ACF Preservation':<25} {acf['acf_preservation']*100:>11.1f}% {'>30%':>12} {'PASS' if acf_pass else 'FAIL':>10}")

    critical_pass = kurt_pass  # Kurtosis is the critical metric we can't regress

    print("\n" + "=" * 70)
    if critical_pass and ctx_pass:
        print("OVERALL: SUCCESS! Kurtosis preserved AND context contributing")
    elif critical_pass:
        print("OVERALL: PARTIAL SUCCESS - Kurtosis preserved, context needs work")
    else:
        print("OVERALL: FAIL - Kurtosis regressed")
    print("=" * 70)

    # Save model if requested
    if save_model:
        save_dir = Path("models/backfill/two_stage/gated_residual")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "gated_residual_best.pt"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "results": results,
        }
        torch.save(checkpoint, save_path)
        print(f"\nModel saved to: {save_path}")

        # Save results JSON
        results_path = save_dir / "gated_residual_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

    return model, results


if __name__ == "__main__":
    model, results = run_experiment(epochs=100, save_model=True)
