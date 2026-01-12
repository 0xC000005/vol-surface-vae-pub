"""
Train Main VAE's Student-t Decoder

This script trains the main VAE's CVAETwoStageStudentT class (which uses
LSTM + FiLM + Deconv architecture) and evaluates if it achieves comparable
kurtosis recovery to the experiment's simple MLP decoder.

Success Criteria:
- Kurtosis recovery: >100% (experiment achieved 135.8%)
- MSE: comparable to experiment (~0.06)
- Direction accuracy: >45% (with correct alignment)

If this doesn't work, we'll fall back to porting the experiment's simple
MLP decoder to the main VAE.

Usage:
    python experiments/backfill/two_stage_vae/train_main_vae_student_t.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import CVAETwoStageStudentT
from config.two_stage_config import TWO_STAGE_CONFIG


# GT kurtosis values from shape_diagnostics.json (Fisher/excess kurtosis)
GT_KURTOSIS = np.array([
    [5.53, 2.75, 2.21, 15.33, 19.66],    # Row 0 (short-term moneyness)
    [21.96, 3.24, 2.51, 84.76, 18.65],   # Row 1
    [65.47, 9.98, 5.02, 2.82, 12.75],    # Row 2 (ATM row)
    [147.78, 47.95, 25.18, 8.50, 143.95],# Row 3
    [75.01, 73.92, 69.44, 44.23, 244.48] # Row 4 (long-term)
]).flatten()  # (25,)


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

    tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_z_contribution(model, val_loader, device):
    """Evaluate z contribution to mean prediction."""
    model.eval()

    oracle_mses = []
    zero_z_mses = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            x = {"surface": batch_data}

            # Get embeddings
            ctx_emb = model.ctx_encoder(x)
            z_mean, z_logvar, z = model.encoder(x)

            # Mean with oracle z
            mean_oracle, L_oracle, nu = model.decoder(ctx_emb, z)
            target = batch_data[:, 1:]
            pred_oracle = mean_oracle[:, :-1]
            oracle_mse = ((pred_oracle - target) ** 2).mean().item()
            oracle_mses.append(oracle_mse)

            # Mean with z=0
            z_zero = torch.zeros_like(z)
            mean_zero, _, _ = model.decoder(ctx_emb, z_zero)
            pred_zero = mean_zero[:, :-1]
            zero_z_mse = ((pred_zero - target) ** 2).mean().item()
            zero_z_mses.append(zero_z_mse)

    oracle_mse = np.mean(oracle_mses)
    zero_z_mse = np.mean(zero_z_mses)
    z_contribution = (zero_z_mse - oracle_mse) / zero_z_mse * 100 if zero_z_mse > 0 else 0

    return {
        "oracle_mse": oracle_mse,
        "zero_z_mse": zero_z_mse,
        "z_contribution": z_contribution,
    }


def train_main_vae_student_t(model, train_loader, val_loader, config, epochs=150):
    """
    Two-phase training for main VAE's Student-t decoder.

    Phase A: Train mean with MSE (freeze Cholesky head)
    Phase B: Train covariance with Student-t NLL (freeze encoders)
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    phase_a_epochs = int(epochs * 0.6)
    phase_b_epochs = epochs - phase_a_epochs

    # ========================================
    # PHASE A: Pre-train mean with MSE
    # ========================================
    print(f"\n--- Phase A: Pre-train Mean with MSE ({phase_a_epochs} epochs) ---")

    # Freeze Cholesky head during Phase A
    for param in model.decoder.cholesky_head.parameters():
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
            x = {"surface": batch_data}

            # Forward pass
            mean, L, nu, z_mean, z_logvar, z = model.forward(x, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # MSE loss
            mse_loss = F.mse_loss(pred, target)

            # KL loss
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            loss = mse_loss + kl_weight * kl_loss

            optimizer_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_a.step()

            train_mses.append(mse_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_a_epochs}: Train MSE={np.mean(train_mses):.6f}")

    # Check z contribution after Phase A
    print("\n    Checking z contribution after Phase A...")
    z_results = evaluate_z_contribution(model, val_loader, device)
    print(f"    Z contribution: {z_results['z_contribution']:.1f}%")
    print(f"    Oracle MSE: {z_results['oracle_mse']:.6f}")

    # ========================================
    # PHASE B: Train covariance with Student-t NLL
    # ========================================
    print(f"\n--- Phase B: Train Covariance with Student-t NLL ({phase_b_epochs} epochs) ---")

    # Freeze encoders and mean pathway, unfreeze Cholesky head
    for param in model.ctx_encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Freeze mean-related parts of decoder (FiLM modulation, surface decoder)
    for param in model.decoder.gamma_net.parameters():
        param.requires_grad = False
    for param in model.decoder.beta_net.parameters():
        param.requires_grad = False
    for param in model.decoder.surface_decoder.parameters():
        param.requires_grad = False
    for param in model.decoder.mem.parameters():
        param.requires_grad = False

    # Unfreeze Cholesky head
    for param in model.decoder.cholesky_head.parameters():
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
            x = {"surface": batch_data}

            mean, L, nu, z_mean, z_logvar, z = model.forward(x, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]
            pred_L = L[:, :-1]

            # Student-t NLL (detach mean so NLL only affects Cholesky head)
            nll_loss = model.multivariate_student_t_nll(
                pred.detach(), pred_L, target, nu
            )

            optimizer_b.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            train_nlls.append(nll_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_b_epochs}: Train NLL={np.mean(train_nlls):.4f}")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    return model


def evaluate_kurtosis(model, val_loader, device, n_samples=500):
    """
    Evaluate kurtosis of generated samples vs GT.
    """
    model.eval()

    all_samples = []
    all_targets = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            x = {"surface": batch_data}

            # Get mean and L for sampling
            mean, L, nu, _, _, _ = model.forward(x, return_full_sequence=True)

            # Generate samples
            samples = []
            for _ in range(n_samples):
                sample = model.sample_from_decoder(mean, L, nu)
                samples.append(sample)
            samples = torch.stack(samples)  # (n_samples, B, T, 5, 5)

            # Get target (next step)
            target = batch_data[:, 1:]  # (B, T, 5, 5)

            # Take only last prediction for each sequence
            samples_last = samples[:, :, -1, :, :]  # (n_samples, B, 5, 5)
            target_last = target[:, -1, :, :]  # (B, 5, 5)

            all_samples.append(samples_last.cpu().numpy())
            all_targets.append(target_last.cpu().numpy())

    # Stack all
    all_samples = np.concatenate(all_samples, axis=1)  # (n_samples, N, 5, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 5, 5)

    # Compute per-grid kurtosis
    model_kurtosis = np.zeros((5, 5))
    gt_kurtosis = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            samples_ij = all_samples[:, :, i, j].flatten()
            model_kurtosis[i, j] = kurtosis(samples_ij, fisher=True)
            gt_kurtosis[i, j] = kurtosis(all_targets[:, i, j], fisher=True)

    # Kurtosis recovery
    recovery_ratio = np.abs(model_kurtosis) / np.abs(gt_kurtosis + 1e-8)
    mean_recovery = recovery_ratio.mean()

    return {
        "gt_kurtosis": gt_kurtosis,
        "model_kurtosis": model_kurtosis,
        "recovery_ratio": recovery_ratio,
        "mean_recovery": mean_recovery * 100,
        "atm_gt_kurtosis": gt_kurtosis[2, 2],
        "atm_model_kurtosis": model_kurtosis[2, 2],
        "atm_recovery": recovery_ratio[2, 2] * 100,
    }


def run_experiment(epochs=150):
    """Train and evaluate main VAE's Student-t decoder."""
    print("=" * 70)
    print("Training Main VAE's Student-t Decoder (LSTM + FiLM + Deconv)")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    log_returns, _ = to_log_returns(data["surface"])

    # Config - match experiment settings
    config = TWO_STAGE_CONFIG.copy()
    config["latent_dim"] = 8
    config["decoder_mem_hidden"] = 64
    config["learn_nu"] = False  # Fix nu from GT kurtosis
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfig:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  decoder_mem_hidden: {config['decoder_mem_hidden']}")
    print(f"  learn_nu: {config['learn_nu']}")
    print(f"  device: {config['device']}")

    # Create dataloaders
    context_len = config.get("context_len", 20)
    n_train = int(len(log_returns) * 0.8)

    train_loader = create_dataloader(log_returns[:n_train], context_len, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], context_len, batch_size=32, shuffle=False)

    # Create model
    print("\nCreating CVAETwoStageStudentT from main VAE...")
    model = CVAETwoStageStudentT(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Fix nu from GT kurtosis
    print("\nSetting nu from GT kurtosis...")
    model.fix_nu_from_kurtosis(GT_KURTOSIS)

    # Train
    train_main_vae_student_t(model, train_loader, val_loader, config, epochs=epochs)

    # Evaluate kurtosis
    print("\n--- Evaluating Kurtosis Recovery ---")
    print("Generating samples (may take a minute)...")

    kurtosis_results = evaluate_kurtosis(model, val_loader, config["device"], n_samples=500)

    print(f"\nGT Kurtosis (per grid):")
    print(kurtosis_results["gt_kurtosis"].round(2))
    print(f"\nModel Kurtosis (per grid):")
    print(kurtosis_results["model_kurtosis"].round(2))
    print(f"\nKurtosis Recovery Ratio (per grid, %):")
    print((kurtosis_results["recovery_ratio"] * 100).round(1))

    print(f"\n  ATM GT Kurtosis: {kurtosis_results['atm_gt_kurtosis']:.2f}")
    print(f"  ATM Model Kurtosis: {kurtosis_results['atm_model_kurtosis']:.2f}")
    print(f"  ATM Kurtosis Recovery: {kurtosis_results['atm_recovery']:.1f}%")
    print(f"  Mean Kurtosis Recovery: {kurtosis_results['mean_recovery']:.1f}%")

    # Evaluate MSE
    z_results = evaluate_z_contribution(model, val_loader, config["device"])
    print(f"\n  Final MSE: {z_results['oracle_mse']:.6f}")
    print(f"  Z Contribution: {z_results['z_contribution']:.1f}%")

    # Save model
    output_dir = Path("models/backfill/two_stage/student_t_main_vae")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "student_t_main_vae_best.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "kurtosis_results": kurtosis_results,
        "z_results": z_results,
        "gt_kurtosis": GT_KURTOSIS,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    experiment_kurtosis = 135.8
    experiment_mse = 0.061

    print(f"\n  Kurtosis Recovery:")
    print(f"    Main VAE: {kurtosis_results['mean_recovery']:.1f}%")
    print(f"    Experiment: {experiment_kurtosis:.1f}%")
    print(f"    Target: >100%")
    print(f"    Status: {'PASS' if kurtosis_results['mean_recovery'] > 100 else 'FAIL'}")

    print(f"\n  MSE:")
    print(f"    Main VAE: {z_results['oracle_mse']:.6f}")
    print(f"    Experiment: {experiment_mse:.6f}")
    print(f"    Status: {'PASS' if z_results['oracle_mse'] < 0.1 else 'FAIL'}")

    print(f"\n  Z Contribution:")
    print(f"    Main VAE: {z_results['z_contribution']:.1f}%")
    print(f"    Target: >30%")
    print(f"    Status: {'PASS' if z_results['z_contribution'] > 30 else 'FAIL'}")

    # Overall verdict
    all_pass = (
        kurtosis_results['mean_recovery'] > 100 and
        z_results['oracle_mse'] < 0.1 and
        z_results['z_contribution'] > 30
    )

    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: PASS - Main VAE's Student-t decoder works!")
        print("Next: Run comprehensive_oracle_analysis.py to verify all metrics")
    else:
        print("OVERALL: FAIL - Main VAE's Student-t decoder did not meet targets")
        print("Next: Fall back to Option A - port experiment's simple MLP decoder")
    print("=" * 70)

    return model, kurtosis_results, z_results


if __name__ == "__main__":
    model, kurtosis_results, z_results = run_experiment(epochs=150)
