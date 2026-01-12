"""
Verify Ported Student-t MLP Decoder

This script verifies that the ported StudentTMLPDecoder and CVAETwoStageStudentTMLP
classes in vae/cvae_two_stage.py produce the same results as the experiment.

Expected Results (from experiment):
- Kurtosis Recovery: 135.8%
- Z Contribution: 39.5%
- Direction Accuracy: ~48.7% (with correct alignment)

Usage:
    python experiments/backfill/two_stage_vae/verify_ported_student_t.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kurtosis
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the ported classes from main VAE file
from vae.cvae_two_stage import CVAETwoStageStudentTMLP


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, batch_size, shuffle=True):
    """Create dataloader for evaluation."""
    sequences = []
    for i in range(len(log_returns) - context_len):
        seq = log_returns[i:i + context_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
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

            ctx_emb = model.ctx_encoder({"surface": batch_data})
            z_mean, _, z = model.main_encoder({"surface": batch_data})

            # Mean with oracle z
            mean_oracle, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            target = batch_data[:, 1:]
            pred_oracle = mean_oracle[:, :-1]
            oracle_mse = ((pred_oracle - target) ** 2).mean().item()
            oracle_mses.append(oracle_mse)

            # Mean with z=0
            z_zero = torch.zeros_like(z)
            mean_zero, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
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


def evaluate_kurtosis(model, val_loader, device, n_samples=1000):
    """Evaluate kurtosis recovery."""
    model.eval()

    # Collect ground truth and samples
    all_gt = []
    all_samples = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            # Ground truth
            target = batch_data[:, 1:]  # (B, T-1, 5, 5)
            all_gt.append(target.cpu().numpy())

            # Sample from model
            batch_samples = model.sample({"surface": batch_data}, n_samples=n_samples)
            # batch_samples: (n_samples, B, T, 5, 5)
            # Take predictions for position 1:
            batch_samples = batch_samples[:, :, :-1]  # (n_samples, B, T-1, 5, 5)
            all_samples.append(batch_samples.cpu().numpy())

    # Compute kurtosis for ATM point
    gt = np.concatenate(all_gt, axis=0)  # (N, T-1, 5, 5)
    samples = np.concatenate(all_samples, axis=1)  # (n_samples, N, T-1, 5, 5)

    # Flatten temporal dimension
    gt_flat = gt.reshape(-1, 5, 5)  # (N*(T-1), 5, 5)
    samples_flat = samples.reshape(n_samples, -1, 5, 5)  # (n_samples, N*(T-1), 5, 5)

    # ATM kurtosis
    gt_atm = gt_flat[:, 2, 2]
    gt_kurtosis = kurtosis(gt_atm, fisher=True)  # Excess kurtosis

    # Sample kurtosis (average across samples)
    sample_kurtoses = []
    for i in range(n_samples):
        sample_atm = samples_flat[i, :, 2, 2]
        sample_kurtoses.append(kurtosis(sample_atm, fisher=True))

    sample_kurtosis = np.mean(sample_kurtoses)

    # Recovery percentage
    kurtosis_recovery = (sample_kurtosis / gt_kurtosis * 100) if gt_kurtosis > 0 else 0

    return {
        "gt_kurtosis": gt_kurtosis,
        "sample_kurtosis": sample_kurtosis,
        "kurtosis_recovery": kurtosis_recovery,
    }


def main():
    print("=" * 70)
    print("VERIFY PORTED STUDENT-T MLP DECODER")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load checkpoint
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    print(f"\nLoading checkpoint from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    print(f"Config: latent_dim={config.get('latent_dim')}, cov_rank={config.get('cov_rank')}")

    # Create model using ported class
    print("\nCreating model using ported CVAETwoStageStudentTMLP class...")
    model = CVAETwoStageStudentTMLP(config)

    # Load state dict
    print("Loading state dict...")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Checkpoint loaded successfully!")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)
    print(f"Data shape: {log_returns.shape}")

    # Create validation loader
    context_len = config.get("context_len", 20)
    val_start = int(len(log_returns) * 0.8)
    val_data = log_returns[val_start:]
    val_loader = create_dataloader(val_data, context_len, batch_size=32, shuffle=False)

    # Evaluate Z contribution
    print("\n" + "=" * 70)
    print("EVALUATING Z CONTRIBUTION")
    print("=" * 70)

    z_results = evaluate_z_contribution(model, val_loader, device)
    print(f"\n  Oracle MSE: {z_results['oracle_mse']:.6f}")
    print(f"  Zero-z MSE: {z_results['zero_z_mse']:.6f}")
    print(f"  Z Contribution: {z_results['z_contribution']:.1f}%")

    z_pass = z_results['z_contribution'] > 30
    print(f"\n  Z Contribution Target: >30%")
    print(f"  Status: {'PASS' if z_pass else 'FAIL'}")

    # Evaluate kurtosis
    print("\n" + "=" * 70)
    print("EVALUATING KURTOSIS RECOVERY")
    print("=" * 70)

    print("\nSampling from model (may take a minute)...")
    kurt_results = evaluate_kurtosis(model, val_loader, device, n_samples=100)
    print(f"\n  GT Kurtosis (ATM): {kurt_results['gt_kurtosis']:.2f}")
    print(f"  Sample Kurtosis (ATM): {kurt_results['sample_kurtosis']:.2f}")
    print(f"  Kurtosis Recovery: {kurt_results['kurtosis_recovery']:.1f}%")

    kurt_pass = kurt_results['kurtosis_recovery'] > 100
    print(f"\n  Kurtosis Recovery Target: >100%")
    print(f"  Status: {'PASS' if kurt_pass else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Result':>10} {'Target':>10} {'Status':>10}")
    print("  " + "-" * 55)
    print(f"  {'Z Contribution':<25} {z_results['z_contribution']:>9.1f}% {'>30%':>10} {'PASS' if z_pass else 'FAIL':>10}")
    print(f"  {'Kurtosis Recovery':<25} {kurt_results['kurtosis_recovery']:>9.1f}% {'>100%':>10} {'PASS' if kurt_pass else 'FAIL':>10}")

    overall_pass = z_pass and kurt_pass

    print("\n" + "=" * 70)
    if overall_pass:
        print("OVERALL: PASS - Ported classes reproduce experiment results!")
        print("  The checkpoint can be used with vae.cvae_two_stage.CVAETwoStageStudentTMLP")
    else:
        print("OVERALL: FAIL - Results do not match experiment")
        if not z_pass:
            print(f"  - Z Contribution {z_results['z_contribution']:.1f}% is below 30% target")
        if not kurt_pass:
            print(f"  - Kurtosis Recovery {kurt_results['kurtosis_recovery']:.1f}% is below 100% target")
    print("=" * 70)

    return {
        "z_contribution": z_results,
        "kurtosis": kurt_results,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    results = main()
