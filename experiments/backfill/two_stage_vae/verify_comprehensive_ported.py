"""
Run Comprehensive Analysis with Ported Classes

This verifies the ported CVAETwoStageStudentTMLP produces identical results
to the experiment's CVAETwoStageStudentT by running the same analysis.

Expected Results (from experiment):
- Z Contribution: 39.5%
- Kurtosis Recovery: 135.8%

Usage:
    python experiments/backfill/two_stage_vae/verify_comprehensive_ported.py
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


CONTEXT_LEN = 20


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


def test_bottleneck_capacity(model, log_returns, device):
    """
    Exact copy of comprehensive_oracle_analysis.py bottleneck test.

    Tests MSE when using:
    - Full model (ctx_emb + z)
    - Context only (ctx_emb + z=0)
    - Z only (ctx_emb=0 + z)
    """
    print("\n" + "=" * 70)
    print("BOTTLENECK CAPACITY TEST (using comprehensive methodology)")
    print("=" * 70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=64, shuffle=False)

    mse_full = []
    mse_ctx_only = []
    mse_z_only = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            # Get embeddings
            ctx_emb = model.ctx_encoder({"surface": batch_data})
            z_mean, z_logvar, z = model.main_encoder({"surface": batch_data})

            # Target
            target = batch_data[:, 1:]

            # Full model
            mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full.append(((pred_full - target) ** 2).mean().item())

            # Context only (z=0)
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

    # z_contribution: how much does z reduce error compared to ctx-only?
    z_contribution = (mse_ctx_avg - mse_full_avg) / mse_ctx_avg * 100 if mse_ctx_avg > 0 else 0
    # ctx_contribution: how much does ctx reduce error compared to z-only?
    ctx_contribution = (mse_z_avg - mse_full_avg) / mse_z_avg * 100 if mse_z_avg > 0 else 0

    print(f"\n  MSE (full model):     {mse_full_avg:.6f}")
    print(f"  MSE (ctx only, z=0):  {mse_ctx_avg:.6f}")
    print(f"  MSE (z only, ctx=0):  {mse_z_avg:.6f}")
    print(f"\n  Z Contribution:       {z_contribution:.1f}%")
    print(f"  Ctx Contribution:     {ctx_contribution:.1f}%")

    return {
        "mse_full": mse_full_avg,
        "mse_ctx_only": mse_ctx_avg,
        "mse_z_only": mse_z_avg,
        "z_contribution_pct": z_contribution,
        "ctx_contribution_pct": ctx_contribution,
    }


def analyze_kurtosis_per_grid(model, log_returns, device, n_samples=200):
    """
    Exact copy of comprehensive_oracle_analysis.py kurtosis test.
    """
    print("\n" + "=" * 70)
    print("KURTOSIS PER GRID ANALYSIS (using comprehensive methodology)")
    print("=" * 70)

    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for batch_idx, (batch_data,) in enumerate(val_loader):
            if batch_idx >= 20:  # Limit for speed (same as comprehensive)
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
    recovery = np.clip(recovery, 0, 2)  # Cap at 200% for display

    print("\n  GT Kurtosis (Fisher):")
    print(np.round(gt_kurtosis, 1))

    print("\n  Model Kurtosis (Fisher):")
    print(np.round(model_kurtosis, 2))

    print("\n  Recovery Ratio (%):")
    print(np.round(recovery * 100, 1))

    print(f"\n  Mean Recovery:      {recovery.mean()*100:.1f}%")
    print(f"  ATM GT Kurtosis:    {gt_kurtosis[2, 2]:.1f}")
    print(f"  ATM Model Kurtosis: {model_kurtosis[2, 2]:.2f}")

    return {
        "mean_recovery": float(recovery.mean()),
        "atm_gt_kurtosis": float(gt_kurtosis[2, 2]),
        "atm_model_kurtosis": float(model_kurtosis[2, 2]),
    }


def main():
    print("=" * 70)
    print("VERIFY PORTED CLASSES WITH COMPREHENSIVE METHODOLOGY")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load checkpoint
    model_path = "models/backfill/two_stage/student_t/student_t_best.pt"
    print(f"\nLoading checkpoint from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    # Create model using ported class
    print("\nCreating model using ported CVAETwoStageStudentTMLP class...")
    model = CVAETwoStageStudentTMLP(config)
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

    # Run bottleneck test
    bottleneck_results = test_bottleneck_capacity(model, log_returns, device)

    # Run kurtosis analysis
    print("\nSampling from model (may take a minute)...")
    kurtosis_results = analyze_kurtosis_per_grid(model, log_returns, device, n_samples=200)

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Ported':>12} {'Expected':>12} {'Match?':>10}")
    print("  " + "-" * 60)

    z_contrib = bottleneck_results['z_contribution_pct']
    z_expected = 39.5
    z_match = abs(z_contrib - z_expected) < 5  # Within 5% tolerance

    kurt_recov = kurtosis_results['mean_recovery'] * 100
    kurt_expected = 135.8
    kurt_match = abs(kurt_recov - kurt_expected) < 20  # Within 20% tolerance

    print(f"  {'Z Contribution':<25} {z_contrib:>11.1f}% {z_expected:>11.1f}% {'OK' if z_match else 'DIFF':>10}")
    print(f"  {'Kurtosis Recovery':<25} {kurt_recov:>11.1f}% {kurt_expected:>11.1f}% {'OK' if kurt_match else 'DIFF':>10}")

    overall_pass = z_match and kurt_match

    print("\n" + "=" * 70)
    if overall_pass:
        print("OVERALL: PASS - Ported classes produce matching results!")
        print("  The checkpoint can be used with vae.cvae_two_stage.CVAETwoStageStudentTMLP")
    else:
        print("OVERALL: RESULTS DIFFER (some variance expected due to sampling)")
        if not z_match:
            print(f"  - Z Contribution: {z_contrib:.1f}% vs expected {z_expected:.1f}%")
        if not kurt_match:
            print(f"  - Kurtosis Recovery: {kurt_recov:.1f}% vs expected {kurt_expected:.1f}%")
    print("=" * 70)

    return {
        "bottleneck": bottleneck_results,
        "kurtosis": kurtosis_results,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    results = main()
