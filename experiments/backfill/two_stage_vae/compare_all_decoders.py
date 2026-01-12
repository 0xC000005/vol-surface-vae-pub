"""
Compare All Decoder Architectures

This script evaluates all trained decoder variants using consistent methodology
to enable fair comparison. Uses the same data and evaluation functions as
comprehensive_oracle_analysis.py.

Models compared:
1. StudentTMLPDecoder (baseline) - 139.9% kurtosis, 0% ctx
2. StudentTDualPathDecoder - additive combination
3. StudentTGatedResidualDecoder - gated residual

All use the same:
- Full dataset for evaluation (not just validation split)
- Batch size and limits
- Kurtosis/ACF computation methodology

Usage:
    python experiments/backfill/two_stage_vae/compare_all_decoders.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kurtosis
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import (
    CVAETwoStageStudentTMLP,
    CVAETwoStageDualPath,
    CVAETwoStageGatedResidual,
    CVAETwoStageDualPathAR,
)


CONTEXT_LEN = 20


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def create_dataloader(log_returns, context_len, batch_size, shuffle=False):
    """Create dataloader - uses shuffle=False for deterministic eval."""
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


def evaluate_bottleneck(model, val_loader, device, model_type="mlp"):
    """
    Evaluate bottleneck capacity - exactly matching comprehensive methodology.
    """
    model.eval()

    mse_full = []
    mse_ctx_only = []
    mse_z_only = []

    # Check if this is an AR decoder (needs prev_x)
    is_ar_model = hasattr(model.decoder, 'get_phi')

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            ctx_emb = model.ctx_encoder(batch)
            z_mean, z_logvar, z = model.main_encoder(batch)

            target = batch_data[:, 1:]

            # For AR decoder, need prev_x
            prev_x = batch_data.clone() if is_ar_model else None

            # Full model
            if is_ar_model:
                mean_full, _, _, _ = model.decoder(ctx_emb, z, prev_x=prev_x, sample=False)
            else:
                mean_full, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            pred_full = mean_full[:, :-1]
            mse_full.append(((pred_full - target) ** 2).mean().item())

            # Ctx only (z=0)
            z_zero = torch.zeros_like(z)
            if is_ar_model:
                mean_ctx, _, _, _ = model.decoder(ctx_emb, z_zero, prev_x=prev_x, sample=False)
            else:
                mean_ctx, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
            pred_ctx = mean_ctx[:, :-1]
            mse_ctx_only.append(((pred_ctx - target) ** 2).mean().item())

            # Z only (ctx=0)
            ctx_zero = torch.zeros_like(ctx_emb)
            if is_ar_model:
                mean_z, _, _, _ = model.decoder(ctx_zero, z, prev_x=prev_x, sample=False)
            else:
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
    """
    Evaluate kurtosis - exactly matching comprehensive methodology.
    """
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
        "gt_kurtosis_grid": gt_kurtosis,
        "model_kurtosis_grid": model_kurtosis,
    }


def evaluate_acf(model, val_loader, device, n_samples=100):
    """Evaluate ACF preservation at ATM point."""
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

            # ATM point (2,2) across all timesteps
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy()
            gt_atm = target[:, :, 2, 2].cpu().numpy()

            all_samples.append(samples_atm)
            all_gt.append(gt_atm)

    # Compute ACFs
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


def load_model(model_class, checkpoint_path, device):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = model_class(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(model, val_loader, device, name):
    """Run full evaluation on a model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    # Bottleneck
    print("  Computing bottleneck capacity...")
    bottleneck = evaluate_bottleneck(model, val_loader, device)

    # Kurtosis
    print("  Computing kurtosis (sampling)...")
    kurt = evaluate_kurtosis(model, val_loader, device, n_samples=200)

    # ACF
    print("  Computing ACF...")
    acf = evaluate_acf(model, val_loader, device, n_samples=100)

    return {
        "name": name,
        "bottleneck": bottleneck,
        "kurtosis": kurt,
        "acf": acf,
    }


def main():
    print("=" * 70)
    print("DECODER ARCHITECTURE COMPARISON")
    print("=" * 70)
    print("\nUsing comprehensive methodology (full dataset, consistent params)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data - use FULL dataset like comprehensive analysis
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)
    print(f"Full data shape: {log_returns.shape}")

    # Create dataloader from FULL dataset
    val_loader = create_dataloader(log_returns, CONTEXT_LEN, batch_size=32, shuffle=False)

    # Define models to compare
    models_to_eval = [
        {
            "name": "StudentTMLPDecoder (baseline)",
            "class": CVAETwoStageStudentTMLP,
            "path": "models/backfill/two_stage/student_t/student_t_best.pt",
        },
    ]

    # Check for optional models
    dual_path_path = Path("models/backfill/two_stage/dual_path/dual_path_best.pt")
    if dual_path_path.exists():
        models_to_eval.append({
            "name": "StudentTDualPathDecoder",
            "class": CVAETwoStageDualPath,
            "path": str(dual_path_path),
        })

    gated_path = Path("models/backfill/two_stage/gated_residual/gated_residual_best.pt")
    if gated_path.exists():
        models_to_eval.append({
            "name": "StudentTGatedResidualDecoder",
            "class": CVAETwoStageGatedResidual,
            "path": str(gated_path),
        })

    warmstart_path = Path("models/backfill/two_stage/warmstart_context/warmstart_context_best.pt")
    if warmstart_path.exists():
        models_to_eval.append({
            "name": "WarmStart Gated Residual",
            "class": CVAETwoStageGatedResidual,
            "path": str(warmstart_path),
        })

    dual_path_ar_path = Path("models/backfill/two_stage/dual_path_ar/dual_path_ar_best.pt")
    if dual_path_ar_path.exists():
        models_to_eval.append({
            "name": "DualPath + AR(1) (ACF fix)",
            "class": CVAETwoStageDualPathAR,
            "path": str(dual_path_ar_path),
        })

    # Evaluate each model
    results = []
    for model_info in models_to_eval:
        path = Path(model_info["path"])
        if not path.exists():
            print(f"\nSkipping {model_info['name']} - checkpoint not found: {path}")
            continue

        try:
            model, _ = load_model(model_info["class"], model_info["path"], device)
            result = evaluate_model(model, val_loader, device, model_info["name"])
            results.append(result)
        except Exception as e:
            print(f"\nError loading {model_info['name']}: {e}")
            continue

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Header
    print(f"\n  {'Model':<35} {'Z%':>8} {'Ctx%':>8} {'Kurt%':>8} {'ACF%':>8}")
    print("  " + "-" * 70)

    for r in results:
        name = r["name"][:35]
        z_pct = r["bottleneck"]["z_contribution_pct"]
        ctx_pct = r["bottleneck"]["ctx_contribution_pct"]
        kurt_pct = r["kurtosis"]["mean_recovery"] * 100
        acf_pct = r["acf"]["acf_preservation"] * 100

        print(f"  {name:<35} {z_pct:>7.1f}% {ctx_pct:>7.1f}% {kurt_pct:>7.1f}% {acf_pct:>7.1f}%")

    # Detailed results
    print("\n" + "=" * 70)
    print("DETAILED METRICS")
    print("=" * 70)

    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Bottleneck:")
        print(f"    MSE (full):     {r['bottleneck']['mse_full']:.6f}")
        print(f"    MSE (ctx=0):    {r['bottleneck']['mse_ctx_only']:.6f}")
        print(f"    MSE (z=0):      {r['bottleneck']['mse_z_only']:.6f}")
        print(f"    Z contribution: {r['bottleneck']['z_contribution_pct']:.1f}%")
        print(f"    Ctx contribution: {r['bottleneck']['ctx_contribution_pct']:.1f}%")
        print(f"  Kurtosis:")
        print(f"    GT ATM:         {r['kurtosis']['atm_gt_kurtosis']:.2f}")
        print(f"    Model ATM:      {r['kurtosis']['atm_model_kurtosis']:.2f}")
        print(f"    Mean Recovery:  {r['kurtosis']['mean_recovery']*100:.1f}%")
        print(f"  ACF:")
        print(f"    GT lag-1:       {r['acf']['gt_acf_lag1']:.4f}")
        print(f"    Model lag-1:    {r['acf']['model_acf_lag1']:.4f}")
        print(f"    Preservation:   {r['acf']['acf_preservation']*100:.1f}%")

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)
    print(f"\n  Target metrics:")
    print(f"    Kurtosis Recovery: >100% (don't regress)")
    print(f"    Ctx Contribution:  >5% (context is useful)")
    print(f"    ACF Preservation:  >30% (temporal structure)")

    best_combined = None
    best_score = -float('inf')

    for r in results:
        kurt_pass = r['kurtosis']['mean_recovery'] > 1.0
        ctx_pass = r['bottleneck']['ctx_contribution_pct'] > 5
        acf_pass = abs(r['acf']['acf_preservation']) > 0.3

        print(f"\n  {r['name']}:")
        print(f"    Kurtosis: {'PASS' if kurt_pass else 'FAIL'} ({r['kurtosis']['mean_recovery']*100:.1f}%)")
        print(f"    Context:  {'PASS' if ctx_pass else 'FAIL'} ({r['bottleneck']['ctx_contribution_pct']:.1f}%)")
        print(f"    ACF:      {'PASS' if acf_pass else 'FAIL'} ({r['acf']['acf_preservation']*100:.1f}%)")

        # Score: prioritize kurtosis, then ctx, then acf
        score = (
            (100 if kurt_pass else 0) +
            (50 if ctx_pass else 0) +
            (25 if acf_pass else 0)
        )
        if score > best_score:
            best_score = score
            best_combined = r['name']

    print(f"\n  Best overall: {best_combined}")

    return results


if __name__ == "__main__":
    results = main()
