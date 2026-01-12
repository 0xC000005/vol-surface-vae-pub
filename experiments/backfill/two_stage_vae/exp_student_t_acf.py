"""
Student-t with Direct ACF Loss: Elegant Solution for 50% ACF Preservation

Goal: Achieve 50% ACF preservation while maintaining Student-t's directional accuracy (4.2% CI violations)

Approach (Bitter Lesson):
- Don't add architectural complexity (no gates, no momentum networks)
- Just add ACF loss to training - let the model learn to preserve temporal correlations
- The model will figure out *how* to achieve ACF preservation on its own

Loss: NLL + λ_kl * KL + λ_acf * ACF_loss

Usage:
    python experiments/backfill/two_stage_vae/exp_student_t_acf.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def evaluate_acf_preservation(model, val_loader, config, n_samples=50):
    """Evaluate ACF preservation of model samples vs ground truth."""
    device = config["device"]
    model.eval()

    all_samples = []
    all_gt = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Generate samples
            samples = model.sample(batch, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            # Get ATM predictions (grid point 2,2)
            samples_atm = samples[:, :, :, 2, 2].cpu().numpy()  # (n_samples, B, T)
            gt_atm = batch_data[:, :, 2, 2].cpu().numpy()  # (B, T)

            all_samples.append(samples_atm)
            all_gt.append(gt_atm)

    # Compute ACF for samples and GT
    sample_acfs = []
    for samples_batch in all_samples:
        for s in range(samples_batch.shape[0]):  # n_samples
            for b in range(samples_batch.shape[1]):  # batch
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


def evaluate_kurtosis(model, val_loader, config, n_samples=50):
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

    recovery = sample_kurt / gt_kurt * 100 if abs(gt_kurt) > 1e-8 else 0

    return {
        "gt_kurtosis": float(gt_kurt),
        "model_kurtosis": float(sample_kurt),
        "kurtosis_recovery": float(recovery),
    }


def train_student_t_with_acf(
    model, train_loader, val_loader, config,
    epochs=100, lambda_acf=0.1, phase1_ratio=0.8
):
    """
    Train Student-t model with direct ACF loss using curriculum learning.

    Phase 1 (80%): Train with NLL only to establish fat tails
    Phase 2 (20%): Add ACF loss to preserve temporal correlations

    This curriculum prevents ACF loss from interfering with fat-tail learning.
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    # Curriculum: Phase 1 without ACF, Phase 2 with ACF
    phase1_epochs = int(epochs * phase1_ratio)
    phase2_epochs = epochs - phase1_epochs

    best_loss = float('inf')
    best_state = None

    print(f"\n{'='*70}")
    print(f"Training Student-t + ACF Loss (λ_acf={lambda_acf})")
    print(f"Phase 1: {phase1_epochs} epochs (NLL only)")
    print(f"Phase 2: {phase2_epochs} epochs (NLL + ACF)")
    print(f"{'='*70}")

    # Phase 1: NLL only
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_nlls = []
        train_acfs = []

        # Curriculum: ACF loss only in Phase 2
        current_lambda_acf = lambda_acf if epoch >= phase1_epochs else 0.0

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Forward pass
            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            # Target and prediction
            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # NLL loss (Student-t)
            nll_loss = model.decoder.compute_student_t_nll(pred, target, factor, log_diag)

            # KL loss
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            # ACF loss on ATM grid point (only in Phase 2)
            if current_lambda_acf > 0:
                pred_atm = pred[:, :, 2, 2]  # (B, T-1)
                target_atm = target[:, :, 2, 2]  # (B, T-1)

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
            total_loss = nll_loss + kl_weight * kl_loss + current_lambda_acf * acf_loss

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
            phase = "P1" if epoch < phase1_epochs else "P2"
            print(f"  [{phase}] Epoch {epoch+1}/{epochs}: "
                  f"NLL={np.mean(train_nlls):.4f}, "
                  f"ACF_loss={np.mean(train_acfs):.4f}, "
                  f"Val={val_loss:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_lambda_sweep():
    """Run sweep over λ_acf values to find optimal balance."""
    print("=" * 70)
    print("Student-t + ACF Loss: λ_acf Sweep Experiment")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)

    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]

    context_len = 20
    batch_size = 64

    train_loader = create_dataloader(train_data, context_len, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, context_len, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    config = TWO_STAGE_CONFIG.copy()
    config["device"] = device

    # λ_acf values to sweep - use very low values to preserve kurtosis
    # Phase 1 is 90% of training (NLL only), Phase 2 is 10% (NLL + ACF)
    lambda_values = [0.02, 0.03, 0.05, 0.08]

    results = []

    for lambda_acf in lambda_values:
        print(f"\n{'='*70}")
        print(f"Training with λ_acf = {lambda_acf} (Phase 1: 90%, Phase 2: 10%)")
        print(f"{'='*70}")

        # Create fresh model
        model = CVAETwoStageStudentTMLP(config)

        # Train with more epochs and even longer Phase 1 for better fat-tail preservation
        model = train_student_t_with_acf(
            model, train_loader, val_loader, config,
            epochs=200, lambda_acf=lambda_acf, phase1_ratio=0.9
        )

        # Evaluate
        print("\nEvaluating...")
        acf_results = evaluate_acf_preservation(model, val_loader, config)
        kurt_results = evaluate_kurtosis(model, val_loader, config)

        result = {
            "lambda_acf": lambda_acf,
            "acf_preservation": acf_results["acf_preservation"],
            "gt_acf": acf_results["gt_acf_lag1"],
            "model_acf": acf_results["model_acf_lag1"],
            "kurtosis_recovery": kurt_results["kurtosis_recovery"],
        }
        results.append(result)

        print(f"\n  Results for λ_acf={lambda_acf}:")
        print(f"    ACF Preservation: {result['acf_preservation']*100:.1f}%")
        print(f"    Kurtosis Recovery: {result['kurtosis_recovery']:.1f}%")

        # Save best model if ACF > 40%
        if result["acf_preservation"] > 0.4:
            save_path = Path("models/backfill/two_stage/student_t_acf")
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "lambda_acf": lambda_acf,
                "acf_preservation": result["acf_preservation"],
                "kurtosis_recovery": result["kurtosis_recovery"],
            }, save_path / f"student_t_acf_lambda{lambda_acf}.pt")
            print(f"    Model saved to {save_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: λ_acf Sweep Results")
    print("=" * 70)
    print(f"{'λ_acf':>10} | {'ACF Pres.':>12} | {'Kurtosis Rec.':>14}")
    print("-" * 45)
    for r in results:
        print(f"{r['lambda_acf']:>10.2f} | {r['acf_preservation']*100:>11.1f}% | {r['kurtosis_recovery']:>13.1f}%")

    # Find best λ_acf (target: ACF > 50% with good kurtosis)
    best = max(results, key=lambda x: x['acf_preservation'] if x['kurtosis_recovery'] > 80 else 0)
    print(f"\nBest: λ_acf={best['lambda_acf']} → ACF={best['acf_preservation']*100:.1f}%, Kurt={best['kurtosis_recovery']:.1f}%")

    return results


def evaluate_baseline():
    """Evaluate pure Student-t baseline (no ACF loss) for comparison."""
    print("=" * 70)
    print("Student-t Baseline Evaluation (No ACF Loss)")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, _ = to_log_returns(surfaces)

    # Split data
    train_end = int(len(log_returns) * 0.7)
    val_end = int(len(log_returns) * 0.85)

    train_data = log_returns[:train_end]
    val_data = log_returns[train_end:val_end]

    context_len = 20
    batch_size = 64

    train_loader = create_dataloader(train_data, context_len, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, context_len, batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TWO_STAGE_CONFIG.copy()
    config["device"] = device

    # Create and train baseline model (no ACF loss = λ_acf = 0)
    print("\nTraining baseline Student-t model...")
    model = CVAETwoStageStudentTMLP(config)
    model = train_student_t_with_acf(
        model, train_loader, val_loader, config,
        epochs=150, lambda_acf=0.0, phase1_ratio=1.0  # All NLL, no ACF
    )

    # Evaluate
    print("\nEvaluating baseline...")
    acf_results = evaluate_acf_preservation(model, val_loader, config)
    kurt_results = evaluate_kurtosis(model, val_loader, config)

    print(f"\n{'='*70}")
    print("BASELINE Student-t Results:")
    print(f"  ACF Preservation: {acf_results['acf_preservation']*100:.1f}%")
    print(f"  Kurtosis Recovery: {kurt_results['kurtosis_recovery']:.1f}%")
    print(f"{'='*70}")

    return {
        "acf_preservation": acf_results["acf_preservation"],
        "kurtosis_recovery": kurt_results["kurtosis_recovery"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true", help="Only evaluate baseline")
    args = parser.parse_args()

    if args.baseline_only:
        # Just evaluate baseline
        evaluate_baseline()
    else:
        # First evaluate baseline
        baseline = evaluate_baseline()
        print("\n\n")
        # Then run sweep
        run_lambda_sweep()
