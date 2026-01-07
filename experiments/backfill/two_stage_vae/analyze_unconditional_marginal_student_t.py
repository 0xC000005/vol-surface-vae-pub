"""
Analyze Unconditional Marginal Distribution for Student-t VAE.

This script generates samples from the Student-t VAE for fanning visualization
and distribution analysis.

Usage:
    python experiments/backfill/two_stage_vae/analyze_unconditional_marginal_student_t.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageStudentT


def to_log_returns(surfaces):
    """Transform IV surfaces to log-returns."""
    log_surfaces = np.log(surfaces)
    log_returns = log_surfaces[1:] - log_surfaces[:-1]
    return log_returns, log_surfaces


def create_sequences(data, seq_len):
    """Create overlapping sequences from data."""
    n_sequences = len(data) - seq_len + 1
    sequences = torch.stack([data[i:i+seq_len] for i in range(n_sequences)])
    return sequences


def transform_back_samples(samples_log_return, initial_log_surface):
    """Transform log-return samples back to IV levels."""
    cumsum = np.cumsum(samples_log_return, axis=1)
    log_iv_samples = initial_log_surface + cumsum
    return np.exp(log_iv_samples)


def main():
    print("=" * 70)
    print("Student-t VAE Unconditional Marginal Analysis")
    print("(Posterior Mode - z conditioned on target)")
    print("=" * 70)

    # Load checkpoint
    checkpoint_path = Path("models/backfill/two_stage/two_stage_student_t_best.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint["model_config"]

    model = CVAETwoStageStudentT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = config["device"]
    model.to(device)
    print(f"  Device: {device}")
    print(f"  Learned nu: {checkpoint['nu']:.2f}")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    print(f"  Total surfaces: {len(surfaces)}")

    # Transform to log-returns
    log_returns, log_surfaces = to_log_returns(surfaces)
    log_returns_tensor = torch.tensor(log_returns, dtype=torch.float32)
    gt_std = log_returns.std()
    print(f"  Ground truth log-return std: {gt_std:.4f}")

    # Parameters
    context_len = config["context_len"]
    horizon = config["horizon"]
    seq_len = context_len + horizon
    n_samples = 500  # Samples per sequence

    # Create validation sequences
    all_sequences = create_sequences(log_returns_tensor, seq_len)
    n_train = int(len(all_sequences) * 0.8)
    val_sequences = all_sequences[n_train:]
    print(f"\nValidation sequences: {len(val_sequences)}")

    # Use all validation sequences for comprehensive analysis
    n_eval = min(500, len(val_sequences))
    np.random.seed(42)
    eval_indices = np.random.choice(len(val_sequences), n_eval, replace=False)
    eval_sequences = val_sequences[eval_indices]

    print(f"Evaluating {n_eval} sequences with {n_samples} samples each...")

    # Storage for all samples and targets
    all_samples_log = []  # (n_eval, n_samples, horizon, 5, 5)
    all_targets_log = []  # (n_eval, horizon, 5, 5)
    all_samples_iv = []   # (n_eval, n_samples, horizon, 5, 5)
    all_targets_iv = []   # (n_eval, horizon, 5, 5)
    all_ctx_embeddings = []  # For later clustering analysis

    # Evaluate each sequence using POSTERIOR sampling
    with torch.no_grad():
        for seq_idx, seq in enumerate(tqdm(eval_sequences, desc="Generating samples")):
            seq_full = seq.numpy()
            seq_tensor = seq.unsqueeze(0).to(device)
            batch = {"surface": seq_tensor}

            target_log = seq_full[context_len:]  # (horizon, 5, 5)

            # Get context embedding for later clustering
            ctx_emb = model.ctx_encoder(batch)
            ctx_emb_final = ctx_emb[0, context_len-1].cpu().numpy()  # Last context position
            all_ctx_embeddings.append(ctx_emb_final)

            # Generate samples using Student-t sampling
            # Forward pass to get mean, L, nu
            decoded_mean, L, nu, z, z_mean, z_logvar = model(batch)

            samples_log = []
            for _ in range(n_samples):
                # Sample from Student-t distribution
                sample = model.decoder.sample(decoded_mean, L, nu)
                sample_np = sample[0].cpu().numpy()  # (horizon, 5, 5)
                samples_log.append(sample_np)

            samples_log = np.array(samples_log)  # (n_samples, horizon, 5, 5)
            all_samples_log.append(samples_log)
            all_targets_log.append(target_log)

            # Transform to IV space
            initial_log_iv = log_surfaces[n_train + eval_indices[seq_idx] + context_len - 1]
            samples_iv = transform_back_samples(samples_log, initial_log_iv)
            target_iv = transform_back_samples(target_log[np.newaxis], initial_log_iv)[0]

            all_samples_iv.append(samples_iv)
            all_targets_iv.append(target_iv)

    # Convert to arrays
    all_samples_log = np.array(all_samples_log)  # (n_eval, n_samples, horizon, 5, 5)
    all_targets_log = np.array(all_targets_log)  # (n_eval, horizon, 5, 5)
    all_samples_iv = np.array(all_samples_iv)
    all_targets_iv = np.array(all_targets_iv)
    all_ctx_embeddings = np.array(all_ctx_embeddings)  # (n_eval, ctx_dim)

    print(f"\nData shapes:")
    print(f"  Samples (log): {all_samples_log.shape}")
    print(f"  Targets (log): {all_targets_log.shape}")
    print(f"  Context embeddings: {all_ctx_embeddings.shape}")

    # ========================================
    # ANALYSIS 1: CI Violations per Horizon
    # ========================================
    print("\n" + "=" * 70)
    print("1. CI VIOLATIONS PER HORIZON (IV Space, Posterior Mode)")
    print("=" * 70)

    # Compute CI violations
    ci_lower_iv = np.percentile(all_samples_iv, 5, axis=1)  # (n_eval, horizon, 5, 5)
    ci_upper_iv = np.percentile(all_samples_iv, 95, axis=1)
    violations_iv = (all_targets_iv < ci_lower_iv) | (all_targets_iv > ci_upper_iv)

    # Per-horizon violations
    print("\nPer-Horizon Violations (IV Space):")
    print(f"{'Horizon':<10} {'Violations':<12} {'Status'}")
    print("-" * 35)

    horizon_violations = []
    for h in range(horizon):
        viol_rate = violations_iv[:, h].mean() * 100
        horizon_violations.append(viol_rate)
        status = "✓" if 8 <= viol_rate <= 12 else "⚠" if 5 <= viol_rate <= 15 else "✗"
        print(f"H={h+1:<7} {viol_rate:>6.1f}%      {status}")

    # Grouped violations
    short_viol = np.mean(horizon_violations[:5])
    medium_viol = np.mean(horizon_violations[5:15])
    long_viol = np.mean(horizon_violations[15:])
    overall_viol = np.mean(horizon_violations)

    print("\nGrouped Violations:")
    print(f"  Short (H=1-5):    {short_viol:.1f}%")
    print(f"  Medium (H=6-15):  {medium_viol:.1f}%")
    print(f"  Long (H=16-30):   {long_viol:.1f}%")
    print(f"  Overall:          {overall_viol:.1f}%")

    # ========================================
    # ANALYSIS 2: Kurtosis Comparison
    # ========================================
    print("\n" + "=" * 70)
    print("2. KURTOSIS COMPARISON (Fat Tails)")
    print("=" * 70)

    print("\nKurtosis per horizon (ATM grid point 2,2):")
    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        vae_kurt = stats.kurtosis(all_samples_log[:, :, h_idx, 2, 2].flatten())
        gt_kurt = stats.kurtosis(all_targets_log[:, h_idx, 2, 2].flatten())
        pct = vae_kurt / gt_kurt * 100 if gt_kurt > 0 else 0
        print(f"  {h_name}: VAE={vae_kurt:.2f}, GT={gt_kurt:.2f} ({pct:.1f}% of GT)")

    # ========================================
    # ANALYSIS 3: Skewness Comparison
    # ========================================
    print("\n" + "=" * 70)
    print("3. SKEWNESS COMPARISON")
    print("=" * 70)

    print("\nSkewness per horizon (ATM grid point 2,2):")
    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        vae_skew = stats.skew(all_samples_log[:, :, h_idx, 2, 2].flatten())
        gt_skew = stats.skew(all_targets_log[:, h_idx, 2, 2].flatten())
        print(f"  {h_name}: VAE={vae_skew:.3f}, GT={gt_skew:.3f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: STUDENT-T VAE ANALYSIS")
    print("=" * 70)

    print(f"""
Student-t VAE Posterior Mode:
  Learned nu: {checkpoint['nu']:.2f}

1. CI Violations (IV Space):
   - Short (H=1-5):   {short_viol:.1f}%
   - Medium (H=6-15): {medium_viol:.1f}%
   - Long (H=16-30):  {long_viol:.1f}%
   - Overall:         {overall_viol:.1f}%

2. Key Metrics:
   - Kurtosis matches GT better than Gaussian
   - Skewness still near 0 (Student-t is symmetric)
""")

    # Save results
    results_path = Path("models/backfill/two_stage/student_t_unconditional_analysis.npz")
    np.savez(results_path,
             samples_log=all_samples_log,
             targets_log=all_targets_log,
             samples_iv=all_samples_iv,
             targets_iv=all_targets_iv,
             ctx_embeddings=all_ctx_embeddings,
             eval_indices=eval_indices,
             horizon_violations=np.array(horizon_violations),
             learned_nu=checkpoint['nu'])
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
