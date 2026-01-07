"""
Analyze Unconditional Marginal Distribution Matching.

This script validates that the heteroscedastic VAE (posterior mode) produces
marginal distributions that match ground truth across all horizons.

Metrics:
- CI violations per horizon group (short/medium/long)
- Distribution statistics (mean, std, skewness, kurtosis)
- KS test for distribution similarity

Usage:
    python experiments/backfill/prior_encoder_ablation/analyze_unconditional_marginal.py
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import sys
sys.path.insert(0, ".")

from vae.cvae_two_stage import CVAETwoStageHeteroscedastic


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
    print("Unconditional Marginal Distribution Analysis")
    print("(Posterior Mode - z conditioned on target)")
    print("=" * 70)

    # Load checkpoint
    checkpoint_path = Path("models/backfill/two_stage/two_stage_log_return_nll_best.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    config = checkpoint["model_config"]

    model = CVAETwoStageHeteroscedastic(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = config["device"]
    print(f"  Device: {device}")

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

            target_log = seq_full[context_len:]  # (horizon, 5, 5)

            # Get context embedding for later clustering
            ctx_emb = model.ctx_encoder({"surface": seq_tensor})
            ctx_emb_final = ctx_emb[0, context_len-1].cpu().numpy()  # Last context position
            all_ctx_embeddings.append(ctx_emb_final)

            # Generate samples using POSTERIOR (z conditioned on target)
            samples_log = []
            for _ in range(n_samples):
                # Full forward pass - z sees the target
                surface_mean, surface_logvar, z_mean, z_logvar, z = model.forward(
                    {"surface": seq_tensor}
                )

                pred_mean = surface_mean.cpu().numpy()[0]  # (horizon, 5, 5)
                pred_logvar = surface_logvar.cpu().numpy()[0]

                # Sample from predicted distribution
                pred_std = np.exp(0.5 * pred_logvar)
                sample = pred_mean + pred_std * np.random.randn(*pred_mean.shape)
                samples_log.append(sample)

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
    # ANALYSIS 2: Distribution Statistics
    # ========================================
    print("\n" + "=" * 70)
    print("2. DISTRIBUTION STATISTICS (Log-Return Space)")
    print("=" * 70)

    # Flatten samples and targets for distribution comparison
    # Compare VAE sample distribution to ground truth log-return distribution

    print("\n2a. Sample Mean Comparison:")
    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        sample_mean = all_samples_log[:, :, h_idx].mean()
        target_mean = all_targets_log[:, h_idx].mean()
        print(f"  {h_name}: VAE mean={sample_mean:.6f}, GT mean={target_mean:.6f}")

    print("\n2b. Sample Std Comparison:")
    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        sample_std = all_samples_log[:, :, h_idx].std()
        target_std = all_targets_log[:, h_idx].std()
        print(f"  {h_name}: VAE std={sample_std:.4f}, GT std={target_std:.4f}, ratio={sample_std/target_std*100:.1f}%")

    # ========================================
    # ANALYSIS 3: KS Test for Distribution Similarity
    # ========================================
    print("\n" + "=" * 70)
    print("3. KOLMOGOROV-SMIRNOV TEST (Distribution Similarity)")
    print("=" * 70)

    print("\nKS test compares VAE sample distribution to GT target distribution")
    print("p-value > 0.05 suggests distributions are similar")
    print()

    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        # Flatten samples and targets
        vae_samples = all_samples_log[:, :, h_idx].flatten()
        gt_targets = all_targets_log[:, h_idx].flatten()

        # KS test
        ks_stat, p_value = stats.ks_2samp(vae_samples, gt_targets)
        status = "✓ Similar" if p_value > 0.05 else "✗ Different"
        print(f"  {h_name}: KS stat={ks_stat:.4f}, p-value={p_value:.4f} → {status}")

    # ========================================
    # ANALYSIS 4: Skewness and Kurtosis
    # ========================================
    print("\n" + "=" * 70)
    print("4. HIGHER MOMENTS (Skewness, Kurtosis)")
    print("=" * 70)

    print("\n4a. Skewness (0 = symmetric):")
    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        vae_skew = stats.skew(all_samples_log[:, :, h_idx].flatten())
        gt_skew = stats.skew(all_targets_log[:, h_idx].flatten())
        print(f"  {h_name}: VAE skew={vae_skew:.3f}, GT skew={gt_skew:.3f}")

    print("\n4b. Kurtosis (0 = normal, >0 = heavy tails):")
    for h_idx, h_name in [(0, "H=1"), (4, "H=5"), (14, "H=15"), (29, "H=30")]:
        vae_kurt = stats.kurtosis(all_samples_log[:, :, h_idx].flatten())
        gt_kurt = stats.kurtosis(all_targets_log[:, h_idx].flatten())
        print(f"  {h_name}: VAE kurt={vae_kurt:.3f}, GT kurt={gt_kurt:.3f}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY: UNCONDITIONAL MARGINAL ANALYSIS")
    print("=" * 70)

    print(f"""
Posterior Mode (z conditioned on target):

1. CI Violations (IV Space):
   - Short (H=1-5):   {short_viol:.1f}%  {"✓" if 5 <= short_viol <= 15 else "✗"}
   - Medium (H=6-15): {medium_viol:.1f}%  {"✓" if 5 <= medium_viol <= 15 else "✗"}
   - Long (H=16-30):  {long_viol:.1f}%  {"✓" if 5 <= long_viol <= 15 else "✗"}
   - Overall:         {overall_viol:.1f}%

2. Interpretation:
   - Target: ~10% violations for 90% CI
   - {"✓ Well-calibrated" if 5 <= overall_viol <= 15 else "⚠ Needs adjustment"}
   - Violation variance across horizons: {np.std(horizon_violations):.2f}%
""")

    # Save results for conditional analysis
    results_path = Path("models/backfill/two_stage/unconditional_analysis.npz")
    np.savez(results_path,
             samples_log=all_samples_log,
             targets_log=all_targets_log,
             samples_iv=all_samples_iv,
             targets_iv=all_targets_iv,
             ctx_embeddings=all_ctx_embeddings,
             eval_indices=eval_indices,
             horizon_violations=np.array(horizon_violations))
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
