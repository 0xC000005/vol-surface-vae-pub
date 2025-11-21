"""
Evaluate CI calibration for VAE Prior generation (in-sample).

Tests realistic VAE Prior generation (z ~ N(0,1) for future) vs oracle reconstruction.

Expected degradation due to prior mismatch:
- Oracle: ~18% CI violations (encodes full sequence)
- VAE Prior: ~20-22% CI violations (samples future from N(0,1))

Computes CI violation rates with breakdown by:
- Overall (all 4000 training days)
- Crisis period (2008-2010)
- Normal periods (pre-2008 + post-2010)
- Comparison to oracle baseline
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("CI CALIBRATION EVALUATION - VAE PRIOR (IN-SAMPLE)")
print("=" * 80)
print()
print("Generation Strategy: VAE Prior Sampling (Strategy 2)")
print("  - Context latents: Encoded from real observations")
print("  - Future latents: Sampled from N(0,1) (NO target encoding)")
print("  - Context embeddings: Zero-padded for future timesteps")
print()
print("Expected: ~20-22% CI violations (vs ~18% oracle due to prior mismatch)")
print()

# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Load VAE Prior predictions
vae_prior_file = "models/backfill/vae_prior_insample_16yr.npz"
vae_prior_data = np.load(vae_prior_file)

print(f"Loaded: {vae_prior_file}")
print("Contents:")
for key in vae_prior_data.files:
    print(f"  {key}: {vae_prior_data[key].shape}")
print()

# Load oracle reconstruction for comparison
oracle_file = "models/backfill/insample_reconstruction_16yr.npz"
oracle_data = np.load(oracle_file)
print(f"Loaded oracle baseline: {oracle_file}")
print()

# Load ground truth
gt_data = np.load("data/vol_surface_with_ret.npz")
vol_surf_gt = gt_data["surface"]

# ============================================================================
# Define Regime Periods
# ============================================================================

# Crisis period: 2008-2010 (roughly indices 2000-2765)
crisis_start = 2000
crisis_end = 2765

print("Regime definitions:")
print(f"  Crisis period: indices [{crisis_start}, {crisis_end}]")
print(f"  Normal periods: < {crisis_start} or > {crisis_end}")
print()

# ============================================================================
# Evaluate Each Horizon
# ============================================================================

horizons = [1, 7, 14, 30]
vae_prior_results = []
oracle_results = []

for horizon in horizons:
    print("=" * 80)
    print(f"HORIZON = {horizon} days")
    print("=" * 80)

    # Load VAE Prior predictions and indices
    recon_key = f'recon_h{horizon}'
    indices_key = f'indices_h{horizon}'

    vae_prior_recons = vae_prior_data[recon_key]  # (N, 3, 5, 5)
    indices = vae_prior_data[indices_key]  # (N,)

    # Load oracle predictions (same indices)
    oracle_recons = oracle_data[recon_key]  # (N, 3, 5, 5)

    num_samples = len(indices)
    print(f"Number of samples: {num_samples}")
    print(f"Date range: [{indices[0]}, {indices[-1]}]")
    print()

    # Extract ground truth
    gt = vol_surf_gt[indices]  # (N, 5, 5)

    # ========================================================================
    # VAE Prior Statistics
    # ========================================================================

    print("VAE PRIOR GENERATION (z ~ N(0,1) for future)")
    print("-" * 80)

    # Extract quantiles
    vae_p05 = vae_prior_recons[:, 0, :, :]  # (N, 5, 5)
    vae_p50 = vae_prior_recons[:, 1, :, :]  # (N, 5, 5)
    vae_p95 = vae_prior_recons[:, 2, :, :]  # (N, 5, 5)

    # CI violations
    vae_below_p05 = gt < vae_p05
    vae_above_p95 = gt > vae_p95
    vae_outside_ci = vae_below_p05 | vae_above_p95

    vae_num_violations = np.sum(vae_outside_ci)
    total_points = vae_outside_ci.size
    vae_pct_violations = 100 * vae_num_violations / total_points

    vae_num_below = np.sum(vae_below_p05)
    vae_num_above = np.sum(vae_above_p95)
    vae_pct_below = 100 * vae_num_below / total_points
    vae_pct_above = 100 * vae_num_above / total_points

    # RMSE
    vae_rmse = np.sqrt(np.mean((vae_p50 - gt) ** 2))

    # CI width
    vae_ci_width = np.mean(vae_p95 - vae_p05)

    print(f"Total points: {total_points:,}")
    print(f"CI violations: {vae_num_violations:,} ({vae_pct_violations:.2f}%)")
    print(f"  Below p05: {vae_num_below:,} ({vae_pct_below:.2f}%)")
    print(f"  Above p95: {vae_num_above:,} ({vae_pct_above:.2f}%)")
    print(f"RMSE (p50): {vae_rmse:.6f}")
    print(f"Mean CI width: {vae_ci_width:.6f}")
    print()

    # ========================================================================
    # Oracle Statistics (for comparison)
    # ========================================================================

    print("ORACLE RECONSTRUCTION (encodes full sequence)")
    print("-" * 80)

    # Extract quantiles
    oracle_p05 = oracle_recons[:, 0, :, :]
    oracle_p50 = oracle_recons[:, 1, :, :]
    oracle_p95 = oracle_recons[:, 2, :, :]

    # CI violations
    oracle_below_p05 = gt < oracle_p05
    oracle_above_p95 = gt > oracle_p95
    oracle_outside_ci = oracle_below_p05 | oracle_above_p95

    oracle_num_violations = np.sum(oracle_outside_ci)
    oracle_pct_violations = 100 * oracle_num_violations / total_points

    oracle_num_below = np.sum(oracle_below_p05)
    oracle_num_above = np.sum(oracle_above_p95)
    oracle_pct_below = 100 * oracle_num_below / total_points
    oracle_pct_above = 100 * oracle_num_above / total_points

    # RMSE
    oracle_rmse = np.sqrt(np.mean((oracle_p50 - gt) ** 2))

    # CI width
    oracle_ci_width = np.mean(oracle_p95 - oracle_p05)

    print(f"CI violations: {oracle_num_violations:,} ({oracle_pct_violations:.2f}%)")
    print(f"  Below p05: {oracle_num_below:,} ({oracle_pct_below:.2f}%)")
    print(f"  Above p95: {oracle_num_above:,} ({oracle_pct_above:.2f}%)")
    print(f"RMSE (p50): {oracle_rmse:.6f}")
    print(f"Mean CI width: {oracle_ci_width:.6f}")
    print()

    # ========================================================================
    # Degradation Analysis
    # ========================================================================

    print("DEGRADATION: VAE Prior vs Oracle")
    print("-" * 80)

    degradation_violations = vae_pct_violations - oracle_pct_violations
    degradation_rmse = ((vae_rmse - oracle_rmse) / oracle_rmse) * 100
    degradation_ci_width = ((vae_ci_width - oracle_ci_width) / oracle_ci_width) * 100

    print(f"CI violations: +{degradation_violations:.2f} pp ({oracle_pct_violations:.2f}% → {vae_pct_violations:.2f}%)")
    print(f"RMSE: {degradation_rmse:+.2f}% ({oracle_rmse:.6f} → {vae_rmse:.6f})")
    print(f"CI width: {degradation_ci_width:+.2f}% ({oracle_ci_width:.6f} → {vae_ci_width:.6f})")
    print()

    # ========================================================================
    # Regime Breakdown (VAE Prior)
    # ========================================================================

    print("REGIME BREAKDOWN (VAE PRIOR)")
    print("-" * 80)

    # Identify regime for each sample
    is_crisis = (indices >= crisis_start) & (indices <= crisis_end)
    is_normal = ~is_crisis

    num_crisis = np.sum(is_crisis)
    num_normal = np.sum(is_normal)

    print(f"Crisis samples: {num_crisis}")
    print(f"Normal samples: {num_normal}")
    print()

    # Crisis statistics (VAE Prior)
    if num_crisis > 0:
        vae_crisis_violations = np.sum(vae_outside_ci[is_crisis])
        vae_crisis_total = vae_outside_ci[is_crisis].size
        vae_crisis_pct = 100 * vae_crisis_violations / vae_crisis_total
        vae_crisis_rmse = np.sqrt(np.mean((vae_p50[is_crisis] - gt[is_crisis]) ** 2))
        vae_crisis_ci_width = np.mean(vae_p95[is_crisis] - vae_p05[is_crisis])

        # Oracle crisis statistics
        oracle_crisis_violations = np.sum(oracle_outside_ci[is_crisis])
        oracle_crisis_total = oracle_outside_ci[is_crisis].size
        oracle_crisis_pct = 100 * oracle_crisis_violations / oracle_crisis_total

        print("Crisis Period:")
        print(f"  VAE Prior violations: {vae_crisis_violations:,}/{vae_crisis_total:,} ({vae_crisis_pct:.2f}%)")
        print(f"  Oracle violations: {oracle_crisis_violations:,}/{oracle_crisis_total:,} ({oracle_crisis_pct:.2f}%)")
        print(f"  Degradation: +{vae_crisis_pct - oracle_crisis_pct:.2f} pp")
        print(f"  VAE Prior RMSE: {vae_crisis_rmse:.6f}")
        print(f"  VAE Prior CI width: {vae_crisis_ci_width:.6f}")
    else:
        vae_crisis_pct = np.nan
        vae_crisis_rmse = np.nan
        vae_crisis_ci_width = np.nan
        oracle_crisis_pct = np.nan
        print("Crisis Period: No samples")

    print()

    # Normal statistics (VAE Prior)
    if num_normal > 0:
        vae_normal_violations = np.sum(vae_outside_ci[is_normal])
        vae_normal_total = vae_outside_ci[is_normal].size
        vae_normal_pct = 100 * vae_normal_violations / vae_normal_total
        vae_normal_rmse = np.sqrt(np.mean((vae_p50[is_normal] - gt[is_normal]) ** 2))
        vae_normal_ci_width = np.mean(vae_p95[is_normal] - vae_p05[is_normal])

        # Oracle normal statistics
        oracle_normal_violations = np.sum(oracle_outside_ci[is_normal])
        oracle_normal_total = oracle_outside_ci[is_normal].size
        oracle_normal_pct = 100 * oracle_normal_violations / oracle_normal_total

        print("Normal Periods:")
        print(f"  VAE Prior violations: {vae_normal_violations:,}/{vae_normal_total:,} ({vae_normal_pct:.2f}%)")
        print(f"  Oracle violations: {oracle_normal_violations:,}/{oracle_normal_total:,} ({oracle_normal_pct:.2f}%)")
        print(f"  Degradation: +{vae_normal_pct - oracle_normal_pct:.2f} pp")
        print(f"  VAE Prior RMSE: {vae_normal_rmse:.6f}")
        print(f"  VAE Prior CI width: {vae_normal_ci_width:.6f}")
    else:
        vae_normal_pct = np.nan
        vae_normal_rmse = np.nan
        vae_normal_ci_width = np.nan
        oracle_normal_pct = np.nan
        print("Normal Periods: No samples")

    print()

    # Store results
    vae_prior_results.append({
        'horizon': horizon,
        'method': 'VAE_Prior',
        'overall_violations': vae_pct_violations,
        'overall_rmse': vae_rmse,
        'overall_ci_width': vae_ci_width,
        'crisis_violations': vae_crisis_pct,
        'crisis_rmse': vae_crisis_rmse,
        'crisis_ci_width': vae_crisis_ci_width,
        'normal_violations': vae_normal_pct,
        'normal_rmse': vae_normal_rmse,
        'normal_ci_width': vae_normal_ci_width,
    })

    oracle_results.append({
        'horizon': horizon,
        'method': 'Oracle',
        'overall_violations': oracle_pct_violations,
        'overall_rmse': oracle_rmse,
        'overall_ci_width': oracle_ci_width,
        'crisis_violations': oracle_crisis_pct,
        'crisis_rmse': np.sqrt(np.mean((oracle_p50[is_crisis] - gt[is_crisis]) ** 2)) if num_crisis > 0 else np.nan,
        'crisis_ci_width': np.mean(oracle_p95[is_crisis] - oracle_p05[is_crisis]) if num_crisis > 0 else np.nan,
        'normal_violations': oracle_normal_pct,
        'normal_rmse': np.sqrt(np.mean((oracle_p50[is_normal] - gt[is_normal]) ** 2)) if num_normal > 0 else np.nan,
        'normal_ci_width': np.mean(oracle_p95[is_normal] - oracle_p05[is_normal]) if num_normal > 0 else np.nan,
    })

# ============================================================================
# Summary Tables
# ============================================================================

print("=" * 80)
print("SUMMARY: VAE PRIOR vs ORACLE")
print("=" * 80)
print()

# Combine results
all_results = vae_prior_results + oracle_results
df = pd.DataFrame(all_results)

# Pivot for easier comparison
pivot_violations = df.pivot(index='horizon', columns='method', values='overall_violations')
pivot_rmse = df.pivot(index='horizon', columns='method', values='overall_rmse')
pivot_ci_width = df.pivot(index='horizon', columns='method', values='overall_ci_width')

print("OVERALL CI VIOLATIONS (%)")
print(pivot_violations.to_string())
print()

print("OVERALL RMSE")
print(pivot_rmse.to_string())
print()

print("MEAN CI WIDTH")
print(pivot_ci_width.to_string())
print()

# Save detailed results
output_csv = "models/backfill/vae_prior_ci_insample_16yr.csv"
df.to_csv(output_csv, index=False)
print(f"Saved detailed results to: {output_csv}")
print()

# ============================================================================
# Degradation Summary
# ============================================================================

print("=" * 80)
print("DEGRADATION SUMMARY")
print("=" * 80)
print()

print("Prior Mismatch Effect (VAE Prior - Oracle):")
print()

for h in horizons:
    vae_row = df[(df['horizon'] == h) & (df['method'] == 'VAE_Prior')].iloc[0]
    oracle_row = df[(df['horizon'] == h) & (df['method'] == 'Oracle')].iloc[0]

    deg_violations = vae_row['overall_violations'] - oracle_row['overall_violations']
    deg_rmse_pct = ((vae_row['overall_rmse'] - oracle_row['overall_rmse']) / oracle_row['overall_rmse']) * 100

    print(f"Horizon {h}:")
    print(f"  CI violations: +{deg_violations:.2f} pp ({oracle_row['overall_violations']:.2f}% → {vae_row['overall_violations']:.2f}%)")
    print(f"  RMSE: {deg_rmse_pct:+.2f}%")

    # Regime-specific degradation
    if not np.isnan(vae_row['crisis_violations']):
        deg_crisis = vae_row['crisis_violations'] - oracle_row['crisis_violations']
        print(f"  Crisis degradation: +{deg_crisis:.2f} pp")

    if not np.isnan(vae_row['normal_violations']):
        deg_normal = vae_row['normal_violations'] - oracle_row['normal_violations']
        print(f"  Normal degradation: +{deg_normal:.2f} pp")

    print()

# ============================================================================
# Assessment
# ============================================================================

print("=" * 80)
print("ASSESSMENT")
print("=" * 80)
print()

print("Expected behavior:")
print("  - Oracle: ~18% CI violations (cheating, encodes target)")
print("  - VAE Prior: ~20-22% CI violations (realistic, prior mismatch)")
print("  - Degradation: ~2-4 pp is EXPECTED and acceptable")
print()

target_degradation = 4.0  # Target degradation in pp

for h in horizons:
    vae_row = df[(df['horizon'] == h) & (df['method'] == 'VAE_Prior')].iloc[0]
    oracle_row = df[(df['horizon'] == h) & (df['method'] == 'Oracle')].iloc[0]

    deg_violations = vae_row['overall_violations'] - oracle_row['overall_violations']

    print(f"Horizon {h}:")
    print(f"  Degradation: +{deg_violations:.2f} pp")

    if deg_violations <= target_degradation:
        status = "✓ EXPECTED (within target)"
    elif deg_violations <= 8.0:
        status = "⚠ ACCEPTABLE (higher than expected but reasonable)"
    else:
        status = "✗ CONCERNING (degradation too large, may indicate issue)"

    print(f"  Status: {status}")
    print()

print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print()
print("Key takeaway:")
print("  - VAE Prior uses z ~ N(0,1) for future (theoretically correct)")
print("  - Oracle uses z ~ q(z|context,target) for future (cheating)")
print("  - Degradation measures the cost of NOT encoding the target")
print("  - This is the REALISTIC generation performance")
print()
