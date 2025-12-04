"""
Phase 2: Feature Distribution Analysis - Test if OOS Features are Out-of-Distribution
"""

import numpy as np
import pandas as pd
from scipy import stats

print("="*80)
print("PHASE 2: FEATURE DISTRIBUTION ANALYSIS")
print("="*80)
print()

# Load data
data = np.load("results/vae_baseline/analysis/prior/sequence_ci_width_stats.npz")
print("Loaded CI width statistics data")
print()

distribution_results = []
feature_names = ['abs_returns', 'realized_vol_30d', 'skews', 'slopes', 'atm_vol']

for h in [1, 7, 14, 30]:
    print(f"{'='*80}")
    print(f"HORIZON = {h} days")
    print(f"{'='*80}")
    print()

    for feature in feature_names:
        # Load data
        ins_vals = data[f'insample_h{h}_{feature}']
        oos_vals = data[f'oos_h{h}_{feature}']

        # KS test (two-sample)
        ks_stat, ks_pval = stats.ks_2samp(ins_vals, oos_vals)

        # Distribution statistics
        ins_mean, ins_std = ins_vals.mean(), ins_vals.std()
        ins_median = np.median(ins_vals)
        ins_q25, ins_q75 = np.percentile(ins_vals, [25, 75])

        oos_mean, oos_std = oos_vals.mean(), oos_vals.std()
        oos_median = np.median(oos_vals)
        oos_q25, oos_q75 = np.percentile(oos_vals, [25, 75])

        # Percentage outside insample range
        ins_min, ins_max = ins_vals.min(), ins_vals.max()
        oos_outside = ((oos_vals < ins_min) | (oos_vals > ins_max)).mean()

        # Mean shift as z-score
        mean_shift_z = (oos_mean - ins_mean) / ins_std if ins_std > 0 else 0

        # Variance ratio
        var_ratio = (oos_std / ins_std) if ins_std > 0 else 0

        distribution_results.append({
            'horizon': h,
            'feature': feature,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'ks_significant': ks_pval < 0.01,
            'insample_mean': ins_mean,
            'insample_std': ins_std,
            'insample_median': ins_median,
            'insample_q25': ins_q25,
            'insample_q75': ins_q75,
            'insample_min': ins_min,
            'insample_max': ins_max,
            'oos_mean': oos_mean,
            'oos_std': oos_std,
            'oos_median': oos_median,
            'oos_q25': oos_q25,
            'oos_q75': oos_q75,
            'mean_shift_z': mean_shift_z,
            'variance_ratio': var_ratio,
            'oos_outside_range_pct': oos_outside * 100
        })

        # Print feature summary
        print(f"Feature: {feature.upper()}")
        print(f"{'─'*80}")
        print(f"  In-sample: mean={ins_mean:.4f}, std={ins_std:.4f}, median={ins_median:.4f}")
        print(f"  OOS:       mean={oos_mean:.4f}, std={oos_std:.4f}, median={oos_median:.4f}")
        print(f"  Mean shift: {mean_shift_z:+.2f}σ")
        print(f"  Variance ratio (OOS/In): {var_ratio:.2f}×")
        print(f"  OOS outside in-sample range: {oos_outside*100:.1f}%")
        print(f"  KS test: statistic={ks_stat:.3f}, p-value={ks_pval:.4f}", end="")
        if ks_pval < 0.01:
            print(" ⚠️ SIGNIFICANT")
        else:
            print()
        print()

    print()

# Save results
df_dist = pd.DataFrame(distribution_results)
df_dist.to_csv('feature_distribution_shifts.csv', index=False)

print("="*80)
print("SUMMARY - SIGNIFICANT DISTRIBUTION SHIFTS")
print("="*80)
print()
print("Significant shifts: p < 0.01 AND (|mean shift| > 0.5σ OR variance ratio > 1.5 OR OOS outside > 5%)")
print()

# Filter for significant shifts
significant = df_dist[
    (df_dist['ks_pvalue'] < 0.01) &
    ((abs(df_dist['mean_shift_z']) > 0.5) |
     (df_dist['variance_ratio'] > 1.5) |
     (df_dist['oos_outside_range_pct'] > 5))
]

if len(significant) > 0:
    print(f"Found {len(significant)} significant shifts:")
    print()
    print(f"{'Horizon':>8} {'Feature':<20} {'Mean Shift':>12} {'Var Ratio':>12} {'OOS Outside':>12} {'KS p-val':>12}")
    print(f"{'─'*80}")
    for _, row in significant.iterrows():
        print(f"{row['horizon']:>8} {row['feature']:<20} {row['mean_shift_z']:>11.2f}σ "
              f"{row['variance_ratio']:>11.2f}× {row['oos_outside_range_pct']:>11.1f}% "
              f"{row['ks_pvalue']:>12.6f}")
else:
    print("No significant shifts detected")

print()
print("="*80)
print("INTERPRETATION")
print("="*80)
print()

# Analyze patterns
max_mean_shift = df_dist['mean_shift_z'].abs().max()
max_var_ratio = df_dist['variance_ratio'].max()
max_outside = df_dist['oos_outside_range_pct'].max()
n_significant_ks = (df_dist['ks_pvalue'] < 0.01).sum()

print(f"Maximum mean shift: {max_mean_shift:.2f}σ")
print(f"Maximum variance ratio: {max_var_ratio:.2f}×")
print(f"Maximum OOS outside range: {max_outside:.1f}%")
print(f"Significant KS tests (p<0.01): {n_significant_ks}/{len(df_dist)}")
print()

# Feature-specific analysis
print("Per-feature analysis:")
print(f"{'─'*80}")
for feat in feature_names:
    feat_data = df_dist[df_dist['feature'] == feat]
    avg_shift = feat_data['mean_shift_z'].mean()
    avg_var = feat_data['variance_ratio'].mean()
    max_outside = feat_data['oos_outside_range_pct'].max()
    n_sig = (feat_data['ks_pvalue'] < 0.01).sum()

    print(f"  {feat:<20}: avg shift={avg_shift:+.2f}σ, avg var ratio={avg_var:.2f}×, "
          f"max outside={max_outside:.1f}%, {n_sig}/4 horizons significant")

print()
print(f"{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print()

if max_mean_shift > 1.0 or max_var_ratio > 2.0 or max_outside > 10.0:
    print("⚠️  STRONG OUT-OF-DISTRIBUTION SIGNAL DETECTED")
    print()
    print("OOS features show substantial shifts from in-sample distribution:")
    if max_mean_shift > 1.0:
        print(f"  - Mean shifts exceed 1σ (max: {max_mean_shift:.2f}σ)")
    if max_var_ratio > 2.0:
        print(f"  - Variance changes exceed 2× (max: {max_var_ratio:.2f}×)")
    if max_outside > 10.0:
        print(f"  - More than 10% of OOS values outside training range (max: {max_outside:.1f}%)")
    print()
    print("Implication: Model is extrapolating beyond training distribution")
    print("→ CI violations likely occur in extrapolation regions")
    print("→ Model needs uncertainty calibration for out-of-distribution detection")
elif n_significant_ks >= 10:
    print("⚠️  MODERATE DISTRIBUTION SHIFTS DETECTED")
    print()
    print(f"Many features show statistically significant distribution changes ({n_significant_ks}/{len(df_dist)})")
    print("→ OOS period has different statistical properties")
    print("→ Combined with Phase 1 findings (spatial dominance decrease), explains CI violation increase")
else:
    print("✓ MILD DISTRIBUTION SHIFTS")
    print()
    print("OOS features remain within reasonable range of in-sample distribution")
    print("→ Distribution shift alone doesn't fully explain CI violation increase")
    print("→ Problem likely in VAE model's learned mapping or latent space")

print()
print(f"Results saved to: feature_distribution_shifts.csv")
print()
