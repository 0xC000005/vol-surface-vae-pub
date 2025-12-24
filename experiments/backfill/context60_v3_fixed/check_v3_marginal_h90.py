"""
Quick Marginal Distribution Check for V3 FIXED Model - H=90

Checks if the p50 (median) predictions at H=90 match the ground truth marginal distribution.
Uses pre-generated predictions from results/context60_latent12_v3_FIXED/predictions/
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
from pathlib import Path

# Constants
CONTEXT_LEN = 60
HORIZON = 90
ATM_6M = (2, 2)  # Grid point: K/S=1.00, 6-month maturity

def main():
    print("="*80)
    print("V3 FIXED MODEL - P50 H=90 MARGINAL DISTRIBUTION CHECK")
    print("="*80)

    # Load V3 FIXED predictions (prior mode)
    pred_file = Path("results/context60_latent12_v3_FIXED/predictions/teacher_forcing/prior/vae_tf_insample_h90.npz")
    print(f"\nLoading predictions from: {pred_file}")
    pred_data = np.load(pred_file)

    surfaces = pred_data['surfaces']  # (N, 90, 3, 5, 5)
    indices = pred_data['indices']
    print(f"  Loaded {len(indices)} sequences")

    # Extract p50 (median) at final horizon
    # surfaces[:, -1, 1, :, :] = p50 predictions at day 90
    p50_final = surfaces[:, -1, 1, ATM_6M[0], ATM_6M[1]]  # (N,)

    print(f"\nP50 predictions at H=90:")
    print(f"  Shape: {p50_final.shape}")
    print(f"  Mean: {p50_final.mean():.6f}")
    print(f"  Std:  {p50_final.std():.6f}")
    print(f"  Range: [{p50_final.min():.6f}, {p50_final.max():.6f}]")

    # Load ground truth
    gt_data = np.load("data/vol_surface_with_ret.npz")
    gt_surface = gt_data['surface'][:, ATM_6M[0], ATM_6M[1]]  # (5793,)

    print(f"\nLoading ground truth...")

    # Extract GT values at the same horizon positions
    gt_at_h90 = []
    for idx in indices:
        target_idx = idx + HORIZON - 1  # idx is end of context, +89 gets to day 90
        if target_idx < len(gt_surface):
            gt_at_h90.append(gt_surface[target_idx])

    gt_at_h90 = np.array(gt_at_h90)

    print(f"  Shape: {gt_at_h90.shape}")
    print(f"  Mean: {gt_at_h90.mean():.6f}")
    print(f"  Std:  {gt_at_h90.std():.6f}")
    print(f"  Range: [{gt_at_h90.min():.6f}, {gt_at_h90.max():.6f}]")

    # Compute marginal distribution metrics
    print(f"\n" + "="*80)
    print("MARGINAL DISTRIBUTION METRICS")
    print("="*80)

    # 1. Wasserstein distance (earth mover's distance)
    wasserstein = wasserstein_distance(gt_at_h90, p50_final)
    print(f"\n1. Wasserstein-1 Distance: {wasserstein:.6f}")
    print(f"   Interpretation: {wasserstein/gt_at_h90.std():.2%} of GT std")
    print(f"   Status: {'✅ PASS' if wasserstein < 0.01 else '⚠️  MODERATE' if wasserstein < 0.02 else '❌ FAIL'}")

    # 2. KS test
    ks_stat, ks_pval = ks_2samp(gt_at_h90, p50_final)
    print(f"\n2. Kolmogorov-Smirnov Test:")
    print(f"   KS Statistic: {ks_stat:.6f}")
    print(f"   p-value: {ks_pval:.6f}")
    print(f"   Status: {'✅ PASS' if ks_pval > 0.05 else '❌ FAIL'} (p > 0.05 = distributions match)")

    # 3. Quantile comparison
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    gt_quantiles = np.percentile(gt_at_h90, [q*100 for q in quantiles])
    model_quantiles = np.percentile(p50_final, [q*100 for q in quantiles])
    quantile_rmse = np.sqrt(np.mean((gt_quantiles - model_quantiles)**2))

    print(f"\n3. Quantile RMSE: {quantile_rmse:.6f}")
    print(f"   Individual quantiles:")
    for i, q in enumerate(quantiles):
        diff = model_quantiles[i] - gt_quantiles[i]
        pct_diff = diff / gt_quantiles[i] * 100
        print(f"     p{int(q*100):02d}: GT={gt_quantiles[i]:.6f}, Model={model_quantiles[i]:.6f}, Diff={diff:+.6f} ({pct_diff:+.2f}%)")
    print(f"   Status: {'✅ PASS' if quantile_rmse < 0.01 else '⚠️  MODERATE' if quantile_rmse < 0.02 else '❌ FAIL'}")

    # 4. Moment matching
    print(f"\n4. Moment Matching:")

    mean_bias = p50_final.mean() - gt_at_h90.mean()
    print(f"   Mean: GT={gt_at_h90.mean():.6f}, Model={p50_final.mean():.6f}, Bias={mean_bias:+.6f} ({mean_bias/gt_at_h90.mean()*100:+.2f}%)")
    print(f"         Status: {'✅ PASS' if abs(mean_bias) < 0.005 else '⚠️  MODERATE' if abs(mean_bias) < 0.01 else '❌ FAIL'}")

    var_ratio = p50_final.var() / gt_at_h90.var()
    print(f"   Variance: GT={gt_at_h90.var():.6f}, Model={p50_final.var():.6f}, Ratio={var_ratio:.3f}")
    print(f"         Status: {'✅ PASS' if 0.8 < var_ratio < 1.2 else '⚠️  MODERATE' if 0.6 < var_ratio < 1.4 else '❌ FAIL'}")

    skew_gt = stats.skew(gt_at_h90)
    skew_model = stats.skew(p50_final)
    print(f"   Skewness: GT={skew_gt:.3f}, Model={skew_model:.3f}, Diff={skew_model-skew_gt:+.3f}")

    kurt_gt = stats.kurtosis(gt_at_h90)
    kurt_model = stats.kurtosis(p50_final)
    print(f"   Kurtosis: GT={kurt_gt:.3f}, Model={kurt_model:.3f}, Diff={kurt_model-kurt_gt:+.3f}")

    # Overall assessment
    print(f"\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    passes = {
        'wasserstein': wasserstein < 0.01,
        'ks_test': ks_pval > 0.05,
        'quantile_rmse': quantile_rmse < 0.01,
        'mean_bias': abs(mean_bias) < 0.005,
        'variance': 0.8 < var_ratio < 1.2,
    }

    n_pass = sum(passes.values())
    n_total = len(passes)

    print(f"\nPassed: {n_pass}/{n_total} metrics")
    for metric, passed in passes.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {metric}")

    if n_pass >= 4:
        print(f"\n✅ V3 FIXED p50 H=90 marginal distribution is GOOD")
        print(f"   Model successfully replicates ground truth distribution at H=90")
    elif n_pass >= 2:
        print(f"\n⚠️  V3 FIXED p50 H=90 marginal distribution is ACCEPTABLE")
        print(f"   Some metrics off target but generally reasonable")
    else:
        print(f"\n❌ V3 FIXED p50 H=90 marginal distribution has ISSUES")
        print(f"   Model does not match ground truth distribution well")

    print("="*80)

if __name__ == "__main__":
    main()
