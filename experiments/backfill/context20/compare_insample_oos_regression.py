"""
Phase 1: Quick Validation - Compare Feature Importance Between In-Sample and OOS
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

print("="*80)
print("PHASE 1: QUICK VALIDATION - FEATURE IMPORTANCE COMPARISON")
print("="*80)
print()

# Load data
data = np.load("results/vae_baseline/analysis/prior/sequence_ci_width_stats.npz")
print("Loaded CI width statistics data")
print()

results = []
feature_names = ['abs_returns', 'realized_vol_30d', 'skews', 'slopes', 'atm_vol']

for h in [1, 7, 14, 30]:
    print(f"{'='*80}")
    print(f"HORIZON = {h} days")
    print(f"{'='*80}")

    # Load insample data
    X_ins = np.column_stack([
        data[f'insample_h{h}_abs_returns'],
        data[f'insample_h{h}_realized_vol_30d'],
        data[f'insample_h{h}_skews'],
        data[f'insample_h{h}_slopes'],
        data[f'insample_h{h}_atm_vol']
    ])
    y_ins = data[f'insample_h{h}_avg_ci_width'][:, 2, 2]  # ATM 6M

    # Load OOS data
    X_oos = np.column_stack([
        data[f'oos_h{h}_abs_returns'],
        data[f'oos_h{h}_realized_vol_30d'],
        data[f'oos_h{h}_skews'],
        data[f'oos_h{h}_slopes'],
        data[f'oos_h{h}_atm_vol']
    ])
    y_oos = data[f'oos_h{h}_avg_ci_width'][:, 2, 2]

    print(f"\nData shapes:")
    print(f"  In-sample: X={X_ins.shape}, y={y_ins.shape}")
    print(f"  OOS: X={X_oos.shape}, y={y_oos.shape}")

    # Standardize and fit
    scaler_ins = StandardScaler()
    scaler_oos = StandardScaler()

    X_ins_scaled = scaler_ins.fit_transform(X_ins)
    X_oos_scaled = scaler_oos.fit_transform(X_oos)

    model_ins = sm.OLS(y_ins, sm.add_constant(X_ins_scaled)).fit()
    model_oos = sm.OLS(y_oos, sm.add_constant(X_oos_scaled)).fit()

    # Compute spatial vs temporal R²
    # Spatial only (indices 2,3,4 = skews, slopes, atm_vol)
    X_ins_spatial = X_ins_scaled[:, [2,3,4]]
    X_oos_spatial = X_oos_scaled[:, [2,3,4]]

    model_ins_spatial = sm.OLS(y_ins, sm.add_constant(X_ins_spatial)).fit()
    model_oos_spatial = sm.OLS(y_oos, sm.add_constant(X_oos_spatial)).fit()

    # Temporal only (indices 0,1 = abs_returns, realized_vol_30d)
    X_ins_temporal = X_ins_scaled[:, [0,1]]
    X_oos_temporal = X_oos_scaled[:, [0,1]]

    model_ins_temporal = sm.OLS(y_ins, sm.add_constant(X_ins_temporal)).fit()
    model_oos_temporal = sm.OLS(y_oos, sm.add_constant(X_oos_temporal)).fit()

    # Store results
    ins_spatial_dom = model_ins_spatial.rsquared / model_ins_temporal.rsquared
    oos_spatial_dom = model_oos_spatial.rsquared / model_oos_temporal.rsquared

    results.append({
        'horizon': h,
        'insample_full_r2': model_ins.rsquared,
        'oos_full_r2': model_oos.rsquared,
        'r2_drop_pct': (model_ins.rsquared - model_oos.rsquared) / model_ins.rsquared * 100,
        'insample_spatial_r2': model_ins_spatial.rsquared,
        'oos_spatial_r2': model_oos_spatial.rsquared,
        'insample_temporal_r2': model_ins_temporal.rsquared,
        'oos_temporal_r2': model_oos_temporal.rsquared,
        'insample_spatial_dominance': ins_spatial_dom,
        'oos_spatial_dominance': oos_spatial_dom,
        'dominance_ratio_change': (oos_spatial_dom - ins_spatial_dom) / ins_spatial_dom * 100
    })

    # Print results
    print(f"\n{'─'*80}")
    print("FULL MODEL (All Features)")
    print(f"{'─'*80}")
    print(f"  In-sample R² = {model_ins.rsquared:.3f}")
    print(f"  OOS R²       = {model_oos.rsquared:.3f}")
    print(f"  Δ R²         = {model_ins.rsquared - model_oos.rsquared:.3f} ({(model_ins.rsquared - model_oos.rsquared) / model_ins.rsquared * 100:.1f}% drop)")

    print(f"\n{'─'*80}")
    print("SPATIAL FEATURES ONLY (skews, slopes, atm_vol)")
    print(f"{'─'*80}")
    print(f"  In-sample R² = {model_ins_spatial.rsquared:.3f}")
    print(f"  OOS R²       = {model_oos_spatial.rsquared:.3f}")
    print(f"  Δ R²         = {model_ins_spatial.rsquared - model_oos_spatial.rsquared:.3f}")

    print(f"\n{'─'*80}")
    print("TEMPORAL FEATURES ONLY (abs_returns, realized_vol_30d)")
    print(f"{'─'*80}")
    print(f"  In-sample R² = {model_ins_temporal.rsquared:.3f}")
    print(f"  OOS R²       = {model_oos_temporal.rsquared:.3f}")
    print(f"  Δ R²         = {model_ins_temporal.rsquared - model_oos_temporal.rsquared:.3f}")

    print(f"\n{'─'*80}")
    print("SPATIAL DOMINANCE RATIO (Spatial R² / Temporal R²)")
    print(f"{'─'*80}")
    print(f"  In-sample dominance = {ins_spatial_dom:.2f}×")
    print(f"  OOS dominance       = {oos_spatial_dom:.2f}×")
    print(f"  Change              = {(oos_spatial_dom - ins_spatial_dom) / ins_spatial_dom * 100:+.1f}%")

    # Print coefficients
    print(f"\n{'─'*80}")
    print("COEFFICIENTS (Standardized)")
    print(f"{'─'*80}")
    print(f"{'Feature':<20} {'In-sample':>12} {'OOS':>12} {'Difference':>12}")
    print(f"{'─'*80}")
    for i, name in enumerate(feature_names):
        coef_ins = model_ins.params[i+1]
        coef_oos = model_oos.params[i+1]
        diff = coef_ins - coef_oos
        print(f"{name:<20} {coef_ins:>12.4f} {coef_oos:>12.4f} {diff:>12.4f}")

    print()

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv('quick_validation_period_comparison.csv', index=False)

print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Results saved to: quick_validation_period_comparison.csv")
print()
print("Key Findings:")
print(f"{'─'*80}")
print(f"{'Horizon':>8} {'ΔR² (drop)':>12} {'In-sample':>15} {'OOS':>15} {'Dominance':>15}")
print(f"{'':>8} {'':>12} {'Dominance':>15} {'Dominance':>15} {'Δ%':>15}")
print(f"{'─'*80}")
for r in results:
    print(f"{r['horizon']:>8} {r['r2_drop_pct']:>11.1f}% "
          f"{r['insample_spatial_dominance']:>14.2f}× "
          f"{r['oos_spatial_dominance']:>14.2f}× "
          f"{r['dominance_ratio_change']:>14.1f}%")

print()
print("="*80)
print("INTERPRETATION")
print("="*80)
print()

# Decision point
max_r2_drop = max(r['r2_drop_pct'] for r in results)
max_dom_change = max(abs(r['dominance_ratio_change']) for r in results)

print(f"Maximum R² drop: {max_r2_drop:.1f}%")
print(f"Maximum dominance ratio change: {max_dom_change:.1f}%")
print()

if max_r2_drop > 20:
    print("⚠️  SIGNIFICANT R² DROP DETECTED (>20%)")
    print("→ Model relationship breaks down in OOS")
    print("→ Recommend proceeding to Phase 2: Feature Distribution Analysis")
    decision = "proceed"
elif max_dom_change > 30:
    print("⚠️  SIGNIFICANT DOMINANCE RATIO CHANGE DETECTED (>30%)")
    print("→ Feature importance shifts between periods")
    print("→ Recommend proceeding to Phase 2: Feature Distribution Analysis")
    decision = "proceed"
else:
    print("✓ Modest changes detected")
    print("→ Feature importance relatively stable")
    print("→ Problem may be elsewhere (latent space, distributional shift)")
    decision = "investigate_elsewhere"

print()
print(f"Recommendation: {decision.upper()}")
print()
