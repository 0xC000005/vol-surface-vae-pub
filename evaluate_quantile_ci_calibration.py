"""
Evaluate confidence interval calibration for quantile regression models.
Compares quantile model CI violations with baseline model violations.
"""
import numpy as np
import pandas as pd

print("=" * 80)
print("CONFIDENCE INTERVAL CALIBRATION EVALUATION")
print("=" * 80)

# Load ground truth
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
start_day = 5000
# Generation starts at day 5000, predicts days 5001-5822 (822 days)
# But actual data only goes to 5821 (5822-1), so we have 821 predictions with ground truth
num_test_days = vol_surf_data.shape[0] - start_day - 1
ground_truth = vol_surf_data[start_day+1:, :, :]  # (821, 5, 5)

print(f"\nGround truth shape: {ground_truth.shape}")
print(f"Number of test days: {ground_truth.shape[0]}")
print(f"Grid size: {ground_truth.shape[1]}x{ground_truth.shape[2]}")

# Models to evaluate
models = ["no_ex", "ex_no_loss", "ex_loss"]

# Storage for results
results = {
    "model": [],
    "generation_type": [],
    "total_predictions": [],
    "ci_violations": [],
    "ci_violation_rate": [],
    "below_p05": [],
    "above_p95": [],
    "mean_ci_width": [],
}

print("\n" + "=" * 80)
print("QUANTILE REGRESSION MODELS")
print("=" * 80)

for model_name in models:
    print(f"\n>>> {model_name.upper()}")
    print("-" * 80)

    for gen_type in ["stochastic", "mle"]:
        # Load generated surfaces
        if gen_type == "stochastic":
            file_path = f"test_spx/quantile_regression/{model_name}_quantile_gen5.npz"
        else:
            file_path = f"test_spx/quantile_regression/{model_name}_quantile_mle_gen5.npz"

        gen_data = np.load(file_path)
        p05 = gen_data["surfaces_p05"][:num_test_days, :, :]  # (821, 5, 5)
        p50 = gen_data["surfaces_p50"][:num_test_days, :, :]  # (821, 5, 5)
        p95 = gen_data["surfaces_p95"][:num_test_days, :, :]  # (821, 5, 5)

        print(f"\n{gen_type.capitalize()} generation:")
        print(f"   Loaded: {file_path}")
        print(f"   p05 shape: {p05.shape}")
        print(f"   p50 shape: {p50.shape}")
        print(f"   p95 shape: {p95.shape}")

        # Calculate CI violations
        total = ground_truth.size
        below_p05 = np.sum(ground_truth < p05)
        above_p95 = np.sum(ground_truth > p95)
        violations = below_p05 + above_p95
        violation_rate = violations / total

        # Calculate mean CI width
        ci_width = p95 - p05
        mean_ci_width = np.mean(ci_width)

        print(f"\n   Calibration metrics:")
        print(f"   - Total predictions: {total}")
        print(f"   - Below p05: {below_p05} ({below_p05/total*100:.2f}%)")
        print(f"   - Above p95: {above_p95} ({above_p95/total*100:.2f}%)")
        print(f"   - Total CI violations: {violations} ({violation_rate*100:.2f}%)")
        print(f"   - Expected violations: ~10%")
        print(f"   - Mean CI width: {mean_ci_width:.6f}")

        # Store results
        results["model"].append(model_name)
        results["generation_type"].append(gen_type)
        results["total_predictions"].append(total)
        results["ci_violations"].append(violations)
        results["ci_violation_rate"].append(violation_rate)
        results["below_p05"].append(below_p05)
        results["above_p95"].append(above_p95)
        results["mean_ci_width"].append(mean_ci_width)

        # Check if well-calibrated
        if 8 <= violation_rate*100 <= 12:
            print(f"   ✓ WELL CALIBRATED (within 8-12% range)")
        else:
            print(f"   ✗ POORLY CALIBRATED (outside 8-12% range)")

print("\n" + "=" * 80)
print("BASELINE MSE MODELS (for comparison)")
print("=" * 80)
print("\nNote: Baseline models need 1000 samples to compute empirical CIs")
print("Loading baseline model surfaces...")

try:
    for model_name in models:
        file_path = f"test_spx/2024_11_09/{model_name}_gen5.npz"
        try:
            baseline_data = np.load(file_path)

            # Check if this is old format (1000 samples) or new format (3 quantiles)
            if "surfaces_p05" in baseline_data:
                # New format - already quantiles
                print(f"\n{model_name}: Already has quantile format")
                p05_baseline = baseline_data["surfaces_p05"]
                p95_baseline = baseline_data["surfaces_p95"]

                # Align with ground truth
                p05_baseline = p05_baseline[5000:, :, :]
                p95_baseline = p95_baseline[5000:, :, :]

                below_p05_baseline = np.sum(ground_truth < p05_baseline)
                above_p95_baseline = np.sum(ground_truth > p95_baseline)
                violations_baseline = below_p05_baseline + above_p95_baseline
                violation_rate_baseline = violations_baseline / total

                ci_width_baseline = p95_baseline - p05_baseline
                mean_ci_width_baseline = np.mean(ci_width_baseline)

                print(f"   - CI violations: {violations_baseline} ({violation_rate_baseline*100:.2f}%)")
                print(f"   - Mean CI width: {mean_ci_width_baseline:.6f}")

                results["model"].append(f"{model_name}_baseline")
                results["generation_type"].append("mse_empirical")
                results["total_predictions"].append(total)
                results["ci_violations"].append(violations_baseline)
                results["ci_violation_rate"].append(violation_rate_baseline)
                results["below_p05"].append(below_p05_baseline)
                results["above_p95"].append(above_p95_baseline)
                results["mean_ci_width"].append(mean_ci_width_baseline)
            else:
                # Old format - need to compute empirical quantiles
                surfaces = baseline_data["surfaces"]  # (days, 1000, 5, 5)
                if surfaces.shape[1] == 1000:
                    print(f"\n{model_name}: Computing empirical quantiles from 1000 samples...")

                    # Align with ground truth (surfaces starts at day 5)
                    surfaces_aligned = surfaces[5000-5:, :, :, :]  # (822, 1000, 5, 5)

                    p05_baseline = np.percentile(surfaces_aligned, 5, axis=1)
                    p95_baseline = np.percentile(surfaces_aligned, 95, axis=1)

                    below_p05_baseline = np.sum(ground_truth < p05_baseline)
                    above_p95_baseline = np.sum(ground_truth > p95_baseline)
                    violations_baseline = below_p05_baseline + above_p95_baseline
                    violation_rate_baseline = violations_baseline / total

                    ci_width_baseline = p95_baseline - p05_baseline
                    mean_ci_width_baseline = np.mean(ci_width_baseline)

                    print(f"   - CI violations: {violations_baseline} ({violation_rate_baseline*100:.2f}%)")
                    print(f"   - Mean CI width: {mean_ci_width_baseline:.6f}")

                    results["model"].append(f"{model_name}_baseline")
                    results["generation_type"].append("mse_empirical")
                    results["total_predictions"].append(total)
                    results["ci_violations"].append(violations_baseline)
                    results["ci_violation_rate"].append(violation_rate_baseline)
                    results["below_p05"].append(below_p05_baseline)
                    results["above_p95"].append(above_p95_baseline)
                    results["mean_ci_width"].append(mean_ci_width_baseline)
                else:
                    print(f"\n{model_name}: Unexpected shape {surfaces.shape}, skipping")
        except FileNotFoundError:
            print(f"\n{model_name}: File not found, skipping")
except Exception as e:
    print(f"\nWarning: Could not load baseline models: {e}")
    print("Continuing with quantile model results only...")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

df = pd.DataFrame(results)
df["violation_rate_pct"] = df["ci_violation_rate"] * 100
df_display = df[["model", "generation_type", "violation_rate_pct", "mean_ci_width", "below_p05", "above_p95"]]
df_display.columns = ["Model", "Generation", "Violation Rate (%)", "Mean CI Width", "Below p05", "Above p95"]

print("\n" + df_display.to_string(index=False))

# Save results
output_file = "test_spx/quantile_regression/ci_calibration_results.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")

# Print key findings
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\n1. QUANTILE REGRESSION PERFORMANCE:")
for model_name in models:
    mle_row = df[(df["model"] == model_name) & (df["generation_type"] == "mle")]
    if len(mle_row) > 0:
        vr = mle_row["violation_rate_pct"].values[0]
        print(f"   {model_name:15s} MLE: {vr:5.2f}% violations")

print("\n2. CALIBRATION STATUS:")
well_calibrated = df[(df["violation_rate_pct"] >= 8) & (df["violation_rate_pct"] <= 12)]
print(f"   Well-calibrated models (8-12% range): {len(well_calibrated)}/{len(df)}")

print("\n3. COMPARISON WITH BASELINE:")
if len(df[df["model"].str.contains("baseline")]) > 0:
    print("   Baseline MSE models:")
    for model_name in models:
        baseline_row = df[df["model"] == f"{model_name}_baseline"]
        quantile_row = df[(df["model"] == model_name) & (df["generation_type"] == "mle")]
        if len(baseline_row) > 0 and len(quantile_row) > 0:
            vr_baseline = baseline_row["violation_rate_pct"].values[0]
            vr_quantile = quantile_row["violation_rate_pct"].values[0]
            improvement = vr_baseline - vr_quantile
            print(f"   {model_name:15s}: {vr_baseline:5.2f}% → {vr_quantile:5.2f}% (Δ {improvement:+.2f}%)")
else:
    print("   Baseline data not available")

print("\n" + "=" * 80)
print("✓ EVALUATION COMPLETE")
print("=" * 80)
