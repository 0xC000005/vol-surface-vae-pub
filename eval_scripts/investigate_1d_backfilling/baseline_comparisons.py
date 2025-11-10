"""
Baseline Comparisons for Backfilling Task.

Tests if simple methods can outperform the VAE realistic scenario:
1. Linear Regression: AMZN[t] ~ β1·MSFT[t] + β2·SP500[t]
2. Historical Mean: Predict mean of past AMZN returns
3. Last Value: Naive persistence (AMZN[t] = AMZN[t-1])
4. Zero (Random Walk): AMZN[t] = 0

Compares against VAE S3 (Realistic Original) performance.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
DATA_FILE = "data/stock_returns_multifeature.npz"
INVESTIGATION_FILE = "models_1d_backfilling/latent_selection_investigation.npz"

print("=" * 80)
print("BASELINE COMPARISONS FOR BACKFILLING")
print("=" * 80)
print()

# Load data
print("Loading data...")
data = np.load(DATA_FILE)
investigation = np.load(INVESTIGATION_FILE)

all_features = data["all_features"]
actuals = investigation["actuals"]  # Test set AMZN returns

# Test set indices
test_start = 5000
ctx_len = 5
first_day = test_start + ctx_len

# Extract features for prediction days
amzn_returns = all_features[first_day:first_day+len(actuals), 0]
msft_returns = all_features[first_day:first_day+len(actuals), 4]
sp500_returns = all_features[first_day:first_day+len(actuals), 8]

# Previous AMZN returns (for context)
amzn_prev = all_features[first_day-1:first_day-1+len(actuals), 0]

print(f"  Test samples: {len(actuals)}")
print()

# ============================================================================
# Baseline 1: Linear Regression (MSFT + SP500 → AMZN)
# ============================================================================

print("=" * 80)
print("BASELINE 1: Linear Regression (MSFT + SP500)")
print("=" * 80)
print()

# Split into train/test within test set (use first 400 for training LR, last 400 for eval)
split_idx = len(actuals) // 2

X_train = np.column_stack([msft_returns[:split_idx], sp500_returns[:split_idx]])
y_train = amzn_returns[:split_idx]

X_test = np.column_stack([msft_returns[split_idx:], sp500_returns[split_idx:]])
y_test = amzn_returns[split_idx:]

# Fit linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred_lr = lr.predict(X_test)

# Metrics
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
direction_lr = np.mean(np.sign(y_pred_lr) == np.sign(y_test)) * 100

print(f"Linear Regression: AMZN ~ {lr.coef_[0]:.3f}·MSFT + {lr.coef_[1]:.3f}·SP500 + {lr.intercept_:.4f}")
print(f"  RMSE: {rmse_lr:.4f}")
print(f"  MAE: {mae_lr:.4f}")
print(f"  R²: {r2_lr:.4f}")
print(f"  Direction Accuracy: {direction_lr:.2f}%")
print()

# ============================================================================
# Baseline 2: Historical Mean
# ============================================================================

print("=" * 80)
print("BASELINE 2: Historical Mean")
print("=" * 80)
print()

# Predict the mean of historical AMZN returns
historical_mean = np.mean(amzn_returns[:split_idx])
y_pred_mean = np.full_like(y_test, historical_mean)

rmse_mean = np.sqrt(mean_squared_error(y_test, y_pred_mean))
mae_mean = mean_absolute_error(y_test, y_pred_mean)
r2_mean = r2_score(y_test, y_pred_mean)
direction_mean = np.mean(np.sign(y_pred_mean) == np.sign(y_test)) * 100

print(f"Historical mean: {historical_mean:.4f}")
print(f"  RMSE: {rmse_mean:.4f}")
print(f"  MAE: {mae_mean:.4f}")
print(f"  R²: {r2_mean:.4f}")
print(f"  Direction Accuracy: {direction_mean:.2f}%")
print()

# ============================================================================
# Baseline 3: Last Value (Naive Persistence)
# ============================================================================

print("=" * 80)
print("BASELINE 3: Last Value (Naive Persistence)")
print("=" * 80)
print()

# Predict AMZN[t] = AMZN[t-1]
y_pred_last = amzn_prev[split_idx:]

rmse_last = np.sqrt(mean_squared_error(y_test, y_pred_last))
mae_last = mean_absolute_error(y_test, y_pred_last)
r2_last = r2_score(y_test, y_pred_last)
direction_last = np.mean(np.sign(y_pred_last) == np.sign(y_test)) * 100

print(f"Naive persistence: AMZN[t] = AMZN[t-1]")
print(f"  RMSE: {rmse_last:.4f}")
print(f"  MAE: {mae_last:.4f}")
print(f"  R²: {r2_last:.4f}")
print(f"  Direction Accuracy: {direction_last:.2f}%")
print()

# ============================================================================
# Baseline 4: Zero (Random Walk)
# ============================================================================

print("=" * 80)
print("BASELINE 4: Zero (Random Walk)")
print("=" * 80)
print()

# Predict AMZN[t] = 0
y_pred_zero = np.zeros_like(y_test)

rmse_zero = np.sqrt(mean_squared_error(y_test, y_pred_zero))
mae_zero = mean_absolute_error(y_test, y_pred_zero)
r2_zero = r2_score(y_test, y_pred_zero)
direction_zero = np.mean(np.sign(y_pred_zero) == np.sign(y_test)) * 100

print(f"Zero prediction: AMZN[t] = 0")
print(f"  RMSE: {rmse_zero:.4f}")
print(f"  MAE: {mae_zero:.4f}")
print(f"  R²: {r2_zero:.4f}")
print(f"  Direction Accuracy: {direction_zero:.2f}%")
print()

# ============================================================================
# VAE Comparison (using full test set)
# ============================================================================

print("=" * 80)
print("VAE COMPARISON (Full Test Set)")
print("=" * 80)
print()

# Load VAE predictions
s3_p50 = investigation["s3_p50"]  # Realistic Original
s4_p50 = investigation["s4_p50"]  # Realistic Fixed

# VAE S3 (Realistic Original - z[T-1])
rmse_s3 = np.sqrt(mean_squared_error(actuals, s3_p50))
mae_s3 = mean_absolute_error(actuals, s3_p50)
r2_s3 = r2_score(actuals, s3_p50)
direction_s3 = np.mean(np.sign(s3_p50) == np.sign(actuals)) * 100

# VAE S4 (Realistic Fixed - z[T])
rmse_s4 = np.sqrt(mean_squared_error(actuals, s4_p50))
mae_s4 = mean_absolute_error(actuals, s4_p50)
r2_s4 = r2_score(actuals, s4_p50)
direction_s4 = np.mean(np.sign(s4_p50) == np.sign(actuals)) * 100

print("VAE S3 (Realistic Original - z[T-1]):")
print(f"  RMSE: {rmse_s3:.4f}")
print(f"  MAE: {mae_s3:.4f}")
print(f"  R²: {r2_s3:.4f}")
print(f"  Direction Accuracy: {direction_s3:.2f}%")
print()

print("VAE S4 (Realistic Fixed - z[T]):")
print(f"  RMSE: {rmse_s4:.4f}")
print(f"  MAE: {mae_s4:.4f}")
print(f"  R²: {r2_s4:.4f}")
print(f"  Direction Accuracy: {direction_s4:.2f}%")
print()

# ============================================================================
# Comparison Table
# ============================================================================

print("=" * 80)
print("COMPARISON TABLE (Second Half of Test Set)")
print("=" * 80)
print()

results = {
    "Method": [
        "Linear Regression (MSFT+SP500)",
        "Historical Mean",
        "Last Value (Persistence)",
        "Zero (Random Walk)",
        "VAE S3 (z[T-1]) - 2nd Half",
        "VAE S4 (z[T]) - 2nd Half",
    ],
    "RMSE": [rmse_lr, rmse_mean, rmse_last, rmse_zero,
             np.sqrt(mean_squared_error(actuals[split_idx:], s3_p50[split_idx:])),
             np.sqrt(mean_squared_error(actuals[split_idx:], s4_p50[split_idx:]))],
    "Direction Acc": [direction_lr, direction_mean, direction_last, direction_zero,
                      np.mean(np.sign(s3_p50[split_idx:]) == np.sign(actuals[split_idx:])) * 100,
                      np.mean(np.sign(s4_p50[split_idx:]) == np.sign(actuals[split_idx:])) * 100],
    "R²": [r2_lr, r2_mean, r2_last, r2_zero,
           r2_score(actuals[split_idx:], s3_p50[split_idx:]),
           r2_score(actuals[split_idx:], s4_p50[split_idx:])],
}

import pandas as pd
df = pd.DataFrame(results)
df = df.sort_values("RMSE")

print(df.to_string(index=False))
print()

# Highlight findings
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

best_baseline = df[df["Method"].str.contains("VAE") == False].iloc[0]
vae_s3_rmse = df[df["Method"] == "VAE S3 (z[T-1]) - 2nd Half"]["RMSE"].values[0]
vae_s4_rmse = df[df["Method"] == "VAE S4 (z[T]) - 2nd Half"]["RMSE"].values[0]

print(f"1. Best baseline: {best_baseline['Method']}")
print(f"   RMSE: {best_baseline['RMSE']:.4f}, Direction: {best_baseline['Direction Acc']:.2f}%")
print()

if vae_s3_rmse > best_baseline["RMSE"]:
    pct_worse = ((vae_s3_rmse - best_baseline["RMSE"]) / best_baseline["RMSE"]) * 100
    print(f"2. VAE S3 is {pct_worse:.1f}% WORSE than best baseline")
    print("   → VAE failed to learn useful backfilling")
else:
    print(f"2. VAE S3 is better than baselines")

print()

if direction_lr > 52:
    print(f"3. Linear regression has {direction_lr:.1f}% direction accuracy")
    print("   → MSFT/SP500 DO have predictive signal for AMZN!")
    print("   → VAE failed to capture this relationship")
elif direction_lr > 50:
    print(f"3. Linear regression has weak signal ({direction_lr:.1f}%)")
    print("   → MSFT/SP500 have minimal cointegration with AMZN")
else:
    print(f"3. Linear regression has no signal ({direction_lr:.1f}%)")
    print("   → MSFT/SP500 are uncorrelated with AMZN")

print()

# Correlations in raw data
corr_amzn_msft = np.corrcoef(amzn_returns, msft_returns)[0, 1]
corr_amzn_sp500 = np.corrcoef(amzn_returns, sp500_returns)[0, 1]
corr_msft_sp500 = np.corrcoef(msft_returns, sp500_returns)[0, 1]

print(f"4. Raw data correlations (test set):")
print(f"   AMZN ↔ MSFT: r={corr_amzn_msft:.3f}")
print(f"   AMZN ↔ SP500: r={corr_amzn_sp500:.3f}")
print(f"   MSFT ↔ SP500: r={corr_msft_sp500:.3f}")
print()

if corr_amzn_msft > 0.5 or corr_amzn_sp500 > 0.5:
    print("   → Strong correlation exists, VAE should have learned it!")
elif corr_amzn_msft > 0.3 or corr_amzn_sp500 > 0.3:
    print("   → Moderate correlation, VAE could have helped")
else:
    print("   → Weak correlation, backfilling is inherently hard")
print()
