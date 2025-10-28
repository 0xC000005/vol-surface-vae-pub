# Confidence Interval Calibration: Empirical Observations

**Date**: 2025-10-28
**Test Period**: 2019-02-27 to 2023-02-14 (1000 days)
**Models Evaluated**: No EX, EX No Loss, EX Loss

---

## What Was Measured

This document reports empirical observations from three trained VAE models evaluated on the test set. Measurements include:

1. **Point Forecast Accuracy**: Regression of actual values against mean of generated samples
2. **Confidence Interval Coverage**: Percentage of days where actual value falls outside model's 90% CI (5th-95th percentile of 1000 generated samples)
3. **Reconstruction Loss**: Mean squared error between predictions and actual values
4. **Distribution Comparison**: Marginal distributions (pooled across all days) vs conditional distributions (per-day)

---

## Observation 1: Point Forecast Accuracy (Mean Tracking)

### Regression Analysis: `Actual = α + β₁ × Mean(Generated Samples)`

| Model | Grid Point | β₁ | R² | RMSE |
|-------|-----------|-----|-----|------|
| No EX | ATM 3-Month | 1.2326 | 0.8503 | 0.0325 |
| No EX | ATM 1-Year | 1.0570 | 0.8915 | 0.0140 |
| No EX | OTM Put 1-Year | 1.2975 | 0.7424 | 0.0252 |
| EX No Loss | ATM 3-Month | 1.2524 | 0.8776 | 0.0248 |
| EX No Loss | ATM 1-Year | 0.9715 | 0.8610 | 0.0140 |
| EX No Loss | OTM Put 1-Year | 0.9087 | 0.8078 | 0.0165 |
| EX Loss | ATM 3-Month | 1.2446 | 0.8774 | 0.0257 |
| EX Loss | ATM 1-Year | 1.0250 | 0.9255 | 0.0102 |
| EX Loss | OTM Put 1-Year | 1.0260 | 0.8249 | 0.0199 |

**Summary**:
- R² ranges from 0.74 to 0.93
- β₁ ranges from 0.91 to 1.30 (mostly close to 1.0)
- Mean of generated samples tracks actual values with high correlation

### Reconstruction Loss (Teacher Forcing Period)

| Model | Surface MSE |
|-------|-------------|
| No EX | 0.000729 |
| EX No Loss | 0.000930 |
| EX Loss | 0.000798 |

**Summary**:
- All models achieve low reconstruction error (< 0.001 MSE)

---

## Observation 2: Confidence Interval Coverage

### CI Violation Rates (90% CI should contain actual value 90% of time)

**Expected**: ~10% of days should fall outside 90% CI
**Observed**:

| Model | Grid Point | Violations | Violation Rate |
|-------|-----------|------------|----------------|
| **No EX** | ATM 3-Month | 353/1000 | **35.3%** |
| **No EX** | ATM 1-Year | 184/1000 | **18.4%** |
| **No EX** | OTM Put 1-Year | 207/1000 | **20.7%** |
| **EX No Loss** | ATM 3-Month | 540/1000 | **54.0%** |
| **EX No Loss** | ATM 1-Year | 716/1000 | **71.6%** |
| **EX No Loss** | OTM Put 1-Year | 666/1000 | **66.6%** |
| **EX Loss** | ATM 3-Month | 580/1000 | **58.0%** |
| **EX Loss** | ATM 1-Year | 500/1000 | **50.0%** |
| **EX Loss** | OTM Put 1-Year | 647/1000 | **64.7%** |
| **EX Loss** | Returns | 563/1000 | **56.3%** |

**Summary by Model (averaged across grid points)**:
- No EX: 24.8% violation rate (expected: 10%)
- EX No Loss: 64.1% violation rate (expected: 10%)
- EX Loss: 57.6% violation rate (expected: 10%)

### Average CI Width

| Model | Grid Point | CI Width (p95-p5) |
|-------|-----------|-------------------|
| No EX | ATM 3-Month | 0.0419 |
| No EX | ATM 1-Year | 0.0244 |
| No EX | OTM Put 1-Year | 0.0452 |
| EX No Loss | ATM 3-Month | 0.0289 |
| EX No Loss | ATM 1-Year | 0.0179 |
| EX No Loss | OTM Put 1-Year | 0.0228 |
| EX Loss | ATM 3-Month | 0.0297 |
| EX Loss | ATM 1-Year | 0.0161 |
| EX Loss | OTM Put 1-Year | 0.0276 |

**Pattern observed**: Models with narrower CI width tend to have higher violation rates.

---

## Observation 3: Point Accuracy vs Uncertainty Calibration

### Relationship Between Metrics

For the same models on the same test period:

| Metric | Value Range | Interpretation |
|--------|-------------|----------------|
| R² (mean tracking) | 0.74 - 0.93 | High correlation between predicted mean and actual |
| CI violation rate | 18% - 72% | Majority fall outside expected range (10%) |
| Reconstruction MSE | 0.0007 - 0.0009 | Low mean squared error |

**Observation**: High R² and low MSE coexist with high CI violation rates.

### Example: EX Loss Model, ATM 1-Year

| Metric | Value | Expected/Interpretation |
|--------|-------|-------------------------|
| R² | 0.9255 | Excellent mean tracking |
| RMSE | 0.0102 | Low prediction error |
| CI violations | 50.0% | Half of actual values fall outside 90% CI |

---

## Observation 4: Marginal vs Conditional Distribution Behavior

### Marginal Distribution (pooled across all 1000 days)

**Method**: Compare histogram of 1000 actual values vs histogram of 1,000,000 generated samples (1000 days × 1000 samples)

**Result**: Generated marginal distributions overlap substantially with historical distributions. Peak locations and spread appear similar visually.

### Conditional Distribution (per-day 90% CI)

**Method**: For each day, generate 1000 samples and compute 5th-95th percentile. Check if actual value falls within this range.

**Result**: Actual values fall outside the per-day 90% CI at rates of 18-72% (see Observation 2).

**Pattern**: Marginal distributions appear well-matched, but conditional (per-day) CIs fail to contain actual values at the expected rate.

---

## Observation 5: Z-Score Distribution

### Paper's Histogram Test (Training Set, 3997 days)

**Method**: For each day, compute z = (actual - mean_generated) / std_generated

**Results reported in paper**:
- Surface levels: Most z-scores within [-2, 2] range
- Positive bias observed (z-scores skewed toward positive values)
- Returns: Well-centered around zero
- Skew: Best calibrated (z-scores closest to zero)
- Slope: More outliers, wider z-score distribution

**Interpretation**: Most actual values fall within ±2 standard deviations of generated mean, but systematic bias present (underestimation).

---

## Data Sources

All metrics verified from:
1. `visualize_teacher_forcing.py` output (2025-10-28)
2. `verify_mean_tracking_vs_ci.py` output (2025-10-28)
3. `compare_reconstruction_losses.py` output (2025-10-28)
4. Paper figures (Figure 1, Figure 2)

---

## Summary of Observations

1. **Point forecasts are accurate**: R² = 0.74-0.93, MSE = 0.0007-0.0009
2. **Confidence intervals are poorly calibrated**: 18-72% violation rates vs expected 10%
3. **Pattern varies by model**:
   - No EX: Best CI calibration (18-35% violations)
   - EX No Loss: Worst CI calibration (54-72% violations)
   - EX Loss: Intermediate (50-65% violations)
4. **Marginal distributions look reasonable**: Overall spread matches historical
5. **Conditional distributions fail**: Per-day uncertainty bands too narrow
6. **Both accurate means and miscalibrated CIs occur simultaneously**: High R² does not guarantee calibrated uncertainty

---

**Note**: This document reports empirical observations only. Root cause analysis and proposed solutions are left for future investigation to avoid confirmation bias.
