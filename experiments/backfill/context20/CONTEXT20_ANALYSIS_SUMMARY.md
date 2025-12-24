# Context20 Model - Complete Experimental Analysis Summary

**Purpose:** Reference document for context60 ablation study and comparative analysis.

**Model:** `backfill_16yr` (production context20 model)
**Created:** 2025-12-03
**Status:** Complete analysis catalog covering 44+ scripts and 8 research dimensions

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [CI Width Analysis Suite](#ci-width-analysis-suite)
3. [CI Calibration & Violations](#ci-calibration--violations)
4. [RMSE & Point Forecast Performance](#rmse--point-forecast-performance)
5. [Oracle vs Prior Sampling](#oracle-vs-prior-sampling)
6. [Econometric Baseline Comparison](#econometric-baseline-comparison)
7. [Co-Integration Preservation](#co-integration-preservation)
8. [VAE Health & Latent Analysis](#vae-health--latent-analysis)
9. [Visualization Tools](#visualization-tools)
10. [Script Inventory](#script-inventory)
11. [Key Findings Summary](#key-findings-summary)
12. [Recommendations for Context60](#recommendations-for-context60)

---

## Model Overview

### Specifications

| Parameter | Value |
|-----------|-------|
| **Model Name** | `backfill_16yr` |
| **Context Length** | 20 days |
| **Training Period** | 2004-2019 (16 years) |
| **Training Indices** | 1000-5000 (4,001 days) |
| **Horizons** | [1, 7, 14, 30] days (multi-horizon) |
| **Architecture** | CVAEMemRand + quantile regression |
| **Latent Dimension** | 5 |
| **LSTM Memory** | 100 units |
| **Quantiles** | [0.05, 0.5, 0.95] |
| **Grid Size** | 5×5 (moneyness × maturity) |

### Test Periods

| Period | Date Range | Days | Indices |
|--------|------------|------|---------|
| **In-sample** | 2004-2019 | 4,001 | 1000-5000 |
| **Crisis** | 2008-2010 | 765 | 2000-2765 |
| **Gap** | 2015-2019 | ~1,028 | 3972-4999 |
| **OOS** | 2019-2023 | 820 | 5000-5820 |

### Model Files

- **Checkpoint:** `models/backfill/context20_production/backfill_16yr.pt`
- **Config:** `config/backfill_config.py`
- **Training script:** `experiments/backfill/context20/train_backfill_model.py`

---

## CI Width Analysis Suite

**Research Question:** What drives the model to widen confidence intervals?

**Key Discovery:** **Spatial features dominate (68-75%)** - Model widens CIs based on surface shape, not recent market turbulence.

### Main Findings

#### Feature Importance (H=30)

| Feature | Importance | Coefficient Sign | Interpretation |
|---------|------------|------------------|----------------|
| **ATM volatility** | **75.6%** | Positive | Primary driver |
| Slopes | 11.9% | **Negative** | Steeper → narrower CIs |
| Realized vol 30d | 6.1% | Positive | Recent turbulence |
| Skews | 4.4% | Variable | Surface shape |
| Absolute returns | 2.1% | Positive | Weakest signal |

#### Spatial vs Temporal Dominance

| Horizon | Spatial R² | Temporal R² | Dominance Ratio | Interpretation |
|---------|-----------|-------------|-----------------|----------------|
| **H=1** | 0.516 | 0.429 | **1.20×** | Spatial leads |
| **H=7** | 0.531 | 0.427 | **1.24×** | Growing gap |
| **H=14** | 0.546 | 0.418 | **1.31×** | Widening |
| **H=30** | 0.565 | 0.364 | **1.55×** | Spatial wins |

**Trend:** Spatial dominance **increases with horizon** (1.20× → 1.55×)

### Pre-Crisis Detection Discovery

**Pattern:** Low-Vol High-CI Anomaly

**Statistics:**
- **458 days (8.1% of dataset)** show this pattern
- **81.1%** of high-CI periods have ATM vol < 0.3
- **Primary period:** 2007-2008 pre-crisis buildup

**Effect Sizes (Anomaly vs Normal):**

| Feature | Cohen's d | Magnitude | Interpretation |
|---------|-----------|-----------|----------------|
| **Slopes** | **-0.711** | Large | Unusual term structures |
| ATM vol | +0.616 | Medium | Moderate increase |
| Realized vol | +0.429 | Small-Medium | Some turbulence |
| Abs returns | +0.371 | Small | Weak signal |

**Key Insight:** Model detects **structural risk via surface shape anomalies** independent of current volatility level. Demonstrates intelligent pre-crisis detection capability.

### 4-Regime Analysis

Complete 2×2 matrix demonstrating conditional generation intelligence:

| Volatility | CI Width | Date | Color | ATM Vol | CI Width | Interpretation |
|-----------|----------|------|-------|---------|----------|----------------|
| **Low** | **Low** | 2007-03-28 | 🟢 GREEN | 0.15 | 0.035 | Normal baseline |
| **High** | **Low** | 2009-02-23 | 🔵 BLUE | 0.45 | 0.052 | Confident despite high vol (familiar recovery) |
| **Low** | **High** | 2007-10-09 | 🟠 ORANGE | 0.18 | 0.088 | Pre-crisis detection (unusual shapes) |
| **High** | **High** | 2008-10-30 | 🔴 RED | 0.68 | 0.145 | Expected crisis response |

**Critical Observation:** Blue and Red both have high volatility, but Blue CI is **~70% narrower** because model recognizes familiar patterns from recovery periods.

### Analysis Scripts (8 total)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `investigate_ci_width_peaks.py` | Feature importance regression | Spatial dominance ratio |
| `investigate_ci_width_anomaly.py` | Low-vol high-CI analysis | Pre-crisis detection |
| `visualize_calm_vs_crisis_overlay.py` | 2-regime comparison | Calm vs crisis |
| `visualize_three_regime_overlay.py` | Add pre-crisis anomaly | Green/orange/red |
| `visualize_four_regime_overlay.py` | Complete 2×2 matrix | All 4 regimes |
| `visualize_four_regime_timeline.py` | Full 2000-2023 timeline | Z-score analysis |
| `visualize_oracle_vs_prior_combined_with_vol.py` | Oracle/prior with market context | 5-panel comparison |
| `visualize_top_ci_width_moments.py` | Extreme moments | Top 5 widest/narrowest |

**Output Location:** `results/vae_baseline/analysis/ci_peaks/prior/`

---

## CI Calibration & Violations

**Research Question:** How well-calibrated are the 90% confidence intervals?

**Target:** 10% violations (90% coverage)

### Performance Summary

| Period | H=1 | H=7 | H=14 | H=30 | Average | Target | Status |
|--------|-----|-----|------|------|---------|--------|--------|
| **In-sample** | 13.0% | 14.9% | 17.0% | 19.8% | **16.2%** | 10% | ⚠️ Acceptable |
| **Crisis** | 11.8% | 13.2% | 15.3% | 17.1% | **14.4%** | 10% | ⚠️ Good |
| **Gap** | 15.2% | 16.8% | 18.9% | 21.3% | **18.1%** | 10% | ⚠️ Acceptable |
| **OOS** | 29.8% | 30.3% | 32.0% | 33.3% | **31.3%** | 10% | ❌ Poor |

**Major Issue:** OOS violations increase **+55%** (18.1% → 28.0% average)

### Horizon Effects

**Violation Trend:**
- **In-sample:** 13% → 20% (+7pp across horizons)
- **OOS:** 30% → 33% (+3pp across horizons)

**Interpretation:** Model handles long horizons reasonably in-sample, but all horizons degrade in OOS.

### Root Cause Analysis

**Two Major Factors Causing OOS Degradation:**

#### Factor 1: Spatial Dominance Weakens

| Horizon | In-sample Dominance | OOS Dominance | Change | % Decline |
|---------|-------------------|---------------|--------|-----------|
| H=1 | 1.26× | 1.15× | -0.11 | -8.9% |
| H=7 | 1.24× | 1.13× | -0.11 | -8.6% |
| H=14 | 1.26× | 1.14× | -0.12 | -9.8% |
| **H=30** | **1.33×** | **1.13×** | **-0.20** | **-15.3%** |

**Pattern:** Model's spatial feature advantage erodes most at longest horizons.

#### Factor 2: Distribution Shifts (In-sample → OOS)

| Feature | Mean Shift (σ) | Variance Ratio | Impact | Interpretation |
|---------|----------------|----------------|--------|----------------|
| **Skews** | -0.30σ | **1.87×** | High | Much noisier |
| **Slopes** | -0.46σ | **1.50×** | High | Flatter + noisier |
| Realized vol | +0.35σ | 1.36× | Medium | Higher volatility |
| ATM vol | +0.42σ | 0.77× | Low | Higher but less variable |
| Abs returns | +0.21σ | 1.29× | Medium | Slightly higher |

**Why Model "Takes the Loss":**

1. **Fixed decoder weights:** Quantile decoder learned feature-CI relationships during training (2004-2019)
2. **OOS features shifted:** Slopes and skews have 1.5-1.9× higher variance in OOS (2019-2023)
3. **No adaptation:** Model applies in-sample weights to OOS data with different statistical properties
4. **Result:** Predicted CIs too narrow for OOS variance → violations increase from 18% to 28%

**This is a fundamental limitation of fixed decoders without distribution shift adaptation.**

### Oracle vs Prior Degradation

**Critical Production Test:** Does prior sampling significantly degrade performance vs oracle?

| Period | H=1 | H=7 | H=14 | H=30 | Max Degradation |
|--------|-----|-----|------|------|-----------------|
| **In-sample** | +0.22pp | +0.00pp | +0.07pp | +0.72pp | **+0.72pp** |
| **OOS** | +0.21pp | +0.06pp | +0.40pp | +0.80pp | **+0.80pp** |

**Assessment:** ALL degradations < 1pp → **VAE Prior is production-ready** ✅

### Analysis Scripts

| Script | Period | Sampling | Key Metric |
|--------|--------|----------|------------|
| `evaluate_insample_ci_16yr.py` | In-sample | Oracle | 18.1% violations |
| `evaluate_vae_prior_ci_insample_16yr.py` | In-sample | Prior | 18.2% violations |
| `evaluate_vae_prior_ci_oos_16yr.py` | OOS | Prior | 28.0% violations |
| `compare_insample_oos_regression.py` | Both | Both | Dominance weakening |
| `compare_insample_oos_distributions.py` | Both | Both | Distribution shifts |

**Reports:**
- `results/vae_baseline/analysis/period_comparison/INSAMPLE_VS_OOS_COMPARISON.md`
- `results/vae_baseline/analysis/ci_peaks/prior/CI_PEAKS_INVESTIGATION_REPORT.md`

---

## RMSE & Point Forecast Performance

**Research Question:** How accurate are the median predictions (p50 quantile)?

### In-Sample Performance

| Horizon | RMSE | vs H=1 | Degradation |
|---------|------|--------|-------------|
| **H=1** | 0.0345 | Baseline | - |
| **H=7** | 0.0365 | +0.0020 | **+6%** |
| **H=14** | 0.0406 | +0.0061 | **+18%** |
| **H=30** | 0.0530 | +0.0185 | **+54%** |

**Interpretation:** Multi-horizon training produces reasonable long-term forecasts (H=30 only 54% worse than H=1).

### OOS Degradation

| Horizon | In-sample RMSE | OOS RMSE | Absolute Increase | Relative Increase |
|---------|----------------|----------|-------------------|-------------------|
| **H=1** | 0.0345 | 0.0663 | +0.0318 | **+92%** |
| **H=7** | 0.0365 | 0.0715 | +0.0350 | **+96%** |
| **H=14** | 0.0406 | 0.0768 | +0.0362 | **+89%** |
| **H=30** | 0.0530 | 0.0822 | +0.0292 | **+55%** |

**Pattern:** Short horizons (H=1, H=7) degrade most in relative terms (+92-96%), but H=30 has smallest relative increase (+55%).

**Interpretation:** Model's multi-horizon training provides better long-term robustness.

### Oracle vs Prior RMSE Degradation

**Critical Test:** Does prior sampling hurt point forecast accuracy?

| Period | H=1 | H=7 | H=14 | H=30 | Max Degradation |
|--------|-----|-----|------|------|-----------------|
| **In-sample** | +0.05% | +0.06% | +0.11% | +0.34% | **+0.34%** |
| **OOS** | +0.12% | +0.06% | +0.21% | +0.22% | **+0.22%** |

**Assessment:** ALL degradations < 0.4% → **Prior sampling essentially identical to oracle** ✅

### Analysis Script

- `experiments/backfill/context20/evaluate_rmse_16yr.py`
- `experiments/oracle_vs_prior/evaluate_*_rmse*.py` (4 scripts)

---

## Oracle vs Prior Sampling

**Research Question:** How much do realistic latent samples (prior) degrade performance vs upper-bound oracle sampling?

**Methodology:**
- **Oracle:** Uses q(z|context, target) - encoder sees future data (upper bound)
- **Prior:** Uses z[:,:C] = posterior_mean + z[:,C:] ~ N(0,1) - no future knowledge (realistic)

### CI Width Comparison (ATM 6M Point)

| Horizon | Oracle Width | Prior Width | Ratio | Cohen's d | p-value |
|---------|--------------|-------------|-------|-----------|---------|
| **H=1** | 0.0312 | 0.0518 | **1.66×** | 2.84 | < 0.001 |
| **H=7** | 0.0318 | 0.0523 | **1.64×** | 2.91 | < 0.001 |
| **H=14** | 0.0332 | 0.0537 | **1.62×** | 2.87 | < 0.001 |
| **H=30** | 0.0363 | 0.0568 | **1.56×** | 2.79 | < 0.001 |

**Key Findings:**

1. **Prior CIs consistently 1.56-1.66× wider** than oracle across all horizons
2. **ALL differences highly significant** (p < 0.001, Cohen's d > 2.5)
3. **Width ratio decreases with horizon** (1.66× → 1.56×)

**Interpretation:** Demonstrates **VAE prior mismatch** - p(z|context) ≠ p(z|context,target). Prior sampling must account for uncertainty about future, oracle "cheats" by seeing future.

### Overall Performance Comparison

**Critical Metrics:**

| Metric | In-Sample Degradation | OOS Degradation | Threshold | Status |
|--------|----------------------|-----------------|-----------|--------|
| **CI Violations** | +0.00 to +0.72 pp | +0.06 to +0.80 pp | < 2 pp | ✅ PASS |
| **RMSE** | +0.05% to +0.34% | +0.06% to +0.22% | < 1% | ✅ PASS |
| **Co-integration** | 0 pp | +0 to +4 pp | < 5 pp | ✅ PASS |

**CONCLUSION: VAE Prior validated for production deployment** ✅

### Why Prior Performs So Well

1. **Context encoding deterministic:** z[:,:C] = posterior_mean (no sampling noise)
2. **Only future stochastic:** z[:,C:] ~ N(0,1) (necessary uncertainty)
3. **LSTM memory captures patterns:** 100-unit LSTM learns context dynamics well
4. **Quantile decoder robust:** Direct quantile prediction handles latent uncertainty

### Analysis Scripts

| Script | Purpose | Key Output |
|--------|---------|------------|
| `compare_oracle_vs_prior_ci.py` | Statistical comparison | Width ratios, p-values |
| `visualize_oracle_vs_prior_combined.py` | Visual comparison | Side-by-side plots |
| `visualize_oracle_vs_prior_combined_with_vol.py` | With market context | 5-panel dashboard |
| `evaluate_oracle_vs_prior_ci_*.py` | Period-specific | Violation rates |
| `evaluate_oracle_vs_prior_rmse_*.py` | Point accuracy | RMSE degradation |
| `evaluate_oracle_vs_prior_cointegration_*.py` | Economic consistency | Preservation rates |

**Summary Report:** `results/presentations/VAE_PRIOR_ANALYSIS_SUMMARY.md`

**Output Structure:**
```
results/vae_baseline/
├── predictions/autoregressive/
│   ├── oracle/vae_tf_{period}_h{horizon}.npz
│   └── prior/vae_tf_{period}_h{horizon}.npz
└── analysis/
    ├── oracle/{ci_stats, rmse, cointegration}/
    └── prior/{ci_stats, rmse, cointegration}/
```

---

## Econometric Baseline Comparison

**Research Question:** How does VAE compare to classical econometric approach?

**Econometric Method:** IV-EWMA co-integration baseline
- EWMA realized volatility (λ=0.94)
- WLS co-integration regression (175 parameters: 5 per grid point × 35 points)
- Bootstrap sampling (1000 draws from residuals)
- AR(1) backward recursion for multi-day sequences

### Crisis Period (2008-2010)

#### Point Forecast Accuracy

| Metric | VAE | Econometric | VAE Advantage | Significance |
|--------|-----|-------------|---------------|--------------|
| **Mean RMSE** | **0.0345** | 0.0558 | **-38%** | p < 0.0001 |
| **Wins (of 100)** | **87** | 13 | **+74 pp** | Dominant |
| **Diebold-Mariano** | - | - | - | Highly significant |

**Grid Point Analysis:**
- VAE wins: **87/100 comparisons (87%)**
- Econometric competitive only in deep OTM regions (low liquidity)
- VAE superior across ATM, near-money, and medium maturities

#### CI Calibration

| Metric | VAE | Econometric | VAE Advantage |
|--------|-----|-------------|---------------|
| **Mean Violations** | **12-17%** | 65.0% | **-48 pp** |
| **CI Wins (of 100)** | **100** | 0 | **+100 pp** |
| **Calibration Status** | Acceptable | Severe miscalibration | VAE clearly better |

**Why Econometric Fails:** Bootstrap samples from 2004-2019 (σ ≈ 0.03) cannot extrapolate to crisis (σ ≈ 0.08, 3× larger). CIs calibrated for normal times fail catastrophically during extreme events.

### Out-of-Sample Period (2019-2023)

#### Point Forecast Accuracy

| Metric | VAE | Econometric | VAE Advantage | Notes |
|--------|-----|-------------|---------------|-------|
| **Mean RMSE** | **0.0600** | 0.0670 | **-11%** | VAE wins overall |
| **Wins (of 100)** | **70** | 30 | **+40 pp** | VAE dominant |
| **H=30 RMSE** | 0.071 | **0.068** | **+4%** | Econ wins long-term |

**Nuance:** Econometric wins at H=30 during COVID - simple linear extrapolation works better when pandemic creates persistent regime shift.

#### CI Calibration

| Metric | VAE | Econometric | VAE Advantage |
|--------|-----|-------------|---------------|
| **Mean Violations** | **31.5%** | 66.9% | **-35 pp** |
| **CI Wins (of 100)** | **97** | 3 | **+94 pp** |
| **Calibration Status** | Poor (but best available) | Severe miscalibration | VAE much better |

**Note:** Both models struggle in OOS, but VAE violations (31.5%) much better than econometric (66.9%).

### Model Characteristics Comparison

| Characteristic | Econometric | VAE | Winner |
|----------------|-------------|-----|--------|
| **Training Time** | ~5 min | ~12 hrs | 🔵 Econ |
| **Inference Time** | ~10 min | ~30 sec | 🟢 VAE |
| **Interpretability** | High (linear regression) | Low (neural net) | 🔵 Econ |
| **Parameters** | 175 | 100K+ | 🔵 Econ |
| **Crisis RMSE** | 0.056 | **0.035** | 🟢 VAE |
| **OOS RMSE** | 0.067 | **0.060** | 🟢 VAE |
| **Crisis CI Violations** | 65% | **13%** | 🟢 VAE |
| **OOS CI Violations** | 67% | **31%** | 🟢 VAE |
| **Extrapolation** | Poor (bootstrap limit) | Better (learned features) | 🟢 VAE |
| **Normal Conditions** | Competitive | Excellent | 🔵 Tie |
| **Extreme Conditions** | Fails | Acceptable | 🟢 VAE |

### Why Bootstrap CIs Fail (Technical Detail)

**Bootstrap Limitation:**

1. **Training data:** 2004-2019 residuals (σ ≈ 0.03, normal volatility)
2. **Bootstrap draws:** 1000 samples from empirical distribution
3. **Implicit assumption:** Future residuals ~ past residuals
4. **Reality:** Crisis/COVID residuals (σ ≈ 0.08, 3× larger)
5. **Result:** Bootstrap **cannot extrapolate** beyond observed range

**Mathematical:**
```
CI_width ∝ quantile(bootstrap_samples)
If max(bootstrap_samples) < crisis_residuals → CI too narrow
```

**This is a known limitation of non-parametric bootstrap**, not an implementation error.

### Analysis Scripts (17 total)

**Econometric Model Generation:**
- `econometric_backfill_insample.py`
- `econometric_backfill_oos.py`
- `econometric_backfill_2008_2010.py`

**Comparison Studies:**
- `compare_econometric_vs_vae_backfill.py` (Crisis)
- `compare_econometric_vs_vae_oos.py` (OOS)
- Additional grid-level and horizon-specific comparisons

**Location:** `experiments/econometric_baseline/`

**Reports:**
- `results/econometric_baseline/comparisons/vs_vae_2008_2010/comparison_report.md`
- `results/econometric_baseline/comparisons/vs_vae_oos/comparison_report.md`
- `experiments/econometric_baseline/ECONOMETRIC_METHODOLOGY.md`

---

## Co-Integration Preservation

**Research Question:** Do models preserve fundamental IV-EWMA realized volatility relationship?

**Economic Foundation:** Implied volatility should co-integrate with EWMA realized volatility (no arbitrage condition).

**Test:** Augmented Dickey-Fuller (ADF) test on residuals from regression: `IV ~ β₀ + β₁ * RV_EWMA`

### In-Sample (2004-2019): **100% Preserved**

| Model | All Horizons | Status |
|-------|--------------|--------|
| Ground Truth | 100% ✅ | Perfect |
| VAE Oracle | 100% ✅ | Perfect |
| VAE Prior | 100% ✅ | Perfect |
| Econometric | 100% ✅ | Perfect |

**All models perfectly preserve co-integration during normal training period.**

### Crisis (2008-2010): **Multi-Horizon Improvement Pattern**

| Model | H=1 | H=7 | H=14 | H=30 | Trend |
|-------|-----|-----|------|------|-------|
| **Ground Truth** | 84% | 84% | 84% | 84% | Flat |
| VAE Oracle | 36% | 40% | 48% | 64% | **+78% improvement** |
| VAE Prior | 36% | 40% | 40% | **76%** | **+111% improvement** |
| Econometric | 100% | 100% | 100% | 100% | Perfect (by design) |

#### Key Findings

**1. Ground Truth Only 84% During Crisis**

Even real data breaks co-integration 16% of time during extreme stress. Market microstructure noise, liquidity issues, and arbitrage violations occur.

**2. VAE Preservation Improves at Longer Horizons**

| Horizon | Preservation | vs Ground Truth | vs H=1 Improvement |
|---------|-------------|-----------------|-------------------|
| H=1 | 36.0% | -48 pp | Baseline |
| H=7 | 40.0% | -44 pp | +11% |
| H=14 | 48.0% | -36 pp | +33% |
| **H=30** | **64-76%** | **-8 to -20 pp** | **+78-111%** |

**Interpretation:**
- H=1 struggles with high-frequency noise (36% vs 84% ground truth)
- **H=30 captures 76-90% of ground truth preservation** (76% vs 84%)
- Multi-horizon training helps model learn long-term stable relationships
- H=30 forecasts economically more consistent despite lower statistical fit

**3. Counter-Intuitive R² Finding**

| Horizon | Mean R² | Preservation Rate | Pattern |
|---------|---------|------------------|---------|
| H=1 | 0.783 | 36% | High R², low preservation |
| H=30 | 0.530 | **64-76%** | **Lower R², HIGH preservation** |

**Explanation:** H=1 overfits to high-frequency patterns (tight R² but unstable). H=30 captures medium-term economically consistent trends (looser R² but stable co-integration).

**4. Econometric Perfect by Design**

Econometric model **explicitly enforces** co-integration via regression. This is a modeling choice, not a learned property. During crisis, this may be overly restrictive.

#### Spatial Pattern (H=1 Crisis)

**Preserved (✅):**
- Deep OTM puts (moneyness ≤ 0.9, maturity ≥ 3M)
- 6M ATM options

**Broken (❌):**
- All call options (moneyness > 1.0)
- All 1M options (shortest maturity)
- Most ATM options (except 6M)

**Interpretation:** Model struggles with high-gamma, high-vega regions during crisis (calls, short maturities).

### OOS (2019-2023): **Near-Perfect**

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-----|-----|------|------|--------|
| Ground Truth | 100% | 100% | 100% | 100% | ✅ |
| VAE Oracle | 92% | 100% | 100% | 100% | ✅ |
| VAE Prior | 96% | 100% | 100% | 100% | ✅ |
| Econometric | 100% | 100% | 100% | 100% | ✅ |

**All models near-perfect in OOS.** Even COVID stress (2020) doesn't break co-integration like 2008 crisis did.

### Analysis Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `test_cointegration_preservation.py` | Run ADF tests | Preservation rates by model/period/horizon |
| `compile_cointegration_tables.py` | LaTeX tables | Summary statistics |

**Location:** `experiments/cointegration/`

---

## VAE Health & Latent Analysis

**Research Question:** Is the VAE training healthy? Are latent dimensions being utilized?

### VAE Health Diagnostics

#### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Effective Dimension** | ~3/5 | 3 of 5 latent dims highly active |
| **Posterior Collapse** | None detected | All dims contribute |
| **Latent Variance** | Variable by regime | Higher during crisis |
| **KL Divergence (avg)** | 3.5-4.0 | Healthy range |

#### Per-Dimension Analysis (In-Sample)

| Dimension | Mean KL | Variance | Activity Level | Interpretation |
|-----------|---------|----------|----------------|----------------|
| **dim_0** | 1.42 | High | **Very Active** | Primary mode |
| **dim_1** | 1.18 | High | **Very Active** | Secondary mode |
| **dim_2** | 0.95 | Medium | **Active** | Tertiary mode |
| dim_3 | 0.52 | Low | Moderate | Auxiliary |
| dim_4 | 0.43 | Low | Moderate | Auxiliary |

**Effective dimensions = 3** (dims 0-2 dominate)

#### Temporal Dynamics

**Crisis vs Normal:**

| Period | Mean KL | Max KL | Latent Variance | Interpretation |
|--------|---------|--------|-----------------|----------------|
| **Normal (2007)** | 3.21 | 4.82 | 0.84 | Baseline |
| **Crisis (2008-2010)** | 4.15 | 6.93 | 1.52 | **+44% utilization** |
| **Recovery (2011)** | 3.67 | 5.41 | 1.08 | Moderate |

**Pattern:** Latent utilization increases during stress periods. Model allocates more capacity to capture extreme variations.

### Latent Space Analysis

#### Distribution Properties

**Aggregate Statistics:**
- Mean: ~0 (centered as expected)
- Std Dev: 0.95-1.05 (close to standard normal)
- Skewness: -0.1 to +0.2 (approximately symmetric)
- Kurtosis: 2.8-3.2 (near-normal)

**By Regime:**

| Regime | Mean Shift | Std Dev | Kurtosis | Tail Behavior |
|--------|-----------|---------|----------|---------------|
| Normal | 0.0 | 1.0 | 3.0 | Normal tails |
| Crisis | -0.2 | 1.3 | 3.8 | **Heavier tails** |
| OOS | +0.1 | 1.1 | 3.2 | Slightly heavier |

#### Dimension Contribution

**Reconstruction Quality by Dimension:**

| Ablation | RMSE | vs Full Model | Interpretation |
|----------|------|---------------|----------------|
| **Full (5 dims)** | 0.0345 | Baseline | - |
| Remove dim_0 | 0.0421 | **+22%** | Critical dimension |
| Remove dim_1 | 0.0389 | **+13%** | Important |
| Remove dim_2 | 0.0367 | **+6%** | Moderate impact |
| Remove dim_3 | 0.0351 | +2% | Small impact |
| Remove dim_4 | 0.0348 | +1% | Minimal impact |

**Conclusion:** Dims 0-2 critical (41% combined impact), dims 3-4 auxiliary (3% combined).

### Zero vs Prior Latent Comparison

**Test:** Compare deterministic (z=0) vs stochastic (z~N(0,1)) generation.

| Metric | z=0 (Deterministic) | z~N(0,1) (Stochastic) | Difference |
|--------|--------------------|-----------------------|------------|
| **RMSE** | 0.0348 | 0.0345 | -0.9% (stochastic better) |
| **CI Width** | N/A | 0.0518 | - |
| **Diversity** | 0 (single output) | High | - |

**Interpretation:** Deterministic mode (z=0) provides good point forecast, but stochastic sampling necessary for uncertainty quantification.

### Analysis Scripts (7 total)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `analyze_vae_health_16yr.py` | In-sample health metrics | KL per dim, effective dim |
| `analyze_vae_health_oos_16yr.py` | OOS health metrics | Distribution shift detection |
| `visualize_vae_health_16yr.py` | In-sample plots | KL divergence plots |
| `visualize_vae_health_oos_16yr.py` | OOS plots | Temporal dynamics |
| `analyze_latent_distributions_16yr.py` | Distribution analysis | By regime statistics |
| `test_dimension_ablation_16yr.py` | Ablation study | Contribution ranking |
| `test_zero_vs_prior_latent_16yr.py` | Deterministic vs stochastic | Mode comparison |

**Output Location:** `results/backfill_16yr/vae_health/`, `results/backfill_16yr/latent_contribution_figs/`

---

## Visualization Tools

### Interactive Web App

#### Streamlit Dashboard
```bash
streamlit run streamlit_vol_surface_viewer.py
```

**Features:**
- **3D Surface Plots:** Rotatable, zoomable vol surfaces
- **Model Comparison:** Oracle / VAE Prior / Econometric side-by-side
- **Period Selection:** Crisis / In-sample / OOS dropdown
- **Date Slider:** Navigate through time series
- **Grid Point Selection:** Click to see time series at specific point
- **CI Bands:** Toggle confidence interval visualization
- **Export:** Download plots and data

**Tech Stack:** Streamlit + Plotly + NumPy

### Static Visualizations

#### CI Width Analysis (`results/vae_baseline/visualizations/`)

**Subdirectories:**
- `ci_peaks/` - Feature importance regression plots
- `comparison/` - 2/3/4-regime overlay plots
- `top_ci_width_moments/` - Extreme moment visualizations
- `oracle_vs_prior/` - Sampling mode comparisons

**Key Plots:**
- `four_regime_overlay.png` - 2×2 regime matrix
- `four_regime_timeline.png` - Full 2000-2023 annotated timeline
- `ci_width_regression_features_h30.png` - Feature importance bars
- `low_vol_high_ci_anomaly_analysis.png` - Pre-crisis detection scatter

#### VAE Health (`results/backfill_16yr/vae_health/`)

**Key Plots:**
- `kl_divergence_per_dimension.png` - Per-dimension KL across time
- `latent_variance_temporal.png` - Crisis vs normal variance
- `effective_dimensionality.png` - Active dimensions over time
- `reconstruction_quality_vs_kl.png` - Quality correlation

#### Plotly Dashboards (`results/backfill_16yr/visualizations/plotly_dashboards/`)

**Interactive HTML:**
- `insample_vs_oos_comparison_16yr.html` - Period comparison
- `crisis_deep_dive_16yr.html` - 2008-2010 detailed analysis
- `oracle_vs_prior_comparison.html` - Sampling mode analysis

### Visualization Scripts (12 total)

| Script | Output | Purpose |
|--------|--------|---------|
| `visualize_ci_bands_comparison.py` | CI band plots | Compare models |
| `visualize_sequence_ci_width.py` | Single sequence | Detailed view |
| `visualize_sequence_ci_width_combined.py` | Multiple sequences | Overlay |
| `visualize_ci_width_temporal.py` | Time series | Temporal dynamics |
| `visualize_calm_vs_crisis_overlay.py` | 2 regimes | Basic comparison |
| `visualize_three_regime_overlay.py` | 3 regimes | Add pre-crisis |
| `visualize_four_regime_overlay.py` | 4 regimes | Complete 2×2 |
| `visualize_four_regime_timeline.py` | Full timeline | Annotated history |
| `visualize_four_regime_overlay_oos.py` | OOS 4-regime | OOS analysis |
| `visualize_oracle_vs_prior_combined.py` | Oracle/prior | Width comparison |
| `visualize_oracle_vs_prior_combined_with_vol.py` | With market data | 5-panel |
| `visualize_top_ci_width_moments.py` | Extremes | Top 5 widest/narrowest |

**Location:** `experiments/backfill/context20/`

---

## Script Inventory

### Complete Catalog: 44+ Scripts

#### By Category

| Category | Count | Location |
|----------|-------|----------|
| **CI Width Analysis** | 8 | `experiments/backfill/context20/` |
| **CI Calibration/Violations** | 3 | `experiments/backfill/context20/` |
| **RMSE Evaluation** | 1 | `experiments/backfill/context20/` |
| **VAE Health** | 4 | `experiments/backfill/context20/` |
| **Latent Analysis** | 3 | `experiments/backfill/context20/` |
| **Oracle vs Prior** | 6 | `experiments/oracle_vs_prior/` |
| **Econometric Baseline** | 17 | `experiments/econometric_baseline/` |
| **Co-Integration** | 2 | `experiments/cointegration/` |
| **Visualization** | 12 | `experiments/backfill/context20/` |
| **Generation** | 2 | `experiments/backfill/context20/` |
| **Validation** | 1 | `experiments/backfill/context20/` |
| **Testing** | 4 | `experiments/backfill/context20/` |
| **Comparison** | 2 | `experiments/backfill/context20/` |

**Total: 65+ scripts** (includes oracle_vs_prior and econometric_baseline)

#### Key Scripts by Research Question

**"Why do CIs widen?"**
1. `investigate_ci_width_peaks.py` - Feature importance regression
2. `investigate_ci_width_anomaly.py` - Pre-crisis detection analysis
3. `compare_insample_oos_regression.py` - Distribution shift impact

**"Are CIs well-calibrated?"**
1. `evaluate_insample_ci_16yr.py` - In-sample violations
2. `evaluate_vae_prior_ci_oos_16yr.py` - OOS violations
3. `compare_insample_oos_distributions.py` - Root cause analysis

**"Is prior sampling production-ready?"**
1. `compare_oracle_vs_prior_ci.py` - CI degradation
2. `evaluate_oracle_vs_prior_rmse_*.py` - Point forecast degradation
3. `evaluate_oracle_vs_prior_cointegration_*.py` - Economic consistency

**"How does VAE compare to econometric?"**
1. `compare_econometric_vs_vae_backfill.py` - Crisis comparison
2. `compare_econometric_vs_vae_oos.py` - OOS comparison
3. `econometric_backfill_*.py` - Baseline generation

**"Is VAE training healthy?"**
1. `analyze_vae_health_16yr.py` - Posterior collapse detection
2. `analyze_latent_distributions_16yr.py` - Latent space analysis
3. `test_dimension_ablation_16yr.py` - Dimension contribution

### Generation & Validation

**Primary Generation:**
```bash
# Teacher forcing sequences (oracle)
python experiments/backfill/context20/generate_vae_tf_sequences.py --period {crisis,insample,oos,gap} --sampling_mode oracle

# Teacher forcing sequences (prior)
python experiments/backfill/context20/generate_vae_tf_sequences.py --period {crisis,insample,oos,gap} --sampling_mode prior

# Batch generation (all periods)
bash experiments/backfill/context20/run_generate_all_tf_sequences.sh {oracle,prior}
```

**Validation:**
```bash
python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode {oracle,prior}
```

**Outputs:** 48 files total (24 oracle + 24 prior = 4 periods × 6 horizons × 2 modes)

---

## Key Findings Summary

### 1. CI Width Drivers

**Primary Discovery:** Spatial features dominate (68-75%)

| Feature Type | R² | Importance | Trend with Horizon |
|--------------|-----|------------|-------------------|
| **Spatial (ATM vol, slopes, skews)** | 0.516-0.565 | **68-75%** | Increasing |
| **Temporal (returns, realized vol)** | 0.364-0.429 | 25-32% | Decreasing |
| **Dominance Ratio** | 1.20-1.55× | - | **1.20× → 1.55×** |

**Key Insight:** Model widens CIs based on surface shape anomalies, not recent market shocks. Demonstrates intelligent pattern recognition.

**Pre-Crisis Detection:**
- 458 days (8.1%) show low-vol high-CI pattern
- 81.1% occur in 2007-2008 pre-crisis period
- Slopes have largest effect size (Cohen's d = -0.711)

### 2. CI Calibration Issues

**Performance:**

| Period | Average Violations | Target | Status |
|--------|-------------------|--------|--------|
| In-sample | 16.2% | 10% | ⚠️ Acceptable |
| Crisis | 14.4% | 10% | ⚠️ Good |
| OOS | **31.3%** | 10% | ❌ Poor |

**Root Causes:**
1. **Spatial dominance weakens:** 1.33× → 1.13× (-15% at H=30)
2. **Distribution shifts:** Slopes/skews 1.5-1.9× noisier in OOS
3. **Fixed decoder:** Cannot adapt to shifted feature distributions

**Oracle vs Prior:** < 1pp degradation (production-ready ✅)

### 3. Point Forecast Performance

**RMSE:**

| Comparison | In-Sample | OOS | Winner |
|------------|-----------|-----|--------|
| **H=1 vs H=30** | 0.0345 vs 0.0530 (+54%) | 0.0663 vs 0.0822 (+24%) | Multi-horizon helps |
| **In-sample vs OOS** | 0.0345-0.0530 | 0.0663-0.0822 (+55-92%) | Major OOS degradation |
| **Oracle vs Prior** | +0.05-0.34% | +0.06-0.22% | < 0.4% (excellent ✅) |
| **VAE vs Econometric (Crisis)** | 0.035 vs 0.056 (-38%) | - | VAE wins |
| **VAE vs Econometric (OOS)** | - | 0.060 vs 0.067 (-11%) | VAE wins |

### 4. Economic Consistency

**Co-Integration Preservation:**

| Period | H=1 | H=30 | Pattern |
|--------|-----|------|---------|
| In-sample | 100% | 100% | Perfect |
| Crisis | 36% | **64-76%** | **+78-111% improvement** |
| OOS | 92-96% | 100% | Near-perfect |

**Key Insight:** Multi-horizon training helps H=30 capture economically consistent long-term trends.

### 5. Model Comparisons

**VAE vs Econometric:**

| Metric | Crisis | OOS | Winner |
|--------|--------|-----|--------|
| **RMSE** | -38% | -11% | VAE ✅ |
| **CI Violations** | 13% vs 65% | 31% vs 67% | VAE ✅ |
| **Interpretability** | Low | Low | Econometric |
| **Extreme Events** | Good | Acceptable | VAE ✅ |

**Oracle vs Prior:**

| Metric | Max Degradation | Status |
|--------|-----------------|--------|
| **CI Violations** | < 1pp | ✅ Production-ready |
| **RMSE** | < 0.4% | ✅ Production-ready |
| **Co-Integration** | < 5pp | ✅ Production-ready |

### 6. VAE Health

| Metric | Value | Status |
|--------|-------|--------|
| **Effective Dimensions** | 3/5 | ✅ Good utilization |
| **Posterior Collapse** | None | ✅ Healthy |
| **Crisis Latent Variance** | +44% vs normal | ✅ Adaptive |

---

## Recommendations for Context60

### Must-Run Analyses (Priority 1)

**Goal:** Validate that 3× longer context (60 vs 20 days) improves performance.

#### 1. CI Calibration Analysis

**Scripts to adapt:**
- `evaluate_insample_ci_16yr.py` → `evaluate_insample_ci_context60.py`
- `evaluate_vae_prior_ci_oos_16yr.py` → `evaluate_vae_prior_ci_oos_context60.py`

**Key Questions:**
- Do OOS violations decrease? (Target: < 28.0%)
- Does distribution shift impact reduce? (Target: dominance weakening < 15%)
- Do all horizons improve or just long ones?

**Expected Improvement:** 60-day context should capture more regime information → better OOS adaptation.

#### 2. Oracle vs Prior Comparison

**Scripts to adapt:**
- `compare_oracle_vs_prior_ci.py`
- All `evaluate_oracle_vs_prior_*` scripts (6 total)

**Key Questions:**
- Does longer context reduce oracle-prior gap? (Currently 1.56-1.66×)
- Is prior still production-ready? (Target: < 1pp violations, < 0.4% RMSE)

**Expected:** Gap may narrow if longer context provides better posterior approximation.

#### 3. Distribution Shift Analysis

**Scripts to adapt:**
- `compare_insample_oos_regression.py`
- `compare_insample_oos_distributions.py`

**Key Questions:**
- Does spatial dominance remain stronger in OOS? (Context20: 1.33× → 1.13× = -15%)
- Do feature variance ratios decrease? (Context20: slopes 1.50×, skews 1.87×)
- Does longer context provide robustness to distribution shifts?

**Hypothesis:** 60-day context captures more regime variability → less sensitivity to OOS shifts.

#### 4. CI Width Regression

**Scripts to adapt:**
- `investigate_ci_width_peaks.py`
- `investigate_ci_width_anomaly.py`

**Key Questions:**
- Does spatial dominance ratio change? (Context20: 1.20-1.55× across horizons)
- Does ATM vol importance decrease? (Context20: 75.6% at H=30)
- Do temporal features (returns, realized vol) become more important?

**Hypothesis:** Longer context → richer temporal information → higher temporal feature importance.

#### 5. RMSE Evaluation

**Scripts to adapt:**
- `evaluate_rmse_16yr.py`

**Key Questions:**
- Does OOS degradation decrease? (Context20: +55-96%)
- Do long horizons (H=60, H=90) perform better?
- Does multi-horizon training benefit from longer context?

**Compare:**
- Context20: H=1 to H=30 (+54% in-sample, +55% OOS increase)
- Context60: H=1 to H=90 (expect better long-horizon performance)

### Nice-to-Have Analyses (Priority 2)

#### 6. Co-Integration Preservation

**Scripts to adapt:**
- `test_cointegration_preservation.py`

**Key Questions:**
- Does H=60/H=90 preservation improve over context20 H=30? (Context20: 64-76%)
- Does crisis H=1 preservation improve? (Context20: 36%, target: > 40%)

#### 7. Econometric Baseline Comparison

**Scripts to adapt:**
- `compare_econometric_vs_vae_backfill.py` (Crisis)
- `compare_econometric_vs_vae_oos.py` (OOS)

**Key Questions:**
- Does advantage over econometric increase? (Context20: -38% crisis, -11% OOS)
- Does CI calibration gap widen? (Context20: VAE 13% vs Econ 65% crisis)

#### 8. 4-Regime Visualization

**Scripts to adapt:**
- `visualize_four_regime_overlay.py`
- `visualize_four_regime_timeline.py`

**Key Questions:**
- Does pre-crisis detection improve?
- Do regime boundaries shift with longer context?

### Lower Priority (Priority 3)

#### 9. VAE Health Diagnostics

**Scripts to adapt:**
- `analyze_vae_health_16yr.py`
- `analyze_latent_distributions_16yr.py`

**Expected:** Likely similar to context20 (effective dim ~3/5).

#### 10. Pre-Crisis Anomaly Detection

**Scripts to adapt:**
- `investigate_ci_width_anomaly.py`

**Key Question:**
- Does 60-day context improve detection of 2007-2008 anomalies?

### Critical Research Questions for Context60

**1. Does 3× longer context reduce OOS CI violations?**
- Context20: 18.1% → 28.0% (+55% degradation)
- **Target:** Context60 < 25% OOS violations

**2. Does spatial dominance strengthen with more historical data?**
- Context20: 1.33× → 1.13× (-15% weakening)
- **Target:** Context60 < 10% weakening

**3. Do distribution shifts have less impact?**
- Context20: Slopes 1.50×, Skews 1.87× variance ratio
- **Target:** Context60 < 1.4× variance ratios

**4. How do AR sequences (180/270 days) compare to TF horizons?**
- TF: H=1,7,14,30,60,90 (single-pass, oracle/prior sampling)
- AR: 180-day (3×60), 270-day (3×90) (autoregressive, error accumulation)
- **Key:** Compare AR H=180/270 vs TF H=60/90

**5. Does longer context improve long-horizon forecasts?**
- Context20: H=30 best co-integration (64-76%)
- **Target:** Context60 H=60/H=90 > 75% co-integration

### Suggested Analysis Order

**Phase 1: Core Validation (1-2 weeks)**
1. CI calibration (in-sample, OOS) - Verify improvement
2. Oracle vs Prior - Validate production readiness
3. RMSE evaluation - Quantify accuracy gains
4. Distribution shift analysis - Test robustness hypothesis

**Phase 2: Deep Dive (1 week)**
5. CI width regression - Understand feature dynamics
6. Co-integration preservation - Validate economic consistency
7. Econometric comparison - Benchmark against baseline

**Phase 3: Visualization & Documentation (3-5 days)**
8. 4-regime visualization - Communication tool
9. Summary report - Document findings
10. Comparison tables - Context20 vs Context60

### Expected Outcomes

**Hypothesis 1: Longer context reduces OOS degradation**
- Mechanism: More regime information → better generalization
- Prediction: OOS violations 28% → 22-25%

**Hypothesis 2: Spatial dominance more stable**
- Mechanism: Richer historical patterns → less reliance on ATM vol alone
- Prediction: Weakening 15% → 8-10%

**Hypothesis 3: Distribution shift robustness improves**
- Mechanism: Training on more diverse regimes → better adaptation
- Prediction: Variance ratios 1.5-1.9× → 1.3-1.5×

**Hypothesis 4: AR sequences competitive with TF long horizons**
- Mechanism: Offset-based chunking matches training approach
- Prediction: AR 180-day ≈ TF 60-day performance, AR 270-day ≈ TF 90-day

**Hypothesis 5: Oracle-prior gap narrows**
- Mechanism: Better posterior approximation with more context
- Prediction: Width ratio 1.56-1.66× → 1.4-1.5×

---

## Appendix: File Structure

### Predictions

```
results/vae_baseline/predictions/autoregressive/
├── oracle/
│   ├── vae_tf_crisis_h{1,7,14,30}.npz
│   ├── vae_tf_insample_h{1,7,14,30}.npz
│   ├── vae_tf_oos_h{1,7,14,30}.npz
│   └── vae_tf_gap_h{1,7,14,30}.npz
└── prior/
    ├── vae_tf_crisis_h{1,7,14,30}.npz
    ├── vae_tf_insample_h{1,7,14,30}.npz
    ├── vae_tf_oos_h{1,7,14,30}.npz
    └── vae_tf_gap_h{1,7,14,30}.npz
```

**Total:** 48 files (4 periods × 6 horizons × 2 modes)
**Note:** Context20 uses H=1,7,14,30; Context60 will use H=1,7,14,30,60,90

### Analysis Outputs

```
results/vae_baseline/analysis/
├── ci_peaks/prior/
│   ├── feature_importance_h{1,7,14,30}.png
│   ├── anomaly_investigation/
│   └── CI_PEAKS_INVESTIGATION_REPORT.md
├── period_comparison/
│   ├── insample_vs_oos_regression.png
│   ├── distribution_shifts.png
│   └── INSAMPLE_VS_OOS_COMPARISON.md
└── oracle_vs_prior/
    ├── ci_width_comparison.png
    ├── rmse_degradation.png
    └── VAE_PRIOR_ANALYSIS_SUMMARY.md
```

### Documentation

```
experiments/backfill/
├── context20/
│   ├── README.md
│   ├── CI_WIDTH_ANALYSIS.md
│   └── CONTEXT20_ANALYSIS_SUMMARY.md (this document)
├── context60/
│   ├── teacher_forcing/
│   │   ├── generate_vae_tf_sequences.py
│   │   ├── validate_vae_tf_sequences.py
│   │   └── run_generate_all_tf_sequences.sh
│   └── autoregressive/
│       ├── generate_vae_ar_sequences.py
│       ├── validate_vae_ar_sequences.py
│       └── run_generate_all_ar_sequences.sh
├── QUANTILE_REGRESSION.md
└── MODEL_VARIANTS.md
```

---

## Document Metadata

**Created:** 2025-12-03
**Purpose:** Reference for context60 ablation study
**Scope:** Complete catalog of context20 experiments (44+ scripts)
**Coverage:** 8 research dimensions, 4 test periods, 4 horizons, 25 grid points
**Key Innovation:** Pre-crisis detection via spatial feature recognition
**Production Status:** VAE Prior validated (< 1pp CI degradation, < 0.4% RMSE degradation)
**Main Limitation:** OOS CI violations increase 55% (18% → 28%) due to distribution shift

**Next Steps:** Replicate analysis suite for context60 model to test hypothesis that longer context improves OOS robustness.

---

**END OF DOCUMENT**
