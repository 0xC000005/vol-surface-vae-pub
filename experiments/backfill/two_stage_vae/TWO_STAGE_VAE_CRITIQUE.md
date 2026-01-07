# Two-Stage VAE Critique: Posterior Mode Analysis

## Overview

This document summarizes the distributional issues identified in the Two-Stage Heteroscedastic VAE when evaluated under **posterior mode** (z conditioned on target). The conditional variance prediction has been addressed via the heteroscedastic decoder with NLL loss, but fundamental distributional mismatches remain.

**Evaluation Setup:**
- 500 evaluation sequences, 500 samples per sequence
- 30-day horizon
- Posterior mode: z ~ q(z|context, target)
- Grid: 5x5 (moneyness x tenor)
- Primary analysis point: ATM, mid-tenor (2,2)

---

## Critical Issues (SEVERE)

### 1. Fat Tails Completely Missing

**Problem:** The VAE generates near-Gaussian samples while ground truth has extreme fat tails.

| Metric | Ground Truth | VAE | Ratio |
|--------|-------------|-----|-------|
| Kurtosis (H=1, ATM) | 22.84 | 0.79 | 0.03x |
| Kurtosis (H=15, ATM) | 21.34 | 1.35 | 0.06x |
| Kurtosis (H=30, ATM) | 2.82 | 0.90 | 0.32x |

**Kurtosis across entire grid (H=15):**
```
GT Kurtosis:                    VAE Kurtosis:
  6.5   5.7   4.3  10.8  28.3     3.8   0.2   0.2   0.8   6.3
 76.7  19.5   9.7  15.3  42.6     1.3   0.4   0.8  -0.3   6.8
 41.4  36.4  21.3  39.6  42.2     1.7   1.5   1.3  -0.0   4.5
 59.3  52.7  30.7  43.7  41.2     0.3   1.0   0.6   0.3  -0.3
 44.2  42.5  39.6 289.8  97.6    -0.0   0.4   0.6   1.2   0.1
```

**Impact:** VAE cannot generate extreme market events (crises, vol spikes). Risk metrics based on VAE samples will severely underestimate tail risk.

**Root Cause:** Gaussian likelihood (MSE/NLL) penalizes outliers quadratically, causing the decoder to avoid extreme predictions.

---

### 2. Tail Asymmetry Reversed

**Problem:** Ground truth shows positive skewness (vol spikes up sharply, drifts down slowly). VAE reverses this to slight negative skewness.

| Threshold | GT Pos/Neg Ratio | VAE Pos/Neg Ratio | Issue |
|-----------|------------------|-------------------|-------|
| 1.5 std | 1.80 | 0.80 | Reversed |
| 2.0 std | 1.78 | 0.75 | Reversed |
| 2.5 std | 2.32 | 0.72 | Reversed |
| 3.0 std | 3.19 | 0.67 | Severely Reversed |

**Raw counts at 2 std threshold:**
- GT: 237 negative extremes, 423 positive extremes
- VAE: 1,593 negative extremes, 1,198 positive extremes

**Skewness comparison:**
| Horizon | GT Skewness | VAE Skewness |
|---------|-------------|--------------|
| H=1 | +2.34 | -0.13 |
| H=5 | +2.66 | -0.15 |
| H=10 | +1.85 | -0.10 |
| H=15 | +1.60 | -0.26 |
| H=20 | +2.31 | -0.24 |

**Impact:** VAE generates too many large negative jumps (vol crashes) which are rare in reality, while under-generating large positive jumps (vol spikes) which are common.

**Root Cause:** Symmetric Gaussian decoder cannot capture asymmetric distributions. The slight negative bias in mean prediction exacerbates this.

---

### 3. Cross-Grid Correlation Destroyed

**Problem:** VAE generates grid points nearly independently, destroying the correlation structure of volatility surfaces.

| Correlation Pair | GT | VAE |
|-----------------|-----|-----|
| OTM-short (0,0) vs ATM-mid (2,2) | 0.32 | 0.02 |
| ITM-long (4,4) vs ATM-mid (2,2) | 0.50 | 0.03 |
| Mean absolute correlation difference | - | 0.44 |
| Frobenius norm of correlation matrix difference | - | 13.38 |

**Impact:** Generated surfaces lack coherent structure. Moneyness smile and term structure dynamics are not preserved across the grid.

**Root Cause:** Heteroscedastic decoder predicts mean and variance for each grid point independently. No mechanism to enforce cross-point correlations.

---

## Moderate Issues

### 4. Systematic Negative Bias

**Problem:** VAE predictions have consistent negative bias across all horizons.

| Horizon | GT Mean | VAE Mean | Bias |
|---------|---------|----------|------|
| H=1 | +0.0019 | -0.0089 | -0.0108 |
| H=5 | +0.0012 | -0.0045 | -0.0057 |
| H=15 | -0.0002 | -0.0065 | -0.0063 |
| H=30 | +0.0011 | -0.0050 | -0.0062 |

**Cumulative impact:** Over 30 days, bias compounds to ~-0.18 in log-return space, causing systematic underestimation of future IV levels.

**Root Cause:** Likely training data imbalance or decoder bias term initialization.

---

### 5. Paths Too Rough

**Problem:** VAE-generated sequences have 73% higher roughness than ground truth.

| Metric | GT | VAE | Ratio |
|--------|-----|-----|-------|
| Mean |change| per step | 0.0454 | 0.0784 | 1.73x |

**Impact:** Generated sequences are noisier than realistic market data, potentially affecting downstream applications that depend on path smoothness.

**Root Cause:** Independent sampling at each timestep. No temporal smoothness constraint in the decoder.

---

### 6. Reconstruction Quality Varies by Grid Position

**Problem:** Reconstruction error is good at ATM but poor at corners (extreme moneyness/tenor).

**RMSE per grid point (H=15):**
```
  0.590   0.093   0.097   0.746   0.376
  0.169   0.054   0.062   0.086   0.487
  0.067   0.042   0.049   0.059   0.345
  0.043   0.032   0.036   0.044   0.046
  0.056   0.041   0.032   0.056   0.063
```

**Pattern:**
- Center (ATM, mid-tenor): RMSE ~0.03-0.05 (good)
- Corners (OTM-short, ITM-long): RMSE ~0.35-0.75 (poor)

**Root Cause:** Less training data at extreme strikes/tenors, or model capacity concentrated on ATM region.

---

## Quantitative Summary

| Issue | GT Value | VAE Value | Severity |
|-------|----------|-----------|----------|
| Kurtosis (ATM, H=15) | 21.3 | 1.4 | SEVERE |
| Skewness (ATM, H=15) | +1.6 | -0.3 | SEVERE |
| Pos/Neg extreme ratio (2std) | 1.78 | 0.75 | SEVERE |
| Cross-grid correlation | 0.32-0.50 | 0.02-0.03 | SEVERE |
| Mean bias (per step) | ~0 | -0.006 | MODERATE |
| Path roughness | baseline | 1.73x | MODERATE |
| Corner RMSE | - | 0.35-0.75 | MODERATE |

---

## Root Cause Analysis

The fundamental issue is the **Gaussian assumption in the decoder**:

```
p(x|z) = N(mu(z), sigma(z)^2)
```

This assumption forces:
1. **Symmetric predictions** - Gaussian is symmetric, cannot capture skewness
2. **Light tails** - Gaussian has kurtosis=3, cannot capture fat tails (kurtosis 20+)
3. **Independence across grid points** - Diagonal covariance, no cross-correlation

The heteroscedastic extension (learning sigma per point) helps with variance calibration but does not address shape (skewness, kurtosis) or correlation issues.

---

## Potential Solutions

### For Fat Tails (Issue 1)

1. **Student-t Decoder**
   - Learn degrees of freedom (df) parameter per grid point
   - As df -> infinity, approaches Gaussian; low df gives fat tails
   - Loss: Negative log Student-t likelihood

2. **Mixture of Gaussians Decoder**
   - Learn mixture weights, means, variances
   - Can approximate fat-tailed distributions

### For Asymmetry (Issue 2)

1. **Skew-Normal Decoder**
   - Add skewness parameter alpha to Gaussian
   - `p(x) = 2 * phi(x) * Phi(alpha * x)`

2. **Asymmetric Laplace / Quantile Regression**
   - Directly model asymmetric loss
   - Pinball loss with learned quantiles

3. **Split-Normal Decoder**
   - Separate variance for positive and negative deviations

### For Correlation (Issue 3)

1. **Full Covariance Decoder**
   - Learn 25x25 covariance matrix (or Cholesky factor)
   - Multivariate Gaussian likelihood

2. **Copula Layer**
   - Separate marginal and dependence modeling
   - Learn correlation structure via copula

3. **Autoregressive Decoder**
   - Generate grid points sequentially
   - Each point conditioned on previous points

### Combined Solutions

1. **Normalizing Flow Decoder**
   - Can model arbitrary distributions
   - Captures fat tails, skewness, and correlations
   - Higher computational cost

2. **Diffusion Model**
   - Modern alternative to VAE
   - Better at multi-modal, complex distributions
   - Requires architecture change

---

## Recommended Priority

1. **High Priority:** Student-t or Skew-Normal decoder to fix tails and asymmetry
2. **Medium Priority:** Full covariance or autoregressive decoder for correlations
3. **Lower Priority:** Bias correction, path smoothness (can be post-processed)

---

## Files Referenced

- Analysis script: `experiments/backfill/prior_encoder_ablation/analyze_unconditional_marginal.py`
- Visualization: `experiments/backfill/prior_encoder_ablation/visualize_fanning_patterns.py`
- Data: `models/backfill/two_stage/unconditional_analysis.npz`
- Model: `vae/cvae_two_stage.py`

---

## Notes

- All analysis performed under **posterior mode** (z conditioned on target)
- Prior mode (z ~ N(0,1)) will have additional issues due to prior mismatch
- The heteroscedastic variance prediction is working (CI coverage ~2% vs target 10% indicates over-conservative CIs)
- Next step: Address distributional shape issues before tackling prior mode

---

*Document created: January 2025*
*Based on analysis of Two-Stage Heteroscedastic VAE with context encoder*
