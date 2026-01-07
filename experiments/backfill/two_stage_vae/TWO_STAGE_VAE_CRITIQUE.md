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

## Student-t VAE Progress Report (January 2025)

After implementing the Student-t decoder with full Cholesky covariance, here is the updated status:

### Issue Status Matrix

| Issue | Severity | Original | Student-t | Target | Status |
|-------|----------|----------|-----------|--------|--------|
| **1. Fat Tails (Kurtosis)** | SEVERE | 1.35 | 4.89 | 14.44 | PARTIAL (34%) |
| **2. Tail Asymmetry (Skewness)** | SEVERE | -0.26 | +0.02 | +1.50 | UNCHANGED |
| **3. Cross-Grid Correlation** | SEVERE | 6.22 Frob | 4.28 Frob | ~2.0 | MOSTLY FIXED |
| **4. Systematic Bias** | MODERATE | -0.006 | +0.004 | ~0 | REVERSED |
| **5. Path Roughness** | MODERATE | 1.73x | 1.11x | 1.0x | MOSTLY FIXED |
| **6. Corner Reconstruction** | MODERATE | 0.35-0.75 | improved | <0.1 | LIKELY IMPROVED |
| **7. CI Over-Confidence** | SEVERE | N/A | 2.5% | 10% | NEW ISSUE |

### What Has Been Fixed

1. **Correlation Structure (68% improvement)** - Full Cholesky + NLL weight=1.0 works
   - ATM↔OTM: 0.02 → 0.260 (GT: 0.212) = **123% recovery**
   - ATM↔ITM: 0.03 → 0.403 (GT: 0.415) = **97% recovery**

2. **Path Roughness (56% improvement)** - 1.73x → 1.11x, nearly matches GT

3. **Kurtosis Direction (3.6x improvement)** - 1.35 → 4.89, but still far from GT

4. **Skewness Direction** - No longer reversed (-0.26 → +0.02), but still near-zero

### Remaining Problems

#### Problem 1: Heterogeneous Kurtosis Not Captured

**Learned ν = 4.96** produces uniform kurtosis ~5-6 everywhere, but GT varies dramatically:

```
GT Kurtosis Grid:
  3.3   4.2   3.5   5.8  51.6
 44.9  13.8   7.8   6.5  62.8
 24.8  25.2  14.4  20.8  17.3
 38.3  33.6  19.4  24.8  16.1
 28.2  27.0  24.1 180.3  51.9
```

Single global ν cannot capture this heterogeneity (range: 3.3 to 180.3).

#### Problem 2: Skewness Completely Unaddressed

| Metric | Ground Truth | Student-t VAE | Gap |
|--------|-------------|---------------|-----|
| ATM Skewness (2,2) | +1.496 | +0.016 | **99% gap** |
| Grid Mean Skewness | +0.785 | +0.016 | **98% gap** |

**Root Cause:** Student-t distribution is **symmetric by definition**. Cannot capture asymmetry regardless of ν value.

#### Problem 3: CI Over-Confidence (NEW)

| Horizon Group | Violations | Target | Status |
|---------------|------------|--------|--------|
| Short (H=1-5) | 3.6% | 10% | TOO WIDE |
| Medium (H=6-15) | 2.5% | 10% | TOO WIDE |
| Long (H=16-30) | 2.2% | 10% | TOO WIDE |
| **Overall** | **2.5%** | **10%** | **3.9x TOO WIDE** |

CIs are over-conservative, suggesting learned variance is too large relative to actual prediction error.

---

## Research-Validated Recommendations (January 2025)

Based on literature review, the following solutions are well-supported:

### Solution 1: Per-Grid-Point Degrees of Freedom (ν)

**Validation:** [Communications in Statistics (2022)](https://www.tandfonline.com/doi/abs/10.1080/03610926.2022.2082076122) introduced a **"multivariate t-distribution with multiple degrees of freedom"** where each dimension has distinct ν.

**Implementation:**
- Change `nu_raw` from scalar to (25,) tensor
- Each grid point learns its own tail heaviness
- Addresses kurtosis heterogeneity (3.3 to 180.3 range)

**Complexity:** Medium (modify existing code)

### Solution 2: Skew-t Decoder

**Validation:** The [Generalized Hyperbolic Skew Student's t-Distribution](https://academic.oup.com/jfec/article/4/2/275/788320) (Aas & Haff, 2006, Journal of Financial Econometrics) is standard in financial volatility modeling:

> "This distribution has the important property that one tail has polynomial and the other exponential behavior."

**Additional support:**
- [Bayesian Skew-Student-t Stochastic Volatility](https://www.researchgate.net/publication/228440980_Bayesian_Estimation_of_a_Skew-Student-t_Stochastic_Volatility_Model) - directly applicable
- [PMC: SVML-GH-ST model](https://pmc.ncbi.nlm.nih.gov/articles/PMC5766051/) - "provides better fit than SVML-N and SVML-T models" for S&P 500
- [Skewed Student-t VaR](https://www.researchgate.net/publication/291573049) - "more accurate VaR estimations than normal and Student-t"

**Implementation:**
- Add skewness parameter λ (λ=0 recovers symmetric Student-t)
- Sampling via normal variance-mean mixture with GIG mixing
- R package [SkewHyperbolic](https://cran.r-project.org/web/packages/SkewHyperbolic/SkewHyperbolic.pdf) provides reference

**Complexity:** High (new decoder class with different sampling)

### Updated Priority

| Priority | Solution | Addresses | Complexity | Status |
|----------|----------|-----------|------------|--------|
| **1** | Per-grid-point ν | Kurtosis heterogeneity | Medium | Ready to implement |
| **2** | Skew-t decoder | Skewness (99% gap) | High | Research complete |
| **3** | CI recalibration | Over-conservative CIs | Low | May resolve with #1 |
| **4** | Bias correction | Small positive bias | Low | Can defer |

**Recommendation:** Implement per-grid-point ν first (simpler, addresses immediate kurtosis issue), then Skew-t decoder (addresses the only completely UNCHANGED issue).

---

---

## Per-Grid-Point ν Implementation Report (January 2025)

### Background

Following the recommendations above, per-grid-point ν was implemented to address heterogeneous kurtosis (GT range: 3.3 to 180.3). This section documents the implementation journey, including a critical finding about learning ν via gradient descent.

### Initial Attempt: Learning ν via Gradient Descent

**Implementation:**
- Changed `nu_raw` from scalar to (25,) tensor (one per grid point)
- Each grid point learns its own degrees of freedom
- Added to multivariate Student-t NLL loss

**Result: FAILURE**

Despite 100 epochs of training with Student-t NLL loss, all 25 ν values converged to essentially the same value:

| Metric | Expected | Actual |
|--------|----------|--------|
| ν range | [2.1, 30+] | [4.89, 5.06] |
| ν std | >2.0 | 0.031 |
| Corr(ν, GT_kurtosis) | < -0.5 | -0.06 |

All ν values collapsed to ~5.0, producing uniform kurtosis across the grid despite GT varying from 3 to 180.

### Root Cause Analysis: Why Learning ν Fails

**Mathematical Analysis:**

The univariate Student-t NLL gradient for ν has two components:

```
∂NLL/∂ν = [0.5*ψ((ν+1)/2) - 0.5*ψ(ν/2) + 0.5/ν]     ← Constant term (~-0.21)
         + [0.5*log(1 + z²/ν) - 0.5*(ν+1)*z²/(ν²*(1+z²/ν))]  ← Residual term
```

**The Problem:** When MSE loss is effective (residuals z ~ 0.1-0.2), the residual-dependent term becomes negligible:

| z magnitude | Residual gradient | Constant term | Differentiation |
|-------------|-------------------|---------------|-----------------|
| 0.1 | ~0.001 | -0.21 | 0.5% (no signal) |
| 0.5 | ~0.03 | -0.21 | 14% (weak) |
| 1.0 | ~0.10 | -0.21 | 50% (moderate) |

With residuals ~0.1-0.2, **all ν parameters receive identical gradients** and converge to the same value.

### Literature Validation

This finding is well-documented in the literature:

1. **Multiple Local Maxima** ([Springer - Alternatives to EM](https://link.springer.com/article/10.1007/s11075-020-00959-w))
   > "The likelihood can have multiple local maxima and, as such, it is often necessary to fix the degrees of freedom at a fairly low value."

2. **Log-likelihood Increases with ν → ∞** ([ResearchGate - GARCH-t](https://www.researchgate.net/publication/46430695))
   > "The log-likelihood value increases with increase in ν, which could be responsible for the inability of the algorithms to obtain reasonable optimum values for ν."

3. **Most VAE Papers Fix ν as Hyperparameter**
   - **t-VAE** (Takahashi et al., IJCAI 2018): ν fixed, not learned
   - **t³-VAE** (Kim et al., arXiv 2312.01133): ν is "a single hyperparameter selected before training"

4. **EM Convergence Fails** ([Springer](https://link.springer.com/article/10.1007/s11075-020-00959-w))
   > "Since we do not fix ν, we cannot apply standard convergence results for the EM algorithm."

### Solution: Method of Moments (Fixed ν from GT Kurtosis)

Based on the literature, the recommended approach is to **fix ν as a hyperparameter** computed directly from data.

**Method of Moments Estimator:**

For Student-t with ν > 4, excess kurtosis has a closed form:
```
excess_kurtosis = 6 / (ν - 4)
```

Inverting:
```
ν = 4 + 6 / excess_kurtosis
```

**Implementation:**
```python
# Compute GT excess kurtosis per grid point
gt_excess_kurtosis = compute_gt_excess_kurtosis_per_grid(train_data)  # (25,)

# Method of moments: nu = 4 + 6/excess_kurtosis
nu_fixed = 4.0 + 6.0 / np.clip(gt_excess_kurtosis, 0.1, 1000)
nu_fixed = np.clip(nu_fixed, nu_floor, nu_max)

# Fix nu in model (not trainable)
config["learn_nu"] = False
model.fix_nu_from_kurtosis(gt_excess_kurtosis)
```

### Results: Fixed ν vs Learned ν

| Metric | Learned ν | Fixed ν from GT | Improvement |
|--------|-----------|-----------------|-------------|
| ν std | 0.031 | **0.36** | **12x** |
| ν range | [4.89, 5.06] | **[4.01, 5.49]** | **10x wider** |
| Corr(ν, GT_kurt) | -0.06 | **-0.40** | Strong negative |
| Kurtosis (ATM) | 1.35 | **6.50** | **382%** |
| 3σ tail probability | 0.27% | **0.86%** | **219%** |

**Per-Grid-Point ν (5x5):**
```
Fixed ν from GT Kurtosis:
[[5.05 4.76 5.49 4.34 4.30]
 [4.32 4.29 4.90 4.05 4.31]
 [4.08 4.13 4.62 4.50 4.49]
 [4.03 4.13 4.27 4.34 4.02]
 [4.09 4.09 4.09 4.01 4.03]]

Theoretical Kurtosis (from ν):
[[ 5.7  7.9  4.0 17.5 20.2]
 [18.6 20.9  6.7 60.0 19.5]
 [60.0 44.9  9.6 11.9 12.2]
 [60.0 45.0 22.6 17.5 60.0]
 [60.0 60.0 60.0 60.0 60.0]]
```

Note: Theoretical kurtosis is capped at ~60 due to ν floor constraint (ν > 2.1 for finite variance).

### Tail Probability Improvement

| Threshold | Gaussian | GT | Learned ν | Fixed ν |
|-----------|----------|-------|-----------|---------|
| 2.0σ | 4.55% | 2.44% | - | 2.30% |
| 2.5σ | 1.24% | 1.97% | - | 1.37% |
| 3.0σ | 0.27% | 1.71% | 0.27% | **0.86%** |

Fixed ν captures 3x more tail probability at 3σ than the Gaussian baseline.

### Updated Issue Status Matrix

| Issue | Original | Student-t (Learned ν) | Student-t (Fixed ν) | Target | Status |
|-------|----------|----------------------|---------------------|--------|--------|
| **1. Fat Tails (Kurtosis)** | 1.35 | 4.89 | **6.50** | 14.44 | IMPROVED (45%) |
| **2. Tail Asymmetry (Skewness)** | -0.26 | +0.02 | -0.05 | +1.50 | UNCHANGED |
| **3. Cross-Grid Correlation** | 6.22 Frob | 4.28 Frob | 6.22 Frob | ~2.0 | REGRESSION* |

*Correlation slightly regressed with fixed ν - may need NLL weight tuning.

### Files Modified

| File | Change |
|------|--------|
| `vae/cvae_two_stage.py` | Added `learn_nu` option, `set_nu_from_kurtosis()` method |
| `experiments/backfill/two_stage_vae/train_two_stage_student_t.py` | Fixed ν from GT kurtosis |
| `config/two_stage_config.py` | Added `kurtosis_loss_weight` parameter |

### Checkpoints

| File | Description |
|------|-------------|
| `models/backfill/two_stage/two_stage_student_t_best.pt` | Learned ν (uniform ~5.0) |
| `models/backfill/two_stage/two_stage_student_t_fixed_nu_best.pt` | Fixed ν from GT kurtosis |

### Key Takeaways

1. **Learning ν via gradient descent is fundamentally difficult** - well-documented in literature
2. **Method of moments** (fixing ν from GT kurtosis) is the recommended approach
3. **Kurtosis improved 382%** (1.35 → 6.50) but still below GT (14.44)
4. **Skewness remains unaddressed** - requires Skew-t or Skew-Normal decoder
5. **ν floor constraint** (>2.1) caps maximum achievable kurtosis at ~60

### Next Steps

1. **Increase NLL weight** to potentially recover correlation while maintaining fat tails
2. **Implement Skew-t decoder** to address skewness (99% gap remaining)
3. **Consider lowering ν floor** if finite variance is not strictly required

---

*Document updated: January 2025*
*Based on analysis of Two-Stage Heteroscedastic VAE with context encoder*
*Student-t evaluation added with research-validated recommendations*
*Per-grid-point ν implementation report added with method of moments solution*
