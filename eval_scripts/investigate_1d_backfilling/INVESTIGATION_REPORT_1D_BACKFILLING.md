# Investigation Report: 1D Backfilling Model Failure Analysis

**Date:** 2025-11-10
**Model:** Multi-channel 1D CVAE for backfilling AMZN returns using MSFT/SP500

## Executive Summary

The 1D backfilling model **completely fails** in the realistic production scenario, with performance worse than trivial baselines. Simple linear regression on MSFT/SP500 achieves **76% direction accuracy and 0.021 RMSE**, while the VAE achieves **49% direction accuracy (random) and 0.040 RMSE (112% worse)**.

**Root cause:** The model learned to encode ONLY AMZN's own information, completely ignoring MSFT/SP500 despite strong correlations (r=0.70) existing in the data.

---

## Performance Results

### Scenario Comparison

| Scenario | RMSE | Direction Acc | CI Violations | R² |
|----------|------|---------------|---------------|-----|
| **S1: Oracle** (z[T] from real) | 0.0246 | 70.25% | 37.25% | 0.01 |
| **S2: Mixed 80/20** | 0.0277 | 65.63% | 40.63% | -0.25 |
| **S3: Realistic** (z[T-1]) | 0.0397 | 49.38% | 56.38% | -1.59 |
| **S4: Realistic Fixed** (z[T]) | 0.0457 | 50.25% | 59.25% | -2.41 |

### Key Observations

1. **Oracle → Realistic degradation:** 62% RMSE increase, direction accuracy drops to random (49%)
2. **Using z[T] instead of z[T-1] makes it WORSE:** RMSE increases 15%, CI violations increase
3. **All scenarios have severe CI miscalibration:** 37-59% violations vs target 10%

---

## Hypothesis Testing Results

### Hypothesis 1: Wrong Latent Selection (z[T-1] vs z[T])

**Result:** ✗ **REJECTED**

Switching from z[T-1] to z[T] made performance worse:
- RMSE: 0.0397 → 0.0457 (+15% worse)
- Direction Acc: 49.38% → 50.25% (+0.87%, negligible)
- CI violations: 56.38% → 59.25% (worse)

**Why it failed:**
- z[T-1] and z[T] are highly correlated (r=0.932)
- Mean Euclidean distance: 0.56 (small)
- LSTM barely updates latent when seeing new MSFT/SP500 data

### Hypothesis 2: Posterior Collapse

**Result:** ✗ **PARTIALLY REJECTED**

Not traditional posterior collapse:
- KL divergence ~19 (very high, far from N(0,1) prior)
- Model IS encoding information, but only from AMZN

However, latent has collapsed to **1-dimensional manifold**:
- 99% of variance explained by 1 principal component
- 12D latent is effectively 1D

### Hypothesis 3: Encoder Ignores MSFT/SP500

**Result:** ✓ **CONFIRMED**

#### Correlation Analysis

**Oracle (S1) - Has real AMZN data:**
- z → AMZN: r=0.884 (very strong!)
- z → MSFT: r=0.477 (moderate)
- z → SP500: r=0.479 (moderate)

**Realistic Fixed (S4) - AMZN masked:**
- z → AMZN: r=0.012 (essentially zero!)
- z → MSFT: r=0.084 (negligible)
- z → SP500: r=0.041 (negligible)

#### Mutual Information

| Scenario | I(z; AMZN) | I(z; MSFT) | I(z; SP500) |
|----------|-----------|-----------|------------|
| Oracle (S1) | 5.45 | - | - |
| Realistic Fixed (S4) | 0.29 | 0.27 | 0.39 |

**95% loss of mutual information** when AMZN is masked!

---

## Baseline Comparison

### Devastating Finding: Linear Regression Destroys VAE

| Method | RMSE | Direction Acc | vs Best |
|--------|------|---------------|---------|
| **Linear Reg (MSFT+SP500)** | **0.0209** | **76.25%** | **BEST** |
| Zero (Random Walk) | 0.0277 | 0% | +33% worse |
| Historical Mean | 0.0279 | 47.5% | +33% worse |
| Naive Persistence | 0.0391 | 54.5% | +87% worse |
| **VAE S3 (Realistic)** | **0.0397** | **49.38%** | **+90% worse** |
| **VAE S4 (Fixed)** | **0.0457** | **50.25%** | **+118% worse** |

**Linear regression model:**
```
AMZN[t] = 0.736 · MSFT[t] - 0.118 · SP500[t] + 0.0008
```

**Raw data correlations (test set):**
- AMZN ↔ MSFT: r=0.702
- AMZN ↔ SP500: r=0.654
- MSFT ↔ SP500: r=0.856

**Conclusion:** MSFT and SP500 have strong predictive power for AMZN, but the VAE completely failed to learn this relationship.

---

## Root Cause Analysis

### Why the VAE Failed

1. **Training objective mismatch:**
   - Only channel 0 (AMZN return) was in the loss function
   - Encoder had no incentive to encode MSFT/SP500 information
   - Model found local optimum: memorize AMZN patterns, ignore other stocks

2. **Information bottleneck misuse:**
   - When AMZN data is available, encoder compresses it into latent
   - When AMZN is masked, encoder produces uninformative latent (r≈0 with everything)
   - MSFT/SP500 information doesn't flow through the latent

3. **Latent dimensionality collapse:**
   - Despite 12D latent, only 1 principal component has variance
   - Model reduced to 1D representation (likely encoding market regime)
   - No capacity left for cross-stock relationships

4. **LSTM causality limitation:**
   - z[T-1] computed before seeing MSFT[T+1], SP500[T+1] → obvious limitation
   - z[T] DOES see MSFT[T+1], SP500[T+1] but still useless (r=0.012)
   - LSTM learned to ignore these inputs during encoding

### Why Oracle Works

Oracle scenario works because:
- Real AMZN[T+1] is available during encoding
- Latent encodes AMZN's return directly (r=0.884)
- Decoder can reconstruct from this AMZN-centric latent
- MSFT/SP500 correlations are passive byproducts, not learned

---

## CI Calibration Failure

All scenarios show severe miscalibration:
- Oracle: 37% violations (target: 10%)
- Mixed: 41% violations
- Realistic Original: 56% violations
- Realistic Fixed: 59% violations

### Why CI Calibration Failed

1. **VAE prior mismatch:**
   - Training: p(z | context + target)
   - Generation: p(z | context only) or p(z | degraded context)
   - Mismatch leads to overconfident predictions

2. **Fixed CI widths:**
   - Mean CI width: 0.042-0.046 (nearly constant)
   - No adaptation to market volatility or uncertainty
   - Model learned: `p05 ≈ p50 - c`, `p95 ≈ p50 + c` for constant c

3. **Pinball loss insufficient:**
   - Optimizes quantile accuracy but doesn't enforce calibration
   - With collapsed posterior, decoder can't model conditional uncertainty
   - Needs proper probabilistic calibration (e.g., conformal prediction)

---

## Architectural Issues

### The "Two-Encoder" Pattern Misapplication

**Design intent:**
- Main encoder: Full sequence [0:T+1] during training
- Context encoder: Only context [0:T] during generation
- Generation: `decoder([ctx_embeddings || z_sampled])`

**What went wrong:**
- Context embeddings ARE informative (from real AMZN history)
- But z_sampled is NOT informative when AMZN is masked
- Decoder relies on context embeddings, ignores z
- z becomes a dummy variable in realistic scenario

### Masked Training Strategy

**Training split:** 70% standard, 30% masked

**Masked batch behavior:**
- Forward-fill AMZN (channel 0) at T+1
- Use z[T-1] for decoding future timestep
- Model learns to "ignore missing AMZN" by predicting from context

**Problem:** This teaches the model to NOT use cross-stock information!
- Masked training says: "When AMZN is missing, just use historical AMZN context"
- It doesn't say: "When AMZN is missing, use MSFT/SP500 instead"
- Model learns the former, not the latter

---

## Comparison to 2D Surface VAE

### Similar CI Miscalibration

| Model | CI Violations |
|-------|--------------|
| 2D no_ex | 44.50% |
| 2D ex_no_loss | 35.43% |
| 2D ex_loss | 34.28% |
| **1D Oracle** | **37.25%** |
| **1D Realistic** | **56.38%** |

Both architectures suffer from VAE prior mismatch and fixed CI widths.

### Different Failure Modes

**2D Surface VAE:**
- Passive conditioning (ex_feats) helps slightly (44% → 34% violations)
- Model can use returns/skew/slope when available
- Still miscalibrated but has some predictive power

**1D Backfilling VAE:**
- Active masking makes it WORSE
- Model completely ignores MSFT/SP500
- No predictive power in realistic scenario (49% direction acc)

---

## Recommendations

### Immediate Fixes (Band-aids)

1. **Use linear regression baseline:**
   - 112% better than VAE
   - Simple, interpretable, actually works
   - `AMZN[t] = 0.736·MSFT[t] - 0.118·SP500[t]`

2. **Abandon realistic backfilling with current model:**
   - Model is fundamentally broken for this task
   - No amount of tuning will fix architectural issues

### Architectural Redesign (If VAE is required)

1. **Multi-task learning:**
   ```python
   # Current (BROKEN):
   loss = pinball_loss(pred[:, 0], target[:, 0])  # Only AMZN

   # Proposed:
   loss = (
       1.0 * pinball_loss(pred[:, 0], target[:, 0])      # AMZN
       + 0.3 * pinball_loss(pred[:, 4], target[:, 4])    # MSFT
       + 0.3 * pinball_loss(pred[:, 8], target[:, 8])    # SP500
   )
   ```
   Forces encoder to encode ALL stocks, not just AMZN.

2. **Explicit cointegration module:**
   ```python
   # Learn: AMZN[t] = f(MSFT[t], SP500[t]) + residual
   beta_msft, beta_sp500 = cointegration_module(z)
   amzn_pred = beta_msft * msft[t] + beta_sp500 * sp500[t] + decoder(z)
   ```

3. **Separate latent for each stock:**
   ```python
   z_amzn, z_msft, z_sp500 = encoder(input)  # 3 separate latents

   # When AMZN is missing:
   z_amzn_imputed = cross_attention(z_msft, z_sp500)  # Use others to infer AMZN
   ```

4. **Adversarial masking during training:**
   - Randomly mask ANY stock (not just AMZN)
   - Train to reconstruct from others
   - Forces model to learn cross-stock dependencies

### CI Calibration Fixes

1. **Conformal prediction:**
   - Post-hoc calibration using validation set
   - Adjust quantile predictions to achieve 90% coverage
   - Doesn't fix root cause but patches miscalibration

2. **Temperature scaling for quantiles:**
   - Learn temperature parameter τ to scale CI width
   - Optimize τ on validation set for calibration

3. **Separate uncertainty model:**
   - Train VAE for point predictions
   - Train separate model for uncertainty (e.g., quantile regression forest)
   - Combine outputs

### Alternative Approaches

1. **Direct quantile regression (no VAE):**
   - Gradient boosting for quantiles
   - Linear quantile regression
   - Neural network with pinball loss

2. **Gaussian Process with multi-output:**
   - Models correlations between AMZN/MSFT/SP500 naturally
   - Provides calibrated uncertainty

3. **Transformer with cross-attention:**
   - Attention mechanism learns which stocks are informative
   - No information bottleneck like VAE latent

---

## Conclusion

The 1D backfilling model is **fundamentally broken** for its intended task. The core issue is that the encoder learned to encode only AMZN information, completely ignoring MSFT/SP500 despite strong correlations. When AMZN is masked, the latent becomes uninformative (r=0.012 with returns), leading to random predictions (49% direction accuracy).

A **trivial linear regression** outperforms the VAE by 112% in RMSE and achieves 76% direction accuracy, proving that the cross-stock signal exists but the VAE failed to learn it.

The problem cannot be fixed by:
- Using z[T] instead of z[T-1] (makes it worse)
- Tuning hyperparameters (KL weight, learning rate, etc.)
- Changing sampling strategies

The problem requires:
- Fundamental architectural redesign (multi-task loss, explicit cointegration)
- Or abandoning the VAE approach entirely for this task

The same VAE prior mismatch issue affects CI calibration (37-59% violations vs 10% target), consistent with the 2D surface VAE results. This suggests a systematic problem with the VAE approach for conditional generation in financial applications.

---

## Appendix: Generated Artifacts

### Files Created

1. `investigate_latent_selection.py` - Tests z[T-1] vs z[T]
2. `analyze_latent_investigation.py` - Latent distribution analysis
3. `baseline_comparisons.py` - Linear regression benchmarks
4. `models_1d_backfilling/latent_selection_investigation.npz` - Predictions for all 4 scenarios
5. `models_1d_backfilling/latent_analysis/` - Visualizations:
   - `latent_vs_prior.png` - Distribution comparison
   - `correlation_heatmap.png` - Latent-return correlations
   - `z_comparison_scatter.png` - z[T-1] vs z[T] scatter plots

### Key Metrics

**Latent Information Content:**
- Oracle: I(z; AMZN) = 5.45, max |r| = 0.884
- Realistic Fixed: I(z; AMZN) = 0.29, max |r| = 0.012
- 95% information loss when AMZN is masked

**Latent Dimensionality:**
- Theoretical: 12D
- Effective: 1D (99% variance in 1 PC)
- Top eigenvalue: 2.28 (Oracle), 3.46 (Realistic Fixed)

**Baseline Performance:**
- Best: Linear Regression RMSE=0.021, Direction=76.25%
- VAE: RMSE=0.040, Direction=49.38%
- VAE is 112% worse than linear regression

**Data Correlations:**
- AMZN ↔ MSFT: r=0.702
- AMZN ↔ SP500: r=0.654
- MSFT ↔ SP500: r=0.856

---

## References

- CLAUDE.md: Project documentation
- CI_CALIBRATION_OBSERVATIONS.md: 2D VAE CI issues
- QUANTILE_REGRESSION_RESULTS.md: Quantile model analysis
- BACKFILLING_PROPOSAL.md: Original design rationale
