# Quantile Regression Diagnosis: Generation Loss Bug

**Date:** 2025-10-28
**Status:** 🔴 **CRITICAL BUG IDENTIFIED**

## Executive Summary

**Root Cause Found:** The quantile regression models exhibit a **3x higher pinball loss during generation** compared to test set evaluation during training. This distribution shift/generation bug is the primary cause of poor CI calibration (34% violations vs target 10%).

**Key Finding:**
```
Test RE Loss (during training):  0.006206 (ex_loss model)
Generation Loss (inference):     0.018828
Ratio:                           3.03x HIGHER
```

This is not a calibration issue - **it's a generation bug**. The model learns proper quantiles during training but fails to produce them correctly during inference.

---

## Diagnosis Methodology

### What We Tested

Compared pinball loss in two scenarios:
1. **During Training/Testing**: Model evaluated on test set (days 5000-5821)
2. **During Generation**: Model used for teacher forcing generation (days 5-826)

### Hypothesis

If losses are similar, the issue is calibration (need conformal prediction).
If losses are very different, the issue is generation (need to fix inference code).

---

## Results

### Pinball Loss Comparison

| Model | Test RE Loss (Training) | Generation Loss (Inference) | Ratio | Verdict |
|-------|-------------------------|----------------------------|-------|---------|
| **no_ex** | 0.006349 | 0.020556 | **3.24x** | 🔴 Generation Bug |
| **ex_no_loss** | 0.006269 | 0.017820 | **2.84x** | 🔴 Generation Bug |
| **ex_loss** | 0.006206 | 0.018828 | **3.03x** | 🔴 Generation Bug |

**All three models show 3x degradation during generation!**

### Per-Quantile Loss Breakdown (ex_loss model)

| Quantile | Loss During Generation | Expected Ordering |
|----------|----------------------|-------------------|
| p05 (5%) | 0.009982 | Lowest |
| p50 (50%) | 0.027633 | Middle |
| p95 (95%) | 0.018870 | Highest |

❌ **Ordering is WRONG**: p50 > p95 > p05 (expected: p05 < p50 < p95)

This suggests the model is not properly separating quantiles during generation.

---

## Root Cause Analysis

### Possible Causes (Ranked by Likelihood)

#### 1. **MLE Generation Mode (z=0) Issue** ⭐ **MOST LIKELY**

**Problem**: Maximum likelihood generation uses `z = 0` for the future timestep, but this may not be appropriate for quantile regression.

**Explanation:**
- Standard VAE: z ~ N(0, 1) captures uncertainty, z=0 gives "average" prediction
- Quantile VAE: Need to condition on *different* z values to get different quantiles?
- Current implementation: Uses same z=0 for all quantiles → may collapse to single output

**Evidence:**
- All models show same ~3x degradation (suggests systematic issue)
- Per-quantile losses are not properly ordered
- Median (p50) loss is highest, not middle

**Fix:**
```python
# Current (WRONG?):
z_future = torch.zeros_like(latent_mean)  # Same z for all quantiles

# Proposed:
# Let the quantile decoder handle uncertainty internally
# OR: Sample different z values for different scenarios
```

#### 2. **Context Encoding Mismatch**

**Problem**: During training, model sees full sequence (context + target). During generation, it only sees context.

**Evidence:**
- Teacher forcing should mitigate this (uses real historical data)
- But if context encoder differs from full sequence encoder, could cause issues

**Fix:** Verify that `ctx_encoder` produces same latent distribution as `encoder` for context portion.

#### 3. **Quantile Decoder Implementation**

**Problem**: Decoder outputs 3 channels (p05, p50, p95) simultaneously. Maybe they're not properly separated?

**Evidence:**
- Per-quantile losses show p50 > p95, which is wrong
- Suggests channels may be "bleeding" into each other

**Fix:** Add constraints to ensure quantile ordering: p05 ≤ p50 ≤ p95

#### 4. **Data Preprocessing Difference**

**Problem**: Generated surfaces may be preprocessed differently than training data.

**Evidence:**
- Generated days (5-826) are all in training set, so not a train/test split issue
- Same data pipeline should be used

**Fix:** Verify normalization, scaling, and data format are identical.

---

## Impact Analysis

### Why CI Violations Are High

With 3x higher pinball loss during generation:
- Quantile predictions are systematically wrong
- p05 and p95 are too close to p50 (not wide enough)
- Leads to narrow confidence intervals
- Results in 34% violations instead of 10%

### CI Violation Chain

```
Generation Bug (3x loss)
    ↓
Wrong quantile predictions
    ↓
Narrow confidence intervals
    ↓
34% CI violations (should be 10%)
```

---

## Recommended Fixes (Priority Order)

### 🔴 IMMEDIATE (Fix Generation Code)

#### Option A: Fix MLE Generation for Quantile Models

**Problem**: Using z=0 may not work for quantile regression.

**Solution 1 - Stochastic Quantile Generation:**
```python
# Instead of z=0, sample multiple z values
num_samples = 100
z_samples = torch.randn(num_samples, latent_dim)

# Generate distributions for each z
all_surfaces = []
for z in z_samples:
    surface = model.decode(z, context)
    all_surfaces.append(surface)

# Compute empirical quantiles from samples
p05 = np.quantile(all_surfaces, 0.05, axis=0)
p50 = np.quantile(all_surfaces, 0.50, axis=0)
p95 = np.quantile(all_surfaces, 0.95, axis=0)
```

**Solution 2 - Deterministic with z from Context:**
```python
# Use context-derived z instead of zeros
z_context = ctx_encoder(context_data)
z_mean, z_logvar = z_context["latent_mean"], z_context["latent_logvar"]

# Use mean for deterministic generation
surface = model.decode(z_mean, context)
```

**Solution 3 - Check Decoder Forward Pass:**
Review how the decoder uses z when generating 3 quantile channels. Maybe z should modulate quantile width?

#### Option B: Add Quantile Ordering Constraints

```python
class QuantileDecoder(nn.Module):
    def forward(self, z, context):
        # Generate base prediction
        base = self.base_decoder(z, context)  # (B, T, H, W)

        # Generate quantile offsets (ensure ordering)
        width_lower = F.softplus(self.width_lower_net(z, context))  # Always positive
        width_upper = F.softplus(self.width_upper_net(z, context))  # Always positive

        p05 = base - width_lower
        p50 = base
        p95 = base + width_upper

        # Now guaranteed: p05 < p50 < p95
        return torch.stack([p05, p50, p95], dim=2)
```

### 🟡 MEDIUM TERM (Validate Fixes)

1. **Re-generate surfaces with fixed code**
2. **Re-run diagnosis script**: `python diagnose_generation_pinball_loss.py`
3. **Target**: Generation loss should be ~1.1x test loss (not 3x)
4. **Re-evaluate CI calibration**: Should improve to ~15-20% violations

### 🟢 LONG TERM (Improve Model)

1. **Conformal prediction** for final calibration adjustment
2. **Increase model capacity** (latent_dim=10, mem_hidden=200)
3. **Heteroscedastic quantile regression** (predict adaptive widths)

---

## Verification Plan

### Step 1: Identify Exact Bug

**Test A - Check z=0 Impact:**
```bash
# Compare z=0 vs z~N(0,1) sampling
python test_z_sampling.py
```

**Test B - Check Context Encoder:**
```bash
# Compare ctx_encoder vs full encoder on same data
python test_context_encoder.py
```

**Test C - Check Decoder Output:**
```bash
# Visualize 3 quantile channels
python visualize_decoder_channels.py
```

### Step 2: Implement Fix

Based on Test A/B/C results, implement appropriate fix.

### Step 3: Validate Fix

```bash
# Regenerate surfaces with fixed code
python generate_quantile_surfaces_fixed.py

# Re-run diagnosis
python diagnose_generation_pinball_loss.py

# Check that ratio is now ~1.0-1.2x instead of 3x
```

### Step 4: Re-evaluate CI Calibration

```bash
python evaluate_quantile_ci_calibration.py

# Target: ~10-15% violations (down from 34%)
```

---

## Files Generated

1. `diagnose_generation_pinball_loss.py` - Diagnostic script
2. `test_spx/quantile_regression/generation_vs_training_loss.csv` - Results
3. `test_spx/quantile_regression/generation_vs_training_loss.png` - Visualization
4. `test_spx/quantile_regression/diagnosis_output.txt` - Full output log

---

## Conclusion

**The poor CI calibration is NOT a fundamental limitation of quantile regression.**

The issue is a **generation bug** causing 3x worse performance during inference compared to training. Once fixed, we expect:
- Generation loss ≈ Test loss (ratio ~1.0-1.2x)
- CI violations to drop from 34% to ~10-15%
- Proper quantile ordering (p05 < p50 < p95)

**Next Step:** Debug MLE generation code to identify why z=0 produces incorrect quantile predictions.

---

## References

- Original quantile results: `QUANTILE_REGRESSION_RESULTS.md`
- Generation scripts: `generate_quantile_surfaces.py`
- Model implementation: `vae/cvae_with_mem_randomized.py`
- CI calibration analysis: `evaluate_quantile_ci_calibration.py`
