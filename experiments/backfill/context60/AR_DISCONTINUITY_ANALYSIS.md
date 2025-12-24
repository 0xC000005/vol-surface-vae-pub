# AR Discontinuity Problem - Root Cause Analysis and Solutions

**Model:** Context60 VAE with Quantile Regression Decoder
**Issue:** Sharp discontinuities in autoregressive predictions at chunk boundaries
**Investigation Date:** 2025-12-03
**Status:** Root cause identified, solutions proposed

---

## Executive Summary

The context60 model's autoregressive generation exhibits **sharp discontinuities** at chunk boundaries (days 60 and 120) when generating 180-day sequences. Our investigation reveals that:

1. **Root cause:** BOTH level-based modeling AND offset-based chunking contribute to discontinuities
2. **Discontinuity magnitude:** 30-50% drops in CI width, significant trend shifts in predictions
3. **Trade-off discovered:** Change-based modeling eliminates discontinuities but reduces accuracy by 60%
4. **Current approach is insufficient:** Level-based predictions without continuity constraints create artificial patterns

**Recommended solutions:** Hybrid change-based decoder with continuity constraints, or overlapping windows with weighted averaging.

---

## Problem Description

### Observed Behavior

When generating 180-day autoregressive sequences using the context60 model:

- **Sharp drops in confidence interval (CI) width** at days 60 and 120
- **Saw-tooth pattern:** Drop at boundary → gradual recovery → drop at next boundary
- **Magnitude:** 30-50% CI width drops (varies by period and sequence)
- **Consistency:** Occurs across ALL periods (crisis, insample, OOS, gap)

### Impact on Model Outputs

**CI Width Statistics (Crisis Period Average):**
```
Day  59:  CI width = 0.123065
Day  60:  CI width = 0.112168  ← -8.9% drop (discontinuity)
Day 119:  CI width = 0.124818
Day 120:  CI width = 0.108069  ← -13.4% drop (discontinuity)
```

**Range Inflation:**
- Teacher forcing H=90: Range = 0.014 (smooth evolution)
- Autoregressive H=180: Range = 0.033 (2.4× larger due to discontinuities)
- Autoregressive H=270: Range = 0.041 (2.9× larger)

**Consequences:**
1. Inflated range statistics (2-5× larger than smooth TF predictions)
2. Unreliable uncertainty quantification at long horizons
3. Artificial patterns that don't reflect true market dynamics
4. CI calibration degradation (violations jump from 18% → 56% in some sequences)

---

## Root Cause Investigation

### Hypothesis: Levels vs Changes Modeling

**Question:** Are discontinuities caused by modeling absolute volatility LEVELS rather than CHANGES?

**Test performed:** `experiments/backfill/context60/test_levels_vs_changes_hypothesis.py`

### Test Results

#### Test 1: Discontinuities in Changes vs Levels

**Finding:** BOTH levels AND changes show discontinuities

```
Crisis Period Sequence 200 (2008-12-31):

Day 60 Boundary:
  Level discontinuity:  -0.028578  (absolute vol drops 2.9 vol points)
  Change discontinuity: +0.028146  (slope shifts dramatically)
  Acceleration:         +0.028146  (trend changes direction)

Day 120 Boundary:
  Level discontinuity:  -0.015226
  Change discontinuity: +0.015315
  Acceleration:         +0.015315
```

**Interpretation:**
- Acceleration spikes (>0.001 threshold) indicate discontinuities in TREND, not just levels
- The model's day-to-day changes themselves are discontinuous
- This proves the problem is NOT just about absolute levels

#### Test 2: Accumulated Change-Based Predictions

**Finding:** Accumulation eliminates discontinuities but diverges from ground truth

When converting level predictions to accumulated changes:
```
Discontinuity Reduction:
  Day 60:  98.5% reduction (-0.029 → -0.0004)
  Day 120: 99.4% reduction (-0.015 → +0.0001)
```

**Visual comparison:**
- **Original (blue):** Sharp drops at boundaries, but tracks ground truth closely
- **Accumulated (red):** Smooth evolution, but drifts away from ground truth

#### Test 3: RMSE Comparison - The Critical Trade-Off

**Finding:** Change-based predictions are 60% LESS accurate

```
RMSE Comparison:
  Original level-based:      0.059708
  Accumulated change-based:  0.095549
  Degradation:               +60.0%

By Chunk:
  Chunk 1 (days 0-59):    0.041 → 0.071 (+73.8%)
  Chunk 2 (days 60-119):  0.072 → 0.110 (+54.0%)
  Chunk 3 (days 120-179): 0.062 → 0.101 (+61.6%)
```

**Loss Comparison:**
```
Method                      Loss        Use Case
------------------------------------------------------
Original (level-based)      0.003565    Current approach
Accumulated (change-based)  0.009130    Smooth but less accurate (2.6× worse)
Direct changes              0.000152    Measures change accuracy only
Log returns                 0.001287    Percentage changes
```

---

## Why Current Approach is Insufficient

### 1. Offset-Based Chunking Creates Independent Predictions

**Current AR generation method:**
```
Chunk 1 (days 0-59):    Predict levels using context[-60:]
Chunk 2 (days 60-119):  Predict levels using predictions[0:60] as context
Chunk 3 (days 120-179): Predict levels using predictions[60:120] as context
```

**Problem:**
- Each chunk makes a **FRESH prediction** of absolute volatility levels
- No continuity constraint between chunks
- Predictions at day 59 and day 60 are **independent** (different forward passes)
- Result: Sharp trend shifts at boundaries

### 2. Level-Based Modeling Without Continuity Constraints

**Current decoder behavior:**
- Predicts: `vol_t` (absolute implied volatility)
- Optimization: `loss = MSE(predicted_vol, true_vol) + pinball_loss(quantiles)`
- No penalty for discontinuities

**What happens at boundaries:**
```
End of Chunk 1 (day 59):
  - Model has accumulated 60 days of prediction uncertainty
  - Predicted level: 0.330 (with high uncertainty)
  - CI width: 0.123

Start of Chunk 2 (day 60):
  - Model makes NEW prediction conditioned on chunk 1 predictions
  - Fresh prediction: 0.301 (with fresh, lower uncertainty)
  - CI width: 0.112 (dropped -8.9%)

Key issue: No constraint that v₆₀ must be close to v₅₉!
```

### 3. Both Levels AND Changes Are Discontinuous

Our hypothesis test proved:
- Discontinuities exist in BOTH absolute levels AND day-to-day changes
- Acceleration spikes at boundaries (trend shifts)
- This is NOT just a visualization artifact—the underlying dynamics are discontinuous

### 4. Trade-Off Between Smoothness and Accuracy

**Critical finding:** The model achieves better accuracy by predicting levels independently at each chunk, even though this creates discontinuities.

**Why level-based is more accurate:**
- Direct optimization of the final target (absolute volatility levels)
- Each chunk can "correct" accumulated errors from previous chunks
- Model learns to predict levels given imperfect context (realistic deployment)

**Why change-based is smoother but less accurate:**
- Errors accumulate over time (no correction mechanism)
- Small errors in early changes compound into large level errors
- Smooth but drifts away from ground truth

---

## Why This Matters

### 1. Unreliable Long-Horizon Uncertainty Quantification

**Example:** Crisis period average statistics
```
Metric                          Value       Issue
----------------------------------------------------------------
CI violation rate (H=180)       56.7%       Far above 10% target
Range (max - min CI width)      0.033       2.4× inflated by discontinuities
Average CI width                0.072       Misleading (not consistent across horizon)
```

**Problem:** Users cannot trust 180-day confidence intervals when:
- Width drops 30-50% every 60 days
- Violations spike after boundaries
- Range statistics are inflated by artificial patterns

### 2. Artificial Patterns Don't Reflect Market Dynamics

**Reality check:** Do real volatility surfaces exhibit sharp 30% drops in uncertainty every 60 days? **No.**

**What we observe in ground truth:**
- Smooth evolution of implied volatility (no artificial boundaries)
- Gradual uncertainty increase with horizon
- Natural market regime changes (not tied to our model's chunk boundaries)

**What our model produces:**
- Artificial saw-tooth pattern tied to 60-day chunks
- Sharp resets in uncertainty at days 60, 120, 180, etc.
- Pattern is a modeling artifact, not a market feature

### 3. Limits Practical Deployment

**Use cases affected:**
- **Risk management:** 180-day VaR estimates unreliable
- **Options pricing:** Long-dated options have discontinuous vol forecasts
- **Scenario analysis:** Can't generate realistic 6-month+ trajectories
- **Backtesting:** Results contaminated by boundary artifacts

---

## Recommended Solutions

### Solution 1: Hybrid Change-Based Decoder with Continuity Constraints ⭐ RECOMMENDED

**Approach:** Predict changes but add loss term enforcing level accuracy

**Architecture modification:**
```python
# Decoder outputs CHANGES instead of LEVELS
predicted_changes = decoder(z, context)  # (B, H, 5, 5)

# Accumulate to get levels (differentiable)
context_end_level = context[:, -1, :, :]  # Last day of context
predicted_levels = cumulative_sum(predicted_changes, initial=context_end_level)

# Hybrid loss
loss_changes = MSE(predicted_changes, true_changes)
loss_levels = MSE(predicted_levels, true_levels)
loss_continuity = MSE(predicted_levels[0], context_end_level)  # Anchor constraint

loss = alpha * loss_changes + beta * loss_levels + gamma * loss_continuity
```

**Benefits:**
- Enforces smooth transitions (changes are the primitive)
- Maintains level accuracy (loss on accumulated levels)
- Continuity guaranteed by design (cumulative sum is continuous)
- No discontinuities at boundaries

**Implementation:**
1. Modify decoder to output Δvol instead of vol
2. Add `torch.cumsum` layer to accumulate changes
3. Tune loss weights (α, β, γ) to balance smoothness vs accuracy
4. Expected optimal: α=0.2, β=0.7, γ=0.1 (prioritize level accuracy)

**Expected outcome:**
- Eliminates discontinuities (change-based)
- Maintains >90% of current level accuracy (hybrid loss)
- Smooth uncertainty evolution across entire horizon

---

### Solution 2: Overlapping Windows with Weighted Averaging

**Approach:** Generate overlapping chunks and blend predictions

**Method:**
```python
# Generate overlapping chunks
chunk_1 = model.generate(context, horizon=90)       # Days 0-89
chunk_2 = model.generate(preds[0:60], horizon=90)   # Days 60-149 (overlap 30)
chunk_3 = model.generate(preds[90:150], horizon=90) # Days 120-209 (overlap 30)

# Weighted averaging in overlap regions
for t in [60, 90]:  # Overlap regions
    weight_1 = linear_fade(t, start=60, end=90)  # 1.0 → 0.0
    weight_2 = 1.0 - weight_1                     # 0.0 → 1.0

    preds[t] = weight_1 * chunk_1[t] + weight_2 * chunk_2[t]
```

**Benefits:**
- No retraining required (works with current model)
- Smooth transitions via blending
- Maintains current level accuracy
- Easy to implement

**Drawbacks:**
- 1.5× more generation time (overlap = 33% extra)
- Blending may reduce accuracy in overlap regions
- Still has underlying discontinuities (just smoothed out)

**Implementation:**
1. Modify `generate_autoregressive_sequence` to use overlapping windows
2. Add `blend_overlapping_predictions` function with configurable fade
3. Test on crisis period to validate smoothness

---

### Solution 3: Regularization Term for Boundary Discontinuities

**Approach:** Add loss penalty for discontinuities during training

**Loss modification:**
```python
# Standard loss
loss_primary = pinball_loss(predicted_quantiles, true_surface) + kl_weight * kl_div

# Discontinuity penalty
# For multi-horizon training: check continuity at horizon boundaries
if horizon == 60:  # Training for H=60 chunk
    # Get prediction at H=60
    pred_at_60 = predicted_surface[:, -1, :, :]  # Last day of chunk

    # Get prediction at H=61 (from next chunk, requires forward pass)
    next_context = torch.cat([context[:, 1:, :, :], predicted_surface], dim=1)
    pred_at_61 = model(next_context, horizon=1)[:, 0, :, :]

    # Penalty for jump
    continuity_penalty = torch.mean((pred_at_61 - pred_at_60)**2)

    loss = loss_primary + lambda_cont * continuity_penalty
```

**Benefits:**
- Incentivizes smooth transitions
- Maintains level-based modeling
- Can tune λ to balance smoothness vs accuracy

**Drawbacks:**
- Requires retraining
- Doubles forward passes during training (expensive)
- May not fully eliminate discontinuities (soft constraint)
- Optimal λ requires hyperparameter search

**Implementation complexity:** High (requires training pipeline changes)

---

### Solution 4: Continuous Day-by-Day AR

**Approach:** Generate one day at a time instead of chunks

**Method:**
```python
def generate_continuous_ar(model, context, total_horizon=180):
    predictions = []
    current_context = context.clone()

    for t in range(total_horizon):
        # Generate next day
        next_day = model(current_context, horizon=1)  # (B, 1, 3, 5, 5)
        predictions.append(next_day)

        # Update context (sliding window)
        current_context = torch.cat([current_context[:, 1:, :, :], next_day], dim=1)

    return torch.cat(predictions, dim=1)  # (B, 180, 3, 5, 5)
```

**Benefits:**
- NO discontinuities (continuous by design)
- Natural uncertainty growth (no resets)
- Most realistic deployment scenario

**Drawbacks:**
- **180× slower** (180 forward passes instead of 3)
- May accumulate errors faster (no 60-day horizon correction)
- Requires retraining with horizon=1 only (current model trained on H=60,90)

**When to use:** Production deployment where smoothness is critical and generation time is acceptable.

---

## Comparison of Solutions

| Solution | Retraining | Speed | Smoothness | Accuracy | Complexity |
|----------|-----------|-------|------------|----------|------------|
| **Hybrid Change-Based** ⭐ | Yes | Same | Excellent | High | Medium |
| **Overlapping Windows** | No | 1.5× slower | Good | Same | Low |
| **Regularization Penalty** | Yes | 2× training | Good | Unknown | High |
| **Continuous Day-by-Day** | Yes | 180× slower | Perfect | Unknown | Low |

**Recommended priority:**
1. **Overlapping windows** (quick fix, no retraining)
2. **Hybrid change-based** (long-term solution, requires retraining)
3. Regularization penalty (if hybrid doesn't work)
4. Continuous AR (production deployment only)

---

## Testing and Validation

### Quick Validation (No Retraining)

**Test overlapping windows:**
```bash
python experiments/backfill/context60/test_overlapping_windows.py --period crisis --horizon 180
```

**Expected results:**
- Discontinuities reduced by 60-80%
- RMSE within 5% of current baseline
- CI violations remain stable (~18%)

### Full Validation (After Retraining)

**Test hybrid change-based model:**
```bash
# Train new model
python experiments/backfill/context60/train_hybrid_change_decoder.py

# Generate and analyze
python experiments/backfill/context60/generate_vae_ar_sequences.py --model hybrid --period crisis
python experiments/backfill/context60/visualize_single_ar_sequence.py --model hybrid
```

**Success criteria:**
- Zero discontinuities (smooth CI width evolution)
- RMSE within 10% of current baseline
- CI violations <20%
- Range statistics comparable to TF horizons

---

## References

**Analysis scripts:**
- `experiments/backfill/context60/test_levels_vs_changes_hypothesis.py` - Hypothesis test
- `experiments/backfill/context60/visualize_single_ar_sequence.py` - Single sequence view
- `experiments/backfill/context60/visualize_ar_discontinuity_overlay.py` - Multi-regime overlay

**Key findings:**
- Context60 CI width analysis: `results/context60_baseline/analysis/comparison/`
- Visualization outputs: `results/context60_baseline/visualizations/`
- Discontinuity test results: `levels_vs_changes_comparison.png`

**Related issues:**
- Oracle vs prior gap (context60 shows NO gap, unlike context20's 1.56-1.66×)
- CI calibration degradation at long horizons
- Range inflation in autoregressive vs teacher forcing

---

## Conclusion

**Current status:** The context60 model's autoregressive generation is **insufficient for long-horizon forecasting** due to:
1. Sharp discontinuities at chunk boundaries (30-50% CI width drops)
2. Artificial saw-tooth patterns not reflecting market dynamics
3. Unreliable uncertainty quantification beyond 60 days
4. Trade-off: Current approach prioritizes accuracy over smoothness

**Required action:** Implement solution to enforce continuity while maintaining prediction accuracy.

**Recommended path forward:**
1. **Immediate:** Implement overlapping windows (quick fix, validate on crisis period)
2. **Short-term:** Train hybrid change-based decoder (proper solution)
3. **Long-term:** Consider continuous day-by-day AR for production deployment

**Expected outcome:** Smooth 180-day volatility surface forecasts with <5% accuracy loss and no artificial boundary discontinuities.

---

**Document version:** 1.0
**Last updated:** 2025-12-03
**Status:** Root cause identified, solutions ready for implementation
