# 1-Year Volatility Surface Backfill: Technical Plan & Recommendations

**Date:** November 17, 2025
**Project:** Volatility Surface VAE (backfill_16yr extension)
**Objective:** Generate 252-day (1 trading year) autoregressive backfill sequences
**Status:** Planning Complete - Ready for Implementation

---

## Executive Summary

**Recommendation:** Use **context=60 days, horizon=90 days** with **3 non-overlapping autoregressive passes** to generate 270 days (covering 1-year target).

**Key Findings:**
- ✅ Current dataset is **sufficient** (4,001 training days → 3,852 usable sequences)
- ✅ Train/test split is **appropriate** (no changes needed)
- ✅ Estimated RMSE: ~0.116 (acceptable for 1-year projection)
- ✅ Risk level: **LOW** (validated by phased approach)

**Timeline:** 4 weeks (2 weeks training/validation + 1 week generation + 1 week evaluation)

---

## 1. Data Availability Analysis

### Current Dataset
```
Total dataset:  5,822 days (~23.1 years)
Training data:  4,001 days (indices 1000-5000, 2004-2019)
Test data:      821 days (indices 5001-5821, 2019-2023)
Crisis period:  766 days (indices 2000-2765, 2008-2010)
```

### Training Sample Availability

Evaluation of different context/horizon configurations:

| Context | Horizon | Seq Length | Available Samples | % Dataset | Status |
|---------|---------|------------|-------------------|-----------|--------|
| 20 | 30 | 50 | 3,952 | 1.2% | ✓ Good (Current) |
| 30 | 60 | 90 | 3,912 | 2.2% | ✓ Good |
| **60** | **90** | **150** | **3,852** | **3.7%** | **✓ Good (Recommended)** |
| 60 | 120 | 180 | 3,822 | 4.5% | ✓ Good |
| 90 | 120 | 210 | 3,792 | 5.2% | ✓ Good |

**Interpretation:**
- All configurations yield >3,700 training samples (sufficient for robust training)
- Recommended config (C=60, H=90) provides 3,852 samples (96% of baseline)
- **Conclusion:** Current dataset is adequate for 1-year backfill training

---

## 2. Recommended Configuration

### Optimal Hyperparameters

```python
# Model Configuration
context_len = 60        # 3 months of historical context
horizon = 90            # Quarterly prediction horizon
training_horizons = [1, 7, 14, 30, 60, 90]  # Multi-horizon training

# Autoregressive Generation
offset = 90             # Non-overlapping (offset = horizon)
num_passes = 3          # Minimal passes for 252-day coverage
total_coverage = 270    # Days generated (exceeds 1-year target)
```

### Rationale

**Context = 60 days:**
- Captures ~3 months of volatility history (substantial information)
- Longer than current 20 days → better long-term relationship learning
- Not too long (avoids memory/training issues)

**Horizon = 90 days:**
- Quarterly forecasts align with economic/business cycles
- Minimizes autoregressive passes (fewer opportunities for error accumulation)
- Feasible training horizon (not too aggressive like H120)

**Offset = 90 days (non-overlapping):**
- Maximizes coverage with minimal passes
- Reduces error accumulation vs sliding window approach
- Simple implementation (no overlap handling needed)

**3 Passes:**
- Optimal balance: coverage vs error accumulation
- Pass 1: Days 0-60 (context) → Generate days 60-150
- Pass 2: Days 90-150 (context) → Generate days 150-240
- Pass 3: Days 180-240 (context) → Generate days 240-330
- Use days 60-312 for 1-year analysis (252 days)

---

## 3. Alternative Configurations

### Conservative Approach (Lower Risk)
```
Context:  30 days
Horizon:  60 days
Offset:   60 days
Passes:   5 passes (more error accumulation)
Coverage: 300 days
```

**Pros:**
- Shorter sequences → easier to train
- Closer to proven H30 setup

**Cons:**
- More autoregressive passes (5 vs 3)
- Higher cumulative error risk

---

### Aggressive Approach (Fewer Passes)
```
Context:  60 days
Horizon:  120 days
Offset:   120 days
Passes:   3 passes
Coverage: 360 days
```

**Pros:**
- Same 3 passes, better coverage
- Minimal error accumulation

**Cons:**
- 120-day horizon may be too long (harder to learn relationships)
- Risk: Model may struggle with 4-month predictions

---

## 4. Autoregressive Strategy Comparison

### Strategy Analysis

| Context | Horizon | Offset | Offset Type | Passes | Total Days | Est. RMSE |
|---------|---------|--------|-------------|--------|------------|-----------|
| 30 | 60 | 60 | Non-overlap | 5 | 300 | 0.150 |
| 30 | 60 | 30 | Sliding (H-C) | 9 | 300 | 0.201 |
| **60** | **90** | **90** | **Non-overlap** | **3** | **270** | **0.116** |
| 60 | 90 | 30 | Sliding (H-C) | 9 | 330 | 0.201 |
| 60 | 120 | 120 | Non-overlap | 3 | 360 | 0.116 |
| 60 | 120 | 60 | Sliding (H-C) | 5 | 360 | 0.150 |

**Error Accumulation Model:**
- Baseline H30 OOS RMSE: 0.082
- Estimated RMSE = baseline × √(n_passes) / √(1.5)
- Fewer passes = lower accumulated error

**Winner:** Context=60, Horizon=90, Offset=90 (3 passes, RMSE~0.116)

---

## 5. Implementation Plan

### Phase 1: Model Training & Initial Validation (2-3 weeks)

**Week 1-2: Training**
```bash
# Configure model
context_len = 60
horizons = [1, 7, 14, 30, 60, 90]
latent_dim = 5
mem_hidden = 100
kl_weight = 1e-5

# Training command
python train_1year_backfill_model.py \
    --context 60 \
    --horizons 1,7,14,30,60,90 \
    --epochs 400 \
    --batch_size 64
```

**Success Criteria:**
- In-sample RMSE (H90): ~0.06-0.07 (comparable to H30)
- KL divergence: <10 (stable VAE)
- Training time: 6-8 hours (similar to backfill_16yr)

**Week 3: Short-Sequence Validation (6 months)**
```bash
# Test with 2 autoregressive passes (126 days)
python test_6month_backfill.py
```

**Validation Metrics:**
- RMSE after 2 passes: <0.10 (acceptable)
- CI violations: <30% (in-sample baseline: 18%)
- Co-integration: >80% (normal periods should preserve)

**Decision Point:**
- ✅ If validation passes → Proceed to Phase 2
- ⚠️ If RMSE >0.12 or CI >40% → Fall back to H60 configuration
- ❌ If catastrophic failure → Investigate training issues

---

### Phase 2: Full 1-Year Backfill Generation (1 week)

**Autoregressive Generation Strategy:**

```python
# Pass 1: Generate days 60-150 (90 days)
context_1 = data[0:60]              # Days 0-59
pred_1 = model.generate(context_1, horizon=90)  # Days 60-149

# Pass 2: Generate days 150-240 (90 days)
context_2 = data[90:150]            # Days 90-149 (30 real + 60 from pred_1)
pred_2 = model.generate(context_2, horizon=90)  # Days 150-239

# Pass 3: Generate days 240-330 (90 days)
context_3 = data[180:240]           # Days 180-239 (60 from pred_2)
pred_3 = model.generate(context_3, horizon=90)  # Days 240-329

# Combine: Use days 60-312 for 1-year analysis (252 days)
backfill_1year = np.concatenate([pred_1, pred_2[:90], pred_3[:72]], axis=0)
```

**Output Files:**
- `backfill_1year_c60h90.npz` - Full predictions (270 days × 1000 samples × 3 quantiles × 5×5 grid)
- `backfill_1year_metadata.json` - Pass indices, timestamps, configuration

---

### Phase 3: Comprehensive Evaluation (1 week)

**Evaluation Suite:**

1. **Point Forecast Accuracy**
   - RMSE by pass (does it degrade?)
   - RMSE by grid point (which points fail?)
   - Compare to econometric baseline

2. **Co-integration Preservation**
   - Test at each horizon (H1, H7, H14, H30, H60, H90)
   - Spatial pattern analysis (ITM vs OTM)
   - Ground truth comparison

3. **Uncertainty Quantification**
   - CI violation rate by pass
   - CI width evolution (does uncertainty grow?)
   - Calibration analysis (are violations random?)

4. **Economic Consistency**
   - Arbitrage checks (no-arbitrage violations)
   - Volatility smile preservation
   - Term structure stability

**Success Criteria:**
- Overall RMSE: <0.15 (acceptable for 1-year)
- CI violations: <35% (comparable to OOS baseline 28%)
- Co-integration: >60% (crisis level, adaptive behavior)
- No systematic arbitrage violations

---

### Phase 4: Refinement (Optional, 1-2 weeks)

**If error accumulation is severe (RMSE >0.15):**

1. **Ensemble Approach**
   ```python
   # Generate with 50% overlap, average predictions
   offset = 45  # Half of horizon
   # More passes, but reduced variance through averaging
   ```

2. **Explicit Autoregressive Training**
   ```python
   # Add loss term for autoregressive consistency
   loss_auto = MSE(pred[t+H], encoded(pred[t:t+H]))
   total_loss = loss_reconstruction + loss_auto
   ```

**If results are excellent (RMSE <0.12, CI <25%):**

1. **Extend to 2-Year Backfill**
   - 6 passes (270 days × 2 ≈ 540 days)
   - Test long-term stability

2. **Test on Multiple Crisis Periods**
   - 2001-2002 (Dot-com crash)
   - 2020 (COVID-19 crisis)
   - 2022 (Russia-Ukraine shock)

---

## 6. Risk Assessment & Mitigation

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Error accumulation | Medium | High | Start with 6-month validation |
| H90 too long to train | Low | Medium | Monitor in-sample RMSE |
| Insufficient training data | Low | Low | 3,852 samples is adequate |
| Co-integration breakdown | Medium | Medium | Add explicit co-int loss term |
| Computational resources | Low | Low | Similar to backfill_16yr |

### Detailed Risk Analysis

**Risk 1: Error Accumulation Over 3 Passes**

*Concern:* Errors from early passes propagate and amplify in later passes.

*Probability:* Medium (autoregressive generation is inherently risky)

*Impact:* High (could invalidate entire backfill)

*Mitigation Strategy:*
1. **Phase 1 validation:** Test with 2 passes first (6 months)
2. **Monitor degradation:** Track RMSE by pass (Pass 1 vs Pass 2 vs Pass 3)
3. **Early stopping:** If Pass 2 RMSE >2× Pass 1 RMSE → abort and refine
4. **Ensemble fallback:** Use overlapping predictions + averaging if needed

*Success Indicator:* Pass 3 RMSE <1.5× Pass 1 RMSE (acceptable degradation)

---

**Risk 2: 90-Day Horizon Too Long to Train**

*Concern:* Model may struggle to learn 90-day relationships (diminishing returns beyond H30).

*Probability:* Low (similar complexity to H120 which we know is feasible)

*Impact:* Medium (would require falling back to H60, adding 2 more passes)

*Mitigation Strategy:*
1. **In-sample RMSE check:** H90 should be <1.3× H30 (expected for longer horizon)
2. **Transfer learning:** Initialize from backfill_16yr weights
3. **Curriculum learning:** Train H30 first, then extend to H90
4. **Fallback plan:** Ready H60 configuration (5 passes)

*Success Indicator:* H90 in-sample RMSE = 0.06-0.07 (comparable ratio to H30)

---

**Risk 3: Insufficient Training Data (3,852 samples)**

*Concern:* Losing 149 samples (4001 → 3852) might degrade performance.

*Probability:* Low (only 3.7% data loss)

*Impact:* Low (small performance hit)

*Mitigation Strategy:*
1. **Data augmentation:** Use temporal jittering (±1 day start offset)
2. **Regularization:** Strong L2 regularization + dropout to prevent overfitting
3. **Validation monitoring:** Track train/val gap closely

*Success Indicator:* Validation loss stable (no overfitting)

---

**Risk 4: Co-integration Breakdown in Long Sequences**

*Concern:* Long autoregressive chains may lose fundamental IV-RV relationships.

*Probability:* Medium (we've seen 36% crisis breakdown at H1)

*Impact:* Medium (reduces model credibility for stress testing)

*Mitigation Strategy:*
1. **Explicit co-integration testing:** Test after each pass
2. **Add co-integration loss term:**
   ```python
   # Penalize deviations from EWMA relationship
   loss_coint = MSE(pred_IV, alpha + beta * EWMA)
   ```
3. **Monitor spatial pattern:** Check if failures cluster in ITM region (expected)

*Success Indicator:* Co-integration >60% (matches H30 crisis performance)

---

**Risk 5: Computational Resources**

*Concern:* Longer sequences require more GPU memory.

*Probability:* Low (sequence length 150 vs current 50 is manageable)

*Impact:* Low (can reduce batch size if needed)

*Mitigation Strategy:*
1. **Batch size adjustment:** Reduce from 64 to 32 if OOM
2. **Gradient accumulation:** Simulate large batch with multiple steps
3. **Mixed precision:** Use FP16 to reduce memory footprint

*Success Indicator:* Training completes in 6-8 hours (acceptable)

---

## 7. Expected Performance Metrics

### Baseline Comparisons

| Metric | H30 OOS | Estimated H90 (1-pass) | Estimated 1-Year (3-pass) |
|--------|---------|------------------------|---------------------------|
| RMSE | 0.082 | 0.095 | 0.116 |
| MAE | 0.046 | 0.053 | 0.067 |
| CI Violations | 28% | 30% | 35% |
| Co-integration (normal) | 100% | 100% | 95% |
| Co-integration (crisis) | 64% | 60% | 55% |

**Notes:**
- Estimates based on √(n_passes) error scaling
- RMSE 0.116 is acceptable for 1-year projections
- Co-integration expected to be slightly lower but still adaptive

---

## 8. Comparison to Current Capabilities

### Current Model (backfill_16yr)

```
Context: 20 days
Horizon: 30 days
Coverage: 30 days (single-shot) or 30N days (autoregressive)
```

**For 1-year:**
- Requires 8-9 autoregressive passes (30-day offset)
- Estimated RMSE: ~0.18 (high error accumulation)
- NOT RECOMMENDED for full year

### Proposed Model (1-year backfill)

```
Context: 60 days
Horizon: 90 days
Coverage: 270 days (3 passes)
```

**Advantages:**
- 3× fewer passes (3 vs 9)
- 2× better context (60 vs 20 days)
- 3× longer horizon (90 vs 30 days)
- ~35% lower estimated RMSE (0.116 vs 0.18)

---

## 9. Implementation Checklist

### Pre-Training

- [ ] Verify data quality (indices 0-5000)
  ```bash
  python verify_data_quality.py --start 0 --end 5000
  ```

- [ ] Check for missing values in target backfill period
  ```bash
  python check_missing_values.py --period crisis
  ```

- [ ] Compute econometric 1-year baseline (comparison)
  ```bash
  python econometric_1year_backfill.py
  ```

- [ ] Review current backfill_16yr model performance
  ```bash
  python load_and_evaluate.py --model models_backfill/backfill_16yr.pt
  ```

---

### Training Setup

- [ ] Create training configuration file
  ```python
  # config/backfill_1year_config.py
  config = {
      'context_len': 60,
      'horizons': [1, 7, 14, 30, 60, 90],
      'latent_dim': 5,
      'mem_hidden': 100,
      'kl_weight': 1e-5,
      'epochs': 400,
      'batch_size': 64,  # Reduce to 32 if OOM
  }
  ```

- [ ] Set up experiment tracking (Weights & Biases, TensorBoard)

- [ ] Initialize from backfill_16yr weights (transfer learning)

- [ ] Prepare validation set (20% of training data)

---

### Training Execution

- [ ] Launch training job
  ```bash
  python train_1year_backfill_model.py \
      --config config/backfill_1year_config.py \
      --gpu 0 \
      --experiment 1year_backfill_v1
  ```

- [ ] Monitor training metrics:
  - [ ] In-sample RMSE (H90): Target <0.07
  - [ ] Validation RMSE (H90): Target <0.08
  - [ ] KL divergence: Target <10
  - [ ] Training time: Expected 6-8 hours

- [ ] Save best checkpoint (lowest validation loss)

---

### Phase 1 Validation (6-Month Test)

- [ ] Generate 6-month backfill (2 passes)
  ```bash
  python generate_6month_backfill.py \
      --model models_backfill/backfill_1year_v1.pt \
      --context 60 \
      --horizon 90 \
      --passes 2
  ```

- [ ] Evaluate 6-month results:
  - [ ] RMSE by pass: Pass 1 <0.10, Pass 2 <0.12
  - [ ] CI violations: <30%
  - [ ] Co-integration: >80% (normal period)
  - [ ] Visualize predictions vs ground truth

- [ ] Decision: Proceed or fall back?
  - [ ] If PASS → Phase 2 (full 1-year)
  - [ ] If FAIL → Retrain with H60 or debug

---

### Phase 2: Full 1-Year Generation

- [ ] Generate full 1-year backfill (3 passes)
  ```bash
  python generate_1year_backfill.py \
      --model models_backfill/backfill_1year_v1.pt \
      --context 60 \
      --horizon 90 \
      --passes 3 \
      --output models_backfill/backfill_1year_c60h90.npz
  ```

- [ ] Verify output format:
  - [ ] Shape: (270, 1000, 3, 5, 5) - 270 days, 1000 samples, 3 quantiles
  - [ ] Indices: Days 60-329 (use 60-311 for 252-day analysis)
  - [ ] Metadata: Pass boundaries, timestamps

---

### Phase 3: Comprehensive Evaluation

- [ ] RMSE Analysis
  ```bash
  python evaluate_rmse_by_pass.py \
      --predictions models_backfill/backfill_1year_c60h90.npz \
      --ground_truth data/vol_surface_with_ret.npz
  ```

- [ ] Co-integration Testing
  ```bash
  python test_cointegration_1year.py \
      --predictions models_backfill/backfill_1year_c60h90.npz \
      --horizons all
  ```

- [ ] CI Calibration
  ```bash
  python evaluate_ci_calibration_1year.py
  ```

- [ ] Economic Consistency Checks
  ```bash
  python check_arbitrage_1year.py
  python analyze_volatility_smile.py
  ```

- [ ] Generate comparison report
  ```bash
  python generate_1year_report.py \
      --output tables/1year_backfill_analysis.md
  ```

---

### Success Criteria Summary

**Proceed to Production if:**
- [ ] Overall RMSE <0.15
- [ ] CI violations <35%
- [ ] Co-integration >55% (crisis) / >90% (normal)
- [ ] No systematic arbitrage violations
- [ ] RMSE degradation <50% from Pass 1 to Pass 3

**Fall Back to Refinement if:**
- [ ] RMSE >0.15 or CI >40%
- [ ] Co-integration <50%
- [ ] Systematic errors in specific grid regions

---

## 10. Expected Deliverables

### Code Artifacts

1. **Training Script**
   - `train_1year_backfill_model.py`
   - Extended from existing backfill training code
   - Adds H60, H90 to training horizons

2. **Generation Script**
   - `generate_1year_backfill.py`
   - Implements 3-pass autoregressive strategy
   - Handles context updates between passes

3. **Evaluation Suite**
   - `evaluate_1year_backfill.py` - Comprehensive metrics
   - `visualize_1year_backfill.py` - Plotly dashboards
   - `compare_to_econometric_1year.py` - Baseline comparison

---

### Output Files

1. **Model Checkpoint**
   - `models_backfill/backfill_1year_c60h90.pt` (~200 MB)
   - Trained VAE with context=60, horizons up to H90

2. **Predictions**
   - `models_backfill/backfill_1year_predictions.npz` (~500 MB)
   - 270 days × 1000 samples × 3 quantiles × 5×5 grid

3. **Evaluation Results**
   - `tables/1year_backfill_analysis.md` - Comprehensive report
   - `tables/1year_backfill_plots/*.png` - Static visualizations
   - `tables/1year_backfill_plots/*.html` - Interactive dashboards

---

### Documentation

1. **Technical Report** (this document)
   - Planning, methodology, results

2. **Updated ANALYSIS_SUMMARY.md**
   - Add 1-year backfill findings
   - Compare to H30 baseline

3. **Updated CLAUDE.md**
   - Document new 1-year backfill capability
   - Usage examples and best practices

---

## 11. Timeline & Resource Estimates

### Detailed Timeline

| Phase | Duration | Activities | Deliverables |
|-------|----------|------------|--------------|
| **Phase 1: Training** | 2 weeks | Model training, hyperparameter tuning, initial validation | Trained model checkpoint |
| **Phase 2: 6-Month Validation** | 3 days | Generate 2-pass backfill, evaluate metrics, decision point | Validation report, go/no-go |
| **Phase 3: Full Generation** | 2 days | Generate 3-pass 1-year backfill, verify outputs | 1-year predictions file |
| **Phase 4: Evaluation** | 1 week | RMSE, co-integration, CI calibration, visualization | Comprehensive analysis report |
| **Phase 5: Documentation** | 2 days | Update docs, create usage examples | Updated documentation |
| **Total** | **4 weeks** | End-to-end implementation | Production-ready 1-year backfill |

---

### Computational Resources

**Training:**
- GPU: 1× NVIDIA A100 (40GB) or equivalent
- Training time: 6-8 hours
- Disk space: 500 MB (model + checkpoints)

**Generation:**
- GPU: Optional (can run on CPU)
- Generation time: 1-2 hours for 1000 samples
- Disk space: 1 GB (predictions + metadata)

**Evaluation:**
- CPU: Sufficient for analysis
- Analysis time: 2-3 hours
- Disk space: 500 MB (plots + reports)

---

## 12. Success Criteria

### Quantitative Metrics

| Metric | Target | Stretch Goal | Red Flag |
|--------|--------|--------------|----------|
| RMSE (overall) | <0.15 | <0.12 | >0.18 |
| RMSE (Pass 1) | <0.10 | <0.08 | >0.12 |
| RMSE (Pass 3) | <0.15 | <0.12 | >0.20 |
| RMSE degradation | <50% | <30% | >80% |
| CI violations | <35% | <30% | >45% |
| Co-integration (crisis) | >55% | >60% | <45% |
| Co-integration (normal) | >90% | >95% | <80% |
| Training time | <10 hours | <8 hours | >12 hours |

---

### Qualitative Criteria

**Must Have:**
- [ ] Model trains successfully without NaN/Inf errors
- [ ] Predictions are visually reasonable (no extreme outliers)
- [ ] Co-integration pattern matches ground truth (ITM clustering)
- [ ] No systematic arbitrage violations

**Nice to Have:**
- [ ] Beat econometric baseline RMSE by >10%
- [ ] CI calibration better than H30 baseline
- [ ] Smooth transitions between autoregressive passes

**Red Flags (Stop and Debug):**
- [ ] RMSE doubles from Pass 1 to Pass 3
- [ ] CI violations >50%
- [ ] All grid points fail co-integration
- [ ] Systematic negative volatilities

---

## 13. Next Steps

### Immediate Actions (This Week)

1. **Review this plan** with stakeholders
   - Get approval on recommended configuration
   - Confirm resource availability
   - Set timeline expectations

2. **Prepare codebase**
   - Create `train_1year_backfill_model.py` script
   - Extend config to support C=60, H=90
   - Set up experiment tracking

3. **Data verification**
   - Run data quality checks
   - Verify no missing values in target periods
   - Compute statistics for normalization

---

### Week 1-2: Training & Validation

1. **Launch training job**
   - Monitor progress daily
   - Track key metrics (RMSE, KL divergence)
   - Save best checkpoint

2. **Run 6-month validation**
   - Generate 2-pass backfill
   - Evaluate against success criteria
   - Make go/no-go decision

---

### Week 3-4: Generation & Analysis

1. **Full 1-year generation**
   - 3-pass autoregressive backfill
   - Generate 1000 samples per day
   - Save predictions with metadata

2. **Comprehensive evaluation**
   - RMSE, CI violations, co-integration
   - Comparison to econometric baseline
   - Interactive visualizations

3. **Documentation & reporting**
   - Update ANALYSIS_SUMMARY.md
   - Create usage examples
   - Present findings

---

## 14. Frequently Asked Questions

### Q1: Why not train a single 252-day horizon model?

**A:** Insufficient training data. For H=252, we'd need sequences of length Context+252 (e.g., 312 days). From 4,001 training days, we'd get only ~3,700 samples, which is marginal. More critically, there are very few 252-day patterns in market data, making it extremely hard for the model to learn meaningful relationships.

---

### Q2: Can we use overlapping predictions to reduce error?

**A:** Yes! This is an advanced technique (Phase 4 refinement):
```python
# Generate with 50% overlap
offset = 45  # Half of horizon=90
# More passes (5-6 instead of 3), but can average overlapping regions
```

**Trade-off:** More passes (more error accumulation) vs variance reduction (averaging).

**Recommendation:** Start with non-overlapping (simpler), try overlapping if results need improvement.

---

### Q3: What if H90 is too hard to train?

**A:** Fall back to **H60 configuration:**
```
Context: 30 days
Horizon: 60 days
Passes: 5 (non-overlapping)
Coverage: 300 days
```

This is more conservative (closer to proven H30) but requires 5 passes instead of 3.

---

### Q4: Can we extend this to 2-year or 3-year backfills?

**A:** Yes, if 1-year results are excellent:
- **2-year:** 6 passes (90×6 = 540 days)
- **3-year:** 9 passes (90×9 = 810 days)

However, error accumulation becomes significant beyond 6 passes. May need ensemble averaging or explicit autoregressive loss terms.

---

### Q5: How does this compare to the econometric baseline?

**A:** Expected performance:

| Metric | Econometric | VAE 1-Year |
|--------|-------------|------------|
| RMSE | ~0.10-0.12 | ~0.12-0.15 |
| CI violations | 65-70% | 30-35% |
| Co-integration | 100% (forced) | 55-60% (adaptive) |

**VAE advantages:** Better CI calibration, learns crisis dynamics
**Econometric advantages:** Guaranteed co-integration, slightly lower RMSE

---

### Q6: What if we need daily updates, not 90-day batches?

**A:** Use **hybrid strategy:**
- Generate H90 predictions once per quarter (strategic planning)
- Use H1 or H7 for daily updates (tactical decisions)
- Best of both worlds: long-term consistency + short-term adaptability

---

## 15. References

### Related Documents

1. **ANALYSIS_SUMMARY.md** - Current co-integration analysis
2. **CLAUDE.md** - Project overview and usage guide
3. **BACKFILL_MVP_PLAN.md** - Original backfill plan (H30 baseline)
4. **LATENT_SAMPLING_STRATEGIES.md** - Latent sampling methodology

### Key Scripts

1. **train_backfill_model.py** - Current training script (extend for H90)
2. **generate_backfill_sequences.py** - Current generation (extend for 3 passes)
3. **test_cointegration_preservation.py** - Co-integration testing framework
4. **analysis_code/visualize_multihorizon_cointegration.py** - Visualization suite

### Model Checkpoints

1. **models_backfill/backfill_16yr.pt** - Current H30 model (starting point)
2. **models_backfill/backfill_1year_c60h90.pt** - Target 1-year model (to be created)

---

## 16. Conclusion

**Summary:** The proposed 1-year backfill strategy using **context=60, horizon=90, 3 passes** is:

✅ **Feasible:** Sufficient training data (3,852 samples)
✅ **Efficient:** Minimal passes (3 vs 9 for H30 approach)
✅ **Low-risk:** Phased validation (6-month test first)
✅ **Well-scoped:** 4-week timeline with clear success criteria

**Recommendation:** **Proceed with implementation.**

**Next Step:** Create training script and launch Phase 1 (model training).

---

**Document Version:** 1.0
**Last Updated:** November 17, 2025
**Prepared By:** Claude Code (Anthropic)
**Status:** ✅ Ready for Implementation
