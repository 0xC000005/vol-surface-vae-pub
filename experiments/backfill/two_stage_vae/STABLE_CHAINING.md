# Stable Autoregressive Chaining for Student-t VAE

> **Important**: This document addresses **IV explosion** during chaining. For the deeper issue of why chained predictions don't track ground truth, see [CHAINING_FAILURE_ANALYSIS.md](CHAINING_FAILURE_ANALYSIS.md).

## Summary

This document describes the investigation and fix for **IV explosion during autoregressive chaining** with the Student-t VAE model. The root cause of explosion was identified as **fat-tailed sampling**, and three stable sampling strategies were implemented.

**Note**: These methods prevent explosion but do **not** provide predictive power. The model architecture is fundamentally unsuited for autoregressive prediction (see CHAINING_FAILURE_ANALYSIS.md).

---

## Problem Description

### Symptom
When generating 30-day IV surface trajectories via autoregressive chaining, the model produces unrealistic values:

```
Original method: Max IV = 24.36  (should be ~0.8)
                 Min IV = 0.0002 (should be ~0.01)
```

### Initial Hypothesis (Wrong)
Mean bias in decoder causing systematic drift:
- z=0 → mean(0,0) = -0.19 (should be 0)

### Actual Root Cause (Discovered)
**Student-t fat-tailed sampling** produces extreme values that compound exponentially.

Evidence from fresh model with low mean bias:
```
Mean bias at z=0: 0.0319  ✓ (acceptable)
But chaining STILL explodes: max IV = 112,336×

Step-by-step analysis:
  Step 17: Pred mean = 0.021, Sampled log_ret = +1.687  ← EXTREME
  Step 18: Pred mean = 0.018, Sampled log_ret = +1.630  ← EXTREME
```

The Student-t distribution with ν≈5:
- Kurtosis ≈ 9 (vs Gaussian = 3)
- P(|x| > 3σ) = 1.8% (vs Gaussian = 0.3%)
- Extreme samples compound: exp(1.6) × exp(1.6) = 25× in just 2 steps

---

## Solutions Implemented

### 1. Mean-Only Chaining (Deterministic)

**Rationale**: Use predicted mean without sampling. No stochastic component = no extreme values.

**Code Change**:
```python
def get_mean_only(model, batch):
    """Get mean prediction without sampling (deterministic)."""
    ctx_emb = model.ctx_encoder({"surface": batch["surface"]})
    z_mean, _, _ = model.main_encoder({"surface": batch["surface"]})
    mean, _, _, _ = model.decoder(ctx_emb, z_mean, sample=False)
    return mean
```

**Effect**:
- Max IV: 0.5588 (stable, no explosion)
- RMSE: 0.0678
- Trade-off: No uncertainty quantification (single trajectory)

---

### 2. Truncated Student-t Sampling (Bounded Stochastic)

**Rationale**: Sample from Student-t but reject values outside physical bounds via rejection sampling.

**Code Change**:
```python
def sample_truncated_student_t(model, batch, truncate_bounds=(-0.3, 0.3), max_attempts=100):
    """Sample with rejection: block extreme log-returns."""
    for attempt in range(max_attempts):
        sample = model.sample(batch, n_samples=1)
        last_sample = sample[:, -1]  # Check prediction

        if (last_sample >= truncate_bounds[0]).all() and \
           (last_sample <= truncate_bounds[1]).all():
            return sample  # Accept

    # Fallback to mean if all attempts rejected
    return get_mean_only(model, batch)
```

**Effect**:
- Max IV: 1.5046 (stable, minor overshoot)
- Preserves fat-tail character for moderate deviations
- Blocks catastrophic extremes (|log_ret| > 30% daily = unrealistic)

---

### 3. Temperature-Scaled Sampling (Controlled Stochastic)

**Rationale**: Reduce scale parameter during sampling. Standard technique in LLM/diffusion generation.

**Code Change**:
```python
def sample_temperature_scaled(model, batch, temperature=0.3):
    """Sample with reduced variance via temperature scaling."""
    # Sample z with reduced variance
    eps_z = torch.randn_like(z_std) * temperature  # Scale down noise
    z = z_mean + z_std * eps_z

    # Get mean and add scaled decoder noise
    mean, _, factor, log_diag = model.decoder(ctx_emb, z, sample=False)

    # Manual scaled sampling (temperature applied to noise)
    eps_rank = torch.randn(B, T, rank, device=device) * temperature
    eps_diag = torch.randn(B, T, 25, device=device) * temperature
    # ... combine with mean
```

**Effect**:
- Max IV: 0.7130 (closest to GT max 0.8175)
- Best balance of stability and realistic spread
- Trade-off: Underestimates true uncertainty (narrower CIs)

---

## Validation Results

### Quantitative Comparison

| Method | Max IV | Min IV | Exploded | Notes |
|--------|--------|--------|----------|-------|
| **Ground Truth** | 0.8175 | 0.0137 | N/A | Target |
| Mean-Only | 0.5588 | 0.0013 | **No** | Most stable |
| Truncated (±0.3) | 1.5046 | 0.0006 | **No** | Preserves fat-tails |
| Temperature (0.3) | 0.7130 | 0.0008 | **No** | Best balance |
| **Original** | 24.36 | 0.0002 | **YES** | Explosion |

### Quality Metrics (Mean-Only)

```
RMSE:           0.0678
MAE:            0.0350
Max Error:      0.6820
ATM Correlation: 0.6381
```

---

## Files

| File | Description |
|------|-------------|
| `visualize_3d_iv_stable_chaining.py` | Main script with all 3 stable methods |
| `results/two_stage_vae/3d_iv_stable_chaining/` | Output plots |

---

## Usage

```bash
# Mean-only (deterministic, most stable)
python experiments/backfill/two_stage_vae/visualize_3d_iv_stable_chaining.py \
    --method mean_only

# Temperature-scaled (stochastic, recommended)
python experiments/backfill/two_stage_vae/visualize_3d_iv_stable_chaining.py \
    --method temperature --temperature 0.3

# Truncated Student-t (stochastic, bounded fat-tails)
python experiments/backfill/two_stage_vae/visualize_3d_iv_stable_chaining.py \
    --method truncated --truncate_lower -0.3 --truncate_upper 0.3

# Compare all methods
python experiments/backfill/two_stage_vae/visualize_3d_iv_stable_chaining.py \
    --method all
```

---

## Recommendations

### For Stable Autoregressive Generation

1. **Use temperature=0.3** for best balance of stability and realistic uncertainty
2. **Use mean-only** when deterministic trajectories are acceptable
3. **Use truncated** when fat-tail preservation is important

### For Best Long-Horizon Forecasting

Train a **Direct Multi-Step (MIMO) model** that predicts all 30 steps in a single forward pass:
- Eliminates recursive feedback loop entirely
- No opportunity for error compounding
- Well-supported in time series forecasting literature

---

## Limitation: No Predictive Power

**Important**: While these methods prevent IV explosion, they do **not** provide actual predictive power. Investigation revealed that:

1. Single-step prediction correlation with ground truth: **~0** (essentially random)
2. Model outputs near-zero log-returns (variance collapse)
3. The VAE architecture is designed for reconstruction, not prediction

See [CHAINING_FAILURE_ANALYSIS.md](CHAINING_FAILURE_ANALYSIS.md) for the complete root cause analysis.

---

## References

- [Stratify: Unifying Multi-Step Forecasting](https://arxiv.org/html/2412.20510v1) - MIMO vs recursive comparison
- [Long Horizon Temperature Scaling](https://arxiv.org/abs/2302.03686) - Temperature scaling for sequences
- [Efficient Truncated Student-t Sampling](https://www.researchgate.net/publication/302480425) - Rejection sampling
- [Machine Learning Strategies for Multi-step Forecasting](https://souhaib-bentaieb.com/papers/2014_phd.pdf) - PhD thesis on forecasting strategies
