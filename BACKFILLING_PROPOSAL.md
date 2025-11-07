# Multi-Stock Backfilling Architecture for 1D VAE

## Overview

Proposal to extend the 1D time series VAE for **backfilling applications**, where AMZN returns are missing at T+1 but contemporaneous market data (MSFT, SP500) is available. This architecture leverages cointegration relationships between stocks to improve prediction accuracy and prevent posterior collapse.

## Backfilling Problem Description

### What is Backfilling?

**Backfilling** refers to the task of estimating missing historical data points when:
1. The target time series has gaps or delays (e.g., AMZN returns unavailable at T+1)
2. Related time series are fully observed (e.g., MSFT and SP500 available through T+1)
3. The goal is to reconstruct the missing target value using available information

This differs from **forecasting** where no future information is available, and from **nowcasting** which estimates current values using partial observations.

### The Cointegration Assumption

Classical econometric backfilling relies on **cointegration**: a long-term stable relationship between time series.

For stock returns, this manifests as:
```
AMZN[t] = β₀ + β₁·MSFT[t] + β₂·SP500[t] + ε[t]
```

Where:
- β coefficients capture systematic co-movement (market exposure)
- ε[t] is idiosyncratic AMZN-specific risk
- The relationship is stable over time (stationarity assumption)

**Key insight:** If we observe MSFT[T+1] and SP500[T+1], we can estimate AMZN[T+1] as:
- Systematic component: β₁·MSFT[T+1] + β₂·SP500[T+1]
- Plus uncertainty: ε[T+1] drawn from historical residual distribution

### Why VAE for Backfilling?

Traditional econometric methods (OLS regression, Kalman filters, ECM) have limitations:

1. **Linear relationships**: Assume constant β coefficients
2. **Homoskedastic errors**: Fixed variance for ε[t]
3. **No temporal dynamics**: Don't capture lagged effects or regime changes

**VAE advantages:**
- **Non-linear cointegration**: LSTM learns time-varying relationships
- **Heteroskedastic uncertainty**: Latent z captures regime-dependent volatility
- **Temporal context**: Encoder sees full history [0:T] for adaptive prediction
- **Probabilistic output**: Quantile regression provides calibrated confidence intervals

### Backfilling vs Forecasting in Our Context

**Forecasting (current 1D model):**
```
Given: AMZN[0:T], MSFT[0:T], SP500[0:T]
Predict: AMZN[T+1]
Information: All series lag by 1 day
```

**Backfilling (proposed architecture):**
```
Given: AMZN[0:T], MSFT[0:T+1], SP500[0:T+1]
Predict: AMZN[T+1]
Information: Market data available contemporaneously, AMZN delayed
```

**Advantage:** Backfilling leverages same-day market information, which provides much stronger signal through cointegration than lagged data.

### Real-World Applications

1. **Data quality issues**: Stock was halted, delisted, or had reporting errors
2. **International markets**: US data available, but Asian/European stock delayed
3. **Private securities**: VC/PE valuations based on public market comparables
4. **Factor models**: Estimate portfolio returns when some holdings unreported

## Motivation

### Problems with Current Architecture
1. **Low dimensionality**: Single stock input (1D per timestep) → weak anti-collapse pressure
2. **Posterior collapse**: Full encoder sees only 1 additional feature at T+1 (AMZN[T+1])
3. **Overconfident predictions**: CI violations ~44% (target: ~10%)

### Backfilling Use Case
- **Scenario**: AMZN data delayed/missing at T+1, but MSFT and SP500 data available
- **Goal**: Predict AMZN[T+1] using:
  - Historical AMZN data [0:T]
  - Contemporaneous market data MSFT[T+1], SP500[T+1]
  - Cointegration relationships learned by VAE

## Proposed Architecture

### Multi-Stock, Multi-Feature Input

**Each stock has 4 features per timestep:**

| Feature | Formula | Economic Interpretation |
|---------|---------|-------------------------|
| **Log Return** | `log(P_t / P_{t-1})` | Price change |
| **Realized Volatility** | `std(r_{t-20:t})` or `sqrt(sum(r_{t-5:t}^2))` | Recent volatility regime |
| **Volume Change** | `log(Volume_t / Volume_{t-1})` | Liquidity shocks, institutional flow |
| **Intraday Range** | `log(High_t / Low_t)` | Parkinson volatility estimator |

**Total input dimensionality:** 3 stocks × 4 features = **12 dimensions**

### Encoder Architecture

**Context Encoder (deterministic):**
```python
Input: [AMZN[0:T], MSFT[0:T], SP500[0:T]], each with 4 features
Shape: (B, T, 12)
Output: ctx_embeddings (B, T, mem_hidden)
```

**Full Encoder (stochastic):**
```python
Input: [AMZN[0:T+1], MSFT[0:T+1], SP500[0:T+1]], each with 4 features
Shape: (B, T+1, 12)
Output: z_mean, z_logvar → reparameterize → z
Shape: (B, T+1, latent_dim)
```

**Information asymmetry:** Full encoder sees 12 additional features at T+1:
- AMZN[T+1]: 4 features (target)
- MSFT[T+1]: 4 features (auxiliary)
- SP500[T+1]: 4 features (auxiliary)

This 12D asymmetry is comparable to the 2D surface VAE's 25-point asymmetry (5×5 grid).

### Decoder Architecture

```python
Input: [ctx_embeddings || z] (concatenated, padded to T+1)
Output: AMZN quantiles at T+1, shape (B, 1, 3)
  - Channel 0: p05 (5th percentile)
  - Channel 1: p50 (median)
  - Channel 2: p95 (95th percentile)
```

## Training Strategy: 80/20 Split

To improve CI calibration and train on realistic generation scenarios, use a mixed training regime.

### 80% of batches: Standard VAE training
```python
# Full information available
mu, logvar = full_encoder([AMZN[0:T+1], MSFT[0:T+1], SP500[0:T+1]])
z = reparameterize(mu[:, T+1], logvar[:, T+1])

ctx_embeddings = context_encoder([AMZN[0:T], MSFT[0:T], SP500[0:T]])
pred = decoder([ctx_embeddings || z])

loss = pinball_loss(pred, AMZN[T+1]) + kl_weight * KL(mu, logvar)
```

### 20% of batches: Simulated backfilling scenario
```python
# Simulate missing AMZN[T+1] using forward fill
amzn_fwd_fill = torch.cat([amzn[0:T], amzn[T:T+1]])  # Repeat last value

# Encoder sees "stale" AMZN, but current market data
mu_prev, logvar_prev = full_encoder([amzn_fwd_fill, MSFT[0:T+1], SP500[0:T+1]])

# Use previous timestep latent (not T+1!)
z_prev = reparameterize(mu_prev[:, T], logvar_prev[:, T])

ctx_embeddings = context_encoder([AMZN[0:T], MSFT[0:T], SP500[0:T]])
pred = decoder([ctx_embeddings || z_prev])

loss = pinball_loss(pred, AMZN[T+1]) + kl_weight * KL(mu_prev, logvar_prev)
```

**Key insight:** The 20% case trains the model to:
- Make predictions with "degraded information" (previous day latent)
- Learn appropriate uncertainty (wider logvar when information is limited)
- Directly match the generation scenario (realistic test conditions)

### Rationale

**Why 80/20 split:**
- Model primarily learns from full information (80%)
- But regularized to handle missing target data (20%)
- Prevents overconfident predictions when using lagged latent at generation

**Why forward fill over zero-padding/mask token:**
- Forward fill (repeat AMZN[T]) is closer to true AMZN[T+1] than zeros
- Simpler than learnable mask token
- Still provides realistic "degraded information" scenario

**Why use mu[:, T] instead of mu[:, T+1] in 20% case:**
- mu[:, T+1] would be computed from forward-filled data (uninformative)
- mu[:, T] represents "best estimate given data up to T"
- Matches generation scenario: use most recent true latent

## Generation (Backfilling)

During actual backfilling at test time:

```python
# Available data
amzn_history = amzn[0:T]          # Real historical AMZN
msft_current = msft[0:T+1]        # Real MSFT through T+1
sp500_current = sp500[0:T+1]      # Real SP500 through T+1

# Forward fill AMZN (or use zeros/mask)
amzn_fwd_fill = torch.cat([amzn_history, amzn_history[-1:]])

# Encode to get previous day latent
mu_prev, logvar_prev = full_encoder([amzn_fwd_fill, msft_current, sp500_current])
z_prev = reparameterize(mu_prev[:, T], logvar_prev[:, T])

# Generate prediction
ctx_embeddings = context_encoder([amzn_history, msft_current[:-1], sp500_current[:-1]])
quantile_pred = decoder([ctx_embeddings || z_prev])

# quantile_pred[:, 0, 0] = p05 (lower CI bound)
# quantile_pred[:, 0, 1] = p50 (point forecast)
# quantile_pred[:, 0, 2] = p95 (upper CI bound)
```

## Expected Benefits

### Anti-Collapse Mechanisms
1. **12D information asymmetry** (vs 1D in current model)
2. **Multi-feature conditioning** captures richer market dynamics
3. **Cointegration structure** forces z to encode joint market state

### Improved Calibration
1. **Training on degraded information (20%)** → learns appropriate uncertainty
2. **Day-specific mu, logvar** from contemporaneous market data
3. **Adaptive CI width** based on market regime (volatility, volume)

### Economically Justified
1. **Cointegration backfilling** matches classical econometric approach
2. **Uses available market data** (MSFT/SP500 at T+1)
3. **Captures cross-stock dynamics** (systematic vs idiosyncratic risk)

## Implementation Checklist

### Data Preparation
- [ ] Fetch MSFT and SP500 historical data from Yahoo Finance
- [ ] Compute 4 features per stock: log_return, realized_vol, volume_change, parkinson_range
- [ ] Create dataset: `data/stock_returns_multifeature.npz`
  - `amzn_features`: (N, 4)
  - `msft_features`: (N, 4)
  - `sp500_features`: (N, 4)

### Model Changes
- [ ] Update `CVAE1DMemRand` to accept 12D input (3 stocks × 4 features)
- [ ] Modify encoder to handle multi-stock concatenated input
- [ ] Increase `latent_dim` from 5 to 10-15 (handle richer state space)
- [ ] Implement 80/20 training split with forward fill logic

### Training Script
- [ ] Create `train_1d_backfilling.py`
- [ ] Add random masking logic (20% probability per batch)
- [ ] Log separate losses for full vs degraded batches
- [ ] Monitor CI violation rates on validation set

### Evaluation
- [ ] Implement backfilling generation in `generate_1d_quantile_predictions.py`
- [ ] Compute metrics: RMSE, MAE, R², Direction Accuracy, CI Violation Rate
- [ ] Compare against:
  - Baseline: Single-stock model (AMZN only)
  - Ablation: 80/20 vs 100% full training
  - Alternative: Zero-padding vs forward fill

## Configuration Example

```python
backfill_config = {
    "model_type": "cvae_1d_mem_rand",
    "latent_dim": 5,              
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    "mem_type": "lstm",
    "mem_hidden": 32,
    "kl_weight": 1e-5,

    # Multi-stock input
    "num_stocks": 3,               # AMZN, MSFT, SP500
    "features_per_stock": 4,       # return, vol, volume, range
    "ex_feats_dim": 12,            # Total: 3 × 4

    # Training strategy
    "degraded_prob": 0.3,          # 30% use previous day latent
    "forward_fill_method": "repeat_last",  # or "zero" or "mask_token"
}
```

## Open Questions

1. **Optimal latent_dim:** 10, 15, or 20? (Trade-off: expressiveness vs overfitting)
2. **Forward fill vs mask token:** Simple repeat vs learned embedding?
3. **CI calibration target:** Tune for 10% violations or allow 15% for better coverage?
4. **Multi-day horizon:** Extend to T+2, T+3 predictions (cumulative error concerns)
5. **Stock selection:** Expand to more stocks (tech sector) or keep parsimonious?

## References

- **RVRAE (2024):** "Variational Recurrent Autoencoder for Stock Returns Prediction" - Uses cross-sectional returns as encoder input, achieves R²=1.57
- **Conditional Quantile VAE:** "Asset pricing via the conditional quantile variational autoencoder" - Achieves 30.9% higher Sharpe ratios
- **Econometric backfilling:** Kalman filters, error-correction models with contemporaneous auxiliary series
- **β-VAE literature:** Free bits, KL annealing strategies for preventing posterior collapse

## Version History

- **2025-11-07:** Initial proposal - multi-stock backfilling architecture with 80/20 training strategy
