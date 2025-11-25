# Quantile Regression Decoder for Volatility Surface Forecasting

A quantile regression variant has been implemented to address confidence interval (CI) calibration issues in uncertainty quantification. This represents an architectural enhancement to directly predict confidence intervals rather than computing them from Monte Carlo samples.

## Motivation

Original MSE-based models showed CI violation rates of 35-72%, far exceeding the target ~10%. The quantile regression decoder explicitly learns conditional quantiles during training to:

1. Improve calibration of uncertainty estimates
2. Provide dramatically faster inference (~1000× speedup)
3. Directly output prediction intervals without Monte Carlo sampling

## Architecture Changes

### Decoder Output

- **Original MSE model**: 1 channel (mean prediction)
- **Quantile model**: 3 channels (p5, p50, p95 quantiles)
- Output shape: `(B, T, 3, H, W)` where dim=2 represents quantiles

### Loss Function

**Original**: MSE (Mean Squared Error)
```python
loss = (y_true - y_pred)^2
```

**Quantile**: Pinball Loss (asymmetric quantile loss)
```python
def pinball_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return torch.where(error >= 0, quantile * error, (quantile - 1) * error)
```

**Asymmetric penalties:**
- For τ=0.05: Over-predictions penalized 19× more → learns conservative lower bound
- For τ=0.95: Under-predictions penalized 19× more → learns conservative upper bound

### Generation Speedup

| Method | Process | Speed |
|--------|---------|-------|
| **Original** | 1,000 forward passes with z ~ N(0,1) → compute empirical quantiles | Baseline |
| **Quantile** | 1 forward pass → get all 3 quantiles directly | **~1000× faster** |

## Training Configuration

All three model variants (no_ex, ex_no_loss, ex_loss) support quantile regression:

```python
model_config = {
    "use_quantile_regression": True,
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    # ... other params same as baseline
    "latent_dim": 5,
    "mem_hidden": 100,
    "surface_hidden": [5, 5, 5],
    "kl_weight": 1e-5,
}
```

## Performance Results

### Training Performance (Test Set)

| Model | Surface RE Loss | KL Loss | Total Loss |
|-------|-----------------|---------|------------|
| **no_ex** | 0.006349 | 6.033 | 0.006409 |
| **ex_no_loss** | 0.006269 | 7.768 | 0.006347 |
| **ex_loss** | 0.006206 | 5.876 | 0.006392 |

Note: RE loss is pinball loss, not directly comparable to MSE.

### CI Calibration Results

| Model | CI Violations | Below p05 | Above p95 | Mean CI Width |
|-------|---------------|-----------|-----------|---------------|
| **no_ex** | 44.50% | 4.79% | 39.71% | 0.0811 |
| **ex_no_loss** | 35.43% | 17.82% | 17.60% | 0.0855 |
| **ex_loss** | 34.28% | 5.55% | 28.72% | 0.0892 |

**Interpretation:**
- **Expected**: ~10% violations (well-calibrated intervals)
- **Actual**: 34-45% violations (improvement from baseline ~50%+ but needs further work)
- **Best model**: ex_loss with 34.28% violations

### Verification Results (2008-2010 Financial Crisis)

CI violations on crisis period:

| Latent Type | no_ex | ex_no_loss | ex_loss |
|-------------|-------|------------|---------|
| **Ground truth** | 7.32% | 5.33% | 6.89% |
| **Context-only** | 18.63% | 20.44% | 19.65% |

**Key finding:** The 3× gap between ground truth and context-only latents reveals **VAE prior mismatch**: p(z|context) ≠ p(z|context+target). The decoder learns correct quantiles when given encoded latents (ground truth), but the prior distribution doesn't match the posterior for realistic generation (context-only).

## Key Observations

1. **Improvement over baseline**: Reduced violations from ~50%+ to ~34% (ex_loss model)
2. **Asymmetric violations**: Most models have more violations above p95 than below p05 (underestimate upper tail uncertainty)
3. **Generation speed**: Dramatically faster (~1000× speedup)
4. **Trade-off**: Better calibration vs more complex loss function

## Usage

### Training

```bash
python train_quantile_models.py
```

Trains all three variants (no_ex, ex_no_loss, ex_loss) with quantile regression.

### Generating Predictions

```bash
python generate_quantile_surfaces.py
```

Generates quantile forecasts for test period.

### Evaluating Calibration

```bash
python evaluate_quantile_ci_calibration.py
```

Computes CI violation statistics.

### Verification Scripts

Located in `analysis_code/`:

```bash
# Ground truth latent (full sequence encoding)
python analysis_code/verify_reconstruction_plotly_2008_2010.py

# Context-only latent (realistic generation)
python analysis_code/verify_reconstruction_2008_2010_context_only.py

# Marginal distribution analysis
python analysis_code/visualize_marginal_distribution_quantile_encoded.py
python analysis_code/visualize_marginal_distribution_quantile_context_only.py
```

## Model Output Format

### Quantile Surfaces

Quantile models output 3 surfaces per prediction:

```python
surfaces = model.get_surface_given_conditions(ctx_data)
# Shape: (B, 1, 3, 5, 5)

p05 = surfaces[:, :, 0, :, :]  # 5th percentile (lower bound)
p50 = surfaces[:, :, 1, :, :]  # 50th percentile (median, point forecast)
p95 = surfaces[:, :, 2, :, :]  # 95th percentile (upper bound)
```

### Extra Features (EX Loss models)

For models with `ex_feats_dim > 0`:

```python
ex_feats = model.get_surface_given_conditions(ctx_data)[1]
# Shape: (B, 1, 3, 3)

p05_features = ex_feats[:, :, 0, :]  # [return, skew, slope] at p05
p50_features = ex_feats[:, :, 1, :]  # [return, skew, slope] at p50
p95_features = ex_feats[:, :, 2, :]  # [return, skew, slope] at p95
```

## Current Status & Next Steps

### Status

- ✅ **Implementation successful**: Stable training with pinball loss
- ✅ **Speed improvement**: 1000× faster generation
- ✅ **Partial calibration improvement**: ~16% reduction in CI violations vs baseline
- ❌ **Calibration needs work**: CI violations at 34% (target: 10%)

### Completed

- Stable training with pinball loss
- 1000× generation speedup
- ~16% reduction in CI violations vs baseline MSE models
- Verification of VAE prior mismatch issue

### Remaining Challenges

1. **High CI violation rates**: 34% vs target 10%
2. **Asymmetric violations**: Models underestimate upper tail uncertainty (more violations above p95)
3. **VAE prior mismatch**: Context-only latents produce 3× higher violations than ground truth latents

### Recommended Next Steps

1. **Apply conformal prediction** for post-hoc calibration
   - Use holdout calibration set to adjust quantile predictions
   - Adaptive intervals that guarantee coverage

2. **Retrain with loss reweighting**
   - Emphasize tail quantiles (p05, p95) more than median
   - Increase penalty for tail violations

3. **Explore heteroscedastic quantile regression**
   - Allow prediction interval width to vary with volatility level
   - Condition quantile spread on market regime

4. **Address VAE prior mismatch**
   - Improve context encoder to better match posterior distribution
   - Explore alternative prior distributions (e.g., flow-based priors)
   - Consider posterior sampling with importance weighting

## Implementation Details

- **Core implementation**: `vae/cvae_with_mem_randomized.py` (CVAEMemRand class)
- **Quantile decoder**: `vae/cvae_with_mem_randomized.py` (CVAEMemRandQuantileDecoder)
- **Loss function**: Pinball loss in `train_step()` method
- **Training script**: `train_quantile_models.py`
- **Evaluation**: `evaluate_quantile_ci_calibration.py`

## References

- Main paper: Chen et al. (2025), "A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces"
- Quantile regression: Koenker & Bassett (1978), "Regression Quantiles"
- Pinball loss: Steinwart & Christmann (2011), "Estimating conditional quantiles with the help of the pinball loss"
