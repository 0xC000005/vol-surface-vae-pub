# Experiment 11: Heteroscedastic VAE Results

## Objective
Train a VAE with heteroscedastic decoder that outputs (mean, log_var) per grid point, aiming to:
1. Learn context-dependent variance (high vol regime → higher predicted var)
2. Achieve ~90% CI coverage from the learned variance
3. Maintain unconditional marginal matching (Gen Var ≈ GT Var)

## Architecture
- **Encoder/Decoder**: Same as baseline CVAEFullCovPrior
- **Decoder modification**: Parallel log_var head alongside mean head
- **Loss**: Gaussian NLL instead of MSE for reconstruction
  - NLL = 0.5 * (log_var + (x - mu)² / exp(log_var))
- **Variance regularization**: Optional penalty on deviation from target variance

## Key Results

### Without Variance Regularization
| Metric | Value |
|--------|-------|
| Predicted Variance | ~0.00001 (collapses to near-zero) |
| Coverage (90% target) | 18.4% |
| Variance Ratio (high/low vol) | 1.00x |

**Finding**: Variance collapses because NLL incentivizes minimizing log_var when mean predictions are accurate.

### With Variance Regularization (target = GT variance)
| Metric | Value | Target |
|--------|-------|--------|
| Predicted Variance | 0.0095 | 0.0096 (GT) |
| Coverage | 31% | 90% |
| Variance Ratio | 1.00x | ~2x |

**Finding**: Regularization successfully matches variance to GT, but coverage is still low because prediction errors (from prior mismatch) are larger than GT variance.

### With Higher Target Variance
| Target Variance | Pred Var | Coverage | Notes |
|----------------|----------|----------|-------|
| 1x GT (0.01) | 0.0095 | 31% | Matches GT, low coverage |
| 3x GT (0.03) | 0.0298 | 65% | Better coverage |
| 10x GT (0.10) | 0.100 | 19% | Mean quality degraded! |

**Finding**: Higher target variance improves coverage up to a point (65% at 3x), but too high causes mean prediction quality to degrade (NLL penalty for mean error becomes small with large variance).

## Fundamental Limitation

The heteroscedastic approach fails to learn **context-dependent** variance because:

1. **Decoder sees z, not regime**: The decoder receives sampled z and context embedding, but these don't carry explicit regime information
2. **Training-test distribution shift**: During training, z comes from posterior (informed by target). During testing, z comes from prior (no target info)
3. **Constant variance is optimal**: For the NLL objective, outputting constant variance across all contexts minimizes loss

### Two Types of Uncertainty

| Type | Source | Can Decoder Learn It? |
|------|--------|----------------------|
| **Aleatoric** | Inherent noise in y given z | Yes (but constant) |
| **Epistemic** | Uncertainty about z given context | No |

The main uncertainty source in VAE generation is **epistemic** (prior-posterior mismatch), not aleatoric. The decoder cannot capture this because it doesn't know what z "should have been".

## Monte Carlo Sampling Analysis

The evaluate_coverage function samples multiple z's from prior and decodes each:
```python
for _ in range(n_samples):
    sample = model.get_surface_given_conditions(ctx_dict, sample_from_decoder=True)
```

Total variance = Var(E[y|z]) + E[Var(y|z)]
- First term: variance in mean predictions across z's (epistemic)
- Second term: average decoder variance (aleatoric, learned)

When decoder variance is constant, total variance is constant across regimes, giving var_ratio = 1.0x.

## Comparison with Regime-Based Calibration (exp10c)

| Approach | Coverage | Var Ratio | Learns Variance? |
|----------|----------|-----------|------------------|
| Heteroscedastic VAE (this exp) | 31-65% | 1.0x | No (constant) |
| Regime-based (exp10c) | 89.5% | 2.05x | Yes (post-hoc) |

**Conclusion**: Post-hoc regime-based calibration outperforms end-to-end heteroscedastic learning because it directly estimates variance from empirical prediction errors grouped by regime.

## Recommendations

1. **For calibrated CIs**: Use regime-based post-hoc calibration (exp10c approach)
   - Classify contexts into low/mid/high vol terciles
   - Estimate empirical variance for each regime
   - Apply regime-specific variance to predictions

2. **For end-to-end learning**: Consider architectural changes
   - Explicit context feature conditioning (ATM IV as direct input to variance head)
   - Separate mean and variance training phases
   - Quantile regression instead of Gaussian variance

3. **Alternative**: Conformal prediction with proper exchangeability handling
   - Use random splits instead of time-ordered (requires caution)
   - Or: Use sliding window conformalization

## Files
- `vae/cvae_heteroscedastic.py`: Model implementation
- `exp11_heteroscedastic_vae.py`: Training script
- `exp10c_regime_variance.py`: Better-performing regime-based approach
