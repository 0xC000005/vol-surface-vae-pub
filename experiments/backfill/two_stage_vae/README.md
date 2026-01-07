# Two-Stage VAE Experiments

This folder contains experiments for the Two-Stage Conditional VAE architecture for volatility surface forecasting.

## Architecture Overview

The Two-Stage VAE decouples representation learning from forecasting:

1. **Stage 1 (Context Encoder)**: Encodes historical context into embeddings
2. **Stage 2 (Latent Predictor + Decoder)**: Predicts latent distributions and decodes to log-returns

Key features:
- **Log-return space**: Predicts Δlog(IV) instead of absolute IV levels
- **Heteroscedastic decoder**: Learns per-point variance via NLL loss
- **Context embedding**: 3D context representation for downstream predictors

## Related Files

### Core Model (in `vae/`)
- `vae/cvae_two_stage.py` - Main model implementation
- `vae/predictors.py` - LatentPredictor and ContextPredictor modules

### Configuration (in `config/`)
- `config/two_stage_config.py` - TwoStageConfig class

### Model Checkpoints (in `models/backfill/two_stage/`)
- `two_stage_log_return_nll_best.pt` - Best heteroscedastic model
- `unconditional_analysis.npz` - Pre-computed validation samples
- `conditional_analysis.npz` - Cluster analysis results
- `visualizations/` - Generated plots

## Scripts in This Folder

### Training
| Script | Description |
|--------|-------------|
| `train_two_stage_autoencoder.py` | Train basic two-stage autoencoder |
| `train_two_stage_log_return.py` | Train with log-return prediction (MSE) |
| `train_two_stage_log_return_nll.py` | Train with heteroscedastic NLL loss |

### Validation
| Script | Description |
|--------|-------------|
| `validate_two_stage_log_return.py` | Validate log-return model |
| `validate_two_stage_log_return_nll.py` | Validate heteroscedastic model |
| `validate_two_stage_forecast.py` | Validate forecasting performance |
| `validate_conditional_variance.py` | Test variance predictions |

### Analysis
| Script | Description |
|--------|-------------|
| `analyze_unconditional_marginal.py` | Analyze marginal distributions |
| `analyze_conditional_marginal.py` | Analyze per-cluster distributions |

### Visualization
| Script | Description |
|--------|-------------|
| `visualize_fanning_patterns.py` | Fanning plots in log-return space |
| `visualize_fanning_iv_space.py` | Fanning plots in IV level space |

### Tests
| Script | Description |
|--------|-------------|
| `test_causal_ctx_encoder.py` | Test causal context encoder |
| `test_film_decoder.py` | Test FiLM decoder conditioning |
| `test_log_return_variance.py` | Test variance predictions |

## Documentation

- `TWO_STAGE_VAE_CRITIQUE.md` - Comprehensive analysis of model issues

## Usage

All scripts should be run from the repository root:

```bash
# Train the heteroscedastic model
python experiments/backfill/two_stage_vae/train_two_stage_log_return_nll.py

# Validate
python experiments/backfill/two_stage_vae/validate_two_stage_log_return_nll.py

# Analyze marginal distributions
python experiments/backfill/two_stage_vae/analyze_unconditional_marginal.py

# Generate visualizations
python experiments/backfill/two_stage_vae/visualize_fanning_patterns.py
```

## Known Issues

See `TWO_STAGE_VAE_CRITIQUE.md` for detailed analysis. Key issues:

1. **Fat tails missing** - Kurtosis 1.4 vs GT 21.3
2. **Tail asymmetry reversed** - Pos/Neg ratio 0.75 vs GT 1.78
3. **Cross-grid correlation destroyed** - Corr 0.02 vs GT 0.32
4. **Systematic negative bias** - -0.006 per step

Root cause: Gaussian decoder assumption in heteroscedastic NLL loss.

## Next Steps

1. Implement Student-t decoder for fat tails
2. Add skewness parameter for asymmetry
3. Full covariance decoder for correlations
4. Evaluate under prior mode (z ~ N(0,1))
