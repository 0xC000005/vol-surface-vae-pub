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
| `exp_low_rank_cov.py` | **CI calibration fix**: Low-rank covariance decoder with mean-reversion correction |

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

## Low-Rank Covariance Decoder (CI Calibration Fix)

### Problem
Vanilla VAE decoder only outputs **mean** E[x|z], missing the **variance** term E[Var(x|z)]. This caused CI violations of 0-2% (CIs too wide) instead of target 10%.

### Solution: Low-Rank Covariance Decoder

**Script:** `exp_low_rank_cov.py`

#### Key Changes from Vanilla VAE

1. **Low-Rank Covariance Structure**
   - Added learnable covariance: Σ = FF^T + D
   - `F`: (5,5,rank) factor matrix for spatial correlation
   - `D`: (5,5) diagonal for per-grid residual variance
   - Sampling: `x = mean + F@ε_rank + sqrt(D)*ε_diag`

2. **GT Variance Initialization**
   - Changed `log_diag = zeros(5,5)` → `log_diag = log(GT_variance)`
   - Gives the model a good starting point instead of learning from scratch

3. **Two-Stage Training**
   - Stage 1: Train mean decoder with MSE (freeze covariance)
   - Stage 2: Train covariance with NLL (freeze mean), epochs=200, lr=0.01

4. **Mean-Reversion Correction** (key discovery)
   - GT log-returns have **negative autocorrelation** (ACF(1) = -0.37)
   - Cumulative variance grows slower than H × single_var:
     - H=7: 33% of i.i.d. expected
     - H=30: 20% of i.i.d. expected
   - Added `compute_empirical_variance_ratios()` to scale cumulative variance

5. **Evaluation Fix**
   - Use **cumulative noise** (single draw per H-step) instead of per-timestep noise
   - `cum_var = H × learned_var × var_ratio[H]`

### Results (Oracle Case)

| Horizon | Before | After |
|---------|--------|-------|
| H=1     | 0-2%   | 4%    |
| H=7     | 0-2%   | 9%    |
| H=14    | 0-2%   | 9%    |
| H=30    | 0-2%   | 11%   |

### Usage

```bash
python experiments/backfill/two_stage_vae/exp_low_rank_cov.py
```

### Output
- Models: `models/backfill/two_stage/low_rank_cov/`
- Summary: `models/backfill/two_stage/low_rank_cov/summary.json`

### Key Files
- `exp_low_rank_cov.py` - Main experiment script
- `LowRankCovarianceDecoder` - Decoder with Σ = FF^T + D
- `compute_empirical_variance_ratios()` - Mean-reversion correction
- `evaluate_ci_violations()` - CI evaluation with corrections

### References & Inspirations

#### Heteroscedastic Regression
- **β-NLL Loss**: Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks." *ICLR 2022*. [arXiv:2203.09168](https://arxiv.org/abs/2203.09168), [GitHub](https://github.com/martius-lab/beta-nll)
  - Key insight: Standard NLL can lead to "lazy variance" where model inflates variance for hard examples
  - Solution: Weight loss by `variance^β` (β=0.5 recommended)

- **Faithful Heteroscedastic Regression**: Stirn, A., Wessels, H., et al. (2023). "Faithful Heteroscedastic Regression with Neural Networks." *AISTATS 2023*, PMLR 206:5593-5613. [arXiv:2212.09184](https://arxiv.org/abs/2212.09184), [PDF](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf)
  - Key insight: Heteroscedastic models can produce worse mean estimates than homoscedastic
  - Solution: Two-stage training that provably retains mean accuracy

#### Low-Rank Covariance in VAEs
- **SOS-VAE (Structured Observation Space)**: "Structured Uncertainty in the Observation Space of Variational Autoencoders." [arXiv:2205.12533](https://arxiv.org/abs/2205.12533)
  - Key insight: Standard VAE uses pixel-wise independent distributions → spatially-incoherent samples
  - Solution: Low-rank parameterization of observation covariance

- **Spatial VAE via Matrix-Variate Normal**: "Spatial Variational Auto-Encoding via Matrix-Variate Normal Distributions." [OpenReview](https://openreview.net/forum?id=ryOBB6g-M)
  - Proposes low-rank MVN distributions to capture spatial dependencies

- **σ-VAE**: "Simple and Effective VAE Training with Calibrated Decoders." [Project Page](https://orybkin.github.io/sigma-vae/)
  - Learning shared decoder variance rather than per-pixel variance

#### Mean-Reversion in Volatility
- **Stylized Facts**: Mean reversion is well-documented in implied volatility: "While volatility clusters in the short term, it exhibits long-run mean reversion." [Bloomberg](https://www.bloomberg.com/professional/insights/trading/volatilitys-a-mean-reverting-asset-but-speed-level-are-unclear/)

- **Negative Autocorrelation**: "Mean reversion is synonymous with negative auto- or serial-correlation." Our empirical finding of ACF(1) = -0.37 aligns with established stylized facts in financial econometrics. [Bionic Turtle](https://forum.bionicturtle.com/threads/square-root-rule-with-mean-reversion-autocorrelation-var-volatility.4463/)

- **Hull-White Models**: Mean reversion affects the swaption volatility surface shape. [S&P Global](https://www.spglobal.com/market-intelligence/en/news-insights/research/implied-interest-rate-volatility-and-xva-how-the-onefactor-hul)

#### Two-Stage VAE Training
- **Diagnosing and Enhancing VAE Models**: Dai, B., & Wipf, D. (2019). [arXiv:1903.05789](https://arxiv.org/pdf/1903.05789)
  - Key insight: "Separate training of stages is important" - joint training performs no better than first stage alone
  - Our approach: Stage 1 (MSE for mean), Stage 2 (NLL for variance with frozen mean)

## Next Steps

1. Implement Student-t decoder for fat tails
2. Add skewness parameter for asymmetry
3. Full covariance decoder for correlations
4. Evaluate under prior mode (z ~ N(0,1))
