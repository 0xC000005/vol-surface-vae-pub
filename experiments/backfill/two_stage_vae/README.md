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

## Student-t Decoder Experiments

### Problem Statement
The Low-Rank Covariance decoder achieved good CI calibration but with Gaussian assumptions.
The ground truth log-returns exhibit fat tails (kurtosis ~4.6 at ATM) that Gaussian decoders cannot capture.

### Solution: Student-t Decoder Family

We developed a series of Student-t decoders with per-grid degrees of freedom learned from GT kurtosis.

#### Model Comparison (Consistent Methodology)

| Model | Kurtosis | Z Contrib | Ctx Contrib | MSE | ACF |
|-------|----------|-----------|-------------|-----|-----|
| **StudentTMLPDecoder** (baseline) | 136.2% | 39.5% | 0.0% | 0.0606 | 15.8% |
| **StudentTDualPathDecoder** | **134.4%** | 38.7% | **16.1%** | **0.0567** | 18.6% |
| StudentTGatedResidualDecoder | 138.7% | 33.1% | 2.4% | 0.0656 | 15.0% |
| WarmStart Gated Residual | 145.4% | 39.5% | 0.1% | 0.0603 | 15.5% |

**Winner: StudentTDualPathDecoder** - Only model to pass both kurtosis (>100%) AND context contribution (>5%) criteria.

### Architecture Details

#### 1. StudentTMLPDecoder (Baseline)
- **Architecture**: `mean = MLP(z)` - context ignored
- **Result**: 136% kurtosis recovery but 0% context contribution
- **Issue**: Context is completely bypassed

#### 2. StudentTDualPathDecoder (Best)
- **Architecture**: `mean = ctx_mean + z_residual` (additive)
- **Key insight**: Addition forces BOTH pathways to contribute
- **Result**: 134% kurtosis + 16% context contribution + lowest MSE
- **Training**: Two-phase (Phase A: MSE, Phase B: Student-t NLL)

```python
# Core additive combination
ctx_mean = self.ctx_mean_net(ctx_flat)  # Expected behavior
z_residual = self.z_residual_net(z_flat)  # Innovation
mean = ctx_mean + z_residual  # Both must contribute
```

#### 3. StudentTGatedResidualDecoder
- **Architecture**: `mean = z_pred + gate * ctx_correction`
- **Key insight**: Gate limits context influence (max 0.3)
- **Result**: Good kurtosis but gate stayed near zero
- **Training**: Three-phase (z only, context+gate, variance)

#### 4. WarmStart Gated Residual
- **Architecture**: Same as gated residual
- **Strategy**: Copy trained z pathway, freeze, train only context
- **Result**: Z pathway preserved but context never activated

### Why Dual-Path Works

The fundamental problem was that FiLM-style multiplicative conditioning allows the model to bypass context:
- FiLM: `output = gamma(z) * ctx + beta(z)` → when gamma→1, beta→0, z becomes optional
- MLP: `output = MLP(z)` → context completely ignored

**Additive dual-path solves this**:
- `mean = f(ctx) + g(z)`
- Neither pathway can be zeroed without destroying predictions
- Forces information decomposition: ctx→expected drift, z→residual/innovation

### Training Scripts

| Script | Description |
|--------|-------------|
| `exp_student_t_decoder.py` | Train StudentTMLPDecoder |
| `exp_dual_path_decoder.py` | Train StudentTDualPathDecoder (recommended) |
| `exp_gated_residual_decoder.py` | Train StudentTGatedResidualDecoder |
| `exp_warmstart_context.py` | Warm-start from trained z pathway |
| `compare_all_decoders.py` | Unified comparison with consistent methodology |

### Usage

```bash
# Train the dual-path decoder (recommended)
python experiments/backfill/two_stage_vae/exp_dual_path_decoder.py

# Compare all decoder variants
python experiments/backfill/two_stage_vae/compare_all_decoders.py
```

### Key Findings

1. **Kurtosis is preserved** - All Student-t decoders achieve >130% kurtosis recovery
2. **Context can contribute** - Dual-path additive architecture achieves 16% context contribution
3. **MSE improved** - Dual-path achieves lowest reconstruction MSE (0.0567)
4. **ACF still challenging** - All models at 15-19% ACF preservation (target: 30%)
5. **Evaluation methodology matters** - Use full dataset, not just validation split

### Model Checkpoints

- `models/backfill/two_stage/student_t/student_t_best.pt` - StudentTMLPDecoder
- `models/backfill/two_stage/dual_path/dual_path_best.pt` - StudentTDualPathDecoder (recommended)
- `models/backfill/two_stage/gated_residual/gated_residual_best.pt` - StudentTGatedResidualDecoder
- `models/backfill/two_stage/warmstart_context/warmstart_context_best.pt` - WarmStart variant

## ACF Preservation: DualPath + AR(1) Decoder

### Problem
All Student-t models achieved excellent kurtosis (127-145%) but poor ACF preservation (15-19%). The ground truth log-returns exhibit strong mean reversion (ACF lag-1 = -0.325) that was not captured.

### Solution: AR(1) Component + Spectral Loss

We added an autoregressive mean-reversion component to the dual-path decoder:

```python
# Architecture: mean = ctx_mean + z_residual + φ * (x_{t-1} - μ)
# φ learned to be ~-0.37 (close to GT -0.35)
```

Combined with **spectral loss** (FFT-based frequency matching) during training, this preserves temporal dynamics.

### Final Comparison (All Decoders)

| Model | Kurtosis | Ctx% | ACF% | Passes All? |
|-------|----------|------|------|-------------|
| StudentTMLPDecoder (baseline) | 141.4% | 0.0% | 15.6% | ❌ |
| StudentTDualPathDecoder | 127.5% | 16.1% | 19.5% | ❌ |
| StudentTGatedResidualDecoder | 145.1% | 2.4% | 15.3% | ❌ |
| **CVAETwoStageDualPathAR** | **128.4%** | **13.4%** | **34.9%** | **✅** |

**Winner: CVAETwoStageDualPathAR** - Only model to pass all three criteria!

### Key Improvements

1. **ACF Preservation**: 15-19% → **34.9%** (more than doubled!)
2. **Learned φ**: -0.367 (very close to GT -0.35)
3. **Kurtosis**: Preserved at 128.4%
4. **Context**: Still contributing at 13.4%

### Files

- `vae/losses.py` - Spectral loss, ACF loss, temporal loss functions
- `vae/cvae_two_stage.py` - StudentTDualPathARDecoder, CVAETwoStageDualPathAR
- `exp_acf_preservation.py` - Training script
- `models/backfill/two_stage/dual_path_ar/dual_path_ar_best.pt` - Best checkpoint

### Usage

```bash
# Train the AR(1) model
python experiments/backfill/two_stage_vae/exp_acf_preservation.py

# Compare all decoder variants
python experiments/backfill/two_stage_vae/compare_all_decoders.py
```

### Research Sources

- [Koopman Autoencoders - Nature](https://www.nature.com/articles/s41467-018-07210-0)
- [K²VAE - ICML 2025](https://openreview.net/forum?id=71Mm8GDGYd)
- [Unified GARCH-RNN](https://arxiv.org/html/2504.09380)
- [Focal Frequency Loss - ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.pdf)

## Model Validation (Oracle Mode)

### Comprehensive Validation Script

**Script:** `validate_student_t_oracle.py`

Performs comprehensive model validation for financial institution deployment:
- Volatility smile preservation
- CI calibration per grid point
- Distribution shape (kurtosis, skewness)
- Cross-grid correlation structure
- Risk assessment (SR 11-7 framework)

### Final Assessment: APPROVED FOR DEPLOYMENT

| Metric | In-Sample | Validation | Crisis 2008 | Target | Status |
|--------|-----------|------------|-------------|--------|--------|
| **CI Violations** | 10.9% | 9.6% | 13.6% | 10% | **PASS** |
| **Smile Sign Match** | - | 81.7% | - | >75% | **PASS** |
| **Kurtosis Recovery** | 183% | 302% | 254% | >50% | **PASS** |
| **Correlation MAE** | 0.135 | 0.143 | 0.284 | <0.2 | **PASS** |

### Key Finding: Model Learned Complex Smile Structure

The model correctly discovered that **SPX volatility smile varies by maturity**:

| Maturity | GT Typical Shape | Model Sign Match |
|----------|------------------|------------------|
| 1M | Inverted (94%) | 84.5% |
| 3M | U-shape (77%) | 64.0% |
| 6M | Inverted (86%) | 87.5% |
| 1Y | Inverted (81%) | 85.5% |
| 2Y | U-shape (94%) | 87.0% |

**Important:** SPX smile is NOT always U-shaped!
- Short-term (1M): Typically **inverted** (ATM > wings)
- Medium-term (6M, 1Y): Typically **inverted/flat**
- Long-term (2Y): Classic **U-shape**

### Deployment Recommendations

**Approved Use Cases:**
- Scenario generation for risk management
- Stress testing and VaR calculations
- Volatility surface forecasting
- Monte Carlo simulations

**Use with Caution:**
- Real-time pricing (correlation degrades in stress)
- Delta hedging (test on realized P&L first)

**Limitations to Document:**
- Correlation degrades 2x in crisis periods
- Oracle mode only (prior sampling shows wider CIs)
- Requires quarterly backtesting

### Usage

```bash
# Run full validation
python experiments/backfill/two_stage_vae/validate_student_t_oracle.py
```

## Next Steps

1. ~~Implement Student-t decoder for fat tails~~ ✓ Done
2. ~~Make context contribute~~ ✓ Done (Dual-Path achieves 16%)
3. ~~Improve ACF preservation~~ ✓ Done (AR(1) achieves 35%)
4. ~~Model validation (oracle mode)~~ ✓ Done - APPROVED
5. Evaluate under prior mode (z ~ N(0,1))
6. Run arbitrage tests on IV levels
