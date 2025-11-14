# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase implements a Variational Autoencoder (VAE) approach for conditional generation of future volatility surfaces, as described in the paper:

**Jacky Chen, John Hull, Zissis Poulos, Haris Rasul, Andreas Veneris, Yuntao Wu**, "*A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces*", Journal of Financial Data Science, 2025.

The system uses a Conditional VAE (CVAE) combined with LSTM to generate one-day-ahead volatility surface forecasts based on variable context lengths. The models use **teacher forcing** during inference - conditioning on real historical data rather than previous predictions - to produce independent forecasts for evaluation.

## Development Environment

### Package Management
- Uses `uv` for Python dependency management (Python >=3.13)
- Dependencies defined in `pyproject.toml`
- Install dependencies: `uv sync`
- Key dependencies: PyTorch, NumPy, pandas, scikit-learn, matplotlib

### Common Commands

**Training Models:**
```bash
python param_search.py
```
Trains CVAE models with different configurations (with/without extra features). Saves models to `test_spx/` directory.

**Generate Volatility Surfaces:**
```bash
python generate_surfaces.py  # Generate distributions over time horizon
python generate_surfaces_max_likelihood.py  # Generate maximum likelihood surfaces
```

**Run Analysis and Generate Tables/Plots:**
```bash
python main_analysis.py
```
Generates tables and plots for regression analysis, PCA, arbitrage checks, and classification tasks.

**Visualization and Verification (analysis_code/):**
```bash
# Visualization scripts
python analysis_code/visualize_teacher_forcing.py          # Compare all models (teacher forcing)
python analysis_code/visualize_quantile_teacher_forcing.py # Quantile-specific visualization
python analysis_code/visualize_distribution_comparison.py  # Distribution comparisons

# Verification scripts (2008-2010 crisis period)
python analysis_code/verify_reconstruction_plotly_2008_2010.py        # Ground truth latent (interactive)
python analysis_code/verify_reconstruction_2008_2010_context_only.py  # Context-only latent
python analysis_code/visualize_marginal_distribution_quantile_encoded.py      # Marginal distributions
python analysis_code/visualize_marginal_distribution_quantile_context_only.py # Context-only marginals
```

**Core Training and Evaluation:**
```bash
python train_quantile_models.py            # Train quantile regression models
python generate_quantile_surfaces.py       # Generate quantile predictions
python evaluate_quantile_ci_calibration.py # Evaluate CI calibration
python compare_reconstruction_losses.py    # Compare losses across models
```

**Test Autoregressive Generation:**
```bash
python test_autoregressive_generation.py   # Test multi-step generation (all 3 variants)
```
Tests: shapes, quantile ordering, different horizons, visual checks, ex_feats coherence.

**Multi-Horizon Training (Experimental):**
```bash
python train_horizon5_test.py              # Train horizon=5 model (validation)
python compare_horizon5_to_baseline.py     # Compare horizon=5 vs baseline
python visualize_horizon5_success.py       # 9-panel comparison visualization
python visualize_improvement_heatmap.py    # Grid-wise improvement analysis
```
Validates multi-horizon training: horizon=5 shows 43-54% RMSE improvement and 80% better CI calibration vs autoregressive baseline.

**Data Preprocessing:**
- Data preprocessing requires Jupyter notebooks (not included in main codebase)
- See `data_preproc/` directory for preprocessing utilities

## Architecture

### Core VAE Models (vae/)

**Base Classes (vae/base.py):**
- `BaseVAE`: Abstract base class for all VAE models
  - Implements standard VAE loss: `loss = reconstruction_error + kl_weight * kl_loss`
  - Provides `train_step()`, `test_step()`, `save_weights()`, `load_weights()`
- `BaseEncoder` / `BaseDecoder`: Abstract encoder/decoder interfaces

**Model Hierarchy:**
1. `VAEConv2D` (vae/conv_vae.py): Basic 2D convolutional VAE
2. `CVAE` (vae/cvae.py): Conditional VAE with context encoder
   - Input: (B, T, H, W) where T = context_len + 1
   - Uses separate context encoder for conditioning
3. `CVAEMem` (vae/cvae_with_mem.py): CVAE with LSTM/GRU/RNN memory
4. **`CVAEMemRand` (vae/cvae_with_mem_randomized.py)**: Primary model used in paper
   - Supports variable context lengths (randomized during training)
   - Supports configurable prediction horizon (default: 1 day, validated up to 5 days)
   - Can optionally encode extra features (returns, skew, slope)

### Key Model Features (CVAEMemRand)

**Configuration Parameters:**
- `feat_dim`: Volatility surface dimensions (typically (5, 5))
- `latent_dim`: Latent space dimensionality
- `horizon`: Number of days to predict (default: 1, configurable for multi-horizon training)
- `surface_hidden`: Hidden layer sizes for surface encoding
- `ex_feats_dim`: Number of extra features (0 for surface only, 3 for ret/skew/slope)
- `mem_type`: Memory type (lstm/gru/rnn)
- `mem_hidden`: Hidden size for memory module
- `mem_layers`: Number of memory layers
- `compress_context`: Whether to compress context to latent_dim size
- `use_dense_surface`: Use fully connected layers instead of Conv2D

**Training Data Format:**
Input is a dictionary with keys:
- `"surface"`: Tensor of shape (B, T, 5, 5) - volatility surfaces
- `"ex_feats"`: (Optional) Tensor of shape (B, T, n) - extra features

**Model Components:**
1. **CVAEMemRandEncoder**: Encodes full sequence (context + target)
   - Surface embedding → Optional ex_feats embedding → Concatenate → LSTM → Latent space
2. **CVAECtxMemRandEncoder**: Encodes context only
   - Similar architecture but for conditioning
3. **CVAEMemRandDecoder**: Decodes latent + context to surface
   - LSTM → Split into surface and ex_feats decoders

### Dataset Handling (vae/datasets_randomized.py)

**VolSurfaceDataSetRand:**
- Generates variable-length sequences (min_seq_len to max_seq_len)
- Each data point consists of T consecutive surfaces
- Uses `CustomBatchSampler` to ensure all samples in a batch have the same sequence length

### Training Utilities (vae/utils.py)

Key functions for training, testing, and evaluation:
- `train()`: Training loop with early stopping
- `test()`: Evaluation on validation/test sets
- `set_seeds()`: Set random seeds for reproducibility

## Data Structure

### Input Data Format
Volatility surfaces are stored as numpy arrays:
- `surface`: (N, 5, 5) - N days of 5x5 volatility grids (moneyness × time to maturity)
- `ret`: (N,) - Daily returns
- `skews`: (N,) - Volatility skew measure
- `slopes`: (N,) - Volatility term structure slope
- `ex_data`: (N, 3) - Concatenated [ret, skew, slope]

Data files:
- `data/vol_surface_with_ret.npz`: Main training data
- `data/spx_vol_surface_history_full_data_fixed.parquet`: Full SPX history

### Model Outputs
Generated surfaces stored in npz files:
- `surfaces`: (num_days, num_samples, 5, 5)
- `ex_feats`: (num_days, num_samples, 3) - if model returns extra features

## Analysis Pipeline (analysis_code/)

**Key Analysis Modules:**
- `regression.py`: Regression analysis of latent embeddings vs market indicators
- `latent_pca.py`: PCA analysis of latent space
- `arbitrage.py`: Check for arbitrage-free properties in generated surfaces
- `loss_table.py`: Format and generate loss comparison tables

**Main Analysis Workflow (main_analysis.py):**
1. Load trained models and generate surfaces
2. Compute regression analysis (surface grid accuracy, RMSE benchmarks)
3. Generate latent embeddings and perform PCA
4. Run classification tasks (NBER recession prediction)
5. Check arbitrage violations
6. Generate LaTeX tables and plots

## Model Variants

The codebase trains three model variants to test different conditioning strategies. These variants explore a fundamental hypothesis: **Can we improve volatility surface forecasts by jointly modeling returns and surfaces, or is surface-only modeling sufficient?**

### 1. No EX (Surface Only - Baseline)

**Configuration:**
```python
ex_feats_dim: 0                # No extra features
re_feat_weight: 0.0            # No feature loss
```

**Input:** Only volatility surfaces (5×5 grids)
- Context: Last C days of surfaces
- No additional market information

**Architecture:**
```
Surface[t-C:t] → LSTM Encoder → Latent z → LSTM Decoder → Surface[t+1]
```

**What it learns:**
- Pure surface-to-surface mapping
- Pattern recognition in volatility surface evolution
- No explicit conditioning on returns/skew/slope

**Use case:** Baseline model to test if surface dynamics alone contain sufficient information for forecasting.

---

### 2. EX No Loss (Features as Passive Conditioning)

**Configuration:**
```python
ex_feats_dim: 3                # Has 3 extra features (return, skew, slope)
re_feat_weight: 0.0            # NO loss on features!
```

**Input:** Surfaces + Extra features
- Surface: 5×5 volatility grids
- Extra features: [daily log return, volatility skew, term structure slope]

**Architecture:**
```
Surface[t-C:t] + Features[t-C:t] → LSTM Encoder → Latent z → Decoder → Surface[t+1]
                                                                        (Features ignored in loss)
```

**What it learns:**
- Uses extra features as **conditioning information** only
- Model sees the features during encoding but isn't trained to reconstruct them
- Features help encoder understand market state (e.g., "market crashed yesterday")
- Decoder can use this context to generate better surfaces

**Key insight:** Tests whether passively providing market context improves predictions without requiring the model to learn feature dynamics.

---

### 3. EX Loss (Features with Joint Optimization)

**Configuration:**
```python
ex_feats_dim: 3                # Has 3 extra features
re_feat_weight: 1.0            # YES - optimize feature reconstruction!
ex_loss_on_ret_only: True      # Only optimize returns (not skew/slope)
ex_feats_loss_type: "l2"       # L2 loss for return predictions
```

**Input:** Same as EX No Loss (surfaces + features)

**Architecture:**
```
Surface[t-C:t] + Features[t-C:t] → LSTM Encoder → Latent z → LSTM Decoder → Surface[t+1]
                                                                           → Return[t+1]
                                                                           (Both optimized!)
```

**Loss function:**
```
Total Loss = MSE(surface) + re_feat_weight × L2(return) + kl_weight × KL_divergence
           = MSE(surface) + 1.0 × L2(return) + 1e-5 × KL
```

**What it learns:**
- Jointly optimizes surface AND return prediction
- Multi-task learning forces latent space to capture return dynamics
- Model must generate coherent predictions across both modalities
- Creates shared representation that understands both surface shape and return magnitudes

**Key insight:** Tests whether forcing the model to predict returns improves surface forecasts by creating better latent representations.

---

### Model Comparison Summary

| Aspect | No EX | EX No Loss | EX Loss |
|--------|-------|------------|---------|
| **Input features** | Surface only | Surface + return/skew/slope | Surface + return/skew/slope |
| **Encoder sees extras?** | ❌ No | ✅ Yes (conditioning) | ✅ Yes (conditioning) |
| **Decoder outputs extras?** | ❌ No | ✅ Yes (ignored) | ✅ Yes (optimized) |
| **Loss on features** | 0.0 | 0.0 | 1.0 (returns only) |
| **Learning objective** | Surface only | Surface (conditioned on features) | Surface + Returns (joint learning) |
| **Training data** | `train_simple` | `train_ex` | `train_ex` |

### Test Set Performance (from param_search.py results)

| Model | Surface Reconstruction Loss | Feature Reconstruction Loss | KL Loss |
|-------|------------------------------|----------------------------|---------|
| **No EX** | 0.001722 | 0.000000 | 3.956 |
| **EX No Loss** | 0.002503 | 0.924496 (not optimized) | 3.711 |
| **EX Loss** | 0.001899 | 0.000161 | 4.000 |

**Observations:**
- **No EX** achieves lowest surface reconstruction error (baseline)
- **EX No Loss** has highest surface error - features help conditioning but high feature loss suggests model doesn't learn feature structure well
- **EX Loss** balances both objectives - competitive surface error with excellent return prediction (0.000161)

### Research Questions Addressed

1. **No EX vs EX No Loss**: Does passive conditioning on market features improve predictions?
   - Tests information benefit without optimization overhead

2. **EX No Loss vs EX Loss**: Should we actively optimize feature predictions?
   - Tests multi-task learning hypothesis
   - Does forcing coherent return predictions improve surface quality?

3. **Practical implications**:
   - **No EX**: Simplest, but ignores market context (e.g., doesn't know if crash occurred)
   - **EX No Loss**: Uses context but may not fully leverage feature information
   - **EX Loss**: Joint prediction may create more robust latent representations

### Generation Outputs

All three models generate two types of forecasts:

**Stochastic (probabilistic):**
- `{model}_gen5.npz`: 1,000 samples per day by sampling z ~ N(0, 1)
- Shape: (num_days, 1000, 5, 5) surfaces
- EX Loss also outputs: (num_days, 1000, 3) features [return, skew, slope]

**Maximum Likelihood (deterministic):**
- `{model}_mle_gen5.npz`: 1 sample per day using z = 0 (distribution mode)
- Shape: (num_days, 1, 5, 5) surfaces
- EX Loss also outputs: (num_days, 1, 3) features

### Visualization

Visualization scripts are in `analysis_code/`:
```bash
python analysis_code/visualize_teacher_forcing.py
```

Generates 9-panel comparison (3 models × 3 grid points) showing ground truth vs predictions with uncertainty bands. Demonstrates teacher forcing behavior where models condition on real historical data for independent one-step-ahead forecasts.

## Quantile Regression Decoder (Recent Development)

A quantile regression variant has been implemented to address CI calibration issues in uncertainty quantification. This represents an architectural enhancement to directly predict confidence intervals rather than computing them from Monte Carlo samples.

### Motivation

Original MSE models showed CI violation rates of 35-72%, far exceeding the target ~10%. The quantile decoder explicitly learns conditional quantiles during training to improve calibration and provides dramatically faster inference.

### Architecture Changes

**Decoder Output:**
- **Original**: 1 channel (mean prediction)
- **Quantile**: 3 channels (p5, p50, p95 quantiles)
- Output shape: `(B, T, 3, H, W)` where dim=2 represents quantiles

**Loss Function:**
- **Original**: MSE (Mean Squared Error)
- **Quantile**: Pinball Loss (asymmetric quantile loss)
- For τ=0.05: Over-predictions penalized 19× more → learns conservative lower bound
- For τ=0.95: Under-predictions penalized 19× more → learns conservative upper bound

**Pinball Loss Formula:**
```python
def pinball_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return torch.where(error >= 0, quantile * error, (quantile - 1) * error)
```

**Generation Speedup:**
- **Original**: 1,000 forward passes with z ~ N(0,1) → compute empirical quantiles
- **Quantile**: 1 forward pass → get all 3 quantiles directly
- **Result**: ~1000× faster generation

### Training Configuration

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

### Current Performance

**Training Performance (Test Set):**

| Model | Surface RE Loss | KL Loss | Total Loss |
|-------|-----------------|---------|------------|
| **no_ex** | 0.006349 | 6.033 | 0.006409 |
| **ex_no_loss** | 0.006269 | 7.768 | 0.006347 |
| **ex_loss** | 0.006206 | 5.876 | 0.006392 |

Note: RE loss is pinball loss, not directly comparable to MSE.

**CI Calibration Results:**

| Model | CI Violations | Below p05 | Above p95 | Mean CI Width |
|-------|---------------|-----------|-----------|---------------|
| **no_ex** | 44.50% | 4.79% | 39.71% | 0.0811 |
| **ex_no_loss** | 35.43% | 17.82% | 17.60% | 0.0855 |
| **ex_loss** | 34.28% | 5.55% | 28.72% | 0.0892 |

- **Expected**: ~10% violations (well-calibrated)
- **Actual**: 34-45% violations (improvement from baseline ~50%+ but needs further work)
- **Best model**: ex_loss with 34.28% violations

### Key Observations

1. **Improvement over baseline**: Reduced violations from ~50%+ to ~34% (ex_loss model)
2. **Asymmetric violations**: Most models have more violations above p95 than below p05
3. **Generation speed**: Dramatically faster (1000× speedup)
4. **Trade-off**: Better calibration vs more complex loss function

### Usage

**Training quantile models:**
```bash
python train_quantile_models.py
```

**Generating quantile surfaces:**
```bash
python generate_quantile_surfaces.py
```

**Evaluating CI calibration:**
```bash
python evaluate_quantile_ci_calibration.py
```

**Verification Scripts (in analysis_code/):**
```bash
# Ground truth latent (full sequence encoding)
python analysis_code/verify_reconstruction_plotly_2008_2010.py

# Context-only latent (realistic generation)
python analysis_code/verify_reconstruction_2008_2010_context_only.py

# Marginal distribution analysis
python analysis_code/visualize_marginal_distribution_quantile_encoded.py
python analysis_code/visualize_marginal_distribution_quantile_context_only.py
```

**Verification Results (2008-2010 CI Violations):**

| Latent Type | no_ex | ex_no_loss | ex_loss |
|-------------|-------|------------|---------|
| **Ground truth** | 7.32% | 5.33% | 6.89% |
| **Context-only** | 18.63% | 20.44% | 19.65% |

The 3× gap reveals **VAE prior mismatch**: p(z|context) ≠ p(z|context+target). Decoder learns correct quantiles when given encoded latents, but prior distribution doesn't match posterior for realistic generation.

### Model Output Format

Quantile models output 3 surfaces per prediction:
- `surfaces[:, :, 0, :, :]` - p05 (5th percentile, lower bound)
- `surfaces[:, :, 1, :, :]` - p50 (median, point forecast)
- `surfaces[:, :, 2, :, :]` - p95 (95th percentile, upper bound)

For EX Loss models, also outputs quantiles for extra features:
- `ex_feats[:, :, 0, :]` - p05 features [return, skew, slope]
- `ex_feats[:, :, 1, :]` - p50 features
- `ex_feats[:, :, 2, :]` - p95 features

### Current Status & Next Steps

**Status**: ✓ Implementation successful, ✗ Calibration needs improvement

**Completed:**
- Stable training with pinball loss
- 1000× generation speedup
- ~16% reduction in CI violations vs baseline

**Remaining challenges:**
- CI violations at 34% (target: 10%)
- Models underestimate upper tail uncertainty
- Need calibration techniques (e.g., conformal prediction)

**Recommended next steps:**
1. Apply conformal prediction for post-hoc calibration
2. Retrain with loss reweighting (emphasize tail quantiles)
3. Explore heteroscedastic quantile regression

## Multi-Horizon Training

Models support training with `horizon > 1` to predict multiple days simultaneously, avoiding error accumulation in autoregressive generation.

**Configuration:**
```python
model_config = {
    "horizon": 5,  # Predict 5 days ahead in one shot
    # ... other params same as horizon=1
}
```

**Validation Results (horizon=5 vs baseline):**
- **RMSE improvement**: 43-54% reduction across volatility surface
- **CI calibration**: 80% reduction in violations (89% → 18%)
- **Training time**: ~3.7× longer per epoch (acceptable given improvements)

**Latent Sampling Strategies:**
Three approaches for generating future latents documented in `LATENT_SAMPLING_STRATEGIES.md`:
1. **Ground truth latent** (~7% violations): Oracle reconstruction, testing only
2. **VAE prior sampling** (~19% violations): Theoretically correct, sample z ~ N(0,1)
3. **Zero-padding encoding** (~18% violations): Current implementation, encodes zero-padded future

All three are valid for backfilling (don't use future information). Current implementation uses zero-padding where LSTM propagates context through hidden states.

## Important Implementation Details

### Loss Function

**Standard VAE Models:**
- Reconstruction error uses MSE for surfaces
- KL divergence term weighted by `kl_weight` (typically 1e-5)
- Extra features use L1 or L2 loss weighted by `re_feat_weight`
- When `ex_loss_on_ret_only=True`, only return (first feature) gets loss optimization

**Quantile Regression Models:**
- Reconstruction error uses Pinball Loss (quantile loss) instead of MSE
- Pinball loss computed separately for each quantile (p05, p50, p95)
- KL divergence term still weighted by `kl_weight`
- Extra features (if used) also use pinball loss for quantile prediction
- Total loss: `loss = pinball_loss + kl_weight * kl_loss`

### Sequence Generation and Teacher Forcing

**Training:**
- Context length `C` is variable during training (randomized between min_seq_len and max_seq_len)
- Model learns to predict 1-day-ahead: given surfaces at [t-C, ..., t-1], predict surface at t

**Generation (Inference):**
- Uses **teacher forcing**: always conditions on real historical data, not model predictions
- For each day t, uses actual observed surfaces [t-C, ..., t-1] as context
- Generates prediction for day t
- Next day uses actual surfaces [t-C+1, ..., t] (including real observation at t)
- This produces **independent one-step-ahead forecasts**, not autoregressive multi-step predictions

**Two Generation Modes:**

1. **Stochastic (`generate_surfaces.py`):**
   - Samples latent variable z ~ N(0, 1) for future timestep
   - Generates 1,000 samples per day to capture uncertainty
   - Output: (num_days, 1000, 5, 5) - full distribution of possible surfaces
   - Used for: uncertainty quantification, arbitrage checking, scenario analysis

2. **Maximum Likelihood (`generate_surfaces_max_likelihood.py`):**
   - Sets latent variable z = 0 (mode of prior distribution)
   - Generates 1 deterministic sample per day
   - Output: (num_days, 1, 5, 5) - single "most likely" surface
   - Used for: point forecasts, RMSE evaluation, regression analysis

**Key Implementation Detail:**
Both methods use the same context encoding (from `ctx_encoder`), but differ only in the latent variable for the future timestep:
- Context timesteps [0:C]: z = ctx_latent_mean (deterministic encoding)
- Future timestep [C]: z ~ N(0,1) (stochastic) OR z = 0 (MLE)

### Device Handling
All models support both CPU and CUDA. Tensors are automatically moved to the configured device.

## Autoregressive Backfilling (Crisis Period Generation)

The codebase includes capability to generate 30-day autoregressive sequences for historical periods with limited data (e.g., 2008-2010 financial crisis). Implementation follows `BACKFILL_MVP_PLAN.md`.

**Key Features:**
- **Multi-horizon training**: Model trained on [1, 7, 14, 30] day horizons simultaneously
- **Scheduled sampling**: 2-phase training (teacher forcing → multi-horizon)
- **Limited data**: Trains on 1-3 years of recent data, generates for historical crisis
- **Autoregressive generation**: 30-day sequences by feeding predictions back as context

**Core Methods (in CVAEMemRand):**
- `train_step_multihorizon()` - Train on multiple horizons with weighted loss
- `generate_autoregressive_sequence()` - Generate multi-day sequences autoregressively

**Configuration and Training:**
```bash
# Configure training parameters
# Edit config/backfill_config.py:
#   - train_period_years: 1, 2, or 3 years
#   - context_len: 5, 10, 20, or 30 days
#   - training_horizons: [1, 7, 14, 30]

# Train model with scheduled sampling
python train_backfill_model.py
# Output: models_backfill/backfill_3yr.pt

# Generate 30-day backfill sequences
python generate_backfill_sequences.py  # (Phase 4 - to be implemented)
# Output: models_backfill/backfill_predictions_3yr.npz
```

**Configuration (config/backfill_config.py):**
- `train_period_years`: Years of training data (1, 2, or 3)
- `train_end_idx`: 5000 (before test set)
- `backfill_start_idx`: 2000 (2008 crisis start)
- `backfill_end_idx`: 2765 (2010 end)
- `context_len`: Initial context window (5 days default, can increase to 20-30)
- `training_horizons`: [1, 7, 14, 30] days
- `teacher_forcing_epochs`: 200 (Phase 1), then multi-horizon for remaining epochs

**Context Length Ablation:**
If context_len=5 performs poorly, try longer contexts (10, 20, 30) - see Phase 3.3 in `BACKFILL_MVP_PLAN.md`.

**Validation Scripts:**
```bash
python test_multihorizon_loss.py       # Validate multi-horizon training
python test_scheduled_sampling.py      # Validate 2-phase training
python test_phase3_config.py           # Validate config + full pipeline
```

### Evaluation & Analysis (backfill_16yr)

**Out-of-Sample Evaluation:**
```bash
# Generate test set predictions (2019-2023)
python test_oos_reconstruction_16yr.py
# Output: models_backfill/oos_reconstruction_16yr.npz

# In-sample predictions already in: models_backfill/insample_reconstruction_16yr.npz
```

**VAE Health Analysis:**
```bash
# In-sample (training set 2004-2019)
python analyze_vae_health_16yr.py          # Extract latent metrics
python visualize_vae_health_16yr.py        # Generate 11 figures
# Output: models_backfill/vae_health_16yr.npz, models_backfill/vae_health_figs/

# Out-of-sample (test set 2019-2023)
python analyze_vae_health_oos_16yr.py      # Extract latent metrics
python visualize_vae_health_oos_16yr.py    # Generate 9 figures
# Output: models_backfill/vae_health_oos_16yr.npz, models_backfill/vae_health_figs_oos/
```

**Interactive Visualizations:**
```bash
# Teacher forcing dashboards (12-panel: 3 grid points × 4 horizons)
python analysis_code/visualize_backfill_16yr_plotly.py      # In-sample
python analysis_code/visualize_backfill_oos_16yr_plotly.py  # Out-of-sample
# Output: tables/backfill_plots/*.html (open in browser)
```

**Key Findings (backfill_16yr):**
- In-sample: 18.1% CI violations (moderate)
- Out-of-sample: 28.0% CI violations (+55% degradation)
- VAE architecture healthy (effective dim ~3/5, consistent collapse pattern)
- RMSE increases 57-92% across horizons OOS
- See: `tables/backfill_plots/insample_vs_oos_comparison_16yr.md`

## Common Development Patterns

**Loading a Trained Model:**
```python
model_data = torch.load("path/to/model.pt")
model_config = model_data["model_config"]
model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
```

**Generating Surfaces (Standard Models):**
```python
ctx_data = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}
generated_surface = model.get_surface_given_conditions(ctx_data)
# Returns: (B, 1, 5, 5) surface for next day
```

**Generating Surfaces (Quantile Models):**
```python
ctx_data = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}
generated_surfaces = model.get_surface_given_conditions(ctx_data)
# Returns: (B, 1, 3, 5, 5) - 3 quantile surfaces for next day
# generated_surfaces[:, :, 0, :, :] - p05 (lower bound)
# generated_surfaces[:, :, 1, :, :] - p50 (median)
# generated_surfaces[:, :, 2, :, :] - p95 (upper bound)
```

**Autoregressive Multi-Step Generation:**
```python
# Generate 30-day sequence by feeding predictions back as context
initial_context = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}

with torch.no_grad():
    result = model.generate_autoregressive_sequence(
        initial_context=initial_context,
        horizon=30
    )

# For surface-only models (no_ex):
# result = (B, 30, 3, 5, 5) - 3 quantiles × 30 days

# For ex_feats models (ex_no_loss, ex_loss):
# result = (surfaces, ex_feats) tuple
# surfaces: (B, 30, 3, 5, 5)
# ex_feats: (B, 30, 3)

# Extract quantiles across all 30 days
p05 = result[:, :, 0, :, :]  # (B, 30, 5, 5) - lower bound
p50 = result[:, :, 1, :, :]  # (B, 30, 5, 5) - median forecast
p95 = result[:, :, 2, :, :]  # (B, 30, 5, 5) - upper bound
```

**Key details:**
- Uses p50 (median) as point estimate for context updates
- Sliding window: drops oldest, appends new prediction each step
- Supports all 3 variants without modification
- **Autoregressive** (error accumulation): For better performance, use multi-horizon training (see Multi-Horizon Training section)
- Models trained with `horizon > 1` predict multiple days simultaneously without autoregressive error accumulation

**Data Preprocessing:**
Data should be downloaded from WRDS (OptionMetrics Ivy DB) and preprocessed using notebooks in the project root. The preprocessing pipeline generates 5×5 interpolated volatility surface grids from option prices.

**WRDS Data Download Instructions:**
1. OptionMetrics/Ivy DB US/Options/Option Prices
2. Date Range: 2000-01-01 to 2023-02-28
3. SECID = 108105 (S&P 500)
4. Option Type: Both, Exercise Type: Both, Security Type: Both
5. Query Variables: all
6. Output Format: *.csv, Compression: *.zip, Date Format: YYYY-MM-DD
7. Save as `data/spx.zip`

**Additional Data Required:**
- S&P 500 stock prices from Yahoo Finance (ticker: `^GSPC`)
- Save as `data/GSPC.csv`

**Preprocessing Workflow:**
1. Run `spx_volsurface_generation.ipynb`: Cleans raw data and generates interpolated IVS dataframe
2. Run `spx_convert_to_grid.ipynb`: Converts dataframe to 5×5 numpy grids
3. Output files:
   - `data/vol_surface_with_ret.npz` - Main training data
   - `data/spx_vol_surface_history_full_data_fixed.parquet` - Full SPX history

**Pre-trained Models:**
Models and parsed data available at: https://drive.google.com/drive/folders/1W3KsAJ0YQzK2qnk0c-OIgj26oCAO3NI1?usp=sharing
