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

### Branch Structure
- `main`: Stable baseline models with N(0,1) sampling
- `max`: Development branch
- `quantile-regression-decoder`: Alternative approach using quantile regression for CI prediction
- `empirical-latent-sampling`: Experiments with improved CI calibration via empirical latent sampling

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

**Visualize Teacher Forcing Performance:**
```bash
python visualize_teacher_forcing.py
```
Creates visualizations comparing all 3 models across different grid points, showing ground truth vs predictions with uncertainty bands. Generates:
- `teacher_forcing_implied_vol.png` - 9-panel comparison (3 models × 3 grid points)
- `teacher_forcing_returns.png` - Return predictions for EX Loss model

**Empirical Latent Sampling (CI Calibration Experiments):**
```bash
# Step 1: Extract empirical latent distributions from training data
python extract_empirical_latents.py

# Step 2: Generate surfaces using empirical latents with custom noise and day ranges
python generate_surfaces_empirical.py --noise-scale 0.3 --start-day 5 --days-to-generate 5810

# Step 3: Evaluate CI calibration
python evaluate_empirical_ci.py --noise-scale 0.3

# Step 4: Batch test multiple noise values
python batch_test_noise_scales.py  # Tests noise=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# Step 5: Analyze noise sweep results
python analyze_noise_sweep.py

# Step 6: Compare sampling methods
python compare_sampling_methods.py
```

**Distribution Analysis:**
```bash
# Analyze latent distributions
python analyze_latent_distribution.py

# Analyze train/val/test distribution shift
python analyze_distribution_shift.py

# Visualize marginal distributions
python visualize_marginal_distributions.py
```

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
   - Generates 1 day forward predictions
   - Can optionally encode extra features (returns, skew, slope)

### Key Model Features (CVAEMemRand)

**Configuration Parameters:**
- `feat_dim`: Volatility surface dimensions (typically (5, 5))
- `latent_dim`: Latent space dimensionality (typically 5)
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
- `data/vol_surface_with_ret.npz`: Main training data (5822 days)
- `data/spx_vol_surface_history_full_data_fixed.parquet`: Full SPX history

### Data Splits
```
Training:   days 0-3999     (2000-01-03 to 2015-11-27)
Validation: days 4000-4999  (2015-11-30 to 2019-11-18)
Test:       days 5000-5821  (2019-11-19 to 2023-02-14)
```

**Important**: Evaluation typically uses last 1000 days (days 4822-5821) which spans validation + test sets.

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

Run `visualize_teacher_forcing.py` to compare model performance:
```bash
python visualize_teacher_forcing.py
```

Generates:
- `teacher_forcing_implied_vol.png`: 9-panel comparison (3 models × 3 grid points)
- `teacher_forcing_returns.png`: Return predictions for EX Loss model

Shows how teacher forcing works (models always conditioned on real historical data) and performance differences across conditioning strategies.

## Important Implementation Details

### Loss Function
- Reconstruction error uses MSE for surfaces
- KL divergence term weighted by `kl_weight` (typically 1e-5)
- Extra features use L1 or L2 loss weighted by `re_feat_weight`
- When `ex_loss_on_ret_only=True`, only return (first feature) gets loss optimization

### Sequence Generation and Teacher Forcing

**Training:**
- Context length `C` is variable during training (randomized between min_seq_len and max_seq_len)
- Model learns to predict 1-day-ahead: given surfaces at [t-C, ..., t-1], predict surface at t
- **Critical**: Encoder sees the actual day t surface and encodes it into latent z_t during training

**Generation (Inference):**
- Uses **teacher forcing**: always conditions on real historical data, not model predictions
- For each day t, uses actual observed surfaces [t-C, ..., t-1] as context
- Generates prediction for day t
- Next day uses actual surfaces [t-C+1, ..., t] (including real observation at t)
- This produces **independent one-step-ahead forecasts**, not autoregressive multi-step predictions

**Three Generation Modes:**

1. **Stochastic N(0,1) (`generate_surfaces.py`):**
   - Samples latent variable z ~ N(0, 1) for future timestep
   - Generates 1,000 samples per day to capture uncertainty
   - Output: (num_days, 1000, 5, 5) - full distribution of possible surfaces
   - Used for: uncertainty quantification, arbitrage checking, scenario analysis
   - **Issue**: CI violation rates 35-72% (target: 10%) due to N(0,1) not matching empirical latent distributions

2. **Empirical Latent Sampling (`generate_surfaces_empirical.py`):**
   - Samples latent variable z from empirical training latent pool + optional noise
   - Formula: `z = z_empirical[random_idx] + noise_scale * N(0,1)`
   - Tunable parameters: `--noise-scale` (default 0.3), `--start-day`, `--days-to-generate`
   - Generates 1,000 samples per day
   - Output: (num_days, 1000, 5, 5) with filename `{model}_empirical_gen5_noise{noise_scale}_days{start}-{end}.npz`
   - Used for: CI calibration experiments, exploring noise parameter space

3. **Maximum Likelihood (`generate_surfaces_max_likelihood.py`):**
   - Sets latent variable z = 0 (mode of prior distribution)
   - Generates 1 deterministic sample per day
   - Output: (num_days, 1, 5, 5) - single "most likely" surface
   - Used for: point forecasts, RMSE evaluation, regression analysis

**Key Implementation Detail:**
All methods use the same context encoding (from `ctx_encoder`), but differ in the latent variable for the future timestep:
- Context timesteps [0:C]: z = ctx_latent_mean (deterministic encoding)
- Future timestep [C]:
  - N(0,1) baseline: z ~ N(0,1)
  - Empirical: z ~ Empirical(z_train) + noise
  - MLE: z = 0

### Device Handling
All models support both CPU and CUDA. Tensors are automatically moved to the configured device.

## Common Development Patterns

**Loading a Trained Model:**
```python
model_data = torch.load("path/to/model.pt")
model_config = model_data["model_config"]
model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
```

**Generating Surfaces:**
```python
ctx_data = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}
generated_surface = model.get_surface_given_conditions(ctx_data)
# Returns: (B, 1, 5, 5) surface for next day
```

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

## Confidence Interval Calibration Issues

### CRITICAL FINDING: Structural Mismatch Between Training and Generation

The models exhibit **severe CI miscalibration** with a counter-intuitive pattern: **in-distribution calibration is WORSE than out-of-distribution**. This reveals a fundamental structural issue with how teacher forcing VAEs handle uncertainty quantification.

### Problem Summary

**Baseline N(0,1) Sampling CI Violations (target: 10%):**

| Evaluation Set | No EX | EX No Loss | EX Loss | Average |
|---------------|-------|------------|---------|---------|
| **OOD** (days 4822-5821, test set) | 18-35% | 54-72% | 50-65% | **54%** |
| **In-Dist** (days 1000-1059, training) | 88-100% | 68-100% | 88-100% | **91%** |

**With Optimal Noise (2.0):**

| Evaluation Set | Average Violations | Distance from Target |
|---------------|-------------------|---------------------|
| **OOD** (test set) | 37% | 27% above target |
| **In-Dist** (training) | 83% | **73% above target** |

**Key Observation**: In-distribution calibration is **46 percentage points worse** than out-of-distribution at noise=2.0. This is the opposite of what distribution shift alone would predict.

### Root Cause: Teacher Forcing Structural Mismatch

**The Problem:**

**During Training:**
- Encoder sees the **actual next-day surface** at timestep t and encodes it into latent z_t
- Model learns: p(surface_t | context, z_t_true) where z_t_true comes from encoding the actual future
- Has privileged information: sees the future before predicting it

**During Generation:**
- Model only has context [t-5...t-1], no access to actual day t
- Must **randomly sample** z_t from N(0,1) or empirical latents
- Sampled z_t is uncorrelated with the actual future that will occur
- Model must use: p(surface_t | context, z_t_sampled)

**Why This Breaks:**
The model learned conditional distributions that expect correlated latent variables. During generation, sampling violates these learned correlations, leading to:
- **Systematic bias** on training data where model memorized context-latent patterns
- **100% violations at ATM 3-Month** grid point (all predictions above actual values)
- **Better OOD performance** because model is already uncertain about unfamiliar patterns

### Evidence from Experiments

**1. Systematic Bias at ATM 3-Month:**
- All three models show 98-100% CI violations on in-distribution data
- 100% of actual values fall BELOW the 5th percentile
- Model systematically overestimates volatility at this grid point
- Even noise=2.0 barely helps (98-100% violations persist)

**2. Distribution Shift Analysis:**
- Training period (2000-2015): mean vol = 0.1937
- Validation period (2015-2019): mean vol = 0.1446 (unusually calm)
- Test period (2019-2023): mean vol = 0.2162 (includes COVID spike)
- **But**: Negative correlation (-0.08) between distribution shift magnitude and CI violations
- Grid points with larger shifts have BETTER calibration, not worse

**3. Noise Parameter Sweep:**
- Tested noise scales: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
- OOD improvement (noise 0.3→2.0): 54% → 37% (-17 points)
- In-dist improvement (noise 0.3→2.0): 91% → 83% (-8 points)
- Noise is **twice as effective** on OOD data

**4. In-Distribution vs OOD Results:**

See `in_distribution_vs_ood_comparison.md` for detailed analysis.

| Metric | In-Dist (training) | OOD (test) |
|--------|-------------------|------------|
| Avg violations @ noise=0.3 | 91.11% | 54.00% |
| Avg violations @ noise=2.0 | 83.15% | 37.12% |
| ATM 3M violations @ noise=2.0 | 98-100% | 3.7-20.2% |
| Best case | 53% (ex_no_loss OTM Put) | 3.7% (no_ex ATM 3M) |

### Implementation Files

**Empirical Latent Sampling:**
- `extract_empirical_latents.py`: Extract latent distributions from training data
- `generate_surfaces_empirical.py`: Generate with empirical sampling (supports custom day ranges)
- `evaluate_empirical_ci.py`: Evaluate CI calibration (supports custom evaluation windows)
- `batch_test_noise_scales.py`: Automated testing of multiple noise values

**Analysis:**
- `analyze_latent_distribution.py`: Compare empirical latents vs N(0,1)
- `analyze_noise_sweep.py`: Analyze results across noise parameter space
- `analyze_distribution_shift.py`: Statistical analysis of train/val/test distribution differences
- `compare_sampling_methods.py`: Visualize baseline vs empirical sampling results

**Documentation:**
- `CI_CALIBRATION_OBSERVATIONS.md`: Initial baseline CI miscalibration findings
- `in_distribution_vs_ood_comparison.md`: Detailed comparison of in-dist vs OOD calibration
- `noise_sweep_results.csv`: Violation rates for all models/grid points/noise values

### Recommendations

**❌ DO NOT USE for Uncertainty Quantification:**
- Confidence intervals are severely miscalibrated (37-91% violations vs 10% target)
- Particularly unreliable on training data (backfilling)
- Systematic bias at certain grid points (ATM 3-Month overestimates by ~1 std)

**✓ CAN USE for Point Forecasts:**
- Teacher forcing works well for deterministic predictions (MLE generation)
- Surface reconstruction losses are low (0.0017-0.0025)
- Good performance for relative comparisons and scenario analysis

**For Improved Calibration:**
1. **Remove teacher forcing during training** - force model to generate without seeing future latents
2. **Use variational dropout** - inject noise during training to match generation conditions
3. **Post-calibration** - learn empirical correction factors from validation set
4. **Alternative approaches** - conformal prediction, quantile regression, ensemble methods

**If Forced to Use Current Model:**
- Use `noise_scale=2.0` for best (but still poor) calibration
- Acknowledge 37% violation rate (vs 10% target) in any analysis
- Consider using wider confidence levels (95% or 99%) to compensate
- Focus on qualitative insights rather than precise probability statements

### Key Takeaways

1. **Primary issue is NOT distribution shift** - it's a structural mismatch between training and generation
2. **Empirical latent sampling helps** but doesn't solve the fundamental problem
3. **Noise scaling provides marginal improvement** - optimal at 2.0 but still 27% above target
4. **All three model variants** (no_ex, ex_no_loss, ex_loss) show similar miscalibration patterns
5. **Teacher forcing is excellent for point forecasts** but incompatible with uncertainty quantification in this architecture

### Posterior Collapse: Root Cause Analysis

**NEW FINDING (2025)**: Comprehensive verification tests confirm **severe decoder collapse** as the root cause of CI miscalibration. Even when using exact training latent distributions, CIs remain severely miscalibrated (84% violations vs 10% target).

**Verification Scripts:**
```bash
# Extract empirical latents (includes z_mean AND z_log_var)
python extract_empirical_latents.py

# Visualize uncertainty over time (check if model learned higher variance during crises)
python visualize_uncertainty_over_time.py

# Test with EXACT training latents (2008-2010)
python visualize_exact_training_latents.py

# Residual analysis (quantify decoder underestimation)
python analyze_residuals_exact_latents.py

# Comprehensive posterior collapse verification
python verify_posterior_collapse.py
```

**Evidence of Posterior Collapse:**

1. **Posterior Variance Too Low:**
   - All models: mean(z_log_var) = -2.1 to -2.9 (expected: 0)
   - Actual variance: 0.17-0.50 (expected: 1.0)
   - Encoder learned to produce low-variance posteriors (2-6x smaller than prior)

2. **Decoder Sensitivity Near Zero (SMOKING GUN):**
   - Decoder sensitivity (±2σ perturbation): 0.0015-0.0023
   - Typical residual std: ~0.010
   - **Decoder is 5-7x less sensitive than actual prediction errors**
   - Perturbing latents by ±2σ changes output by only 0.002, but real errors are ~0.01

3. **Residual Analysis Results:**
   - Mean predictions are unbiased (good MSE)
   - BUT: Predicted uncertainty drastically underestimated
   - No EX: 4-7x too narrow
   - EX No Loss: 10-20x too narrow
   - EX Loss: 14-28x too narrow (worst!)

4. **Exact Training Latents Test (2008-2010):**
   - Used z ~ N(μ_train[day], σ²_train[day]) for each day
   - Result: Still 64-94% violations (vs 10% target)
   - **Proves latent sampling is NOT the primary issue**
   - Decoder is fundamentally miscalibrated

5. **Uncertainty Learning Analysis:**
   - Only EX No Loss learned correct behavior (positive correlation +0.39 between |returns| and variance during crises)
   - No EX and EX Loss: NEGATIVE correlation (-0.22, -0.41) - more confident during volatility!

**Mechanism:**

1. Weak KL regularization (1e-5) allows posterior collapse
2. Encoder produces nearly deterministic latents (z ≈ μ)
3. During training, decoder sees stable z values
4. Decoder learns deterministic mapping: f(z) ≈ constant
5. Result: Decoder ignores latent variance → generates narrow distributions regardless of input

**Total Uncertainty Decomposition:**
```
Total Uncertainty = Latent Uncertainty × Decoder Sensitivity

If decoder sensitivity ≈ 0, doesn't matter if latent variance increases 100x during crisis.
```

**Generated Diagnostic Files:**
- `posterior_collapse_analysis.png` - 4-panel verification visualization
- `posterior_collapse_report.txt` - Detailed diagnostic summary
- `exact_training_latents_2008_2010.png` - Teacher forcing with exact latents
- `residual_analysis_exact_latents.png` - Residual vs predicted uncertainty
- `empirical_uncertainty_vs_returns.png` - Time-varying uncertainty analysis

### Related Work and Context

This finding is consistent with known issues in VAE-based probabilistic forecasting:
- VAEs with weak KL regularization (1e-5) develop posterior collapse
- Teacher forcing creates train-test distribution mismatch
- **Decoder becomes deterministic** even when posteriors are well-calibrated
- Empirical latent sampling is insufficient - requires architectural changes

**Why Weak KL Causes Overconfidence:**
```
KL term: KL(q(z|x) || p(z)) where p(z) = N(0,1)
Weak KL weight (1e-5) → encoder can make variance tiny without penalty
Result: z ≈ μ (almost no noise) → decoder learns deterministic mapping
```

**Critical Insight:**
Even if the true latent distribution is not N(0,1), increasing KL weight forces wrong distribution and hurts reconstruction. This is the fundamental VAE dilemma: strong KL → bad fit if prior is wrong, weak KL → posterior collapse.

For applications requiring reliable uncertainty estimates, consider alternative architectures:
- Direct quantile regression (see `quantile-regression-decoder` branch)
- Conformal prediction post-processing
- Ensemble methods with proper diversity
- Explicit heteroscedastic decoder (model uncertainty directly)
- Normalizing flows to learn flexible priors
- Two-stage VAE: learn q(z|x), then fit p(z) to empirical latents
