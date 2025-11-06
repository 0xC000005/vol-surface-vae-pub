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

## Critical Setup Requirements

### dtype Configuration

**CRITICAL**: All models are trained with `torch.float64`. You **MUST** set the default dtype before any model operations:

```python
import torch
torch.set_default_dtype(torch.float64)  # MUST be before model loading/creation
```

**Why this matters:**
- Models are trained with float64 precision
- PyTorch defaults to float32
- Mismatch causes: `RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float`

**Common mistake:**
```python
# WRONG - dtype not set before loading
model = torch.load("model.pt")  # Will load with float32 default
torch.set_default_dtype(torch.float64)  # Too late!

# CORRECT - dtype set first
torch.set_default_dtype(torch.float64)
model = torch.load("model.pt")  # Now loads with float64
```

**Application to all scripts:**
- Training scripts: Set at top of file before imports
- Generation scripts: Set before `torch.load()`
- Evaluation scripts: Set before any model operations
- Visualization scripts: Set before model loading

### Device Handling Pattern

**CRITICAL**: Use a single `.to(device)` call on the entire model to preserve dtype:

```python
# CORRECT - Single .to() call preserves torch.set_default_dtype
class CVAE1DMemRand(BaseVAE):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CVAE1DMemRandEncoder(config)
        self.ctx_encoder = CVAE1DCtxMemRandEncoder(config)
        self.decoder = CVAE1DMemRandDecoder(config)
        self.to(self.device)  # Single call at end

# INCORRECT - Individual .to() calls can override dtype
class CVAE1DMemRand(BaseVAE):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CVAE1DMemRandEncoder(config).to(self.device)  # Breaks dtype!
        self.ctx_encoder = CVAE1DCtxMemRandEncoder(config).to(self.device)
        self.decoder = CVAE1DMemRandDecoder(config).to(self.device)
```

**Why this matters:**
- Individual `.to()` calls can reset dtype to float32
- Single call on parent module preserves `torch.set_default_dtype()`
- Applies to both 2D and 1D VAE implementations

### Model Loading Pattern

**Standard pattern for loading trained models:**

```python
import torch
import numpy as np

# 1. Set dtype FIRST
torch.set_default_dtype(torch.float64)

# 2. Load model checkpoint
model_path = "models_1d/amzn_only.pt"
model_data = torch.load(model_path, map_location=DEVICE, weights_only=False)

# 3. Extract config and create model
model_config = model_data["model_config"]
model = CVAE1DMemRand(model_config)  # or CVAEMemRand for 2D

# 4. Load weights
model.load_weights(dict_to_load=model_data)

# 5. Set to eval mode
model.eval()
```

**Key points:**
- Always use `map_location=DEVICE` to handle CPU/GPU differences
- Use `weights_only=False` for loading full model state (includes config)
- Call `.eval()` to disable dropout/batch norm training behavior

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

**Confidence Interval Calibration Analysis:**
```bash
# Original MSE models - CI computed from MC samples
python verify_mean_tracking_vs_ci.py       # Verify R² vs CI violation coexistence
python compare_reconstruction_losses.py    # Compare MSE with CI calibration
python visualize_distribution_comparison.py # Compare marginal distributions

# Quantile regression models - CI directly predicted
python train_quantile_models.py                 # Train with pinball loss
python generate_quantile_surfaces.py            # Generate quantile surfaces
python evaluate_quantile_ci_calibration.py      # Evaluate CI calibration metrics
python visualize_quantile_teacher_forcing.py    # Visualize quantile predictions
```
These scripts analyze uncertainty calibration in model predictions:
- `verify_mean_tracking_vs_ci.py`: Proves that good mean tracking (high R²) can coexist with poor CI calibration (high violation rates). Generates scatter plots and regression analysis.
- `compare_reconstruction_losses.py`: Shows relationship between reconstruction loss (MSE) and CI calibration metrics.
- `visualize_distribution_comparison.py`: Compares marginal distributions (pooled across all days) vs ground truth histograms.
- `train_quantile_models.py`: Trains models with quantile regression decoder for direct CI prediction.
- `evaluate_quantile_ci_calibration.py`: Evaluates CI calibration for quantile regression models.
- See `CI_CALIBRATION_OBSERVATIONS.md` for MSE model findings and `QUANTILE_REGRESSION_RESULTS.md` for quantile regression analysis.

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
- `latent_dim`: Latent space dimensionality
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

## Architecture Deep Dive

### Forward Pass Behavior (CRITICAL)

**Key Pattern**: The `forward()` method returns **ONLY the future prediction** [:, C:, ...], not the full sequence.

This is a **fundamental design decision** that differs from standard seq2seq models:

```python
def forward(self, x):
    """
    Input:  (B, T, ...)      # T timesteps total
    Output: (B, T-C, ...)    # Only future timesteps (typically T-C=1)
    """
    target = x["target"]  # or x["surface"] for 2D models
    B = target.shape[0]
    T = target.shape[1]
    C = T - 1  # Context length = total timesteps - 1

    # 1. Extract context (first C timesteps)
    ctx_target = target[:, :C, ...]

    # 2. Encode full sequence (including target at T-1)
    z_mean, z_log_var, z = self.encoder(x)

    # 3. Get context embeddings (from context only)
    ctx_embeddings = self.ctx_encoder(ctx_input)

    # 4. Decode to full T timesteps
    decoded_output = self.decoder(...)  # (B, T, ...)

    # 5. Return ONLY future prediction (last timestep)
    return decoded_output[:, C:, ...], z_mean, z_log_var, z
    #                     ^^^^^^^^ CRITICAL: slice out future only
```

**Why this design?**
- Training objective: predict next day given context
- Loss should only be computed on future timestep, not context
- Context timesteps are deterministic (teacher forcing), only future is predicted

**Implications:**
- Loss computation uses `target[:, C:, ...]` to match decoded output shape
- Test assertions expect `(B, 1, ...)` not `(B, T, ...)`
- Generation methods return single timestep predictions

### Context Extraction Pattern

**Universal pattern across all models:**

```python
T = sequence.shape[1]    # Total timesteps in batch
C = T - 1                # Context length
context = sequence[:, :C, ...]      # Extract context [0:C]
future = sequence[:, C:, ...]       # Extract future [C:T]
```

**Key insight**: Context length `C` is **derived** from sequence length `T`, not hardcoded:
- During training: T varies (randomized sequences), so C varies too
- Variable context forces model to work with different history lengths
- At generation time: T = C + 1 (context + 1 future prediction)

**Example with T=6:**
```
Sequence: [t0, t1, t2, t3, t4, t5]
          └───────C=5────────┘  └─┘
              Context         Future

C = T - 1 = 5
context = sequence[:, :5, ...]  = [t0, t1, t2, t3, t4]
future  = sequence[:, 5:, ...]  = [t5]
```

### Two-Encoder Architecture

**Why two separate encoders?**

1. **Main Encoder** (`self.encoder`): Used during training
   - Input: Full sequence [t-C, ..., t-1, t] (context + target)
   - Output: Latent encoding z that captures complete sequence information
   - Purpose: Learn to encode complete patterns including the target

2. **Context Encoder** (`self.ctx_encoder`): Used during generation
   - Input: Context only [t-C, ..., t-1] (no future knowledge)
   - Output: Context embedding for conditioning
   - Purpose: Realistic generation without "cheating" by seeing the future

**Architecture diagram:**

```
TRAINING:
  [context: t-C...t-1, target: t] → encoder → z_mean, z_log_var, z
  [context: t-C...t-1] → ctx_encoder → ctx_embedding
  [ctx_embedding || z] → decoder → prediction for t

GENERATION:
  [context: t-C...t-1] → ctx_encoder → ctx_embedding
  z ~ N(0,1) or z = 0 (MLE)
  [ctx_embedding || z] → decoder → prediction for t
```

**Critical implementation detail:**
```python
# During generation, separate treatment for context vs future latents
z = torch.zeros((B, T, latent_dim), ...)

# Context timesteps: use encoded latent mean (deterministic)
z[:, :C, :] = z_mean[:, :C, :]

# Future timestep: sample from prior (stochastic) or use mode (MLE)
if use_mean:
    z[:, C:, :] = 0  # MLE: use mode of N(0,1)
else:
    z[:, C:, :] = torch.randn(...)  # Sample from N(0,1)
```

### Context Padding Pattern

**Why pad context embeddings?**

The decoder expects input of shape `(B, T, ctx_dim + latent_dim)`, but context encoder only produces `(B, C, ctx_dim)`.

```python
# ctx_encoder output: (B, C, ctx_dim)
ctx_embeddings = self.ctx_encoder(context)

# Need to pad to T timesteps
ctx_padded = torch.zeros((B, T, ctx_dim), device=self.device, dtype=z.dtype)
ctx_padded[:, :C, :] = ctx_embeddings  # Fill first C timesteps
# ctx_padded[:, C:, :] remains zero for future timestep

# Concatenate with latent
decoder_input = torch.cat([ctx_padded, z], dim=-1)  # (B, T, ctx_dim + latent_dim)
```

**What does this mean?**
- Context timesteps [0:C]: have both context embedding and latent encoding
- Future timestep [C]: only has latent variable (context embedding is zero)
- Decoder learns to use context info for [0:C] and generate from latent for [C:]

**Visualization:**
```
ctx_padded:  [ctx_emb, ctx_emb, ..., ctx_emb, 0, 0, ...]
                   └────────C────────┘  └─T-C─┘

z:           [z_mean, z_mean, ..., z_mean, z_sample, ...]
                   └────────C────────┘  └──T-C───┘

decoder_in:  [[ctx||z], [ctx||z], ..., [ctx||z], [0||z_sample], ...]
                   └──────────C──────────┘    └─────T-C──────┘
```

### Loss Computation Pattern

**Standard VAE loss with context-aware masking:**

```python
def compute_loss(self, x, decoded_output, z_mean, z_log_var):
    target = x["target"]  # (B, T, ...)
    T = target.shape[1]
    C = T - 1

    # Extract future ground truth (to match decoded_output shape)
    target_future = target[:, C:, ...]  # (B, T-C, ...)

    # Reconstruction loss (MSE or Pinball)
    recon_loss = loss_fn(decoded_output, target_future)

    # KL divergence (computed on full sequence encoding)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    kl_loss /= (B * T)  # Normalize by batch and sequence length

    # Total loss
    total_loss = recon_loss + self.kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
```

**Key points:**
- `decoded_output` is `(B, T-C, ...)` from forward pass
- `target_future` must be extracted to match shape
- KL loss uses full sequence z_mean/z_log_var (shape B, T, latent_dim)
- Both reconstruction and KL losses normalized appropriately

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

Run `visualize_teacher_forcing.py` to compare model performance:
```bash
python visualize_teacher_forcing.py
```

Generates:
- `teacher_forcing_implied_vol.png`: 9-panel comparison (3 models × 3 grid points)
- `teacher_forcing_returns.png`: Return predictions for EX Loss model

Shows how teacher forcing works (models always conditioned on real historical data) and performance differences across conditioning strategies.

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

**Visualizing quantile predictions:**
```bash
python visualize_quantile_teacher_forcing.py
```

**Verification Scripts (Ground Truth vs Context-Only):**
```bash
# Ground truth latent verification (uses full sequence encoding)
python verify_reconstruction_plotly_2008_2010.py  # Interactive HTML + analysis for 2008-2010

# Context-only verification (realistic generation without future knowledge)
python verify_reconstruction_2008_2010_context_only.py  # Both Plotly HTML and Matplotlib PNG

# Marginal distribution analysis
python visualize_marginal_distribution_quantile_encoded.py      # Ground truth latent
python visualize_marginal_distribution_quantile_context_only.py  # Context-only latent
```

These scripts test different aspects of model performance:
- **Ground truth latent**: Encodes full sequence [t-5,...,t] → tests decoder quality when given perfect information
- **Context-only latent**: Encodes only context [t-5,...,t-1] → tests realistic generation scenario
- **Marginal distributions**: Pool predictions across all days to test if model captures unconditional distribution

**Key Findings from Verification (2008-2010 period):**

| Latent Type | no_ex | ex_no_loss | ex_loss | Interpretation |
|-------------|-------|------------|---------|----------------|
| **Ground truth** | 7.32% | 5.33% | 6.89% | Well-calibrated when decoder has perfect info |
| **Context-only** | 18.63% | 20.44% | 19.65% | ~3× higher violations in realistic generation |

The large gap between ground truth and context-only CI violations reveals a **fundamental VAE prior mismatch**: p(z|context) ≠ p(z|context+target). The decoder learns correct quantiles for encoded latents, but the prior distribution doesn't match the posterior well enough for realistic forecasting.

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

See `QUANTILE_REGRESSION_RESULTS.md` for detailed analysis and recommendations.

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

## Troubleshooting Guide

### Common Errors and Solutions

#### 1. dtype Mismatch Error

**Error:**
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

**Cause:** Model trained with float64 but script using default float32.

**Solution:**
```python
import torch
torch.set_default_dtype(torch.float64)  # Add THIS LINE before any model operations
```

**Where to add:**
- Training scripts: At top of file before imports
- Generation scripts: Before `torch.load()`
- Evaluation scripts: Before model creation
- Visualization scripts: Before model loading

**Verification:** Check model weights dtype:
```python
model_data = torch.load("model.pt")
print(model_data["decoder.output_layer.weight"].dtype)  # Should be torch.float64
```

---

#### 2. Forward Pass Shape Mismatch

**Error:**
```
AssertionError: Expected (1, 1, 1), got (1, 6, 1)
# or
RuntimeError: The size of tensor a (6) must match the size of tensor b (1)
```

**Cause:** forward() returning all T timesteps instead of only future [:, C:, ...].

**Diagnosis:**
```python
# Check forward() implementation
decoded_output = self.decoder(decoder_input)  # (B, T, ...)

# WRONG: returns all timesteps
return decoded_output, z_mean, z_log_var, z

# CORRECT: returns only future timesteps
C = T - 1
return decoded_output[:, C:, ...], z_mean, z_log_var, z
```

**Solution:** Add slicing to extract future timesteps:
```python
def forward(self, x):
    # ... encoding and decoding ...
    T = target.shape[1]
    C = T - 1

    # CRITICAL: Return only future prediction
    return decoded_target[:, C:, :], decoded_cond_feats[:, C:, :], z_mean, z_log_var, z
```

**Update loss computation too:**
```python
def compute_loss(self, x, decoded_target, ...):
    target = x["target"]
    C = target.shape[1] - 1
    target_future = target[:, C:, :]  # Extract future to match decoded_target shape
    recon_loss = F.mse_loss(decoded_target, target_future)
```

---

#### 3. Context Length Confusion

**Error:**
```
IndexError: index 5 is out of bounds for dimension 1 with size 5
# or
RuntimeError: shape mismatch in slicing
```

**Cause:** Hardcoded context length instead of deriving from T.

**Wrong pattern:**
```python
CONTEXT_LEN = 5  # Hardcoded!
ctx_target = target[:, :CONTEXT_LEN, :]  # Breaks when T != 6
```

**Correct pattern:**
```python
T = target.shape[1]
C = T - 1  # Derive context length
ctx_target = target[:, :C, :]  # Works for any T
```

**Key insight:** During training, T is variable (randomized sequences). Always compute C = T - 1 dynamically.

---

#### 4. Device/dtype Ordering in Model Init

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
# After supposedly fixing, dtype error reappears
```

**Cause:** Individual `.to(device)` calls override `torch.set_default_dtype()`.

**Wrong pattern:**
```python
class CVAE1DMemRand(BaseVAE):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CVAE1DMemRandEncoder(config).to(self.device)  # Each .to() call
        self.ctx_encoder = CVAE1DCtxMemRandEncoder(config).to(self.device)  # can reset
        self.decoder = CVAE1DMemRandDecoder(config).to(self.device)  # dtype!
```

**Correct pattern:**
```python
class CVAE1DMemRand(BaseVAE):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CVAE1DMemRandEncoder(config)      # No .to() here
        self.ctx_encoder = CVAE1DCtxMemRandEncoder(config)  # No .to() here
        self.decoder = CVAE1DMemRandDecoder(config)      # No .to() here
        self.to(self.device)  # Single call at end preserves dtype
```

---

#### 5. Model Collapse (Known Limitation)

**Symptom:**
```python
# Generate multiple samples with different z ~ N(0,1)
samples_std = predictions.std(axis=1)
print(samples_std.mean())  # 0.000005 (essentially zero!)
```

**Cause:** For simple prediction tasks (e.g., 1D stock returns), the model may learn to ignore the latent variable and rely solely on context.

**Diagnosis:**
```python
# Test different sampling strategies
z_prior = torch.randn(...)        # z ~ N(0,1): std = 0.000008
z_posterior = torch.randn(...) * z_std + z_mean  # z ~ N(μ, σ): std = 0.000014
z_deterministic = z_mean          # Direct mean: std = 0.000902

# If all produce near-zero variance → model collapsed
```

**When it happens:**
- 1D time series with strong context signal
- Simple prediction tasks where context is sufficient
- High context-to-prediction information ratio

**Impact:**
- Narrow confidence intervals (overconfident predictions)
- CI calibration will be poor (high violation rates)
- Model still predicts well, but uncertainty estimates are unreliable

**Not a bug:** This is trained model behavior. The model learned that context alone is sufficient, making the latent variable redundant.

**Mitigation strategies:**
1. Increase KL weight to force latent usage
2. Add information bottleneck in context encoder
3. Use β-VAE formulation (β > 1)
4. Switch to quantile regression for direct CI prediction

---

#### 6. Quantile vs MSE Model Confusion

**Error:**
```python
# Expecting 3 quantile channels
p05 = predictions[:, :, 0, :, :]  # IndexError: index 2 is out of bounds for dimension 2 with size 1
```

**Cause:** Trying to extract quantiles from MSE model output.

**Diagnosis:**
```python
# Check model configuration
print(model_config.get("use_quantile_regression", False))
# False or not present → MSE model (single output channel)
# True → Quantile regression model (3 output channels)

# Check decoder output shape
output = model(batch)
print(output[0].shape)
# (B, 1, H, W) → MSE model
# (B, 1, 3, H, W) → Quantile model
```

**Solution for MSE models:** Generate empirical quantiles:
```python
# Generate multiple samples
samples = []
for _ in range(1000):
    z = torch.randn((B, 1, latent_dim), ...)  # Sample different z
    prediction = model.generate(context, z)
    samples.append(prediction)

samples = np.array(samples)  # (1000, B, 1, ...)

# Compute empirical quantiles
p05 = np.percentile(samples, 5, axis=0)
p50 = np.percentile(samples, 50, axis=0)
p95 = np.percentile(samples, 95, axis=0)
```

**Solution for quantile models:** Direct extraction:
```python
# Single forward pass returns all quantiles
output = model(context)  # (B, 1, 3, H, W)
p05 = output[:, :, 0, :, :]
p50 = output[:, :, 1, :, :]
p95 = output[:, :, 2, :, :]
```

---

#### 7. Teacher Forcing Misunderstanding

**Confusion:** "Why don't we use previous predictions as context?"

**Clarification:** This codebase uses **teacher forcing** during inference:
- Each day's prediction uses **real historical data** as context
- Predictions are **independent one-step-ahead forecasts**
- NOT autoregressive multi-step prediction

**Example:**
```python
# Teacher forcing (this codebase)
Day 1: context=[real_t-5, ..., real_t-1] → predict t
Day 2: context=[real_t-4, ..., real_t] → predict t+1  # Uses real_t, not predicted_t
Day 3: context=[real_t-3, ..., real_t+1] → predict t+2

# Autoregressive (NOT used here)
Day 1: context=[real_t-5, ..., real_t-1] → predict t
Day 2: context=[real_t-4, ..., pred_t] → predict t+1  # Uses pred_t
Day 3: context=[real_t-3, ..., pred_t+1] → predict t+2
```

**Why teacher forcing?**
- Evaluates model's inherent prediction ability
- Avoids error accumulation from previous predictions
- Standard practice for single-step forecast evaluation

---

### Debugging Checklist

When encountering issues, check in order:

1. **dtype setup:**
   - [ ] `torch.set_default_dtype(torch.float64)` before model operations?
   - [ ] No individual `.to(device)` calls on submodules?

2. **Forward pass behavior:**
   - [ ] Returns `[:, C:, ...]` not full sequence?
   - [ ] Loss uses `target[:, C:, ...]` to match decoded shape?

3. **Context extraction:**
   - [ ] Computes `C = T - 1` dynamically?
   - [ ] Not hardcoded `CONTEXT_LEN`?

4. **Model type:**
   - [ ] Check `use_quantile_regression` in config?
   - [ ] Using appropriate quantile/sampling method?

5. **Shape expectations:**
   - [ ] Test assertions match forward() output shape?
   - [ ] Visualization expects correct number of models?

## 1D Time Series VAE (Stock Returns Forecasting)

A parallel implementation exists for 1D scalar time series (stock returns) using the same architectural principles as the 2D surface VAE. This was created to test conditioning strategies on simpler data.

### Architecture: CVAE1DMemRand

**Location:** `vae/cvae_1d_with_mem_randomized.py`

**Key Differences from 2D Surface VAE:**

| Aspect | 2D Surface VAE | 1D Time Series VAE |
|--------|----------------|-------------------|
| **Input data** | `"surface"` (B,T,5,5) + `"ex_feats"` (B,T,3) | `"target"` (B,T,1) + `"cond_feats"` (B,T,K) |
| **Target** | SPX volatility surfaces | Amazon stock returns |
| **Conditioning features** | SPX return + skew + slope (derived from vol surface) | SP500 and/or MSFT returns (raw returns only) |
| **Encoder** | Conv2D → Flatten → LSTM → Latent | Linear layers → LSTM → Latent |
| **Decoder** | LSTM → ConvTranspose2D → (B,T,5,5) | LSTM → Linear layers → (B,T,1) |
| **Core architecture** | [ctx_embed \|\| z] concatenation | [ctx_embed \|\| z] concatenation ✓ SAME |

**Critical Implementation Note:**
```python
# CORRECT (follows 2D VAE pattern, respects torch.set_default_dtype):
self.encoder = CVAE1DMemRandEncoder(config)
self.ctx_encoder = CVAE1DCtxMemRandEncoder(config)
self.decoder = CVAE1DMemRandDecoder(config)
self.to(self.device)  # Single call on entire model

# INCORRECT (dtype issues):
self.encoder = CVAE1DMemRandEncoder(config).to(self.device)  # Separate .to() calls
```

### Dataset: TimeSeriesDataSetRand

**Location:** `vae/datasets_1d_randomized.py`

Handles 1D time series with variable sequence lengths, analogous to `VolSurfaceDataSetRand` but for scalar sequences.

### Model Variants

Four variants test different conditioning strategies:

1. **Amazon only (baseline)**: `cond_feats_dim=0`, no conditioning
2. **Amazon + SP500 (no loss)**: `cond_feats_dim=1`, passive conditioning on market index
3. **Amazon + MSFT (no loss)**: `cond_feats_dim=1`, passive conditioning on tech sector peer
4. **Amazon + SP500 + MSFT (no loss)**: `cond_feats_dim=2`, both market and sector conditioning

All use `cond_feat_weight=0.0` (passive conditioning) to mirror the "EX No Loss" approach from 2D models.

### Common Commands

**Data Preparation:**
```bash
# Fetch market data from Yahoo Finance
python fetch_market_data.py  # Downloads AMZN, ^GSPC, MSFT, etc.

# Prepare training data (compute returns, create NPZ)
python prepare_stock_data.py
# Output: data/stock_returns.npz
```

**Training:**
```bash
# Test model architecture before training
python test_1d_model_sanity.py  # Runs all sanity checks

# Train all 4 model variants (30-60 min per model)
python train_1d_models.py
# Output: models_1d/*.pt and models_1d/results.csv
```

**Generation and Evaluation:**
```bash
# Generate predictions using teacher forcing
python generate_1d_predictions.py
# Output: predictions_1d/*.npz (stochastic + MLE)

# Visualize predictions (in analysis_code/)
python analysis_code/visualize_1d_teacher_forcing.py
# Output: plots_1d/teacher_forcing_comparison.png

# Compute evaluation metrics
python evaluate_1d_models.py
# Output: results_1d/evaluation_metrics.csv
```

### Data Format

**Input (data/stock_returns.npz):**
```python
{
    "amzn_returns": (5824,),      # Amazon log returns (target)
    "sp500_returns": (5824,),     # SP500 log returns
    "msft_returns": (5824,),      # MSFT log returns
    "dates": (5824,),             # Trading dates
    "cond_sp500": (5824, 1),      # SP500 for conditioning
    "cond_msft": (5824, 1),       # MSFT for conditioning
    "cond_both": (5824, 2),       # [SP500, MSFT] for conditioning
}
```

**Model Input:**
```python
{
    "target": (B, T, 1),           # Amazon returns
    "cond_feats": (B, T, K),       # Optional: K=1 (SP500 or MSFT) or K=2 (both)
}
```

**Model Output:**
```python
# Stochastic generation
predictions: (num_days, num_samples, 1)  # Multiple samples per day

# MLE generation
prediction: (num_days, 1)  # Single deterministic prediction
```

### Evaluation Metrics

The `evaluate_1d_models.py` script computes:
- **RMSE**: Root mean squared error (point forecast accuracy)
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination (predictive power)
- **Direction Accuracy**: % of correct sign predictions (better than random = >50%)
- **CI Violation Rate**: % outside 90% confidence interval (target: ~10%)
- **Mean CI Width**: Average uncertainty band width

### Research Questions

1. **Does market context help?** Compare Amazon-only vs Amazon+SP500
2. **What's more informative?** Compare SP500 vs MSFT vs Both
3. **Is the model better than random?** Direction accuracy >50% indicates signal
4. **Are uncertainties well-calibrated?** CI violations near 10% = well-calibrated

### Architecture Equivalence

Despite different data dimensions, the 1D and 2D VAEs share identical core architecture:

**Decoder Input Construction (both models):**
```python
# Encode context to get embeddings
ctx_embeddings = self.ctx_encoder(context)  # (B, C, ctx_dim)

# Pad context embeddings to full sequence length T
ctx_padded = torch.zeros((B, T, ctx_dim), ...)
ctx_padded[:, :C, :] = ctx_embeddings

# Construct latent variable
z = torch.zeros((B, T, latent_dim), ...)
z[:, :C, :] = encoded_z_mean        # Use encoded mean for context
z[:, C:, :] = sampled_z_future      # Sample for future timesteps

# Concatenate and decode
decoder_input = torch.cat([ctx_padded, z], dim=-1)  # [context || latent]
output = self.decoder(decoder_input)
```

This pattern is identical in both `CVAEMemRand.get_surface_given_conditions()` (2D) and `CVAE1DMemRand.get_prediction_given_context()` (1D).

### Known Limitations: Model Collapse in 1D VAE

The 1D time series models exhibit **model collapse**, where the decoder learns to ignore the latent variable and rely solely on context embeddings.

**Evidence:**
```python
# Generate 1000 samples with different z ~ N(0,1)
samples_std = predictions.std(axis=1)
print(samples_std.mean())  # 0.000005 - essentially zero variance!
```

**Root cause:**
- Stock return prediction from market context is relatively simple
- Context signal (SP500, MSFT returns) is highly informative
- Model learns that context alone is sufficient
- Latent variable becomes redundant

**Testing different sampling strategies (all fail):**
1. `z ~ N(0,1)` (prior sampling): std = 0.000008
2. `z ~ N(z_mean, z_std)` (posterior sampling): std = 0.000014
3. Direct `z_mean` (deterministic): std = 0.000902
4. All methods produce nearly identical predictions

**Impact:**
- Point predictions remain accurate (RMSE, R² unaffected)
- Uncertainty estimates are unreliable (overconfident)
- CI violation rates will be high (>50% instead of target 10%)
- Confidence intervals collapse to near-zero width

**Why this is NOT a bug:**
- This is trained model behavior, not implementation error
- VAE learned optimal strategy: ignore noisy latent, trust context
- Common in VAEs when conditioning information is strong
- Similar to posterior collapse in text VAEs

**Comparison to 2D VAE:**
- 2D volatility surface VAE does NOT collapse
- Surface prediction is more complex than 1D return prediction
- Higher-dimensional output requires latent information
- Stochastic generation produces meaningful uncertainty

**Mitigation strategies** (for future work):
1. **Increase KL weight**: Force model to use latent variable
   ```python
   kl_weight: 1e-3  # vs current 1e-5, penalize ignoring z
   ```

2. **Information bottleneck**: Limit context encoder capacity
   ```python
   mem_hidden: 20  # vs current 100, force reliance on latent
   ```

3. **β-VAE**: Disentangle latent representations
   ```python
   beta: 4.0  # Emphasize KL term
   ```

4. **Conditional prior**: Model p(z|context) instead of p(z)=N(0,1)
   - Learn context-dependent prior distribution
   - Better match between prior and posterior

5. **Quantile regression**: Bypass stochastic generation entirely
   - Directly predict quantiles (p05, p50, p95)
   - No reliance on latent sampling
   - See 2D quantile models for reference

**Current status:**
- Models train successfully and predict accurately
- Collapse documented as known limitation
- Visualization scripts compute empirical quantiles from collapsed samples
- Users should be aware CI calibration metrics will be poor
