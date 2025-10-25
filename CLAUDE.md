# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase implements a Variational Autoencoder (VAE) approach for conditional generation of future volatility surfaces, as described in the paper "A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces" (Journal of Financial Data Science, 2025). The system uses a Conditional VAE (CVAE) combined with LSTM to generate volatility surfaces based on arbitrary context lengths.

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

The codebase trains three model variants:
1. **no_ex**: Surface only, no extra features
2. **ex_no_loss**: Surface + extra features, but no loss on extra features
3. **ex_loss**: Surface + extra features, with loss on both (return only for ex_feats)

## Important Implementation Details

### Loss Function
- Reconstruction error uses MSE for surfaces
- KL divergence term weighted by `kl_weight` (typically 1e-5)
- Extra features use L1 or L2 loss weighted by `re_feat_weight`
- When `ex_loss_on_ret_only=True`, only return (first feature) gets loss optimization

### Sequence Generation
- Context length `C` is variable during training (4-10 days typical)
- At generation time, uses last `C` days to predict day `C+1`
- For multi-day generation, uses autoregressive approach with maximum likelihood (encoded latent mean)

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
Data should be downloaded from WRDS (OptionMetrics Ivy DB) and preprocessed using notebooks in the project root (not included in this repository). The preprocessing pipeline generates 5×5 interpolated volatility surface grids from option prices.
