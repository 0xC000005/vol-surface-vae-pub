# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conditional VAE (CVAE) + LSTM for one-day-ahead volatility surface forecasts using **quantile regression**. All models use teacher forcing (condition on real historical data, not predictions).

**Paper:** Jacky Chen, John Hull, Zissis Poulos, Haris Rasul, Andreas Veneris, Yuntao Wu, "*A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces*", Journal of Financial Data Science, 2025.

## Setup

**Dependencies:**
```bash
uv sync  # Python >=3.13, dependencies in pyproject.toml
```

**CRITICAL - dtype Setup:**
```python
import torch
torch.set_default_dtype(torch.float64)  # MUST be FIRST, before model ops
```
Models trained with float64. PyTorch defaults to float32. Mismatch → runtime error.

**CRITICAL - Device Handling:**
```python
# CORRECT - Single .to() preserves dtype
def __init__(self, config):
    self.encoder = Encoder(config)
    self.decoder = Decoder(config)
    self.to(self.device)  # Single call at end

# WRONG - Individual .to() can break dtype
def __init__(self, config):
    self.encoder = Encoder(config).to(self.device)  # ✗
```

## Commands

**2D Surface VAE:**
```bash
# Training
python param_search.py                          # Train all 3 model variants

# Generation & Analysis
python generate_quantile_surfaces.py            # Generate quantile predictions
python generate_surfaces_max_likelihood.py      # Generate MLE surfaces
python visualize_teacher_forcing.py             # Visualize predictions
python main_analysis.py                         # Regression, PCA, arbitrage
```

**1D Time Series VAE:**
```bash
# Data prep
python fetch_market_data.py                     # Download from Yahoo Finance
python prepare_stock_data.py                    # Create stock_returns.npz

# Training & Evaluation
python test_1d_model_sanity.py                  # Architecture sanity check
python train_1d_models.py                       # Train all 4 variants
python generate_1d_quantile_predictions.py      # Generate predictions
python analysis_code/visualize_1d_quantile_teacher_forcing.py
python evaluate_1d_quantile_models.py           # Compute metrics
```

## Architecture

### Model Hierarchy

1. **CVAEMemRand** (vae/cvae_with_mem_randomized.py) - 2D surface VAE
   - Variable context lengths (randomized during training)
   - Encoder: Surface → Optional extra features → LSTM → Latent (z)
   - Decoder: [Context LSTM embeddings || z] → Quantile surfaces (3 channels)

2. **CVAE1DMemRand** (vae/cvae_1d_with_mem_randomized.py) - 1D time series VAE
   - Same architecture as 2D but for scalars instead of 5×5 grids
   - Encoder: Target → Optional cond features → LSTM → Latent (z)
   - Decoder: [Context LSTM embeddings || z] → Quantile values (3 channels)

### Two-Encoder Pattern

**Critical architectural insight:**
- **Main encoder**: Encodes full sequence [t-C, ..., t] during training
- **Context encoder**: Encodes only context [t-C, ..., t-1] during generation
- Generation uses: `decoder([ctx_embeddings || z_sampled])`
- Context embeddings padded to full sequence length before concatenation

### Quantile Regression

**All models use quantile regression exclusively (no MSE mode).**

**Decoder output:**
- 2D: (B, T, 3, H, W) where 3 = [p05, p50, p95] quantiles
- 1D: (B, T, 3) where 3 = [p05, p50, p95] quantiles

**Loss:**
```python
loss = pinball_loss(predictions, target) + kl_weight * kl_divergence
# Pinball loss: asymmetric quantile loss, optimizes CI calibration
```

**Config:**
```python
{
    "num_quantiles": 3,
    "quantiles": [0.05, 0.5, 0.95],
    "latent_dim": 5,
    "mem_type": "lstm",
    "mem_hidden": 100,
    "kl_weight": 1e-5,
}
```

### Model Variants (2D Surface VAE)

Three variants test conditioning strategies:

| Variant | ex_feats_dim | re_feat_weight | Input Features | Optimization |
|---------|--------------|----------------|----------------|--------------|
| **no_ex** | 0 | 0.0 | Surface only | Surface only |
| **ex_no_loss** | 3 | 0.0 | Surface + ret/skew/slope | Surface only (passive conditioning) |
| **ex_loss** | 3 | 1.0 | Surface + ret/skew/slope | Surface + Returns (joint learning) |

**Research question:** Does conditioning on returns/skew/slope improve surface forecasts?

### Model Variants (1D Time Series VAE)

Four variants test market conditioning:

| Variant | ex_feats_dim | Extra Features |
|---------|--------------|----------------|
| **amzn_only** | 0 | None (baseline) |
| **amzn_sp500** | 1 | SP500 returns |
| **amzn_msft** | 1 | MSFT returns |
| **amzn_both** | 2 | SP500 + MSFT returns |

All use `ex_feat_weight=0.0` (passive extra features).

## Data

**2D Surface VAE:**
- `data/vol_surface_with_ret.npz`: Main training data
  - `surface`: (N, 5, 5) volatility grids
  - `ret`, `skews`, `slopes`: (N,) extra features
  - `ex_data`: (N, 3) concatenated [ret, skew, slope]

**1D Time Series VAE:**
- `data/stock_returns.npz`: Stock returns
  - `amzn_returns`: (N,) target
  - `cond_sp500`, `cond_msft`, `cond_both`: Conditioning features

**Data splits:**
- Train: [0:4000]
- Valid: [4000:5000]
- Test: [5000:]

## Teacher Forcing

**Key insight:** All generation uses teacher forcing = independent one-step-ahead forecasts.

**For each day t:**
1. Context: Real observed data [t-C, ..., t-1]
2. Predict: Day t
3. Next day: Use real observed [t-C+1, ..., t] (includes real observation at t)

**Not autoregressive:** Each prediction is independent, not conditioned on previous predictions.

## Critical Implementation Patterns

### Loading Trained Models

```python
torch.set_default_dtype(torch.float64)  # FIRST!

model_data = torch.load(path, map_location=device, weights_only=False)
model = CVAEMemRand(model_data["model_config"])  # or CVAE1DMemRand
model.load_weights(dict_to_load=model_data)
model.eval()
```

### Forward Pass Returns

**CRITICAL:** Forward pass returns only FUTURE timesteps, not full sequence.

```python
# Input: batch with T timesteps (C context + 1 future)
decoded_target, _, z_mean, z_log_var, z = model.forward(batch)

# Output shapes:
# decoded_target: (B, 1, num_quantiles, H, W)  # Only future timestep!
# z_mean: (B, T, latent_dim)                   # Full sequence
```

### Context Extraction

```python
# Variable sequence length: T is different for each batch
C = T - 1  # Last timestep is target, rest is context

ctx_dict = {
    "surface": batch["surface"][:, :C, :, :],    # [t-C, ..., t-1]
    "ex_feats": batch["ex_feats"][:, :C, :],     # Optional
}
```

### Generation Method

```python
# Single forward pass → get all 3 quantiles
quantile_preds = model.get_surface_given_conditions(ctx_dict)
# Returns: (B, 1, 3, 5, 5) for 2D or (B, 3) for 1D
# quantile_preds[:, :, 0] = p05, [:, :, 1] = p50, [:, :, 2] = p95
```

## Dataset Classes

**VolSurfaceDataSetRand** (vae/datasets_randomized.py):
- Generates variable-length sequences (min_seq_len to max_seq_len)
- Uses `CustomBatchSampler` to ensure same length within batch
- Returns dict: `{"surface": (T, H, W), "ex_feats": (T, K)}`

**TimeSeriesDataSetRand** (vae/datasets_1d_randomized.py):
- 1D version for scalar time series
- Returns dict: `{"target": (T, 1), "ex_feats": (T, K)}`

## Data Preprocessing

**Requires Jupyter notebooks (not in repo):**

1. Download WRDS data: OptionMetrics Ivy DB (SECID=108105, SPX, 2000-2023)
2. Download S&P 500 prices from Yahoo Finance (`^GSPC`)
3. Run `spx_volsurface_generation.ipynb`: Clean and interpolate IVS
4. Run `spx_convert_to_grid.ipynb`: Convert to 5×5 grids
5. Output: `data/vol_surface_with_ret.npz`

**Pre-trained models:** https://drive.google.com/drive/folders/1W3KsAJ0YQzK2qnk0c-OIgj26oCAO3NI1

## Analysis Pipeline

**main_analysis.py workflow:**
1. Load models and generate quantile surfaces
2. Regression analysis (grid accuracy, RMSE benchmarks)
3. PCA on latent embeddings
4. Classification tasks (NBER recession prediction)
5. Arbitrage-free checks
6. Generate LaTeX tables and plots

**Key analysis modules (analysis_code/):**
- `regression.py`: Latent embeddings vs market indicators
- `latent_pca.py`: PCA analysis
- `arbitrage.py`: Check no-arbitrage violations
- `loss_table.py`: Format loss comparison tables

## Evaluation Metrics (1D Models)

- **RMSE/MAE**: Point forecast accuracy using p50 as forecast
- **R²**: Coefficient of determination
- **Direction Accuracy**: % correct sign predictions (>50% = signal)
- **CI Violation Rate**: % outside [p05, p95] (target: ~10%)
- **Mean CI Width**: Average uncertainty band width

## Key Architecture Files

- `vae/base.py`: BaseVAE, BaseEncoder, BaseDecoder
- `vae/cvae_with_mem_randomized.py`: CVAEMemRand (2D surface VAE)
- `vae/cvae_1d_with_mem_randomized.py`: CVAE1DMemRand (1D time series VAE)
- `vae/datasets_randomized.py`: VolSurfaceDataSetRand
- `vae/datasets_1d_randomized.py`: TimeSeriesDataSetRand
- `vae/utils.py`: train(), test(), set_seeds()

## Architecture Equivalence Note

Despite different data dimensions, 2D and 1D VAEs share **identical core architecture**:

**Decoder construction (both models):**
```python
# 1. Encode context
ctx_embeddings = self.ctx_encoder(context)  # (B, C, ctx_dim)

# 2. Pad to full sequence
ctx_padded = torch.zeros((B, T, ctx_dim))
ctx_padded[:, :C, :] = ctx_embeddings

# 3. Create latent variable
z = torch.zeros((B, T, latent_dim))
z[:, :C, :] = encoded_mean       # Encoded mean for context
z[:, C:, :] = sampled_z          # Sampled z for future

# 4. Concatenate and decode
decoder_input = torch.cat([ctx_padded, z], dim=-1)  # [ctx || z]
output = self.decoder(decoder_input)
```

This pattern is identical in both `CVAEMemRand.get_surface_given_conditions()` and `CVAE1DMemRand.get_prediction_given_context()`.

## CI Calibration Results

**Current quantile regression performance:**

| Model | CI Violations | Target |
|-------|--------------|--------|
| **2D no_ex** | 44.50% | ~10% |
| **2D ex_no_loss** | 35.43% | ~10% |
| **2D ex_loss** | 34.28% | ~10% |

**Key findings:**
- Ground truth latent (full sequence encoding): 5-7% violations (well-calibrated)
- Context-only latent (realistic generation): 18-20% violations
- **VAE prior mismatch**: p(z|context) ≠ p(z|context+target)

See `QUANTILE_REGRESSION_RESULTS.md` and `CI_CALIBRATION_OBSERVATIONS.md` for detailed analysis.

## Common Pitfalls

1. **dtype not set first** → RuntimeError about Double vs Float
2. **Individual .to() calls** → dtype reset to float32
3. **Confusing forward() output** → Returns only future [:, C:], not full [:, :]
4. **Context length off by one** → Use C = T - 1, not C = T
5. **Expecting MSE mode** → All models are quantile-only now

## Version History

- **Nov 2025**: Removed MSE mode, quantile regression only
- **Oct 2025**: Added quantile regression variant alongside MSE
- **2025**: Initial implementation for paper
