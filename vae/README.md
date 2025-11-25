# VAE Module - Variational Autoencoder Implementations

This directory contains all VAE model implementations for volatility surface forecasting.

## Module Overview

The `vae/` module provides a hierarchy of variational autoencoder models, from basic 2D convolutional VAEs to advanced conditional VAEs with LSTM memory and variable context lengths.

## Base Classes

### `base.py` - Abstract Base Classes

**BaseVAE**: Abstract base class for all VAE models
- Implements standard VAE loss: `loss = reconstruction_error + kl_weight * kl_loss`
- Provides core methods:
  - `train_step()`: Single training step with loss computation
  - `test_step()`: Evaluation on validation/test sets
  - `save_weights()`: Save model checkpoint
  - `load_weights()`: Load model from checkpoint

**BaseEncoder / BaseDecoder**: Abstract encoder/decoder interfaces
- Define standard interface for all encoder/decoder implementations
- Ensure consistency across model variants

## Model Hierarchy

### 1. VAEConv2D (`conv_vae.py`)
Basic 2D convolutional VAE for single surface encoding/decoding.

### 2. CVAE (`cvae.py`)
Conditional VAE with context encoder.
- Input: (B, T, H, W) where T = context_len + 1
- Uses separate context encoder for conditioning on historical surfaces
- Enables temporal dependencies

### 3. CVAEMem (`cvae_with_mem.py`)
CVAE with LSTM/GRU/RNN memory module.
- Adds recurrent memory to capture temporal dynamics
- Supports LSTM, GRU, or RNN architectures

### 4. CVAEMemRand (`cvae_with_mem_randomized.py`) ⭐ Primary Model
**The main model used in the paper.**

Features:
- **Variable context lengths**: Randomized during training (min_seq_len to max_seq_len)
- **Configurable horizon**: Multi-day prediction (default: 1 day, validated up to 5 days)
- **Optional extra features**: Can encode returns, skew, slope alongside surfaces
- **Quantile regression**: Optional quantile decoder for uncertainty quantification

## CVAEMemRand - Primary Model

### Configuration Parameters

```python
model_config = {
    # Surface dimensions
    "feat_dim": (5, 5),              # Volatility surface grid size

    # Latent space
    "latent_dim": 5,                 # Dimensionality of latent space

    # Prediction horizon
    "horizon": 1,                    # Number of days to predict (1-30)

    # Surface encoding
    "surface_hidden": [5, 5, 5],     # Hidden layer sizes for surface encoder
    "use_dense_surface": False,      # Use fully connected vs Conv2D

    # Extra features
    "ex_feats_dim": 0,               # 0 = surface only, 3 = ret/skew/slope
    "re_feat_weight": 0.0,           # Loss weight for extra features

    # Memory module
    "mem_type": "lstm",              # Memory type: lstm/gru/rnn
    "mem_hidden": 100,               # Hidden size for memory module
    "mem_layers": 1,                 # Number of memory layers

    # Context compression
    "compress_context": True,        # Compress context to latent_dim size

    # Loss weights
    "kl_weight": 1e-5,               # KL divergence weight

    # Quantile regression (optional)
    "use_quantile_regression": False,  # Enable quantile decoder
    "num_quantiles": 3,                # Number of quantiles (p05, p50, p95)
    "quantiles": [0.05, 0.5, 0.95],   # Quantile levels
}
```

### Training Data Format

Input is a dictionary with keys:
- `"surface"`: Tensor of shape (B, T, 5, 5) - Volatility surfaces
  - B = batch size
  - T = sequence length (context_len + horizon)
  - 5×5 = moneyness × time to maturity grid
- `"ex_feats"`: (Optional) Tensor of shape (B, T, n) - Extra features
  - n = 3 for [daily log return, volatility skew, term structure slope]

### Model Components

#### 1. CVAEMemRandEncoder
Encodes full sequence (context + target) to latent space.

**Architecture flow:**
```
Surface[t-C:t+H] → Surface Embedding (Conv2D or Dense)
                 ↓
Ex_Feats[t-C:t+H] → Ex_Feats Embedding (Dense) [optional]
                 ↓
            Concatenate
                 ↓
           LSTM Memory
                 ↓
         Latent Space (μ, σ)
```

#### 2. CVAECtxMemRandEncoder
Encodes context only (for generation/inference).

**Architecture:** Similar to full encoder but only processes context timesteps [t-C:t-1]

#### 3. CVAEMemRandDecoder
Decodes latent + context to predicted surface(s).

**Architecture flow:**
```
Latent z + Context Encoding
          ↓
    LSTM Decoder
          ↓
     ┌─────────┴─────────┐
     ↓                   ↓
Surface Decoder    Ex_Feats Decoder
(Conv2DTranspose)      (Dense)
     ↓                   ↓
 Surface[t+1:t+H]   Ex_Feats[t+1:t+H]
```

**Output shapes:**
- Standard MSE: (B, H, 5, 5) surfaces
- Quantile regression: (B, H, 3, 5, 5) - 3 quantiles per surface

## Dataset Handling

### `datasets_randomized.py`

**VolSurfaceDataSetRand**: Primary dataset class for training
- Generates variable-length sequences (min_seq_len to max_seq_len)
- Each data point: T consecutive surfaces from time series
- Uses `CustomBatchSampler` to ensure all samples in batch have same sequence length

**Why variable length?**
- Improves model robustness to different context window sizes
- Prevents overfitting to specific sequence length
- Enables flexible deployment with varying data availability

## Training Utilities

### `utils.py`

**Key functions:**

- **`train(model, train_loader, val_loader, optimizer, epochs, ...)`**
  - Training loop with early stopping
  - Supports checkpoint saving
  - Validation monitoring

- **`test(model, test_loader, ...)`**
  - Evaluation on test sets
  - Returns loss metrics

- **`set_seeds(seed)`**
  - Sets random seeds for numpy, torch, and torch.cuda
  - Ensures reproducibility

## Model Variants

See `experiments/backfill/MODEL_VARIANTS.md` for detailed comparison of:
- **No EX**: Surface-only baseline
- **EX No Loss**: Features as passive conditioning
- **EX Loss**: Joint optimization of surfaces and returns

## Quantile Regression

See `experiments/backfill/QUANTILE_REGRESSION.md` for:
- Architecture modifications for quantile prediction
- Pinball loss implementation
- Calibration results and analysis

## Usage Examples

See `DEVELOPMENT.md` for code examples:
- Loading trained models
- Generating predictions
- Autoregressive generation

## References

- **Main paper**: Chen et al. (2025), "A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces"
- **VAE theory**: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- **Conditional VAE**: Sohn et al. (2015), "Learning Structured Output Representation using Deep Conditional Generative Models"
