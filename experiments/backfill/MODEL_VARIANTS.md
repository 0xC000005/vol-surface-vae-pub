# VAE Model Variants for Volatility Surface Forecasting

This document details the three model variants implemented in this codebase. These variants explore a fundamental research question: **Can we improve volatility surface forecasts by jointly modeling returns and surfaces, or is surface-only modeling sufficient?**

## Overview

All three variants use the same CVAEMemRand architecture with LSTM memory but differ in how they handle extra market features (returns, skew, slope):

1. **No EX**: Surface-only baseline (no extra features)
2. **EX No Loss**: Uses features as passive conditioning (no optimization)
3. **EX Loss**: Joint optimization of surfaces and returns

## 1. No EX (Surface Only - Baseline)

### Configuration

```python
ex_feats_dim: 0                # No extra features
re_feat_weight: 0.0            # No feature loss
```

### Input

Only volatility surfaces (5×5 grids):
- Context: Last C days of surfaces
- No additional market information

### Architecture

```
Surface[t-C:t] → LSTM Encoder → Latent z → LSTM Decoder → Surface[t+1]
```

### What it learns

- Pure surface-to-surface mapping
- Pattern recognition in volatility surface evolution
- No explicit conditioning on returns/skew/slope

### Use case

Baseline model to test if surface dynamics alone contain sufficient information for forecasting.

---

## 2. EX No Loss (Features as Passive Conditioning)

### Configuration

```python
ex_feats_dim: 3                # Has 3 extra features (return, skew, slope)
re_feat_weight: 0.0            # NO loss on features!
```

### Input

Surfaces + Extra features:
- Surface: 5×5 volatility grids
- Extra features: [daily log return, volatility skew, term structure slope]

### Architecture

```
Surface[t-C:t] + Features[t-C:t] → LSTM Encoder → Latent z → Decoder → Surface[t+1]
                                                                        (Features ignored in loss)
```

### What it learns

- Uses extra features as **conditioning information** only
- Model sees the features during encoding but isn't trained to reconstruct them
- Features help encoder understand market state (e.g., "market crashed yesterday")
- Decoder can use this context to generate better surfaces

### Key insight

Tests whether passively providing market context improves predictions without requiring the model to learn feature dynamics.

---

## 3. EX Loss (Features with Joint Optimization)

### Configuration

```python
ex_feats_dim: 3                # Has 3 extra features
re_feat_weight: 1.0            # YES - optimize feature reconstruction!
ex_loss_on_ret_only: True      # Only optimize returns (not skew/slope)
ex_feats_loss_type: "l2"       # L2 loss for return predictions
```

### Input

Same as EX No Loss (surfaces + features)

### Architecture

```
Surface[t-C:t] + Features[t-C:t] → LSTM Encoder → Latent z → LSTM Decoder → Surface[t+1]
                                                                           → Return[t+1]
                                                                           (Both optimized!)
```

### Loss function

```
Total Loss = MSE(surface) + re_feat_weight × L2(return) + kl_weight × KL_divergence
           = MSE(surface) + 1.0 × L2(return) + 1e-5 × KL
```

### What it learns

- Jointly optimizes surface AND return prediction
- Multi-task learning forces latent space to capture return dynamics
- Model must generate coherent predictions across both modalities
- Creates shared representation that understands both surface shape and return magnitudes

### Key insight

Tests whether forcing the model to predict returns improves surface forecasts by creating better latent representations.

---

## Model Comparison

### Comparison Table

| Aspect | No EX | EX No Loss | EX Loss |
|--------|-------|------------|---------|
| **Input features** | Surface only | Surface + return/skew/slope | Surface + return/skew/slope |
| **Encoder sees extras?** | ❌ No | ✅ Yes (conditioning) | ✅ Yes (conditioning) |
| **Decoder outputs extras?** | ❌ No | ✅ Yes (ignored) | ✅ Yes (optimized) |
| **Loss on features** | 0.0 | 0.0 | 1.0 (returns only) |
| **Learning objective** | Surface only | Surface (conditioned on features) | Surface + Returns (joint learning) |
| **Training data** | `train_simple` | `train_ex` | `train_ex` |

### Test Set Performance

Results from `param_search.py`:

| Model | Surface Reconstruction Loss | Feature Reconstruction Loss | KL Loss |
|-------|------------------------------|----------------------------|---------|
| **No EX** | 0.001722 | 0.000000 | 3.956 |
| **EX No Loss** | 0.002503 | 0.924496 (not optimized) | 3.711 |
| **EX Loss** | 0.001899 | 0.000161 | 4.000 |

**Observations:**
- **No EX** achieves lowest surface reconstruction error (baseline)
- **EX No Loss** has highest surface error - features help conditioning but high feature loss suggests model doesn't learn feature structure well
- **EX Loss** balances both objectives - competitive surface error with excellent return prediction (0.000161)

## Research Questions Addressed

### 1. No EX vs EX No Loss

**Question**: Does passive conditioning on market features improve predictions?

**Test**: Information benefit without optimization overhead

### 2. EX No Loss vs EX Loss

**Question**: Should we actively optimize feature predictions?

**Test**: Multi-task learning hypothesis - does forcing coherent return predictions improve surface quality?

### 3. Practical Implications

- **No EX**: Simplest, but ignores market context (e.g., doesn't know if crash occurred)
- **EX No Loss**: Uses context but may not fully leverage feature information
- **EX Loss**: Joint prediction may create more robust latent representations

## Generation Outputs

All three models generate two types of forecasts:

### Stochastic (Probabilistic)

- File format: `{model}_gen5.npz`
- Sampling: 1,000 samples per day by sampling z ~ N(0, 1)
- Shape: (num_days, 1000, 5, 5) surfaces
- EX Loss also outputs: (num_days, 1000, 3) features [return, skew, slope]

### Maximum Likelihood (Deterministic)

- File format: `{model}_mle_gen5.npz`
- Sampling: 1 sample per day using z = 0 (distribution mode)
- Shape: (num_days, 1, 5, 5) surfaces
- EX Loss also outputs: (num_days, 1, 3) features

## Visualization

Visualization scripts are in `analysis_code/`:

```bash
python analysis_code/visualize_teacher_forcing.py
```

Generates 9-panel comparison (3 models × 3 grid points) showing ground truth vs predictions with uncertainty bands. Demonstrates teacher forcing behavior where models condition on real historical data for independent one-step-ahead forecasts.

## Training Scripts

- **Training**: `param_search.py` (trains all 3 variants)
- **Generation**: `generate_surfaces.py` (stochastic), `generate_surfaces_max_likelihood.py` (MLE)
- **Analysis**: `main_analysis.py`
- **Visualization**: `analysis_code/visualize_teacher_forcing.py`

## References

- Main paper: Chen et al. (2025), "A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces"
- Implementation: `vae/cvae_with_mem_randomized.py` (CVAEMemRand class)
- Training data: `data/vol_surface_with_ret.npz`
