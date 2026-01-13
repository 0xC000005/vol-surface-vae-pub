# Two-Stage VAE Model Architecture

## Overview

The model generates **conditional forecasts of volatility surfaces** using a two-stage Conditional VAE architecture with a Student-t decoder for fat-tailed distributions.

```
INPUT: 30 days of historical volatility surfaces (5×5 grid)
OUTPUT: 30 days of future volatility surface predictions with uncertainty
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TWO-STAGE CONDITIONAL VAE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 1: VAE (Autoencoder)                                            │
│  ════════════════════════════                                          │
│                                                                         │
│  Input: [Context + Target] log-returns                                 │
│         (B, 60, 5, 5) = 30 context + 30 horizon                        │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Context    │    │    Main      │    │   Student-t MLP          │  │
│  │   Encoder    │    │   Encoder    │    │   Decoder                │  │
│  │              │    │              │    │                          │  │
│  │  LSTM →      │    │  LSTM →      │    │   z → MLP → mean        │  │
│  │  ctx_emb     │    │  z_mean      │    │         → L (Cholesky)  │  │
│  │  (3 dim)     │    │  z_logvar    │    │         → ν (d.o.f.)    │  │
│  │              │    │  (16 dim)    │    │                          │  │
│  └──────┬───────┘    └──────┬───────┘    └────────────┬─────────────┘  │
│         │                   │                         │                 │
│         │    ┌──────────────┴──────────────┐         │                 │
│         └───►│      Decoder Input          │◄────────┘                 │
│              │   concat(ctx_emb, z)        │                           │
│              │      (3 + 16 = 19 dim)      │                           │
│              └─────────────────────────────┘                           │
│                                                                         │
│  Loss = NLL(Student-t) + KL(z || N(0,1))                              │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 2: Predictor Network (trained separately)                       │
│  ═══════════════════════════════════════════════                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    LatentPredictor                               │  │
│  │                                                                  │  │
│  │   Context surfaces ──► LSTM ──► Autoregressive loop ──► z_pred  │  │
│  │   (B, 30, 5, 5)              (30 steps)              (B, 30, 16)│  │
│  │                                                                  │  │
│  │   For each horizon step h:                                       │  │
│  │     h_t, c_t = LSTM(feedback(h_{t-1}), (h_{t-1}, c_{t-1}))     │  │
│  │     z_mean[h] = Linear(h_t)                                     │  │
│  │     z_logvar[h] = Linear(h_t)                                   │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Loss = MSE(z_pred, z_true)  where z_true from frozen VAE encoder     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Context Encoder (Tiny Bottleneck)

**Purpose**: Compress historical context into a coarse representation.

```python
Context Encoder:
  Input:  (B, T, 5, 5) log-return surfaces

  Conv2D layers: [2, 4, 2] channels
    → Flatten to 50 features per timestep

  LSTM: hidden=8, layers=1
    → Temporal encoding

  Output: ctx_emb (B, T, 3)  ← TINY (forces z to carry variance)
```

**Key insight**: The 3-dimensional bottleneck is deliberately small (like k-means clusters) to force the latent variable `z` to capture fine-grained variation.

### 2. Main Encoder (Latent z)

**Purpose**: Encode the target information into latent variable z.

```python
Main Encoder:
  Input:  (B, T, 5, 5) log-return surfaces (context + target)

  Conv2D layers: [2, 4, 2] channels
    → Flatten to 50 features per timestep

  LSTM: hidden=8, layers=1
    → Temporal encoding

  Output: z_mean (B, T, 16), z_logvar (B, T, 16)

  Sampling: z = z_mean + exp(0.5 * z_logvar) * ε,  ε ~ N(0,1)
```

**Key parameters**:
- `latent_dim = 16` (expressive enough for variance)
- `z_logvar_floor = -4.0` (prevents variance collapse)

### 3. Student-t MLP Decoder

**Purpose**: Decode (ctx_emb, z) → predicted log-returns with fat tails.

```python
Student-t MLP Decoder:
  Input: z (B, T, 16)  ← Note: ctx_emb is NOT used (prevents z wash-out)

  MLP: Linear(16 → 128) → ReLU → Linear(128 → 25)
    → mean (B, T, 5, 5)

  MLP: Linear(16 → 128) → ReLU → Linear(128 → 25)
    → L (Cholesky factor for covariance)

  Fixed: ν (degrees of freedom) per grid point
    → Computed from GT kurtosis: ν = 4 + 6/excess_kurtosis

  Sampling:
    u ~ Gamma(ν/2, ν/2)           # Scale mixing
    ε ~ N(0, I)                    # Standard normal
    x = mean + L @ ε / √u          # Student-t sample
```

**Why MLP instead of LSTM decoder?**
- LSTM decoder causes **z wash-out** (z contributes only 3.8%)
- MLP forces z to carry all variation (z contributes 39.5%)
- Trade-off: No temporal dynamics in decoder, but preserves uncertainty

### 4. Latent Predictor (Stage 2)

**Purpose**: Predict z from context only (no target information).

```python
LatentPredictor:
  Input: context_surfaces (B, 30, 5, 5)

  Surface embedding: Conv2D [2, 4, 2] → 50 features

  LSTM encoder: Process context
    → h_C, c_C (hidden state after context)

  Autoregressive generation (30 steps):
    For h = 0 to 29:
      input_h = feedback_proj(output_{h-1})  # or context output if h=0
      output_h, (h_n, c_n) = LSTM(input_h, (h_n, c_n))
      z_mean[h] = Linear(output_h)
      z_logvar[h] = Linear(output_h)

  Output: z_mean (B, 30, 16), z_logvar (B, 30, 16)
```

**Training**: Freeze VAE, train predictor to match encoder's z:
```python
z_true = VAE.encoder(context + target)  # Detached
z_pred = Predictor(context)
loss = MSE(z_pred, z_true)
```

---

## Training Process

### Stage 1: Train VAE (100 epochs)

```python
# Input: Full sequence (context + target)
batch = {"surface": log_returns}  # (B, 60, 5, 5)

# Forward pass
mean, z_mean, z_logvar, L, ν = model(batch)

# Loss
nll_loss = student_t_nll(mean, target, L, ν)  # Negative log-likelihood
kl_loss = KL(q(z|x) || p(z))                   # KL divergence
total_loss = nll_loss + 0.001 * kl_loss        # Weak KL weight
```

### Stage 2: Train Predictor (50 epochs)

```python
# Freeze VAE
for param in vae.parameters():
    param.requires_grad = False

# Get ground truth z from encoder
with torch.no_grad():
    z_true, _, _ = vae.encoder(full_sequence)
    z_true = z_true[:, context_len:]  # Only horizon positions

# Predict z from context only
z_pred_mean, z_pred_logvar = predictor(context_only)

# Loss
loss = MSE(z_pred_mean, z_true.detach())
```

---

## Generation Modes

### Oracle Mode (Upper Bound)

Uses encoder to sample z - sees the target data.

```python
# Oracle: z from posterior (encoder sees target)
z = encoder(context + target)
prediction = decoder(z)
```

**Use case**: Benchmarking, understanding VAE capacity

### Predictor Mode (Realistic)

Uses predictor network - only sees context.

```python
# Predictor: z from trained network (no target)
z_mean, z_logvar = predictor(context_only)
z = z_mean + exp(0.5 * z_logvar) * ε
prediction = decoder(z)
```

**Use case**: Actual deployment, forecasting

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Decoder type | MLP (not LSTM) | LSTM causes z wash-out (3.8% → 39.5% z contribution) |
| ctx_emb dimension | 3 (tiny) | Forces z to carry variance |
| Distribution | Student-t | Captures fat tails (kurtosis ~56% recovery) |
| Two-stage training | Yes | Decouples representation learning from forecasting |
| Horizon | 30 days | Balance between accuracy and practical use |

---

## Model Performance

### Single-Step Reconstruction

| Metric | Value |
|--------|-------|
| Kurtosis Recovery | ~56% (per horizon step) |
| 90% CI Violations | ~10% (well-calibrated) |
| Z Contribution | 39.5% |

### Autoregressive Chaining (75 days)

| Hop | Real Context | Kurtosis | RMSE |
|-----|--------------|----------|------|
| 1 | 30 days | 6.6 | 0.35 |
| 2 | 15 days | 8.0 | 0.30 |
| 3 | 0 days | 3.9 | 0.25 |
| 4 | 0 days | 2.8 | 0.25 |

**Key finding**: Chaining is stable - no collapse even with all-synthetic context.

---

## Code Locations

| Component | File | Class |
|-----------|------|-------|
| Full model | `vae/cvae_two_stage.py` | `CVAETwoStageStudentTMLP` |
| Student-t decoder | `vae/cvae_two_stage.py` | `StudentTMLPDecoder` |
| Context encoder | `vae/cvae_two_stage.py` | `ContextEncoder` |
| Main encoder | `vae/cvae_two_stage.py` | `MainEncoder` |
| Latent predictor | `vae/predictors.py` | `LatentPredictor` |
| Config | `config/two_stage_config.py` | `TwoStageConfig` |

---

## Limitations

1. **No volatility clustering**: MLP decoder is position-independent
2. **~56% kurtosis recovery**: Student-t helps but doesn't fully match GT tails
3. **Extreme events**: Model won't capture black swan events (e.g., Lehman spike)
4. **30-day horizon**: Designed for 30-day forecasts; longer requires chaining

---

## References

- Main paper: Chen et al., "A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces", JFDS 2025
- Student-t VAE: Takahashi et al., "Student-t VAE"
- Two-stage training: Similar to VQ-VAE approach
