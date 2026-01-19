# Two-Stage VAE Decoder: Architecture & Engineering Decisions

## Overview

This document describes the decoder architecture for the Two-Stage Cumulative-Aware VAE and summarizes experimental findings from Options 1-6 for preventing IV explosion during autoregressive chaining.

---

## Decoder Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DECODER INPUTS                               │
├─────────────────────────────────────────────────────────────────────┤
│  ctx_emb (B, T, 3)        Context embedding from LSTM encoder       │
│  z (B, T, 16)             Latent variable (sampled from posterior)  │
│  prev_x (B, T, 5, 5)      Previous log-return (for AR correction)   │
│  prev_log_iv (B, T, 5, 5) Previous log-IV level (multi-task only)   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    1. MEAN PREDICTION PATHWAY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ctx_emb ──► ctx_mean_net ──► ctx_mean (B, T, 5, 5)                │
│               MLP [3 → 64 → 32 → 25]                                │
│                                                                      │
│   z ──► z_shared_net ──► return_head ──► return_pred (B, T, 5, 5)   │
│         MLP [16→128→64]  Linear [64→25]                             │
│                                                                      │
│   mean_base = ctx_mean + return_pred                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    2. AR(1) CORRECTION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   φ = tanh(phi_logit)              # Learned, bounded to (-1, 1)    │
│   μ = learned per-grid mean        # 25 learnable parameters        │
│                                                                      │
│   ar_correction = φ × (prev_x - μ)                                  │
│   return_mean = mean_base + ar_correction                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    3. COVARIANCE PREDICTION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   z_pooled = z.mean(dim=1)         # Pool across time dimension     │
│                                                                      │
│   factor = factor_net(z_pooled)    # Low-rank: (B, 25, rank=4)      │
│            MLP [16 → 64 → 100]                                      │
│                                                                      │
│   log_diag = log_diag_net(z_pooled) # Diagonal: (B, 25)             │
│              MLP [16 → 64 → 25], clamped to [-10, 2]                │
│                                                                      │
│   Covariance: Σ = FF^T + diag(exp(log_diag))                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    4. STUDENT-T SAMPLING                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ν[i,j] = GT_NU_MLP[i,j]          # Fixed per-grid DOF (4 to 30)   │
│                                                                      │
│   # Correlated noise via low-rank factor                            │
│   eps_rank ~ N(0, I_rank)                                           │
│   correlated = einsum('bir,btr->bti', factor, eps_rank)             │
│                                                                      │
│   # Independent noise via diagonal                                  │
│   eps_diag ~ N(0, I_25)                                             │
│   independent = sqrt(exp(log_diag)) × eps_diag                      │
│                                                                      │
│   # Convert Gaussian to Student-t                                   │
│   chi2[i] ~ Gamma(ν[i]/2, ν[i]/2)                                   │
│   student_t_factor = 1 / sqrt(chi2)                                 │
│   noise = (correlated + independent) × student_t_factor             │
│                                                                      │
│   sample = return_mean + noise.view(B, T, 5, 5)                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Engineering Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| **Mean pathway** | ctx + z (additive) | Context provides baseline prediction, z adds stochastic variation |
| **AR(1) φ** | Learned via `tanh(phi_logit)` | Real data has ACF(1) ≈ -0.35; tanh bounds φ to (-1, 1) |
| **Long-run mean μ** | 25 learnable params | Each grid point can have different equilibrium |
| **Covariance** | Low-rank (r=4) + diagonal | Captures cross-grid correlation efficiently (125 vs 325 params) |
| **Distribution** | Student-t | Matches fat-tailed empirical distribution (kurtosis 4-40) |
| **ν (DOF)** | Fixed from data | Per-grid kurtosis varies significantly across moneyness/tenor |
| **Architecture** | MLP (not LSTM/CNN) | LSTM causes z wash-out (3.8% contribution vs 39.5% with MLP) |

### Why MLP Instead of LSTM Decoder?

Previous experiments showed LSTM decoder causes **z wash-out**:
- With LSTM: z contributes only 3.8% of output variance
- With MLP: z contributes 39.5% of output variance

The LSTM's recurrent structure allows it to ignore z and rely solely on its hidden state, collapsing the VAE to a deterministic autoencoder.

### Per-Grid Degrees of Freedom (GT_NU_MLP)

The degrees of freedom for Student-t sampling are fixed based on empirical kurtosis:

```python
GT_NU_MLP = [
    [4.22, 5.95, 7.05, 4.89, 4.33],   # Short tenor: fat tails
    [4.45, 7.83, 9.75, 6.62, 4.23],
    [5.48, 10.6, 14.1, 11.1, 4.16],   # ATM [2,2]: ν=14.1
    [8.35, 16.6, 24.1, 19.9, 7.23],
    [6.55, 23.9, 30.0, 25.9, 6.20],   # Long tenor: lighter tails
]
```

Relationship: `kurtosis ≈ 3 + 6/(ν-4)` for ν > 4

---

## Experimental Options Summary

### Option 1: Cumulative Log-Return Conditioning
- **Idea**: Condition decoder on cumulative log-return to let model learn level-dependent dynamics
- **Implementation**: `StudentTCumulativeDecoder` with `cumul_net` MLP
- **Result**: 89.8% explosion rate, all sequences explode
- **Why it failed**: Model doesn't learn to use cumulative info for mean reversion; the cumulative pathway is additive and doesn't constrain output

### Option 2: Level Conditioning
- **Idea**: Condition on current IV level instead of cumulative returns
- **Implementation**: `StudentTLevelDecoder` with `level_net` MLP
- **Result**: 15.2% explosion rate (best among 1-3)
- **Why it partially works**: Direct level info helps, but no explicit reversion mechanism

### Option 3: Sequence-Level Training (DualPathAR)
- **Idea**: Train on full 30-day sequences with dual encoder paths
- **Implementation**: Chain predictions during training, loss on trajectory
- **Result**: 70.4% explosion rate
- **Why it failed**: Training objective doesn't penalize long-horizon drift; model optimizes for local accuracy

### Option 4: Multi-Task Decoder ⭐
- **Idea**: Predict BOTH log-return AND log-IV level with joint loss
- **Implementation**: `StudentTMultiTaskDecoder` with `return_head` and `level_head`
- **Result**: **13.2% mean explosion rate** (best overall)
- **Key finding**: ATM [2,2] has only **0.0-0.4% explosion rate**
- **Why it works**: Level prediction provides implicit consistency constraint
- **Problem**: OTM corners still explode (74%)

### Option 5: Per-Grid Variance-Scaled Reversion
- **Idea**: Stronger level reversion at high-variance OTM corners
- **Implementation**: `StudentTPerGridReversionDecoder` with variance-scaled `per_grid_reversion`
- **Result**: 89.3% explosion rate (worse than baseline)
- **Why it failed**: Model learned φ = +0.98 to compensate for fixed reversion term
- **Lesson**: Can't add fixed terms without model compensating during training

### Option 6: Spatial CNN + Multi-Task
- **Idea**: Add Conv2d spatial smoothing so ATM stability propagates to OTM
- **Implementation**: `SpatialMultiTaskDecoder` with 3-layer Conv2d stacks
- **Result**: **91.5% explosion rate** (worst)
- **Why it failed**: Bidirectional coupling - OTM instability contaminated ATM
- **Key insight**: MLP independence is a feature, not a bug

---

## Comparative Results Table

| Option | Approach | Mean Explosion % | ATM [2,2] % | Max IV | Learned φ |
|--------|----------|------------------|-------------|--------|-----------|
| 1 | Cumulative conditioning | 89.8% | ~87% | ∞ | ~0.9 |
| 2 | Level conditioning | 15.2% | ~3% | 72M | ~0.8 |
| **4** | **Multi-task** | **13.2%** | **0.0-0.4%** | 2.4M | 0.93 |
| 3 | Sequence training | 70.4% | ~65% | 18B | ~0.9 |
| 5 | Per-grid reversion | 89.3% | ~75% | 10^154 | 0.98 |
| 6 | Spatial CNN | 91.5% | ~89% | 10^154 | 0.99 |

---

## The Core Problem: Training vs Chaining Distribution Shift

```
                    TRAINING                          CHAINING
                    ────────                          ────────
prev_x comes from:  GROUND TRUTH data                MODEL'S OWN predictions
                    (bounded, realistic)              (can drift/explode)

What model sees:    Real returns in [-0.1, +0.1]     Potentially extreme returns
What model learns:  φ ≈ +0.9 (works for training)    φ ≈ +0.9 (amplifies errors)
```

**The model never sees its own errors during training**, so it doesn't learn to correct them.

---

## Why ACF Preservation Doesn't Prevent Explosion

ACF operates in **log-return space**, not **level space**:

```
AR(1) with φ = -0.35:
    E[r_t | r_{t-1}] = φ × r_{t-1}

Example trajectory:
    Day 1: r = +0.40 → IV = 0.15 × exp(0.40) = 0.224
    Day 2: r = -0.14 → IV = 0.224 × exp(-0.14) = 0.195
    Day 3: r = +0.30 → IV = 0.195 × exp(0.30) = 0.263
    Day 4: r = -0.11 → IV = 0.263 × exp(-0.11) = 0.236
    Day 5: r = +0.35 → IV = 0.236 × exp(0.35) = 0.334

Returns alternate sign (ACF preserved!) but IV drifts: 0.15 → 0.33 (+120%)
```

**Partial reversion (φ = -0.35) only reverses 35% of each shock**. The remaining 65% accumulates, causing drift proportional to √T.

---

## Key Lessons Learned

### 1. MLP Independence Prevents Cross-Contamination
When one grid point explodes, it doesn't affect others. Option 4 (MLP multi-task) achieved 0.4% ATM explosion while OTM exploded at 74%. Option 6 (CNN) coupled them, causing ATM to explode at 89%.

### 2. Models Compensate for Fixed Constraints
Adding fixed reversion terms causes the model to learn compensating φ values. Option 5 added per-grid reversion, but the model learned φ = +0.98 to cancel it out.

### 3. Multi-Task Learning Provides Implicit Regularization
Predicting both return AND level constrains the solution space. The level head acts as a consistency check, preventing the return head from making predictions that would lead to unrealistic levels.

### 4. Training-Time Distribution ≠ Inference-Time Distribution
The model is trained on ground truth previous returns but tested on its own predictions. This "exposure bias" is well-known in seq2seq but particularly severe here because errors compound multiplicatively in IV space.

### 5. ACF Preservation is Necessary but Not Sufficient
Mean reversion in returns doesn't prevent level drift. The model needs to understand CUMULATIVE effects, not just local return dynamics.

---

## Architecture Summary by Option

| Option | Decoder Class | Key Components |
|--------|---------------|----------------|
| 1 | `StudentTCumulativeDecoder` | cumul_net MLP for cumulative conditioning |
| 2 | `StudentTLevelDecoder` | level_net MLP for current level conditioning |
| 3 | `CVAETwoStageDualPathAR` | Trajectory-level training loss |
| 4 | `StudentTMultiTaskDecoder` | return_head + level_head (dual output) |
| 5 | `StudentTPerGridReversionDecoder` | per_grid_reversion buffer (variance-scaled) |
| 6 | `SpatialMultiTaskDecoder` | Conv2d spatial smoothing + multi-task |

---

## Files Reference

| File | Description |
|------|-------------|
| `vae/cvae_two_stage.py` | All decoder classes (lines 4600-6053) |
| `experiments/backfill/two_stage_vae/train_cumulative_options.py` | Training script for Options 1-4 |
| `experiments/backfill/two_stage_vae/train_per_grid_reversion.py` | Training script for Option 5 |
| `experiments/backfill/two_stage_vae/train_spatial_multitask.py` | Training script for Option 6 |
| `experiments/backfill/two_stage_vae/comprehensive_evaluation.py` | Evaluation for all 6 options |

---

## Next Steps to Consider

Based on the lessons learned, promising directions include:

1. **Fix φ during training**: Freeze φ = -0.35 to prevent compensation
2. **Scheduled sampling**: Gradually replace ground truth with model predictions during training
3. **Explicit level bounds**: Hard constraints on cumulative log-return
4. **Per-grid constraints on OTM**: Regularize high-variance corners without affecting ATM
5. **Exposure bias correction**: Train with model's own predictions (like in seq2seq)

---

## Glossary

| Term | Definition |
|------|------------|
| **IV** | Implied Volatility |
| **ATM** | At-The-Money (grid point [2,2]) |
| **OTM** | Out-of-The-Money (corner grid points) |
| **Log-return** | log(IV_t / IV_{t-1}) |
| **Cumulative log-return** | sum of log-returns from start |
| **φ (phi)** | AR(1) coefficient for mean reversion |
| **ν (nu)** | Degrees of freedom for Student-t distribution |
| **Explosion** | IV exceeds 2.0 or falls below 0.01 during chaining |
