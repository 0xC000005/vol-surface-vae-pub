# Why Autoregressive Chaining Fails: Root Cause Analysis

## Executive Summary

This document explains why autoregressive chaining with the Student-t VAE produces poor results that don't track ground truth patterns. The investigation revealed a **fundamental architecture mismatch**: the model was designed for reconstruction, not prediction.

**Key Finding**: The model has essentially **zero predictive power** for the next time step because the architecture allows the encoder to "see" the target during training.

---

## Problem Statement

### Observed Behavior

When generating 30-day IV surface trajectories via autoregressive chaining:

1. **Predictions don't track GT patterns**: Model median stays flat while GT shows market movements
2. **Fan charts don't fan properly**: Trajectories collapse to near-constant values
3. **IV explosion**: Without temperature scaling, values explode to 7000+ (should be ~0.8)

### Initial Hypotheses (All Wrong)

| Hypothesis | Evidence Against |
|------------|------------------|
| Mean bias in decoder | Fresh model with bias 0.03 still explodes |
| Fat-tailed sampling | Temperature scaling prevents explosion but predictions still don't track GT |
| Prior network not used | Using prior network gives even worse correlation (-0.007) |

---

## Root Cause: Reconstruction vs Prediction Architecture

### The Architecture Issue

The VAE has two encoders:

```
Context Encoder: ctx_emb[t] = f(x[0:t-1])  ← CAUSAL (good)
Main Encoder:    z[t] = g(x[0:t])          ← NON-CAUSAL (problem!)
```

At position `t`, the main encoder `z[t]` **sees input `x[t]`**. This means:

1. **During training**: The model can use `z[t]` to reconstruct `x[t]` (trivial)
2. **During inference**: Without `x[t]`, we must sample `z` from prior → output near-zero

### Training Loss Analysis

The model was trained with:
```python
# Loss on horizon positions only
loss = MSE(output[30:60], input[30:60])  # Reconstruction, NOT prediction
```

At position 30:
- `ctx_emb[30]` sees `x[0:29]` (context only) ✓
- `z[30]` sees `x[0:30]` (includes target!) ✗
- Target is `x[30]` (reconstruction)

The model was **never trained to predict `x[t+1]` from `x[0:t]`**.

---

## Quantitative Evidence

### Test 1: Single-Step Prediction Correlation

```
Method                          | Correlation with Next Step
--------------------------------|---------------------------
VAE forward() output[-1]        | -0.26 (weak negative)
Prior Network prediction        | -0.007 (essentially zero)
Ground truth autocorrelation    | -0.11 (weak mean-reversion)
```

**Conclusion**: Neither method has predictive power.

### Test 2: Output Variance Collapse

```
                    | Predicted std | Actual std | Ratio
--------------------|---------------|------------|-------
VAE reconstruction  | 0.0098        | 0.0314     | 3.2x smaller
Prior Network       | 0.0044        | 0.0394     | 9.0x smaller
```

**Conclusion**: Model outputs near-zero values (no variance).

### Test 3: Prior Network vs Encoder z

```
Metric                              | Value
------------------------------------|----------------
z_encoder std                       | 1.22
z_prior std                         | 0.52 (2.4x smaller)
Per-dimension correlation           | 0.24 - 0.51
Decoded prediction vs actual corr   | -0.17 (no signal)
```

**Conclusion**: Prior network outputs compressed z values near zero.

---

## Why the Prior Network Doesn't Help

The prior network was trained to minimize:
```
MSE(z_prior, z_encoder)
```

Where `z_encoder` comes from the VAE's main encoder which **sees the target**.

This creates a paradox:
1. Prior network tries to predict what encoder would output
2. Encoder has target information that prior doesn't have
3. Best prior can do is output `E[z|context]` ≈ 0

The prior network **cannot predict the actual next step** because:
- It was trained to match encoder z, not predict log-returns
- Encoder z contains target information unavailable at inference
- Without that information, prior defaults to near-zero output

---

## Information Flow Diagram

```
TRAINING (with target):
┌─────────────────────────────────────────────────────────────┐
│ Input: [x_0, x_1, ..., x_29, x_30, ..., x_59]              │
│                              ↑                              │
│                        z[30] sees this!                     │
│                                                             │
│ Output[30] = f(ctx_emb[30], z[30])                         │
│            = f(context_info, target_info)                   │
│            → Can reconstruct x[30] easily                   │
└─────────────────────────────────────────────────────────────┘

INFERENCE (without target):
┌─────────────────────────────────────────────────────────────┐
│ Input: [x_0, x_1, ..., x_29]                               │
│                                                             │
│ z ~ N(0, I) or z = prior(context)                          │
│                                                             │
│ Output[30] = f(ctx_emb[30], z_sampled)                     │
│            = f(context_info, ?????)                         │
│            → Without target info, outputs ~0                │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Temperature Scaling Doesn't Fix It

Temperature scaling (from STABLE_CHAINING.md) prevents explosion but doesn't add predictive power:

| Method | Explodes? | Tracks GT? | Why |
|--------|-----------|------------|-----|
| Original | Yes | No | Fat-tailed extreme values compound |
| Temperature 0.3 | No | No | Suppresses variance, still no prediction signal |
| Mean-only | No | No | Deterministic ~0, no variance |
| Truncated | No | No | Clips extremes, still ~0 predictions |

All methods produce stable but **uninformative** trajectories.

---

## Solution Options

### Option 1: Causal VAE (Architectural Fix)

Modify the main encoder to be causal:
```python
# Current (non-causal):
z[t] = encoder(x[0:t])  # Sees x[t]

# Fixed (causal):
z[t] = encoder(x[0:t-1])  # Only sees past
```

**Pros**: Minimal code change, preserves VAE framework
**Cons**: May reduce reconstruction quality

### Option 2: Direct Prediction Model

Train a model to directly predict `x[t+1]`:
```python
# New loss function
target = input[:, 1:]      # Shifted by 1
pred = output[:, :-1]      # Predict next step
loss = MSE(pred, target)
```

**Pros**: Clear prediction objective
**Cons**: Loses VAE uncertainty quantification

### Option 3: MIMO Forecasting (Recommended)

Predict all H steps in one forward pass:
```python
# Input: context [x_0, ..., x_C-1]
# Output: horizon [x_C, ..., x_C+H-1]
# No recursive feedback = no error compounding
```

**Pros**:
- Eliminates autoregressive error accumulation
- Well-supported in forecasting literature
- Can use existing VAE architecture with modified loss

**Cons**:
- Single forward pass, can't extend beyond trained horizon
- Requires retraining

### Option 4: Autoregressive Transformer

Replace VAE with causal transformer:
```python
# GPT-style autoregressive prediction
x[t+1] = transformer(x[0:t])
```

**Pros**: State-of-the-art for sequence prediction
**Cons**: Complete architecture change

---

## Diagnostic Scripts

| Script | Purpose |
|--------|---------|
| `diagnose_model_output.py` | Compare reconstruction vs prediction |
| `diagnose_model_output_v2.py` | Multi-sample correlation analysis |
| `diagnose_training_target.py` | Verify training loss target |
| `diagnose_prior_network.py` | Analyze prior network predictions |
| `visualize_chaining_with_prior.py` | Test chaining with prior network |

---

## Conclusions

1. **The model has no predictive power** for autoregressive chaining
2. **Root cause**: Non-causal encoder allows reconstruction but not prediction
3. **Prior network doesn't help**: It was trained to predict encoder z, not actual future
4. **Temperature scaling is a band-aid**: Prevents explosion but doesn't add signal
5. **Architectural change required**: Need causal architecture or MIMO approach

---

## References

- [Stratify: Unifying Multi-Step Forecasting](https://arxiv.org/html/2412.20510v1) - MIMO vs recursive
- [Machine Learning Strategies for Multi-step Forecasting](https://souhaib-bentaieb.com/papers/2014_phd.pdf) - Forecasting strategies
- [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363) - Modern forecasting architecture
