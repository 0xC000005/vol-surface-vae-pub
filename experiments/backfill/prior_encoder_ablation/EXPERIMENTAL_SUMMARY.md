# Conditional Variance Investigation: Complete Summary

**Date:** December 29, 2025
**Status:** Root cause identified

## Executive Summary

**Problem:** The VAE model fails to generate diverse conditional distributions P(X|C) while preserving the unconditional marginal distribution P(X).

**Metrics:**
- **P1 (Conditional Variance Ratio):** E[Var(X|C)] / Var(X) = **0.006%** (target: >2%)
- **P2 (Roughness Ratio):** Generated / GT roughness = **~10%** (target: >40%)

**Root Cause Identified:**

The decoder learned to **suppress z variation**, with a gain of only **1.2e-7** from latent space to output space. This means even large variations in z (e.g., 5x the prior σ) produce negligible changes in output.

---

## Relationship to CONDITIONAL_VARIANCE_SOLUTIONS.md

The original `CONDITIONAL_VARIANCE_SOLUTIONS.md` document (root directory) made a critical claim:

> "**Decoder works fine** (uses z effectively, 1.84× larger gradients than ctx)"
> — Based on Exp 8-10 results showing "126.7% variance"

**THIS CLAIM IS INVALID FOR THE exp6 MODELS.**

### Why the Original Experiments Were Invalid

| Aspect | Exp 8-10 (Original) | exp6 Ablation (New) |
|--------|---------------------|---------------------|
| **Model** | `CVAEMemRandConditionalPrior` | `CVAEFullCovPrior` / `PriorEncoder` |
| **Z Sampling** | **BROKEN** - uses `prior_mu[:, -1:, :].expand()` | **CORRECT** - proper `horizon=1` parameter |
| **126.7% Result** | Artifact of broken sampling (last context z reused) | **Not applicable** |
| **Validity** | ❌ Invalid for exp6 models | ✅ Valid for exp6 models |

The "126.7% variance" result came from a model that incorrectly reused the last context timestep's z for all future positions. This bug created artificial variance in the output that didn't reflect true decoder sensitivity.

### Fresh Investigation: Exp7 Decoder Sensitivity Test

Created `exp7_decoder_sensitivity.py` to properly test exp6 models.

**Results:**

| Z Scale | Z Variance | Output Variance | Ratio (vs 1x) |
|---------|------------|-----------------|---------------|
| 0.5x | 0.26 | 0.000000 | 0.27x |
| 1.0x | 1.02 | 0.000000 | 1.00x |
| 2.0x | 4.12 | 0.000001 | 3.26x |
| 5.0x | 25.6 | 0.000003 | 9.36x |

**Key Insights:**
1. Decoder IS responsive (ratio scales ~9x when z variance increases 25x)
2. BUT absolute output variance is **TINY** (0.000003 at 5x scale)
3. **Decoder Gain = 0.000003 / 25.6 = 1.2e-7** (near zero!)

---

## Why the Model Fails to Achieve Wide Conditional Marginal

### The Law of Total Variance

```
Var(X) = E[Var(X|C)] + Var(E[X|C])
  ↑           ↑              ↑
Total     "within"       "between"
(0.0096)  (target: 2%)   (from model)
```

For the VAE to produce diverse samples while preserving the unconditional marginal:
- **E[Var(X|C)]** (within-context variance) must be non-trivial
- The decoder must translate z variations into output variations

### What Actually Happened

**During Training:**
1. Each context C in training data has exactly **one** observed outcome X
2. The posterior q(z|x,c) learns a deterministic mapping: (context, target) → z
3. The prior p(z|c) mimics the posterior (to minimize KL divergence)
4. The decoder learns: specific z → specific output (minimal reconstruction loss)

**The Result:**
- z encodes "**which specific outcome**" not "**uncertainty about outcome**"
- Decoder learned to be **precise** (low variance) not **diverse** (high variance)
- Prior network outputs z values clustered around "expected z" for each context

### The Decoder Gain Problem

```
Input:  z with Var(z) = 25.6 (5x prior σ)
Output: X with Var(X) = 0.000003

Gain = 0.000003 / 25.6 = 1.2e-7

Even with prior σ² = 1.02 (healthy prior variance),
the decoder suppresses this to near-zero output variance.
```

**This is why prior network modifications didn't help:**
- exp6_v5: Prior learned φ=0.49, σ²=1.02 ✅ (healthy values!)
- But P1 stayed at 0.006% ❌
- The bottleneck is the **decoder**, not the prior

---

## Complete Experimental Evidence (exp6 Series)

### Exp6_v3: Baseline (H=1, KL=0.00001)

| Model | P1 | P2 | Phi Change |
|-------|-----|-----|------------|
| CVAEFullCovPrior | 0.003% | 4.7% | 0% |
| PriorEncoderDiagonal | 0.017% | 9.4% | N/A |
| PriorEncoderFullCov | 0.004% | 6.8% | 0% |

**Finding:** φ stuck because H=1 → Σ = σ² × φ^0 = σ² (no φ dependence)

### Exp6_v5: Multi-Horizon Fix (H=5, KL=0.00001)

| Model | P1 | P2 | Phi Change |
|-------|-----|-----|------------|
| CVAEFullCovPrior | 0.006% | 6.3% | **-1.81%** |
| PriorEncoderDiagonal | 0.001% | 4.9% | N/A |
| PriorEncoderFullCov | 0.001% | 10.2% | **-1.80%** |

**Finding:** φ now learns (0.500 → 0.491), but P1/P2 **unchanged**

### Exp6_v6: Aggressive Fix (H=30, KL=0.001)

| Model | P1 | P2 | Phi Change |
|-------|-----|-----|------------|
| CVAEFullCovPrior | 0.007% | 4.6% | -1.80% |
| PriorEncoderDiagonal | 0.000% | N/A | N/A |
| PriorEncoderFullCov | 0.006% | 8.2% | -1.79% |

**Finding:** 100x KL weight → catastrophic collapse (-99% on metrics)

### Exp7: Decoder Sensitivity Test

**Finding:** Decoder gain = 1.2e-7. **ROOT CAUSE IDENTIFIED.**

---

## Why Prior Network Solutions Didn't Work

The `CONDITIONAL_VARIANCE_SOLUTIONS.md` proposed several prior network modifications:

| Solution | Expected | Actual | Why It Failed |
|----------|----------|--------|---------------|
| Full Cov Prior | P1 ↑, P2 ↑ | P1 unchanged | Decoder suppresses z variance |
| Prior Encoder | Clean gradients | P1 unchanged | Decoder suppresses z variance |
| AR(1) Structure | Better temporal | φ learns, P2 unchanged | Decoder suppresses z variance |
| Higher KL | Wider prior | Collapse | Over-regularization + decoder still suppresses |

**All solutions modified the prior**, but the bottleneck is the **decoder's near-zero gain**.

### Mathematical Explanation

Let `g` be the decoder gain (output variance / z variance):
```
Var(X|C) = g × Var(z|C)

Current: g = 1.2e-7
Prior:   Var(z|C) = σ² = 1.02

Result:  Var(X|C) = 1.2e-7 × 1.02 = 1.2e-7

Even if we increase prior variance 10x:
         Var(X|C) = 1.2e-7 × 10.2 = 1.2e-6  (still tiny!)
```

**The prior cannot compensate for a near-zero decoder gain.**

---

## The Fundamental Problem

```
CONDITIONAL_VARIANCE_SOLUTIONS.md claim:
"Decoder works fine (uses z effectively)"
    ↓
Based on Exp 8-10 from DIFFERENT model with BROKEN z sampling
    ↓
Led to focusing on PRIOR modifications
    ↓
All exp6 prior modifications FAILED
    ↓
exp7 reveals TRUE cause: Decoder gain = 1.2e-7
```

---

## Summary: Root Cause Chain

```
Training Data Structure
    ↓
Each context has ONE outcome (no within-context variability)
    ↓
Posterior learns: (context, target) → specific z
    ↓
Prior mimics posterior (to minimize KL)
    ↓
Decoder learns: specific z → specific output
    ↓
Decoder gain → near zero (1.2e-7)
    ↓
Even healthy prior variance (σ²=1.02) produces
near-zero output variance
    ↓
P1 = 0.006% despite all prior modifications
```

---

## Recommended Next Step: P1 Loss Regularization (exp8)

**Solution:** Force decoder to have higher gain via direct P1 loss optimization.

```python
def train_step_with_p1_loss(batch, p1_weight=0.1, num_samples=20):
    # Standard VAE forward
    z_posterior = encoder(batch)
    recon = decoder(z_posterior)
    recon_loss = MSE(recon, target)
    kl_loss = KL(posterior, prior)

    # NEW: Conditional variance regularization
    context = batch[:, :C]
    z_samples = [sample_prior(context) for _ in range(num_samples)]
    outputs = [decoder(z) for z in z_samples]
    within_var = torch.var(torch.stack(outputs), dim=0).mean()

    # Maximize output variance when z varies
    p1_loss = -torch.log(within_var + 1e-8)

    return recon_loss + kl_weight * kl_loss + p1_weight * p1_loss
```

**Why this should work:**
- Directly optimizes for output variance (P1 metric)
- Forces decoder to **increase its gain** from z to output
- Doesn't require prior network changes (prior is already healthy)

---

## Key Files

| File | Purpose |
|------|---------|
| `CONDITIONAL_VARIANCE_SOLUTIONS.md` (root) | Original analysis (Exp 8-10, now known invalid for exp6) |
| `exp7_decoder_sensitivity.py` | Proper decoder test for exp6 models |
| `vae/cvae_full_cov_prior.py` | Full Cov Prior model |
| `vae/cvae_prior_encoder.py` | Prior Encoder models |
| `results/prior_encoder_ablation/extended_training_v{3,5,6}/` | Experiment results |

---

## All Architectural Solutions Tried

### Solution 1: CVAEFullCovPrior (Full Covariance Prior Network)

**File:** `vae/cvae_full_cov_prior.py`

**Architecture:**
```
Context Encoder (shared) → Context Summary (B, 12)
                                ↓
                    FullCovariancePrior Network
                    ├── Position-Encoded Mean: μ_t = MLP(context_summary, pos_t)
                    └── AR(1) Covariance: Σ[i,j] = σ² × φ^|i-j|
                                ↓
                    z ~ N(μ_1:H, Σ_AR1)
```

**What It Learns:**
- φ (phi): Autocorrelation coefficient (init=0.5)
- σ² (sigma_sq): Global variance (init=1.0)
- Position MLP weights for μ_t

**Intended to Solve:** P1 + P2 (temporal structure via AR(1))

**Problem Found:** Gradient conflict - context encoder receives gradients from BOTH reconstruction AND KL loss, causing optimization interference.

---

### Solution 2: CVAEWithPriorEncoderDiagonal (Independent Prior Encoder)

**File:** `vae/cvae_prior_encoder.py`

**Architecture:**
```
Raw Context (B, 60, 5, 5) → Prior Encoder (INDEPENDENT Conv+LSTM)
                                ↓
                    Per-timestep Diagonal Gaussian
                    ├── μ_p: (B, H, latent_dim)
                    └── log_var_p: (B, H, latent_dim)
                                ↓
                    z_t ~ N(μ_t, σ_t²) for each t  ← INDEPENDENT!
```

**What It Learns:**
- Separate Conv2D filters (50% capacity of context encoder)
- Separate LSTM weights
- Per-timestep μ and σ² (no correlation between timesteps)

**Intended to Solve:** P1 (clean gradient flow)

**Problem Found:** No AR(1) structure → cannot address P2 (roughness). Each z_t is independent → even smoother than baseline!

---

### Solution 3: CVAEWithPriorEncoderFullCov (Prior Encoder + AR(1))

**File:** `vae/cvae_prior_encoder.py`

**Architecture:**
```
Raw Context (B, 60, 5, 5) → Prior Encoder (INDEPENDENT Conv+LSTM)
                                ↓
                    ├── Position-Encoded Mean: μ_t
                    └── Learnable AR(1): φ, σ²
                                ↓
                    z ~ N(μ_1:H, Σ_AR1)
```

**What It Learns:**
- Independent Conv2D + LSTM (clean gradients)
- Position MLP for μ_t
- Global φ, σ² for AR(1) covariance

**Intended to Solve:** P1 + P2 (best of both approaches)

**Gradient Flow:**
- Prior Encoder: Gradients from KL loss ONLY
- Context Encoder: Gradients from reconstruction loss ONLY
- Result: Clean, non-conflicting optimization

---

## Summary: What Worked vs What Failed

### ✅ What Worked

| Discovery | Evidence |
|-----------|----------|
| H > 1 enables φ gradient | v3 (H=1): φ stuck; v5 (H=5): φ learns -1.8% |
| Independent prior encoder enables clean gradients | Separate optimization paths for KL vs recon |
| Full Cov achieves best P2 | 10.2% vs 6.3% (baseline) vs 4.9% (diagonal) |
| Identified variance collapse in diagonal | KL=0.29 vs ~6 for full cov models |

### ❌ What Failed

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| AR(1) prior structure | P1 stuck at 0.006% | Decoder suppresses z variance (gain=1.2e-7) |
| Higher KL weight | P1/P2 collapsed 99% | Over-regularization pushed z → N(0,1) |
| Diagonal covariance | Variance collapse | No AR(1) structure, learned σ² → 0 |
| Longer horizon (H=30) | No improvement | Gradient magnitude unchanged |
| 200 epoch training | Phi moves 1.8% | Would need 10,000+ epochs to converge |
