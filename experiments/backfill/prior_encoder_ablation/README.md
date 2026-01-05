# Prior Encoder Ablation Experiment

## Problem Statement

**Current Architecture Issue:** The context encoder receives confounded gradients from TWO sources:
1. Reconstruction loss (through decoder)
2. KL loss (through prior network)

This causes:
- Conditional variance ratio E[Var(X|C)] / Var(X) = **0.44%** (near zero!)
- CI coverage at H=90: ~67% (should be 90%)
- Prior is "too context-specific" (r=0.89 correlation)

**Hypothesis:** A "Prior Encoder" that processes raw context independently will produce cleaner gradient flow and better conditional variance.

---

## Experiment Design: 3 Variants

| Variant | Prior Input | Output Type | Gradient Flow | File |
|---------|-------------|-------------|---------------|------|
| **Baseline** | context_summary (B, 12) | Full Cov AR(1) | Confounded | CVAEFullCovPrior |
| **Prior Encoder Diagonal** | raw_context (B, 60, 5, 5) | Per-timestep (μ, log_var) | Clean | CVAEWithPriorEncoderDiagonal |
| **Prior Encoder Full Cov** | raw_context (B, 60, 5, 5) | μ + AR(1) covariance | Clean | CVAEWithPriorEncoderFullCov |

---

## Quick Start

### Train All Variants

```bash
# Train baseline
python experiments/backfill/prior_encoder_ablation/train_ablation.py --variant baseline

# Train prior encoder diagonal
python experiments/backfill/prior_encoder_ablation/train_ablation.py --variant prior_encoder_diagonal

# Train prior encoder full cov
python experiments/backfill/prior_encoder_ablation/train_ablation.py --variant prior_encoder_full_cov
```

### Checkpoints and Results

Each variant saves to:
```
models/backfill/prior_encoder_ablation/{variant}/checkpoints/
  - prior_encoder_ablation_{variant}_phase1_ep99.pt
  - prior_encoder_ablation_{variant}_phase2_ep399.pt
  - prior_encoder_ablation_{variant}_best.pt

results/prior_encoder_ablation/{variant}/
  - (evaluation results will go here)
```

---

## Architecture Details

### Baseline (Current Design)

```
Raw Context (B, C=60, 5, 5)
        |
        v
+-----------------------------------------------------------+
|  CONTEXT ENCODER (CVAECtxMemRandEncoder)                   |
|  - Conv2D [32, 64, 128] -> LSTM(128) -> compress(12)      |
|  - Receives gradients from BOTH:                           |
|    ✗ Reconstruction loss (via decoder)                     |
|    ✗ KL loss (via prior network)                           |
|  OUTPUT: ctx_embedding (B, C, 12)                          |
+-----------------------------------------------------------+
        |
        v Last timestep: ctx_embedding[:, -1, :] -> (B, 12)
        |
+-----------------------------------------------------------+
|  FULL COVARIANCE PRIOR (FullCovariancePrior)               |
|  - Input: context_summary (B, 12) <- JUST 12-DIM VECTOR!   |
|  - Architecture: MLP/LSTM over 12-dim                      |
|  OUTPUT: mu_p (B, H, 12), Sigma_p (H, H)                   |
+-----------------------------------------------------------+
```

**Problems:**
- Information bottleneck (12-dim summary)
- Confounded gradients
- Prior too context-specific

### Prior Encoder Diagonal

```
Raw Context (B, C=60, 5, 5)
        |
        +---------------------------------------------+
        v                                             v
+------------------------+     +-------------------------------+
|  CONTEXT ENCODER       |     |  PRIOR ENCODER (NEW)          |
|  - Conv2D [32,64,128]  |     |  - Conv2D [16,32,64] (50% cap)|
|  - LSTM(128)           |     |  - LSTM(64)                   |
|  - Gradients from      |     |  - Gradients from KL ONLY      |
|    RECON only          |     |  - Position encoding (32-dim)  |
+------------------------+     +-------------------------------+
        |                                     |
        v                                     v
   ctx_embedding                      mu_p (B, H, 12)
   (for decoder)                      log_var_p (B, H, 12)
```

**Benefits:**
- Clean gradient separation
- More information (1500 values vs 12)
- Per-timestep independent variance

### Prior Encoder Full Cov

```
Raw Context (B, C=60, 5, 5)
        |
        +---------------------------------------------+
        v                                             v
+------------------------+     +-------------------------------+
|  CONTEXT ENCODER       |     |  PRIOR ENCODER (NEW)          |
|  - Conv2D [32,64,128]  |     |  - Conv2D [16,32,64] (50% cap)|
|  - LSTM(128)           |     |  - LSTM(64)                   |
|  - Gradients from      |     |  - Position encoding (32-dim)  |
|    RECON only          |     |  - AR(1) covariance (φ, σ²)   |
+------------------------+     +-------------------------------+
        |                                     |
        v                                     v
   ctx_embedding                      mu_p (B, H, 12)
   (for decoder)                      Sigma_p (H, H)
```

**Benefits:**
- Clean gradient separation
- More information (1500 values vs 12)
- Temporal correlation via AR(1)

---

## Training Schedule

**Phase 1 (Epochs 0-99):** Teacher Forcing (H=1)
- Sequence length: 62 (60 context + 1 horizon + 1 for slicing)
- Focus: Learn basic reconstruction

**Phase 2 (Epochs 100-399):** Multi-Horizon [1, 30, 60, 90]
- Sequence length: 151 (60 context + 90 horizon + 1)
- Focus: Learn multi-step predictions

**Training Time:** ~38 sec/epoch (all variants should be similar)

---

## Evaluation Metrics

### 1. Conditional Variance Ratio

**Current:** E[Var(X|C)] / Var(X) = 0.44% (near zero!)
**Target:** >10%

**Method:**
- For each context C, generate N=1000 samples
- Compute within-context variance Var(X|C)
- Average across contexts: E[Var(X|C)]
- Compare to total variance Var(X)

### 2. CI Coverage

**Current:** ~67% at H=90
**Target:** 90%

**Method:**
- Generate N=1000 samples per context
- Compute empirical p05, p95 quantiles
- Check if ground truth within [p05, p95]
- Report coverage rate

### 3. Path Roughness

**Current:** 9.7% (vs 75% for oracle)
**Target:** >40%

**Method:**
- Measure volatility of generated trajectories
- Compare to ground truth roughness

---

## Expected Outcomes

| Metric | Baseline | Prior Enc Diagonal | Prior Enc Full Cov |
|--------|----------|-------------------|-------------------|
| **Conditional Variance Ratio** | 0.44% | >5% | >10% |
| **CI Coverage at H=90** | ~67% | ~75% | ~85% |
| **Gradient Flow** | Confounded | Clean | Clean |
| **Path Diversity** | Low | Medium | High |
| **Training Time** | 1x | ~1.5x | ~1.5x |

---

## Implementation Files

### Core Classes

```
vae/
  prior_encoder.py              # PriorEncoderBase, Diagonal, FullCov
  cvae_prior_encoder.py         # CVAEWithPriorEncoderDiagonal, CVAEWithPriorEncoderFullCov
  full_covariance_prior.py      # FullCovariancePrior (baseline)
  cvae_full_cov_prior.py        # CVAEFullCovPrior (baseline)

config/
  prior_encoder_ablation_config.py  # Configuration for all variants

experiments/backfill/prior_encoder_ablation/
  train_ablation.py             # Training script
  README.md                     # This file
```

### Evaluation Scripts (To Be Created)

```
experiments/backfill/prior_encoder_ablation/
  evaluate_conditional_variance.py  # E[Var(X|C)] / Var(X)
  evaluate_ci_coverage.py           # CI coverage across horizons
  visualize_paths.py                # Path comparison vs ground truth
  compare_variants.py               # Summary table and plots
```

---

## Next Steps

1. **Train all three variants** using the commands above
2. **Create evaluation scripts** to measure conditional variance, CI coverage, and path quality
3. **Generate comparison tables** and visualizations
4. **Analyze results** to confirm whether Prior Encoder architecture improves conditional path generation

---

## Rationale

### Why This Should Work

**1. Clean Gradient Flow**
- Context encoder optimizes only for reconstruction (what decoder needs)
- Prior encoder optimizes only for KL (learning appropriate uncertainty)
- No conflicting objectives

**2. More Information**
- Prior encoder sees full context (60 × 5 × 5 = 1500 values)
- vs context summary (12 values)
- Can learn more nuanced patterns

**3. Reduced Over-Discrimination**
- Scaled-down architecture (~50% capacity)
- Prevents prior from being too context-specific
- Encourages grouping similar contexts

**4. Independent Training**
- Each encoder learns for its own objective
- No forced trade-offs between reconstruction and uncertainty modeling
- Should lead to better specialization

---

## References

See main documentation:
- `/home/max/Documents/vol-surface-vae-pub/CONDITIONAL_DISTRIBUTION_FRAMEWORK.md`
- `/home/max/Documents/vol-surface-vae-pub/CONDITIONAL_VARIANCE_SOLUTIONS.md`
- Plan file: `/home/max/.claude/plans/replicated-shimmying-mango.md`
