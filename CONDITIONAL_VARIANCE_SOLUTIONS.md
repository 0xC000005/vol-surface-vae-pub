# Solution Analysis: Diverse Conditional Distributions While Preserving Marginals

## Problem Recap

**Root cause identified from 10 diagnostic experiments:**
- Prior network is too context-specific (correlation r=0.89 between context and prior μ)
- z encodes **between-context** variance, not **within-context** variance
- Similar contexts have 108.8% outcome variance → conditional distribution EXISTS in data
- Decoder works fine (uses z effectively, 1.84× larger gradients than ctx)
- NOT posterior collapse (KL=12.1, z-space utilized at 46%)

**The requirement:**
1. Generate diverse conditional samples P(X|context)
2. Preserve unconditional marginal: ∫ P(X|C)P(C)dC = P(X)

---

## ✅ Prior Parameter Reuse Verification - COMPLETED

### Problem: Constant Prior for All Future Timesteps

The prior network outputs `(B, C, latent_dim)` parameters for **context positions**, but generation logic reuses **only the last timestep** for all H future positions:

```python
# cvae_conditional_prior.py lines 176-177
future_prior_mean = prior_mean[:, -1:, :].expand(B, horizon, -1)      # REUSE!
future_prior_logvar = prior_logvar[:, -1:, :].expand(B, horizon, -1)  # REUSE!
```

**Impact:** All future z values sampled from the SAME distribution → unrealistically smooth paths.

### Verification Experiments Results

**Model:** `backfill_context60_latent12_v3_conditional_prior_phase2_ep599.pt`
**Script:** `experiments/backfill/context60_v3_fixed/verify_prior_reuse_problem.py`
**Results:** `/tmp/prior_reuse_verification_results.txt`

#### Experiment A: Prior Scheme Modifications

| Condition | Roughness Ratio | Improvement vs Baseline |
|-----------|-----------------|-------------------------|
| Baseline (constant reuse) | 9.7% | — |
| Random walk μ | 9.7% | +0% |
| Scaled σ (sqrt) | 10.6% | +0.9% |
| Both | 10.7% | +1.0% |

**Finding:** Simple heuristic modifications provide negligible improvement (<1%).

#### Experiment B: Oracle vs Constant Prior ⭐ KEY FINDING

| Method | Roughness Ratio | Gap |
|--------|-----------------|-----|
| **Oracle (posterior z)** | **75.0%** | — |
| **Constant (current)** | **9.7%** | **-65.3%** |

**Finding:** Using posterior z (which has different values per timestep) achieves **7.7× higher roughness** than constant prior. The 65.3 percentage point gap **definitively proves prior parameter reuse is the bottleneck**.

#### Experiment C: Variance Scaling Functions

| Scale Function | Roughness Ratio | Formula |
|----------------|-----------------|---------|
| None | 9.7% | — |
| Sqrt | 10.5% | σ_t = σ × √(1 + t/H) |
| Linear | 10.7% | σ_t = σ × (1 + 0.5t/H) |
| Log | 17.1% | σ_t = σ × log(log(2+t)) |
| Strong | 11.6% | σ_t = σ × (1 + t/H) |

**Finding:** Even aggressive variance scaling only reaches **17.1%**, far below the 40% target. Mean still constant → paths remain too correlated.

### Success Criteria Assessment

| Metric | Target | Baseline (Constant) | Oracle (Posterior) |
|--------|--------|---------------------|-------------------|
| Roughness ratio | >40% | **9.7%** ❌ | **75.0%** ✅ |
| Autocorrelation | >0.3 | 0.21 ❌ | ~0.3 ✅ |
| Marginal std | Match GT ±10% | ✓ OK | ✓ OK |

**Conclusion:** Prior parameter reuse is confirmed as the root cause. Heuristic fixes insufficient → need architectural modification.

---

## ✅ Temporal Structure Necessity Investigation - COMPLETED

### Question: Is Autoregressive Prior Required for H=90 Horizon?

**Motivation:** The 65.3% gap (9.7% → 75.0%) proves we need timestep-specific prior parameters. For H=90, should we:
- **P1 (Autoregressive):** Sequential RNN (~90× slower inference)
- **P2 (Position-Encoded):** Parallel network with regularization (~1× baseline speed)

**Script:** `experiments/backfill/context60_v3_fixed/confirm_temporal_necessity.py`
**Results:** `/tmp/temporal_necessity_confirmation.txt`
**Date:** 2025-12-23

### Initial Autocorrelation Analysis

**Oracle z temporal correlation (H=30, 60, 90):**

| Horizon | Lag-1 Autocorr | Lag-5 Autocorr | Lag-10 Autocorr |
|---------|----------------|----------------|-----------------|
| H=30 | 0.7083 | 0.6373 | 0.6139 |
| H=60 | 0.7303 | 0.6772 | 0.6476 |
| H=90 | 0.7230 | 0.6687 | 0.6477 |

**Variance structure:**

| Horizon | Temporal Var | Cross-Sample Var | Ratio |
|---------|--------------|------------------|-------|
| H=30 | 0.052 | 0.086 | 0.60 |
| H=60 | 0.052 | 0.080 | 0.64 |
| H=90 | 0.054 | 0.078 | 0.70 |

**Initial conclusion:** HIGH autocorrelation (>0.7) → autoregressive prior necessary?

### Confirmation Experiments - SURPRISING RESULTS! ⭐

#### Experiment 1: Shuffled Oracle Z (Break Temporal Structure)

| Condition | Autocorr | Roughness | Change |
|-----------|----------|-----------|--------|
| Original oracle | 0.70 | 0.0364 | — |
| Shuffled oracle | 0.62 | 0.0398 | **+9.4%** |

**Finding:** Breaking temporal structure **INCREASED** roughness by 9.4%, not decreased it!

**Interpretation:** Temporal correlation makes paths **smoother**, not rougher. The opposite of expected if temporal structure was essential for achieving oracle's 75% roughness.

#### Experiment 2: Synthetic Z with Varying Autocorrelation

| Target Autocorr | Actual Autocorr | Roughness | vs Oracle |
|-----------------|-----------------|-----------|-----------|
| 0.0 (iid) | 0.00 | 0.0504 | **138.5%** (TOO ROUGH) |
| 0.3 | 0.31 | 0.0430 | **118.1%** (rough) |
| 0.5 | 0.49 | 0.0400 | **110.0%** (close!) |
| 0.7 | 0.70 | 0.0317 | **87.1%** (too smooth) |
| Oracle | 0.70 | 0.0364 | **100.0%** (target) |

**Finding:** Clear **inverse relationship** - higher autocorr → lower roughness (smoother paths).

**Critical Insight:** The "Goldilocks zone" is **autocorr ≈ 0.4-0.6**, not 0.7! Oracle has autocorr=0.70 but synthetic z with autocorr=0.50 gets closer to matching oracle roughness.

#### Experiment 3: Post-hoc Smoothing of IID Z

| Method | Autocorr | Roughness | vs Oracle |
|--------|----------|-----------|-----------|
| IID | 0.00 | 0.0506 | 139.0% (TOO ROUGH) |
| Smooth window=3 | 0.66 | 0.0238 | **65.4%** (TOO SMOOTH) |
| Smooth window=5 | 0.80 | 0.0151 | 41.4% (TOO SMOOTH) |
| Smooth window=7 | 0.86 | 0.0114 | 31.3% (TOO SMOOTH) |
| Oracle | 0.70 | 0.0364 | 100.0% (target) |

**Finding:** All smoothing approaches make paths **too smooth**. Even the best (window=3) only achieves 65% of oracle roughness.

#### Experiment 4: Hierarchical Chunks

| Config | Autocorr | Roughness | vs Oracle |
|--------|----------|-----------|-----------|
| 3×10 days | 0.99 | 0.0076 | 20.9% (TOO SMOOTH) |
| 5×6 days | 0.97 | 0.0119 | 32.7% (TOO SMOOTH) |
| 6×5 days | 0.95 | 0.0154 | 42.2% (TOO SMOOTH) |
| Oracle | 0.70 | 0.0364 | 100.0% (target) |

**Finding:** Chunked approaches introduce **too much correlation** (0.95-0.99), making paths unrealistically smooth.

### Key Insights

1. **Temporal correlation ≠ roughness:** Higher autocorr makes paths smoother, not rougher
2. **Goldilocks zone is autocorr ≈ 0.5:** Not 0.7 like oracle z
3. **Post-hoc smoothing fails:** All approaches overshoot and become too smooth
4. **Autoregressive may be overkill:** Oracle achieves roughness despite having high autocorr

### REVISED RECOMMENDATION ⭐

**DO NOT implement P1 (Autoregressive Prior)** - expensive and may not be optimal!

**Instead: P2 (Position-Encoded) + AR(1) Regularization**

```python
# During prior network training, add regularization loss
ar_target_phi = 0.5  # Target autocorrelation in Goldilocks zone

def ar1_regularization_loss(prior_mu):
    """Encourage AR(1)-like temporal structure in prior μ"""
    # μ_t ≈ φ * μ_{t-1} + ε, where φ = target autocorr
    residuals = prior_mu[:, 1:, :] - ar_target_phi * prior_mu[:, :-1, :]
    return torch.mean(residuals ** 2)

# Total loss
total_loss = (reconstruction_loss +
              kl_weight * kl_loss +
              lambda_ar * ar1_regularization_loss(prior_mu))
```

**Why this works:**
1. ✅ Targets optimal autocorr≈0.5 (Goldilocks zone)
2. ✅ Single forward pass (1000× faster than autoregressive)
3. ✅ Trains with standard backprop (no exposure bias)
4. ✅ Expected performance: 110% of oracle roughness (based on Exp 2)

**Expected performance:** Synthetic z with autocorr=0.5 achieved roughness within 10% of oracle. With proper training, should get close to oracle's 75% roughness ratio.

**Computational comparison:**
- P1 (Autoregressive): ~90 seconds for 100×100 samples (sequential)
- P2 + AR(1) Reg: ~1 second for 100×100 samples (parallel)
- **90× speedup** while maintaining similar quality

### Success Criteria for P2 + AR(1) Regularization

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Prior μ autocorr | 0.4-0.6 | Measure lag-1 correlation of prior_mean |
| Generated roughness ratio | >40% | Compare to ground truth paths |
| Inference speed | <2s for 100×100 | Benchmark generation time |

### Files Created

- `experiments/backfill/context60_v3_fixed/investigate_prior_solutions.py` - Initial autocorrelation analysis
- `experiments/backfill/context60_v3_fixed/confirm_temporal_necessity.py` - Confirmation experiments
- `/tmp/prior_solution_viability_results.txt` - Initial analysis results
- `/tmp/temporal_necessity_confirmation.txt` - Confirmation experiment results

---

## ⭐ FINAL SOLUTION: Full Covariance Prior with Cholesky Sampling

**Date:** 2025-12-24
**Status:** APPROVED FOR IMPLEMENTATION

This section documents the complete solution that addresses both Problem 1 (prior too context-specific) and Problem 2 (prior parameter reuse), superseding earlier recommendations.

### Overview

| Component | Change | Rationale |
|-----------|--------|-----------|
| **Prior Network** | Full Covariance with AR(1) structure | Generates correlated z samples |
| **Quantile Decoder** | REMOVE | No longer needed - use sampling |
| **Context Length** | FIXED to 60 days | Consistent methodology |
| **Horizon** | H=30 (recommended) | Computational efficiency |

### Key Insight: Sample-Based Quantiles Replace Quantile Decoder

**OLD APPROACH (Quantile Decoder):**
```python
Prior → z (not diverse) → Quantile Decoder → [p05, p50, p95]
                                              (fixed quantiles only)
```

**NEW APPROACH (Full Covariance + Sampling):**
```python
Prior → z_1, z_2, ..., z_1000 (diverse!) → Decoder → surfaces
                                                        ↓
                                          Empirical quantiles:
                                          p05 = percentile(surfaces, 5%)
                                          pXX = percentile(surfaces, XX%)
                                          (ANY quantile!)
```

**Why sampling is better:**
| Aspect | Quantile Decoder | Full Cov + Sampling |
|--------|------------------|---------------------|
| Output | Fixed (p05, p50, p95) | Any quantile |
| Flexibility | Retrain for new quantiles | Just resample |
| Full distribution | ❌ Only 3 points | ✅ Complete |
| Decoder complexity | 3× channels | 1× (simpler) |

### Architecture: Full Covariance Prior

```
Context (60 days) → Context Encoder → context_embedding (100,)
                                              ↓
                    ┌─────────────────────────────────────────────┐
                    │      FULL COVARIANCE PRIOR NETWORK          │
                    │                                             │
                    │  ┌──────────────────────────────────────┐  │
                    │  │ Position Encoder (MLP)               │  │
                    │  │   for t in [1, 2, ..., H]:           │  │
                    │  │     μ_t = f(ctx, pos_enc(t))         │  │
                    │  │   output: μ = [μ_1, μ_2, ..., μ_H]   │  │
                    │  └──────────────────────────────────────┘  │
                    │                                             │
                    │  ┌──────────────────────────────────────┐  │
                    │  │ Global Covariance (2 params only!)   │  │
                    │  │   φ = sigmoid(learnable_φ)   ~0.5    │  │
                    │  │   σ² = exp(learnable_log_σ²) ~1.0    │  │
                    │  │   Σ[i,j] = σ² × φ^|i-j|  (Toeplitz)  │  │
                    │  └──────────────────────────────────────┘  │
                    │                                             │
                    │  ┌──────────────────────────────────────┐  │
                    │  │ Cholesky Sampling                    │  │
                    │  │   L = cholesky(Σ)                    │  │
                    │  │   ε ~ N(0, I)                        │  │
                    │  │   z = μ + L @ ε  ← CORRELATED!       │  │
                    │  └──────────────────────────────────────┘  │
                    └─────────────────────────────────────────────┘
                                              ↓
                              z (H, latent_dim) with autocorr = φ
                                              ↓
                                    Standard Decoder → surfaces
```

### How Problems Are Solved

**Problem 1 (Prior too context-specific, σ too small):**
- OLD: Both μ and σ are context-dependent (~187K params) → overfit
- NEW: μ is context-dependent, but σ² is GLOBAL (1 param) → consistent diversity
- Result: Similar contexts still get diverse samples

**Problem 2 (Prior parameter reuse):**
- OLD: Same (μ, σ) reused for all H timesteps → IID samples → wrong autocorr
- NEW: Position-encoded μ_t varies per timestep + Cholesky gives correlated samples
- Result: Samples have exact autocorr = φ (controllable!)

### Implementation Details

**1. Position-Encoded Mean Network:**
```python
class PositionEncodedPriorMean(nn.Module):
    def __init__(self, context_dim=100, horizon=30, latent_dim=12):
        self.pos_encoder = SinusoidalPositionEncoding(d_model=64)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, context_emb):  # (B, context_dim)
        # Output: μ = [μ_1, μ_2, ..., μ_H] where each μ_t is DIFFERENT
        ...
```

**2. AR(1) Covariance Matrix:**
```python
def build_ar1_covariance(phi, sigma_sq, horizon):
    """
    Σ[i,j] = σ² × φ^|i-j|

    Example (φ=0.5, σ²=1.0, H=5):
    [[1.00, 0.50, 0.25, 0.12, 0.06],
     [0.50, 1.00, 0.50, 0.25, 0.12],
     [0.25, 0.50, 1.00, 0.50, 0.25],
     [0.12, 0.25, 0.50, 1.00, 0.50],
     [0.06, 0.12, 0.25, 0.50, 1.00]]
    """
    indices = torch.abs(torch.arange(horizon).unsqueeze(1) -
                       torch.arange(horizon).unsqueeze(0))
    return sigma_sq * (phi ** indices)
```

**3. Cholesky Sampling:**
```python
def sample_with_covariance(mu, Sigma, num_samples=1000):
    """
    Sample z ~ N(μ, Σ) using Cholesky decomposition.

    z = μ + L @ ε, where L = chol(Σ), ε ~ N(0, I)

    This gives samples with EXACT autocorrelation = φ
    """
    L = torch.linalg.cholesky(Sigma)  # (H, H)
    epsilon = torch.randn(num_samples, H, latent_dim)
    z = mu.unsqueeze(0) + (L @ epsilon.transpose(-1, -2)).transpose(-1, -2)
    return z  # (num_samples, H, latent_dim)
```

### Training Process

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Fixed context length = 60 (NOT variable!)
        context = batch['surface'][:, :60, :, :]  # Always 60 days
        future = batch['surface'][:, 60:60+H, :, :]  # H = 30

        # 2. Encoder: Get posterior z (ground truth)
        z_posterior = encoder(context, future)

        # 3. Prior: Sample z from Full Covariance Prior
        context_emb = context_encoder(context)
        z_prior, mu_prior, Sigma = prior(context_emb)

        # 4. Decoder: Reconstruct surfaces
        surfaces_recon = decoder(z_posterior)

        # 5. Losses
        recon_loss = MSE(surfaces_recon, future)
        kl_loss = KL_divergence(z_posterior, mu_prior, Sigma)
        total_loss = recon_loss + beta * kl_loss

        # 6. Backprop - φ and σ² are learned automatically!
        total_loss.backward()
        optimizer.step()
```

### Inference Process

```python
# 1. Get context (60 days)
context = get_recent_60_days()  # (60, 5, 5)
context_emb = context_encoder(context)  # (100,)

# 2. Sample from Full Covariance Prior
z_samples, mu, Sigma = prior(context_emb, num_samples=1000)
# z_samples: (1000, 30, 12) - 1000 correlated trajectories

# 3. Decode each sample (STANDARD decoder, not quantile!)
surfaces = decoder(z_samples)  # (1000, 30, 5, 5)

# 4. Compute ANY quantile empirically
p05 = np.percentile(surfaces, 5, axis=0)   # (30, 5, 5)
p50 = np.percentile(surfaces, 50, axis=0)  # (30, 5, 5)
p95 = np.percentile(surfaces, 95, axis=0)  # (30, 5, 5)
# Can compute p10, p25, p75, p90, etc. - ANY quantile!
```

### Key Parameters

| Parameter | Value | Learned? | Purpose |
|-----------|-------|----------|---------|
| Context length | 60 days | Fixed | Consistent methodology |
| Horizon (H) | 30 days | Fixed | Computational efficiency |
| φ (autocorr) | ~0.5 | Yes | Controls temporal correlation |
| σ² (variance) | ~1.0 | Yes | Controls spread |
| Position dim | 64 | Fixed | Sinusoidal encoding |
| Prior MLP | 256→128→12 | Yes | Maps (ctx, pos) → μ_t |

### Why φ ≈ 0.5 (Not Oracle's 0.7)

Experiments showed oracle z has autocorr ≈ 0.7, but:

| Autocorr | Roughness vs Oracle | Assessment |
|----------|---------------------|------------|
| 0.0 (iid) | 138.5% | TOO ROUGH |
| 0.5 | 110.0% | ⭐ OPTIMAL |
| 0.7 (oracle) | 87.1% | Too smooth |

Oracle sees future (cheating). Prior only sees context → needs lower autocorr to match roughness.

### Computational Comparison

| Horizon | Cholesky Cost | Relative |
|---------|--------------|----------|
| H=30 | O(30³) = 27K ops | 1.0× |
| H=60 | O(60³) = 216K ops | 8.0× |
| H=90 | O(90³) = 729K ops | 27.0× |

**Recommendation:** Start with H=30, chain for longer sequences if needed.

### Files to Modify

| File | Change |
|------|--------|
| `vae/cvae_conditional_prior.py` | Add FullCovariancePrior class |
| `vae/cvae_with_mem_randomized.py` | Remove quantile decoder option |
| Training scripts | Fix context_len=60, horizon=30 |
| Generation scripts | Use sampling for quantiles |

### Files to Remove/Deprecate

| File | Reason |
|------|--------|
| Quantile decoder code | No longer needed |
| `train_quantile_models.py` | Replace with standard training |
| Quantile-specific generation | Use empirical percentiles |

### Validation Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Sample autocorr | = φ ± 0.05 | `np.corrcoef(z[:, :-1], z[:, 1:])` |
| Roughness ratio | > 40% | Compare std of daily changes |
| CI coverage | 90% ± 5% | Empirical coverage test |
| Inference speed | < 2s for 1000 samples | Benchmark |

### Summary

This solution:
1. ✅ **Solves Problem 1:** σ is GLOBAL (not context-specific)
2. ✅ **Solves Problem 2:** μ_t varies + samples are correlated
3. ✅ **Removes quantile decoder:** Use sampling instead
4. ✅ **Fixes context length:** Always 60 (not variable)
5. ✅ **Efficient:** Single forward pass + Cholesky
6. ✅ **Flexible:** Any quantile from samples
7. ✅ **Simpler:** Only 2 extra params (φ, σ²) vs 187K

### Reference Implementation

See `full_covariance_prior_walkthrough.py` for complete working code.

---

## Previously Documented Solutions (from CONDITIONAL_DISTRIBUTION_FRAMEWORK.md)

### ❌ OBSOLETE - Based on Wrong Diagnosis

| Solution | Why Obsolete |
|----------|--------------|
| **Inverse Lipschitz decoder** | Exp 8-10 prove decoder uses z effectively |
| **KL annealing** | Not posterior collapse (KL=12.1 healthy) |
| **Scale-VAE** | Not posterior collapse |
| **Global variance scaling** | Exp 1 shows prior is context-specific, need context-specific fix |

### ⚠️ PARTIALLY VALID - May Help But Don't Address Root Cause

| Solution | Status | Notes |
|----------|--------|-------|
| **Bootstrap from residuals** | Works | Bypasses z entirely, but doesn't improve model |
| **Conformal prediction** | Works | Post-hoc calibration, doesn't fix generation |
| **K-NN conditional variance** | Works | Borrows from neighbors, validated by Exp 7 |

### ✅ VALID - Address Root Cause

| Solution | Status | Notes |
|----------|--------|-------|
| **Regularize prior network** | Needs retraining | Force similar contexts → similar priors |
| **Reduce prior network capacity** | Needs retraining | Simpler prior = more grouping |
| **Mixture/GMM prior** | Needs retraining | Cluster contexts into groups |

---

## New Solutions from Research (2024-2025)

### 1. Diffusion Models for Financial Time Series

**CoFinDiff (IJCAI 2025)** - https://arxiv.org/abs/2503.04164
- Controllable financial diffusion model with cross-attention conditioning
- Can specify trends and volatility as conditions
- Generated data accurately meets specified conditions

**Synthetic Financial Time Series (2024)** - https://arxiv.org/abs/2410.18897
- DDPM with wavelet transformation
- Captures stylized facts (fat tails, volatility clustering)

**Relevance:** Could replace VAE entirely. Diffusion models naturally produce diverse samples.

### 2. Normalizing Flows for Conditional Distributions

**TarFlow (2024-2025)** - https://arxiv.org/html/2412.06329v3
- Transformer-based Masked Autoregressive Flows
- Quality comparable to diffusion, faster sampling

**CAFLOW** - https://www.aimsciences.org/article/doi/10.3934/fods.2024028
- Conditional autoregressive flows for diverse generation
- Models conditional distribution of latent encodings

**Relevance:** Flows explicitly model P(X|C), no prior-posterior mismatch.

### 3. Learned/Structured Priors

**Prior Learning in Introspective VAEs (Aug 2024)** - https://arxiv.org/abs/2408.13805
- Multimodal trainable prior
- Responsibility regularization to prevent inactive modes
- Adaptive variance clipping

**Multi-stage VAE (2024)** - https://www.sciencedirect.com/science/article/abs/pii/S0010482524008709
- Learn optimal prior as aggregated posterior
- Closes gap between variational posterior and prior

**Hierarchical Prior VAE (2024)** - https://www.sciencedirect.com/science/article/abs/pii/S0306457324000013
- Learns hierarchical prior from neighbors
- Prior captures community preference + personalized info

### 4. Quantile Regression Approaches

**QR-VAE** - https://pmc.ncbi.nlm.nih.gov/articles/PMC8321392/
- Quantile regression decoder instead of variance estimation
- Addresses variance shrinkage in VAEs

**Heteroscedastic Calibration** - https://arxiv.org/pdf/1910.14179
- Repurpose heteroscedastic regression for calibration
- Quantile-HC for aleatoric uncertainty

**Relevance:** Already implemented (quantile decoder), but needs proper calibration.

### 5. VAE for Time Series Forecasting

**VAEneu (2024)** - https://link.springer.com/article/10.1007/s10489-024-06203-5
- CVAE with CRPS loss for probabilistic forecasting
- Superior uncertainty quantification

**K²VAE (2024)** - https://arxiv.org/pdf/2505.23017
- Koopman-Kalman enhanced VAE
- Linear dynamics in latent space reduces error accumulation

**VAR-VAE (2025)** - https://www.sciencedirect.com/science/article/pii/S0020025525003160
- VAR model in latent space
- Better uncertainty modeling

---

## Solution Categories

### Category A: Fix Current Model (No Retraining)

**A1. K-NN Conditional Variance Injection**
```
Method: For context C, find K nearest in prior μ space
        Use neighbor outcome variance as σ²_conditional
        Sample: X ~ N(model_prediction, σ_conditional)

Pros: Validated by Exp 7 (regimes cluster, 2.77× separation)
Cons: Adds variance post-hoc, doesn't improve model predictions
Preserves marginal: Yes (if σ calibrated correctly)
```

**A2. Context-Dependent Bootstrap**
```
Method: Bootstrap residuals from K nearest contexts
        Sample: X = prediction + bootstrap_residual

Pros: Non-parametric, uses real errors
Cons: Residuals may not be i.i.d.
Preserves marginal: Approximately (depends on residual distribution)
```

**A3. Conformal Prediction Wrapper**
```
Method: Compute nonconformity scores on calibration set
        Construct prediction intervals with coverage guarantee

Pros: Guaranteed coverage (asymptotically)
Cons: Intervals, not full distribution
Preserves marginal: N/A (interval method)
```

### Category B: Modify Prior Network (Requires Retraining)

**B1. Prior Network Regularization**
```
Method: Add loss term to group similar contexts
        L_reg = λ * ||μ(C₁) - μ(C₂)||² when sim(C₁, C₂) > threshold

Pros: Directly addresses root cause
Cons: Requires retraining, hyperparameter tuning
Preserves marginal: Should, if done correctly
```

**B2. Reduce Prior Network Capacity**
```
Method: Use simpler prior network (fewer layers/units)
        Force it to learn coarser groupings

Pros: Simple, interpretable
Cons: May lose some useful context discrimination
Preserves marginal: Likely, but needs validation
```

**B3. Mixture Prior (GMM or Learned)**
```
Method: Prior outputs K Gaussian components
        Each component represents a context cluster
        Sample: z ~ Σπₖ * N(μₖ, σₖ)

Pros: Explicit multi-modal, diverse sampling
Cons: More complex training, mode collapse risk
Preserves marginal: Yes (if modes trained correctly)
```

**B4. Hierarchical Prior (from neighbors)**
```
Method: Prior conditioned on both context AND neighbors
        Borrows uncertainty from similar contexts

Pros: Naturally groups similar contexts
Cons: More complex, requires neighbor computation
Preserves marginal: Should, similar to K-NN
```

### Category C: Alternative Architectures (Major Change)

**C1. Diffusion Model Replacement**
```
Method: Replace VAE with conditional diffusion model
        Train: Learn denoising score ∇logP(X|C)
        Sample: Reverse diffusion from noise

Pros: Naturally diverse, no prior-posterior gap
Cons: Complete architecture change, slower sampling
Preserves marginal: By design
```

**C2. Normalizing Flow Replacement**
```
Method: Replace VAE with conditional normalizing flow
        Explicit density P(X|C) = |det(∂f/∂x)| * P(f(X)|C)

Pros: Exact likelihood, diverse sampling
Cons: Architecture change, may need flow-specific tricks
Preserves marginal: By design (exact density)
```

**C3. Quantile Regression Decoder (Already Have)**
```
Method: Decoder outputs p05, p50, p95 directly
        Already implemented but CIs too narrow

Fix needed: Train with prior-mode z, not just posterior
            Or: Apply heteroscedastic calibration post-hoc
```

### Category D: Hybrid Approaches

**D1. Two-Stage Generation**
```
Stage 1: VAE generates "mean trajectory" (current model)
Stage 2: Add stochastic component from learned residual distribution

Method: Train separate residual model on (context, prediction_error)
        At inference: X = VAE(context) + ResidualModel(context)

Pros: Keeps existing model, adds diversity layer
Cons: Two models to maintain
```

**D2. VAE + Flow Hybrid**
```
Method: Use VAE for coarse structure
        Use normalizing flow for fine-grained diversity

Pros: Best of both worlds
Cons: Complex architecture
```

---

## Recommendation Ranking

### Immediate Implementation (No Retraining)

| Rank | Solution | Effort | Expected Impact |
|------|----------|--------|-----------------|
| 1 | **K-NN Conditional Variance (A1)** | Low | High - validated |
| 2 | **Context-Dependent Bootstrap (A2)** | Low | Medium-High |
| 3 | **Conformal Prediction (A3)** | Low | Medium (intervals only) |

### Medium-Term (Retraining Required)

| Rank | Solution | Effort | Expected Impact |
|------|----------|--------|-----------------|
| 1 | **Prior Network Regularization (B1)** | Medium | High - direct fix |
| 2 | **Hierarchical Prior (B4)** | Medium-High | High |
| 3 | **Mixture Prior (B3)** | Medium | High |
| 4 | **Quantile Decoder with Prior Samples** | Medium | Medium-High |

### Long-Term (Architecture Change)

| Rank | Solution | Effort | Expected Impact |
|------|----------|--------|-----------------|
| 1 | **Diffusion Model (C1)** | High | Very High |
| 2 | **Normalizing Flow (C2)** | High | Very High |
| 3 | **Two-Stage VAE+Residual (D1)** | Medium-High | High |

---

## Key Insight for Marginal Preservation

**Law of Total Variance constraint:**
```
Var(X) = E[Var(X|C)] + Var(E[X|C])
  ↑           ↑              ↑
Total     "within"       "between"
(known)   (to inject)    (from model)
```

**To preserve marginal while adding conditional variance:**
1. Measure Var(E[X|C]) from current model predictions
2. Compute required: E[Var(X|C)] = Var(X) - Var(E[X|C])
3. Inject this much variance through K-NN/bootstrap
4. Verify: aggregated samples should have Var ≈ Var(X)

**Current measurements (H=90, ATM 6M):**
```
Var(X) = 0.00371
Var(E[X|C]) = 0.00660 (overspending!)
E[Var(X|C)] = 0.00001 (near zero)

Issue: Model predictions have MORE variance than ground truth!
This means model is OVERCONFIDENT in differentiating contexts.
```

---

## Time-Series Prior Solutions - Detailed Comparison

Based on verification experiments showing **9.7% baseline** vs **75% oracle** roughness ratio, the prior parameter reuse must be addressed with architectural changes.

### Current Architecture Analysis

**ConditionalPriorNetwork** (~187K parameters):
- LSTM-based: `context → Surface Embedding → LSTM → Compression → (μ, σ)`
- Already outputs **per-timestep parameters** `(B, C, latent_dim)` for context
- **Limitation:** Generation logic at lines 176-177 collapses to single set:
  ```python
  future_prior_mean = prior_mean[:, -1:, :].expand(B, horizon, -1)  # ← BOTTLENECK
  ```

---

### Solution P1: Autoregressive Prior (RNN-based)

**Architecture:** `z_t ~ N(μ(context, z_{<t}), σ(context, z_{<t}))`

**How it works:**
- Prior RNN takes context encoding + previous z values
- Outputs (μ_t, σ_t) sequentially for each future timestep
- Sample z_1, then z_2|z_1, then z_3|z_{1:2}, etc.

| ✅ Pros | ❌ Cons |
|---------|---------|
| Captures temporal dependencies in z explicitly | **Not parallelizable** - must sample sequentially |
| Theoretically most correct for time series | H forward passes (slower inference) |
| z values correlated like real market dynamics | Complex training (need teacher forcing for prior) |
| Expected roughness: **60-75%** | Risk of exposure bias (train vs inference) |

**Implementation:** Add RNN layer after context LSTM, train with autoregressive prior loss
**Retraining:** Required (extensive)
**Complexity:** Medium-High
**Risk:** Medium (exposure bias, slower inference)

---

### Solution P2: Position-Encoded Prior ⭐ RECOMMENDED

**Architecture:** `μ_t, σ_t = PriorNet(context_encoding, pos_encoding(t))`

**How it works:**
- Add sinusoidal position encoding for each future timestep t ∈ [1, H]
- Prior network takes concatenated [context, position] input
- Single forward pass outputs (μ_1..H, σ_1..H)

| ✅ Pros | ❌ Cons |
|---------|---------|
| **Fully parallelizable** - single forward pass | No explicit z_t → z_{t+1} dependency |
| Simple modification to existing architecture | Position encoding is learned heuristic |
| Well-understood (Transformer heritage) | May not capture complex temporal patterns |
| Moderate parameter increase (~10-20%) | |
| Easy to implement and debug | |
| Expected roughness: **40-60%** | |

**Implementation:**
1. Add position encoding layer: `pos_enc = SinusoidalPositionEncoding(latent_dim)`
2. Modify prior forward to accept horizon input
3. Concat position encoding to context encoding
4. Output (B, H, latent_dim) directly

**Retraining:** Required
**Complexity:** Low-Medium
**Risk:** Low

---

### Solution P3: Transformer Prior (Cross-Attention)

**Architecture:** `horizon_queries (learnable) × context_keys → (μ_1..H, σ_1..H)`

**How it works:**
- H learnable query vectors (one per future timestep)
- Cross-attend to context encoding via multi-head attention
- Each query outputs (μ_t, σ_t) for its timestep

| ✅ Pros | ❌ Cons |
|---------|---------|
| **Most expressive** - can learn complex patterns | **Highest parameter count** (~2-3× current) |
| Each timestep attends to relevant context | Risk of overfitting with limited data (~5800 days) |
| Captures long-range dependencies | More complex implementation |
| State-of-the-art for sequence modeling | Slower training, harder to debug |
| Parallelizable (single forward pass) | Extensive hyperparameter tuning needed |
| Expected roughness: **50-75%** | |

**Implementation:**
1. Replace prior MLP with TransformerDecoder
2. Create H learnable query embeddings
3. Cross-attention: queries attend to context LSTM outputs
4. Output heads: μ_head and σ_head for each query

**Retraining:** Required (extensive)
**Complexity:** High
**Risk:** High (overfitting, complexity)

---

### Solution P4: Variance Scaling (No Retraining)

**Architecture:** `σ_t = σ_base × scale(t)` where `scale(t) = sqrt(1 + t/H)`

**How it works:**
- Multiply σ by horizon-dependent scaling factor
- Sample z with monotonically increasing variance
- No changes to network, only generation logic

| ✅ Pros | ❌ Cons |
|---------|---------|
| **No retraining required** | **Mean still constant** - paths remain correlated |
| Trivial to implement (~3 lines of code) | Heuristic, not learned from data |
| Can test immediately | **Only reaches 17.1%** (verified) |
| Zero risk to existing model | Doesn't address root cause |
| Good for quick validation | Marginal error increases to 129% |

**Implementation:** Add scaling to logvar in `get_surface_given_conditions`:
```python
t = torch.arange(horizon).float()
scale = 0.5 * torch.log(1 + t/horizon)  # sqrt scaling
future_prior_logvar = prior_logvar[:, -1:, :] + scale.view(1, -1, 1)
```

**Retraining:** No
**Complexity:** Trivial
**Risk:** None
**Expected roughness:** 10-17% (verified)

---

### Solution D1: Hybrid VAE + Residual Model

**Architecture:** `final_surface = VAE(context) + ResidualModel(context)`

**How it works:**
- Keep existing VAE frozen
- Train separate model to predict residuals to add to VAE output
- Residual model learns to add stochastic variation

| ✅ Pros | ❌ Cons |
|---------|---------|
| **Preserves existing VAE** - no retraining | Two models to maintain |
| Modular - can swap residual model | May not integrate well with latent space |
| Explicit separation: VAE for mean, Residual for variance | Residual model needs own training pipeline |
| Lower risk - existing model unchanged | Total parameter count increases |
| Can use simpler architecture for residual | Harder to ensure marginal preservation |
| Expected roughness: **30-50%** | |

**Implementation:**
1. Train ResidualNet on (context, VAE_prediction_error) pairs
2. At inference: `pred = VAE(ctx) + ResidualNet(ctx).sample()`
3. ResidualNet can be simple: LSTM + Gaussian output

**Retraining:** New model only
**Complexity:** Medium
**Risk:** Medium (integration complexity)

---

## Solution Comparison Summary

| Solution | Complexity | Retraining | Parallelizable | Expected Roughness | Risk | Recommendation |
|----------|------------|------------|----------------|-------------------|------|----------------|
| **P1: Autoregressive** | Medium-High | Yes | ❌ No | 60-75% | Medium | Backup if P2 insufficient |
| **P2: Position-Encoded** | Low-Medium | Yes | ✅ Yes | 40-60% | Low | ⭐ **PRIMARY** |
| **P3: Transformer** | High | Yes (extensive) | ✅ Yes | 50-75% | High | If overfitting not concern |
| **P4: Variance Scaling** | Trivial | No | ✅ Yes | 10-17% ⚠️ | None | Quick validation only |
| **D1: Hybrid Residual** | Medium | New model | ✅ Yes | 30-50% | Medium | If preserving VAE critical |

---

## Recommendation: Position-Encoded Prior (P2)

**Rationale:**
1. **Best balance** of complexity vs expected improvement
2. **Fully parallelizable** - no inference slowdown
3. **Minimal change** to existing architecture (add position encoding)
4. **Well-understood** technique with predictable behavior
5. **Low risk** of breaking existing functionality
6. **Sufficient improvement** - 40-60% target vs 75% oracle upper bound

**Next steps:**
1. Implement sinusoidal position encoding layer
2. Modify ConditionalPriorNetwork to accept horizon + position input
3. Retrain with same hyperparameters
4. Validate: Target roughness >40%, autocorr >0.3

**If P2 insufficient (< 40% roughness):**
- Escalate to **P1 (Autoregressive)** for stronger temporal dependencies
- Accept slower inference (sequential sampling) for higher fidelity

---

## Critical Question for User

Before implementing, need to clarify:

**Which approach do you prefer?**

1. **Quick win (A1/A2):** K-NN or bootstrap for immediate results
   - No retraining
   - Adds post-hoc diversity
   - Preserves existing model

2. **Proper fix (B1):** Regularize prior network
   - Requires retraining
   - Fixes root cause
   - Better long-term solution

3. **Architecture change (C1/C2):** Diffusion or flow model
   - Major rework
   - State-of-the-art approach
   - Highest potential

4. **Hybrid (D1):** Keep VAE + add residual model
   - Moderate effort
   - Keeps existing investment
   - Adds explicit stochastic layer

---

## Files to Modify

**For A1 (K-NN Conditional Variance):**
- New: `experiments/backfill/context60/knn_conditional_variance.py`
- Modify: Generation scripts to add variance injection

**For B1 (Prior Network Regularization):**
- Modify: `vae/cvae_conditional_prior.py` (add regularization loss)
- Modify: Training scripts

**For C1 (Diffusion Model):**
- New: `vae/diffusion_model.py` (new architecture)
- New: Training and generation scripts

---

## References

### VAE Prior & Regularization
- [Prior Learning in Introspective VAEs](https://arxiv.org/abs/2408.13805) - Aug 2024
- [Multi-stage VAE for Learned Prior](https://www.sciencedirect.com/science/article/abs/pii/S0010482524008709) - 2024
- [Hierarchical Prior VAE](https://www.sciencedirect.com/science/article/abs/pii/S0306457324000013) - 2024
- [VAE Priors Overview](https://jmtomczak.github.io/blog/7/7_priors.html)

### Diffusion Models for Finance
- [CoFinDiff](https://arxiv.org/abs/2503.04164) - IJCAI 2025
- [Synthetic Financial Time Series](https://arxiv.org/abs/2410.18897) - 2024

### Normalizing Flows
- [TarFlow](https://arxiv.org/html/2412.06329v3) - 2024
- [CAFLOW](https://www.aimsciences.org/article/doi/10.3934/fods.2024028) - 2024
- [Conditional Normalizing Flows](https://scipost.org/SciPostPhys.16.5.132/pdf) - 2024

### Time Series VAE
- [VAEneu](https://link.springer.com/article/10.1007/s10489-024-06203-5) - 2024
- [K²VAE](https://arxiv.org/pdf/2505.23017) - 2024
- [VAR-VAE](https://www.sciencedirect.com/science/article/pii/S0020025525003160) - 2025

### Uncertainty Quantification
- [QR-VAE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321392/) - 2021
- [Heteroscedastic Calibration](https://arxiv.org/pdf/1910.14179) - 2019
