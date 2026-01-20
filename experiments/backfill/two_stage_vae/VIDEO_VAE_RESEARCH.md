# Causal 3D VAE Research for Temporal Surface Generation

This document tracks research into using video VAE architectures for volatility surface forecasting.

---

# IMPLEMENTATION STATUS

## Files Created

| File | Description |
|------|-------------|
| `vae/causal_3d_blocks.py` | CausalConv3d, ResnetBlockCausal3D, Downsample/Upsample modules |
| `vae/causal_3d_vae.py` | EncoderCausal3D, DecoderCausal3D, AutoencoderCausal3D |
| `config/causal_3d_config.py` | Training configuration dataclass |
| `experiments/backfill/two_stage_vae/train_causal_3d_vae.py` | Training script |
| `experiments/backfill/two_stage_vae/eval_causal_3d_coverage.py` | Basic CI coverage evaluation |
| `experiments/backfill/two_stage_vae/eval_causal_3d_comprehensive.py` | Full evaluation suite |
| `experiments/backfill/two_stage_vae/diagnose_causal_3d_posterior.py` | Posterior collapse diagnostic |

## Model Architecture (Scaled Down for 5×5 Grid)

```
Parameters: 473,249 (vs 9.8M original)

ENCODER:
  CausalConv3d(1 → 8)
  ResBlock(8 → 16) + Downsample(T → T/2)
  ResBlock(16 → 32)
  MidBlock(32)
  Conv → z(4, T/2, 5, 5)

DECODER (mirrors encoder):
  CausalConv3d(4 → 32)
  MidBlock(32)
  ResBlock(32 → 16) + Upsample(T/2 → T)
  ResBlock(16 → 8)
  CausalConv3d(8 → 1)
```

## Training Configuration

- Context length: 60 days
- Prediction horizon: 60 days
- Total sequence: 120 days
- KL weight: 1e-6
- Learning rate: 1e-4
- Batch size: 32

---

# TRAINING RESULTS: 50 Epochs Oracle Mode (No Autoregressive)

## Training Progress

| Epoch | Train Loss | Val Loss | KL | Notes |
|-------|-----------|----------|-----|-------|
| 1 | 0.0203 | 0.0068 | 2.09 | Initial |
| 10 | 0.0039 | 0.0042 | 1.20 | Converging |
| 25 | 0.0034 | 0.0035 | 1.26 | Stable |
| 50 | 0.0031 | 0.0032 | 1.25 | Final |

Model saved to: `models/backfill/causal_3d/best_model.pt`

## Comprehensive Evaluation Results (Oracle Mode)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **90% CI Coverage** | **33.4%** | 90% | ❌ Poor |
| Explosion Rate | 0.3% | 0% | ✅ Good |
| CRPS | 0.0089 | Lower is better | - |
| Sample Kurtosis | 1.34 | 0.59 (GT) | ❌ 2.28× higher |
| Sample Skewness | 0.12 | 0.08 (GT) | ~OK |
| Smile Curvature | 54% of GT | 100% | ⚠️ Reduced |
| Term Structure | 89% sign match | 100% | ⚠️ Mostly OK |

## Per-Horizon Coverage Degradation

| Horizon | Coverage |
|---------|----------|
| h=1 | 51% |
| h=7 | 42% |
| h=14 | 35% |
| h=30 | 27% |
| h=60 | 22% |

Coverage degrades rapidly with horizon, indicating the model cannot produce wide enough confidence intervals.

---

# CRITICAL ISSUE: Out of Memory During Autoregressive Training

## Problem

At epoch 50, when autoregressive (AR) training was enabled, the script crashed with OOM:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.41 GiB
```

## Root Cause

AR training requires gradient accumulation over 60 sequential steps:
- Each step: forward + backward through full model
- Memory grows linearly with sequence length
- 60 steps × model activations = OOM

## Attempted Fixes

1. Added `ar_training_steps: int = 10` config parameter to limit AR steps
2. Added `--ar-steps` and `--no-ar-training` CLI flags
3. Reduced batch size during AR phase

## Status

AR training not completed. Results above are from **oracle mode only** (encoder sees full context+target).

---

# ROOT CAUSE ANALYSIS: Why CI Coverage is 33% Instead of 90%

## Initial Hypothesis: Posterior Collapse

Suspected the small KL weight (1e-6) caused posterior collapse where encoder outputs near-zero variance.

## Diagnostic Results (diagnose_causal_3d_posterior.py)

| Test | Result | Diagnosis |
|------|--------|-----------|
| Logvar mean | **-0.10** | ✅ Healthy (not -30) |
| Posterior std | **0.97** | ✅ Healthy (not collapsed) |
| z sample std (100 samples) | **0.96** | ✅ Diverse latents |
| **Output std (100 samples)** | **0.008** | ❌ **PROBLEM!** |

**Finding: Posterior is NOT collapsed. The encoder produces diverse z samples.**

## Real Root Cause: Decoder Squashes Variance

100 different z samples (std=0.96) → nearly identical outputs (std=0.008)

The decoder maps all z variations to essentially the same output.

### Why Decoder Ignores Z Variance

1. **GroupNorm layers (5-7 in decoder)**: Normalize activations to std≈1 at each layer, erasing z-dependent variance
2. **MSE loss**: Rewards matching the mean perfectly, no incentive to preserve variance
3. **Scaling factor (0.18215)**: From Stable Diffusion, compresses z before decoding

### Decoder Sensitivity Test

| z Perturbation σ | Output MAE | Relative Change |
|------------------|------------|-----------------|
| 0.01 | 0.000012 | 0.0001 |
| 0.10 | 0.000115 | 0.0011 |
| 0.50 | 0.000583 | 0.0056 |
| 1.00 | 0.001167 | 0.0112 |
| 2.00 | 0.002334 | 0.0224 |

Even with σ=2.0 perturbation, output changes by only 2.2%. **Decoder is nearly insensitive to z.**

---

# CRITICAL FINDING: Video VAEs Are NOT Designed for Probabilistic Forecasting

## The Architecture Mismatch

Video VAEs (HunyuanVideo, etc.) are designed as **deterministic codecs**:

```
VIDEO GENERATION PIPELINE (HunyuanVideo):

Video → [VAE Encoder] → Latent z (compressed)
                              ↓
         [DIFFUSION MODEL] ← This is where diversity comes from!
         (60 transformer layers, 50 denoising steps)
                              ↓
Latent z' → [VAE Decoder] → Generated Video
              (DETERMINISTIC)
```

**Key insight**: The VAE decoder is SUPPOSED to be deterministic. Diversity comes from the diffusion model, not from VAE sampling.

## What We Were Trying vs What Video VAEs Do

| Our Approach | Video VAE Pipeline |
|--------------|-------------------|
| Sample z ~ q(z\|x) from encoder | Sample z ~ diffusion model |
| Expect decoder to produce diverse outputs | Decoder is deterministic |
| 1 forward pass | 50+ denoising steps |
| VAE sampling for diversity | Diffusion sampling for diversity |

**We were using the VAE architecture for something it was never designed to do.**

## GroupNorm's Purpose

GroupNorm in video VAE decoders is intentional - it makes the decoder a **stable, consistent mapping** from z to output. The diffusion model is supposed to feed in different z values; the decoder just faithfully decodes them.

---

# HUNYUAN DIFFUSION MODEL ARCHITECTURE

## HunyuanVideo (Text-to-Video)

The diffusion model we're missing:

| Component | Value |
|-----------|-------|
| Total transformer blocks | **60 layers** |
| Dual-stream blocks | 20 layers (separate video/text) |
| Single-stream blocks | 40 layers (merged multimodal) |
| Attention heads | 24 heads × 128 dim = 3072 hidden |
| Denoising steps | 50 iterations |
| Parameters | **Billions** |

This is the component that generates diversity - not the VAE.

## HunyuanCustom (Video-Conditioned)

For our temporal forecasting problem, we need **video conditioning**, not text conditioning.

Research into [HunyuanCustom](https://github.com/Tencent-Hunyuan/HunyuanCustom) (arXiv:2505.04512) shows:

### Video Conditioning Mechanism

```
CONDITIONING VIDEO (past/context):
  1. VAE Encoder → z_cond
  2. 4-layer FC alignment network → aligned_features
  3. Frame-by-frame ADDITION to noisy latents

DIFFUSION PROCESS:
  z_t (noisy) + aligned_features → Transformer → ε (predicted noise)
  Iterate 50 steps → z_0 (clean, diverse)

VAE DECODER:
  z_0 → Output (deterministic)
```

### Key Design Choices

- **Injection**: Addition (not concatenation) - preserves dimensions
- **Alignment**: 4-layer FC maps clean→noisy latent space
- **Temporal**: Frame-by-frame along temporal dimension

---

# OPTIONS GOING FORWARD

## Option 1: Add a Latent Diffusion Model

Build a small diffusion transformer that:
- Takes context latents as conditioning
- Generates diverse future latents via denoising
- Feed to deterministic VAE decoder

**Pros**: Architecturally correct, proven approach
**Cons**: Significantly more complex, longer training

## Option 2: Redesign for Probabilistic Decoding

Modify decoder to output distributions:
- Remove GroupNorm layers
- Add heteroscedastic output head (predict mean + variance)
- Use NLL loss instead of MSE

**Pros**: Simpler than diffusion, keeps current pipeline
**Cons**: Fighting against the architecture's design

## Option 3: Use Ensemble/Dropout Uncertainty

Keep deterministic decoder, use:
- Dropout at inference time (MC Dropout)
- Train ensemble of models
- Combine predictions for uncertainty

**Pros**: Simple to implement
**Cons**: May not capture full uncertainty

## Option 4: Different Architecture Entirely

Use architectures designed for probabilistic time series:
- TimeGrad (diffusion for time series)
- CSDI (conditional score-based diffusion)
- D3VAE (decomposed diffusion VAE)

**Pros**: Purpose-built for our problem
**Cons**: Need to start over

---

# RESEARCH SYNTHESIS: Multi-Step IV Surface Diffusion

## The Innovation Gap

### What Exists in Literature

| Domain | Paper | What They Do | Limitation |
|--------|-------|--------------|------------|
| **Finance** | [IV Surface DDPM (arxiv:2511.07571)](https://arxiv.org/abs/2511.07571) | One-day-ahead IV surface forecasting with 90% CI coverage | **One-step only** |
| **Video** | [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), [PA-VDM](https://arxiv.org/html/2410.08151v2) | Coherent multi-frame video generation | **No CI validation** |
| **Time Series** | [ARMD](https://arxiv.org/html/2412.09328v1), [REDI](https://dl.acm.org/doi/10.1145/3627673.3679808) | Multi-horizon forecasting | **Not applied to IV surfaces** |

### What's Missing (Our Innovation)

**Nobody has combined:**
1. Multi-step generation (60 days) with temporal coherence
2. CI coverage validation at ALL horizons (h=1, 7, 14, 30, 60)
3. IV surface domain constraints (arbitrage-free)

```
EXISTING:
  One-step IV DDPM:    Day k → Day k+1 (90% CI ✓)
  Video Diffusion:     Frame 1-60 coherent (CI not measured)

OUR GOAL:
  Multi-step IV DDPM:  Days 1-60 → Days 61-120 (90% CI at ALL horizons)
```

---

## Key Techniques from Video Diffusion

### 1. Progressive Noise Levels (from PA-VDM)

**Problem**: Uniform noise across all frames causes abrupt transitions and error accumulation.

**Solution**: Assign progressively increasing noise levels per frame:
```
τ = {0, T/S, 2T/S, ..., T}

Day 61: noise level τ₁ (low)    → more certain
Day 62: noise level τ₂          →
...
Day 120: noise level τ₆₀ (high) → less certain
```

**Benefit**: Earlier frames guide later frames. Later frames with higher uncertainty follow patterns from earlier, more certain frames.

**Source**: [Progressive Autoregressive Video Diffusion Models (CVPR 2025)](https://arxiv.org/html/2410.08151v2)

### 2. Chunked Frame Denoising (from PA-VDM)

**Problem**: Naive progressive noise causes cumulative error in latent video diffusion.

**Solution**: Treat chunks of C frames as a unit:
- Assign identical noise levels to frames within a chunk
- Add/remove chunks together from attention window
- Prevents divergence in long sequences

**Source**: PA-VDM

### 3. Overlapped Conditioning (from PA-VDM)

**Mechanism**: Prepend clean (context) frames to attention window:
```
Attention window: [Clean context frames | Noisy future frames]
                   ↑                      ↑
                   Already denoised       Being denoised
```

Later frames attend to clean frames, ensuring temporal consistency.

### 4. Rolling KV Cache (from Rolling Forcing)

**For long sequences**: Maintain two types of cached context:
- **Recent frames**: Short-term consistency
- **Initial frames**: Long-term consistency (prevents drift)

**Source**: [Rolling Forcing: Autoregressive Long Video Diffusion](https://arxiv.org/html/2509.25161v1)

### 5. Distribution Matching Distillation (from CausVid)

**For faster inference**: Distill 50-step diffusion → 4-step generator:
- Train bidirectional model first (sees all frames)
- Distill to autoregressive model (causal)
- Reduces inference time while maintaining quality

**Source**: [CausVid (CVPR 2025)](https://github.com/tianweiy/CausVid)

---

## Key Techniques from Time Series Diffusion

### 1. Sliding Diffusion (from ARMD)

**Key insight**: Frame the forecasting as diffusion trajectory:
```
Future series = initial state (x₀)
History series = final state (xₜ)
Intermediate states = sliding between them
```

**Benefit**: Generate ALL future timesteps at once through reverse diffusion, avoiding autoregressive error accumulation.

**Source**: [Auto-Regressive Moving Diffusion Models](https://arxiv.org/html/2412.09328v1)

### 2. Multi-Resolution Decomposition (from mr-Diff)

**Approach**: Seasonal-trend decomposition, coarse→fine:
1. Extract multi-scale trends
2. Forward diffusion: fine→coarse
3. Reverse diffusion: coarse→fine (easy-to-hard)

**Benefit**: Captures both global patterns and local dynamics.

**Source**: [Multi-Resolution Diffusion Models (ICLR 2024)](https://iclr.cc/media/iclr-2024/Slides/17883_mrXtGgm.pdf)

### 3. Recurrent Forward Process (from REDI)

**Mechanism**: Weight recent history more heavily in diffusion process:
- Recent past has stronger influence on near future
- Distant past has weaker influence

**Source**: [REDI (CIKM 2024)](https://dl.acm.org/doi/10.1145/3627673.3679808)

---

## Key Techniques from IV Surface DDPM

### 1. FiLM Conditioning (from arxiv:2511.07571)

**For scalar features** (VIX, returns, etc.):
```python
# Scalar features → 2-layer FC → scale (γ) and shift (β)
# Conv output = γ * conv_output + β

class FiLMLayer(nn.Module):
    def __init__(self, scalar_dim, channel_dim):
        self.fc = nn.Sequential(
            nn.Linear(scalar_dim, channel_dim),
            nn.SiLU(),
            nn.Linear(channel_dim, channel_dim * 2)  # γ and β
        )

    def forward(self, x, scalars):
        gamma, beta = self.fc(scalars).chunk(2, dim=-1)
        return gamma * x + beta
```

### 2. SNR-Weighted Arbitrage Penalty

**Problem**: Arbitrage constraints unreliable at high noise levels.

**Solution**: Weight penalty by signal-to-noise ratio:
```
L_arb_weighted = w_SNR(t) · Φ(surface)

where w_SNR(t) = ᾱₜ / (1 - ᾱₜ + ε)
```

- High noise (early steps): Low weight (unreliable estimates)
- Low noise (late steps): High weight (reliable estimates)

### 3. Multi-Timescale EWMA Context

**Input channels**:
- Current surface (1×5×5)
- 5-day EWMA surface (1×5×5) - short-term trend
- 20-day EWMA surface (1×5×5) - long-term trend

**Benefit**: Captures both recent dynamics and longer-term patterns.

---

## Proposed Architecture: Multi-Horizon IV Surface DDPM

### Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│            MULTI-HORIZON IV SURFACE DIFFUSION MODEL                          │
│            Combining: Video (temporal) + Time Series (multi-step) + Finance  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ CONTEXT ENCODER (from video diffusion)                                  │ │
│  │                                                                         │ │
│  │ Input: Past 60 days (60×5×5)                                           │ │
│  │   + 5-day EWMA surface (5×5)                                           │ │
│  │   + 20-day EWMA surface (5×5)                                          │ │
│  │                                                                         │ │
│  │ Architecture: 3D Conv Encoder → Context Features (C×T×5×5)             │ │
│  │                                                                         │ │
│  │ Scalar Features (via FiLM):                                            │ │
│  │   - Timestep t                                                          │ │
│  │   - VIX level                                                           │ │
│  │   - Return EWMA (5d, 20d)                                              │ │
│  │   - Squared return EWMA (vol proxy)                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ PROGRESSIVE NOISE ASSIGNMENT (from PA-VDM)                              │ │
│  │                                                                         │ │
│  │ Future 60 days with progressive noise levels:                          │ │
│  │                                                                         │ │
│  │   Day 61:  τ₁  = T/60  (low noise, high certainty)                     │ │
│  │   Day 62:  τ₂  = 2T/60                                                 │ │
│  │   Day 70:  τ₁₀ = 10T/60                                                │ │
│  │   Day 90:  τ₃₀ = 30T/60                                                │ │
│  │   Day 120: τ₆₀ = T      (high noise, low certainty)                    │ │
│  │                                                                         │ │
│  │ Benefit: Earlier days guide later days naturally                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 3D U-NET WITH CAUSAL TEMPORAL ATTENTION                                 │ │
│  │                                                                         │ │
│  │ Input: Concat(Context_features, Noisy_future) along channel dim        │ │
│  │                                                                         │ │
│  │ Architecture:                                                           │ │
│  │   Encoder: 4 → 32 → 64 → 128 channels                                  │ │
│  │   Bottleneck: 128 channels + Temporal Self-Attention (causal)          │ │
│  │   Decoder: 128 → 64 → 32 → 1 channel                                   │ │
│  │   Skip connections at each level                                        │ │
│  │   FiLM conditioning injected at each block                             │ │
│  │                                                                         │ │
│  │ Causal Attention: Frame t only attends to frames ≤ t                   │ │
│  │                                                                         │ │
│  │ Output: Predicted noise ε (60×5×5) for all future days                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                       ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ LOSS FUNCTION                                                           │ │
│  │                                                                         │ │
│  │ L = L_MSE + λ_arb · L_arbitrage + λ_temp · L_temporal                  │ │
│  │                                                                         │ │
│  │ L_MSE = ||ε - ε_pred||²                                                │ │
│  │                                                                         │ │
│  │ L_arbitrage = w_SNR(t) · [                                             │ │
│  │     calendar_spread_violation +                                         │ │
│  │     call_spread_violation +                                             │ │
│  │     butterfly_spread_violation                                          │ │
│  │ ]                                                                       │ │
│  │                                                                         │ │
│  │ L_temporal = ||surface_t - surface_{t-1}||² (smoothness)               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Source | Rationale |
|----------|--------|-----------|
| Progressive noise levels | PA-VDM | Prevents error accumulation, earlier frames guide later |
| All horizons at once | ARMD | Avoids autoregressive compounding errors |
| Causal temporal attention | Video VAE | Frame t only sees ≤t, enables streaming |
| FiLM for scalars | IV DDPM | Proven for market indicators |
| SNR-weighted arbitrage | IV DDPM | Constraints reliable only at low noise |
| Multi-timescale EWMA | IV DDPM | Captures short and long-term patterns |

### Model Size Estimate

```
For 5×5 grid with 60-day horizon:

Input:  (B, 4, 60, 5, 5)  = 6,000 values per sample
Output: (B, 1, 60, 5, 5)  = 1,500 values per sample

Estimated parameters: 500K - 2M
(Much smaller than HunyuanVideo's billions - appropriate for our data scale)
```

### Training Procedure

```python
def train_step(model, context, future_gt, scalars):
    """
    Progressive noise training for multi-horizon diffusion.
    """
    B, T_future, H, W = future_gt.shape  # T_future = 60

    # 1. Assign progressive noise levels to each future day
    # Day 1: low noise, Day 60: high noise
    noise_levels = torch.linspace(0.1, 1.0, T_future)  # Progressive τ

    # 2. Sample noise
    epsilon = torch.randn_like(future_gt)

    # 3. Create noisy future with per-day noise levels
    noisy_future = []
    for t in range(T_future):
        alpha_t = get_alpha(noise_levels[t])
        noisy_day = sqrt(alpha_t) * future_gt[:, t] + sqrt(1 - alpha_t) * epsilon[:, t]
        noisy_future.append(noisy_day)
    noisy_future = torch.stack(noisy_future, dim=1)

    # 4. Predict noise for all days
    epsilon_pred = model(context, noisy_future, noise_levels, scalars)

    # 5. Compute losses
    loss_mse = F.mse_loss(epsilon_pred, epsilon)
    loss_arb = compute_arbitrage_penalty(denoise(noisy_future, epsilon_pred), noise_levels)
    loss_temp = compute_temporal_smoothness(denoise(noisy_future, epsilon_pred))

    loss = loss_mse + lambda_arb * loss_arb + lambda_temp * loss_temp
    return loss
```

### Sampling Procedure

```python
def sample(model, context, scalars, num_samples=100):
    """
    Generate diverse future scenarios with proper uncertainty.
    """
    B = context.shape[0]
    T_future, H, W = 60, 5, 5

    all_samples = []

    for _ in range(num_samples):
        # Start from pure noise
        x = torch.randn(B, T_future, H, W)

        # Progressive noise levels (same as training)
        noise_levels = torch.linspace(0.1, 1.0, T_future)

        # Denoise iteratively
        for step in reversed(range(num_denoise_steps)):
            t = step / num_denoise_steps
            epsilon_pred = model(context, x, noise_levels * t, scalars)
            x = denoise_step(x, epsilon_pred, t)

        all_samples.append(x)

    return torch.stack(all_samples, dim=1)  # (B, num_samples, T_future, H, W)
```

---

## Evaluation Framework (Our Contribution)

### CI Coverage at All Horizons

```python
def evaluate_coverage(samples, ground_truth, ci_level=0.90):
    """
    Evaluate CI coverage at each horizon.

    samples: (num_test, num_samples, T_future, H, W)
    ground_truth: (num_test, T_future, H, W)
    """
    alpha = (1 - ci_level) / 2

    results = {}
    for h in [1, 7, 14, 30, 60]:
        # Get samples at horizon h
        samples_h = samples[:, :, h-1, :, :]  # (num_test, num_samples, H, W)
        gt_h = ground_truth[:, h-1, :, :]      # (num_test, H, W)

        # Compute quantiles
        lower = np.percentile(samples_h, alpha * 100, axis=1)
        upper = np.percentile(samples_h, (1 - alpha) * 100, axis=1)

        # Coverage
        covered = (gt_h >= lower) & (gt_h <= upper)
        coverage_rate = covered.mean()

        results[f'h={h}'] = coverage_rate
        print(f"Horizon {h}: Coverage = {coverage_rate:.1%} (target: {ci_level:.0%})")

    return results
```

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| 90% CI Coverage @ h=1 | 90% | Near-term |
| 90% CI Coverage @ h=7 | 90% | One week |
| 90% CI Coverage @ h=14 | 90% | Two weeks |
| 90% CI Coverage @ h=30 | 90% | One month |
| 90% CI Coverage @ h=60 | 90% | Two months |
| CRPS (all horizons) | Lower is better | Proper scoring rule |
| Arbitrage violations | < 1% | No-arb constraints |
| Explosion rate | 0% | Stability |

### Calibration Plot

```python
def calibration_plot(samples, ground_truth):
    """
    Plot nominal vs empirical coverage across confidence levels.
    Perfect calibration = diagonal line.
    """
    nominal_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    empirical_coverage = []

    for level in nominal_levels:
        alpha = (1 - level) / 2
        lower = np.percentile(samples, alpha * 100, axis=1)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=1)
        covered = (ground_truth >= lower) & (ground_truth <= upper)
        empirical_coverage.append(covered.mean())

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(nominal_levels, empirical_coverage, 'o-', label='Model')
    plt.xlabel('Nominal Coverage')
    plt.ylabel('Empirical Coverage')
    plt.title('Calibration Plot')
    plt.legend()
```

---

## Implementation Roadmap

### Phase 1: Core Diffusion Model

1. Implement 3D U-Net with causal temporal attention
2. Implement FiLM conditioning for scalars
3. Implement progressive noise scheduler
4. Basic MSE training (no arbitrage penalty yet)

### Phase 2: Finance-Specific Additions

1. Add SNR-weighted arbitrage penalty
2. Add EWMA context channels
3. Add temporal smoothness loss

### Phase 3: Evaluation

1. Implement CI coverage evaluation at all horizons
2. Implement CRPS computation
3. Implement calibration plots
4. Compare against baselines (VAE, one-step DDPM)

### Phase 4: Optimization

1. Tune progressive noise schedule
2. Tune loss weights (λ_arb, λ_temp)
3. Experiment with model size
4. Distillation for faster inference (optional)

---

# REFERENCES

## Video Diffusion
- [HunyuanVideo GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo) - Reference VAE implementation
- [HunyuanCustom Paper](https://arxiv.org/abs/2505.04512) - Video conditioning mechanism
- [HunyuanCustom GitHub](https://github.com/Tencent-Hunyuan/HunyuanCustom) - Video-conditioned diffusion
- [PA-VDM (CVPR 2025)](https://arxiv.org/html/2410.08151v2) - Progressive Autoregressive Video Diffusion
- [Rolling Forcing](https://arxiv.org/html/2509.25161v1) - Long video generation with KV cache
- [CausVid (CVPR 2025)](https://github.com/tianweiy/CausVid) - Bidirectional to causal distillation

## Time Series Diffusion
- [ARMD](https://arxiv.org/html/2412.09328v1) - Auto-Regressive Moving Diffusion
- [REDI (CIKM 2024)](https://dl.acm.org/doi/10.1145/3627673.3679808) - Recurrent Diffusion for Time Series
- [mr-Diff (ICLR 2024)](https://iclr.cc/media/iclr-2024/Slides/17883_mrXtGgm.pdf) - Multi-Resolution Diffusion
- [Diffusion-TS (ICLR 2024)](https://github.com/Y-debug-sys/Diffusion-TS) - Interpretable Time Series Diffusion
- [Awesome Time Series Diffusion](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model) - Curated paper list

## IV Surface / Finance
- [IV Surface DDPM (arxiv:2511.07571)](https://arxiv.org/abs/2511.07571) - One-step IV forecasting with DDPM
- [IV-VAE Paper](https://arxiv.org/abs/2411.06449) - Group causal convolution for IV
- [Meta-Learning Neural Process (SABR prior)](https://arxiv.org/html/2509.11928v1) - Pretrained prior for IV surfaces
- [Deep Learning from IV Surfaces](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4531181) - IV surface as image features

---

# CRITICAL FINDING #1: Ground Truth Coverage is NOT Validated in Video VAE Literature

## The User's Question

> "Is there evidence that video VAEs can generate, by varying latent, a conditional marginal distribution wide enough to include the ground truth?"

## Answer: **NO clear evidence exists.**

### What Video VAE Papers Actually Measure

| Evaluation Method | What It Measures | Does It Validate Coverage? |
|-------------------|------------------|---------------------------|
| **Best of N samples** | Whether at least 1 of 100 samples is close to GT | ❌ No - just proximity, not coverage |
| **CRPS** | Distributional agreement with GT | ⚠️ Partial - but rarely used for video |
| **Diversity metrics** | Sample variance | ❌ No - diverse ≠ covers GT |
| **SSIM/PSNR** | Reconstruction quality of best sample | ❌ No - single sample metric |

### Key Quotes from Literature

**[SAVP Paper](https://arxiv.org/abs/1804.01523):**
> "One weakness of this approach is that samples may be diverse but still not cover the feasible output space."

**[RVD Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10606505/):**
> "Although CRPS is not commonly used in evaluating video prediction methods, it adds a valuable perspective on a model's uncertainty calibration."
> *Note: Even this paper does NOT validate that ground truth falls within generated distributions.*

### The "Best of 100" Methodology Problem

Video VAE papers typically:
1. Generate 100 samples
2. Pick the one closest to ground truth
3. Report metrics on that "best" sample

**This does NOT answer:** "Does the 90% CI of the predicted distribution contain the ground truth?"

### What Would Be Needed for Our Use Case

For volatility surface forecasting, we need:
- **Coverage**: P(GT ∈ CI) = 90% (if using 90% CI)
- **Calibration**: Predicted uncertainty matches actual uncertainty
- **CRPS or similar proper scoring rules**

**Video VAE literature does not provide this evidence.**

---

# Part 2: Research Findings (Architecture)

## Problem Statement

The current decoder architecture fails during autoregressive chaining because:

1. **Architectural mismatch**: Encoder is CNN→LSTM (temporal), decoder is MLP-only (no temporal/spatial structure)
2. **Bandaid fixes (Options 1-6)** addressed symptoms, not root cause
3. **Exposure bias**: Training on ground truth, inference on own predictions
4. **No spatial coherence**: MLP outputs 25 independent scalars

This is fundamentally a **video generation problem** - generating a sequence of 2D surfaces where variations should contain ground truth.

---

## Research: How Video VAEs Solve This

### Key Papers Reviewed

| Paper | Key Contribution |
|-------|------------------|
| [IV-VAE (CVPR 2025)](https://arxiv.org/abs/2411.06449) | Keyframe-based temporal compression + Group Causal Convolution |
| [CausVid](https://arxiv.org/html/2412.07772v1) | Bidirectional→Causal distillation, distribution matching |
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | Causal 3D VAE with CausalConv3d (open-source PyTorch) |
| [AR-Diffusion](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_AR-Diffusion_Asynchronous_Video_Generation_with_Auto-Regressive_Diffusion_CVPR_2025_paper.pdf) | Time-agnostic encoder + Temporal causal decoder |

### Core Architectural Patterns

**1. Causal 3D Convolution**
```python
# Asymmetric temporal padding: (kernel-1, 0) ensures frame t only sees t and earlier
padding = (W_pad, W_pad, H_pad, H_pad, kernel_t-1, 0)  # (left,right) for W,H,T
```

**2. Encoder-Decoder Symmetry**
```
Encoder: Input (B,C,T,H,W) → CausalConv3d → Downsample → Latent (B,C',T/4,H/8,W/8)
Decoder: Latent → CausalConvTranspose3d → Upsample → Output (B,C,T,H,W)
```

**3. Group Causal Convolution (IV-VAE)**
- Standard convolution WITHIN frame groups (inter-frame equivalence)
- Causal padding BETWEEN groups (no future leakage)

**4. Exposure Bias Solutions**
- **Distribution Matching Distillation**: Train causal student with bidirectional teacher
- **Scheduled Sampling**: Gradually replace GT with model predictions during training

---

## Current vs Proposed Architecture

### Current (Broken)
```
Encoder: Surface (B,T,5,5) → Flatten → MLP → LSTM → z (B,T,16)
Decoder: z (B,T,16) → MLP [16→128→64→25] → Reshape → Output (B,T,5,5)

Problems:
- Encoder LSTM is bidirectional (sees future)
- Decoder has NO temporal structure
- Decoder has NO spatial structure (25 independent outputs)
- z is the ONLY input to decoder (no context)
```

### Proposed: Causal 3D VAE
```
Encoder: Surface (B,1,T,5,5) → CausalConv3d stack → z (B,C,T',H',W')
Decoder: z → CausalConvTranspose3d stack → Output (B,1,T,5,5)

Key properties:
- BOTH encoder and decoder are causal (temporal masking)
- BOTH preserve spatial structure (3D convolution)
- Symmetric architecture (decoder mirrors encoder)
- Supports streaming inference (frame-by-frame)
```

---

## Implementation Plan

### Phase 1: CausalConv3d Module (ACTUAL CODE from HunyuanVideo)

Source: https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/vae/unet_causal_3d_blocks.py

```python
class CausalConv3d(nn.Module):
    """
    Implements a causal 3D convolution layer where each position only depends
    on previous timesteps and current spatial locations.
    This maintains temporal causality in video generation tasks.

    Source: HunyuanVideo (Tencent)
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode='replicate',
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        # KEY INSIGHT: Asymmetric temporal padding (kernel-1, 0) ensures causality
        # Padding order: (W_left, W_right, H_left, H_right, T_left, T_right)
        padding = (kernel_size // 2, kernel_size // 2,    # W: symmetric
                   kernel_size // 2, kernel_size // 2,    # H: symmetric
                   kernel_size - 1, 0)                     # T: left-only (CAUSAL!)
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride,
                              dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)
```

**Causal Padding Explained:**
- Spatial (W, H): symmetric padding `kernel//2` on both sides
- Temporal (T): asymmetric padding `(kernel-1, 0)` - only left padding
- This ensures frame t can only see frames ≤t, never t+1

### Phase 2: Port HunyuanVideo Building Blocks

Source: https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/vae/

**Files to port (simplified for our 5x5 grid):**

1. **`unet_causal_3d_blocks.py`** → `vae/causal_3d_blocks.py`
   - `CausalConv3d` - core causal convolution
   - `ResnetBlockCausal3D` - residual block with GroupNorm, SiLU activation
   - `DownsampleCausal3D` - strided causal conv for compression
   - `UpsampleCausal3D` - nearest-neighbor interpolation + causal conv

2. **`vae.py`** → `vae/causal_3d_vae.py`
   - `EncoderCausal3D` - stacks down_blocks with optional attention
   - `DecoderCausal3D` - stacks up_blocks mirroring encoder
   - `DiagonalGaussianDistribution` - for VAE sampling

**Key HunyuanVideo ResnetBlockCausal3D structure:**
```python
# Simplified from HunyuanVideo (full version has time embeddings, etc.)
class ResnetBlockCausal3D(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.nonlinearity = nn.SiLU()

        # Shortcut if channels differ
        self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1) \
                             if in_channels != out_channels else None

    def forward(self, x):
        h = self.nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = self.nonlinearity(self.norm2(h))
        h = self.conv2(h)

        if self.conv_shortcut:
            x = self.conv_shortcut(x)
        return x + h
```

**Key HunyuanVideo UpsampleCausal3D approach:**
```python
# HunyuanVideo uses nearest-neighbor + conv, NOT transposed conv
# This avoids checkerboard artifacts and maintains causality
class UpsampleCausal3D(nn.Module):
    def __init__(self, channels, upsample_factor=(2, 2, 2)):
        self.upsample_factor = upsample_factor
        self.conv = CausalConv3d(channels, channels, kernel_size=3)

    def forward(self, x):
        # Special handling: first frame upsampled only spatially
        B, C, T, H, W = x.shape
        first_h, other_h = x.split((1, T - 1), dim=2)

        if T > 1:
            other_h = F.interpolate(other_h, scale_factor=self.upsample_factor, mode="nearest")

        first_h = first_h.squeeze(2)
        first_h = F.interpolate(first_h, scale_factor=self.upsample_factor[1:], mode="nearest")
        first_h = first_h.unsqueeze(2)

        x = torch.cat((first_h, other_h), dim=2) if T > 1 else first_h
        return self.conv(x)
```

### Phase 3: Adapt for 5x5 Vol Surface Grid

**Key adaptations needed:**

| HunyuanVideo | Our Adaptation |
|--------------|----------------|
| Spatial compression 8x (256→32) | NO spatial compression (5x5 too small) |
| Temporal compression 4x | Keep or reduce based on context length |
| 4 latent channels | 16 latent channels (more capacity needed) |
| Groups=32 in GroupNorm | Groups=8 or less (fewer channels) |

### Phase 4: Training with Scheduled Sampling

```python
def train_step(model, batch, epoch, max_epochs):
    # Scheduled sampling: gradually use model predictions
    teacher_forcing_ratio = max(0.5, 1 - epoch / max_epochs)

    if random.random() < teacher_forcing_ratio:
        # Use ground truth (standard training)
        output = model(batch)
    else:
        # Use model's own predictions (exposure bias correction)
        output = model.generate_autoregressive(batch[:, :context_len])
```

---

## Files to Create/Modify

| File | Action | Source |
|------|--------|--------|
| `vae/causal_3d_blocks.py` | PORT from HunyuanVideo | `hyvideo/vae/unet_causal_3d_blocks.py` |
| `vae/causal_3d_vae.py` | PORT from HunyuanVideo | `hyvideo/vae/vae.py` + `autoencoder_kl_causal_3d.py` |
| `config/causal_3d_config.py` | CREATE | New config for our adaptations |
| `experiments/backfill/two_stage_vae/train_causal_3d_vae.py` | CREATE | Training script with scheduled sampling |
| `experiments/backfill/two_stage_vae/eval_causal_3d_coverage.py` | CREATE | Coverage/CRPS evaluation (NOT in HunyuanVideo) |

### HunyuanVideo Repository Reference

```bash
# Clone for reference (optional - we'll port key classes)
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git /tmp/hunyuan_ref

# Key files to study:
# - hyvideo/vae/unet_causal_3d_blocks.py  # CausalConv3d, ResnetBlock, Up/Downsample
# - hyvideo/vae/vae.py                     # EncoderCausal3D, DecoderCausal3D
# - hyvideo/vae/autoencoder_kl_causal_3d.py # AutoencoderKLCausal3D wrapper
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Causal 3D Conv (not 2D+LSTM)** | Explicit temporal masking, no hidden state drift |
| **Encoder-Decoder symmetry** | Video VAE standard, proven to work |
| **Spatial structure preserved** | 3D conv maintains grid correlations naturally |
| **Scheduled sampling** | Addresses exposure bias at training time |
| **Group causal conv (optional)** | Better frame equivalence if needed |

---

## Adaptations for Vol Surface (5x5 grid vs Video)

| Aspect | HunyuanVideo | Our Adaptation |
|--------|--------------|----------------|
| Spatial size | 256x256 to 1024x1024 | 5x5 (fixed, small) |
| Spatial downsampling | 8x (conv stride) | None (5x5 too small) |
| Temporal downsampling | 4x | Optional (depends on context length) |
| Latent shape | (B, C, T/4, H/8, W/8) | (B, C, T, 5, 5) - keep spatial |
| Channel depth | 4-16 latent channels | 16 latent channels |

**Key adaptation:** Since our spatial grid is only 5x5, we cannot downsample spatially like video VAEs (which go from 256→32 or similar). Instead, we:
1. Keep spatial dimensions fixed at 5x5
2. Use channel expansion to increase capacity
3. Focus temporal compression if needed (T→T/2 or T/4)

---

## Verification Plan

### Standard Video VAE Metrics (for comparison)
1. **Unit test CausalConv3d**: Verify temporal masking (frame t cannot see t+1)
2. **Reconstruction quality**: Compare with current architecture on single-step
3. **Chaining stability**: 30-day sequences, measure explosion rate
4. **Spatial coherence**: Verify smile/skew structure preserved

### Coverage Metrics WE MUST ADD (not standard in video VAE literature)

Since video VAE papers do NOT validate coverage, we must implement:

1. **CI Coverage Rate**:
   ```python
   # For each grid point and horizon:
   # P(GT ∈ [p5, p95]) should ≈ 90%
   coverage = (gt >= p5) & (gt <= p95)
   coverage_rate = coverage.mean()  # Target: 90%
   ```

2. **CRPS (Continuous Ranked Probability Score)**:
   ```python
   # Proper scoring rule for probabilistic forecasts
   def crps(samples, gt):
       # samples: (n_samples,), gt: scalar
       sorted_samples = np.sort(samples)
       n = len(samples)
       # CRPS = E|X-y| - 0.5*E|X-X'|
       term1 = np.mean(np.abs(samples - gt))
       term2 = np.mean(np.abs(samples[:, None] - samples[None, :]))
       return term1 - 0.5 * term2
   ```

3. **Per-Horizon Coverage Degradation**:
   ```python
   # Coverage should not degrade too much over horizon
   for h in [1, 7, 14, 30]:
       print(f"Horizon {h}: Coverage = {coverage_at_horizon(h):.1%}")
   ```

4. **Calibration Plot**:
   - X-axis: Nominal coverage (10%, 20%, ..., 90%)
   - Y-axis: Empirical coverage
   - Perfect calibration = diagonal line

---

## References

- [HunyuanVideo GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo) - Reference PyTorch implementation
- [IV-VAE Paper](https://arxiv.org/abs/2411.06449) - Group causal convolution
- [CausVid](https://arxiv.org/html/2412.07772v1) - Bidirectional to causal distillation
