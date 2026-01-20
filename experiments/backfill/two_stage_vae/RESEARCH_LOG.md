# Research Log: Causal 3D VAE & Multi-Horizon Diffusion

This document tracks the chronological research progress, findings, code changes, and rationale for the volatility surface forecasting project.

---

## 2026-01-20: Multi-Horizon IV Surface Diffusion Research Synthesis

### Context
After discovering that the Causal 3D VAE (ported from HunyuanVideo) cannot produce proper CI coverage due to architectural limitations, researched how to extend one-step IV surface DDPM to multi-step.

### Key Findings

**1. Literature Gap Identified**

| Domain | Best Paper | Capability | Limitation |
|--------|------------|------------|------------|
| Finance | arxiv:2511.07571 | One-step IV DDPM, 90% CI coverage | h=1 only |
| Video | PA-VDM, HunyuanVideo | Coherent 60+ frame sequences | No CI validation |
| Time Series | ARMD, REDI | Multi-horizon forecasting | Not for IV surfaces |

**Nobody has combined multi-step generation with CI coverage validation for IV surfaces.**

**2. Techniques to Combine**

From Video Diffusion:
- Progressive noise levels (PA-VDM): Earlier frames guide later frames
- Chunked frame denoising: Prevents cumulative error
- Causal temporal attention: Frame t only sees ≤t

From Time Series:
- All-at-once generation (ARMD): Avoids autoregressive error accumulation
- Multi-resolution: Coarse→fine decomposition

From IV Surface DDPM:
- FiLM conditioning for scalars (VIX, returns)
- SNR-weighted arbitrage penalty
- EWMA context channels

### Code Changes

**File**: `experiments/backfill/two_stage_vae/VIDEO_VAE_RESEARCH.md`
**Commit**: `dbbcfbf`

Added 464 lines documenting:
- Research synthesis from 15+ papers
- Proposed multi-horizon architecture
- Training/sampling pseudocode
- Evaluation framework for CI at all horizons
- Implementation roadmap

### Rationale

The Causal 3D VAE approach failed because video VAEs are designed as deterministic codecs - diversity comes from a separate diffusion model. Rather than fighting the architecture, we should build a proper multi-horizon diffusion model that combines proven techniques from video, time series, and finance domains.

### Next Steps

1. Implement small 3D U-Net with progressive noise
2. Add FiLM conditioning and arbitrage penalty
3. Evaluate CI coverage at h=1,7,14,30,60

---

## 2026-01-20: Decoder Variance Squashing Root Cause Analysis

### Context
Causal 3D VAE achieved only 33% CI coverage (target: 90%) despite healthy posterior (logvar=-0.1, std=0.97).

### Key Findings

**1. Posterior is NOT Collapsed**

Diagnostic results from `diagnose_causal_3d_posterior.py`:
```
Logvar mean: -0.10 (healthy, not -30)
Posterior std: 0.97 (healthy)
z sample std: 0.96 (diverse latents)
Output std: 0.008 (PROBLEM - nearly identical outputs)
```

**2. Root Cause: Decoder Squashes Variance**

100 different z samples (std=0.96) → nearly identical outputs (std=0.008)

Why:
- GroupNorm layers (5-7 in decoder) normalize to std≈1, erasing z-dependent variance
- MSE loss rewards mean prediction, no incentive for variance
- This is BY DESIGN - video VAE decoders are meant to be deterministic

**3. Architectural Insight**

Video VAE pipeline:
```
Video → VAE Encoder → z → [DIFFUSION MODEL] → z' → VAE Decoder → Output
                           ↑
                     Diversity comes from here, NOT from VAE sampling
```

We were misusing the architecture - trying to get diversity from VAE sampling when it should come from a diffusion model.

### Code Changes

**File**: `experiments/backfill/two_stage_vae/diagnose_causal_3d_posterior.py`
**Commit**: `a29f3d2`

Created diagnostic script with 4 tests:
1. Posterior statistics (logvar, std)
2. Sample diversity (100 z samples)
3. Decoder sensitivity (z perturbation response)
4. KL contribution analysis

### Rationale

Initial hypothesis was posterior collapse (KL weight too small). Diagnostic proved this wrong - the posterior is healthy. The real issue is decoder design, which led to the research pivot toward diffusion models.

---

## 2026-01-20: Causal 3D VAE Training Results (50 Epochs)

### Context
Trained scaled-down Causal 3D VAE (473K params) for 50 epochs in oracle mode.

### Key Findings

**Training Metrics**
```
Epoch 50: Train Loss=0.0031, Val Loss=0.0032, KL=1.25
```

**Evaluation Results**
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| 90% CI Coverage | 33.4% | 90% | FAIL |
| Explosion Rate | 0.3% | 0% | GOOD |
| Kurtosis | 2.28x GT | 1.0x | FAIL |

**Per-Horizon Coverage**
```
h=1:  51%
h=7:  42%
h=14: 35%
h=30: 27%
h=60: 22%
```

Coverage degrades with horizon - model cannot produce wide enough CIs.

### Code Changes

**Files Created** (Commit `a29f3d2`):
- `vae/causal_3d_blocks.py` (542 lines) - CausalConv3d modules
- `vae/causal_3d_vae.py` (664 lines) - Encoder/Decoder/Autoencoder
- `config/causal_3d_config.py` (79 lines) - Configuration
- `train_causal_3d_vae.py` (518 lines) - Training script
- `eval_causal_3d_coverage.py` (457 lines) - Basic evaluation
- `eval_causal_3d_comprehensive.py` (691 lines) - Full evaluation

**Model Saved**: `models/backfill/causal_3d/best_model.pt`

### Rationale

Ported HunyuanVideo's Causal 3D VAE architecture, scaled down for 5×5 grid (473K vs 9.8M params). Good reconstruction and low explosion rate, but poor CI coverage led to root cause investigation (see above).

### Issues Encountered

**OOM during AR training**: At epoch 50, enabling autoregressive training caused CUDA OOM. AR training requires gradients over 60 sequential steps. Added `--no-ar-training` flag as workaround.

---

## 2026-01-20: HunyuanVideo/Custom Architecture Research

### Context
Investigated how HunyuanVideo actually generates diverse videos.

### Key Findings

**1. HunyuanVideo VAE Training**

Actual loss function (not simple MSE+KL):
```
Loss = L₁ + 0.1·L_lpips + 0.05·L_adv + 10⁻⁶·L_kl
```
- L₁: Pixel reconstruction
- L_lpips: Perceptual loss (pretrained VGG)
- L_adv: GAN discriminator
- L_kl: KL divergence

**2. HunyuanVideo Diffusion Model**

The diversity engine (we didn't port this):
- 60 transformer layers
- 24 attention heads × 128 dim
- 50 denoising steps
- Billions of parameters

**3. HunyuanCustom Video Conditioning**

For conditioning on video (not just text):
```
Context video → VAE Encode → z_cond
z_cond → 4-layer FC alignment → aligned_features
aligned_features + noisy_z → Diffusion → clean_z
```

Injection method: Frame-by-frame ADDITION (not concatenation)

### Code Changes

None - research only. Documented in VIDEO_VAE_RESEARCH.md.

### Rationale

Understanding the full HunyuanVideo pipeline revealed why our VAE-only approach failed. The VAE is just a codec; the diffusion model (which we didn't port) is responsible for diversity.

---

## 2026-01-20: IV Surface DDPM Literature Review

### Context
Searched for existing diffusion models for IV surface forecasting.

### Key Findings

**1. arxiv:2511.07571 (Nov 2025)**

"Forecasting Implied Volatility Surface with Generative Diffusion Models"
- One-day-ahead conditional DDPM
- 9×9 IV grid
- U-Net architecture
- FiLM conditioning for scalars
- SNR-weighted arbitrage penalty
- **90% CI coverage achieved** (but only h=1)

Code: https://github.com/Austinjinc/rep_volgan/tree/refactor/new-ddpm

**2. No Multi-Step Validation**

The paper explicitly states: "one-day-ahead conditional forecasting"
- No autoregressive chaining
- No multi-horizon evaluation
- CI coverage only validated for h=1

### Code Changes

None - research only. Documented in VIDEO_VAE_RESEARCH.md.

### Rationale

This paper proves DDPM can achieve proper CI coverage for IV surfaces, but only for one-step. Extending to multi-step is our innovation opportunity.

---

## Research Timeline Summary

| Date | Finding | Code Change | Impact |
|------|---------|-------------|--------|
| 2026-01-20 | Causal 3D VAE: 33% CI coverage | 8 files, 3979 lines | Identified failure mode |
| 2026-01-20 | Decoder squashes variance | diagnose script | Root cause found |
| 2026-01-20 | Video VAE = deterministic codec | - | Architectural insight |
| 2026-01-20 | IV DDPM exists for h=1 | - | Literature baseline |
| 2026-01-20 | Multi-horizon gap identified | VIDEO_VAE_RESEARCH.md | Innovation opportunity |
| 2026-01-20 | Research synthesis | 464 lines docs | Proposed architecture |

---

## Commit History

```
dbbcfbf docs: Add research synthesis for multi-horizon IV surface diffusion
a29f3d2 feat: Add Causal 3D VAE architecture ported from HunyuanVideo
3ef8464 feat: Add cumulative-aware decoder options (1-6) for stable AR chaining
```

---

## Open Questions

1. What is the minimum model size for 5×5 grid diffusion?
2. How many denoising steps needed for proper CI coverage?
3. Can we use the existing VAE encoder for context embedding?
4. What is the optimal progressive noise schedule for 60-day horizon?

---

## References

See VIDEO_VAE_RESEARCH.md for full reference list.
