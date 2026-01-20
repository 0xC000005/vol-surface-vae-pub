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

# REFERENCES

- [HunyuanVideo GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo) - Reference VAE implementation
- [HunyuanCustom Paper](https://arxiv.org/abs/2505.04512) - Video conditioning mechanism
- [HunyuanCustom GitHub](https://github.com/Tencent-Hunyuan/HunyuanCustom) - Video-conditioned diffusion
- [IV-VAE Paper](https://arxiv.org/abs/2411.06449) - Group causal convolution
- [CausVid](https://arxiv.org/html/2412.07772v1) - Bidirectional to causal distillation

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
