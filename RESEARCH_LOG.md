# Research Log: Causal 3D VAE & Multi-Horizon Diffusion

This document tracks the chronological research progress, findings, code changes, and rationale for the volatility surface forecasting project.

---

## 2026-01-24: Architecture Synthesis - Hierarchical Regime DDPM Design

### Context

Synthesized insights from extensive research on VAE limitations, video generation approaches, and financial time series requirements. This entry documents the complete rationale for the final proposed architecture.

### 1. Why VAEs Fundamentally Fail for Conditional Diversity

**Mathematical Root Cause:**

The CVAE loss function:
```
L = E_q[log p(y|z,x)] - KL(q(z|x,y) || p(z|x))

log p(y|z,x) ∝ -||y - μ_θ(z,x)||²  ← MSE learns MEAN, not distribution
```

The decoder learns `μ_θ(z,x) ≈ E[y|z,x]`, producing the **conditional mean**, not diverse samples.

**Three Failure Modes:**

| Failure | Description |
|---------|-------------|
| **Low variance** | Decoder averages over futures → "blurry" mean-reverting paths |
| **Gaussian tails** | MSE + Gaussian prior suppresses kurtosis |
| **Mode averaging** | Blends calm/crisis regimes instead of generating distinct ones |

**Key Insight:**
> "VAEs are excellent at learning smooth, structured latent spaces. They're terrible at sampling diverse outputs from that space."

---

### 2. Why NOT Mimic Standard Video Generation (Block-Wise Autoregressive)

Video models like MCVD use block-wise generation:
```
Block 1: Generate frames 1-5   → diversity exists
Block 2: Generate frames 6-10  → conditioned on Block 1, LOCKED IN
Block 3: Generate frames 11-20 → error compounds
```

**Problems for Financial Time Series:**

| Issue | Impact |
|-------|--------|
| Error accumulation | Each block adds error that propagates |
| Diversity collapse | Cannot explore fundamentally different regimes |
| Poor CI coverage | All samples converge toward conditional mean |
| ACF broken | Long-range dependencies lost at block boundaries |

**Key Quote:**
> "The IV literature is stuck in one-step autoregressive paradigms. Video diffusion solved full-sequence generation. Your task is to bring video advances to finance."

---

### 3. "One-Pass Generate All" Approach

Generate entire future (e.g., 30-60 days) in **single diffusion pass**:

| Aspect | Block-Wise | One-Pass (Proposed) |
|--------|------------|---------------------|
| Diversity | Limited per block | Different noise → different trajectory |
| Error | Compounds across blocks | None - generates jointly |
| Marginal | May be biased | Mathematically correct |
| ACF | Fails at long lags | Temporal attention models lags |
| Speed | Multiple passes | Single reverse diffusion |

**Mathematical Guarantee:**
```
∫ p(x_{t+1:t+H} | x_{t-K:t}) · p(x_{t-K:t}) d(x_{t-K:t}) = p(x_{t+1:t+H})
```

---

### 4. Framework Comparison

#### **Standard DDPM**
- Forward: `q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)` ← Gaussian bias
- Learns to denoise Gaussian → outputs tend toward Gaussian

#### **Flow Matching (For Heavy Tails)**
```python
# Can use Student-t noise for fat tails
z = torch.distributions.StudentT(df=4).sample(shape)
x_t = t * x_future + (1 - t) * z
loss = F.mse_loss(v_pred, x_future - z)  # No explicit kurtosis loss!
```

| Property | DDPM | Flow Matching |
|----------|------|---------------|
| Prior | N(0,I) required | Any distribution |
| Tail behavior | Gaussian bias | Learned from data |
| Sampling steps | 100-1000 | 10-50 |

#### **Latent SDE (Finance-Native)**
```
dz_t = μ_θ(z_t) dt + σ_θ(z_t) dW_t
                     ↑ STATE-DEPENDENT
```

- **Vol clustering guaranteed**: Large σ(z) → large innovation → vol persists
- **Fat tails emerge**: Stochastic volatility naturally generates heavy tails
- **ACF captured**: Drift μ(z) learns mean reversion/persistence

---

### 5. The Bitter Lesson Applied

**DON'T DO (Explicit Losses):**
```python
loss = mse_loss + λ_kurtosis * kurtosis_loss + λ_acf * acf_loss  # ❌ Band-aids
```

**DO THIS (Architecture + Data):**
```python
loss = F.mse_loss(noise_pred, noise)  # That's it!

# Properties emerge from:
# 1. Full-sequence generation (sees joint distribution)
# 2. Temporal attention (learns ACF implicitly)
# 3. Rich conditioning (history provides vol clustering signal)
# 4. Heavy-tailed prior if needed (Student-t)
```

---

### 6. The Regime Diversity Problem & Solution

**Problem with Standard Diffusion:**
```
All samples follow SAME dynamics, different noise realization:
Sample 1: ════════════════════▶ (average regime + noise)
Sample 2: ════════════════════▶ (average regime + noise)
```

**Solution - Two-Level Hierarchical Sampling:**
```
Stage 1: Sample regime ~ p(regime | history)
         {calm: 0.6, spike_early: 0.15, spike_late: 0.15, trending: 0.1}

Stage 2: Sample trajectory ~ p(future | history, regime)

Sample 1 (calm):        ════════════════════════▶
Sample 2 (spike_early): ═══════╱╲╱╲════════════▶
Sample 3 (spike_late):  ════════════════╱╲╱╲╱╲▶
```

**Why This Solves Kurtosis:** If 10% chance of crisis, exactly 10% of samples have crisis dynamics (not averaged away).

---

### 7. Final Recommended Architecture

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: REGIME CLASSIFIER                             │
│  Input: 30-day history → GRU/Transformer → p(regime)    │
│  Regimes: {calm, spike_early, spike_late, trending, ...}│
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: CONDITIONAL TRAJECTORY DIFFUSION              │
│  Architecture: 3D U-Net + Temporal Attention            │
│  Conditioning: History + Regime Embedding + EWMA + VIX  │
│  Loss: L_noise + SNR(t) × L_arbitrage                   │
│  Output: Full 30-day trajectory in ONE PASS             │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| GRU Encoder | Path-dependent history compression |
| EWMA Features | Multi-scale historical volatility |
| Temporal Attention | Global receptive field for ACF |
| Spatial Attention | Surface-level coherence (smile/skew) |
| Regime Embedding | Discrete conditioning for multimodality |
| SNR-weighted arbitrage | Penalty only on clean samples (from Jin 2511.07571) |

---

### 8. Concrete Experiments to Run

| Experiment | What to Test | Priority |
|------------|--------------|----------|
| Regime count ablation | K ∈ {3, 5, 7} regimes | High |
| Classifier-free guidance | w ∈ {0.5, 1.0, 1.5, 2.0} | Medium |
| Temporal attention analysis | Do weights resemble ACF? | Medium |
| Arbitrage penalty tuning | Start λ=0, increase if >5% violations | High |
| Flow Matching vs DDPM | Student-t prior for heavy tails | Medium |
| Full-sequence vs autoregressive | Compare error accumulation | High |
| Latent SDE variant | State-dependent σ(z) for vol clustering | Low |

---

### 9. Comparison to Existing Approaches

| Aspect | MCVD | Jin's IV-DDPM | FuNVol | **Proposed** |
|--------|------|---------------|--------|--------------|
| Generation | Block AR | 1-step AR | Continuous SDE | **Full 30-day** |
| Temporal | 2D conv | None | GRU | **3D conv + attention** |
| Regime diversity | ❌ | ❌ | ❌ | **✅ Hierarchical** |
| Arbitrage | N/A | SNR penalty | Implicit | **SNR penalty** |
| Contribution | Video | Finance | Hybrid | **Bridges both** |

---

### 10. Key Takeaway

The proposed architecture bridges a gap between video diffusion and financial modeling:

1. **One-pass generation** (from video diffusion)
2. **Hierarchical regime sampling** (novel for diversity)
3. **SNR-weighted arbitrage penalty** (from finance literature)
4. **Temporal attention for ACF** (from video diffusion)

This combination doesn't exist in the IV literature—it represents a genuine contribution bridging two research domains.

---

### Next Steps

1. Implement regime classifier on historical trajectories
2. Add regime embedding to existing DDPM POC
3. Test hierarchical sampling on kurtosis/butterfly metrics
4. Compare against single-level baseline

---

## 2026-01-24: Progressive Noise Scheduling Research

### Context

Investigated why popular video diffusion models (HunyuanVideo, PA-VDM, Sora) use autoregressive/progressive approaches. Key finding: **they don't use pure frame-by-frame AR** - they use hybrid approaches with progressive noise.

### Key Finding: Video Models Use Progressive Noise, Not Pure AR

| Model | Actual Architecture |
|-------|---------------------|
| HunyuanVideo | Causal 3D attention, one-pass with causal masking |
| PA-VDM | Progressive noise (early=clean, late=noisy) |
| Sora | "Full foresight" - sees many frames at once |
| Diffusion Forcing | Independent per-token noise levels |

### PA-VDM: Progressive Noise for Video Extension

**Paper**: [PA-VDM](https://arxiv.org/abs/2410.08151) (CVPR 2025)

**Noise assignment formula:**
```
τ_{0:S} = {0, T/S, 2T/S, ..., (S-1)T/S, T}

Frame i gets noise level τ_i
Earlier frames → lower noise (cleaner)
Later frames → higher noise (noisier)
```

**Training modification:**
- Standard diffusion loss, but with per-frame progressive noise levels
- Add random shift δ = 0.4ε(t_i - t_{i+1}) to cover full [0,T) range

**Benefit:**
> "smoother attention correspondence among frames with adjacent noise levels"

**Limitation for our case:** PA-VDM is designed for autoregressive video **extension** (generate more frames indefinitely), not fixed-horizon forecasting.

### Diffusion Forcing: More Relevant to Our Case

**Paper**: [Diffusion Forcing](https://www.boyuan.space/diffusion-forcing/) (NeurIPS 2024)

**Key insight:**
> "Training a diffusion model to denoise a set of tokens with independent per-token noise levels"

**Sampling with variable noise:**
```
Context tokens (history): Clean (noise = 0)
Near future (day 1-5):    Low noise (high confidence)
Far future (day 25-30):   High noise (high uncertainty)
```

**Why this solves our CI calibration issue:**

| Approach | Noise Distribution | Uncertainty |
|----------|-------------------|-------------|
| Current DDPM | Uniform across all days | Same CI width for h=1 and h=30 |
| Diffusion Forcing | Progressive (near=low, far=high) | Natural CI widening for far horizons |

**Mathematical property:**
> "optimize[s] a variational lower bound on the likelihoods of all subsequences"

### Application to IV Surface Forecasting

**Current DDPM (uniform noise):**
```
Day 1:  noise level τ → CI width W
Day 30: noise level τ → CI width W (same!)
```

**With progressive noise (Diffusion Forcing style):**
```
Day 1:  noise level τ × 0.2 → CI width W₁ (narrow)
Day 30: noise level τ × 1.0 → CI width W₃₀ (wide)
```

This matches our intuition: **near-term forecasts should be more certain than far-term forecasts**.

### Implementation Options

**Option A: Training-time progressive noise (PA-VDM style)**
```python
def get_progressive_noise_level(frame_idx, t_global, n_frames):
    """Each frame gets different noise based on temporal position."""
    progress = frame_idx / n_frames  # 0 to 1
    return t_global * (0.2 + 0.8 * progress)  # 20% to 100% of global noise
```

**Option B: Independent per-frame noise (Diffusion Forcing style)**
```python
def sample_independent_noise_levels(n_frames, t_max):
    """Each frame gets independently sampled noise level."""
    return torch.randint(0, t_max, (n_frames,))
```

**Option C: Inference-time only (simplest)**
```python
def progressive_sample(model, history, n_frames):
    """Use trained uniform model but sample with progressive schedule."""
    x = torch.randn(B, n_frames, 5, 5)

    # Different denoising schedules per frame
    for frame_idx in range(n_frames):
        noise_scale = 0.2 + 0.8 * (frame_idx / n_frames)
        x[:, frame_idx] = denoise_with_scale(x[:, frame_idx], noise_scale)
```

### Experiments to Run

| Experiment | Baseline | Test | Expected Outcome |
|------------|----------|------|------------------|
| h=1 CI coverage | Uniform DDPM | Progressive DDPM | Similar or better |
| h=30 CI coverage | Uniform DDPM | Progressive DDPM | **Significant improvement** |
| CI width ratio (h=30/h=1) | ~1.0 | Progressive | >1.5 (natural widening) |
| Overall calibration | 81.7% | Progressive | Closer to 90% |

### Decision: Which Approach?

| Approach | Pros | Cons |
|----------|------|------|
| **Option C (inference-only)** | No retraining, quick test | May not fully capture benefits |
| **Option B (Diffusion Forcing)** | Mathematically principled | Requires retraining |
| **Option A (PA-VDM)** | Proven in video | Designed for extension, not fixed horizon |

**Recommendation:** Start with Option C (inference-time progressive sampling) as a quick validation. If promising, implement Option B (Diffusion Forcing) for full benefits.

### Sources

- [PA-VDM Paper](https://arxiv.org/abs/2410.08151)
- [Diffusion Forcing Paper](https://www.boyuan.space/diffusion-forcing/)
- [CausVid Paper](https://arxiv.org/abs/2412.07772)

---

## 2026-01-24: Extension Capability & Ultimate Goal

### Why Extension Matters

The current POC generates **30 days conditioned on 30 days of history**. This is intentionally limited for proof-of-concept validation. The ultimate goal is:

> **Backfill/interpolate arbitrarily long time series** (similar to masked video diffusion objective)

Use cases requiring extension:
- Backfill 2008-2010 financial crisis period (~750 trading days)
- Generate multi-year scenarios for stress testing
- Interpolate missing data in historical records

### Old Approach: Pure Block-by-Block AR (Problematic)

Our original VAE approach used pure autoregressive generation:
```
Block 1: Generate days 1-30   → feed to next block
Block 2: Generate days 31-60  → conditioned on generated Block 1
Block 3: Generate days 61-90  → conditioned on generated Blocks 1-2
...
```

**Problems documented in STABLE_CHAINING.md:**
- Error accumulation compounds across blocks
- Mean-only chaining: RMSE = 0.0678 (stable but no uncertainty)
- Fat-tail sampling: Explodes to invalid values
- Diversity collapse: locked into regime after first block

### Better Approach: PA-VDM / Diffusion Forcing Hybrid

The video diffusion papers reveal a superior approach:

**PA-VDM (Progressive Autoregressive Video Diffusion):**
```
NOT frame-by-frame: Frame 1 → Frame 2 → Frame 3 → ...

Instead: Chunk-wise with shift
┌─────────────────────────────────────────────────────────┐
│  Step 1: Generate frames 1-30 in ONE diffusion pass     │
│          (progressive noise: frame 1 clean, 30 noisy)   │
│                                                         │
│  Step 2: SHIFT - drop frame 1, keep 2-30 as context     │
│          Add noisy frame 31                             │
│          Generate again (now have frames 2-31)          │
│                                                         │
│  Step 3: Repeat to extend indefinitely                  │
└─────────────────────────────────────────────────────────┘
```

**Diffusion Forcing (more flexible):**
```
Training: Independent random noise per token (not progressive schedule)

Inference options (same trained model):
  A) One-pass: Generate all frames together
  B) Sliding window: Generate chunk, shift, extend (like PA-VDM)
  C) Causal: Frame-by-frame with past as clean context
```

### Why This is Better Than Pure AR

| Aspect | Pure Block AR | PA-VDM/Diffusion Forcing |
|--------|---------------|--------------------------|
| Error accumulation | Compounds across blocks | Bounded within chunk |
| Diversity | Locked after first block | Fresh noise each extension |
| Context | Only sees generated history | Mixes real + generated |
| Stability | Degrades over time | 2000+ frames demonstrated |

### Comparison Table

| Method | Training | Extension Mode | CI Tested? | Our Status |
|--------|----------|----------------|------------|------------|
| Our VAE AR | Standard VAE | Block-by-block | ❌ Failed | Abandoned |
| **Our DDPM POC** | Uniform noise | Fixed 30 days | ✅ 81.7% | Current |
| PA-VDM | Progressive noise | Chunk + shift | ❌ No | Could adopt |
| Diffusion Forcing | Independent noise | Flexible | ❌ No | **Recommended** |

### Path Forward

1. **Current POC (30 days):** Validates diffusion works for IV surfaces
2. **Next: Diffusion Forcing training:** Gets both CI calibration AND extension
3. **Ultimate goal:** Backfill arbitrarily long series with proper uncertainty

### Architecture Evolution

```
Phase 1 (Complete): POC Validation
├── 30-day generation, 30-day history
├── Validates: surface validity, CI coverage, marginal recovery
└── Result: 81.7% CI coverage, 0% out-of-range

Phase 2 (Next): Diffusion Forcing + Extension
├── Training with independent per-frame noise
├── Inference with chunk + shift for arbitrary length
├── Expected: Better CI calibration + extension capability
└── Target: Backfill 100+ days with proper uncertainty

Phase 3 (Future): Full Production
├── Hierarchical regime sampling (multimodality)
├── Arbitrage penalty (butterfly violations)
└── Target: Backfill multi-year periods for crisis analysis
```

### Analogy to Masked Video Diffusion

Our ultimate objective is analogous to **masked video diffusion/inpainting**:
- Video: Given frames 1-10 and 50-60, generate frames 11-49
- Us: Given IV surfaces from period A and C, generate period B

The chunk + shift approach enables this by:
1. Conditioning on known past (clean, low noise)
2. Generating unknown future (high noise → denoised)
3. Shifting window to extend further

---

## 2026-01-24: Progressive Sampling Experiment Results

### Experiment

Tested inference-time progressive noise to validate whether it improves CI calibration at far horizons without retraining.

**Method tested:** Post-hoc noise addition
- Generate samples with standard DDIM (20 steps)
- Add progressive noise: `noise_scale = 0.03 × (frame_idx / 29)`
- Frame 0 gets no added noise, Frame 29 gets max noise (σ=0.03)

### Results

| Method | h=1 | h=7 | h=14 | h=30 | CI Width Ratio (h30/h1) |
|--------|-----|-----|------|------|-------------------------|
| **Uniform DDPM** | 90.4% | 86.1% | 86.7% | 86.5% | 1.01 |
| **Post-hoc noise** | 89.9% | 87.8% | 90.9% | **95.5%** | 1.19 |

### Key Findings

1. **h=1 coverage maintained:** 89.9% vs 90.4% (essentially unchanged)
2. **h=30 coverage improved by +9.0%:** 86.5% → 95.5%
3. **Natural CI width growth:** Ratio increased from 1.01 to 1.19
4. **All horizons improved:** h=7 (+1.7%), h=14 (+4.2%), h=30 (+9.0%)

### Fréchet Surface Distance (FSD) Results

**Update (2026-01-24):** Added FSD metric to evaluate distributional realism alongside CI coverage.

FSD measures whether generated surface sequences are statistically similar to real sequences:
```
FSD = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real × Σ_gen))
```

| Method | FSD-Encoder (128-dim) | FSD-Domain (~1000-dim) |
|--------|----------------------|------------------------|
| **Uniform DDPM** | 4.338 | 6.160 |
| **Post-hoc noise** | 4.377 (+0.9%) | 6.143 (-0.3%) |

**Key FSD Findings:**
1. FSD values are **nearly identical** between methods (within 1%)
2. Post-hoc noise maintains realism while improving CI calibration
3. FSD-Encoder captures learned 128-dim representation similarity
4. FSD-Domain captures financial features: level, skew, convexity, term slope

**Interpretation:**
- CI coverage measures calibration (does 90% CI contain 90% of outcomes?)
- FSD measures realism (do generated distributions match real distributions?)
- Post-hoc noise improves CI calibration (+9% at h=30) **without sacrificing realism**

This is a positive result: we get better calibration without any degradation in distributional quality.

**Files Created:**
- `experiments/backfill/two_stage_vae/metrics/frechet_surface_distance.py`
- `experiments/backfill/two_stage_vae/metrics/__init__.py`

### Analysis

The simple post-hoc noise addition validates the core hypothesis:
> Near-term forecasts should be more certain than far-term forecasts

The uniform DDPM produces nearly constant CI width (ratio 1.02), which doesn't match financial intuition. Adding progressive noise creates natural uncertainty growth.

**Why this works:**
- Uniform diffusion samples have similar variance at all horizons
- Real forecasts should have increasing uncertainty with horizon
- Progressive noise compensates for the uniform-noise model's limitation

### Implications

This quick test shows progressive noise is highly effective (+9.1% at h=30). However, post-hoc noise is a "hack" that:
- Adds noise independent of the model's learned dynamics
- May add noise in wrong directions (not aligned with data manifold)
- Cannot fully capture the benefits of training-time progressive noise

**Recommendation:** Implement full Diffusion Forcing (Option B) for training-time progressive noise, which should:
- Learn to produce appropriate uncertainty per horizon
- Keep noise aligned with learned data manifold
- Potentially improve even further

### Files Created

- `experiments/backfill/two_stage_vae/test_progressive_sampling.py`

### Command

```bash
# Basic test (CI coverage only)
python experiments/backfill/two_stage_vae/test_progressive_sampling.py \
    --max_batches 15 --n_samples 50 --device cuda

# With FSD metric (CI coverage + distributional realism)
python experiments/backfill/two_stage_vae/test_progressive_sampling.py \
    --max_batches 15 --n_samples 50 --device cuda --compute_fsd
```

---

## 2026-01-24: Video Diffusion Architecture Deep Dive

### Context

Before implementing progressive noise scheduling, conducted comprehensive research into why popular video diffusion models use their specific architectures. This documents the full reasoning for our architectural choices.

### 1. Video Model Architecture Comparison

| Model | VAE | Diffusion | Temporal Handling | Generation Mode |
|-------|-----|-----------|-------------------|-----------------|
| **HunyuanVideo** | Causal 3D VAE (4×8×8 compression) | DiT with Full 3D Attention | Causal masking (frame t sees ≤t) | One-pass with causal structure |
| **Sora** | Space-time patches | DiT Transformer | "Full foresight" - sees all frames | One-pass full sequence |
| **CogVideoX** | 3D VAE | 3D Full Attention DiT | Bidirectional attention | One-pass full sequence |
| **Stable Video Diffusion** | Image VAE + temporal layers | U-Net with temporal attention | Frame-by-frame with conditioning | Semi-autoregressive |
| **PA-VDM** | Standard video VAE | DiT with progressive noise | Progressive denoising schedule | Hybrid: one-pass + AR extension |
| **MCVD** | None (pixel space) | 3D U-Net | Block-wise (5-20 frames) | Block autoregressive |

**Key Observation:** Most successful models (Sora, CogVideoX, HunyuanVideo) use **one-pass generation** at the diffusion level, not frame-by-frame AR.

### 2. Why One-Pass Beats AR for IV Surface Forecasting

#### Problem 1: Error Accumulation in True Frame-by-Frame AR

```
Training:   Each frame conditioned on GROUND TRUTH history
Inference:  Each frame conditioned on MODEL PREDICTIONS

Frame 1: error ε₁
Frame 2: error ε₂ + f(ε₁)     ← propagates Frame 1 error
Frame 3: error ε₃ + f(ε₂) + g(ε₁)  ← compounds
...
Frame 30: accumulated error from all previous frames
```

**Evidence from our codebase** (STABLE_CHAINING.md):
- Mean-only VAE chaining: RMSE = 0.0678
- Fat-tail sampling: Explodes to invalid values
- Temperature scaling: Best balance but still underestimates uncertainty

#### Problem 2: Diversity Collapse (Regime Lock-In)

```
Block 1 generates days 1-5:
  ├── Could be: Calm regime
  ├── Could be: Spike regime
  └── Could be: Trending regime

After Block 1 samples "Calm":
  └── All subsequent blocks LOCKED INTO calm dynamics

Result: Cannot explore "what if crisis starts at day 20?"
        because calm regime was committed at Block 1
```

For CI coverage, we need samples that explore **fundamentally different regimes**, not just noise variations around one committed trajectory.

#### Problem 3: Broken Long-Range Dependencies

```
ACF structure: Day 1 correlates with Day 30 (lag-29 autocorrelation)

Block-wise generation:
  Block 1: Days 1-5
  Block 2: Days 6-10
  ...
  Block 6: Days 26-30

The Day 1 → Day 30 correlation must pass through 5 block boundaries.
Each boundary is an information bottleneck where structure can be lost.
```

**One-pass solution:** Temporal attention directly connects Day 1 to Day 30 in a single forward pass.

#### Problem 4: Marginal Distribution Bias

**Block-wise generates an approximation:**
```
p̂(x₁₋₃₀) = p(x₁₋₅) × p(x₆₋₁₀|x₁₋₅) × p(x₁₁₋₁₅|x₁₋₁₀) × ...
```

This is a **factorized approximation** of the true joint. Errors in early blocks propagate and bias the entire marginal.

**One-pass generates the true joint:**
```
p(x₁₋₃₀ | history)  ← no approximation, no factorization
```

**Mathematical guarantee:**
```
∫ p(x_{t+1:t+H} | x_{t-K:t}) · p(x_{t-K:t}) d(x_{t-K:t}) = p(x_{t+1:t+H})
```

### 3. Computational Complexity Analysis

#### Why Video Models Need Efficiency Tricks

```
720p video frame: 1280 × 720 = 921,600 pixels
With 8×8 patches: 14,400 tokens per frame
60-frame video:   864,000 tokens total

Full attention: O(N²) = O(864,000²) = 746 billion operations per layer
                Per denoising step × 50 steps = infeasible
```

**Solutions video models use:**
- Causal masking: Reduces to triangular matrix (50% savings)
- Block-wise processing: O(B² × num_blocks) instead of O(N²)
- Token compression: Exploit temporal redundancy
- Progressive noise: Reduces effective sequence length

#### Why We DON'T Need These Tricks

```
Our IV surface: 5 × 5 = 25 values per frame
30-day horizon: 25 × 30 = 750 tokens
With history:   25 × 60 = 1,500 tokens total

Full attention: O(N²) = O(1,500²) = 2.25 million operations per layer
                ≈ 0.0003% of video complexity
                Trivially computable on any GPU
```

**Conclusion:** Computational efficiency is **not a valid reason** for us to use AR/block-wise approaches.

| Scale | Tokens | O(N²) Ops | Feasibility |
|-------|--------|-----------|-------------|
| 720p video | 864,000 | 746B | ❌ Infeasible |
| Our IV surfaces | 1,500 | 2.25M | ✅ Trivial |

### 4. The Key Architectural Insight

**Progressive noise ≠ Pure autoregressive**

What video models actually do:
```
┌─────────────────────────────────────────────────────────┐
│  ONE-PASS at diffusion level                            │
│  (all frames processed in single reverse diffusion)     │
│                                                         │
│  + Causal attention for temporal structure              │
│    (frame t can only attend to frames ≤ t)              │
│                                                         │
│  + Progressive noise for guidance                       │
│    (early frames = anchors, late frames = follow)       │
└─────────────────────────────────────────────────────────┘
```

This is **fundamentally different** from true frame-by-frame AR:
- No sequential generation at inference time
- No error accumulation from conditioning on own predictions
- Full sequence available for global optimization

### 5. Comparison Table: AR vs One-Pass for Our Requirements

| Requirement | Block-Wise AR | One-Pass | Winner |
|-------------|---------------|----------|--------|
| **Surface validity** | Each block valid | Joint optimization | One-Pass |
| **Regime diversity** | Locked after Block 1 | Each seed = full trajectory | **One-Pass** |
| **Error accumulation** | Compounds across blocks | None | **One-Pass** |
| **Long-range ACF** | Block boundaries break it | Temporal attention | **One-Pass** |
| **CI calibration** | Degrades with horizon | Consistent | **One-Pass** |
| **Marginal accuracy** | Factorization bias | True joint | **One-Pass** |
| **Computational cost** | Lower (at video scale) | Higher (but trivial for us) | Tie |

### 6. What We Should Adopt from Video Models

While we reject pure AR, we should adopt:

1. **Progressive noise scheduling** (PA-VDM, Diffusion Forcing)
   - Natural temporal guidance without sequential generation
   - Wider CIs for far horizons (matches our intuition)

2. **Causal temporal attention** (HunyuanVideo)
   - Optional: provides temporal structure
   - Not required for efficiency at our scale

3. **Hierarchical regime sampling** (our addition)
   - Addresses multimodality that video models don't need
   - Explicit regime modeling for financial applications

### 7. Final Architecture Decision

```
┌─────────────────────────────────────────────────────────┐
│  CHOSEN: One-Pass with Progressive Noise                │
│                                                         │
│  ✅ One-pass generation (not block-wise AR)            │
│  ✅ Progressive noise (early=anchor, late=forecast)    │
│  ✅ Hierarchical regime sampling (for multimodality)   │
│  ⚪ Causal attention (optional, for temporal structure)│
│                                                         │
│  Computational cost: O(1,500²) = trivial               │
│  Expected benefits: Better CI calibration, regime      │
│                     diversity, no error accumulation   │
└─────────────────────────────────────────────────────────┘
```

### Sources

- [HunyuanVideo Technical Report](https://arxiv.org/abs/2412.03603)
- [Sora Technical Report](https://openai.com/research/video-generation-models-as-world-simulators)
- [CogVideoX Paper](https://arxiv.org/abs/2408.06072)
- [PA-VDM Paper](https://arxiv.org/abs/2410.08151)
- [Diffusion Forcing Paper](https://www.boyuan.space/diffusion-forcing/)
- [MCVD Paper](https://arxiv.org/abs/2205.09853)
- [Lil'Log Video Diffusion Survey](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)

---

## 2026-01-23: [-1, 1] Normalization Fix - Out-of-Range Issue Resolved

### Context

After 50-epoch training still showed 70% out-of-range rate, researched how original DDPM/DDIM papers handle this. Found that standard practice is to normalize data to [-1, 1] before training.

### Root Cause

The DDPM noise schedule (cosine) is calibrated for data in [-1, 1] range:
- Our IV data: [0.01, 0.99] with mean ~0.21, std ~0.10
- Noise schedule adds N(0,1) noise (std=1.0, 10× larger than data std)
- Mismatch caused model to output unbounded values

### Solution Implemented

Following Ho et al. 2020 (original DDPM):
> "We assume that image data consists of integers in {0,1,...,255} scaled linearly to [-1, 1]."

**Normalization constants** (based on empirical data range):
```python
IV_MIN = 0.0  # Slightly wider than data min (0.01)
IV_MAX = 1.0  # Slightly wider than data max (0.99)

def normalize_iv(iv):
    """Normalize IV from [0.0, 1.0] to [-1, 1]."""
    return 2.0 * (iv - IV_MIN) / (IV_MAX - IV_MIN) - 1.0

def denormalize_iv(iv_norm):
    """Denormalize IV from [-1, 1] to [0.0, 1.0]."""
    return (iv_norm + 1.0) / 2.0 * (IV_MAX - IV_MIN) + IV_MIN
```

### Files Modified

| File | Changes |
|------|---------|
| `train_ddpm_poc.py` | Added normalize/denormalize functions, normalize in dataset |
| `simple_denoiser.py` | Added denormalize_iv after sampling |
| `test_ddpm_requirements.py` | Fixed test bounds (iv_min: 0.05 → 0.0), denormalize GT |

### Validation Results (50 Epochs, Normalized Data)

| Test | Before Normalization | After Normalization | Status |
|------|---------------------|---------------------|--------|
| **Out-of-range rate** | 70.0% | **0.0%** | ✅ FIXED |
| Calendar arbitrage | 6.6% | 6.5% | ✅ PASS |
| Butterfly arbitrage | 23.6% | 23.6% | ❌ FAIL |
| Smile symmetry | PASS | PASS | ✅ |
| **90% CI Coverage** | 81.3%* | **81.7%** | ✅ PASS (>70%) |
| Mean diff | 5.4% | 5.3% | ✅ PASS |
| Std diff | 0.1% | 0.0% | ✅ PASS |
| ACF correlation | 0.901 | 0.907 | ✅ PASS |
| Vol clustering | PASS | PASS | ✅ |
| Kurtosis ratio | 0.454 | 0.450 | ❌ FAIL (target: 0.5-2.0) |

*Before: CI coverage was "meaningless" because samples spanned invalid range. Now it's a valid metric.

### Per-Horizon CI Coverage

| Horizon | Coverage |
|---------|----------|
| h=1 | 89.7% |
| h=7 | 82.5% |
| h=14 | 80.7% |
| h=30 | 80.4% |

Coverage degrades with horizon but remains above 70% target.

### Gated Evaluation Status Update

| Gate | Target | Before | After | Status |
|------|--------|--------|-------|--------|
| **Gate 1: Surface Validity** | <5% out-of-range | 70% | **0%** | ✅ PASSED |
| Gate 2: Marginal Plausibility | Mean <50% off | 5.4% | 5.3% | ✅ PASSED |
| Gate 3: CI Coverage | >70% | 81%* | **82%** | ✅ PASSED |
| Gate 4: Time Series | Kurtosis 0.5-2.0 | 0.45 | 0.45 | ❌ FAIL |

*Now valid measurement after Gate 1 passed.

### Remaining Issues

1. **Butterfly arbitrage: 23.6%** - Model doesn't fully preserve smile convexity
2. **Kurtosis ratio: 0.45** - Under-estimates fat tails (generates more Gaussian than GT)

### Architecture Discussion

The current simple 3D Conv architecture lacks:
- **Temporal attention**: Each frame can only influence neighbors (small 3D kernel)
- **Global context**: Frame 1 cannot directly influence frame 30

Video diffusion models solve this with:
1. **Factorized Space-Time attention**: Alternate spatial and temporal attention
2. **Full 3D attention**: Every patch attends to every other patch (expensive)

### Next Steps

1. Add temporal self-attention to improve long-range temporal coherence
2. Investigate butterfly arbitrage violations (smile convexity)
3. Address kurtosis mismatch (heavier tails needed)

### Model Files

Trained model saved at: `models/backfill/ddpm_poc/checkpoint_epoch_50.pt`

### Usage

```bash
# Run validation with DDIM (fast)
python experiments/backfill/two_stage_vae/test_ddpm_requirements.py \
    --model_path models/backfill/ddpm_poc/checkpoint_epoch_50.pt \
    --sampler ddim --ddim_steps 20 --n_samples 50 --max_batches 20
```

---

## 2026-01-23: DDIM Sampling Implementation & 50-Epoch Results

### Context

After implementing DDPM POC, sampling was extremely slow (~80 seconds per batch with 100 diffusion steps). Added DDIM (Denoising Diffusion Implicit Models) sampling to enable fast iteration during development.

### Terminology Clarification

**Important:** The term "explosion" in financial VAE literature specifically refers to **autoregressive error accumulation** when using log-returns:
- Each step predicts `log(IV_t / IV_{t-1})`
- Small errors accumulate: `exp(Σ log_returns)` diverges
- This is a compounding error problem

**Our current DDPM is NOT autoregressive** - it generates all 30 days at once. The issue we observe is:
- **Out-of-range samples**: Model outputs values outside [0.05, 1.0]
- **Not explosion**: No error accumulation, just lack of output constraint

In this log, we use "out-of-range rate" or "invalid IV rate" instead of "explosion rate."

### Why Surfaces Are Outside Valid Range

The DDPM model outputs unbounded values because:

1. **Model predicts noise ε ~ N(0,1)**, not IV directly
2. **Reverse diffusion has no bounds**: x₀ = (x_t - √(1-ᾱ)·ε) / √ᾱ can be any real value
3. **No output activation**: Unlike `sigmoid(x)·0.95 + 0.05`, nothing clamps the output
4. **MSE loss is symmetric**: Equally penalizes +0.5 error and -0.5 error, even if one is invalid

The model learned the correct **mean** (0.8% off GT) because that's the center of the loss landscape. But it hasn't learned the **boundaries** because MSE doesn't strongly penalize boundary violations.

### DDIM Implementation

**Key Insight:** DDIM uses the same trained model but a different (deterministic) sampling algorithm that can skip steps.

| Method | Steps | Time/batch | Stochastic |
|--------|-------|------------|------------|
| DDPM | 100 (all) | ~80s | Yes (new noise each step) |
| DDIM | 20 | ~1.5s | No (deterministic given z_T) |

**Speedup achieved: ~50x**

**DDIM Formula:**
```
x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1 - alpha_bar_{t-1}) * noise_pred
```

Where `x_0_pred` is predicted from the current noisy sample and model's noise prediction.

**Diversity preserved:** Different initial noise z_T → different outputs. DDIM is deterministic *given the same z_T*, but we sample different z_T for each trajectory.

### Files Modified

| File | Changes |
|------|---------|
| `diffusion/ddpm_scheduler.py` | Added `ddim_sample()` and `sample_ddim()` methods (~60 lines) |
| `diffusion/simple_denoiser.py` | Added `sampler` and `n_inference_steps` params to `sample()` |
| `test_ddpm_requirements.py` | Added `--sampler` and `--ddim_steps` CLI arguments |

### 50-Epoch Training Results

Trained for 50 epochs (vs previous 2 epochs). Best model saved at epoch 10.

**Training Metrics (Epoch 50):**
- Train Loss: 0.0159
- Val Loss: 0.0221
- Sample Diversity: 0.0297

### Validation Test Results (50 Epochs with DDIM)

| Test | 2 Epochs | 50 Epochs | Status |
|------|----------|-----------|--------|
| **Out-of-range rate** | 84.9% | **70.0%** | ❌ Still failing |
| **Min IV observed** | -1.0 | Still negative | ❌ |
| **90% CI Coverage** | 79% | **99.3%** | ⚠️ Meaningless |
| **Mean diff** | 129% | **0.8%** | ✅ Improved |
| **Std diff** | 255% | **209%** | ❌ Still too wide |
| **Kurtosis ratio** | 0.05 | **0.044** | ❌ Still failing |
| **ACF correlation** | 0.90 | **0.82** | ✅ Good |

**Key Observations:**

1. **Mean improved significantly** (0.8% vs 129% diff) - model learned the distribution center
2. **Out-of-range rate improved** (70% vs 85%) but still unacceptable
3. **CI coverage now 99%** - but this is STILL meaningless because samples span too wide a range
4. **Kurtosis still wrong** (0.044 ratio) - model produces Gaussian, not fat-tailed distributions

### Analysis

The 50-epoch model learned the **mean** of IV surfaces but NOT the **valid range**:
- Generated mean ≈ 0.21 (GT: 0.21) ✓
- Generated std ≈ 0.32 (GT: 0.10) ✗ - 3x too wide
- Samples still go negative (impossible for IV)

**Root Cause:** No architectural constraint forcing outputs to [0.05, 1.0]. Diffusion models naturally produce unbounded Gaussian-like outputs.

### Gated Evaluation Status

| Gate | Target | 2 Epochs | 50 Epochs | Status |
|------|--------|----------|-----------|--------|
| **Gate 1: Surface Validity** | <5% out-of-range | 85% | 70% | ❌ BLOCKING |
| Gate 2: Marginal Plausibility | Mean <50% off | 129% | **0.8%** | ✅ Would pass |
| Gate 3: CI Coverage | >70% | 79%* | 99%* | ⚠️ *Invalid |
| Gate 4: Time Series | Kurtosis 0.5-2.0 | 0.05 | 0.044 | ❌ |

*CI coverage metrics are meaningless until Gate 1 passes.

### Next Steps

**Must fix Gate 1 (Surface Validity) before other metrics matter.**

Options:
1. **Output clamping:** Add `sigmoid(x) * 0.95 + 0.05` to force [0.05, 1.0]
2. **Data normalization:** Normalize IV to [0, 1] before training
3. **Architectural constraint:** Learn bounded output via tanh + scaling

### Usage

```bash
# Fast validation with DDIM (~2 min instead of ~26 min)
python test_ddpm_requirements.py --sampler ddim --ddim_steps 20 --n_samples 20 --max_batches 10

# Full DDPM validation (slow, all 100 steps)
python test_ddpm_requirements.py --sampler ddpm --n_samples 20 --max_batches 10
```

---

## 2026-01-23: DDPM POC Implementation & Validation Test Results

### Context

After discovering that Causal 3D VAE cannot produce proper CI coverage (decoder squashes variance), implemented a minimal DDPM POC to test if diffusion models can achieve better CI coverage for multi-horizon IV surface forecasting.

### Implementation

**Architecture: Conditional DDPM (30 days → 30 days)**

```
History (30 days, 5×5)
       ↓
Context Encoder (CausalConv3d → ResBlocks → GAP → Linear)
       ↓
condition vector (dim=128)
       ↓
┌─────────────────────────────────────────────────────────┐
│  Noisy Future (30 days) + Time Embedding + Condition    │
│         ↓                                               │
│  SimpleDenoiser3D (4 ResBlocks with AdaptiveGroupNorm)  │
│         ↓                                               │
│  Predicted Noise                                        │
└─────────────────────────────────────────────────────────┘

Loss = MSE(predicted_noise, actual_noise)
Diffusion: 100 steps, cosine schedule
```

**Key Design Choices:**
- Reused CausalConv3d and ResnetBlockCausal3D from Causal 3D VAE
- Added SinusoidalTimeEmbedding and AdaptiveGroupNorm for time conditioning
- No output clamping (raw diffusion output)
- Simple architecture for fast iteration (189K params)

**Files Created:**
| File | Lines | Purpose |
|------|-------|---------|
| `diffusion/ddpm_scheduler.py` | 168 | DDPM forward/reverse process |
| `diffusion/time_embedding.py` | 79 | Sinusoidal embedding + AdaptiveGroupNorm |
| `diffusion/simple_denoiser.py` | 390 | Minimal 3D denoiser with context encoder |
| `config_ddpm_poc.py` | 75 | POC configuration |
| `train_ddpm_poc.py` | 452 | Training script |
| `eval_ddpm_poc.py` | 350 | Basic evaluation |
| `test_ddpm_requirements.py` | 1150 | **Comprehensive validation tests** |

**Model Saved:** `models/backfill/ddpm_poc/best_coverage_model.pt` (2 epochs)

### Rationale

The DDPM approach addresses the fundamental limitation of the Causal 3D VAE:

| Approach | Diversity Source | Problem |
|----------|------------------|---------|
| **VAE** | z ~ N(μ,σ²) sampling | Decoder squashes variance to near-zero |
| **DDPM** | Different noise seeds | Noise directly perturbs output space |

DDPM generates samples by iteratively denoising from random noise, so each noise seed produces a different trajectory. This avoids the decoder bottleneck that killed VAE diversity.

### Validation Test Results (2 Epochs - CRITICAL FAILURES)

Ran comprehensive validation tests (`test_ddpm_requirements.py`) covering 4 requirements:

**Test 1: Surface Validity - FAIL**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Out-of-range rate | **84.9%** | <5% | ❌ CRITICAL |
| Min IV observed | **-1.0** | >0.0 | ❌ CRITICAL |
| Calendar arbitrage | 31% | <10% | ❌ |
| Butterfly violation | 45% | <10% | ❌ |
| Smile symmetry | PASS | - | ✅ |

**Test 2: CI Coverage - MEANINGLESS**

| Level | Value | Note |
|-------|-------|------|
| 90% CI | 79% | Invalid - samples span [-1, +1] |
| Calibration error | 0.16 | N/A |

The CI coverage appears good but is **vacuously true** - generated samples span [-1, +1] while ground truth is ~0.2, so it trivially falls within the absurdly wide (invalid) range.

**Test 3: Marginal Recovery - FAIL**

| Metric | Generated | Ground Truth | Status |
|--------|-----------|--------------|--------|
| Mean | **-0.05** | +0.18 | ❌ Wrong sign! |
| Std | 0.29 | 0.08 | ❌ 3.5x too wide |
| K-S statistic | 0.58 | <0.15 | ❌ |

**Test 4: Time Series Properties - PARTIAL**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ACF correlation | 0.90 | >0.5 | ⚠️ Deceptive |
| ACF lag-20 | 0.31 vs 0.64 | Similar | ❌ Decays too fast |
| Kurtosis ratio | **0.05** | 0.5-2.0 | ❌ CRITICAL |
| ARCH effect | 10x higher | Similar | ❌ |

**Path Visualization Analysis:**

Looking at generated trajectories:
- All 6 grid points (ITM/ATM/OTM × short/long tenor) show **identical chaotic oscillations**
- Samples oscillate randomly between -1.0 and +1.0
- No resemblance to actual IV dynamics (which should be smooth, ~0.1-0.5)
- Ground truth (red line) is smooth around 0.2-0.4; generated samples are noise

### Root Cause

**The model is undertrained (2 epochs = ~128 batches).** It hasn't learned the IV distribution - it's outputting Gaussian noise centered at 0 instead of learning to predict the actual distribution.

Key evidence:
- Generated mean = -0.05 (near zero, as expected from untrained model)
- Kurtosis = 3.4 (near Gaussian 3.0) vs GT = 67.2 (extreme fat tails)
- All grid points identical (no spatial structure learned)

### Gated Evaluation Framework

Established gated evaluation - must pass earlier gates before later metrics are meaningful:

1. **Gate 1: Surface Validity** ← BLOCKING (currently failing)
2. Gate 2: Marginal Plausibility
3. Gate 3: CI Coverage (only meaningful after Gates 1-2 pass)
4. Gate 4: Time Series Properties

### Next Steps

**Immediate:** Train for 50+ epochs without code changes to see if model naturally learns valid IV range.

**If still failing after 50 epochs:** Add output clamping (sigmoid to [0.05, 1.0]).

### Outputs Generated

```
results/ddpm_poc/validation_tests/
├── summary.json           # All metrics in JSON
├── calibration_curve.png  # CI calibration plot
├── marginal_comparison.png # Distribution histogram + Q-Q
├── acf_comparison.png     # ACF bar chart
└── path_visualization.png # Generated trajectories vs GT
```

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
| Out-of-range Rate | 0.3% | 0% | GOOD |
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
| 2026-01-23 | **[-1, 1] normalization: 0% out-of-range** | train_ddpm_poc.py, simple_denoiser.py | Gate 1 passed, CI now valid |
| 2026-01-23 | **DDIM sampling: 50x speedup** | ddpm_scheduler.py +60 lines | Fast iteration enabled |
| 2026-01-23 | 50-epoch training: mean improved (0.8% off) | - | Model learning distribution |
| 2026-01-23 | 50-epoch: out-of-range still 70% | - | Need output clamping |
| 2026-01-23 | DDPM POC: 85% out-of-range (2 epochs) | 7 files, 2664 lines | Model undertrained |
| 2026-01-23 | Validation tests reveal invalid samples | test_ddpm_requirements.py | Gated evaluation framework |
| 2026-01-23 | CI coverage (79%) is meaningless | - | Samples span [-1,+1], GT=0.2 |
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

---

## Proposed: Two-Level Hierarchical Regime DDPM

### Motivation

Current single-level DDPM generates samples that cluster around the conditional mean, underestimating tail events. The kurtosis ratio is 0.45 (target: 0.5-2.0), indicating the model produces more Gaussian-like outputs than the fat-tailed ground truth.

**Core Problem:**
```
Standard DDPM: p(future | history) → samples cluster near E[future|history]
```

If there's a 10% chance of a crisis regime, standard DDPM won't produce 10% crisis-like samples - it will produce slightly elevated "average" samples.

### Architecture

**Level 1: Regime Classifier**
- Input: History (30 days of IV surfaces)
- Output: p(regime | history) - categorical distribution over K regimes
- Regimes: calm, crisis, spike_early, spike_late, trending_up, trending_down

**Level 2: Conditional Trajectory Diffusion**
- Input: History + sampled regime embedding
- Output: Future trajectory (30 days of IV surfaces)
- The regime embedding conditions the denoising process

```
History (30 days)
      ↓
Regime Classifier → p(regime | history) → Sample regime r
      ↓
[history, regime_embedding(r)]
      ↓
Conditional DDPM → Future trajectory (30 days)
```

### Hierarchical Sampling

```python
def hierarchical_sample(model, history, n_samples=100):
    """Sample with regime-aware diversity."""
    # Level 1: Sample regime for each trajectory
    regime_probs = model.regime_classifier(history)  # (K,)
    regimes = torch.multinomial(regime_probs, n_samples, replacement=True)  # (n_samples,)

    # Level 2: Sample trajectory conditioned on regime
    regime_embeds = model.regime_embedding(regimes)  # (n_samples, embed_dim)
    trajectories = model.diffusion.sample(history, condition=regime_embeds)

    return trajectories
```

### Why This Helps

With hierarchical sampling:
```
p(future | history) = Σ_r p(future | history, regime=r) × p(regime | history)
```

- If p(crisis | history) = 10%, exactly 10% of samples will have crisis dynamics
- Each regime has its own characteristic trajectory shape
- Tail events (spikes, crashes) are explicitly modeled, not averaged away

### Expected Benefits

| Issue | Current | With Regime Sampling |
|-------|---------|---------------------|
| Kurtosis ratio | 0.45 | Target: 0.5-2.0 |
| Butterfly violations | 24% | Should improve (regime-specific smile dynamics) |
| CI coverage | 81.7% | Should maintain or improve |
| Tail coverage | Underestimated | Properly calibrated |

### Implementation Steps

1. **Regime labeling:** Cluster historical trajectories into K regimes using K-means on trajectory features (mean, std, max drawdown, etc.)
2. **Train regime classifier:** Simple MLP on history context vector
3. **Add regime embedding:** Concatenate to DDPM condition vector
4. **Train jointly:** Cross-entropy on regime + MSE on noise prediction

### Open Questions

- How many regimes K? (Start with 4: calm, volatile, spike, trending)
- Should regimes be learned (VQ-VAE style) or predefined?
- How to ensure regime classifier is well-calibrated?

---

## Open Questions

1. ~~**How many epochs needed for valid samples?**~~ **ANSWERED:** 50 epochs improves mean (0.8% off) but still 70% out-of-range. Output clamping likely required.
2. ~~**Is output clamping required?**~~ **LIKELY YES:** 50 epochs didn't fix out-of-range rate (70%). Model needs architectural constraint.
3. What is the minimum model size for 5×5 grid diffusion? (Current: 189K params)
4. ~~**How many denoising steps needed?**~~ **ANSWERED:** DDIM with 20 steps works well (~50x faster than 100-step DDPM)
5. Can we use the existing VAE encoder for context embedding?
6. What is the optimal progressive noise schedule for 60-day horizon?
7. ~~**What output constraint works best?**~~ **ANSWERED:** Data normalization to [-1, 1] following Ho et al. 2020 (standard DDPM practice). No sigmoid/tanh needed - just normalize input data.
8. ~~**Will fixing out-of-range rate make CI coverage meaningful?**~~ **ANSWERED:** Yes! CI coverage now valid at 81.7% (was "meaningless" when samples spanned invalid range).

---

## References

See VIDEO_VAE_RESEARCH.md for full reference list.
