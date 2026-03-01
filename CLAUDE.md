# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Standing Directives (ALWAYS ACTIVE)

1. **HEDA Cycle**: Follow the Hypothesize → Experiment → Document → Analyze loop.
   After each experiment, IMMEDIATELY document results in RESEARCH_LOG.md before moving on.
2. **Don't stop** until ALL 8 test suites PASS on the raw model (no conformal), or you have
   exhausted all reasonable hypotheses. Check the **todo list** for current hypotheses and
   **MEMORY.md** for accumulated findings. After context compaction, recover state from these.
3. **Bitter Lesson**: Prefer approaches that scale with compute. No hand-tuned post-hoc fixes.
   No assumptions about the data — everything must be LEARNED from data. No precomputed
   per-cell/per-tenor constants, no data-derived lookup tables, no domain-specific heuristics.
   The method must generalize to any conditional scenario generation problem (IV, rates, FX, etc.).

## Project Overview

**Branch: `diffusion-poc-v1`** - DDPM-based approach for multi-horizon IV surface forecasting.

This codebase implements generative models for volatility surface forecasting. The **diffusion approach** addresses fundamental VAE limitations in sample diversity, achieving 81.7% CI coverage vs 33% VAE baseline.

For research rationale and architecture decisions, see `RESEARCH_LOG.md`.

## Repository Structure

```
vol-surface-vae-pub/
├── diffusion/              # Core DDPM implementation (PRIMARY)
│   ├── ddpm_scheduler.py   # Forward/reverse diffusion, DDPM & DDIM sampling
│   ├── simple_denoiser.py  # 3D denoiser, ConditionalDDPM wrapper
│   └── time_embedding.py   # Sinusoidal encoding, AdaptiveGroupNorm (FiLM)
├── experiments/backfill/diffusion_poc/  # DDPM experiments
│   ├── train_ddpm_poc.py   # Training script
│   ├── test_ddpm_requirements.py  # Full validation (4 test suites)
│   ├── test_progressive_sampling.py  # CI calibration tests
│   ├── config_ddpm_poc.py  # Configuration
│   └── metrics/            # FSD (Fréchet Surface Distance) metric
├── vae/                    # VAE baseline (for comparison)
├── data/                   # Input data files
├── models/backfill/ddpm_poc/  # DDPM checkpoints
└── results/ddpm_poc/       # Generated results
```

## Development Environment

- Uses `uv` for Python (>=3.13): `uv sync`
- Key packages: PyTorch, NumPy, scipy, matplotlib
- **Run all scripts from repository root**

## Diffusion Module (diffusion/)

### Architecture

**ConditionalDDPM** generates 30-day IV surface sequences conditioned on 30-day history:

```
History (30×5×5) → HistoryEncoder → condition (128-dim)
                                         ↓
Noise (30×5×5) + Time Embedding → SimpleDenoiser3D → Predicted Noise
                                         ↓
                              Reverse Diffusion (DDIM 20 steps)
                                         ↓
                              Future Surfaces (30×5×5)
```

**Key Components:**
- `DDPMScheduler`: Cosine noise schedule, supports DDPM (all steps) and DDIM (accelerated)
- `SimpleDenoiser3D`: 4 ResBlocks with AdaptiveGroupNorm for time/condition injection
- `HistoryEncoder`: Reuses `CausalConv3d` from VAE for temporal encoding

**Data Normalization:** IV surfaces normalized to [-1, 1] following Ho et al. 2020.

### Common Commands

```bash
# Train DDPM (50 epochs, ~2 hours on GPU)
python experiments/backfill/diffusion_poc/train_ddpm_poc.py --epochs 50

# Quick training test
python experiments/backfill/diffusion_poc/train_ddpm_poc.py --fast

# Full validation (surface validity, CI coverage, marginals, time series)
python experiments/backfill/diffusion_poc/test_ddpm_requirements.py \
    --model_path models/backfill/ddpm_poc/checkpoint_epoch_50.pt \
    --sampler ddim --ddim_steps 20 --max_batches 20

# Progressive sampling with FSD metric
python experiments/backfill/diffusion_poc/test_progressive_sampling.py \
    --max_batches 15 --n_samples 50 --compute_fsd
```

### Key Results

| Metric | VAE Baseline | DDPM POC |
|--------|--------------|----------|
| 90% CI Coverage | 33% | **81.7%** |
| Out-of-range rate | N/A | 0% |
| Kurtosis ratio | N/A | 0.45 (target: 0.5-2.0) |

**Progressive Noise Results (h=30):**
| Method | CI Coverage | CI Width Ratio |
|--------|-------------|----------------|
| Uniform DDPM | 86.5% | 1.01 |
| Post-hoc noise | **95.5%** | 1.19 |

**FSD (Fréchet Surface Distance):** Measures distributional realism. Lower = better.
- FSD-Encoder: 4.338 (uniform) vs 4.377 (progressive) - nearly identical
- Post-hoc noise improves CI calibration without hurting realism

### Validation Test Suites

`test_ddpm_requirements.py` runs 4 test suites:

1. **Surface Validity**: Explosion rate, calendar/butterfly arbitrage, smile symmetry
2. **CI Coverage**: Per-horizon (h=1,7,14,30), calibration curve
3. **Marginal Recovery**: K-S test, mean/std comparison
4. **Time Series**: ACF preservation, vol clustering, kurtosis matching

### Why DDPM Beats VAE

VAE decoder learns `μ_θ(z,x) ≈ E[y|z,x]` (conditional mean), squashing variance. DDPM samples directly in output space - different noise seeds → genuinely different trajectories.

**Remaining Issues:**
- Butterfly arbitrage: 24% (target: <5%)
- Kurtosis ratio: 0.45 (target: 0.5-2.0)

**Next Steps:** Hierarchical regime sampling, Diffusion Forcing training

## Data Format

**Input:** `data/vol_surface_with_ret.npz`
- `surface`: (N, 5, 5) - 5×5 IV grids (moneyness × tenor)
- `ret`: (N,) - Daily returns

**DDPM Training Data:**
- History: (B, 30, 5, 5) - 30 days context
- Future: (B, 30, 5, 5) - 30 days to predict

## Loading Models

```python
import torch
from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig

checkpoint = torch.load("models/backfill/ddpm_poc/checkpoint_epoch_50.pt", weights_only=False)
model = ConditionalDDPM(DenoiserConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate samples
samples = model.sample(history, n_samples=50, sampler="ddim", n_inference_steps=20)
# samples: (B, n_samples, 30, 5, 5)
```

## Import Structure

```python
from diffusion.simple_denoiser import ConditionalDDPM, DenoiserConfig, denormalize_iv
from diffusion.ddpm_scheduler import DDPMScheduler
from experiments.backfill.diffusion_poc.config_ddpm_poc import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from experiments.backfill.diffusion_poc.metrics import compute_fsd, extract_domain_features
```

## VAE Baseline (Reference)

The `vae/` module contains the VAE baseline for comparison. Key model: `CVAEMemRand`.

VAE achieves only 33% CI coverage due to decoder variance squashing. See `vae/README.md` for details.

## Path Conventions

- DDPM checkpoints: `models/backfill/ddpm_poc/checkpoint_epoch_*.pt`
- DDPM results: `results/ddpm_poc/`
- Data: `data/vol_surface_with_ret.npz`
