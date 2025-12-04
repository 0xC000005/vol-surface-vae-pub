# Volatility Surface VAE

**A comprehensive research platform for conditional generation of volatility surfaces using Variational Autoencoders with quantile regression and multi-horizon training.**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Paper

Jacky Chen, John Hull, Zissis Poulos, Haris Rasul, Andreas Veneris, Yuntao Wu
**"A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces"**
*Journal of Financial Data Science*, 2025

This repository implements the paper's methodology and extends it with quantile regression for improved uncertainty quantification, multi-horizon training infrastructure, and comprehensive baseline comparisons.

---

## Key Features

### Core Capabilities

**Quantile Regression Decoder** - Direct confidence interval prediction with pinball loss
- Outputs 3 quantiles [p05, p50, p95] in a single forward pass
- ~1000× faster than Monte Carlo sampling (1 pass vs 1000 samples)
- CI violations: 34-45% (improvement from baseline 50%+)

**Three Model Variants** - Tests whether multi-task learning improves forecasting
- **no_ex**: Surface-only baseline
- **ex_no_loss**: Surface + features (passive conditioning)
- **ex_loss**: Surface + features (joint optimization)

**Multi-Horizon Training** - Simultaneous training on horizons [1, 7, 14, 30] days
- Prevents autoregressive error accumulation
- RMSE reduction: 43-54% vs sequential forecasting
- CI calibration improvement: 80% reduction in violations

**Oracle vs Prior Sampling** - Two inference strategies for uncertainty quantification
- **Oracle**: Posterior sampling with future knowledge (upper bound performance)
- **Prior**: Context-only sampling (realistic deployment scenario)
- Reveals VAE prior mismatch: Prior CIs 21-42% wider than oracle

**Production Models** - Battle-tested implementations
- **Context20**: 16-year training (2004-2019), 34 analysis scripts
- **Context60**: Context length ablation study with 4-phase curriculum training

**Baselines** - Multiple comparison methods
- Bootstrap baseline (non-parametric, 10 scripts)
- Oracle vs Prior analysis (4 scripts)
- Additional comparison methods available

---

## Quick Start

### Installation

**Requirements:**
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

```bash
# Install dependencies
uv sync
```

### Download Pre-trained Models & Data

Download from [Google Drive](https://drive.google.com/drive/folders/1W3KsAJ0YQzK2qnk0c-OIgj26oCAO3NI1?usp=sharing):
- Pre-trained model checkpoints (models/)
- Preprocessed data (`data/vol_surface_with_ret.npz`)

Place in respective directories (`models/`, `data/`).

### Basic Usage

**Train quantile regression models:**
```bash
python train_quantile_models.py
```

**Generate forecasts:**
```bash
# Fast quantile forecasts (single pass)
python generate_quantile_surfaces.py

# Stochastic forecasts (Monte Carlo sampling)
python generate_surfaces.py

# Deterministic maximum likelihood
python generate_surfaces_max_likelihood.py
```

**Run analysis pipeline:**
```bash
python main_analysis.py
```

**Interactive visualization:**
```bash
streamlit run streamlit_vol_surface_viewer.py
```

For detailed setup and examples, see [`DEVELOPMENT.md`](DEVELOPMENT.md).

---

## Repository Structure

```
vol-surface-vae-pub/
├── vae/                     # Core VAE implementations
│   ├── base.py             # Base classes
│   ├── cvae_with_mem_randomized.py  # Primary model (CVAEMemRand)
│   ├── datasets_randomized.py       # Variable context datasets
│   └── utils.py            # Training utilities
├── analysis_code/           # 29 reusable analysis modules
│   ├── regression.py       # Latent space regression
│   ├── latent_pca.py       # PCA analysis
│   ├── diagnose_distribution_shape.py
│   ├── visualize_*.py      # Plotly dashboards
│   └── ...
├── experiments/             # Organized by research question
│   ├── backfill/
│   │   ├── context20/      # Production model (34 scripts)
│   │   ├── context60/      # Context ablation (15+ scripts)
│   │   └── horizon5/       # Multi-horizon validation (4 scripts)
│   ├── bootstrap_baseline/  # Non-parametric baseline (10 scripts)
│   ├── oracle_vs_prior/    # Sampling strategy comparison (4 scripts)
│   ├── cointegration/      # Co-integration tests (2 scripts)
│   └── diagnostics/        # Model diagnostics
├── config/                  # Model configurations
│   ├── backfill_config.py
│   └── backfill_context60_config.py
├── data/                    # Input data (not in git)
│   └── vol_surface_with_ret.npz
├── models/                  # Trained checkpoints (not in git)
│   └── backfill/
│       ├── context20_production/
│       └── context60_experiment/
├── results/                 # Generated outputs (not in git)
├── data_preproc/           # Data preprocessing notebooks
├── eval_scripts/           # Evaluation utilities
└── [training scripts]      # Core scripts in root
```

**Important:** Run all scripts from repository root:
```bash
python experiments/backfill/context20/train_backfill_model.py
```

---

## Architecture

### Model Hierarchy

1. **VAEConv2D** - Basic 2D convolutional VAE
2. **CVAE** - Conditional VAE with context encoder
3. **CVAEMem** - CVAE + LSTM/GRU/RNN memory
4. **CVAEMemRand** - Primary model (variable context, multi-horizon, quantile regression)

### Key Features

- **Variable context lengths** - Randomized during training (1-20 days)
- **Multi-horizon prediction** - Configurable horizon (1-30 days)
- **Quantile regression** - Decoder outputs [p05, p50, p95]
- **Optional extra features** - Returns, volatility skew, term structure slope

### Three Model Variants

| Variant | Features | Loss on Features | Research Question |
|---------|----------|------------------|-------------------|
| **no_ex** | Surface only | 0.0 | Baseline performance |
| **ex_no_loss** | Surface + [ret, skew, slope] | 0.0 (passive) | Does conditioning help? |
| **ex_loss** | Surface + [ret, skew, slope] | 1.0 (joint) | Does multi-task learning help? |

### Input/Output Format

**Input:**
```python
{
    "surface": (B, T, 5, 5),   # B batches, T timesteps, 5×5 IV grids
    "ex_feats": (B, T, 3)       # [returns, skew, slope]
}
```

**Output (Quantile Models):**
```python
(B, T, 3, 5, 5)  # 3 quantiles [p05, p50, p95]
```

For detailed architecture, see [`vae/README.md`](vae/README.md).

---

## Main Components

### Training

**Quantile regression:**
```bash
python train_quantile_models.py  # Trains all 3 variants
```

**Hyperparameter search:**
```bash
python param_search.py
```

**Production model (Context20):**
```bash
python experiments/backfill/context20/train_backfill_model.py
```

### Generation

**Quantile forecasts (fast):**
```bash
python generate_quantile_surfaces.py
# Output: (num_days, 1, 3, 5, 5)
```

**Stochastic forecasts:**
```bash
python generate_surfaces.py
# Output: (num_days, 1000, 5, 5) - 1000 samples per day
```

**Maximum likelihood:**
```bash
python generate_surfaces_max_likelihood.py
# Deterministic forecast with z=0
```

### Analysis

**Full pipeline:**
```bash
python main_analysis.py
```

**CI calibration:**
```bash
python compute_grid_ci_stats.py
```
---

## Experiments & Baselines

### Context20 Production Model

**Configuration:**
- Training period: 2004-2019 (16 years)
- Context length: 20 days
- Horizons: [1, 7, 14, 30] days
- 34 analysis scripts

**Training:**
```bash
python experiments/backfill/context20/train_backfill_model.py
```

**Documentation:** See [`experiments/backfill/context20/README.md`](experiments/backfill/context20/README.md)

### Context60 Ablation Study

Tests impact of longer context on forecasting performance.

**Training approach:**
- 4-phase curriculum training
- Multi-offset autoregressive
- 15+ analysis scripts

**Documentation:** See [`experiments/backfill/context60/README.md`](experiments/backfill/context60/README.md)

### Baselines

**Bootstrap Baseline:**
- Non-parametric sampling from historical residuals
- 10 scripts including in-sequence cointegration tests
- See [`experiments/bootstrap_baseline/`](experiments/bootstrap_baseline/)

**Oracle vs Prior Analysis:**
- Quantifies VAE prior mismatch
- 4 comparison and visualization scripts
- See [`experiments/oracle_vs_prior/`](experiments/oracle_vs_prior/)

**Additional comparison methods available** for comprehensive evaluation.

---

## Key Results

### Quantile Regression Performance

**Test Set CI Calibration:**

| Model | CI Violations | Target |
|-------|---------------|--------|
| **no_ex** | 44.50% | ~10% |
| **ex_no_loss** | 35.43% | ~10% |
| **ex_loss** | 34.28% ✓ | ~10% |

- Best model: **ex_loss** (34.28% violations)
- Improvement from baseline: 16% reduction (50%+ → 34%)
- Target: ~10% (well-calibrated) - work in progress
- Inference speedup: ~1000× faster than Monte Carlo

### Multi-Horizon Training (H=5)

**vs Baseline Autoregressive:**
- RMSE reduction: 43-54%
- CI violations: 80% reduction (89% → 18%)
- Training time: 3.7× longer per epoch (acceptable tradeoff)

### Oracle vs Prior Gap

**CI Width Comparison (2004-2007 calm period):**

| Horizon | Avg Width Ratio | Max Width Ratio |
|---------|-----------------|-----------------|
| **H=60** | 1.032× | 1.081× |
| **H=90** | 1.049× | 1.087× |

- Prior CIs are 21-42% wider than oracle (realistic uncertainty)
- Gap grows from ~1.0× at day 1 to ~1.15× at day 60-90
- Demonstrates VAE prior mismatch: p(z|context) ≠ p(z|context,target)

For detailed results, see [`experiments/backfill/QUANTILE_REGRESSION.md`](experiments/backfill/QUANTILE_REGRESSION.md).

---

## Documentation

### Core Guides

- **[`CLAUDE.md`](CLAUDE.md)** (425 lines) - Comprehensive project guide for developers
- **[`DEVELOPMENT.md`](DEVELOPMENT.md)** - Code examples and development patterns
- **[`experiments/README.md`](experiments/README.md)** - Experiment directory overview

### Method Documentation

- **[`experiments/backfill/QUANTILE_REGRESSION.md`](experiments/backfill/QUANTILE_REGRESSION.md)** - Pinball loss, architecture, performance
- **[`experiments/backfill/MODEL_VARIANTS.md`](experiments/backfill/MODEL_VARIANTS.md)** - 3-variant comparison with test results
- **[`experiments/backfill/context20/CI_WIDTH_ANALYSIS.md`](experiments/backfill/context20/CI_WIDTH_ANALYSIS.md)** - Analysis suite documentation

### Training Plans

- **[`BACKFILL_MVP_PLAN.md`](BACKFILL_MVP_PLAN.md)** - Multi-horizon backfilling methodology
- **[`CONTEXT60_TRAINING_PLAN.md`](CONTEXT60_TRAINING_PLAN.md)** - Context ablation study plan
- **[`LATENT_SAMPLING_STRATEGIES.md`](LATENT_SAMPLING_STRATEGIES.md)** - Oracle vs Prior sampling

### Experiment-Specific READMEs

Each `experiments/` subdirectory contains detailed documentation:
- `experiments/backfill/context20/README.md` - Production model
- `experiments/backfill/context60/README.md` - Context ablation
- `experiments/bootstrap_baseline/README_insequence_cointegration.md` - Bootstrap methodology

---

## Data Preprocessing

### Data Sources

**SPX Option Prices:**
- Source: WRDS OptionMetrics Ivy DB US
- Period: 2000-01-01 to 2023-02-28
- SECID: 108105 (S&P 500 Index)

**SPX Index Prices:**
- Source: Yahoo Finance (ticker `^GSPC`)

### Preprocessing Pipeline

1. **Raw data cleaning** - `spx_volsurface_generation.ipynb`
   - Uses `data_preproc/data_preproc.py`
   - Cleans WRDS data and generates interpolated IVS

2. **Grid generation** - `spx_convert_to_grid.ipynb`
   - Converts to 5×5 numpy grids (moneyness × maturity)
   - Output: `data/vol_surface_with_ret.npz`

### WRDS Data Download Instructions

**Step 1:** Date Range: 2000-01-01 to 2023-02-28

**Step 2:** Filters
- SECID = 108105
- Option Type: Both
- Exercise Type: Both
- Security Type: Both

**Step 3:** Query Variables: all

**Step 4:** Output Format
- Format: *.csv
- Compression: *.zip
- Date Format: YYYY-MM-DD

### Pre-trained Models

Download from [Google Drive](https://drive.google.com/drive/folders/1W3KsAJ0YQzK2qnk0c-OIgj26oCAO3NI1?usp=sharing).

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{chen2025vae,
  title={A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces},
  author={Chen, Jacky and Hull, John and Poulos, Zissis and Rasul, Haris and Veneris, Andreas and Wu, Yuntao},
  journal={Journal of Financial Data Science},
  year={2025}
}
```

---

## License

[Specify license here - MIT, Apache 2.0, etc.]

---

## Contributing

This repository contains research code associated with the published paper. For questions or collaboration inquiries, please contact the authors through the affiliated institutions.

---

**Version:** 2.0 (Quantile Regression + Multi-Horizon Training)
**Last Updated:** December 2024
