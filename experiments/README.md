# Experiments Directory

This directory contains experiment-specific scripts organized by research question or analysis type.

## Directory Structure

### Backfill Experiments (`backfill/`)

Studies on multi-horizon volatility surface backfilling using VAE models.

- **`context20/`** - Main 16-year backfill model (production)
  - Training: `train_backfill_model.py`
  - Config: `config/backfill_config.py`
  - Testing: `test_insample_reconstruction_16yr.py`, `test_oos_reconstruction_16yr.py`
  - Evaluation: `evaluate_*.py` scripts
  - Analysis: `analyze_*.py` scripts
  - Visualization: `visualize_*.py` scripts

- **`context60/`** - Context length ablation study
  - Extended context window (60 days vs 20 days)
  - Multi-phase training experiments
  - Config: `config/backfill_context60_config.py`

- **`horizon5/`** - Multi-horizon validation
  - Horizon=5 model validation
  - Comparison vs autoregressive baseline
  - Improvement analysis

### Baselines & Comparisons

- **`econometric_baseline/`** - Co-integration baseline implementation ([Full Methodology](econometric_baseline/ECONOMETRIC_METHODOLOGY.md))
  - EWMA realized volatility model
  - Bootstrap sampling with AR(1) backward recursion
  - Comparison scripts vs VAE models

- **`bootstrap_baseline/`** - Non-parametric bootstrap baseline
  - Bootstrap sampling from historical residuals
  - Autoregressive 30-day sequence generation
  - In-sequence co-integration testing
  - RMSE and CI calibration comparisons

- **`oracle_vs_prior/`** - VAE prior mismatch analysis
  - Oracle (posterior sampling) vs Prior (z ~ N(0,1))
  - Performance evaluation scripts

### Additional Analyses

- **`cointegration/`** - Co-integration preservation tests
  - Tests whether VAE preserves economic relationships
  - Multi-horizon co-integration analysis

- **`vol_smile/`** - Volatility smile preservation
  - Analyzes shape preservation across moneyness
  - Training smile statistics

- **`diagnostics/`** - Model diagnostic scripts
  - Dimension ablation studies
  - Latent distribution analysis
  - Multi-offset training experiments

## Usage

All scripts should be run from the repository root directory:

```bash
# Example: Run context20 in-sample reconstruction
python experiments/backfill/context20/test_insample_reconstruction_16yr.py

# Example: Train context60 model
python experiments/backfill/context60/train_backfill_context60.py
```

## Results

Generated results are stored in the `results/` directory at the repository root.

## Models

Trained models are stored in the `models/` directory at the repository root.
