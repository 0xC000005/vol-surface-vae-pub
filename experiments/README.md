# Experiments Directory

This directory contains experiment-specific scripts organized by research question or analysis type.

## Directory Structure

### Backfill Experiments (`backfill/`)

Studies on multi-horizon volatility surface backfilling using VAE models.

**Documentation:**
- [Model Variants Guide](backfill/MODEL_VARIANTS.md) - Detailed comparison of 3 model variants (no_ex, ex_no_loss, ex_loss)
- [Quantile Regression Guide](backfill/QUANTILE_REGRESSION.md) - Quantile decoder methodology and performance

**Subdirectories:**
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
  - Bootstrap sampling with AR(1) recursion (backward & forward)
  - Cross-sectional testing: Backward recursion for endpoint predictions
  - In-sequence testing: Forward autoregressive 30-day sequences
  - Comparison scripts vs VAE and bootstrap models

- **`bootstrap_baseline/`** - Non-parametric bootstrap baseline
  - Bootstrap sampling from historical residuals
  - Autoregressive 30-day sequence generation
  - In-sequence co-integration testing
  - RMSE and CI calibration comparisons

- **`oracle_vs_prior/`** - VAE prior mismatch analysis
  - Oracle (posterior sampling) vs Prior (z ~ N(0,1))
  - Performance evaluation scripts

### Co-integration Testing Methodology

**Important methodological note:** Co-integration tests across different baselines use different experimental designs:

| Method | Test Type | What It Measures |
|--------|-----------|------------------|
| **VAE** | Cross-sectional only | Tests predictions at specific horizons (H=1,7,14,30) across many dates. Each prediction is independent. Tests whether the model preserves IV-EWMA relationship at different forecast distances. |
| **Econometric** | Both cross-sectional & in-sequence | Cross-sectional: Backward recursion for endpoint predictions. In-sequence: Forward autoregressive 30-day sequences for trajectory testing. |
| **Bootstrap** | In-sequence only | Tests whether all 30 days within each generated autoregressive sequence maintain co-integration. Tests trajectory coherence over time. |

**Key distinctions:**
- **Cross-sectional** (VAE, Econometric endpoints): N independent dates × 1 prediction per horizon → Tests endpoint accuracy
- **In-sequence** (Bootstrap, Econometric forward AR): N sequences × 30 days each → Tests autoregressive stability

Both use ADF tests but measure fundamentally different properties. Cross-sectional and in-sequence results are **not directly comparable** - cross-sectional tests measure model accuracy at specific horizons, while in-sequence tests measure whether the autoregressive generation process maintains economic relationships throughout the entire trajectory.

**Example interpretation:**
- VAE cross-sectional: "84% of grid points co-integrated at H30" = Model predictions 30 days ahead preserve relationship across many dates
- Econometric in-sequence: "75% pass rate" = 75% of generated 30-day autoregressive sequences maintain relationship throughout all intermediate days
- Bootstrap in-sequence: "70% pass rate" = 70% of generated 30-day sequences maintain relationship throughout all intermediate days

**Econometric baseline supports both modes:**
- **Backward recursion (Eq. 10)**: Uses future v(t+1), suitable for cross-sectional endpoint predictions
- **Forward recursion (Eq. 9)**: Uses past v(t-1), suitable for in-sequence autoregressive testing

This dual capability enables direct comparison between econometric and bootstrap in-sequence results to test the hypothesis that parametric structure (explicit IV-EWMA enforcement) outperforms non-parametric sampling by 5-10%.

### Additional Analyses

- **`cointegration/`** - Co-integration preservation tests (cross-sectional)
  - Tests whether VAE preserves economic relationships
  - Multi-horizon co-integration analysis (H=1,7,14,30)
  - Cross-sectional testing: predictions at specific horizons across many dates

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
