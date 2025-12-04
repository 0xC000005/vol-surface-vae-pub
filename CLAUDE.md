# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This codebase implements a Variational Autoencoder (VAE) approach for conditional generation of future volatility surfaces, as described in the paper:

**Jacky Chen, John Hull, Zissis Poulos, Haris Rasul, Andreas Veneris, Yuntao Wu**, "*A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces*", Journal of Financial Data Science, 2025.

The system uses a Conditional VAE (CVAE) combined with LSTM to generate one-day-ahead volatility surface forecasts based on variable context lengths. The models use **teacher forcing** during inference - conditioning on real historical data rather than previous predictions - to produce independent forecasts for evaluation.

## Repository Structure

The codebase is organized into clear directories separating core functionality from experiments:

```
vol-surface-vae-pub/
├── vae/                    # Core VAE model implementations
├── analysis_code/          # Reusable analysis modules
├── config/                 # Configuration files (backfill configs)
├── data/                   # Input data files
├── experiments/            # Experiment-specific scripts (organized by research question)
│   ├── backfill/          # Multi-horizon backfilling experiments
│   │   ├── context20/     # Production model (16-year training, 34 scripts)
│   │   ├── context60/     # Context length ablation study (3 scripts)
│   │   └── horizon5/      # Multi-horizon validation (4 scripts)
│   ├── econometric_baseline/  # Co-integration baseline (9 scripts)
│   ├── oracle_vs_prior/   # VAE prior mismatch analysis (4 scripts)
│   ├── cointegration/     # Co-integration preservation tests (2 scripts)
│   ├── vol_smile/         # Volatility smile preservation (3 scripts)
│   └── diagnostics/       # Model diagnostics (1 script)
├── models/                 # Trained model checkpoints
│   ├── backfill/          # Backfill model variants
│   │   ├── context20_production/  # Main 16yr model
│   │   ├── context60_experiment/  # Context ablation checkpoints
│   │   └── archived/      # Old experiment models
│   └── 1d_backfilling/    # 1D backfilling experiments
├── results/                # All generated results and analyses (not in git)
│   ├── presentations/     # Main reports and documentation
│   ├── backfill_16yr/     # Context20 model results
│   ├── econometric_baseline/  # Baseline comparison results
│   └── ...                # Other analysis results
├── archived_experiments/   # Old test outputs and validation scripts
├── eval_scripts/          # Evaluation utilities
├── data_preproc/          # Data preprocessing code
├── test_code/             # Unit tests
├── test_spx/              # Test models directory (not in git, 31GB)
└── [core scripts]         # 10 core scripts in root
```

**Key Principles:**
- **Core scripts** (training, generation, analysis) stay in root
- **Experiment-specific scripts** go in `experiments/` subdirectories
- **Config files** stay in root `config/` directory for easy importing
- **Results** go in `results/` with clear organization (not committed to git)
- **Models** go in `models/` organized by experiment type (not committed to git)
- **Run all scripts from repository root:** `python experiments/backfill/context20/script.py`

## Development Environment

### Package Management
- Uses `uv` for Python dependency management (Python >=3.13)
- Dependencies: `pyproject.toml`, install with `uv sync`
- Key packages: PyTorch, NumPy, pandas, scikit-learn, matplotlib

### Common Commands

**Training:**
```bash
python param_search.py                                              # Train 3 model variants
python train_quantile_models.py                                     # Train quantile models
python experiments/backfill/context20/train_backfill_model.py       # Train backfill model
```

**Generation & Analysis:**
```bash
python generate_surfaces.py                  # Stochastic forecasts
python generate_quantile_surfaces.py         # Quantile forecasts
python main_analysis.py                      # Full analysis pipeline
python evaluate_quantile_ci_calibration.py   # CI calibration metrics
```

**Visualization:**
```bash
python analysis_code/visualize_teacher_forcing.py                # Model comparison
python analysis_code/visualize_backfill_16yr_plotly.py           # Interactive dashboards
```

For complete command reference, see `experiments/README.md` and subdirectory READMEs. Code examples in `DEVELOPMENT.md`.

## Architecture

### Core VAE Models (vae/)

**Model Hierarchy:**
1. `VAEConv2D`: Basic 2D convolutional VAE
2. `CVAE`: Conditional VAE with context encoder
3. `CVAEMem`: CVAE with LSTM/GRU/RNN memory
4. **`CVAEMemRand`**: Primary model (variable context lengths, multi-horizon prediction, optional extra features)

**Key features:**
- Variable context lengths (randomized during training)
- Configurable prediction horizon (1-30 days)
- Optional extra features (returns, skew, slope)
- Quantile regression support for uncertainty quantification

**Input format:** Dictionary with `"surface"` (B, T, 5, 5) and optional `"ex_feats"` (B, T, 3)

For detailed architecture documentation, configuration parameters, and model components, see `vae/README.md`.

## Data Structure

### Input Data Format
Volatility surfaces are stored as numpy arrays:
- `surface`: (N, 5, 5) - N days of 5x5 volatility grids (moneyness × time to maturity)
- `ret`: (N,) - Daily returns
- `skews`: (N,) - Volatility skew measure
- `slopes`: (N,) - Volatility term structure slope
- `ex_data`: (N, 3) - Concatenated [ret, skew, slope]

Data files:
- `data/vol_surface_with_ret.npz`: Main training data
- `data/spx_vol_surface_history_full_data_fixed.parquet`: Full SPX history

### Model Outputs
Generated surfaces stored in npz files:
- `surfaces`: (num_days, num_samples, 5, 5)
- `ex_feats`: (num_days, num_samples, 3) - if model returns extra features

## Analysis Pipeline (analysis_code/)

**Key Analysis Modules:**
- `regression.py`: Regression analysis of latent embeddings vs market indicators
- `latent_pca.py`: PCA analysis of latent space
- `arbitrage.py`: Check for arbitrage-free properties in generated surfaces
- `loss_table.py`: Format and generate loss comparison tables
- `diagnose_distribution_shape.py`: Distribution shape diagnostics (kurtosis/skewness)
- `analyze_shape_vs_horizon.py`: Shape mismatch degradation across horizons
- `create_shape_diagnostic_heatmaps.py`: 5×5 grid heatmaps of shape differences
- `visualize_tail_behavior.py`: Q-Q plots and tail probability analysis

**Main Analysis Workflow (main_analysis.py):**
1. Load trained models and generate surfaces
2. Compute regression analysis (surface grid accuracy, RMSE benchmarks)
3. Generate latent embeddings and perform PCA
4. Run classification tasks (NBER recession prediction)
5. Check arbitrage violations
6. Generate LaTeX tables and plots

## Model Variants

Three model variants test the hypothesis: **Can we improve forecasts by jointly modeling returns and surfaces, or is surface-only modeling sufficient?**

| Variant | Features | Loss on Features | Use Case |
|---------|----------|------------------|----------|
| **No EX** | Surface only | 0.0 | Baseline (surface dynamics alone) |
| **EX No Loss** | Surface + ret/skew/slope | 0.0 | Passive conditioning (features as context) |
| **EX Loss** | Surface + ret/skew/slope | 1.0 (returns only) | Joint optimization (multi-task learning) |

**Test Set Performance:**

| Model | Surface Loss | Feature Loss | KL Loss |
|-------|--------------|--------------|---------|
| **No EX** | 0.001722 | 0.000000 | 3.956 |
| **EX No Loss** | 0.002503 | 0.924496 (not optimized) | 3.711 |
| **EX Loss** | 0.001899 | 0.000161 | 4.000 |

**Key findings:**
- No EX: Lowest surface error (baseline)
- EX Loss: Balances both objectives, excellent return prediction

**Generation outputs:**
- Stochastic: `{model}_gen5.npz` (1000 samples/day)
- Deterministic: `{model}_mle_gen5.npz` (1 sample/day, z=0)

For detailed architecture, research questions, and analysis, see `experiments/backfill/MODEL_VARIANTS.md`.

## Quantile Regression Decoder

Quantile regression variant addresses CI calibration issues by directly predicting confidence intervals instead of Monte Carlo sampling.

**Architecture changes:**
- Decoder outputs 3 channels: p05, p50, p95 quantiles
- Pinball loss (asymmetric) instead of MSE
- ~1000× faster generation (1 forward pass vs 1000)

**CI Calibration Results:**

| Model | CI Violations | Best: ex_loss |
|-------|---------------|---------------|
| **Target** | 10% | (well-calibrated) |
| **Achieved** | 34-45% | 34.28% violations |
| **Baseline MSE** | ~50%+ | 16% improvement |

**Key findings:**
- Significant speedup and partial calibration improvement
- VAE prior mismatch: 3× gap between ground truth (7%) and context-only (19%) latents
- Asymmetric violations (underestimate upper tail uncertainty)

**Next steps:** Conformal prediction, loss reweighting, heteroscedastic quantile regression

For detailed methodology, pinball loss formula, verification results, and implementation, see `experiments/backfill/QUANTILE_REGRESSION.md`.

## VAE Sampling Strategies

The VAE teacher forcing generation supports two sampling strategies for latent variable z, controlled via `--sampling_mode` parameter:

**Oracle (Posterior) Sampling:**
- Uses `model.forward(context+target)` → z ~ q(z|context, target)
- Encoder sees full sequence including future target data
- Represents upper bound performance (not realistic deployment)
- Produces tighter confidence intervals (~2-3× narrower than prior)

**Prior (Realistic) Sampling:**
- Uses `model.get_surface_given_conditions(context)` with hybrid sampling
- z[:,:C] = posterior_mean (deterministic context encoding)
- z[:,C:] ~ N(0,1) (stochastic future, no target knowledge)
- Represents realistic deployment scenario
- Produces wider confidence intervals due to VAE prior mismatch

**File Organization:**
```
results/vae_baseline/
├── predictions/autoregressive/
│   ├── oracle/           # Posterior sampling results
│   │   └── vae_tf_*.npz
│   └── prior/            # Prior sampling results
│       └── vae_tf_*.npz
└── analysis/
    ├── oracle/           # Oracle CI statistics and reports
    └── prior/            # Prior CI statistics and reports
```

**Usage:**
```bash
# Oracle sampling (default)
python experiments/backfill/context20/generate_vae_tf_sequences.py --period crisis --sampling_mode oracle

# Prior sampling (realistic)
python experiments/backfill/context20/generate_vae_tf_sequences.py --period crisis --sampling_mode prior

# Run all periods for prior mode
bash experiments/backfill/context20/run_generate_all_tf_sequences.sh prior

# Compare oracle vs prior
python experiments/backfill/context20/compare_oracle_vs_prior_ci.py
```

**Key Findings:**
- Prior CIs are ~2-3× wider than oracle CIs on average
- All significant differences (p < 0.001) across periods and horizons
- Demonstrates VAE prior mismatch: p(z|context) ≠ p(z|context+target)
- Critical for understanding realistic model uncertainty in deployment

All analysis scripts support `--sampling_mode oracle/prior` parameter for separate analysis of each strategy.

## Multi-Horizon Training

Models support `horizon > 1` to predict multiple days simultaneously, avoiding autoregressive error accumulation.

**Configuration:** Set `"horizon": 5` in model_config to predict 5 days in one shot.

**Results (horizon=5 vs baseline):**
- RMSE: 43-54% reduction
- CI violations: 80% reduction (89% → 18%)
- Training time: 3.7× longer per epoch

For latent sampling strategies and implementation details, see `experiments/backfill/horizon5/README.md`.

## Important Implementation Details

### Loss Function

- **Standard VAE**: `loss = MSE(surface) + kl_weight * KL_divergence`
- **Quantile VAE**: `loss = pinball_loss(quantiles) + kl_weight * KL_divergence`
- **With extra features**: Add `re_feat_weight * L2(return)` when `ex_loss_on_ret_only=True`

### Teacher Forcing

**Training:** Variable context length (randomized), predicts 1-day-ahead
**Inference:** Always conditions on real historical data, not model predictions
**Result:** Independent one-step-ahead forecasts (not autoregressive)

**Generation modes:**
- **Stochastic**: Sample z ~ N(0,1), 1000 samples/day, captures uncertainty
- **Maximum Likelihood**: Set z = 0, deterministic point forecast

For code examples, see `DEVELOPMENT.md`.

## Autoregressive Backfilling

Generates 30-day autoregressive sequences for historical periods with limited data (e.g., 2008-2010 financial crisis).

**Key features:**
- Multi-horizon training: [1, 7, 14, 30] day horizons simultaneously
- 2-phase training: teacher forcing → multi-horizon
- Context length: 20 days (production model)

**Training:**
```bash
python experiments/backfill/context20/train_backfill_model.py
```
Output: `models/backfill/context20_production/backfill_16yr.pt`

**Results (backfill_16yr / context20):**
- In-sample CI violations: 18.1%
- Out-of-sample CI violations: 28.0% (+55% degradation)
- RMSE increase: 57-92% across horizons (OOS)

For complete evaluation, analysis scripts, and interactive visualizations, see `experiments/backfill/context20/README.md`.

## Econometric Baseline

Econometric baseline using IV-EWMA co-integration for comparison with VAE.

**Components:** EWMA realized volatility (λ=0.94), co-integration regression, weighted least squares, bootstrap sampling with AR(1) backward recursion.

**Performance:**
- Crisis (2008-2010): VAE wins 87% comparisons, 38% lower RMSE
- OOS (2019-2023): VAE better on extremes, econometric competitive on ATM
- CI calibration: Econometric 65-68% violations vs VAE 18-28%

For methodology, scripts, and detailed comparison, see `experiments/econometric_baseline/README.md`.

## CI Width Analysis & Regime Visualization

8-script suite for investigating why the VAE widens confidence intervals and demonstrating intelligent pattern recognition across market regimes. Includes systematic regression analysis, multi-regime overlay visualizations, and extreme moment analysis.

**Key findings:** Spatial features (slopes, skews) dominate CI width explanation (68-75%); model demonstrates pre-crisis detection by widening CIs based on surface shape anomalies, not just volatility levels.

See `experiments/backfill/context20/CI_WIDTH_ANALYSIS.md` for detailed documentation of all scripts, usage examples, and outputs.

## Visualization Tools

**Interactive Streamlit App:**
```bash
streamlit run streamlit_vol_surface_viewer.py
```
Web-based 3D visualization comparing Oracle, VAE Prior, and Econometric predictions. Features rotatable plots, period selection, date slider, grid point selection.

**Analysis outputs:** `results/presentations/` (summary reports), `results/backfill_16yr/visualizations/` (interactive Plotly dashboards), `results/distribution_analysis/` (distribution plots).

See `results/README.md` for complete structure.

## Additional Evaluation Scripts

**Core utility:** `compute_grid_ci_stats.py` (CI violations per grid point)

**Experiment-specific scripts:** See `experiments/` subdirectories for evaluation scripts (backfill, oracle_vs_prior, econometric_baseline, cointegration, vol_smile).

**Utilities:** `eval_scripts/` (SABR model, evaluation utilities), `test_code/` (autoregressive tests, quantile decoder tests)

For complete script listing, see `experiments/README.md`.

## Bootstrap Baseline

Non-parametric baseline using bootstrap sampling from historical residuals and autoregressive 30-day sequence generation. Includes in-sequence IV-EWMA co-integration tests and comparison vs VAE/econometric baselines.

**Important methodological note:** Bootstrap co-integration tests use **in-sequence** testing (testing all 30 days within each generated sequence), while VAE and econometric tests use **cross-sectional** testing (testing predictions at specific horizons across many dates). Both use ADF tests but measure different properties - cross-sectional tests measure endpoint accuracy, in-sequence tests measure trajectory coherence. Results are not directly comparable.

See `experiments/bootstrap_baseline/` for implementation and `experiments/README.md` for detailed methodology comparison.

## Baseline Comparisons

Additional baseline comparison studies available. See `results/presentations/ANALYSIS_SUMMARY.md` for details.

## Common Development Patterns

**Loading a model:**
```python
model_data = torch.load("path/to/model.pt")
model = CVAEMemRand(model_data["model_config"])
model.load_weights(dict_to_load=model_data)
model.eval()
```

**Generating predictions:**
- Standard models: Returns (B, 1, 5, 5)
- Quantile models: Returns (B, 1, 3, 5, 5) for p05/p50/p95

**Autoregressive generation:** `model.generate_autoregressive_sequence(initial_context, horizon=30)`

For detailed code examples, data preprocessing workflow, and usage patterns, see `DEVELOPMENT.md`.

## Data Preprocessing

**Source data:** WRDS OptionMetrics Ivy DB (SPX options, 2000-2023) + Yahoo Finance (SPX prices)

**Preprocessing pipeline:**
1. `spx_volsurface_generation.ipynb`: Clean raw data, generate interpolated IVS
2. `spx_convert_to_grid.ipynb`: Convert to 5×5 grids
3. Output: `data/vol_surface_with_ret.npz` (main training data)

**Pre-trained models:** https://drive.google.com/drive/folders/1W3KsAJ0YQzK2qnk0c-OIgj26oCAO3NI1?usp=sharing

For detailed download instructions and workflow, see `DEVELOPMENT.md`.

## Important Notes After Reorganization

**Running Scripts:**
- All scripts must be run from the repository root directory
- Example: `python experiments/backfill/context20/test_insample_reconstruction_16yr.py`
- Do NOT cd into subdirectories before running scripts

**Import Structure:**
- Core modules: `from vae.utils import train` (vae/ at root)
- Config files: `from config.backfill_config import BackfillConfig` (config/ at root)
- Data files: Use relative paths from root (e.g., `"data/vol_surface_with_ret.npz"`)
- Output paths: Write to `results/` or `models/` from root

**Path Conventions:**
- Model checkpoints: `models/backfill/context20_production/backfill_16yr.pt`
- Generated results: `results/backfill_16yr/predictions/`
- Config files: `config/backfill_config.py`
- Never use old paths: `models_backfill/` or `tables/` (these are obsolete)

**Git Ignored Directories:**
- `results/` - Generated analysis outputs (230MB)
- `models/` - Model checkpoints (GB-sized)
- `test_spx/` - Test experiments (31GB)
- `archived_experiments/test_*/` - Old test outputs
- `data/` - Input data files

See `REORGANIZATION_SUMMARY.md` for complete details on the recent codebase reorganization.
