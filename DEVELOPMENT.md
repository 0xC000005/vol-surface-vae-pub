# Development Guide

Quick reference for common development patterns and code examples.

## Loading a Trained Model

```python
import torch
from vae.cvae_with_mem_randomized import CVAEMemRand

# Load model checkpoint
model_data = torch.load("path/to/model.pt")
model_config = model_data["model_config"]

# Initialize model with saved configuration
model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
model.eval()
```

## Generating Predictions

### Standard MSE Models

For models trained with MSE loss (original approach):

```python
import torch

# Prepare context data
ctx_data = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}

# Generate single-day forecast
generated_surface = model.get_surface_given_conditions(ctx_data)
# Returns: (B, 1, 5, 5) surface for next day
```

### Quantile Regression Models

For models trained with quantile regression:

```python
import torch

# Prepare context data (same format as MSE models)
ctx_data = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}

# Generate quantile forecasts
generated_surfaces = model.get_surface_given_conditions(ctx_data)
# Returns: (B, 1, 3, 5, 5) - 3 quantile surfaces for next day

# Extract specific quantiles
p05 = generated_surfaces[:, :, 0, :, :]  # 5th percentile (lower bound)
p50 = generated_surfaces[:, :, 1, :, :]  # 50th percentile (median)
p95 = generated_surfaces[:, :, 2, :, :]  # 95th percentile (upper bound)
```

## Autoregressive Multi-Step Generation

Generate multi-day sequences by feeding predictions back as context:

```python
import torch

# Prepare initial context
initial_context = {
    "surface": torch.tensor(...),  # (B, C, 5, 5)
    "ex_feats": torch.tensor(...)  # (B, C, 3) - optional
}

# Generate 30-day autoregressive sequence
with torch.no_grad():
    result = model.generate_autoregressive_sequence(
        initial_context=initial_context,
        horizon=30
    )

# For surface-only models (no_ex):
# result = (B, 30, 3, 5, 5) - 3 quantiles × 30 days

# For ex_feats models (ex_no_loss, ex_loss):
# result = (surfaces, ex_feats) tuple
# surfaces: (B, 30, 3, 5, 5)
# ex_feats: (B, 30, 3)

# Extract quantiles across all 30 days
p05 = result[:, :, 0, :, :]  # (B, 30, 5, 5) - lower bound
p50 = result[:, :, 1, :, :]  # (B, 30, 5, 5) - median forecast
p95 = result[:, :, 2, :, :]  # (B, 30, 5, 5) - upper bound
```

**Key implementation details:**
- Uses p50 (median) as point estimate for context updates each step
- Sliding window: drops oldest surface, appends new prediction
- Supports all 3 model variants (no_ex, ex_no_loss, ex_loss) without modification
- **Autoregressive error accumulation**: For better long-horizon performance, use multi-horizon training (see CLAUDE.md Multi-Horizon Training section)
- Models trained with `horizon > 1` predict multiple days simultaneously, avoiding autoregressive error accumulation

## Data Preprocessing

### WRDS Data Download

Data should be downloaded from WRDS (OptionMetrics Ivy DB):

1. **Dataset**: OptionMetrics/Ivy DB US/Options/Option Prices
2. **Date Range**: 2000-01-01 to 2023-02-28
3. **SECID**: 108105 (S&P 500)
4. **Option Type**: Both
5. **Exercise Type**: Both
6. **Security Type**: Both
7. **Query Variables**: all
8. **Output Format**: *.csv
9. **Compression**: *.zip
10. **Date Format**: YYYY-MM-DD
11. **Save location**: `data/spx.zip`

### Additional Data Required

- **S&P 500 stock prices**: Yahoo Finance ticker `^GSPC`
- **Save location**: `data/GSPC.csv`

### Preprocessing Workflow

The preprocessing pipeline generates 5×5 interpolated volatility surface grids from option prices:

1. **Step 1**: Run `spx_volsurface_generation.ipynb`
   - Cleans raw WRDS data
   - Generates interpolated implied volatility surface (IVS) dataframe

2. **Step 2**: Run `spx_convert_to_grid.ipynb`
   - Converts dataframe to 5×5 numpy grids
   - Computes extra features (returns, skew, slope)

3. **Output files**:
   - `data/vol_surface_with_ret.npz` - Main training data (5×5 grids + features)
   - `data/spx_vol_surface_history_full_data_fixed.parquet` - Full SPX history

### Data Format

The processed data contains:

- `surface`: (N, 5, 5) - N days of 5×5 volatility grids (moneyness × time to maturity)
- `ret`: (N,) - Daily log returns
- `skews`: (N,) - Volatility skew measure
- `slopes`: (N,) - Volatility term structure slope
- `ex_data`: (N, 3) - Concatenated [ret, skew, slope]

## Pre-trained Models

Pre-trained models and parsed data are available at:

https://drive.google.com/drive/folders/1W3KsAJ0YQzK2qnk0c-OIgj26oCAO3NI1?usp=sharing

## Common Tasks

### Training a New Model

```bash
# Train all 3 variants (no_ex, ex_no_loss, ex_loss)
python param_search.py

# Train quantile regression models
python train_quantile_models.py

# Train backfill model (multi-horizon)
python experiments/backfill/context20/train_backfill_model.py
```

### Generating Forecasts

```bash
# Generate stochastic forecasts (1000 samples/day)
python generate_surfaces.py

# Generate maximum likelihood forecasts (deterministic)
python generate_surfaces_max_likelihood.py

# Generate quantile forecasts
python generate_quantile_surfaces.py
```

### Running Analysis

```bash
# Main analysis pipeline
python main_analysis.py

# CI calibration evaluation
python evaluate_quantile_ci_calibration.py

# Visualization
python analysis_code/visualize_teacher_forcing.py
python analysis_code/visualize_backfill_16yr_plotly.py
```

## Development Tips

### Device Handling

All models support both CPU and CUDA. Tensors are automatically moved to the configured device:

```python
# Model automatically detects CUDA availability
model = CVAEMemRand(model_config)
# model.device will be 'cuda' if available, else 'cpu'
```

### Random Seeds

For reproducibility, set random seeds before training:

```python
from vae.utils import set_seeds

set_seeds(42)  # Sets numpy, torch, and torch.cuda seeds
```

### Running Scripts

All scripts must be run from the repository root directory:

```bash
# Correct
python experiments/backfill/context20/test_insample_reconstruction_16yr.py

# Incorrect (will fail with import errors)
cd experiments/backfill/context20
python test_insample_reconstruction_16yr.py
```

### Import Structure

- Core modules: `from vae.utils import train`
- Config files: `from config.backfill_config import BackfillConfig`
- Data files: Use relative paths from root (e.g., `"data/vol_surface_with_ret.npz"`)
- Output paths: Write to `results/` or `models/` from root

## References

- **Model variants**: See `experiments/backfill/MODEL_VARIANTS.md`
- **Quantile regression**: See `experiments/backfill/QUANTILE_REGRESSION.md`
- **Architecture details**: See `vae/` module and CLAUDE.md
- **Experiment documentation**: See `experiments/README.md` and subdirectory READMEs
