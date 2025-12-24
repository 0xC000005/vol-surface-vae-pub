# Context60 V4: Full Covariance Prior (Single-Output Decoder)

## Overview

This experiment folder contains scripts for training and evaluating VAE models with **single-output decoder** (no quantile decoder). This is a preparatory step for implementing Full Covariance Prior, where quantiles will be computed empirically from multiple samples using Cholesky decomposition.

## Key Changes from V3

### Architecture

**V3 (Quantile Decoder):**
- Decoder outputs 3 quantile levels: `(B, T, 3, H, W)` where 3 = [p05, p50, p95]
- Uses pinball loss for direct quantile prediction
- Fast generation (~1 forward pass)

**V4 (Single-Output Decoder):**
- Decoder outputs single prediction: `(B, T, H, W)`
- Uses MSE loss for prediction
- Quantiles computed empirically from multiple samples (future work)
- Preparation for Full Covariance Prior implementation

### Model Configuration

**Removed Parameters:**
- `num_quantiles`: Number of quantile levels (was 3)
- `quantiles`: Quantile levels [0.05, 0.50, 0.95]
- `quantile_loss_weights`: Weights for each quantile

**Retained Parameters:**
- All other model architecture parameters unchanged
- `use_conditional_prior`: Still supported for learned prior network

### Output Shapes

| Component | V3 Shape | V4 Shape |
|-----------|----------|----------|
| Decoder output | `(B, T, 3, H, W)` | `(B, T, H, W)` |
| Saved predictions | `(n_days, H, 3, 5, 5)` | `(n_days, H, 5, 5)` |
| NPZ files | includes `quantiles` array | no `quantiles` array |

## Directory Structure

```
context60_v4_full_cov/
├── teacher_forcing/
│   ├── generate_vae_tf_sequences.py    # Generate predictions (oracle/prior modes)
│   └── validate_vae_tf_sequences.py    # Validate output shapes and values
├── train_backfill_context60_latent12_v4_full_cov.py  # Training script
└── README.md                            # This file
```

## Usage

### 1. Training

Train a new V4 model from scratch:

```bash
python experiments/backfill/context60_v4_full_cov/train_backfill_context60_latent12_v4_full_cov.py
```

Resume from checkpoint:

```bash
python experiments/backfill/context60_v4_full_cov/train_backfill_context60_latent12_v4_full_cov.py \
    --resume_from models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_phase1_ep199.pt
```

**Training Schedule:**
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1, 7, 14, 30, 60, 90]

**Output:**
- `models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_phase1_ep199.pt`
- `models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_phase2_ep599.pt`
- `models/backfill/context60_v4_full_cov/checkpoints/backfill_context60_latent12_v4_full_cov_best.pt`

### 2. Generation

Generate teacher forcing sequences for all periods and horizons:

**Oracle sampling** (posterior with future knowledge):
```bash
# Generate for one period
python experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py \
    --period crisis \
    --sampling_mode oracle

# Generate for all periods
for period in crisis insample oos gap; do
    python experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py \
        --period $period \
        --sampling_mode oracle
done
```

**Prior sampling** (realistic, no future knowledge):
```bash
# Generate for one period
python experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py \
    --period crisis \
    --sampling_mode prior

# Generate for all periods
for period in crisis insample oos gap; do
    python experiments/backfill/context60_v4_full_cov/teacher_forcing/generate_vae_tf_sequences.py \
        --period $period \
        --sampling_mode prior
done
```

**Output:**
- `results/context60_v4_full_cov/predictions/teacher_forcing/oracle/vae_tf_{period}_h{horizon}.npz`
- `results/context60_v4_full_cov/predictions/teacher_forcing/prior/vae_tf_{period}_h{horizon}.npz`

Each file contains:
- `surfaces`: `(n_days, H, 5, 5)` - Single prediction per timestep
- `indices`: `(n_days,)` - Start indices
- `horizon`: Scalar - Forecast horizon
- `sampling_mode`: String - 'oracle' or 'prior'
- `context_len`: Scalar - 60
- `method`: String - 'teacher_forcing'

### 3. Validation

Validate generated predictions:

```bash
# Validate oracle predictions
python experiments/backfill/context60_v4_full_cov/teacher_forcing/validate_vae_tf_sequences.py \
    --sampling_mode oracle

# Validate prior predictions
python experiments/backfill/context60_v4_full_cov/teacher_forcing/validate_vae_tf_sequences.py \
    --sampling_mode prior
```

**Validation checks:**
1. File existence (24 files: 4 periods × 6 horizons)
2. Correct shapes `(n_days, H, 5, 5)`
3. Value ranges (0.01 < IV < 5.0)
4. No NaN/Inf values
5. Index uniqueness
6. Correct metadata (sampling_mode, context_len=60)

## Differences from V3 Scripts

### generate_vae_tf_sequences.py

**Removed:**
- `quantiles=[0.05, 0.50, 0.95]` from npz save
- Printing of `model_config['quantiles']` and `quantile_loss_weights`
- All references to quantile arrays

**Updated:**
- All shape comments: `(n_days, H, 3, 5, 5)` → `(n_days, H, 5, 5)`
- Default paths: `context60_latent12_v3` → `context60_v4_full_cov`
- Docstring: Added "(Single-Output Decoder)" to title

### validate_vae_tf_sequences.py

**Removed:**
- `'quantiles'` from required keys
- Quantile value checks
- Quantile ordering checks (p05 ≤ p50 ≤ p95)
- Quantile-specific summary statistics

**Updated:**
- Shape validation: `(n_days, H, 3, 5, 5)` → `(n_days, H, 5, 5)`
- Default paths: `context60_latent12_v3` → `context60_v4_full_cov`
- Error messages: Updated script paths to v4

### train_backfill_context60_latent12_v4_full_cov.py

**Removed from model_config:**
- `"num_quantiles": cfg.num_quantiles`
- `"quantiles": cfg.quantiles`
- `"quantile_loss_weights": cfg.quantile_loss_weights`

**Updated:**
- Docstring: Changed to "Full Covariance Prior" focus
- Usage examples: Updated paths to v4
- Log file name: `context60_latent12_v4_full_cov_training_log.txt`
- Description: "Train Context60 Latent12 V4 model with Full Covariance Prior"

**Note:** Currently reuses v3 config file (quantile params will be ignored by new model). Create separate v4 config if needed.

## Model Compatibility

**V4 models are NOT compatible with V3 evaluation scripts** due to shape differences:
- V3 scripts expect `(B, T, 3, H, W)` outputs
- V4 scripts expect `(B, T, H, W)` outputs

Use the corresponding script versions:
- V3 models → `experiments/backfill/context60_v3_fixed/` scripts
- V4 models → `experiments/backfill/context60_v4_full_cov/` scripts

## Future Work: Full Covariance Prior

The single-output decoder is a preparatory step for Full Covariance Prior implementation:

1. **Multiple samples**: Generate N samples from prior p(z|context)
2. **Empirical covariance**: Compute full 25×25 covariance matrix across surface grid
3. **Cholesky decomposition**: Generate correlated samples preserving spatial structure
4. **Empirical quantiles**: Compute p05, p50, p95 from correlated samples

This approach will:
- Preserve spatial correlations in uncertainty estimates
- Eliminate quantile crossing artifacts
- Better calibrate confidence intervals
- Capture regime-specific uncertainty patterns

## Testing Status

- [x] Scripts created and updated for single-output decoder
- [x] Shape comments updated throughout
- [x] Default paths updated to v4
- [ ] Generation script tested with actual model
- [ ] Validation script tested with generated outputs
- [ ] Training script tested (end-to-end training)

## Contact

For questions about this experiment, see:
- Main documentation: `CLAUDE.md`
- VAE architecture: `vae/README.md`
- Experiment inventory: `experiments/EXPERIMENT_INVENTORY.md`
