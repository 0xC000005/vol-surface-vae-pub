# Context20 Backfill Model (16-Year Training)

This directory contains the main production backfill model trained on 16 years of data (2004-2019) with context length of 20 days.

## Model Configuration

- **Training data**: 2004-2019 (indices 1000-5000)
- **Context length**: 20 days
- **Horizons**: [1, 7, 14, 30] days
- **Architecture**: CVAEMemRand with quantile regression decoder
- **Latent dimension**: 5
- **Memory**: LSTM with 100 hidden units
- **Quantiles**: [0.05, 0.5, 0.95]

## Training

**Main training script:**
```bash
python experiments/backfill/context20/train_backfill_model.py
```

**Training phases:**
1. Phase 1 (epochs 0-199): Teacher forcing on horizon=1
2. Phase 2 (epochs 200-349): Multi-horizon [1, 7, 14, 30]
3. Phase 3 (epochs 350-499): Multi-horizon with adjusted weights

**Output:**
- Model checkpoint: `models/backfill/context20_production/backfill_16yr.pt`
- Training logs: `models/backfill/context20_production/training_logs/`

## Testing & Generation

**In-sample reconstruction (2004-2019):**
```bash
python experiments/backfill/context20/test_insample_reconstruction_16yr.py
```
Output: `results/backfill_16yr/predictions/insample_reconstruction_16yr.npz`

**Out-of-sample reconstruction (2019-2023):**
```bash
python experiments/backfill/context20/test_oos_reconstruction_16yr.py
```
Output: `results/backfill_16yr/predictions/oos_reconstruction_16yr.npz`

**VAE Prior testing:**
```bash
# In-sample
python experiments/backfill/context20/test_vae_prior_insample_16yr.py

# Out-of-sample
python experiments/backfill/context20/test_vae_prior_oos_16yr.py
```

## Teacher Forcing Sequence Generation (Multi-Period, Multi-Horizon)

**Generate full H-day sequences with configurable sampling strategy:**

The generation pipeline supports two sampling modes:
- **`oracle`** (default): Posterior sampling q(z|context,target) - uses future knowledge (upper bound)
- **`prior`**: Realistic sampling with context only - z[:,:C] = posterior_mean, z[:,C:] ~ N(0,1)

```bash
# Generate single period with oracle sampling
python experiments/backfill/context20/generate_vae_tf_sequences.py --period crisis --sampling_mode oracle

# Generate single period with prior sampling (realistic)
python experiments/backfill/context20/generate_vae_tf_sequences.py --period oos --sampling_mode prior

# Generate all periods (crisis, insample, oos, gap) for one sampling mode
bash experiments/backfill/context20/run_generate_all_tf_sequences.sh oracle
bash experiments/backfill/context20/run_generate_all_tf_sequences.sh prior

# Validate generated files
python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode oracle
python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode prior
```

**Output structure:**
```
results/vae_baseline/predictions/autoregressive/
├── oracle/
│   ├── vae_tf_crisis_h{1,7,14,30}.npz
│   ├── vae_tf_insample_h{1,7,14,30}.npz
│   ├── vae_tf_oos_h{1,7,14,30}.npz
│   └── vae_tf_gap_h{1,7,14,30}.npz
└── prior/
    ├── vae_tf_crisis_h{1,7,14,30}.npz
    ├── vae_tf_insample_h{1,7,14,30}.npz
    ├── vae_tf_oos_h{1,7,14,30}.npz
    └── vae_tf_gap_h{1,7,14,30}.npz
```

**Analysis Pipeline:**

All analysis scripts support `--sampling_mode oracle/prior` parameter:

```bash
# 1. Compute CI width statistics
python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode oracle
python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode prior

# 2. Merge gap period statistics (if needed)
python experiments/backfill/context20/compute_gap_ci_stats.py --sampling_mode oracle

# 3. Visualize time series
python experiments/backfill/context20/visualize_sequence_ci_width.py --period insample --sampling_mode oracle
python experiments/backfill/context20/visualize_sequence_ci_width_combined.py --sampling_mode prior

# 4. Correlation analysis
python experiments/backfill/context20/analyze_sequence_ci_correlations.py --sampling_mode oracle

# 5. Identify extreme events
python experiments/backfill/context20/identify_ci_width_events.py --sampling_mode prior

# 6. Compare oracle vs prior
python experiments/backfill/context20/compare_oracle_vs_prior_ci.py
```

**Key Findings:**
- Prior CIs are ~2-3× wider than oracle CIs (VAE prior mismatch)
- All differences statistically significant (p < 0.001)
- Demonstrates realistic uncertainty quantification vs upper bound performance

## Evaluation

**CI Calibration:**
```bash
python experiments/backfill/context20/evaluate_insample_ci_16yr.py
python experiments/backfill/context20/evaluate_vae_prior_ci_insample_16yr.py
python experiments/backfill/context20/evaluate_vae_prior_ci_oos_16yr.py
```

**RMSE Evaluation:**
```bash
python experiments/backfill/context20/evaluate_rmse_16yr.py
```

## Analysis

**VAE Health Analysis:**
```bash
# In-sample
python experiments/backfill/context20/analyze_vae_health_16yr.py
python experiments/backfill/context20/visualize_vae_health_16yr.py

# Out-of-sample
python experiments/backfill/context20/analyze_vae_health_oos_16yr.py
python experiments/backfill/context20/visualize_vae_health_oos_16yr.py
```

**Latent Analysis:**
```bash
python experiments/backfill/context20/analyze_latent_distributions_16yr.py
python experiments/backfill/context20/analyze_latent_contribution_16yr.py
```

**Comparisons:**
```bash
# Oracle vs VAE Prior
python experiments/backfill/context20/compare_oracle_vs_vae_prior_16yr.py

# Zero latent vs Prior sampling
python experiments/backfill/context20/test_zero_vs_prior_latent_16yr.py

# Dimension ablation
python experiments/backfill/context20/test_dimension_ablation_16yr.py
```

## Key Results

**Performance (Out-of-Sample 2019-2023):**
- CI Violations: ~28% (target: 10%, but improvement from baseline)
- RMSE: Varies by grid point and horizon
- VAE Health: Effective dimension ~3/5 latent dimensions

**Comparison vs Baselines:**
- vs Econometric (2008-2010 Crisis): 38% lower RMSE, 87% win rate
- vs Econometric (OOS): Better on extremes, competitive on ATM

See `results/presentations/MILESTONE_PRESENTATION.md` for detailed results.
