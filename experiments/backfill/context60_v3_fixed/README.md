# Context60 Latent12 V3 FIXED Experiments

This folder contains all experiments related to the **Context60 Latent12 V3 model with conditional prior network** (FIXED version).

## Model Overview

**Model checkpoint:** `models/backfill/context60_experiment/checkpoints/backfill_context60_latent12_v3_conditional_prior_best.pt`

**Key features:**
- Context length: 60 days
- Latent dimension: 12
- **Conditional prior network**: p(z|context) - learned prior that depends on context
- Training: Phase 1 (teacher forcing) + Phase 2 (multi-horizon)
- Horizons: [1, 7, 14, 30, 60, 90] days

**Results:** `results/context60_latent12_v3_FIXED/`

## Key Findings

### ✅ What Model CAN Do: Unconditional Distribution
The model successfully captures the **unconditional marginal distribution** across contexts:
- At H=90: std ratio = 103.5% (matches ground truth fanning pattern)
- Point predictions across different contexts have correct spread
- Fanning pattern matches ground truth (99.4% at H=90)

### ❌ What Model CANNOT Do: Conditional Distribution
The model fails to capture **conditional distribution** for single context:
- E[Var(X|C)] / Var(X) = 0.44% (near zero conditional variance)
- Sampling same context 1000× gives nearly identical outputs
- Cannot generate diverse scenarios for risk management
- All samples cluster tightly around p50, but p50 ≠ GT exactly

**Variance decomposition:**
- Var(X) = E[Var(X|C)] + Var(E[X|C])
- E[Var(X|C)] ≈ 0 (no conditional variance)
- Var(E[X|C]) = 178% of Var(X) (overspends uncertainty budget)

## Scripts

### Training
- `train_backfill_context60_latent12_v3_conditional_prior.py` - Train V3 model with conditional prior

### Verification & Analysis
- `verify_conditional_vs_unconditional.py` - Verify conditional vs unconditional distribution capture
- `check_v3_marginal_h90.py` - Quick marginal distribution check at H=90
- `test_smoothness_experiments.py` - Test path smoothness (VAE produces smoother paths)
- `analyze_ci_coverage_by_regime.py` - CI coverage by market regime

### Visualization
- `visualize_fanning_pattern_v3_fixed.py` - Fanning pattern (std vs horizon) visualization

### Supporting
- `test_encoder_caching_verification.py` - Verify encoder caching optimization
- `FIXED_CONTEXT_OPTIMIZATION.md` - Documentation on fixed context optimization

### Teacher Forcing (subfolder)
- `generate_vae_tf_sequences.py` - Generate teacher forcing sequences (oracle/prior mode)
- `validate_vae_tf_sequences.py` - Validate generated sequences
- `run_generate_all_tf_sequences.sh` - Batch generation for all periods

## Usage

All scripts should be run from repository root:

```bash
# Training
python experiments/backfill/context60_v3_fixed/train_backfill_context60_latent12_v3_conditional_prior.py

# Verification experiment
python experiments/backfill/context60_v3_fixed/verify_conditional_vs_unconditional.py

# Visualization
python experiments/backfill/context60_v3_fixed/visualize_fanning_pattern_v3_fixed.py

# Generate sequences (teacher forcing)
python experiments/backfill/context60_v3_fixed/teacher_forcing/generate_vae_tf_sequences.py --period crisis --sampling_mode prior

# Run all periods
bash experiments/backfill/context60_v3_fixed/teacher_forcing/run_generate_all_tf_sequences.sh prior
```

## Next Steps

See `CONDITIONAL_DISTRIBUTION_FRAMEWORK.md` in the repository root for proposed solutions:
- Solution A: Post-hoc variance calibration
- Solution B: Cluster-based conditional variance
- Solution C: Regime-dependent variance
- Solution D: Bootstrap from residuals
- Solution E: Conformal prediction
- Solution F: Training with prior samples (most promising)
