# VAE Experiment Inventory & Standardized Testing Guide

**Purpose:** Comprehensive catalog of all VAE experiment scripts for standardized model testing and evaluation.

**Scope:** Context20 and Context60 models (v1/v2/v3) - VAE experiments only

**Last Updated:** December 2025

---

## Table of Contents

1. [Quick Start: New Model Testing Checklist](#1-quick-start-new-model-testing-checklist)
2. [Model Configurations & Checkpoints](#2-model-configurations--checkpoints)
3. [Training Scripts](#3-training-scripts)
4. [Generation Scripts](#4-generation-scripts)
5. [Validation & CI Calibration](#5-validation--ci-calibration)
6. [Analysis Scripts](#6-analysis-scripts)
7. [Visualization Scripts](#7-visualization-scripts)
8. [Reusable Modules (analysis_code/)](#8-reusable-modules-analysis_code)
9. [Shell Scripts & Batch Runners](#9-shell-scripts--batch-runners)
10. [Appendix: Full Script Index](#10-appendix-full-script-index)

---

## 1. Quick Start: New Model Testing Checklist

When you train a new model, follow this standardized workflow to generate all results and evaluations.

### Standard Testing Pipeline (7 Phases)

```bash
# ========================================
# PHASE 1: TRAINING
# ========================================
python experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py

# Output: models/backfill/context60_experiment/backfill_context60_v3_best.pt

# ========================================
# PHASE 2: TEACHER FORCING GENERATION
# ========================================
# Generate predictions for all periods (crisis, insample, oos, gap) with both sampling modes

bash experiments/backfill/context60/teacher_forcing/run_generate_all_tf_sequences.sh prior
bash experiments/backfill/context60/teacher_forcing/run_generate_all_tf_sequences.sh oracle

# Output: results/vae_baseline/predictions/autoregressive/{oracle,prior}/vae_tf_*.npz

# ========================================
# PHASE 3: VALIDATION
# ========================================
python experiments/backfill/context60/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode prior
python experiments/backfill/context60/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode oracle

# Validates file shapes, checks for NaN/Inf, verifies horizons

# ========================================
# PHASE 4: CI WIDTH STATISTICS
# ========================================
python experiments/backfill/context60/compute_sequence_ci_width_stats_context60.py

# Output: results/vae_baseline/analysis/{oracle,prior}/sequence_ci_width_stats.csv

# ========================================
# PHASE 5: ORACLE VS PRIOR COMPARISON
# ========================================
python experiments/backfill/context60/compare_oracle_vs_prior_ci_context60.py

# Output: Statistical comparison of oracle vs prior CI widths, identifies VAE prior mismatch

# ========================================
# PHASE 6: 4-REGIME VISUALIZATION
# ========================================
python experiments/backfill/context60/visualize_four_regime_overlay_context60.py
python experiments/backfill/context60/visualize_four_regime_overlay_context60_h90.py

# Output: results/vae_baseline/visualizations/comparison/four_regime_overlay_context60_*.png

# ========================================
# PHASE 7: VAE HEALTH ANALYSIS
# ========================================
python experiments/backfill/context60/analyze_latent_space.py

# Output: Latent dimension utilization, effective rank, PCA analysis
```

### Dependency Chain

```
train_backfill_model.py (Phase 1)
    │
    ├→ generate_vae_tf_sequences.py (Phase 2)
    │     │
    │     ├→ validate_vae_tf_sequences.py (Phase 3)
    │     │
    │     ├→ compute_sequence_ci_width_stats.py (Phase 4)
    │     │     │
    │     │     ├→ visualize_sequence_ci_width_combined.py
    │     │     ├→ analyze_sequence_ci_correlations.py
    │     │     └→ identify_ci_width_events.py
    │     │
    │     ├→ compare_oracle_vs_prior_ci.py (Phase 5)
    │     │     │
    │     │     └→ visualize_oracle_vs_prior_combined.py
    │     │
    │     └→ visualize_four_regime_overlay.py (Phase 6)
    │
    └→ analyze_latent_space.py (Phase 7)
```

### Critical Files Generated

| Phase | Output | Path |
|-------|--------|------|
| Training | Model checkpoint | `models/backfill/context60_experiment/backfill_*.pt` |
| TF Generation | Prediction sequences | `results/vae_baseline/predictions/autoregressive/{oracle,prior}/` |
| CI Stats | CI width statistics | `results/vae_baseline/analysis/{oracle,prior}/sequence_ci_width_stats.csv` |
| Visualization | 4-regime plots | `results/vae_baseline/visualizations/comparison/` |

---

## 2. Model Configurations & Checkpoints

### Context20 (Production Model)

| Attribute | Value |
|-----------|-------|
| **Name** | Context20 Production (16-year model) |
| **Context Length** | 20 days |
| **Latent Dimension** | 5 |
| **Training Period** | 2004-2019 (indices 1000-5000) |
| **Horizons** | [1, 7, 14, 30] days |
| **Config File** | `config/backfill_config.py` |
| **Checkpoint** | `models/backfill/context20_production/backfill_16yr.pt` |
| **Training Script** | `experiments/backfill/context20/train_backfill_model.py` |
| **Documentation** | `experiments/backfill/context20/README.md` |

### Context60 Variants

#### Base Model
| Attribute | Value |
|-----------|-------|
| **Name** | Context60 Base |
| **Context Length** | 60 days |
| **Latent Dimension** | 5 |
| **Config File** | `config/backfill_context60_config.py` |
| **Checkpoint** | `models/backfill/context60_experiment/backfill_context60_best.pt` |
| **Training Script** | `experiments/backfill/context60/train_backfill_context60.py` |

#### Latent12 v1
| Attribute | Value |
|-----------|-------|
| **Name** | Context60 Latent12 v1 |
| **Latent Dimension** | 12 (expanded capacity) |
| **Config File** | `config/backfill_context60_config_latent12.py` |
| **Training Script** | `experiments/backfill/context60/train_backfill_context60_latent12.py` |

#### Latent12 v2 (Improved)
| Attribute | Value |
|-----------|-------|
| **Name** | Context60 Latent12 v2 |
| **Latent Dimension** | 12 (improved architecture) |
| **Config File** | `config/backfill_context60_config_latent12_v2.py` |
| **Training Script** | `experiments/backfill/context60/train_backfill_context60_latent12_v2.py` |
| **KL Diagnostics** | `experiments/backfill/context60/diagnose_kl_divergence_v2.py` |

#### Latent12 v3 (Conditional Prior) 🚀
| Attribute | Value |
|-----------|-------|
| **Name** | Context60 Latent12 v3 - Conditional Prior |
| **Latent Dimension** | 12 |
| **Key Innovation** | Context-adaptive prior p(z\|context) instead of fixed N(0,1) |
| **Config File** | `config/backfill_context60_config_latent12_v3_conditional_prior.py` |
| **Training Script** | `experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py` |
| **Status** | **Current recommended model** |

---

## 3. Training Scripts

### Context20 Training

#### train_backfill_model.py
**Path:** `experiments/backfill/context20/train_backfill_model.py`

**Purpose:** Train production 16-year backfill model with context length 20.

**Configuration:**
- Config: `config/backfill_config.py`
- Context: 20 days
- Latent dim: 5
- Horizons: [1, 7, 14, 30]
- Training data: 2004-2019 (4,001 days)

**Training Phases:**
1. Phase 1 (epochs 0-199): Teacher forcing H=1
2. Phase 2 (epochs 200-349): Multi-horizon [1, 7, 14, 30]
3. Phase 3 (epochs 350-499): Multi-horizon with adjusted weights

**Output:**
- Checkpoint: `models/backfill/context20_production/backfill_16yr.pt`
- Logs: `models/backfill/context20_production/training_logs/`

**Usage:**
```bash
python experiments/backfill/context20/train_backfill_model.py
```

---

### Context60 Training

#### train_backfill_context60.py
**Path:** `experiments/backfill/context60/train_backfill_context60.py`

**Purpose:** Initial training of context60 base model.

**Configuration:**
- Config: `config/backfill_context60_config.py`
- Context: 60 days
- Latent dim: 5

**Output:** `models/backfill/context60_experiment/backfill_context60_best.pt`

**Usage:**
```bash
python experiments/backfill/context60/train_backfill_context60.py
```

---

#### train_backfill_context60_resume.py
**Path:** `experiments/backfill/context60/train_backfill_context60_resume.py`

**Purpose:** Resume context60 training at Phase 2.

**Usage:**
```bash
python experiments/backfill/context60/train_backfill_context60_resume.py
```

---

#### train_backfill_context60_resume_phase3.py
**Path:** `experiments/backfill/context60/train_backfill_context60_resume_phase3.py`

**Purpose:** Resume context60 training at Phase 3.

**Usage:**
```bash
python experiments/backfill/context60/train_backfill_context60_resume_phase3.py
```

---

#### train_backfill_context60_latent12.py
**Path:** `experiments/backfill/context60/train_backfill_context60_latent12.py`

**Purpose:** Train context60 with expanded latent dimension (12 instead of 5).

**Configuration:**
- Config: `config/backfill_context60_config_latent12.py`
- Latent dim: 12

**Hypothesis:** Larger latent space may capture more fine-grained patterns.

**Usage:**
```bash
python experiments/backfill/context60/train_backfill_context60_latent12.py
```

---

#### train_backfill_context60_latent12_v2.py
**Path:** `experiments/backfill/context60/train_backfill_context60_latent12_v2.py`

**Purpose:** Improved latent12 training with better initialization and KL annealing.

**Configuration:**
- Config: `config/backfill_context60_config_latent12_v2.py`
- Latent dim: 12

**Improvements over v1:**
- Better KL weight scheduling
- Improved decoder initialization
- Adjusted learning rate

**Usage:**
```bash
python experiments/backfill/context60/train_backfill_context60_latent12_v2.py
```

---

#### train_backfill_context60_latent12_v3_conditional_prior.py 🚀
**Path:** `experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py`

**Purpose:** Train context60 with **conditional prior network** - learns p(z|context) instead of fixed N(0,1).

**Configuration:**
- Config: `config/backfill_context60_config_latent12_v3_conditional_prior.py`
- Latent dim: 12
- **Key Feature:** Context-adaptive uncertainty

**Key Innovation:**
- Prior network learns context-dependent distribution
- Eliminates systematic VAE bias
- Regime-specific behavior (automatically widens CIs during crises)

**Usage:**
```bash
python experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py
```

**Status:** ✅ **Currently recommended model**

---

## 4. Generation Scripts

### Teacher Forcing (TF) Generation

#### Context20: generate_vae_tf_sequences.py
**Path:** `experiments/backfill/context20/generate_vae_tf_sequences.py`

**Purpose:** Generate multi-horizon teacher forcing sequences for all periods.

**Parameters:**
- `--period`: crisis, insample, oos, gap
- `--sampling_mode`: oracle (posterior) or prior (realistic)
- `--horizons`: [1, 7, 14, 30] (default)

**Sampling Modes:**
- **Oracle**: z ~ q(z|context, target) - uses future knowledge (upper bound)
- **Prior**: z[:,:C] = posterior_mean (deterministic), z[:,C:] ~ N(0,1) (stochastic)

**Usage:**
```bash
# Single period with oracle sampling
python experiments/backfill/context20/generate_vae_tf_sequences.py --period crisis --sampling_mode oracle

# Single period with prior sampling (realistic)
python experiments/backfill/context20/generate_vae_tf_sequences.py --period oos --sampling_mode prior
```

**Output Structure:**
```
results/vae_baseline/predictions/autoregressive/
├── oracle/
│   ├── vae_tf_crisis_h1.npz
│   ├── vae_tf_crisis_h7.npz
│   ├── vae_tf_crisis_h14.npz
│   ├── vae_tf_crisis_h30.npz
│   ├── vae_tf_insample_h*.npz
│   ├── vae_tf_oos_h*.npz
│   └── vae_tf_gap_h*.npz
└── prior/
    └── (same structure as oracle/)
```

**File Format:**
- `surfaces`: (num_days, num_samples, 3, 5, 5) - [p05, p50, p95] quantiles
- `dates`: (num_days,) - Date index for each prediction
- `ex_feats`: (num_days, num_samples, 3, 3) - Extra features if model returns them

---

#### Context60: generate_vae_tf_sequences.py
**Path:** `experiments/backfill/context60/teacher_forcing/generate_vae_tf_sequences.py`

**Purpose:** Same as context20 but for context60 models.

**Usage:**
```bash
python experiments/backfill/context60/teacher_forcing/generate_vae_tf_sequences.py --period crisis --sampling_mode prior
```

---

### Autoregressive (AR) Generation

#### Context60: generate_vae_ar_sequences.py
**Path:** `experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py`

**Purpose:** Generate 30-day autoregressive sequences (model predicts, then uses own predictions as context).

**Usage:**
```bash
python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py --period crisis
```

**Output:** `results/vae_baseline/predictions/autoregressive/ar/vae_ar_crisis.npz`

**Note:** AR generation shows discontinuity issues documented in `experiments/backfill/context60/AR_DISCONTINUITY_ANALYSIS.md`

---

## 5. Validation & CI Calibration

### Validation Scripts

#### validate_vae_tf_sequences.py (Context20)
**Path:** `experiments/backfill/context20/validate_vae_tf_sequences.py`

**Purpose:** Validate generated TF sequences for correctness.

**Checks:**
- File existence for all periods and horizons
- Shape correctness: (num_days, num_samples, 3, 5, 5)
- No NaN or Inf values
- Date continuity
- Quantile ordering (p05 < p50 < p95)

**Usage:**
```bash
python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode prior
python experiments/backfill/context20/validate_vae_tf_sequences.py --sampling_mode oracle
```

---

#### validate_vae_tf_sequences.py (Context60)
**Path:** `experiments/backfill/context60/teacher_forcing/validate_vae_tf_sequences.py`

**Purpose:** Same validation for context60 TF sequences.

**Usage:**
```bash
python experiments/backfill/context60/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode prior
```

---

#### validate_vae_ar_sequences.py (Context60)
**Path:** `experiments/backfill/context60/autoregressive/validate_vae_ar_sequences.py`

**Purpose:** Validate autoregressive sequences.

**Usage:**
```bash
python experiments/backfill/context60/autoregressive/validate_vae_ar_sequences.py
```

---

### CI Statistics Scripts

#### compute_sequence_ci_width_stats.py (Context20)
**Path:** `experiments/backfill/context20/compute_sequence_ci_width_stats.py`

**Purpose:** Compute comprehensive CI width statistics across all periods and horizons.

**Metrics Computed:**
- Mean CI width by period and horizon
- 5th, 50th, 95th percentiles of CI width
- Correlation with volatility measures
- Temporal evolution

**Usage:**
```bash
python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode prior
python experiments/backfill/context20/compute_sequence_ci_width_stats.py --sampling_mode oracle
```

**Output:**
- `results/vae_baseline/analysis/prior/sequence_ci_width_stats.csv`
- `results/vae_baseline/analysis/oracle/sequence_ci_width_stats.csv`

---

#### compute_sequence_ci_width_stats_context60.py
**Path:** `experiments/backfill/context60/compute_sequence_ci_width_stats_context60.py`

**Purpose:** Same CI width statistics for context60 models.

**Usage:**
```bash
python experiments/backfill/context60/compute_sequence_ci_width_stats_context60.py
```

---

#### compute_gap_ci_stats.py (Context20)
**Path:** `experiments/backfill/context20/compute_gap_ci_stats.py`

**Purpose:** Merge gap period statistics into main CI width analysis.

**Usage:**
```bash
python experiments/backfill/context20/compute_gap_ci_stats.py --sampling_mode prior
```

---

### Evaluation Scripts

#### evaluate_insample_ci_16yr.py
**Path:** `experiments/backfill/context20/evaluate_insample_ci_16yr.py`

**Purpose:** Evaluate in-sample CI calibration (2004-2019).

**Metrics:**
- CI violation rate (target: ~10%)
- Violations below p05 and above p95 (should be ~5% each)
- Mean CI width

**Usage:**
```bash
python experiments/backfill/context20/evaluate_insample_ci_16yr.py
```

---

#### evaluate_vae_prior_ci_insample_16yr.py
**Path:** `experiments/backfill/context20/evaluate_vae_prior_ci_insample_16yr.py`

**Purpose:** Evaluate VAE prior sampling CI calibration (in-sample).

**Usage:**
```bash
python experiments/backfill/context20/evaluate_vae_prior_ci_insample_16yr.py
```

---

#### evaluate_vae_prior_ci_oos_16yr.py
**Path:** `experiments/backfill/context20/evaluate_vae_prior_ci_oos_16yr.py`

**Purpose:** Evaluate VAE prior sampling CI calibration (out-of-sample 2019-2023).

**Usage:**
```bash
python experiments/backfill/context20/evaluate_vae_prior_ci_oos_16yr.py
```

---

#### evaluate_rmse_16yr.py
**Path:** `experiments/backfill/context20/evaluate_rmse_16yr.py`

**Purpose:** Compute RMSE metrics across horizons and grid points.

**Metrics:**
- RMSE by horizon [1, 7, 14, 30]
- RMSE by grid position (5×5)
- In-sample vs OOS comparison

**Usage:**
```bash
python experiments/backfill/context20/evaluate_rmse_16yr.py
```

---

### Testing Scripts

#### test_insample_reconstruction_16yr.py
**Path:** `experiments/backfill/context20/test_insample_reconstruction_16yr.py`

**Purpose:** Test in-sample reconstruction quality.

**Usage:**
```bash
python experiments/backfill/context20/test_insample_reconstruction_16yr.py
```

---

#### test_oos_reconstruction_16yr.py
**Path:** `experiments/backfill/context20/test_oos_reconstruction_16yr.py`

**Purpose:** Test out-of-sample reconstruction quality.

**Usage:**
```bash
python experiments/backfill/context20/test_oos_reconstruction_16yr.py
```

---

#### test_vae_prior_insample_16yr.py
**Path:** `experiments/backfill/context20/test_vae_prior_insample_16yr.py`

**Purpose:** Test VAE prior sampling (realistic deployment scenario) in-sample.

**Usage:**
```bash
python experiments/backfill/context20/test_vae_prior_insample_16yr.py
```

---

#### test_vae_prior_oos_16yr.py
**Path:** `experiments/backfill/context20/test_vae_prior_oos_16yr.py`

**Purpose:** Test VAE prior sampling out-of-sample.

**Usage:**
```bash
python experiments/backfill/context20/test_vae_prior_oos_16yr.py
```

---

#### test_zero_vs_prior_latent_16yr.py
**Path:** `experiments/backfill/context20/test_zero_vs_prior_latent_16yr.py`

**Purpose:** Compare deterministic (z=0) vs stochastic prior sampling.

**Usage:**
```bash
python experiments/backfill/context20/test_zero_vs_prior_latent_16yr.py
```

---

#### test_dimension_ablation_16yr.py
**Path:** `experiments/backfill/context20/test_dimension_ablation_16yr.py`

**Purpose:** Ablation study - how many latent dimensions are actually used?

**Usage:**
```bash
python experiments/backfill/context20/test_dimension_ablation_16yr.py
```

---

## 6. Analysis Scripts

### CI Width Analysis Suite (8 scripts)

**Research Question:** What drives the model to widen confidence intervals?

**Key Discovery:** Spatial features (surface shape) dominate (68-75%) over temporal features (recent returns/volatility).

#### investigate_ci_width_peaks.py
**Path:** `experiments/backfill/context20/investigate_ci_width_peaks.py`

**Purpose:** Systematic peak identification and regression decomposition of CI width drivers.

**Features:**
- Identifies top 10% CI width peaks
- Regression analysis: CI width ~ temporal + spatial features
- Feature importance decomposition

**Temporal Features:**
- abs_returns: Absolute daily returns
- realized_vol_30d: 30-day realized volatility

**Spatial Features:**
- atm_vol: At-the-money volatility level
- slopes: Volatility term structure slope
- skews: Volatility smile skew

**Usage:**
```bash
python experiments/backfill/context20/investigate_ci_width_peaks.py --sampling_mode prior --percentile_threshold 90
```

**Parameters:**
- `--sampling_mode`: oracle or prior (default: prior)
- `--percentile_threshold`: 90 (top 10%) or 95 (top 5%)

**Output:**
- Analysis: `results/vae_baseline/analysis/ci_peaks/prior/`
- Visualizations: `results/vae_baseline/visualizations/ci_peaks/prior/`
- Report: `CI_PEAKS_INVESTIGATION_REPORT.md`

**Key Findings (H=30):**
| Feature | Importance | Coefficient Sign |
|---------|------------|------------------|
| ATM volatility | 75.6% | Positive |
| Slopes | 11.9% | **Negative** (steeper → narrower CIs) |
| Realized vol 30d | 6.1% | Positive |
| Skews | 4.4% | Variable |
| Absolute returns | 2.1% | Positive |

---

#### investigate_ci_width_anomaly.py
**Path:** `experiments/backfill/context20/investigate_ci_width_anomaly.py`

**Purpose:** Deep investigation of "low-vol high-CI anomaly" in pre-crisis period (2007-2008).

**Phenomenon:** Model widens CIs despite low ATM volatility, demonstrating pre-crisis detection capability.

**Analysis Pipeline:**
1. Regime identification (anomaly vs normal vs crisis)
2. Feature comparison by regime
3. Temporal evolution analysis
4. Visualization generation
5. Comprehensive reporting

**Usage:**
```bash
python experiments/backfill/context20/investigate_ci_width_anomaly.py
```

**Output:**
- Directory: `results/vae_baseline/analysis/ci_peaks/prior/anomaly_investigation/`
- CSVs: regime_identification.csv, feature_comparison_by_regime.csv, regression_by_regime.csv
- Visualizations: 5 PNG files
- Report: `CI_WIDTH_ANOMALY_REPORT.md`

**Key Discovery:**
- 458 days (8.1% of dataset) show low-vol high-CI pattern
- 81.1% of high-CI periods have ATM vol < 0.3
- Primary occurrence: 2007-2008 pre-crisis buildup
- Model detects structural risk beyond ATM volatility

---

#### visualize_calm_vs_crisis_overlay.py
**Path:** `experiments/backfill/context20/visualize_calm_vs_crisis_overlay.py`

**Purpose:** Simple 2-regime comparison showing same model produces different CI widths based on context.

**Regimes:**
- **2007-03-28** (calm, narrow CI) - BLUE
- **2008-10-30** (crisis, wide CI) - RED

**Visualization:** Overlaid 5×5 grids with p05/p50/p95 quantiles for both dates.

**Usage:**
```bash
python experiments/backfill/context20/visualize_calm_vs_crisis_overlay.py
```

**Output:** `results/vae_baseline/visualizations/comparison/calm_vs_crisis_overlay_prior.png`

---

#### visualize_three_regime_overlay.py
**Path:** `experiments/backfill/context20/visualize_three_regime_overlay.py`

**Purpose:** 3-regime comparison emphasizing anomalous pre-crisis detection pattern.

**Regimes:**
- **2007-03-28** (normal/calm) - GREEN
- **2007-10-09** (low-vol high-CI anomaly, pre-crisis) - ORANGE ← Key finding
- **2008-10-30** (high-vol high-CI crisis) - RED

**Key Insight:** Orange date shows model detects structural risk before crisis manifests in ATM volatility.

**Usage:**
```bash
python experiments/backfill/context20/visualize_three_regime_overlay.py
```

**Output:** `results/vae_baseline/visualizations/comparison/three_regime_overlay_prior.png`

---

#### visualize_four_regime_overlay.py ⭐
**Path:** `experiments/backfill/context20/visualize_four_regime_overlay.py`

**Purpose:** Complete 2×2 regime matrix demonstrating intelligent pattern recognition beyond volatility levels.

**Regimes (2×2 matrix):**
- **Green** (Low Vol, Low CI): 2007-03-28 - Normal baseline
- **Blue** (High Vol, Low CI): 2009-02-23 - **Intelligent confidence** despite high vol
- **Orange** (Low Vol, High CI): 2007-10-09 - **Pre-crisis detection** anomaly
- **Red** (High Vol, High CI): 2008-10-30 - Expected crisis behavior

**Key Insights:**
- Blue and Red both have high context volatility but Blue CI is ~70% narrower
- Demonstrates model recognizes familiar patterns → confident even in volatile conditions
- Orange shows model detects structural risk beyond ATM volatility

**Usage:**
```bash
python experiments/backfill/context20/visualize_four_regime_overlay.py
```

**Output:** `results/vae_baseline/visualizations/comparison/four_regime_overlay_prior.png`

---

#### visualize_four_regime_timeline.py
**Path:** `experiments/backfill/context20/visualize_four_regime_timeline.py`

**Purpose:** Multi-panel timeline showing where the 4 regime dates appear in full 2000-2023 history.

**Panels:**
1. SPX price history with 4 regime markers
2. ATM volatility timeline
3. CI width evolution (H=30)
4. Regime classification scatter plot

**Usage:**
```bash
python experiments/backfill/context20/visualize_four_regime_timeline.py
```

**Output:** `results/vae_baseline/visualizations/comparison/four_regime_timeline_prior.png`

---

#### visualize_four_regime_overlay_oos.py
**Path:** `experiments/backfill/context20/visualize_four_regime_overlay_oos.py`

**Purpose:** 4-regime visualization for out-of-sample period (2019-2023).

**Usage:**
```bash
python experiments/backfill/context20/visualize_four_regime_overlay_oos.py
```

---

#### visualize_top_ci_width_moments.py
**Path:** `experiments/backfill/context20/visualize_top_ci_width_moments.py`

**Purpose:** Visualize extreme CI width moments (top 5 widest, narrowest).

**Usage:**
```bash
python experiments/backfill/context20/visualize_top_ci_width_moments.py --sampling_mode prior
```

**Output:** `results/vae_baseline/visualizations/ci_peaks/prior/top_ci_width_moments.png`

---

### VAE Health & Latent Analysis (8+ scripts)

#### analyze_vae_health_16yr.py
**Path:** `experiments/backfill/context20/analyze_vae_health_16yr.py`

**Purpose:** Comprehensive VAE health diagnostics (in-sample).

**Metrics:**
- Effective latent dimension (PCA-based)
- KL divergence per dimension
- Posterior collapse detection
- Latent variance utilization

**Usage:**
```bash
python experiments/backfill/context20/analyze_vae_health_16yr.py
```

**Output:** `results/vae_baseline/analysis/vae_health_insample.json`

---

#### analyze_vae_health_oos_16yr.py
**Path:** `experiments/backfill/context20/analyze_vae_health_oos_16yr.py`

**Purpose:** VAE health diagnostics for out-of-sample period.

**Usage:**
```bash
python experiments/backfill/context20/analyze_vae_health_oos_16yr.py
```

---

#### visualize_vae_health_16yr.py
**Path:** `experiments/backfill/context20/visualize_vae_health_16yr.py`

**Purpose:** Visualize VAE health metrics (in-sample).

**Plots:**
- Latent dimension utilization bar chart
- KL divergence by dimension
- Posterior variance distribution

**Usage:**
```bash
python experiments/backfill/context20/visualize_vae_health_16yr.py
```

**Output:** `results/vae_baseline/visualizations/vae_health_insample.png`

---

#### visualize_vae_health_oos_16yr.py
**Path:** `experiments/backfill/context20/visualize_vae_health_oos_16yr.py`

**Purpose:** Visualize VAE health metrics (OOS).

**Usage:**
```bash
python experiments/backfill/context20/visualize_vae_health_oos_16yr.py
```

---

#### analyze_latent_distributions_16yr.py
**Path:** `experiments/backfill/context20/analyze_latent_distributions_16yr.py`

**Purpose:** Analyze latent space distributions and clustering.

**Analysis:**
- Per-dimension distribution statistics
- Cross-dimension correlation
- Temporal evolution of latent means

**Usage:**
```bash
python experiments/backfill/context20/analyze_latent_distributions_16yr.py
```

---

#### analyze_latent_contribution_16yr.py
**Path:** `experiments/backfill/context20/analyze_latent_contribution_16yr.py`

**Purpose:** Assess contribution of each latent dimension to reconstruction.

**Usage:**
```bash
python experiments/backfill/context20/analyze_latent_contribution_16yr.py
```

---

#### test_dimension_ablation_16yr.py
**Path:** `experiments/backfill/context20/test_dimension_ablation_16yr.py`

**Purpose:** Ablation study - systematically remove latent dimensions and measure impact.

**Usage:**
```bash
python experiments/backfill/context20/test_dimension_ablation_16yr.py
```

---

#### test_zero_vs_prior_latent_16yr.py
**Path:** `experiments/backfill/context20/test_zero_vs_prior_latent_16yr.py`

**Purpose:** Compare deterministic (z=0, MLE) vs stochastic prior sampling.

**Usage:**
```bash
python experiments/backfill/context20/test_zero_vs_prior_latent_16yr.py
```

---

### Context60-Specific Analysis Scripts

#### analyze_latent_space.py
**Path:** `experiments/backfill/context60/analyze_latent_space.py`

**Purpose:** Comprehensive latent space analysis for context60 models.

**Usage:**
```bash
python experiments/backfill/context60/analyze_latent_space.py
```

---

#### analyze_context_length_effect.py
**Path:** `experiments/backfill/context60/analyze_context_length_effect.py`

**Purpose:** Compare context20 vs context60 performance to assess benefit of longer context.

**Usage:**
```bash
python experiments/backfill/context60/analyze_context_length_effect.py
```

---

#### analyze_oracle_prior_horizon_progression.py
**Path:** `experiments/backfill/context60/analyze_oracle_prior_horizon_progression.py`

**Purpose:** Analyze how oracle vs prior gap evolves across horizons [1, 7, 14, 30, 60, 90].

**Usage:**
```bash
python experiments/backfill/context60/analyze_oracle_prior_horizon_progression.py
```

---

#### analyze_long_horizon_convergence_v2.py
**Path:** `experiments/backfill/context60/analyze_long_horizon_convergence_v2.py`

**Purpose:** Analyze convergence behavior at long horizons (H=60, H=90).

**Usage:**
```bash
python experiments/backfill/context60/analyze_long_horizon_convergence_v2.py
```

---

#### analyze_final_epoch599_comprehensive.py
**Path:** `experiments/backfill/context60/analyze_final_epoch599_comprehensive.py`

**Purpose:** Comprehensive analysis of final trained model at epoch 599.

**Usage:**
```bash
python experiments/backfill/context60/analyze_final_epoch599_comprehensive.py
```

---

#### diagnose_kl_divergence_v2.py
**Path:** `experiments/backfill/context60/diagnose_kl_divergence_v2.py`

**Purpose:** Diagnose KL divergence issues in latent12 v2 model.

**Usage:**
```bash
python experiments/backfill/context60/diagnose_kl_divergence_v2.py
```

---

#### test_levels_vs_changes_hypothesis.py
**Path:** `experiments/backfill/context60/test_levels_vs_changes_hypothesis.py`

**Purpose:** Test hypothesis that model predicts changes better than levels.

**Usage:**
```bash
python experiments/backfill/context60/test_levels_vs_changes_hypothesis.py
```

---

#### validate_fitted_prior.py
**Path:** `experiments/backfill/context60/validate_fitted_prior.py`

**Purpose:** Validate conditional prior network (v3 model).

**Usage:**
```bash
python experiments/backfill/context60/validate_fitted_prior.py
```

---

#### test_encoder_caching_verification.py
**Path:** `experiments/backfill/context60/test_encoder_caching_verification.py`

**Purpose:** Verify encoder caching correctness for efficiency.

**Usage:**
```bash
python experiments/backfill/context60/test_encoder_caching_verification.py
```

---

### Oracle vs Prior Comparison Scripts

#### compare_oracle_vs_vae_prior_16yr.py (Context20)
**Path:** `experiments/backfill/context20/compare_oracle_vs_vae_prior_16yr.py`

**Purpose:** Statistical comparison of oracle vs prior sampling modes (context20).

**Metrics:**
- CI width ratio (prior/oracle)
- Statistical significance tests
- Horizon-specific comparisons

**Usage:**
```bash
python experiments/backfill/context20/compare_oracle_vs_vae_prior_16yr.py
```

**Key Finding:** Prior CIs are ~2-3× wider than oracle CIs, demonstrating VAE prior mismatch.

---

#### compare_oracle_vs_prior_ci.py (Context20)
**Path:** `experiments/backfill/context20/compare_oracle_vs_prior_ci.py`

**Purpose:** Comprehensive oracle vs prior CI comparison with visualizations.

**Usage:**
```bash
python experiments/backfill/context20/compare_oracle_vs_prior_ci.py
```

**Output:**
- Statistical reports
- Comparison tables
- Visualizations

---

#### compare_oracle_vs_prior_ci_context60.py
**Path:** `experiments/backfill/context60/compare_oracle_vs_prior_ci_context60.py`

**Purpose:** Oracle vs prior comparison for context60 models.

**Usage:**
```bash
python experiments/backfill/context60/compare_oracle_vs_prior_ci_context60.py
```

---

#### compare_gt_oracle_prior_percentile_bands_v2.py
**Path:** `experiments/backfill/context60/compare_gt_oracle_prior_percentile_bands_v2.py`

**Purpose:** Compare ground truth, oracle, and prior percentile bands with detailed analysis.

**Usage:**
```bash
python experiments/backfill/context60/compare_gt_oracle_prior_percentile_bands_v2.py
```

---

#### confirm_oracle_prior_ci_width_2004_2007.py
**Path:** `experiments/backfill/context60/confirm_oracle_prior_ci_width_2004_2007.py`

**Purpose:** Confirm oracle/prior CI width differences for calm period (2004-2007).

**Usage:**
```bash
python experiments/backfill/context60/confirm_oracle_prior_ci_width_2004_2007.py
```

---

### Co-integration Testing

#### test_cointegration_preservation.py
**Path:** `experiments/cointegration/test_cointegration_preservation.py`

**Purpose:** Test whether VAE preserves IV-EWMA co-integration relationships.

**Method:** Augmented Dickey-Fuller (ADF) test on residuals.

**Usage:**
```bash
python experiments/cointegration/test_cointegration_preservation.py
```

**Output:** Co-integration pass rates by period and horizon.

**Benchmark:**
- Crisis (2008-2010): 70-76% preservation target
- Normal periods: 95-100% preservation target

---

#### compile_cointegration_tables.py
**Path:** `experiments/cointegration/compile_cointegration_tables.py`

**Purpose:** Compile co-integration results into LaTeX tables.

**Usage:**
```bash
python experiments/cointegration/compile_cointegration_tables.py
```

---

### Volatility Smile Analysis

#### analyze_vol_smile_training.py
**Path:** `experiments/vol_smile/analyze_vol_smile_training.py`

**Purpose:** Analyze volatility smile preservation during training.

**Usage:**
```bash
python experiments/vol_smile/analyze_vol_smile_training.py
```

---

#### compare_vol_smile_preservation.py
**Path:** `experiments/vol_smile/compare_vol_smile_preservation.py`

**Purpose:** Compare vol smile preservation across models.

**Usage:**
```bash
python experiments/vol_smile/compare_vol_smile_preservation.py
```

---

#### compile_vol_smile_tables.py
**Path:** `experiments/vol_smile/compile_vol_smile_tables.py`

**Purpose:** Compile vol smile results into tables.

**Usage:**
```bash
python experiments/vol_smile/compile_vol_smile_tables.py
```

---

### RMSE Evaluation

#### evaluate_oracle_rmse_with_grid_stats.py
**Path:** `experiments/oracle_vs_prior/evaluate_oracle_rmse_with_grid_stats.py`

**Purpose:** Evaluate RMSE for oracle sampling with per-grid-point statistics.

**Usage:**
```bash
python experiments/oracle_vs_prior/evaluate_oracle_rmse_with_grid_stats.py
```

---

#### evaluate_oracle_oos_rmse_grid.py
**Path:** `experiments/oracle_vs_prior/evaluate_oracle_oos_rmse_grid.py`

**Purpose:** Oracle RMSE evaluation (OOS) with grid-level detail.

**Usage:**
```bash
python experiments/oracle_vs_prior/evaluate_oracle_oos_rmse_grid.py
```

---

#### evaluate_vae_prior_insample_rmse.py
**Path:** `experiments/oracle_vs_prior/evaluate_vae_prior_insample_rmse.py`

**Purpose:** VAE prior RMSE evaluation (in-sample).

**Usage:**
```bash
python experiments/oracle_vs_prior/evaluate_vae_prior_insample_rmse.py
```

---

#### evaluate_vae_prior_oos_rmse_grid.py
**Path:** `experiments/oracle_vs_prior/evaluate_vae_prior_oos_rmse_grid.py`

**Purpose:** VAE prior RMSE evaluation (OOS) with grid-level detail.

**Usage:**
```bash
python experiments/oracle_vs_prior/evaluate_vae_prior_oos_rmse_grid.py
```

---

### In-Sample vs OOS Comparisons

#### compare_insample_oos_regression.py
**Path:** `experiments/backfill/context20/compare_insample_oos_regression.py`

**Purpose:** Compare in-sample vs OOS regression performance.

**Usage:**
```bash
python experiments/backfill/context20/compare_insample_oos_regression.py
```

---

#### compare_insample_oos_distributions.py
**Path:** `experiments/backfill/context20/compare_insample_oos_distributions.py`

**Purpose:** Compare in-sample vs OOS prediction distributions.

**Usage:**
```bash
python experiments/backfill/context20/compare_insample_oos_distributions.py
```

---

## 7. Visualization Scripts

### Regime Overlay Visualizations

#### Context20 Regime Overlays

**visualize_calm_vs_crisis_overlay.py**
- Path: `experiments/backfill/context20/visualize_calm_vs_crisis_overlay.py`
- Purpose: 2-regime comparison (calm vs crisis)
- Output: `results/vae_baseline/visualizations/comparison/calm_vs_crisis_overlay_prior.png`

**visualize_three_regime_overlay.py**
- Path: `experiments/backfill/context20/visualize_three_regime_overlay.py`
- Purpose: 3-regime with pre-crisis anomaly
- Output: `results/vae_baseline/visualizations/comparison/three_regime_overlay_prior.png`

**visualize_four_regime_overlay.py** ⭐
- Path: `experiments/backfill/context20/visualize_four_regime_overlay.py`
- Purpose: Complete 2×2 regime matrix
- Output: `results/vae_baseline/visualizations/comparison/four_regime_overlay_prior.png`

**visualize_four_regime_timeline.py**
- Path: `experiments/backfill/context20/visualize_four_regime_timeline.py`
- Purpose: Timeline showing regime dates in full history
- Output: `results/vae_baseline/visualizations/comparison/four_regime_timeline_prior.png`

**visualize_four_regime_overlay_oos.py**
- Path: `experiments/backfill/context20/visualize_four_regime_overlay_oos.py`
- Purpose: 4-regime for OOS period
- Output: `results/vae_baseline/visualizations/comparison/four_regime_overlay_oos_prior.png`

---

#### Context60 Regime Overlays

**visualize_four_regime_overlay_context60.py**
- Path: `experiments/backfill/context60/visualize_four_regime_overlay_context60.py`
- Purpose: 4-regime for context60 H=30
- Usage: `python experiments/backfill/context60/visualize_four_regime_overlay_context60.py`

**visualize_four_regime_overlay_context60_h90.py**
- Path: `experiments/backfill/context60/visualize_four_regime_overlay_context60_h90.py`
- Purpose: 4-regime for context60 H=90
- Usage: `python experiments/backfill/context60/visualize_four_regime_overlay_context60_h90.py`

**visualize_four_regime_overlay_latent12v2_h30.py**
- Path: `experiments/backfill/context60/visualize_four_regime_overlay_latent12v2_h30.py`
- Purpose: 4-regime for latent12 v2 at H=30

**visualize_four_regime_overlay_latent12v2_h90.py**
- Path: `experiments/backfill/context60/visualize_four_regime_overlay_latent12v2_h90.py`
- Purpose: 4-regime for latent12 v2 at H=90

**visualize_four_regime_marginal_comparison.py**
- Path: `experiments/backfill/context60/visualize_four_regime_marginal_comparison.py`
- Purpose: Compare marginal distributions across 4 regimes

**visualize_four_regime_median_comparison.py**
- Path: `experiments/backfill/context60/visualize_four_regime_median_comparison.py`
- Purpose: Compare median forecasts across 4 regimes

---

### CI Width Temporal Visualizations

#### visualize_ci_width_temporal.py
**Path:** `experiments/backfill/context20/visualize_ci_width_temporal.py`

**Purpose:** Visualize CI width evolution over time.

**Usage:**
```bash
python experiments/backfill/context20/visualize_ci_width_temporal.py --sampling_mode prior
```

**Output:** `results/vae_baseline/visualizations/ci_width_temporal_prior.png`

---

#### visualize_sequence_ci_width.py
**Path:** `experiments/backfill/context20/visualize_sequence_ci_width.py`

**Purpose:** Visualize CI width for specific period and horizon.

**Parameters:**
- `--period`: crisis, insample, oos, gap
- `--sampling_mode`: oracle, prior

**Usage:**
```bash
python experiments/backfill/context20/visualize_sequence_ci_width.py --period insample --sampling_mode prior
```

---

#### visualize_sequence_ci_width_combined.py
**Path:** `experiments/backfill/context20/visualize_sequence_ci_width_combined.py`

**Purpose:** Combined CI width visualization across all periods.

**Usage:**
```bash
python experiments/backfill/context20/visualize_sequence_ci_width_combined.py --sampling_mode prior
```

**Output:** `results/vae_baseline/visualizations/sequence_ci_width_combined_prior.png`

---

#### visualize_ci_bands_comparison.py
**Path:** `experiments/backfill/context20/visualize_ci_bands_comparison.py`

**Purpose:** Compare CI bands (oracle vs prior).

**Usage:**
```bash
python experiments/backfill/context20/visualize_ci_bands_comparison.py
```

---

#### analyze_sequence_ci_correlations.py
**Path:** `experiments/backfill/context20/analyze_sequence_ci_correlations.py`

**Purpose:** Analyze correlations between CI width and market features.

**Usage:**
```bash
python experiments/backfill/context20/analyze_sequence_ci_correlations.py --sampling_mode prior
```

---

#### identify_ci_width_events.py
**Path:** `experiments/backfill/context20/identify_ci_width_events.py`

**Purpose:** Identify and catalog extreme CI width events.

**Usage:**
```bash
python experiments/backfill/context20/identify_ci_width_events.py --sampling_mode prior
```

---

#### compute_ci_width_timeseries.py
**Path:** `experiments/backfill/context20/compute_ci_width_timeseries.py`

**Purpose:** Compute CI width time series for all dates.

**Usage:**
```bash
python experiments/backfill/context20/compute_ci_width_timeseries.py
```

---

### Oracle vs Prior Visualizations

#### visualize_oracle_vs_prior_combined.py
**Path:** `experiments/backfill/context20/visualize_oracle_vs_prior_combined.py`

**Purpose:** Combined visualization comparing oracle and prior CI widths.

**Usage:**
```bash
python experiments/backfill/context20/visualize_oracle_vs_prior_combined.py
```

**Output:** `results/vae_baseline/visualizations/oracle_vs_prior_combined.png`

---

#### visualize_oracle_vs_prior_combined_with_vol.py
**Path:** `experiments/backfill/context20/visualize_oracle_vs_prior_combined_with_vol.py`

**Purpose:** Oracle vs prior comparison with volatility overlay.

**Usage:**
```bash
python experiments/backfill/context20/visualize_oracle_vs_prior_combined_with_vol.py
```

---

#### visualize_oracle_vs_prior_combined_timeseries_context60.py
**Path:** `experiments/backfill/context60/visualize_oracle_vs_prior_combined_timeseries_context60.py`

**Purpose:** Oracle vs prior timeseries for context60.

**Usage:**
```bash
python experiments/backfill/context60/visualize_oracle_vs_prior_combined_timeseries_context60.py
```

---

#### visualize_oracle_prior_groundtruth_2004_2007.py
**Path:** `experiments/backfill/context60/visualize_oracle_prior_groundtruth_2004_2007.py`

**Purpose:** Compare oracle, prior, and ground truth for calm period.

**Usage:**
```bash
python experiments/backfill/context60/visualize_oracle_prior_groundtruth_2004_2007.py
```

---

#### visualize_oracle_prior_multihorizon_overlay.py
**Path:** `experiments/backfill/context60/visualize_oracle_prior_multihorizon_overlay.py`

**Purpose:** Multi-horizon oracle vs prior overlay.

**Usage:**
```bash
python experiments/backfill/context60/visualize_oracle_prior_multihorizon_overlay.py
```

---

#### visualize_sequence_ci_width_evolution_2004_2007.py
**Path:** `experiments/backfill/context60/visualize_sequence_ci_width_evolution_2004_2007.py`

**Purpose:** CI width evolution for calm period (2004-2007).

**Usage:**
```bash
python experiments/backfill/context60/visualize_sequence_ci_width_evolution_2004_2007.py
```

---

### Context60-Specific Visualizations

#### visualize_ar_discontinuity_overlay.py
**Path:** `experiments/backfill/context60/visualize_ar_discontinuity_overlay.py`

**Purpose:** Visualize autoregressive discontinuity issues.

**Usage:**
```bash
python experiments/backfill/context60/visualize_ar_discontinuity_overlay.py
```

**Documentation:** See `experiments/backfill/context60/AR_DISCONTINUITY_ANALYSIS.md`

---

#### visualize_single_ar_sequence.py
**Path:** `experiments/backfill/context60/visualize_single_ar_sequence.py`

**Purpose:** Visualize single 30-day AR sequence.

**Usage:**
```bash
python experiments/backfill/context60/visualize_single_ar_sequence.py
```

---

#### visualize_ground_truth_fan_heatmap.py
**Path:** `experiments/backfill/context60/visualize_ground_truth_fan_heatmap.py`

**Purpose:** Heatmap showing ground truth quantile "fanning" pattern.

**Usage:**
```bash
python experiments/backfill/context60/visualize_ground_truth_fan_heatmap.py
```

---

#### visualize_fanning_pattern_latent12v2.py
**Path:** `experiments/backfill/context60/visualize_fanning_pattern_latent12v2.py`

**Purpose:** Visualize quantile fanning pattern for latent12 v2 model.

**Usage:**
```bash
python experiments/backfill/context60/visualize_fanning_pattern_latent12v2.py
```

---

#### visualize_fitted_prior_fanning.py
**Path:** `experiments/backfill/context60/visualize_fitted_prior_fanning.py`

**Purpose:** Visualize fanning pattern for conditional prior model (v3).

**Usage:**
```bash
python experiments/backfill/context60/visualize_fitted_prior_fanning.py
```

---

#### visualize_model_limitations.py
**Path:** `experiments/backfill/context60/visualize_model_limitations.py`

**Purpose:** Comprehensive visualization of model limitations.

**Usage:**
```bash
python experiments/backfill/context60/visualize_model_limitations.py
```

---

#### investigate_opposite_dynamics_v2.py
**Path:** `experiments/backfill/context60/investigate_opposite_dynamics_v2.py`

**Purpose:** Investigate cases where model predicts opposite dynamics from ground truth.

**Usage:**
```bash
python experiments/backfill/context60/investigate_opposite_dynamics_v2.py
```

---

### Interactive Dashboards

#### streamlit_vol_surface_viewer.py
**Path:** `streamlit_vol_surface_viewer.py` (root)

**Purpose:** Interactive Streamlit web app for comparing oracle, VAE prior, and econometric predictions.

**Features:**
- Rotatable 3D surface plots
- Period selection (crisis, insample, oos, gap)
- Date slider
- Grid point selection
- Side-by-side comparison of methods

**Usage:**
```bash
streamlit run streamlit_vol_surface_viewer.py
```

**Opens:** Web browser at `http://localhost:8501`

---

#### visualize_backfill_16yr_plotly.py
**Path:** `analysis_code/visualize_backfill_16yr_plotly.py`

**Purpose:** Interactive Plotly dashboard for context20 16yr model.

**Features:**
- Interactive 3D surfaces
- Hoverable data points
- Zoom/pan controls
- Export to HTML

**Usage:**
```bash
python analysis_code/visualize_backfill_16yr_plotly.py
```

---

#### visualize_backfill_oos_16yr_plotly.py
**Path:** `analysis_code/visualize_backfill_oos_16yr_plotly.py`

**Purpose:** Interactive Plotly dashboard for OOS period.

**Usage:**
```bash
python analysis_code/visualize_backfill_oos_16yr_plotly.py
```

---

## 8. Reusable Modules (analysis_code/)

These are general-purpose analysis modules that can be imported and used across experiments.

### Core Analysis Modules

#### regression.py
**Path:** `analysis_code/regression.py`

**Purpose:** Regression analysis utilities for latent embeddings vs market indicators.

**Functions:**
- `run_regression()`: Linear regression
- `compute_r_squared()`: R² calculation
- `cross_validation()`: K-fold CV

**Import:**
```python
from analysis_code.regression import run_regression
```

---

#### latent_pca.py
**Path:** `analysis_code/latent_pca.py`

**Purpose:** PCA analysis of latent space.

**Functions:**
- `compute_pca()`: Perform PCA on latent embeddings
- `explained_variance_ratio()`: Variance explained by components
- `latent_dimension_utilization()`: Effective dimensionality

**Import:**
```python
from analysis_code.latent_pca import compute_pca
```

---

#### arbitrage.py
**Path:** `analysis_code/arbitrage.py`

**Purpose:** Check for arbitrage violations in generated surfaces.

**Checks:**
- Calendar spread arbitrage
- Butterfly spread arbitrage
- Vertical spread arbitrage

**Import:**
```python
from analysis_code.arbitrage import check_arbitrage
```

---

#### cointegration_analysis.py
**Path:** `analysis_code/cointegration_analysis.py`

**Purpose:** Co-integration testing utilities (ADF, Johansen).

**Functions:**
- `adf_test()`: Augmented Dickey-Fuller test
- `johansen_test()`: Johansen co-integration test
- `compute_cointegration_rank()`: Estimate co-integration rank

**Import:**
```python
from analysis_code.cointegration_analysis import adf_test
```

---

#### loss_table.py
**Path:** `analysis_code/loss_table.py`

**Purpose:** Format and generate loss comparison tables for LaTeX.

**Functions:**
- `format_loss_table()`: Format losses as LaTeX table
- `compare_models()`: Generate comparison table

**Import:**
```python
from analysis_code.loss_table import format_loss_table
```

---

### Distribution Analysis

#### diagnose_distribution_shape.py
**Path:** `analysis_code/diagnose_distribution_shape.py`

**Purpose:** Distribution shape diagnostics (kurtosis, skewness).

**Functions:**
- `compute_kurtosis()`: Excess kurtosis
- `compute_skewness()`: Skewness measure
- `test_normality()`: Shapiro-Wilk test

**Import:**
```python
from analysis_code.diagnose_distribution_shape import compute_kurtosis
```

---

#### analyze_shape_vs_horizon.py
**Path:** `analysis_code/analyze_shape_vs_horizon.py`

**Purpose:** Analyze shape mismatch degradation across horizons.

**Usage:**
```bash
python analysis_code/analyze_shape_vs_horizon.py
```

---

#### create_shape_diagnostic_heatmaps.py
**Path:** `analysis_code/create_shape_diagnostic_heatmaps.py`

**Purpose:** Create 5×5 grid heatmaps of shape differences.

**Usage:**
```bash
python analysis_code/create_shape_diagnostic_heatmaps.py
```

---

#### visualize_tail_behavior.py
**Path:** `analysis_code/visualize_tail_behavior.py`

**Purpose:** Q-Q plots and tail probability analysis.

**Usage:**
```bash
python analysis_code/visualize_tail_behavior.py
```

---

### Visualization Modules

#### visualize_teacher_forcing.py
**Path:** `analysis_code/visualize_teacher_forcing.py`

**Purpose:** Visualize teacher forcing behavior (context20 models).

**Usage:**
```bash
python analysis_code/visualize_teacher_forcing.py
```

---

#### visualize_quantile_teacher_forcing.py
**Path:** `analysis_code/visualize_quantile_teacher_forcing.py`

**Purpose:** Visualize quantile decoder teacher forcing.

**Usage:**
```bash
python analysis_code/visualize_quantile_teacher_forcing.py
```

---

#### visualize_marginal_distributions_comparison.py
**Path:** `analysis_code/visualize_marginal_distributions_comparison.py`

**Purpose:** Compare marginal distributions (generated vs ground truth).

**Usage:**
```bash
python analysis_code/visualize_marginal_distributions_comparison.py
```

---

#### visualize_distribution_comparison.py
**Path:** `analysis_code/visualize_distribution_comparison.py`

**Purpose:** General distribution comparison plots.

**Usage:**
```bash
python analysis_code/visualize_distribution_comparison.py
```

---

### Quantile-Specific Modules

#### visualize_marginal_distribution_quantile_encoded.py
**Path:** `analysis_code/visualize_marginal_distribution_quantile_encoded.py`

**Purpose:** Visualize marginal distribution for quantile model with encoded latents.

**Usage:**
```bash
python analysis_code/visualize_marginal_distribution_quantile_encoded.py
```

---

#### visualize_marginal_distribution_quantile_context_only.py
**Path:** `analysis_code/visualize_marginal_distribution_quantile_context_only.py`

**Purpose:** Visualize marginal distribution for quantile model with context-only latents.

**Usage:**
```bash
python analysis_code/visualize_marginal_distribution_quantile_context_only.py
```

---

### Crisis Period Verification

#### verify_reconstruction_plotly_2008_2010.py
**Path:** `analysis_code/verify_reconstruction_plotly_2008_2010.py`

**Purpose:** Verify reconstruction quality during 2008-2010 crisis with interactive plots.

**Usage:**
```bash
python analysis_code/verify_reconstruction_plotly_2008_2010.py
```

---

#### verify_reconstruction_2008_2010_context_only.py
**Path:** `analysis_code/verify_reconstruction_2008_2010_context_only.py`

**Purpose:** Verify crisis reconstruction with context-only latents (realistic scenario).

**Usage:**
```bash
python analysis_code/verify_reconstruction_2008_2010_context_only.py
```

---

#### verify_reconstruction_with_encoded_latents.py
**Path:** `analysis_code/verify_reconstruction_with_encoded_latents.py`

**Purpose:** Verify reconstruction with encoded latents (oracle scenario).

**Usage:**
```bash
python analysis_code/verify_reconstruction_with_encoded_latents.py
```

---

### Variance & Cointegration

#### diagnose_marginal_variance.py
**Path:** `analysis_code/diagnose_marginal_variance.py`

**Purpose:** Diagnose marginal variance issues.

**Usage:**
```bash
python analysis_code/diagnose_marginal_variance.py
```

---

#### visualize_variance_ratio_heatmap.py
**Path:** `analysis_code/visualize_variance_ratio_heatmap.py`

**Purpose:** Heatmap of variance ratio (generated/ground truth).

**Usage:**
```bash
python analysis_code/visualize_variance_ratio_heatmap.py
```

---

#### visualize_multihorizon_cointegration.py
**Path:** `analysis_code/visualize_multihorizon_cointegration.py`

**Purpose:** Visualize co-integration across multiple horizons.

**Usage:**
```bash
python analysis_code/visualize_multihorizon_cointegration.py
```

---

#### visualize_preservation_and_oos.py
**Path:** `analysis_code/visualize_preservation_and_oos.py`

**Purpose:** Visualize preservation metrics and OOS comparison.

**Usage:**
```bash
python analysis_code/visualize_preservation_and_oos.py
```

---

### Crisis Grid Comparison

#### visualize_crisis_grid_comparison.py
**Path:** `analysis_code/visualize_crisis_grid_comparison.py`

**Purpose:** Grid-by-grid comparison during crisis period.

**Usage:**
```bash
python analysis_code/visualize_crisis_grid_comparison.py
```

---

### Other Analysis Tools

#### visulize_autoregressive_returns.py
**Path:** `analysis_code/visulize_autoregressive_returns.py`

**Purpose:** Visualize autoregressive return predictions.

**Usage:**
```bash
python analysis_code/visulize_autoregressive_returns.py
```

---

#### debug_histogram_mismatch.py
**Path:** `analysis_code/debug_histogram_mismatch.py`

**Purpose:** Debug histogram mismatches between generated and ground truth.

**Usage:**
```bash
python analysis_code/debug_histogram_mismatch.py
```

---

#### fit_aggregated_posterior.py
**Path:** `analysis_code/fit_aggregated_posterior.py`

**Purpose:** Fit aggregated posterior distribution for conditional prior.

**Usage:**
```bash
python analysis_code/fit_aggregated_posterior.py
```

---

## 9. Shell Scripts & Batch Runners

### Context20 Shell Scripts

#### run_generate_all_tf_sequences.sh
**Path:** `experiments/backfill/context20/run_generate_all_tf_sequences.sh`

**Purpose:** Generate TF sequences for all periods (crisis, insample, oos, gap) in batch.

**Parameters:**
- `$1`: Sampling mode (oracle or prior)

**Usage:**
```bash
bash experiments/backfill/context20/run_generate_all_tf_sequences.sh prior
bash experiments/backfill/context20/run_generate_all_tf_sequences.sh oracle
```

**Runs:**
```bash
python experiments/backfill/context20/generate_vae_tf_sequences.py --period crisis --sampling_mode $1
python experiments/backfill/context20/generate_vae_tf_sequences.py --period insample --sampling_mode $1
python experiments/backfill/context20/generate_vae_tf_sequences.py --period oos --sampling_mode $1
python experiments/backfill/context20/generate_vae_tf_sequences.py --period gap --sampling_mode $1
```

---

### Context60 Shell Scripts

#### run_generate_all_tf_sequences.sh (Context60 TF)
**Path:** `experiments/backfill/context60/teacher_forcing/run_generate_all_tf_sequences.sh`

**Purpose:** Generate context60 TF sequences for all periods.

**Usage:**
```bash
bash experiments/backfill/context60/teacher_forcing/run_generate_all_tf_sequences.sh prior
```

---

#### run_generate_all_ar_sequences.sh (Context60 AR)
**Path:** `experiments/backfill/context60/autoregressive/run_generate_all_ar_sequences.sh`

**Purpose:** Generate context60 autoregressive sequences for all periods.

**Usage:**
```bash
bash experiments/backfill/context60/autoregressive/run_generate_all_ar_sequences.sh
```

---

## 10. Appendix: Full Script Index

### By Directory Structure

#### experiments/backfill/context20/ (44 scripts)

**Training:**
- `train_backfill_model.py`

**Generation & Validation:**
- `generate_vae_tf_sequences.py`
- `run_generate_all_tf_sequences.sh`
- `validate_vae_tf_sequences.py`
- `migrate_oracle_files.py`

**Testing:**
- `test_insample_reconstruction_16yr.py`
- `test_oos_reconstruction_16yr.py`
- `test_vae_prior_insample_16yr.py`
- `test_vae_prior_oos_16yr.py`
- `test_zero_vs_prior_latent_16yr.py`
- `test_dimension_ablation_16yr.py`

**Evaluation:**
- `evaluate_insample_ci_16yr.py`
- `evaluate_vae_prior_ci_insample_16yr.py`
- `evaluate_vae_prior_ci_oos_16yr.py`
- `evaluate_rmse_16yr.py`

**CI Width Analysis (8 scripts):**
- `investigate_ci_width_peaks.py`
- `investigate_ci_width_anomaly.py`
- `compute_ci_width_timeseries.py`
- `compute_sequence_ci_width_stats.py`
- `compute_gap_ci_stats.py`
- `analyze_sequence_ci_correlations.py`
- `identify_ci_width_events.py`
- `visualize_ci_width_temporal.py`

**CI Width Visualizations:**
- `visualize_sequence_ci_width.py`
- `visualize_sequence_ci_width_combined.py`
- `visualize_ci_bands_comparison.py`
- `visualize_top_ci_width_moments.py`

**Regime Visualizations:**
- `visualize_calm_vs_crisis_overlay.py`
- `visualize_three_regime_overlay.py`
- `visualize_four_regime_overlay.py`
- `visualize_four_regime_timeline.py`
- `visualize_four_regime_overlay_oos.py`

**VAE Health:**
- `analyze_vae_health_16yr.py`
- `analyze_vae_health_oos_16yr.py`
- `visualize_vae_health_16yr.py`
- `visualize_vae_health_oos_16yr.py`

**Latent Analysis:**
- `analyze_latent_distributions_16yr.py`
- `analyze_latent_contribution_16yr.py`

**Oracle vs Prior:**
- `compare_oracle_vs_vae_prior_16yr.py`
- `compare_oracle_vs_prior_ci.py`
- `visualize_oracle_vs_prior_combined.py`
- `visualize_oracle_vs_prior_combined_with_vol.py`

**In-Sample vs OOS:**
- `compare_insample_oos_regression.py`
- `compare_insample_oos_distributions.py`

---

#### experiments/backfill/context60/ (53 scripts)

**Training:**
- `train_backfill_context60.py`
- `train_backfill_context60_resume.py`
- `train_backfill_context60_resume_phase3.py`
- `train_backfill_context60_latent12.py`
- `train_backfill_context60_latent12_v2.py`
- `train_backfill_context60_latent12_v3_conditional_prior.py`

**Generation & Validation:**
- `teacher_forcing/generate_vae_tf_sequences.py`
- `teacher_forcing/run_generate_all_tf_sequences.sh`
- `teacher_forcing/validate_vae_tf_sequences.py`
- `autoregressive/generate_vae_ar_sequences.py`
- `autoregressive/run_generate_all_ar_sequences.sh`
- `autoregressive/validate_vae_ar_sequences.py`

**CI Width:**
- `compute_sequence_ci_width_stats_context60.py`
- `compare_oracle_vs_prior_ci_context60.py`

**Visualizations:**
- `visualize_four_regime_overlay_context60.py`
- `visualize_four_regime_overlay_context60_h90.py`
- `visualize_four_regime_overlay_latent12v2_h30.py`
- `visualize_four_regime_overlay_latent12v2_h90.py`
- `visualize_four_regime_marginal_comparison.py`
- `visualize_four_regime_median_comparison.py`
- `visualize_oracle_vs_prior_combined_timeseries_context60.py`
- `visualize_oracle_prior_groundtruth_2004_2007.py`
- `visualize_oracle_prior_multihorizon_overlay.py`
- `visualize_sequence_ci_width_evolution_2004_2007.py`
- `visualize_ar_discontinuity_overlay.py`
- `visualize_single_ar_sequence.py`
- `visualize_ground_truth_fan_heatmap.py`
- `visualize_fanning_pattern_latent12v2.py`
- `visualize_fitted_prior_fanning.py`
- `visualize_model_limitations.py`

**Analysis:**
- `analyze_latent_space.py`
- `analyze_context_length_effect.py`
- `analyze_oracle_prior_horizon_progression.py`
- `analyze_long_horizon_convergence_v2.py`
- `analyze_final_epoch599_comprehensive.py`
- `analyze_quantile_growth_rates.py`
- `analyze_training_data_mean_reversion.py`
- `analyze_loss_function_bias.py`
- `analyze_decoder_horizon_sensitivity.py`
- `analyze_gt_day1_variance_by_regime.py`
- `analyze_quantile_decoder_calibration.py`
- `analyze_latent_information_bottleneck.py`
- `analyze_latent_information_bottleneck_v2.py`
- `analyze_raw_data_pca.py`

**Comparisons:**
- `compare_gt_oracle_prior_percentile_bands.py`
- `compare_gt_oracle_prior_percentile_bands_v2.py`
- `confirm_oracle_prior_ci_width_2004_2007.py`
- `investigate_opposite_dynamics.py`
- `investigate_opposite_dynamics_v2.py`

**Diagnostics:**
- `diagnose_kl_divergence.py`
- `diagnose_kl_divergence_v2.py`
- `test_levels_vs_changes_hypothesis.py`
- `validate_fitted_prior.py`
- `test_encoder_caching_verification.py`
- `compute_marginal_distribution_metrics.py`

---

#### experiments/oracle_vs_prior/ (4 scripts)

- `evaluate_vae_prior_insample_rmse.py`
- `evaluate_oracle_rmse_with_grid_stats.py`
- `evaluate_vae_prior_oos_rmse_grid.py`
- `evaluate_oracle_oos_rmse_grid.py`

---

#### experiments/cointegration/ (2 scripts)

- `test_cointegration_preservation.py`
- `compile_cointegration_tables.py`

---

#### experiments/vol_smile/ (3 scripts)

- `analyze_vol_smile_training.py`
- `compare_vol_smile_preservation.py`
- `compile_vol_smile_tables.py`

---

#### experiments/backfill/horizon5/ (4 scripts)

- `train_horizon5_test.py`
- `compare_horizon5_to_baseline.py`
- `visualize_horizon5_success.py`
- `visualize_improvement_heatmap.py`

---

#### experiments/diagnostics/ (1 script)

- `test_multi_offset_training.py`

---

#### analysis_code/ (30 scripts)

See Section 8 for complete list.

---

#### Root Scripts (10 scripts)

- `param_search.py` - Train 3 model variants
- `generate_surfaces.py` - Generate surfaces (stochastic)
- `generate_surfaces_max_likelihood.py` - Generate surfaces (MLE)
- `generate_surfaces_arb_free.py` - Generate arbitrage-free surfaces
- `generate_quantile_surfaces.py` - Generate quantile surfaces
- `train_quantile_models.py` - Train quantile models
- `main_analysis.py` - Full analysis pipeline
- `compute_grid_ci_stats.py` - CI violations per grid point
- `table_making.py` - Generate LaTeX tables
- `streamlit_vol_surface_viewer.py` - Interactive web app

---

## Summary Statistics

| Category | Count |
|----------|-------|
| **Training Scripts** | 7 |
| **Generation Scripts** | 6 |
| **Validation Scripts** | 3 |
| **CI Calibration Scripts** | 12 |
| **CI Width Analysis** | 8 |
| **VAE Health & Latent** | 8 |
| **Context60 Analysis** | 28 |
| **Regime Visualizations** | 11 |
| **Oracle vs Prior** | 8 |
| **Reusable Modules** | 30 |
| **Shell Scripts** | 3 |
| **Root Scripts** | 10 |
| **TOTAL** | **134** |

---

## Quick Reference: Common Commands

```bash
# =========================
# STANDARD TESTING WORKFLOW
# =========================

# 1. Train model
python experiments/backfill/context60/train_backfill_context60_latent12_v3_conditional_prior.py

# 2. Generate all periods (both sampling modes)
bash experiments/backfill/context60/teacher_forcing/run_generate_all_tf_sequences.sh prior
bash experiments/backfill/context60/teacher_forcing/run_generate_all_tf_sequences.sh oracle

# 3. Validate outputs
python experiments/backfill/context60/teacher_forcing/validate_vae_tf_sequences.py --sampling_mode prior

# 4. Compute CI statistics
python experiments/backfill/context60/compute_sequence_ci_width_stats_context60.py

# 5. Compare oracle vs prior
python experiments/backfill/context60/compare_oracle_vs_prior_ci_context60.py

# 6. Generate 4-regime visualization
python experiments/backfill/context60/visualize_four_regime_overlay_context60.py
python experiments/backfill/context60/visualize_four_regime_overlay_context60_h90.py

# 7. Analyze latent space
python experiments/backfill/context60/analyze_latent_space.py
```

---

**End of Experiment Inventory**
