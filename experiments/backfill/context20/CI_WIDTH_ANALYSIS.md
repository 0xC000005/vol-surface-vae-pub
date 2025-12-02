# CI Width Analysis & Visualization Suite

Comprehensive investigation and visualization tools for analyzing CVAE confidence interval behavior across market regimes, answering: **Why does the VAE widen confidence intervals?**

## Overview

8 scripts organized into three categories:
1. **Investigation Scripts (2)**: Systematic regression analysis
2. **Regime Overlays (4)**: Demonstrating conditional generation capability
3. **Timeline & Extremes (2)**: Providing temporal context

## Investigation Scripts

### investigate_ci_width_anomaly.py

Deep investigation of "high CI despite low vol" anomaly in pre-crisis period (2006-2008).

**Usage:**
```bash
python experiments/backfill/context20/investigate_ci_width_anomaly.py
```

**Features:**
- 5-phase analysis pipeline (regime identification, feature comparison, temporal evolution, visualization, reporting)
- Regression analysis of CI width vs features
- Identifies when model widens CIs based on surface shape, not just ATM volatility

**Outputs:** `results/vae_baseline/analysis/ci_peaks/prior/anomaly_investigation/`
- CSV files: regime_identification.csv, feature_comparison_by_regime.csv, regression_by_regime.csv, rolling_window_features.csv
- Visualizations: 5 PNG files (timeseries, heatmap, feature importance, drivers, scatter)
- Report: CI_WIDTH_ANOMALY_REPORT.md

### investigate_ci_width_peaks.py

Systematic peak identification and regression decomposition of temporal vs spatial feature contributions.

**Usage:**
```bash
python experiments/backfill/context20/investigate_ci_width_peaks.py --sampling_mode prior --percentile_threshold 90
```

**Parameters:**
- `--sampling_mode`: oracle or prior (default: prior)
- `--percentile_threshold`: 90 for top 10%, 95 for top 5% (default: 90)

**Features:**
- Temporal features: abs_returns, realized_vol_30d
- Spatial features: atm_vol, slopes, skews
- Regression decomposition showing 68-75% spatial dominance

**Outputs:**
- Analysis: `results/vae_baseline/analysis/ci_peaks/{oracle|prior}/`
- Visualizations: `results/vae_baseline/visualizations/ci_peaks/{oracle|prior}/`
- Report: CI_PEAKS_INVESTIGATION_REPORT.md

## Regime Comparison Visualizations

All use `prior` sampling for realistic deployment scenarios. Progressive narrative from simple to complete.

### visualize_calm_vs_crisis_overlay.py

Simple 2-regime comparison showing same model produces different CI widths based on context.

**Regimes:**
- 2007-03-28 (calm, narrow CI) - **BLUE**
- 2008-10-30 (crisis, wide CI) - **RED**

**Output:** `results/vae_baseline/visualizations/comparison/calm_vs_crisis_overlay_prior.png`

### visualize_three_regime_overlay.py

3-regime comparison emphasizing anomalous pre-crisis detection pattern.

**Regimes:**
- 2007-03-28 (normal/calm, narrow CI) - **GREEN**
- 2007-10-09 (low-vol high-CI anomaly, pre-crisis) - **ORANGE** ← Key finding
- 2008-10-30 (high-vol high-CI crisis, widest CI) - **RED**

**Output:** `results/vae_baseline/visualizations/comparison/three_regime_overlay_prior.png`

### visualize_four_regime_overlay.py

Complete 2×2 regime matrix showing intelligent pattern recognition beyond volatility levels.

**Regimes (2×2 matrix):**
- **Green** (Low Vol, Low CI): 2007-03-28 - Normal baseline
- **Blue** (High Vol, Low CI): 2009-02-23 - **Intelligent confidence** despite high vol
- **Orange** (Low Vol, High CI): 2007-10-09 - **Pre-crisis detection** anomaly
- **Red** (High Vol, High CI): 2008-10-30 - Expected crisis behavior

**Key Insights:**
- Blue and Red both have high context volatility but Blue CI is ~70% narrower
- Demonstrates model recognizes familiar patterns → confident even in volatile conditions
- Orange shows model detects structural risk beyond ATM vol

**Output:** `results/vae_baseline/visualizations/comparison/four_regime_overlay_prior.png`

### visualize_four_regime_timeline.py

Multi-panel timeline showing where the 4 regime dates appear in full 2000-2023 history.

**Features:**
- 5 panels: CI Width, ATM Vol, Slopes, Realized Vol 30d, Skews
- Full period coverage with 4 regime dates highlighted
- Z-score computation for each feature at each regime date

**Outputs:**
- Timeline: `results/vae_baseline/visualizations/comparison/four_regime_timeline.png`
- Summary: `results/vae_baseline/visualizations/comparison/four_regime_feature_summary.csv`

## Oracle vs Prior & Extremes

### visualize_oracle_vs_prior_combined_with_vol.py

5-panel comparison of oracle vs prior sampling across all horizons with actual volatility reference.

**Panels:**
- Panels 1-4: H=1, 7, 14, 30 CI width comparison
- Panel 5: Actual ATM 6M implied volatility for market context

**Key Findings:**
- Prior CIs are ~2-3× wider than oracle CIs across all horizons
- Demonstrates VAE prior mismatch: p(z|context) ≠ p(z|context+target)

**Output:** `results/vae_baseline/visualizations/comparison/oracle_vs_prior_combined_timeseries_with_vol_KS1.00_6M.png`

### visualize_top_ci_width_moments.py

Teacher forcing visualization for top 5 widest/narrowest CI moments with full 20-day context + 30-day forecast.

**Usage:**
```bash
# All combinations
python experiments/backfill/context20/visualize_top_ci_width_moments.py --sampling_mode both --ci_type both

# Specific combination
python experiments/backfill/context20/visualize_top_ci_width_moments.py --sampling_mode prior --ci_type widest
```

**Parameters:**
- `--sampling_mode`: oracle, prior, or both
- `--ci_type`: widest, narrowest, or both

**Features:**
- 5-row stacked figures for each combination
- Each row: context line + forecast region + CI band + violations marked
- Metrics: RMSE, CI violations count/%, mean CI width

**Outputs:** `results/vae_baseline/visualizations/top_ci_width_moments/{sampling_mode}_top5_{widest|narrowest}_ci_h30.png`

## Key Findings Summary

1. **Spatial dominance**: Spatial features (ATM vol, slopes, skews) explain 68-75% of CI width variation vs lower % from temporal features

2. **Pre-crisis detection**: Model widens CIs based on surface shape anomalies, not just ATM volatility levels

3. **Intelligent pattern recognition**: Complete 2×2 regime matrix validates:
   - Blue (High Vol, Low CI): Confident in familiar patterns
   - Orange (Low Vol, High CI): Detects structural risk pre-crisis
   - Red (High Vol, High CI): Expected crisis behavior

4. **VAE prior mismatch**: Prior CIs are ~2-3× wider than oracle (all p < 0.001)

5. **OOS feature importance shifts**: Period-stratified analysis reveals spatial dominance weakens in out-of-sample data (1.33× → 1.13×, -15% at H=30) while feature distributions shift significantly (1.5-1.9× variance increase). Explains why CI violations increase 55% in OOS (18% → 28%) - see detailed analysis below.

## Related Documentation

- **Period comparison analysis:** `results/vae_baseline/analysis/period_comparison/INSAMPLE_VS_OOS_COMPARISON.md` - Comprehensive investigation of why OOS CI violations increase instead of model widening CIs
- Main project docs: `CLAUDE.md` (VAE Sampling Strategies section)
- Quantile regression: `experiments/backfill/QUANTILE_REGRESSION.md`
- Context20 workflow: `experiments/backfill/context20/README.md`
