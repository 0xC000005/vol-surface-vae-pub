# 233a Regime Breakdown Diagnostic

Purely analytical pass over existing suite.json files. Per-regime data sourced from `conditionality.per_regime_conditionality` (calm/turb). Coverage and distributional fidelity are **not** regime-stratified in the JSON; only aggregate values are reported for those suites.

## Table 1: Per-Run Overview (all 9 variants × seeds)

| Run | Pass | TC_ratio | calm_wr | turb_wr | WR_gap | calm_MAE% | turb_MAE% | wc_h30_cov | KS_chg | KS_lvl | MR_ratio | Jump_KS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_s42                       | 2/7 | 1.010 | 1.223 | 0.793 | -0.430 | -28.1% | +26.1% | 0.385 | 13/25 | 8/25 | 0.794 | 0.735 |
| full_s1337                     | 2/7 | 0.931 | 1.198 | 0.788 | -0.410 | -15.5% | +15.0% | 0.208 | 15/25 | 10/25 | 0.814 | 0.898 |
| full_s2024                     | 2/7 | 0.986 | 1.216 | 0.783 | -0.434 | -27.0% | +26.2% | 0.245 | 12/25 | 7/25 | 0.925 | 0.838 |
| B_s42                          | 2/7 | 1.006 | 1.213 | 0.800 | -0.414 | -24.2% | +18.5% | 0.245 | 9/25 | 12/25 | 0.901 | 0.845 |
| B_s1337                        | 2/7 | 1.044 | 1.219 | 0.804 | -0.415 | -32.5% | +25.6% | 0.370 | 16/25 | 13/25 | 0.823 | 0.740 |
| B_s2024                        | 1/7 | 1.000 | 1.220 | 0.792 | -0.428 | -14.0% | +11.5% | 0.156 | 15/25 | 5/25 | 0.815 | 0.799 |
| C_s42                          | 2/7 | 1.029 | 1.211 | 0.810 | -0.401 | -21.6% | +15.0% | 0.323 | 15/25 | 9/25 | 0.799 | 0.953 |
| C_s1337                        | 2/7 | 0.971 | 1.225 | 0.783 | -0.443 | -30.6% | +24.1% | 0.453 | 13/25 | 12/25 | 0.825 | 0.928 |
| C_s2024                        | 2/7 | 0.994 | 1.208 | 0.782 | -0.427 | -38.8% | +29.8% | 0.312 | 16/25 | 10/25 | 0.750 | 0.848 |
| _baseline_229a_newproxy        | 3/7 | 1.025 | 1.117 | 0.871 | -0.246 | -3.9% | +24.2% | 0.271 | 19/25 | 6/25 | 1.318 | 0.945 |

_TC_ratio=turb_calm_ratio (gate: >1.15), calm/turb_wr=avg_width_ratio (>1 means wider than unconditional), WR_gap=turb_wr - calm_wr, calm/turb_MAE%=avg_mae_reduction_pct (positive=improved), wc_h30_cov=worst-cell coverage at h=30 (gate: ≥0.90), KS_chg/lvl=cells passing KS gate (gate: 25/25), MR_ratio=gen/gt mean-reversion slope ratio (gate: ≥0.5), Jump_KS=pathwise max-jump KS stat (gate: <0.20)._

## Table 2: Per-Variant Mean (averaged over 3 seeds)

| Variant | mean_pass | TC_ratio | calm_wr | turb_wr | WR_gap | calm_MAE% | turb_MAE% | wc_h30_cov | KS_chg | MR_ratio | Jump_KS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full    | 2.0/7 | 0.976 | 1.212 | 0.788 | -0.425 | -23.6% | +22.4% | 0.280 | 13.3/25 | 0.844 | 0.824 |
| B       | 1.7/7 | 1.016 | 1.218 | 0.799 | -0.419 | -23.6% | +18.5% | 0.257 | 13.3/25 | 0.847 | 0.795 |
| C       | 2.0/7 | 0.998 | 1.215 | 0.791 | -0.424 | -30.3% | +23.0% | 0.363 | 14.7/25 | 0.791 | 0.910 |
| **229a_base** | 3/7 | 1.025 | 1.117 | 0.871 | -0.246 | -3.9% | +24.2% | 0.271 | 19/25 | 1.318 | 0.945 |

## Table 3: 233a vs 229a Baseline — Regime Gap Delta

How much worse is 233a at each regime metric compared to 229a? Positive = 233a *better* than 229a on that metric.

| Metric | 229a | 233a_full | 233a_B | 233a_C | Best_variant |
|---|---|---|---|---|---|
| TC_ratio | 1.025 | 0.976 | 1.016 | 0.998 | B |
| calm_wr | 1.117 | 1.212 | 1.218 | 1.215 | B |
| turb_wr | 0.871 | 0.788 | 0.799 | 0.791 | B |
| calm_MAE% | -3.9 | -23.6 | -23.6 | -30.3 | full |
| turb_MAE% | +24.2 | +22.4 | +18.5 | +23.0 | C |
| wc_h30_cov | 0.271 | 0.280 | 0.257 | 0.363 | C |
| MR_ratio | 1.318 | 0.844 | 0.847 | 0.791 | B |
| Jump_KS | 0.945 | 0.824 | 0.795 | 0.910 | C |
| KS_chg_n | 19 | 13 | 13 | 15 | C |

## Table 4: Per-Cell Width Ratio Heatmaps (full_s42 example)

Width ratio > 1.0 means the model generates WIDER intervals in this regime than the unconditional baseline. Target: calm < 1 (narrower when calm), turb > 1 (wider when turb).

### CALM regime — per-cell width ratio (full_s42)

| row\col | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 0.852 | **1.239** | **1.385** | 0.510 | 0.961 |
| 1 | **1.071** | **1.343** | **1.371** | 0.888 | **1.306** |
| 2 | **1.420** | **1.437** | **1.464** | **1.405** | 0.587 |
| 3 | **1.461** | **1.497** | **1.407** | **1.414** | **1.181** |
| 4 | **1.331** | **1.353** | **1.279** | **1.290** | **1.126** |

avg_wr=1.223, worst_cell_wr=1.497, avg_mae=-28.1%, n_windows=39

### TURB regime — per-cell width ratio (full_s42)

| row\col | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | **1.187** | 0.907 | 0.665 | **1.217** | **1.004** |
| 1 | 0.893 | 0.825 | 0.621 | 1.000 | 0.867 |
| 2 | 0.908 | 0.716 | 0.584 | 0.618 | **1.049** |
| 3 | 0.603 | 0.688 | 0.640 | 0.606 | 0.699 |
| 4 | 0.912 | 0.666 | 0.700 | 0.618 | 0.629 |

avg_wr=0.793, worst_cell_wr=1.217, avg_mae=26.1%, n_windows=39

## Conclusions

### Dominant Failure Regime: **CALM**

Calm regime is systematically over-dispersed: model generates intervals that are too WIDE in calm periods (calm_avg_wr=1.215 > 1.0), causing avg MAE to INCREASE by 25.8% vs unconditional. This pattern holds in 9/9 runs.

Turb regime is correct in direction: turb_avg_wr=0.792 (below 1 = under-wide), MAE reduces by 21.3% on average. However turb_avg_wr < 1 means turb intervals are actually NARROWER than unconditional, which is backwards. Correct behavior would be turb_avg_wr > 1 AND positive MAE reduction.

### Combined Failure Pattern

Both regimes are wrong but in OPPOSITE directions: calm is too wide (overdispersed), turb is too narrow (underdispersed). This is classic FiLM-collapse with sign inversion: the conditioning signal is received but INVERTED or uncalibrated — the model widens when it should narrow and narrows when it should widen.

### FiLM-Collapse Hypothesis Assessment

SUPPORTED by evidence. 9/9 runs show calm_avg_wr > 1.0 (over-wide calm). 9/9 runs show turb_avg_wr < 1.0 (under-wide turb). FiLM signals regime correctly but gate/shift miscalibration inverts the width effect. turb_calm_ratio ≈ 1.0 for all 233a variants confirms near-zero NET differentiation.

### 233a vs 229a Baseline

233a vs 229a baseline: calm MAE worsened by -21.9 pct-pts (229a was -3.9%), turb MAE changed by -2.9 pct-pts (229a was 24.2%). The 233a loss vs 229a is primarily driven by calm-regime degradation.

- 229a: calm_MAE=-3.9%, turb_MAE=24.2%
- 233a: calm_MAE=-25.8%, turb_MAE=21.3%
- Delta: calm -21.9 ppt, turb -2.9 ppt

### Cross-Variant Comparison

- Best at calm regime (least overdispersion): **full**
- Best at turb regime (highest MAE reduction): **C**

  - **full**: calm_MAE=-23.6%, turb_MAE=22.4%, TC_ratio=0.976
  - **B**: calm_MAE=-23.6%, turb_MAE=18.5%, TC_ratio=1.016
  - **C**: calm_MAE=-30.3%, turb_MAE=23.0%, TC_ratio=0.998

### Coverage and Other Suites (Aggregate, Not Regime-Stratified)

Coverage is not regime-stratified in suite.json — only aggregate per-horizon stats available. Worst-cell h=30 coverage averaged across variants: full=0.280, B=0.257, C=0.363. All far below 0.90 gate, consistent with global underdispersion.

Across all 233a variants: MR_ratio 0.876 (gate ≥0.5), Jump_KS 0.853 (gate <0.20). Both failures are regime-agnostic (global underdispersion and slow noise process).

---
_Generated by diagnose_233a_regime_breakdown.py_
