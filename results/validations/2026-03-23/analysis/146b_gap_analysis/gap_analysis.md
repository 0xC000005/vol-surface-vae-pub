# 146b Gap Analysis — Why 4 Suites Still Fail
**Date**: 2026-03-23
**Model**: afcrps_146b (best_model.pt, epoch 17)
**Result dir**: results/block_ar/146b_best_30d/summary.json
**Score**: 5/9 suites pass {1, 3, 4, 5, 6}
**Failing**: Suites 2, 7, 8, 9

## Summary of Gaps

| Suite | What Fails | Current | Need | Gap |
|-------|-----------|---------|------|-----|
| 2 (CI Coverage) | h=1,7 horizon CI; ALL 25 cells at h=1 | h=1: 59.8%, h=7: 73.9% | 90% | 30.2pp / 16.1pp |
| 7 (Regime Coverage) | Both calm and turb, all horizons except turb@h=30 | calm h=1: 51.4% | 90% | 38.6pp worst |
| 8 (Distributional) | median_bias frac+mag; window_floor | 19/25 frac, 21/25 mag, 5.31% bad | 20/25, 22/25, <5.0% | 1 cell each; 4 fewer bad windows |
| 9 (Cross-cell Corr) | eff_rank ratio | 0.4495 | 0.50 | +0.0505 ratio; need +0.254 eff_rank |

---

## Suite 2: CI Coverage

### Pass Logic
- `horizon_pass`: all evaluated horizons must have overall CI ≥ 0.90 at the 0.9 quantile
- `worst_cell_pass`: all 25 cells at all horizons must individually achieve ≥ 0.90

### Per-Horizon Overall CI @ 90%

| Horizon | CI@0.90 | Gap to 90% | Pass? |
|---------|---------|-----------|-------|
| h=1d | 59.81% | +30.19pp | FAIL |
| h=7d | 73.92% | +16.08pp | FAIL |
| h=14d | 78.78% | +11.22pp | PASS |
| h=30d | 84.93% | +5.07pp | PASS |

Overall CI @ 0.90: **77.30%** (vs 144b: 74.01%, +3.29pp improvement).
Calibration error: **0.0619** (systematic underprediction, all quantiles compressed ~12.7pp on average).

### Worst Cell Per Horizon

| Horizon | Worst Cell CI | Location | Gap |
|---------|--------------|----------|-----|
| h=1d | 34.83% | (0,2) — OTM, short tenor | 55.17pp |
| h=7d | 43.01% | (0,3) | 46.99pp |
| h=14d | 42.11% | (0,3) | 47.89pp |
| h=30d | 58.46% | (0,3) | 31.54pp |

### Spatial Pattern of CI Failure at h=1 (ALL 25 cells fail)

The worst cells cluster in **middle tenors (T3/T4) and upper rows (M1-M3 = OTM to ATM)**:

| Cell | Coverage | Gap |
|------|----------|-----|
| (0,2) | 34.83% | 55.2pp |
| (1,2) | 35.81% | 54.2pp |
| (2,2) | 41.29% | 48.7pp |
| (2,3) | 41.46% | 48.5pp |
| (1,3) | 42.44% | 47.6pp |
| (3,3) | 44.48% | 45.5pp |
| (3,2) | 45.05% | 45.0pp |
| (0,3) | 48.81% | 41.2pp |
| ... | ... | ... (all cells fail) |
| (0,1) | 84.22% | 5.8pp (closest to passing) |

At h=7, 23/25 cells fail. Only at h=14 and h=30 do cells start passing.

### Why CI Fails

At h=1d, conditional interval width = **0.038** (from conditionality.per_horizon_conditionality["1"].cond_width). For 90% coverage at short horizons, approximately **0.08-0.09 width** is needed (based on GT spread). The model generates ~2.2x too narrow intervals at h=1.

The factor noise in 146b improved overall CI by +3.3pp but **worsened** the worst cell at h=1 (144b: 50.0% → 146b: 34.8%, a regression of 15.2pp). This is a redistribution effect: factor noise shifts variance into the dominant cross-cell direction, helping average CI but potentially concentrating more mass along a single direction and leaving some cells with less marginal variance.

### How Much to Pass Suite 2

- **h=1 overall CI**: need +30.2pp (from 59.8% to 90%)
- **h=7 overall CI**: need +16.1pp (from 73.9% to 90%)
- **Worst cell gate**: need every cell at every horizon ≥ 90%. Worst cell at h=1 needs +55.2pp — approximately **3x more interval width** than currently generated.
- The growing uncertainty structure is correct (h=1 < h=7 < h=14 < h=30 in variance, monotonic pass), but the absolute scale at short horizons is insufficient.

---

## Suite 7: Regime Coverage

### Pass Logic
- **Layer 1**: All 8 combinations of (calm/turb) × (h=1/7/14/30) must have overall coverage ≥ 0.90
- **Layer 2**: Within each regime×horizon subset, all 25 cells must also reach ≥ 0.90 (0/8 subsets pass)
- **Layer 3**: catastrophic_rate < threshold (currently 7.97%, fail)

### Layer 1: Per-Regime Per-Horizon Coverage

| Horizon | Calm | Turb | Calm Gap | Turb Gap |
|---------|------|------|----------|----------|
| h=1d | 51.4% | 62.3% | **+38.6pp** | +27.7pp |
| h=7d | 68.1% | 76.5% | +21.9pp | +13.5pp |
| h=14d | 70.0% | 82.2% | +20.0pp | +7.8pp |
| h=30d | 75.4% | **90.2%** | +14.6pp | **PASS** |

Only **1 of 8 regime×horizon combos passes**: turb @ h=30d (90.2%). Every calm combo fails; 3/4 turb combos fail.

### Turb/Calm Width Structure

| Horizon | Calm Width | Turb Width | Ratio |
|---------|-----------|-----------|-------|
| h=1d | 0.02436 | 0.05884 | **2.42x** |
| h=7d | 0.04630 | 0.09868 | 2.13x |
| h=14d | 0.06647 | 0.13674 | 2.06x |
| h=30d | 0.10258 | 0.21161 | 2.06x |

The turb/calm separation is good (ratio 2.06-2.42x across horizons, turb_calm_pass=True in Suite 3). The **relative** regime differentiation is correct; the **absolute level** is too low for both, with calm suffering more.

### Directional (Path) Bias

Both regimes show the same persistent downward bias:
- Calm: `mean_frac_median_below_gt = 0.685` (median is below GT in 68.5% of all horizon-steps)
- Turb: `mean_frac_median_below_gt = 0.690`

The model's generated median trajectory systematically runs below the realized GT. This shifts coverage: even if the interval width were sufficient, a downward-biased median reduces coverage asymmetrically (more GT values land above the upper bound than below the lower bound).

### Why Calm Fails Harder

At h=1d, calm width = 0.024 vs turb width = 0.059. For 90% coverage in calm regimes, a width of ~0.06 is needed (based on turb barely passing at h=30). Calm at h=1 is **2.5x too narrow** in absolute terms. The model has correctly learned the ratio but the calm absolute level is anchored too low.

### Layer 2 (Per-Cell by Regime)

0/8 subsets have all 25 cells passing. Best results:
- Turb h=30d: 14/25 cells pass (worst cell at (1,3): 63.3%)
- Calm h=30d: 7/25 cells pass (worst cell at (0,3): 44.1%)
- All h=1d subsets: 0/25 cells pass

### Layer 3 Catastrophic Rate

7.97% of windows have catastrophically low per-cell coverage (2,436 of 30,563 windows). Pass threshold is not exceeded; need to bring this below the threshold.

### How Much to Pass Suite 7

- **Worst single gap**: calm h=1d at +38.6pp
- **Minimum to clear Layer 1**: all 7 currently-failing combos must reach 90%
  - Calm must improve by +14.6pp minimum (h=30d) up to +38.6pp (h=1d)
  - Turb must improve by +7.8pp (h=14d) and +13.5pp (h=7d), +27.7pp (h=1d)
- Same root cause as Suite 2: insufficient absolute interval width at short horizons, plus the downward median bias
- The turb@h=30d pass proves the **mechanism is correct** — the model can do it at long horizon/high volatility. Needs generalization to calm and short horizons.

---

## Suite 8: Distributional Fidelity

### Sub-test Results

| Sub-test | Pass? | Key Metric |
|----------|-------|------------|
| KS daily changes | PASS | 21/25 cells (worst D=0.168, gate 0.15) |
| KS IV levels | PASS | 19/25 cells (worst D=0.328, gate 0.15) |
| median_bias | **FAIL** | frac: 19/25 (need 20), mag: 21/25 (need 22) |
| window_floor | **FAIL** | pct_bad=5.31% (gate: <5.0%) |
| explosion | PASS | |
| cell_mae | PASS | |

### Median Bias — Exact Failure

**Thresholds** (from `test_block_ar_requirements_v2.py`):
- `BIAS_LO = 0.30`, `BIAS_HI = 0.70` (model median must be above GT in 30%-70% of steps)
- `frac_pass`: need n_pass ≥ 20/25 cells in [0.30, 0.70]
- `mag_pass`: need n_pass ≥ 22/25 cells with |mean_bias| < 0.03 IV

**frac_pass**: 19/25 cells in range. **Failing cells** (all below 0.30, i.e., median below GT >70% of steps):

| Cell | Above-Frac | Gap to 0.30 |
|------|-----------|------------|
| (0,0) | 0.211 | **8.90pp** (OTM short-tenor — worst) |
| (1,0) | 0.234 | 6.64pp |
| (2,0) | 0.274 | 2.60pp |
| (0,1) | 0.281 | 1.92pp |
| (2,4) | 0.292 | 0.83pp |
| (0,4) | 0.298 | **0.16pp** (closest to passing) |

6 cells fail, need to fix 1 of them. Cell (0,4) is **0.16pp below threshold** — the absolute closest gap of any failing metric in the entire model. Cell (2,4) needs only +0.83pp.

**mag_pass**: 21/25 cells pass |mean_bias| < 0.03 IV. Failing cells:

| Cell | Mean Bias | Excess over gate |
|------|----------|-----------------|
| (0,0) | -0.127 IV | +0.097 IV (worst) |
| (0,4) | -0.060 IV | +0.030 IV |
| (1,4) | -0.040 IV | +0.010 IV |
| (1,0) | -0.035 IV | +0.005 IV |

4 cells fail, need to fix 2 of them. Cell (1,0) needs only 0.005 IV bias reduction.

**All biases are negative**: the model median systematically undershoots GT. The bias is concentrated in the **edges and corners of the IV surface**: short-tenor (T1) and extreme moneyness (M1). Cell (0,0) has the worst bias at -12.7 IVpts — the model underestimates short-dated OTM vol severely.

### Window Floor — Exact Failure

- `pct_bad = 5.31%` (65 bad windows out of 1,223)
- Gate: `pct_bad < 5.0%`
- Gap: only **+0.31pp** over threshold
- Need to reduce bad windows from **65 to ≤ 61** (eliminate ~4 windows)
- `worst_window_cov = 15.47%` (some windows have near-zero cross-surface coverage)
- vs 144b: 144b had 102 bad windows (8.34%); 146b improved to 65 (5.31%)

### Why 146b Broke Suite 8 (vs 144b)

**144b**: frac_pass=True (24/25), mag_pass=True, median_bias overall=PASS
**146b**: frac_pass=False (19/25), mag_pass=False, median_bias overall=FAIL

The factor noise addition caused a **regression** in median accuracy. Mechanistically: factor noise injects structured variance across cells simultaneously. If the dominant factor direction has any correlation with the direction of forecast error (downward bias), factor noise amplifies the bias effect. The variance budget was redistributed from symmetric noise (equal chance of overshooting/undershooting) toward structured noise aligned with a specific direction.

### How Much to Pass Suite 8

- **frac_pass**: need 1 more cell to cross 0.30. Cell (0,4) is 0.16pp away. Cell (2,4) is 0.83pp away. **Essentially a noise-floor level gap**.
- **mag_pass**: need 1 more cell. Cell (1,0) is 0.005 IV over the gate — a trivially small improvement.
- **window_floor**: need 4 fewer bad windows out of 65 (6.2% reduction in bad windows).
- **Core issue**: the systematic downward bias in the median. Reducing this requires either (a) adding a small positive drift term, (b) adjusting the loss function to penalize median underprediction, or (c) reversing the factor noise direction.

---

## Suite 9: Cross-Cell Correlation

### Metrics

| Metric | GT | Gen (146b) | Ratio | Threshold | Pass? |
|--------|-----|-----------|-------|-----------|-------|
| eff_rank | 5.029 | 2.260 | 0.4495 | ≥ 0.50 | **FAIL** |
| mean_corr | 0.510 | 0.583 | 1.143 | in range | PASS |
| PC1 variance | 60.9% | 83.6% | — | — | — |
| Frobenius dist | — | 16.30 | — | — | — |

The model generates covariance structures where **83.6% of variance is in PC1** vs GT's 60.9%. Near rank-1 behavior persists despite the factor noise improvement.

### Gap to Pass

| Quantity | Value |
|----------|-------|
| gt_eff_rank | 5.029 |
| gen_eff_rank (current) | 2.260 |
| rank_ratio (current) | **0.4495** |
| rank_ratio threshold | 0.50 |
| Gap in ratio | **+0.0505** |
| Required gen_eff_rank | **2.514** |
| Required improvement | **+0.254 eff_rank** (+11.2% relative) |

### Progress and Projection

| Model | gen_eff_rank | rank_ratio | Delta from prev |
|-------|-------------|-----------|----------------|
| 144b | 1.471 | 0.2926 | baseline |
| 146b | 2.260 | 0.4495 | **+0.789** (+53.7%) |
| **Target** | **2.514** | **0.500** | **+0.254** needed |

146b achieved +0.789 eff_rank improvement via factor noise. To pass, only **+0.254 more is needed — 32.2% of the 144b→146b gain**. This is the most tractable gap of all 4 failing suites.

### Why eff_rank is Still Below Threshold

Three compounding mechanisms:
1. **CLN rank-1 bottleneck** (bottleneck 1 from MEMORY): The conditioning layer normalization produces near-identical modulation across all 25 spatial tokens (Jacobian 1.23). This forces all cells to move together.
2. **Skip bypass disabled** (146b has ar_noise_skip=False, ar_skip_bypass_spread=False): All cell-independent noise must route through CLN, which compresses it back toward rank-1.
3. **AR temporal autocorrelation (rho=0.8)**: Early-step noise patterns propagate through all 30 frames. If step-1 noise is near rank-1 (due to CLN), all subsequent steps inherit it.

The factor noise partially overcomes (1) by injecting diversity before CLN, but without skip bypass, the CLN still modulates everything together.

---

## Cross-Suite Root Cause Analysis

### Root Cause A: Insufficient Interval Width at Short Horizons (Suites 2, 7)
At h=1d, conditional interval width = 0.038. Coverage requires ~0.08-0.09 (2.1-2.4x more).
The noise process variance grows correctly with horizon (monotonic, Suite 5 passes), but the **absolute magnitude** at h=1 is anchored too low. The model needs horizon-adaptive noise scaling.

### Root Cause B: Systematic Downward Bias in Median Trajectory (Suites 7, 8)
- `mean_frac_median_below_gt ≈ 0.69` for both regimes
- All 25 cells have negative mean_bias
- Bias concentrated in: short-tenor (col 0), long-tenor extremes (col 4), OTM row (row 0)
- **146b introduced this failure** vs 144b (which passed bias tests). Factor noise redistributed variance in a way that amplified the directional bias.
- In Suite 8, the model is 0.16pp from passing on frac_pass and 4 windows from passing window_floor — these are near-miss gaps.

### Root Cause C: Near Rank-1 Factor Structure (Suite 9, contributing to Suites 2/7)
- gen_pc1_var = 83.6% vs GT's 60.9%
- CLN conditioning collapses cross-cell diversity
- 11.2% more eff_rank needed — smallest proportional gap among all failing suites

---

## Prioritized Fix Plan

### Priority 1: Suite 9 (11.2% eff_rank gain needed — smallest gap)
- Need gen_eff_rank from 2.260 to 2.514 (+0.254)
- Options:
  - Enable skip bypass (H1 in RC11): routes noise directly to cells, bypassing CLN
  - More factor components in 146b's factor noise
  - Stronger noise injection before CLN compression

### Priority 2: Suite 8 Near-Misses (regression from 144b factor noise)
- frac_pass: 1 cell (0.16pp from threshold at cell (0,4))
- mag_pass: 1 cell (0.005 IV from threshold at cell (1,0))
- window_floor: 4 fewer bad windows
- **Options**: Reduce factor noise weight to partially restore 144b's median accuracy, or add a small positive bias correction to short-tenor/OTM cells.

### Priority 3: Suites 2 + 7 (require ~2-3x interval width at h=1, structural change)
- These need horizon-adaptive noise amplitude (H2: heteroscedastic decoder, or H3: condition-dependent noise scale)
- The turb@h=30d passing in Suite 7 confirms the mechanism is correct at long range
- The failing at h=1 is a fundamental scale problem, not a direction problem

---

## Numerical Targets for Passing

| Suite | Metric | Current | Target | Absolute Gap |
|-------|--------|---------|--------|-------------|
| 2 | h=1 overall CI@0.9 | 59.81% | 90.00% | +30.19pp |
| 2 | h=7 overall CI@0.9 | 73.92% | 90.00% | +16.08pp |
| 2 | Worst cell h=1 | 34.83% | 90.00% | +55.17pp |
| 7 | Calm h=1 coverage | 51.40% | 90.00% | +38.60pp |
| 7 | Calm h=7 coverage | 68.13% | 90.00% | +21.87pp |
| 7 | Turb h=1 coverage | 62.33% | 90.00% | +27.67pp |
| 8 | frac_pass cells | 19/25 | 20/25 | +1 cell (0.16pp for (0,4)) |
| 8 | mag_pass cells | 21/25 | 22/25 | +1 cell (0.005 IV for (1,0)) |
| 8 | window_floor pct_bad | 5.31% | <5.00% | 4 fewer bad windows |
| 9 | rank_ratio | 0.4495 | 0.5000 | +0.0505 |
| 9 | gen_eff_rank | 2.260 | 2.514 | +0.254 (+11.2%) |
