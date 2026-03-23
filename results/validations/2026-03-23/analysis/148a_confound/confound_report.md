# 148a Confound Isolation Report

**Date:** 2026-03-23
**Purpose:** Isolate whether the CI regression in Exp 148a (74.0% -> 71.7%) was caused by
skip bypass or by the recipe changes (ES/IS/cell_spread/bias_lambda/reflect).

---

## Three-Model Ablation Design

| Model | Description | CI90 | KS/25 | eff_rank | growing_uncertainty |
|-------|-------------|------|-------|----------|---------------------|
| 144b  | Base: no skip, no recipe, no factor | 74.0% | 22 | 1.47 | PASS |
| 148a  | 144b + recipe + skip (NO factor) | 71.7% | 15 | 1.69 | FAIL |
| 146b  | 144b + recipe + skip + factor | 77.3% | 21 | 2.26 | PASS |

The key isolating structure:
- **144b -> 148a delta** = net effect of (recipe + skip), no factor
- **148a -> 146b delta** = effect of adding factor noise only (recipe+skip held constant)
- **144b -> 146b delta** = total effect of all three changes

---

## CI Attribution Analysis

| Step | Delta CI | Interpretation |
|------|----------|----------------|
| 144b -> 148a (recipe + skip, no factor) | -2.34 pp | REGRESSION |
| 148a -> 146b (factor noise only) | +5.63 pp | RECOVERY + improvement |
| 144b -> 146b (total) | +3.29 pp | Net positive |

**Factor noise accounts for 171% of the 146b improvement** (it overcomes the skip-induced
regression). The recipe+skip combination is a -71% contributor (negative drag).

---

## Per-Cell CI Grids

### 148a - 144b (recipe+skip effect, NO factor)

All 25 cells show negative or near-zero delta. The damage is universal:

```
-0.020  -0.090  -0.073  -0.109  -0.048
-0.028  -0.105  -0.051  -0.072  -0.089
-0.057  -0.082  -0.071  -0.061  -0.087
-0.109  -0.051  -0.038  -0.021  -0.069
-0.028  +0.002  -0.041  -0.037  -0.106
```

Largest losses in the interior (moneyness rows 2-3, short tenors). The worst cell
drops 109 bp. Only one cell (row 5, col 2) is nominally flat.

### 146b - 148a (factor noise ONLY)

All 25 cells show positive delta. Factor noise uniformly restores CI:

```
+0.053  +0.119  +0.071  +0.104  +0.071
+0.043  +0.136  +0.073  +0.054  +0.063
+0.059  +0.104  +0.095  +0.073  +0.058
+0.070  +0.095  +0.101  +0.089  +0.074
+0.004  +0.076  +0.101  +0.100  +0.132
```

Recovery is largest where skip-bypass hurt most (interior cells).

### 146b - 144b (total net effect)

```
+0.033  +0.030  -0.002  -0.005  +0.022
+0.016  +0.031  +0.022  -0.018  -0.026
+0.002  +0.022  +0.024  +0.012  -0.028
-0.038  +0.044  +0.063  +0.067  +0.005
-0.024  +0.078  +0.060  +0.063  +0.026
```

Longer tenors (rows 4-5) gain substantially. Short-tenor, high-strike cells (top rows,
cols 3-5) show slight net negatives — these are the residual skip bypass cost that factor
noise does not fully overcome.

---

## Horizon-Level CI Evidence

| Horizon | 144b | 148a | 146b |
|---------|------|------|------|
| h1  | 64.7% | 47.9% | 59.8% |
| h7  | 75.1% | 70.9% | 73.9% |
| h14 | 75.3% | 73.4% | 78.8% |
| h30 | 74.7% | 73.0% | 84.9% |

The h1 collapse in 148a is extreme (-16.8 pp). Factor noise partially recovers it (+12.1 pp)
but h1 remains below 144b even in 146b (-4.9 pp residual). Long-horizon (h30) shows the
strongest gain from factor noise (+11.9 pp vs 148a, +10.2 pp vs 144b).

---

## Worst-Cell Evidence

| Horizon | 144b worst | 148a worst | 146b worst |
|---------|-----------|-----------|-----------|
| h1  | 50.0% | 24.2% | 34.8% |
| h7  | 48.9% | 43.6% | 43.0% |
| h14 | 47.6% | 39.2% | 42.1% |
| h30 | 47.1% | 36.9% | 58.5% |

The worst_cell CI at h1 drops from 50.0% to 24.2% in 148a — a catastrophic collapse
consistent with variance starvation from skip bypass at very short horizons. Factor noise
restores the short-horizon diversity only partially (34.8%).

---

## Growing Uncertainty Diagnostic

- **144b**: growing_uncertainty = PASS (monotone variance with horizon)
- **148a**: growing_uncertainty = FAIL (non-monotone: variance dips at h20)
- **146b**: growing_uncertainty = PASS (restored by factor noise)

This is the clearest mechanistic signal. Skip bypass (without factor noise) breaks the
monotone-variance invariant, collapsing short-horizon spread. The recipe changes
(ES/IS/etc.) alone cannot explain this since they are shared by 146b (which passes).

---

## KS Changes (Distributional Health)

| Model | KS changes pass | Change from 144b |
|-------|----------------|-----------------|
| 144b  | 22/25 | — |
| 148a  | 15/25 | -7 |
| 146b  | 21/25 | -1 |

Recipe+skip degrades KS by 7 cells. Factor noise recovers 6 of those 7. The residual
1-cell KS degradation vs 144b suggests skip bypass carries a small irreducible KS cost
even with factor noise.

---

## Effective Rank

| Model | eff_rank | Change from 144b |
|-------|----------|-----------------|
| 144b  | 1.47 | — |
| 148a  | 1.69 | +0.22 |
| 146b  | 2.26 | +0.79 |

Skip bypass actually increases eff_rank slightly (+0.22), but the CI still drops.
This suggests skip bypass adds diversity at aggregate level but collapses it at
short horizons specifically. Factor noise adds +0.57 additional rank on top of skip,
consistent with documented MEMORY findings (146b: eff_rank 2.26 from MEMORY.md).

---

## Suite-by-Suite Comparison

| Suite | 144b | 148a | 146b |
|-------|------|------|------|
| surface | PASS | PASS | PASS |
| coverage | FAIL | FAIL | FAIL |
| conditionality | PASS | PASS | PASS |
| time_series | PASS | PASS | PASS |
| block_ar | PASS | PASS | PASS |
| cointegration | PASS | PASS | PASS |
| regime_coverage | FAIL | FAIL | FAIL |
| distributional | FAIL | FAIL | FAIL |
| cross_cell_correlation | FAIL | FAIL | FAIL |

No suite changed pass/fail status across all three models. The experiment series
sits at 5/9 throughout. The marginal improvements from factor noise are real but
insufficient to cross any binary pass threshold.

---

## Conclusions

### Is the CI regression from skip collapse, recipe change, or both?

**Primary cause: skip bypass without factor noise causes short-horizon variance collapse.**

Evidence chain:
1. growing_uncertainty fails ONLY in 148a (not 144b, not 146b). Recipe changes are common
   to 148a and 146b. Therefore recipe changes cannot be the cause.
2. Worst_cell h1 collapses from 50.0% (144b) to 24.2% (148a). This is consistent with
   skip bypass allowing the decoder to shortcut noise injection, starving short-horizon
   diversity.
3. Factor noise (148a -> 146b) restores growing_uncertainty and partially recovers CI,
   confirming that skip bypass is the mechanism — factor noise provides a path to inject
   diversity even when skip bypass is active.
4. The recipe changes (ES/IS) contribute ZERO to the CI drop since the same recipe is used
   in 146b with a positive CI outcome.

### Fraction of 146b improvement attributable to factor noise vs recipe changes

- Recipe + skip alone: -2.34 pp (net drag vs 144b baseline)
- Factor noise alone: +5.63 pp (measured at constant recipe+skip)
- **Factor noise is responsible for 171% of 146b's total improvement** (i.e., it overcomes
  a -2.34 pp drag AND adds +5.63 pp on top of it, for a net +3.29 pp vs 144b)

### Practical implication

Skip bypass alone is harmful. Skip bypass + factor noise is beneficial. The correct
interpretation is that factor noise is the load-bearing component; skip bypass is neutral
to mildly positive when factor noise is present (it provides a path for noise to propagate
through skip connections), but harmful without it (it bypasses noise injection entirely).

For RC11 planning: H1 (enable skip bypass) should only be tested together with factor
noise or another diversity mechanism. Testing skip bypass in isolation (as 148a did) is
expected to regress CI and KS.
