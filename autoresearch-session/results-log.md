# Autoresearch Session — 2026-03-17

**Goal**: 8/8 test suites on raw model (Bitter Lesson, no post-hoc)
**Starting model**: 99m_v2 (5/8 PASS: suites 1,3,4,5,6)
**Branch**: autoresearch-session-20260317

## Iteration Log

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 0 | 99m_v2 | baseline  | 66.31 | 5/8    | BASELINE |
| 1 | 102a   | B: noise_scale_cond | 65.73 | 5/8 | VALUABLE FAILURE |
| 2 | 103a   | A: learned rho      | 65.47 | 5/8 | VALUABLE FAILURE |
| 3 | 103a_v2| A: clamped rho 0.6-0.95 | 66.20 | 5/8 | VALUABLE FAILURE |
| 4 | 104a   | C: mean-reversion | 64.65 | 5/8 | VALUABLE FAILURE |
| 5 | 105a   | E: freeze ep5+60ep | 56.47 | **4/8** | VALUABLE FAILURE |
| 6 | 105a_v2| E: freeze ep7+60ep | 66.04 | 5/8 | VALUABLE FAILURE |
| 7 | 106a   | D: lambda_es=5.0 | 65.51 | 5/8 | VALUABLE FAILURE |

---

### Iteration 1: Exp 102a — Condition-Dependent Per-Cell Noise Scale
- **Direction**: B (encoder-modulated per-cell noise scale)
- **Hypothesis**: noise_scale_head (Linear+softplus) scales skip bypass per-cell based on condition. Should reduce over-spread cells.
- **Prediction**: Suite 2 over-spread cells should tighten; Suite 7 may improve.
- **Result**: 5/8 (same pattern). Score 65.73 (-0.58 vs baseline).
- **Improvements**: h=1 calibration (81.1% vs 74.5% worst), calib err 0.043 vs 0.072, coint 0.720 vs 0.675, KS daily 21 vs 20, median bias 23 vs 18
- **Regressions**: kurtosis 0.605 vs 0.845, CI overall 89.9% vs 91.3%, catastrophic 671 vs 576
- **Suite 2 failure**: Cell (4,0) still over 95% at h=7-30. Noise scale learned only 1.4x range (too modest).
- **WHY**: noise_scale_head is redundant with cell_spread_linear — both are condition→25 scalars on same path. CRPS per-cell gradient has no cross-cell signal, so both converge to near-uniform. The fundamental spread-collapse problem (CRPS MAE dominates diversity) applies to noise_scale_head just as it did to cell_spread.
- **INSIGHT**: Simply adding another learned scalar on the skip path doesn't help. Need to either (a) provide explicit variance target or (b) change the noise injection mechanism more fundamentally.
- **Decision**: VALUABLE FAILURE — confirms learned per-cell scaling is redundant with cell_spread. Next: try Direction A (learnable noise dampening) which changes the noise DYNAMICS, not just amplitude.

### Iteration 2: Exp 103a — Learned Rho for AR Noise
- **Direction**: A (learnable noise dampening)
- **Hypothesis**: Replace fixed rho=0.8 with sigmoid(rho_head(condition)). Model learns condition-dependent noise correlation.
- **Prediction**: Delta autocorrelation should decrease, per-horizon spread calibration improves.
- **Result**: 5/8 (same pattern). Score 65.47 (-0.84 vs baseline).
- **Improvements**: coint 0.791 (BEST), turb/calm 1.721 (BEST), median 24/25, MAE red 90%, h1 worst 82.4%
- **Regressions**: KS daily 17/25 (from 20), CI 88.6% (from 91.3%), catastrophic 808
- **Learned rho converged to 0.289** (model wants low rho for CRPS; this matches 101b: rho=0.3→4/8)
- **Cross-cell corr reached 0.323 at ep5 (near GT 0.38!)** but drifted to 0.661 by ep20
- **WHY**: CRPS-optimal rho ≈ 0.29 produces poor KS. Fixed rho=0.8 is a BETTER inductive bias.
- **INSIGHT**: Tension between CRPS-optimal dynamics and distributional realism. Constrained rho ∈ [0.6, 0.9] might preserve both.
- **Decision**: VALUABLE FAILURE — key insight about CRPS-optimal rho. Next: try 103a_v2 with clamped rho, or Direction C.

### Iteration 3: Exp 103a_v2 — Clamped Learned Rho [0.6, 0.95]
- **Direction**: A refinement
- **Hypothesis**: Clamp rho to [0.6, 0.95] preserves KS quality while allowing condition-dependence.
- **Result**: 5/8, score 66.20 (-0.11 vs baseline). Near-identical to baseline.
- **Key**: Rho stuck at 0.703 — sigmoid+clamp creates gradient desert. No condition-dependence learned.
- **Improvements**: Median bias 25/25 (perfect), catastrophic 547 (best). CI 90.8%.
- **Suite 2**: Cell (4,0) WORSE at 97.7-98.3% (fixed rho=0.7 produces more over-spread than 0.8).
- **WHY**: Effectively a fixed-rho=0.7 model. No information gained beyond confirming rho=0.7 is slightly worse than 0.8.
- **Decision**: VALUABLE FAILURE. Direction A exhausted (attempts 2/3, but root cause clear: can't learn rho without distributional loss). Move to Direction C.

### Iteration 4: Exp 104a — Mean-Reversion Dynamics
- **Direction**: C (non-anchored dynamics)
- **Hypothesis**: Add alpha*(mu(cond) - prev) term. Pulls trajectories toward learned equilibrium.
- **Result**: 5/8, score 64.65 (WORST, -1.66 vs baseline).
- **Alpha converged to 0.036** (3.6% pull per step) — too aggressive for 30 steps.
- **Improvements**: Coint 0.810 (BEST EVER), calm/turb bias more symmetric
- **Regressions**: Kurtosis 0.568 (barely passing), KS 17/25, catastrophic 925 (worst), h=30 worst 69.8% (UNDER 70%!)
- **WHY**: Mean-reversion dampens tails (pulls to mu≈0.5), increases catastrophic failures (wrong-level pull). CRPS doesn't penalize level distribution — it rewards accurate per-step prediction, so mu converges to reduce CRPS regardless of level distribution quality.
- **KS IV levels**: Still 1/25 — mean-reversion didn't help because mu is a single condition-dependent value, not a distribution. Levels are still anchored, just to mu instead of history[-1].
- **INSIGHT**: Mean-reversion changes the anchor from history[-1] to mu, but doesn't widen the level distribution. Would need noise in mu itself, or horizon-dependent mu that grows in variance.
- **Decision**: VALUABLE FAILURE. Direction C needs different implementation.

---

### Iteration 0: Baseline (99m_v2)
- **Hypothesis**: N/A (baseline measurement)
- **Model**: models/backfill/afcrps_99m_v2/best_model.pt
- **Result**: 5/8 PASS (1,3,4,5,6). FAIL: 2 (CI), 7 (regime), 8 (distributional)
- **Composite score**: 66.31/109
- **Key metrics**: CI90=0.913, worst_cell=FAIL, KS daily 20/25, KS levels 1/25, kurtosis 0.845, coint 0.675, median bias 18/25, regime L3 catastrophic=576

### Session 2 (2026-03-19) — Post-Investigation Experiments

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 35 | 120a | B1: AdaGN noise in MLP | — | **4/8** | VALUABLE FAILURE — variance saturation |
| 36 | 123b | C2: Ortho reg on skip | 66.47 | 5/8 | VALUABLE FAILURE — skip only 4% |
| 37 | 123a_v2 | C1: ACF loss | 65.39 | 5/8 | VALUABLE FAILURE — fights CRPS |
| 38 | 120b | B2: Noise-free MLP | PENDING | — | TRAINING |

**Exhausted directions**: B1 (AdaGN), C1 (ACF loss), C2 (ortho reg)
**Key insight**: MLP path (96% of output) dominates — skip-level and loss-level interventions insufficient.
