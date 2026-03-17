# Autoresearch Session — 2026-03-17

**Goal**: 8/8 test suites on raw model (Bitter Lesson, no post-hoc)
**Starting model**: 99m_v2 (5/8 PASS: suites 1,3,4,5,6)
**Branch**: autoresearch-session-20260317

## Iteration Log

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 0 | 99m_v2 | baseline  | 66.31 | 5/8    | BASELINE |
| 1 | 102a   | B: noise_scale_cond | 65.73 | 5/8 | VALUABLE FAILURE |

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

---

### Iteration 0: Baseline (99m_v2)
- **Hypothesis**: N/A (baseline measurement)
- **Model**: models/backfill/afcrps_99m_v2/best_model.pt
- **Result**: 5/8 PASS (1,3,4,5,6). FAIL: 2 (CI), 7 (regime), 8 (distributional)
- **Composite score**: 66.31/109
- **Key metrics**: CI90=0.913, worst_cell=FAIL, KS daily 20/25, KS levels 1/25, kurtosis 0.845, coint 0.675, median bias 18/25, regime L3 catastrophic=576
