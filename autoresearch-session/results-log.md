# Autoresearch Session — 2026-03-17

**Goal**: 8/8 test suites on raw model (Bitter Lesson, no post-hoc)
**Starting model**: 99m_v2 (5/8 PASS: suites 1,3,4,5,6)
**Branch**: autoresearch-session-20260317

## Iteration Log

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 0 | 99m_v2 | baseline  | 66.31 | 5/8    | BASELINE |

---

### Iteration 0: Baseline (99m_v2)
- **Hypothesis**: N/A (baseline measurement)
- **Model**: models/backfill/afcrps_99m_v2/best_model.pt
- **Result**: 5/8 PASS (1,3,4,5,6). FAIL: 2 (CI), 7 (regime), 8 (distributional)
- **Composite score**: 66.31/109
- **Key metrics**: CI90=0.913, worst_cell=FAIL, KS daily 20/25, KS levels 1/25, kurtosis 0.845, coint 0.675, median bias 18/25, regime L3 catastrophic=576
