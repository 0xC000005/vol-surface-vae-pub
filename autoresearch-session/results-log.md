# Autoresearch Session 2026-03-19: Breaking the 5/8 Ceiling

**Goal**: Break the 5/8 ceiling via Research Compass (principled architectural changes)
**Directions**: H1 (diagnostic) → H2 (variogram) or H3 (CLN) → H4 (non-AR)
**Branch**: autoresearch-session-20260319

## Iteration Log

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 0 | 99m_v2 | baseline  | 66.31 | 5/8    | BASELINE |

---

### Iteration 0: Baseline (99m_v2)
- **Model**: models/backfill/afcrps_99m_v2/best_model.pt
- **Result**: 5/8 PASS (1,3,4,5,6). FAIL: 2 (CI), 7 (regime), 8 (distributional)
- **Composite score**: 66.31/109
- **Key metrics**: CI90=0.913, worst_cell=FAIL, KS daily 20/25, KS levels 1/25, kurtosis 0.845, coint 0.675, median bias 18/25, regime L3 catastrophic=576
- **Effective output rank**: 1.57 (critical diagnostic baseline)

