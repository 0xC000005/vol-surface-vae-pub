# Autoresearch Session 2026-03-21: RC6 Principled Architecture

**Goal**: Build principled architecture piece by piece (Karpathy). 5-step roadmap.
**Compass**: RC6 (RESEARCH_LOG.md line 33876)
**Branch**: autoresearch-session-20260321

## Theory Queue (Updated 2026-03-22 after validation audit + course correction)
Step 1 (139a/v2): Unfreeze encoder + ortho reg — COMPLETE
Step 2 (140a): AR causal transformer decoder — COMPLETE
Step 3a (141a): Add CLN to transformer (Gaussian) — NEXT
Step 3b (141b): CLN + Student-t noise (resist CLT) — after 3a
Step 4 (142a): Strip loss to CRPS + VS only — after 3b
Step 5 (143a): Strip vol_scale, cell_spread, freeze — after 4

## Corrections from Validation Audit (2026-03-22)
- Cross-cell corr 0.389 was ENSEMBLE-MEMBER corr, NOT daily-change corr (Suite 9 = 0.178)
- CRPS self-calibration claim was premature (IS confounds coverage measurements)
- CLT kills kurtosis in AR (30 Gaussian steps → Gaussian by CLT) — not in original risk table

## Iteration Log

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 0 | 99m_v2 | baseline  | 66.31 | 5/8    | BASELINE |
| 1 | 139a | RC6 Step 1: unfrozen encoder + ortho reg | 57.98 | 4/8 | VALUABLE FAILURE — ortho reg works (rank max), coverage collapses from CRPS imbalance |
| 2 | 139a_v2 | RC6 Step 1: + freeze encoder at ep10 | **67.54** | **5/8** | BUILD ON — Suite 8 FIRST PASS, KS levels 15/25. Coverage decline is standard CRPS, not encoder. |
| 3 | 140a | RC6 Step 2: AR causal transformer | 53.16 | 4/8 | VALUABLE FAILURE — rank 4-5.6, ensemble corr 0.389. But noise crushed (14%), kurtosis 0.364 (CLT). Architecture validated, noise mechanism is bottleneck. |
| 11 | 144a | RC8v3-H1: Learned scalar vol_scale | 66.95 | 5/8 | VALUABLE FAILURE — delta ratio 0.40→0.70, 77x spread increase, but 8/25 edge cells under-spread. Scalar insufficient, H2 (per-cell) needed. |

### Iteration 1: Exp 139a — Unfrozen Encoder + Ortho Reg
- **Hypothesis**: Joint training + ortho reg prevents encoder rank collapse
- **Result**: 4/8, score 57.98 (-8.33 vs baseline). Coverage 76.4% (was 91.3%).
- **KEY FINDING**: Ortho reg works perfectly (weight rank near-maximum). But weight rank ≠ output rank. MLP decoder still compresses to rank ~1.3.
- **SURPRISE**: KS levels 20/25 (was 1/25), coint 0.942 (was 0.675) — best ever distributional quality.
- **MECHANISM**: Unfrozen encoder gives CRPS accuracy term more leverage → overwhelming spread → under-coverage.
- **Decision**: VALUABLE FAILURE. Encoder anti-collapse solved. Proceed with coverage fix variant.
