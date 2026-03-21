# Autoresearch Session 2026-03-21: RC6 Principled Architecture

**Goal**: Build principled architecture piece by piece (Karpathy). 5-step roadmap.
**Compass**: RC6 (RESEARCH_LOG.md line 33876)
**Branch**: autoresearch-session-20260321

## Theory Queue
Step 1 (139a): Unfreeze encoder + ortho reg
Step 2 (140a): Replace MLP decoder with causal transformer (AFTER Step 1 understood)
Step 3 (141a): Replace noise with CLN
Step 4 (142a): Strip loss to CRPS + VS only
Step 5 (143a): Strip vol_scale, cell_spread, NoiseMLP

## Iteration Log

| # | Exp ID | Direction | Score | Suites | Decision |
|---|--------|-----------|-------|--------|----------|
| 0 | 99m_v2 | baseline  | 66.31 | 5/8    | BASELINE |
| 1 | 139a | RC6 Step 1: unfrozen encoder + ortho reg | 57.98 | 4/8 | VALUABLE FAILURE — ortho reg works (rank max), coverage collapses from CRPS imbalance |

### Iteration 1: Exp 139a — Unfrozen Encoder + Ortho Reg
- **Hypothesis**: Joint training + ortho reg prevents encoder rank collapse
- **Result**: 4/8, score 57.98 (-8.33 vs baseline). Coverage 76.4% (was 91.3%).
- **KEY FINDING**: Ortho reg works perfectly (weight rank near-maximum). But weight rank ≠ output rank. MLP decoder still compresses to rank ~1.3.
- **SURPRISE**: KS levels 20/25 (was 1/25), coint 0.942 (was 0.675) — best ever distributional quality.
- **MECHANISM**: Unfrozen encoder gives CRPS accuracy term more leverage → overwhelming spread → under-coverage.
- **Decision**: VALUABLE FAILURE. Encoder anti-collapse solved. Proceed with coverage fix variant.
