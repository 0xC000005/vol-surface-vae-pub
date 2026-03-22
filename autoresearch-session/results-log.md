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
| 12 | 144b | RC8v3-H2: Per-cell learned scale | **69.28** | **5/8** | BUILD ON — delta ratio 0.993 (target!). KS daily 22/25 (best ever). ρ=0.854 with GT. 0/25 under-spread. CI dropped 78→74%. |
| 13 | 144c | RC8v3-H3: d_model=128 capacity | 65.84 | 5/8 | VALUABLE FAILURE — rank collapsed to 1.17 (GT 6.19). CRPS attractor scales with capacity. d_model=64 is sweet spot. |

### Iteration 1: Exp 139a — Unfrozen Encoder + Ortho Reg
- **Hypothesis**: Joint training + ortho reg prevents encoder rank collapse
- **Result**: 4/8, score 57.98 (-8.33 vs baseline). Coverage 76.4% (was 91.3%).
- **KEY FINDING**: Ortho reg works perfectly (weight rank near-maximum). But weight rank ≠ output rank. MLP decoder still compresses to rank ~1.3.
- **SURPRISE**: KS levels 20/25 (was 1/25), coint 0.942 (was 0.675) — best ever distributional quality.
- **MECHANISM**: Unfrozen encoder gives CRPS accuracy term more leverage → overwhelming spread → under-coverage.
- **Decision**: VALUABLE FAILURE. Encoder anti-collapse solved. Proceed with coverage fix variant.

| 16 | 146a | RC10-H1: cell_var temporal-only variance | 64.92 | 5/9 | VALUABLE FAILURE — CI dropped 74→70.8%. Cell_var pooling is a "beneficial bug" rewarding diversity. Falsified. |

### Iteration 16: Exp 146a — cell_var temporal-only variance
- **Hypothesis**: cell_var pools (B,K,T) suppressing diversity. Fix: var(dim=2).mean(dim=(0,1))
- **Result**: 5/9 {1,3,4,5,6}, CI 70.8% (was 74.0%). h=1 collapsed 64.7%→39.5%.
- **KEY FINDING**: Old cell_var implicitly REWARDS diversity (counts between-member spread toward GT target). Fix removed reward → less spread → worse CI.
- **MECHANISM**: Training spread dropped 26.8→23.4. Between-member fraction ~0% at test time for both.
- **Decision**: VALUABLE FAILURE. Revert cell_var. Use 144b base for H2/H3. CI root cause is NOT cell_var.
| 17 | 146b | RC10-H2: Factor-structured noise skip | 69.14 | 5/9 | PARTIAL SUCCESS — eff_rank 1.47→2.26 (+54%), CI 74→77.3%, ACF 0.75→0.95. But 252d coverage 53→14% (kurtosis compounding). |
