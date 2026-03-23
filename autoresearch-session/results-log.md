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
| 18 | 146c | RC10-H3: cum_cal multi-horizon calibration | 62.05 | 5/9 | VALUABLE FAILURE — CI improved (74→76.2%, h=1: 64.7→75.1%), 252d coverage 53→73%, but KS daily collapsed 22→1/25. Trade-off: calibration vs distributional fidelity. |

## RC11: Complete the Architecture, Don't Add Losses

**Goal**: Break through 5/9 ceiling by completing the noise architecture.
**Compass**: RC11 (RESEARCH_LOG.md — "Research Compass RC11")
**Theme**: Enable disabled pathways + add learned spread control. No new losses.
**Base model**: 144b (69.28, 5/9). Best modification: 146b (eff_rank +54%).

### RC10 Summary
| Exp | Result | Learning |
|-----|--------|----------|
| 146a | FALSIFIED | cell_var = diversity reward, not penalty |
| 146b | PARTIAL (+54% rank) | Skip bypass + factor noise works |
| 146c | FAILS (KS 1/25) | cum_cal incompatible with distributional fidelity |
| 147a | FAILS (KS 4/25) | Even λ=0.1 breaks KS |
| 147b | PARTIAL (KS 24/25 but lost S6) | Factor+cum_cal: KS recovers but cointegration breaks |

### 7 Bottlenecks (complete mechanistic understanding)
1. CLN rank-1 (Jacobian 1.23)
2. Skip bypass DISABLED
3. Encoder = level only (cos 0.9999+)
4. Constant vol_scale (~2x residual heteroscedasticity)
5. Decoder positive bias (+0.123)
6. CRPS rank collapse
7. Calm tail vulnerability (59 persistent bad windows)

### RC11 Theory Queue
| # | Hypothesis | Type | Status |
|---|-----------|------|--------|
| H0 | Inference noise scaling probe | Inference only, 5 min | **DONE — STRUCTURAL bottleneck confirmed** |
| H1 | Enable skip bypass on 144b | Zero code change, 30 min | **DONE — CONFOUNDED. Skip collapsed under CRPS (norm=0.103). Eff_rank +15% but CI −2.3pp.** |
| H2 | Heteroscedastic decoder output | Implementation, 1.5h | Blocked by H1 |
| H3 | Condition-dependent noise amplitude | Zero code change, 30 min | **DONE — noise_scale_head learned 0.75× suppression. KS 25/25 but CI −8.9pp. CRPS spread suppression is root constraint.** |

### Iteration 21: Exp 148_probe — Inference Noise Scaling (H0)
- **Hypothesis**: Scale noise z by beta={1.5, 2.0, 3.0} at inference. Tests amplitude vs structure.
- **Result**: CI 74.0%→73.9% across ALL betas. Zero effect. h=1 CI WORSENED (64.7→60.9%).
- **KEY FINDING**: Bottleneck is 100% STRUCTURAL (rank-1 CLN). Amplitude is irrelevant.
  - Eff_rank unchanged: 1.471→1.479
  - Conditioned width unchanged: 0.0930→0.0931
  - Per-cell CI: h=7 mean −1.14pp (worsened!), h=30 mean +0.63pp (noise)
- **Decision**: VALUABLE FAILURE. Cleanest falsification of amplitude hypothesis. H1 (skip bypass) is now CRITICAL — it's the only way to break rank-1.

### Iteration 22: Exp 148a — Skip Bypass Without Factor Noise (H1)
- **Hypothesis**: Plain Linear skip bypass breaks rank-1 without factor noise.
- **Result**: 5/9 (same suites). CI 74.0→71.7% (−2.3pp). KS 22→15/25 (−7). Eff_rank 1.47→1.69 (+15%).
- **KEY FINDING**: CRPS drove skip proj weights to near-zero (norm=0.103). Skip has eff_rank=10.64 capacity but CRPS suppresses amplitude. Factor noise (146b) resists this.
- **CONFOUND**: Recipe added cell_spread/ES/IS/bias_lambda/reflect that 144b lacked. CI regression likely from recipe, not skip.
- **Decision**: CONFOUNDED. Cannot isolate skip contribution. 146b recipe validated as principled path. Build H2/H3 on 146b.

### Iteration 23: Exp 148c — Noise Scale Conditioning on 146b (H3)
- **Hypothesis**: Enable noise_scale_cond on 146b for condition-dependent spread.
- **Result**: 5/9 (same suites). CI 77.3→68.4% (−8.9pp). KS 21→25/25 (+4, PERFECT!). Eff_rank 2.26→2.23.
- **KEY FINDING**: noise_scale_head learned 0.75× uniform suppression (CV=1.6%). CRPS optimizes toward less noise everywhere. KS-CI trade-off is fundamental under CRPS.
- **KILL**: Turb/calm decreased 2.14→1.76 (−18%, opposite of prediction). Encoder signal used for suppression, not differentiation.
- **Decision**: VALUABLE FAILURE. 4th independent confirmation of CRPS spread suppression. 5/9 ceiling is a LOSS problem, not architecture.

## RC12: Per-Cell Noise Architecture (AIFS-Inspired)

**Goal**: Break through 5/9 ceiling by completing the per-cell noise architecture.
**Compass**: RC12 (RESEARCH_LOG.md — "Research Compass RC12")
**Theme**: More independent noise channels = more structural resistance to CRPS collapse.
**Base model**: 146b (69.14, 5/9). Best modification with eff_rank 2.26.

### Evidence Base (from deep investigation)
- Factor noise resists CRPS collapse: W eff_rank 4.74/5, all 5 factors active
- Skip pathway = 2.7% of variance but 100% of eff_rank improvement
- AIFS uses per-location noise (35K dims) with same loss → no collapse
- Suite 9 needs only +11.2% eff_rank. Suite 8 is 1 cell away.
- 146b broke Suite 8 (was passing in 144b). Factor noise worsened median bias.

### Quantitative Gap Table

| Suite | Metric | Current | Target | Gap | Tractability |
|-------|--------|---------|--------|-----|-------------|
| 9 | eff_rank | 2.26 | 2.51 | +11.2% | **Easiest** |
| 8 | frac_pass | 19/25 | 20/25 | 1 cell | Near-miss |
| 8 | mag_pass | 21/25 | 22/25 | 1 cell | Near-miss |
| 8 | window_floor | 5.31% | <5.0% | ~4 windows | Near-miss |
| 2 | h=1 CI | 59.8% | 90% | +30.2pp | Hard |
| 7 | calm h=1 | 51.4% | 90% | +38.6pp | Hard |

### RC12 Theory Queue
| # | Hypothesis | Type | Status |
|---|-----------|------|--------|
| H1 | n_factors=10→25 (Suite 9) | Single hyperparam, 30 min | NEXT |
| H2 | Fix median bias regression (Suite 8) | ar_bias_lambda probe + training | Parallel with H1 |
| H3 | Per-cell CLN noise injection | ~50 LOC, 2h | After H1/H2 signal |

### Iteration 24: Exp 149a — n_factors=10 on 146b (RC12-H1)
- **Hypothesis**: More factor channels (5→10) = more structural diversity = eff_rank ≥ 2.51.
- **Result**: 4/9 (−1, lost cointegration). eff_rank 2.26→2.17 (−0.09). KS 21→6/25 (−15!). CI −5.1pp.
- **KEY FINDING**: CRPS collapsed 5/10 factors (top-1 energy 80% vs 46.8%). Min/max SV ratio 0.032 vs 0.396. Factor resistance has a SWEET SPOT at 5 for this model scale.
- **Decision**: VALUABLE FAILURE. "More factors = better" cleanly falsified. Path forward is per-cell CLN (H3), not more skip factors.

### Iteration 25: Exp 149b — ar_bias_lambda=0.05 on 146b (RC12-H2)
- **Hypothesis**: 5× bias regularization fixes median bias → Suite 8 PASS.
- **Result**: 4/9 (−1, lost Suite 4: kurtosis 2.03). CI −8.1pp, KS −16, eff_rank −0.28.
- **KEY FINDING**: Bias regularization too strong. Bias is structural (factor W direction + CLN interaction), not simple decoder offset.
- **Decision**: VALUABLE FAILURE. H2 falsified at this level. Proceed to H3 (per-cell CLN).
