# Validation Audit Report — 2026-03-24

## Scope
Audited RC15 experiments (152f, 153a, 153b) from autoresearch session on branch
`autoresearch-session-rc15`. 4 experiments + 1 full evaluation reviewed.

## Original Gap Inventory

| # | Gap Type | Severity | Experiment | Status |
|---|----------|----------|------------|--------|
| 1 | MISSING_RESULTS | MEDIUM | 152f training_history.json corrupted | FIXED |
| 2 | MISSING_ANALYSIS | MEDIUM | 153b no diagnostic JSON on disk | FIXED |
| 3 | MISSING_ANALYSIS | HIGH | 153a no long-horizon 252d test | FILLED |
| 4 | MISSING_ANALYSIS | MEDIUM | 153a no multi-seed verification | FILLED |
| 5 | MISSING_ANALYSIS | MEDIUM | 153a no per-cell breakdown of failures | FILLED |
| 6 | MISSING_ANALYSIS | LOW | 153a best_model vs final_model | FILLED |
| 7 | NO_SCRIPT | LOW | 153b inline diagnostics not saved | FIXED |

**All 7 gaps resolved.** 5 verification agents ran in parallel, all completed successfully.

## Verification Results

### Agent 1: Long-Horizon 252-Day Test (153a)
- **Status**: COMPLETE
- **Key findings**:
  - NO explosion, NO NaN across 252 days — model is stable
  - Mean IV drifts slightly (0.201 → 0.189) but no divergence
  - Spread GROWS from 0.015 → 0.036 (2.4x ratio) — proper uncertainty growth
  - Calendar arbitrage 88-98% per chunk — surfaces lose term structure at long horizon
  - Butterfly arbitrage 90-100% — same issue
- **Verdict**: Stable generation, but surface validity degrades at long horizon

### Agent 2: Multi-Seed Evaluation (153a seeds 42/43/44)
- **Status**: COMPLETE
- **Key findings**:
  - Seed 42: 4/9, Seed 43: 5/9, Seed 44: 4/9
  - Stable PASS (all seeds): S3, S4, S5, S9
  - Stable FAIL (all seeds): S1, S2, S7, S8
  - UNSTABLE (seed-dependent): S6 (Cointegration — 47%, 53.5%, 49.2%)
  - S6 flips to PASS at seed 43 (53.5% > 50%) — this is borderline
- **Verdict**: 4/9 is the robust result. S6 is noise-dependent.

### Agent 3: Per-Cell Breakdown of Failing Suites (153a)
- **Status**: COMPLETE
- **Critical findings**:
  - **S1 (Calendar arb)**: NOT a model failure! Gen arb 39.9% vs GT 43.0% — model
    actually has FEWER violations than GT. The v2 test uses different threshold/convention.
  - **S2 (CI Coverage)**: SPREAD-DOMINATED failure. Ensemble 90% CI width = 41.8% of GT
    variability. Moneyness gradient: ITM 53.5% > ATM 35.2% > OTM 19.8% coverage.
  - **S6 (Cointegration)**: Cell-cell cointegration is 96.3% — excellent. The v2 Suite 6
    tests IV-vs-EWMA which is a different test.
  - **S8 (KS levels)**: Sample size asymmetry makes KS oversensitive. Same spatial pattern
    as CI: OTM worst.
  - **Root cause**: Single mechanism — ODE ensemble is 2.4x too narrow. Not bias (r=0.36),
    but spread (r=0.57 with CI). OTM cells worst because higher GT variability.

### Agent 4: 153b Diagnostics + 152f Fix
- **Status**: COMPLETE
- **153b**: Diagnostics saved. Confirms kurtosis 1.37 (worse than 153a's 1.11).
  Level bias very low (0.002). KS 25/25 on training data.
- **152f**: training_history.json reconstructed from log file. 6 eval checkpoints recovered.

### Agent 5: Best Model vs Final Model (153a)
- **Status**: COMPLETE
- **Critical finding**: best_model (ep28) has 2x wider per-window spread (0.030 vs 0.016)
  and CI worst cell 0.244 vs 0.107. Population eff_rank near-perfect (7.57, GT=7.61).
  But fails conditionality (turb_calm 1.019 < 1.15).
- **Tradeoff**: Early training = wider ensembles + weaker conditioning.
  Late training = narrower ensembles + stronger conditioning. Neither passes both S2 and S3.

## Corrections to Prior Claims

| Claim | Source | Correction |
|-------|--------|------------|
| "S1 cal arb 16.8% is a model failure" | 153a eval | NOT a failure — gen arb 39.9% < GT 43.0%. Different computation convention in v2 test. |
| "153a achieves 4/9 stably" | RC15 summary | Mostly correct, but S6 is borderline (flips at seed 43 to 5/9). Robust result is 4-5/9. |
| "153b kurtosis 1.35" | Research log | Confirmed at 1.37 with independent eval. Close to claimed value. |

## Mechanistic Understanding (from per-cell investigation)

**Single root cause for S2/S7/S8 failures: ODE ensemble is too narrow (2.4x)**

The per-cell analysis reveals:
1. Spread/GT ratio is UNIFORM at 0.35-0.45 across all cells
2. CI coverage has STRONG moneyness gradient (ITM 53.5% > OTM 19.8%)
3. OTM cells are worst because GT variability is highest there
4. KS and CI failures are correlated (r=-0.69) — same cells fail both
5. Bias is secondary (mean shift ~0.01, explains 36% of CI variance)
6. Spread is primary (explains 57% of CI variance)

**Best_model (ep28) finding suggests a training dynamics tradeoff:**
- The velocity field becomes MORE deterministic with continued training
- Early: diverse samples, good eff_rank (7.57), poor conditioning (1.019)
- Late: narrow samples, good conditioning (1.46), poor CI (0.107)
- The ODE's "sharpening" during training improves quality but kills diversity

## Outstanding Items

None requiring immediate human attention. All gaps filled.

**For next research compass**: The key evidence is that ODE ensemble diversity is
the single bottleneck. A hybrid approach or stochastic ODE could address this.
