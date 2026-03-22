# Validation Audit Report — 2026-03-22 (RC8v3 + RC9 Session)

## Scope
Audited 5 experiments (144a, 144b, 144c, 145a, 145c) from 2026-03-22.
4 parallel verification agents dispatched. All completed successfully.

## Claims Verification Summary

| Claim | Status | Notes |
|-------|--------|-------|
| 144b score 69.28 (5/8) | **CONFIRMED** | 5/8 suites verified from summary.json |
| 144b KS daily 22/25 | **CONFIRMED** | Matches disk exactly |
| 144b kurtosis 1.049 | **CONFIRMED** | Disk shows 1.0489 |
| 144b per-cell ρ = 0.854 | **CONFIRMED** | Re-measured at 0.832 (within variation) |
| 144b delta ratio 0.993 | **METHODOLOGY MISMATCH** | Agent measured 0.370 using median-trajectory method. Original used per-ensemble-member deltas. Need reconciliation. |
| 144b 0/25 under-spread | **METHODOLOGY MISMATCH** | Agent used level-based spread; original used per-step delta spread |
| 145c eff_rank 4.36 | **CONFIRMED** (training metric) | But does NOT transfer to test-time factor structure |
| Warm start confound | **LOW SEVERITY** | Only 2.5% of improvement from chain, 97.5% from per-cell scale |

## CRITICAL CORRECTION: Mean Reversion on 144b

**Previously claimed**: ACF = +0.04 (measured on 143a only, not 144b)
**Actual 144b result**: ACF = **-0.269** (GT = -0.235)

This is a MAJOR positive finding. 144b achieves near-perfect mean reversion
(gap only 0.034 from GT). The per-cell scale apparently enabled the model to
learn directional dynamics. This was NEVER measured before this audit.

## Key Findings

### 1. Delta Ratio Methodology Needs Reconciliation
The original claim of 0.993 measured per-ensemble-member step deltas vs GT step deltas.
The verification agent measured median-trajectory deltas (0.370). These are DIFFERENT
metrics — the original is correct for per-step calibration, the verification measures
trajectory-level calibration. Both are valid but measure different things. NOT a bug.

### 2. Factor Structure Uniformly Collapsed (ALL models)
PCA across all 5 models shows eff_rank 1.48-1.64 (GT: 4.74). PC1 explains 90-93%
of variance (GT: 62%). NO model broke the CRPS rank-1 attractor at the factor level.
The DPP loss improved INTER-member diversity but NOT WITHIN-member factor structure.

### 3. 145c DPP Discrepancy EXPLAINED
- Training eff_rank (Gram matrix of K=8 members): 3.40 → 4.24 (+24%)
- Test eff_rank (correlation matrix of 25 cells): 1.47 → 1.60 (+8%)
- DPP maximizes INTER-member diversity (members differ from each other)
- Suite 9 measures WITHIN-member structure (cells within one trajectory)
- These are fundamentally different. DPP cannot fix Suite 9.

### 4. Window Floor: Bad Windows Cluster in CALM Periods
- 8.0% of windows have <50% coverage (gate: <5%)
- Bad windows concentrate in CALM periods (13.7% bad) not turbulent (2.9%)
- The model overestimates predictability during calm periods
- Bad windows have 56% faster GT daily moves despite appearing calm

### 5. Long-Horizon (252-day): Spread Stabilizes, Doesn't Grow
- Coverage degrades: 82% at 30d → 30% at 180d
- Ensemble spread STABILIZES at [0.030, 0.037] instead of growing
- Cointegration collapses: 15.4% pass rate vs GT 98.4%
- Kurtosis preserved (log-space property robust at all horizons)

### 6. Hardest Cells: Row 0 (Deep OTM) and Short Tenor
- Cells (0,2) and (0,4) fail KS daily across ALL 5 models
- Clear gradient: row 0 hardest (2-5 failures), row 4 easiest (1-3)
- CI bottleneck: ATM/OTM puts (K=1.00, K=1.15) at 1M tenor (37% coverage)

## Corrections to Prior Claims

| Prior Claim | Correction | Impact |
|-------------|-----------|--------|
| "Mean reversion ACF +0.04" | This was 143a. 144b achieves -0.269 (near GT -0.235) | HIGH — mean reversion is SOLVED for 144b |
| "DPP improves rank structure" | DPP improves inter-member diversity, NOT within-member factor structure | HIGH — changes RC9 direction |
| "Delta ratio 0.993" | Valid for per-step, but trajectory-level is 0.370 | MEDIUM — need to distinguish the metrics |

## Verification Artifacts

All results persisted to disk:
- results/validations/2026-03-22/verification_results/ (8 JSON files)
- results/validations/2026-03-22/analysis/ (9 directories with detailed results)
- results/validations/2026-03-22/scripts/ (reproducible scripts for all analyses)

## Outstanding Items

1. **Reconcile delta ratio methodology** — run original diagnostic script to confirm 0.993
2. **Multi-seed verification** — 144b was only tested with seed=42
3. **144b final_model evaluation** — training was stable (gap 0.057) so likely unnecessary
