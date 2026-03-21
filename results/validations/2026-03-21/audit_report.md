# Validation Audit Report — 2026-03-21

## Scope
Audited research log entries from 2026-03-19 to 2026-03-21. 12 experiments reviewed
(120b_v5 through 138a_v2), plus RC4, RC5, and session summary entries.

## Gaps Found
| Severity | Count | Types |
|----------|-------|-------|
| HIGH | 2 | MISSING_ANALYSIS (138a follow-up, 120b_v6 long-horizon) |
| MEDIUM | 6 | MISSING_ANALYSIS (ensemble, multi-seed, compute_score, 138a_v2 bestcov), MISSING_RESULTS (lambda sweep), NO_SCRIPT (training commands) |
| LOW | 1 | INCOMPLETE_FOLLOWUP (no RC6 ideation yet) |
| **Total** | **9** | |

## Verification Results

| Task | Status | Key Finding |
|------|--------|-------------|
| T1: 138a multi-seed | **ROBUST** | 6/8 identical across seeds 42, 123, 456. KS levels exactly 15/25 each time. |
| T2: 138a long-horizon | **PARTIAL** | 0% explosion, but CI 98%→55% at 252d. Spread shrinks. AR limitation. |
| T3: 120b_v6 long-horizon | **PARTIAL** | Same variance collapse — CI 99%→42%. Cross-cell corr good (0.50 vs GT 0.48). |
| T4: Ensemble 138a+120b_v6 | **NO BENEFIT** | 6/8, same as 138a alone. KS regresses 21→15. CI improves but per-cell fails. |
| T5: Lambda sweep artifact | **SAVED** | Structured JSON with 4 data points and key findings. |
| T6: 138a compute_score | **CONFIRMED** | v1=76.62 (6/8), v2=85.52 (7/9). All-time best on both systems. |
| T7: Training scripts | **SAVED** | 9 executable .sh files for all 03-21 experiments. |
| T8: 138a_v2 bestcov | **CONFIRMED** | 4/8, CI 88.4%. λ=1.0 too strong even at early epoch. 138a (λ=0.05) superior. |

## Corrections
None — all research log claims verified against disk.

## Key Insights

1. **138a 6/8 is the real deal** — multi-seed confirms robustness. Not a lucky seed.
2. **Long-horizon is the next frontier** — both IS-fix models collapse at 252d.
   Spread shrinks instead of growing. This is the AR(1) rho=0.8 limitation, not IS-related.
3. **Ensemble mixing doesn't help** — 138a's weaker KS dilutes 120b_v6's strength.
   The 6/8 from 138a alone is the best achievable by mixing these two.
4. **Per-horizon IS λ sweet spot is narrow** — λ=0.05 gives 6/8, λ=1.0 gives 4/8.
   The research log correctly identifies λ=0.1-0.3 as the next exploration range.
5. **138a v2=85.52 is all-time best composite** — confirmed by compute_score_v2.

## Outstanding
- RC6 ideation not yet done (RC5 exhausted all 3 hypotheses)
- Per-horizon IS λ sweep between 0.05-0.3 could push past 6/8
- Long-horizon variance collapse needs architectural solution (not loss-level)
