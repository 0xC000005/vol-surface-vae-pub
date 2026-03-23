# Validation Audit Report — 2026-03-23

## Scope
Audited RC11 session experiments from 2026-03-23: 148_probe (H0), 148a (H1), 148c (H3).
Plus baseline verification of 146b.

## Metric Verification
**ALL claimed metrics verified against summary.json on disk — ZERO mismatches.**
Training histories for 148a and 148c also verified (best epoch, val gap).

## Gaps Found

| # | Type | Severity | Status | Finding |
|---|------|----------|--------|---------|
| 1 | DIRTY_FALSIFICATION | HIGH | **RESOLVED** | H1 confound isolated: CI regression from skip bypass WITHOUT factor noise, NOT from recipe changes |
| 2 | MISSING_ANALYSIS | MEDIUM | **DONE** | 148a long-horizon: PARTIAL_PASS. Spread grows but CI collapses at d180 (28%) |
| 3 | MISSING_ANALYSIS | MEDIUM | **DONE** | 148c long-horizon: FAIL. Spread NON-MONOTONIC, CI at d252 = 11.6% |
| 4 | NO_SCRIPT | MEDIUM | **DONE** | 148c noise_scale_head: 0.755 mean ✓, CV 1.54% ✓. NEW: with conditioning, range is [0.006, 5.36] |
| 5 | MISSING_ANALYSIS | MEDIUM | **DONE** | Per-cell CI across betas: zero effect confirmed. No cell >5pp improvement |

## Key Corrections to Prior Understanding

### 1. H1 Confound RESOLVED (was HIGH severity)
The research log said "CONFOUNDED — cannot isolate skip bypass from recipe changes."
**Correction**: The confound IS resolved. Three-model ablation (144b/148a/146b) shows:
- Recipe changes contribute ZERO to CI drop (same recipe in 146b → positive CI)
- Skip bypass WITHOUT factor noise causes −2.34pp CI (short-horizon variance collapse)
- Factor noise accounts for 171% of 146b improvement (overcomes skip drag + adds net positive)
- **Skip bypass alone is harmful. Skip+factor is beneficial. Factor noise is the load-bearing component.**

### 2. noise_scale_head NOT Uniformly Constant (was MEDIUM)
The research log said "0.75× uniform suppression (CV=1.6%)."
**Correction**: The BIAS is nearly constant (0.755 ± 0.012). But with realistic conditioning inputs, the FULL output ranges from 0.006 to 5.36 (CV=117%). The head IS condition-dependent — CRPS just drove the unconditional baseline toward suppression. The log claim should be updated to note this distinction.

### 3. Long-Horizon Results (NEW findings)
- **148a**: PARTIAL_PASS. CI 97.2% at d30, drops to 28-40% at d180-d252. Kurtosis blows up (33× at d252).
- **148c**: FAIL. Spread is non-monotonic (collapses after d30). CI 8.8% at d180, 11.6% at d252. Much worse than 148a.
- **Implication**: noise_scale_cond makes long-horizon performance MUCH worse (148c vs 148a).

## Updated Mechanistic Understanding

The RC11 compass stated the theme "Complete the architecture, don't add losses." After this validation:

1. **H0 (amplitude)**: Cleanly falsified. Zero CI effect at 3× noise. VERIFIED ✓
2. **H1 (skip bypass)**: NOT confounded — skip bypass without factor noise is harmful (variance collapse at h=1). Factor noise is the load-bearing fix. CORRECTED.
3. **H3 (noise_scale_cond)**: noise_scale_head is MORE condition-dependent than claimed (range 0.006-5.36). CRPS drives the bias toward suppression but the weights learn large condition-dependent modulation. Long-horizon performance is severely degraded. CORRECTED + NEW.

## Outstanding Items
- H2 (heteroscedastic) was SKIPPED — should this be tested on 146b?
- RC12 compass was drafted before this validation. The correction to H1 (factor noise is load-bearing) and H3 (noise_scale IS condition-dependent with wide range) may change RC12 priorities.

## Files Produced
- 5 verification_result.json files in `verification_results/`
- 5 reproducible bash scripts in `scripts/`
- 5 analysis directories with detailed outputs in `analysis/`
