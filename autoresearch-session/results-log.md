# RC21 Autoresearch Results Log

**Session**: RC21 — Conditional Prediction Quality
**Branch**: autoresearch-session-rc21
**Started**: 2026-04-02
**Previous**: RC20 → 6/9 (softplus barrier breakthrough)

## Baseline (from RC20.6)

| Metric | Softplus best (ep11) | Softplus final (ep80) | Target |
|--------|---------------------|----------------------|--------|
| Suites | 6/9 | 5/9 | 7+/9 |
| S1 explosion | 2.5% PASS | 5.9% FAIL | <5% |
| S2 worst_cell_cov | 0.652 FAIL | ~0.70 FAIL | >0.70 |
| S3 turb/calm | 1.185 PASS | 1.162 PASS | >1.15 |
| S7 L2 | 0/8 FAIL | 1/8 FAIL | >=6/8 |
| S8 KS-levels | 12/25 FAIL | 17/25 FAIL | >15/25 |

## Key Evidence (from RC20 investigation)
- Oracle per-window debiasing → 100% CI coverage (spread adequate, centering wrong)
- Probe 0: cond IS used (zero→MAE+73%) but weakly discriminative (shuffle→MAE+0.7%)
- S7 never passed in 11 RC20 experiments
- Encoder has 84.2% regime probe accuracy but conditions nearly interchangeable

## Iterations

| # | Exp ID | Direction | Key Metrics | Decision |
|---|--------|-----------|-------------|----------|
| 1 | 165a | Mean+Residual (hard centering) | 4/9 (kurtosis 20.3, lost S3+S4) | VALUABLE FAILURE — centering kills persistence |
| 2 | 165a_v2 | Additive Innovation (MLP mean head) | **6/9** S2 PASS, S3 FAIL (worst_cell -18.4%) | PARTIAL SUCCESS — S2 first pass, S3 tension |
| 3 | 165a_v3 | Spatial Transformer mean head | 5/9 (overfits, lost S2) | FAILURE — bigger head makes it worse |
| 4 | 165b | Direct ensemble mean MSE (no mean head) | 5/9 S3 PASS(+17pp) S2 FAIL(0.062) | VALUABLE FAILURE — centering works but compresses spread |
| — | 165b_v2 | Corrected IS (0.005) + ensemble mean MSE | Target: S2+S3 PASS, ≥7/9 | NEXT EXPERIMENT |

## Key Findings (H1 series + gradient decomposition)
- Mean head approach has S2/S3 tension (shared drift damages conditionality)
- Ensemble mean MSE fixes S3 (+17pp) but compresses spread (S2 worsens)
- Root cause: CRPS allocates only 13% of gradient to centering (spread 2.9x dominant)
- IS at lambda=0.05 is 42% of gradient (10x too strong), starving CRPS
- Stochastic pathway (ConditionalNorm) is ALIVE — sole diversity source, not dormant
- K=16 under-samples: K=100 gets 82.7% coverage (vs 75.8%), but distribution itself still narrow
- Combined fix: correct IS (0.005) + add centering MSE with gradient matching
