# RC19 Autoresearch Results Log

**Session**: RC19 — AR + No-Norm + VS (Literature-Grounded)
**Branch**: autoresearch-session-rc19
**Started**: 2026-03-27
**Baselines**: 155d (test CI=0.330, turb/calm=0.90) + 158a (test CI=0.442, turb/calm=0.988)

## Baseline Metrics

| Metric | 155d (residual, test) | 158a (E2E, test) | Target |
|--------|----------------------|-------------------|--------|
| CI worst | 0.330 | 0.442 | >0.40 |
| turb/calm | 0.90 | 0.988 | >1.15 |
| Corr ratio | 0.820 | 0.888 | >0.80 |
| KS daily | 13/25 | 2/25 | >10 |
| Suites | 4/6 | 2/6 | 5+/6 |

## Iterations

| # | Exp ID | Direction | Key Metrics | Decision |
|---|--------|-----------|-------------|----------|
| 1 | 159a | H2a-S1: No-LN CLN (FCN3) | val turb/calm: peak 1.164@ep20 → 0.921@ep80. Test: CI=0.241, turb/calm=0.940, 4/6 suites. | VALUABLE FAILURE: Mechanism works transiently but afCRPS erases it. No-LN necessary but not sufficient. |
