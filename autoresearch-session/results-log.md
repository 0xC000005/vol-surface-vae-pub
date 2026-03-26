# RC18 Autoresearch Results Log

**Session**: RC18 — Noise Bottleneck + Multivariate Loss
**Branch**: autoresearch-session-rc18
**Started**: 2026-03-26
**Baseline**: 155d (CI=0.748, corr=0.910, KS=25/25, 5/6 suites)

## Baseline Metrics (155d, validated 2026-03-25)

| Metric | Value | Target |
|--------|-------|--------|
| CI worst_cell | 0.748 | >0.80 |
| Corr ratio | 0.910 | >0.80 |
| KS daily | 25/25 | >20/25 |
| Kurtosis | 1.166 | 0.5-2.0 |
| Spread-skill | 1.078 | ~1.0 |
| Growing unc | 1.00 | >0.80 |
| 252d explosion | 0.000 | <0.01 |
| Suites | 5/6 | 6/6 |

## Iterations

| # | Exp ID | Direction | Metric | Decision |
|---|--------|-----------|--------|----------|
| 1 | 156a | H1-S1: noise_dim=4 | CI=0.664, corr=1.168, KS=25/25 | VALUABLE FAILURE — bottleneck too tight |
