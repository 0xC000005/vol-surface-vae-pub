# RC20 Autoresearch Results Log

**Session**: RC20 — Principled AR + no-LN Spatial Transformer + VS
**Branch**: autoresearch-session-rc20
**Started**: 2026-03-30

## Baselines (v2 test suite, apples-to-apples)

| Metric | 99m_v2 (old, 5/8) | 161a (principled, 4/9) | Target |
|--------|-------------------|----------------------|--------|
| Kurtosis | 0.859 PASS | 0.448 FAIL | 0.5-2.0 |
| Cointegration | PASS | 0.431 FAIL | >0.50 |
| turb/calm | 1.49 PASS | 1.209 PASS | >1.15 |
| CI worst | FAIL | 63.6% FAIL | >70% |
| KS daily | 20/25 PASS | 16/25 PASS | >15 |
| v2 suites | 5/8 | 4/9 | 5+/9 |

## Iterations

| # | Exp ID | Direction | Key Metrics | Decision |
|---|--------|-----------|-------------|----------|
