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
| 1 | 164a | AR + no-LN spatial + VS + direct | 4/9 (S1,S5,S6,S9). Boundary 99.4%. tc=1.07. eff_rank=43. | VALUABLE FAILURE → fix output scale |
| 2 | 164a_v2 | Remove reflecting boundary | 4/9 (S1,S4,S5,S9). Delta std=0.019 (correct). tc=1.24 (no FB). | VALUABLE FAILURE → GRU feedback mismatch |
| 3 | 164a_v3 | Add GRU feedback during training | **5/9** (S1,S3,S4,S5,S6). tc=1.23. coint=0.52. eff_rank=1.77. | BUILD ON THIS — rank-1 is remaining bottleneck |
| — | RC20.1 | Research compass: break rank-1 | H1: VS lambda=5.0, H2: per-cell noise (if H1 fails) | Next: H1 |
