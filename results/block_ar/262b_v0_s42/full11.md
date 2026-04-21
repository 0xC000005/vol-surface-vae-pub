# 220h Full 11-Suite Multi-Horizon Validation

- model: `262b`
- checkpoint: `models/backfill/262b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.638`
- h30 cov90: `0.605`
- turb/calm ratio: `0.942`
- MR ratio h1: `0.581`
- MR ratio h30: `0.688`

**Additional v2 Suites**
- time-series ACF corr: `0.960`
- block boundary ratio: `0.987`
- cointegration gen/GT ratio: `0.573`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `18/25`
- level KS pass cells: `14/25`
- corr ratio: `0.364`
- rank ratio: `1.487`
- max-jump KS: `0.952`
