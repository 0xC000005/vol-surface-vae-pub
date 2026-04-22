# 220h Full 11-Suite Multi-Horizon Validation

- model: `266c`
- checkpoint: `models/backfill/266c_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.125`
- h30 cov90: `0.295`
- turb/calm ratio: `0.980`
- MR ratio h1: `1.877`
- MR ratio h30: `1.094`

**Additional v2 Suites**
- time-series ACF corr: `0.718`
- block boundary ratio: `1.235`
- cointegration gen/GT ratio: `0.318`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `2/25`
- level KS pass cells: `0/25`
- corr ratio: `2.113`
- rank ratio: `0.242`
- max-jump KS: `0.989`
