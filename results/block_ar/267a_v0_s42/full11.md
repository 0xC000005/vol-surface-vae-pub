# 220h Full 11-Suite Multi-Horizon Validation

- model: `267a`
- checkpoint: `models/backfill/267a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.002`
- h30 cov90: `0.003`
- turb/calm ratio: `1.468`
- MR ratio h1: `2.051`
- MR ratio h30: `0.879`

**Additional v2 Suites**
- time-series ACF corr: `0.878`
- block boundary ratio: `0.567`
- cointegration gen/GT ratio: `0.568`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `1.205`
- rank ratio: `0.719`
- max-jump KS: `1.000`
