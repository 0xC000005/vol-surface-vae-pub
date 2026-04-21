# 220h Full 11-Suite Multi-Horizon Validation

- model: `262a`
- checkpoint: `models/backfill/262a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.740`
- h30 cov90: `0.867`
- turb/calm ratio: `0.999`
- MR ratio h1: `0.071`
- MR ratio h30: `0.307`

**Additional v2 Suites**
- time-series ACF corr: `0.948`
- block boundary ratio: `0.994`
- cointegration gen/GT ratio: `0.695`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `18/25`
- level KS pass cells: `9/25`
- corr ratio: `0.676`
- rank ratio: `1.709`
- max-jump KS: `0.831`
