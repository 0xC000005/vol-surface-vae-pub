# 220h Full 11-Suite Multi-Horizon Validation

- model: `268c`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/268c_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.930`
- h30 cov90: `0.947`
- turb/calm ratio: `0.892`
- MR ratio h1: `0.004`
- MR ratio h30: `-0.039`

**Additional v2 Suites**
- time-series ACF corr: `0.949`
- block boundary ratio: `0.980`
- cointegration gen/GT ratio: `0.695`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `10/25`
- level KS pass cells: `0/25`
- corr ratio: `0.636`
- rank ratio: `2.478`
- max-jump KS: `0.567`
