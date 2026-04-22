# 220h Full 11-Suite Multi-Horizon Validation

- model: `293h`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/293h_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.928`
- h30 cov90: `0.992`
- turb/calm ratio: `0.876`
- MR ratio h1: `2.321`
- MR ratio h30: `0.854`

**Additional v2 Suites**
- time-series ACF corr: `0.954`
- block boundary ratio: `0.957`
- cointegration gen/GT ratio: `0.811`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `0/25`
- corr ratio: `1.027`
- rank ratio: `1.234`
- max-jump KS: `0.643`
