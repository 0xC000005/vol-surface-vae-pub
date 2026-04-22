# 220h Full 11-Suite Multi-Horizon Validation

- model: `268a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/268a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `surface, coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.900`
- h30 cov90: `0.897`
- turb/calm ratio: `0.967`
- MR ratio h1: `1.907`
- MR ratio h30: `0.852`

**Additional v2 Suites**
- time-series ACF corr: `0.923`
- block boundary ratio: `1.020`
- cointegration gen/GT ratio: `1.560`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `8/25`
- level KS pass cells: `4/25`
- corr ratio: `0.511`
- rank ratio: `2.715`
- max-jump KS: `0.653`
