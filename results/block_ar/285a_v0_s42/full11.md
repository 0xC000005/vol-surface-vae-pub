# 220h Full 11-Suite Multi-Horizon Validation

- model: `285a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/283a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.808`
- h30 cov90: `0.734`
- turb/calm ratio: `1.047`
- MR ratio h1: `0.987`
- MR ratio h30: `1.085`

**Additional v2 Suites**
- time-series ACF corr: `0.953`
- block boundary ratio: `1.072`
- cointegration gen/GT ratio: `0.563`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `24/25`
- level KS pass cells: `3/25`
- corr ratio: `1.002`
- rank ratio: `1.143`
- max-jump KS: `0.451`
