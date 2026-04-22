# 220h Full 11-Suite Multi-Horizon Validation

- model: `283b`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/283b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.813`
- h30 cov90: `0.698`
- turb/calm ratio: `1.121`
- MR ratio h1: `0.995`
- MR ratio h30: `0.752`

**Additional v2 Suites**
- time-series ACF corr: `0.950`
- block boundary ratio: `1.049`
- cointegration gen/GT ratio: `0.756`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `23/25`
- level KS pass cells: `1/25`
- corr ratio: `1.000`
- rank ratio: `1.125`
- max-jump KS: `0.477`
