# 220h Full 11-Suite Multi-Horizon Validation

- model: `283c`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/283c_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.826`
- h30 cov90: `0.700`
- turb/calm ratio: `1.125`
- MR ratio h1: `0.971`
- MR ratio h30: `0.748`

**Additional v2 Suites**
- time-series ACF corr: `0.952`
- block boundary ratio: `1.021`
- cointegration gen/GT ratio: `0.837`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `23/25`
- level KS pass cells: `1/25`
- corr ratio: `0.999`
- rank ratio: `1.146`
- max-jump KS: `0.446`
