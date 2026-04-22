# 220h Full 11-Suite Multi-Horizon Validation

- model: `282a`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/282a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.840`
- h30 cov90: `0.728`
- turb/calm ratio: `1.063`
- MR ratio h1: `1.016`
- MR ratio h30: `0.792`

**Additional v2 Suites**
- time-series ACF corr: `0.935`
- block boundary ratio: `0.978`
- cointegration gen/GT ratio: `0.821`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `4/25`
- level KS pass cells: `0/25`
- corr ratio: `1.038`
- rank ratio: `1.077`
- max-jump KS: `0.460`
