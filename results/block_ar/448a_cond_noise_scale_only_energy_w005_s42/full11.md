# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/448a_cond_noise_scale_only_energy_w005_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.539`
- h30 cov90: `0.510`
- turb/calm ratio: `0.954`
- MR ratio h1: `0.800`
- MR ratio h30: `0.707`

**Additional v2 Suites**
- time-series ACF corr: `0.959`
- block boundary ratio: `1.006`
- cointegration gen/GT ratio: `0.373`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `1/25`
- level KS pass cells: `12/25`
- corr ratio: `1.380`
- rank ratio: `0.826`
- max-jump KS: `0.977`
