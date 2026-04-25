# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/450a_joint_cond_noise_widen_energy_init105_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.850`
- h30 cov90: `0.900`
- turb/calm ratio: `1.089`
- MR ratio h1: `1.012`
- MR ratio h30: `0.777`

**Additional v2 Suites**
- time-series ACF corr: `0.956`
- block boundary ratio: `0.992`
- cointegration gen/GT ratio: `0.587`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `13/25`
- corr ratio: `0.970`
- rank ratio: `1.444`
- max-jump KS: `0.372`
