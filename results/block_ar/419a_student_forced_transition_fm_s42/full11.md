# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/419a_student_forced_transition_fm_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `6/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.893`
- h30 cov90: `0.919`
- turb/calm ratio: `1.102`
- MR ratio h1: `1.010`
- MR ratio h30: `0.817`

**Additional v2 Suites**
- time-series ACF corr: `0.956`
- block boundary ratio: `0.999`
- cointegration gen/GT ratio: `0.753`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `7/25`
- corr ratio: `1.010`
- rank ratio: `1.436`
- max-jump KS: `0.307`
