# 220h Full 11-Suite Multi-Horizon Validation

- model: `340c`
- checkpoint: `models/backfill/468a_recent_condcritic_w0002_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.859`
- h30 cov90: `0.920`
- turb/calm ratio: `1.040`
- MR ratio h1: `1.006`
- MR ratio h30: `0.800`

**Additional v2 Suites**
- time-series ACF corr: `0.957`
- block boundary ratio: `0.990`
- cointegration gen/GT ratio: `0.685`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `11/25`
- corr ratio: `0.948`
- rank ratio: `1.526`
- max-jump KS: `0.365`
