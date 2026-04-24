# 220h Full 11-Suite Multi-Horizon Validation

- model: `421a`
- checkpoint: `models/backfill/421a_joint_score_transition_path_fm_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `4/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.827`
- h30 cov90: `0.978`
- turb/calm ratio: `0.970`
- MR ratio h1: `0.419`
- MR ratio h30: `0.849`

**Additional v2 Suites**
- time-series ACF corr: `0.952`
- block boundary ratio: `0.961`
- cointegration gen/GT ratio: `0.748`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `5/25`
- corr ratio: `0.189`
- rank ratio: `3.734`
- max-jump KS: `0.500`
