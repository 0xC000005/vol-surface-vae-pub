# 405a Interval-Scale Calibrated Validation

- base model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- calibration: `pre-validation interval scale, objective=deadband, regime_bins=True, alpha=1.0`
- scale range: `0.800 / 1.000 / 1.300`
- windows: `192`
- samples per window: `48`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Horizon Summary**
- h1 cov90: `0.852`
- h30 cov90: `0.883`
- turb/calm ratio: `1.059`

**Additional v2 Suites**
- time-series ACF corr: `0.954`
- block boundary ratio: `1.043`
- cointegration gen/GT ratio: `0.663`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio: `0.960`
- rank ratio: `1.500`
- max-jump KS: `0.372`
