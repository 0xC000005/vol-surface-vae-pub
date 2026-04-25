# 427a Validation-Oracle Affine Diagnostic

- base model: `340c`
- checkpoint: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`
- diagnostic: `validation oracle affine median shift + interval scale; not deployable`
- shift range: `-0.2265 / -0.0019 / 0.1588`
- scale range: `0.450 / 0.875 / 1.800`
- windows: `192`
- samples per window: `48`
- suite score: `5/11`
- failed suites: `coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, mean_reversion`

**Fidelity / Structure**
- cov90 overall: `0.856`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `22/25`
- median-bias fraction cells: `25/25`
- median-bias magnitude cells: `24/25`
- regime layer2: `4/8`
- conditionality MAE reduction: `4.83%`
- cointegration gen/GT ratio: `0.671`
- mean-reversion active pass: `0.417`
- max-jump KS: `0.195`
