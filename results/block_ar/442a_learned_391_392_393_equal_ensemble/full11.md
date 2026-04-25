# 442a Equal-Weight Learned Checkpoint Ensemble

- policy: `equal-weight learned checkpoint ensemble`
- deployable: `True`
- uses validation future: `False`
- total samples per window: `48`
- member: `models/backfill/391a_recent_rollout_energy_w02_s42/best_model.pt`, samples: `16`, epoch: `1`
- member: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`, samples: `16`, epoch: `1`
- member: `models/backfill/393a_recent_rollout_energy_w01_s42/best_model.pt`, samples: `16`, epoch: `1`
- windows: `192`
- suite score: `7/11`
- failed suites: `coverage, conditionality, regime_coverage, distributional_fidelity`

**Key Metrics**
- cov90 overall: `0.852`
- h1 cov90: `0.840`
- h30 cov90: `0.864`
- conditionality MAE reduction: `3.84%`
- turb/calm ratio: `1.057`
- regime layer2: `0/8`
- daily-change KS pass cells: `25/25`
- level KS pass cells: `12/25`
- corr ratio/rank ratio: `0.955` / `1.495`
- mean-reversion active pass: `0.833`
- max-jump KS: `0.453`
