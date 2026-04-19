# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_both_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.630`
- h30 cov90: `0.594`
- worst cell h30: `0.229`
- best cell h30: `0.818`

**Conditionality**
- turb/calm ratio: `0.957`
- MAE reduction vs shuffled: `3.6%`

**Mean Reversion**
- aggregate slope ratio: `0.811`
- active pass count: `1/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `17/25`
- level KS pass cells: `6/25`
- worst floor occupancy: `34.009%`

**Cross-Cell / Pathwise**
- corr ratio: `0.868`
- rank ratio: `1.401`
- max-jump KS: `0.789`
