# 220b Multi-Horizon Path Suite

- model: `233a_C`
- checkpoint: `models/backfill/233a_v1_C_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.606`
- h30 cov90: `0.612`
- worst cell h30: `0.323`
- best cell h30: `0.891`

**Conditionality**
- turb/calm ratio: `1.029`
- MAE reduction vs shuffled: `3.6%`

**Mean Reversion**
- aggregate slope ratio: `0.799`
- active pass count: `4/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `15/25`
- level KS pass cells: `9/25`
- worst floor occupancy: `12.368%`

**Cross-Cell / Pathwise**
- corr ratio: `1.250`
- rank ratio: `0.773`
- max-jump KS: `0.953`
