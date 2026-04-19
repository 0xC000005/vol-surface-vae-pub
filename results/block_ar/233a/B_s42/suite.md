# 220b Multi-Horizon Path Suite

- model: `233a_B`
- checkpoint: `models/backfill/233a_v1_B_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.514`
- h30 cov90: `0.596`
- worst cell h30: `0.245`
- best cell h30: `0.984`

**Conditionality**
- turb/calm ratio: `1.006`
- MAE reduction vs shuffled: `3.1%`

**Mean Reversion**
- aggregate slope ratio: `0.901`
- active pass count: `3/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `9/25`
- level KS pass cells: `12/25`
- worst floor occupancy: `5.944%`

**Cross-Cell / Pathwise**
- corr ratio: `1.224`
- rank ratio: `0.924`
- max-jump KS: `0.845`
