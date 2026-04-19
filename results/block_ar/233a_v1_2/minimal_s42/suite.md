# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_minimal_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.589`
- h30 cov90: `0.677`
- worst cell h30: `0.312`
- best cell h30: `0.885`

**Conditionality**
- turb/calm ratio: `1.006`
- MAE reduction vs shuffled: `3.3%`

**Mean Reversion**
- aggregate slope ratio: `0.673`
- active pass count: `3/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `17/25`
- level KS pass cells: `7/25`
- worst floor occupancy: `8.403%`

**Cross-Cell / Pathwise**
- corr ratio: `1.154`
- rank ratio: `1.146`
- max-jump KS: `0.868`
