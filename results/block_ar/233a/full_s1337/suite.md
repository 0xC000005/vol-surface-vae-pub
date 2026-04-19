# 220b Multi-Horizon Path Suite

- model: `233a_full`
- checkpoint: `models/backfill/233a_v1_full_25d_s1337/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.642`
- h30 cov90: `0.692`
- worst cell h30: `0.208`
- best cell h30: `0.948`

**Conditionality**
- turb/calm ratio: `0.931`
- MAE reduction vs shuffled: `4.5%`

**Mean Reversion**
- aggregate slope ratio: `0.814`
- active pass count: `2/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `15/25`
- level KS pass cells: `10/25`
- worst floor occupancy: `33.550%`

**Cross-Cell / Pathwise**
- corr ratio: `0.834`
- rank ratio: `1.708`
- max-jump KS: `0.898`
