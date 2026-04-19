# 220b Multi-Horizon Path Suite

- model: `233a_B`
- checkpoint: `models/backfill/233a_v1_B_25d_s2024/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/7`
- failed suites: `surface, coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.527`
- h30 cov90: `0.520`
- worst cell h30: `0.156`
- best cell h30: `0.818`

**Conditionality**
- turb/calm ratio: `1.000`
- MAE reduction vs shuffled: `3.6%`

**Mean Reversion**
- aggregate slope ratio: `0.815`
- active pass count: `2/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `15/25`
- level KS pass cells: `5/25`
- worst floor occupancy: `16.462%`

**Cross-Cell / Pathwise**
- corr ratio: `1.088`
- rank ratio: `1.125`
- max-jump KS: `0.799`
