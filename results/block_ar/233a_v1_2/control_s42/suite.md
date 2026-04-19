# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_control_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.619`
- h30 cov90: `0.642`
- worst cell h30: `0.255`
- best cell h30: `0.911`

**Conditionality**
- turb/calm ratio: `0.950`
- MAE reduction vs shuffled: `5.0%`

**Mean Reversion**
- aggregate slope ratio: `0.664`
- active pass count: `4/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `14/25`
- level KS pass cells: `11/25`
- worst floor occupancy: `20.357%`

**Cross-Cell / Pathwise**
- corr ratio: `0.978`
- rank ratio: `1.590`
- max-jump KS: `0.531`
