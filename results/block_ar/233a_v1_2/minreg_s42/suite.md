# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_minreg_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.599`
- h30 cov90: `0.616`
- worst cell h30: `0.297`
- best cell h30: `0.885`

**Conditionality**
- turb/calm ratio: `1.065`
- MAE reduction vs shuffled: `2.8%`

**Mean Reversion**
- aggregate slope ratio: `0.519`
- active pass count: `1/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `12/25`
- level KS pass cells: `9/25`
- worst floor occupancy: `4.592%`

**Cross-Cell / Pathwise**
- corr ratio: `1.203`
- rank ratio: `1.028`
- max-jump KS: `0.849`
