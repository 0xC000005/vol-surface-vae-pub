# 220b Multi-Horizon Path Suite

- model: `233a_C`
- checkpoint: `models/backfill/233a_v1_C_25d_s1337/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.623`
- h30 cov90: `0.748`
- worst cell h30: `0.453`
- best cell h30: `0.990`

**Conditionality**
- turb/calm ratio: `0.971`
- MAE reduction vs shuffled: `5.4%`

**Mean Reversion**
- aggregate slope ratio: `0.825`
- active pass count: `2/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `13/25`
- level KS pass cells: `12/25`
- worst floor occupancy: `7.079%`

**Cross-Cell / Pathwise**
- corr ratio: `1.097`
- rank ratio: `1.057`
- max-jump KS: `0.928`
