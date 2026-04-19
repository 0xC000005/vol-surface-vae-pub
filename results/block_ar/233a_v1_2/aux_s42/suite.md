# 220b Multi-Horizon Path Suite

- model: `233a_v1_2`
- checkpoint: `models/backfill/233a_v1_2_aux_25d_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/7`
- failed suites: `coverage, conditionality, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Coverage**
- h1 cov90: `0.584`
- h30 cov90: `0.639`
- worst cell h30: `0.292`
- best cell h30: `0.901`

**Conditionality**
- turb/calm ratio: `1.038`
- MAE reduction vs shuffled: `2.4%`

**Mean Reversion**
- aggregate slope ratio: `0.599`
- active pass count: `2/24`
- full-horizon overall: `False`

**Distributional Fidelity**
- daily-change KS pass cells: `16/25`
- level KS pass cells: `8/25`
- worst floor occupancy: `8.371%`

**Cross-Cell / Pathwise**
- corr ratio: `1.099`
- rank ratio: `1.149`
- max-jump KS: `0.822`
