# 268a Postmortem

## Result
- Score: 3/11
- Passes: block_ar, cointegration, cross_cell_correlation
- Best epoch: 23

## Key Metrics
- coverage90: 0.909375
- calibration_error: 0.032180555555555546
- conditional_mae_reduction: 0.12393025064843374
- turb_calm_width_ratio: 0.9666840434074402
- acf_corr: 0.9233929243248745
- kurtosis_ratio: 0.5755742535861947
- change_ks_pass_cells: 8
- level_ks_pass_cells: 4
- corr_ratio: 0.5112674635176183
- rank_ratio: 2.714504883370997
- mr_ratio: 1.9066844023146095
- mr_h1_ratio: 1.9066844023146095
- mr_h30_ratio: 0.8520008870508421
- max_jump_ks: 0.653125
- q99_jump_ratio: 0.9874111118129297
- cointegration_ratio: 1.559677419354839
- surface_explosion_rate: 0.5901692708333334

## Mechanism Read
- Direct path-space FM is alive as a scenario family: it reaches 3/11 without latent collapse and gets strong unconditional coverage/calibration.
- The dominant failure is weak history use. Conditionality is nearly flat (MAE reduction ~0.1%, turb/calm width ratio <1), which means the model is mostly learning a generic path law rather than a history-sensitive conditional law.
- Mean dynamics are misallocated across horizon: short-horizon mean reversion is too strong (h1 ratio ~1.9) while longer horizons are much closer to gate. This points to an overly global history summary rather than horizon-specific conditioning.
- Path-space support is also too loose: explosion rate is 59%, level KS is 4/25, and pathwise max-jump KS is 0.653. The direct path parameterization can generate spread, but not yet a history-anchored realistic support manifold.

## Decision
- Proceed to 268b-v0: keep direct path-space FM, but replace the single broadcast history context with sequence-aware history-to-future conditioning so the velocity field can use history locally in time without adding hand-engineered side paths.
