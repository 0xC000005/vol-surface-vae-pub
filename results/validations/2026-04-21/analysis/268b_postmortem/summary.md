# 268b Postmortem

## Result
- Score: 1/11
- Passes: block_ar
- Best epoch: 8

## Key Metrics
- coverage90: 0.8793611111111111
- calibration_error: 0.024387345679012344
- conditional_mae_reduction: 0.33981829040643746
- turb_calm_width_ratio: 1.0174368619918823
- acf_corr: 0.9318446086712746
- kurtosis_ratio: 0.6055346255397372
- change_ks_pass_cells: 6
- level_ks_pass_cells: 1
- corr_ratio: 0.461178263674258
- rank_ratio: 2.8825846914145865
- mr_ratio: 1.6824763738210242
- mr_h1_ratio: 1.6824763738210242
- mr_h30_ratio: 0.7512256550967025
- max_jump_ks: 0.3984375
- q99_jump_ratio: 1.1000703613990608
- cointegration_ratio: 2.4854838709677427
- surface_explosion_rate: 0.7027994791666666

## Mechanism Read
- Sequence-aware history conditioning did not fix the decisive problem. Conditionality stayed essentially flat (MAE reduction only 0.3%, turb/calm width ratio 1.017), so the family still does not allocate uncertainty strongly by history/regime.
- The stronger conditioning path actually made support worse: explosion rose to 70.3%, level KS fell to 1/25, and cross-cell mean correlation dropped below gate (corr ratio 0.461).
- This points away from the broadcast-context hypothesis as the primary bottleneck. The deeper issue is that direct level-space path FM is too support-loose and too easy to solve with generic overspread trajectories.
- The direct path-space family remains elegant, but the stochastic object should likely be future change trajectories rather than future levels.

## Decision
- Do not keep patching direct level-space FM. Shift the 268 family to direct conditional future-change flow matching in a generic normalized change coordinate, then evaluate whether that tighter stochastic object restores support and conditional structure without reintroducing latent machinery.
