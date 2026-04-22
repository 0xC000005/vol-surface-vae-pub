# 269a Postmortem

## Result
- Score: 1/11
- Passes: block_ar
- Best epoch: 19

## Key Metrics
- coverage90: 0.9894930555555556
- calibration_error: 0.32505169753086427
- conditional_mae_reduction: 5.161431790191469
- turb_calm_width_ratio: 0.979145348072052
- acf_corr: 0.9484591745089066
- kurtosis_ratio: 0.3171589360913742
- change_ks_pass_cells: 2
- level_ks_pass_cells: 0
- corr_ratio: 0.2983537303628529
- rank_ratio: 3.607088351498044
- mr_ratio: 0.04996822533166132
- mr_h1_ratio: 0.04996822533166132
- mr_h30_ratio: 1.1038324250501226
- max_jump_ks: 0.6520833333333333
- q99_jump_ratio: 1.0202885404930586
- cointegration_ratio: 0.735483870967742
- surface_explosion_rate: 1.0

## Mechanism Read
- Autoregressive state feedback did turn on one desired property: conditional MAE reduction passed at 5.2%. So generated-state feedback matters.
- But the pure direct-generator family is now failing in a different way: it produces massively over-wide trajectories, near-total explosion, and diffuse cross-cell structure (corr ratio 0.298, rank ratio 3.607).
- Mean reversion is still effectively absent through most of the horizon. So simply making the model autoregressive does not recover the right state-dependent law when the generator remains fully unconstrained in observation space.
- Taken together with 268a-268c, this suggests the direct no-structure first-principles line is exhausted. Some learned shared bottleneck is likely necessary, but it should be generic and data-driven rather than a hard low-rank hand-design.

## Decision
- Close the pure direct-generator line. Next step should be research ideation for a new restart family built around a narrow learned latent bottleneck with autoregressive state feedback, still without hard low-rank heads, bounded side paths, or heuristic teacher machinery.
