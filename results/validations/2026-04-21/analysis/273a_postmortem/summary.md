# 273a Postmortem

## Result
- Score: 2/11
- Passes: block_ar, cointegration
- Best epoch: 20

## Key Metrics
- coverage90: 0.996
- calibration_error: 0.188
- conditional_mae_reduction: 1.2%
- turb_calm_width_ratio: 0.996
- acf_corr: 0.940
- kurtosis_ratio: 0.482
- corr_ratio: 1.599
- rank_ratio: 0.337
- mr_ratio: -1.673
- cointegration_ratio: 1.513
- max_jump_ks: 1.000
- surface_explosion_rate: 1.000
- level_ks_pass: 0
- change_ks_pass: 0
- mae_pass: 4

## Mechanism Read
- The tokenized observation model is not the main problem. A direct reconstruction probe on encoded ground-truth future surfaces stays on-manifold:
  - recon_floor_rate: 0.000
  - recon_ceiling_rate: 0.000
  - recon_mae: 0.0328
- The failure appears immediately in the sampled latent transition. A one-step rollout probe from real histories already produces support violations:
  - step1_floor_rate: 0.291
  - step1_ceiling_rate: 0.069
  - step1_min: -1.905
  - step1_max: 7.971
- So `273a` does not fail because the explicit observation autoencoder is too weak. It fails because deterministic FM on raw token-state deltas leaves the learned latent manifold almost immediately, and the decoder then maps those off-manifold states into floor/ceiling explosions, over-common-mode structure, and the wrong MR sign.

## Decision
- Close deterministic token-delta FM as the active `273a` path.
- Do not add target-scaling or latent-clipping knobs.
- Next step: move to a cleaner probabilistic latent-token state-space family where the latent manifold is part of the model specification rather than an unregularized deterministic autoencoder target.
