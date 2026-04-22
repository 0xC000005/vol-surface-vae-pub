# 274a Postmortem

## Result
- Score: 2/11
- Passes: surface, block_ar
- Best epoch: 20

## Key Metrics
- coverage90: 0.648
- calibration_error: 0.154
- conditional_mae_reduction: 0.6%
- turb_calm_width_ratio: 0.979
- acf_corr: 0.948
- kurtosis_ratio: 0.207
- corr_ratio: 1.546
- rank_ratio: 0.450
- mr_ratio: 2.336
- cointegration_ratio: 1.066 (worst-cell ratio: 0.224)
- max_jump_ks: 0.998
- surface_explosion_rate: 0.003
- level_ks_pass: 1
- change_ks_pass: 6
- mae_pass: 23

## Mechanism Read
- The probabilistic latent manifold fixes the `273a` support pathology cleanly. Surface validity now passes across the board, and the evaluator reports zero floor/ceiling saturation.
- The family did not collapse in the naive VAE way:
  - KL stayed nonzero throughout training (`val_kl ~ 0.255`)
  - prior and posterior standard deviations stayed distinct (`prior_std ~ 0.445`, `post_std ~ 0.329`)
- The observation model is still not the blocker:
  - posterior_recon_mae: 0.0267
  - posterior_floor_rate: 0.000
  - posterior_ceiling_rate: 0.000
- The live bottleneck is the one-step latent prior:
  - prior_step1_mae: 0.0363
  - prior_step1_std: 0.0643 vs GT step1 std 0.0892
  - samples stay on-support, but they are too narrow, too smooth, and too common-mode
- That explains the current failure pattern:
  - undercoverage despite good calibration shape
  - weak regime width allocation
  - low jump incidence and tail scale
  - rank ratio just below gate
  - over-strong aggregate mean reversion

## Decision
- Keep the `274` family alive.
- Do not add side paths, error-correction baselines, or collapse hacks.
- Next step: `274b-v0`, keeping the same probabilistic token-state family and explicit observation model, but replacing the pure one-step latent prior with a short-memory latent prior over recent latent-token history.
