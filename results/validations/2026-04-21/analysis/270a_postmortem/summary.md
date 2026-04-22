# 270a Postmortem

## Result
- Score: 2/11
- Passes: block_ar, cointegration
- Best epoch: 15

## Key Metrics
- coverage90: 0.9644305555555556
- calibration_error: 0.1247098765432099
- conditional_mae_reduction: -1.6528801068398387
- turb_calm_width_ratio: 1.009108304977417
- acf_corr: 0.898978276017913
- kurtosis_ratio: 0.04355438880664671
- change_ks_pass_cells: 1
- level_ks_pass_cells: 0
- corr_ratio: 2.2886085258820277
- rank_ratio: 0.17774861075404205
- mr_ratio: 2.855905288011109
- mr_h1_ratio: 2.855905288011109
- mr_h30_ratio: 1.1537845241179896
- max_jump_ks: 0.6229166666666667
- cointegration_ratio: 1.2629032258064516
- surface_explosion_rate: 0.9852430555555556
- latent_prev_std_mean: 0.0014143198495730758
- teacher_forced_corr_ratio: 2.2924038123778687
- teacher_forced_rank_ratio: 0.17553496586466147
- teacher_forced_mr_ratio: 0.6022829897533521
- teacher_forced_kurtosis_ratio: 0.03711850925145931

## Mechanism Read
- The single-vector learned bottleneck collapsed. Teacher-forced latent codes have almost no variation across the dataset (mean latent std ~1.4e-3), so the failure is upstream of stochastic sampling.
- Because the bottleneck/decoder pair already collapses under teacher forcing, the sampled model inherits an almost rank-1 common mode: corr ratio 2.289 and rank ratio 0.178.
- This is not evidence that all bottlenecks are wrong. It is evidence that a single-vector bottleneck is too compressive for this path law. The next restart should keep a generic bottleneck but make it a short latent sequence, not a single code.
- The autoregressive state-feedback idea remains live in principle, but it needs a less degenerate shared representation.

## Decision
- Do not knob-tune 270a. Shift to 270b-v0: autoregressive latent-sequence bottleneck next-change FM, keeping the same first-principles bans while replacing the single latent code with a short learned latent token state.
