# 270b Postmortem

## Result
- Score: 1/11
- Passes: block_ar
- Best epoch: 20

## Key Metrics
- coverage90: 0.9544375
- calibration_error: 0.12643595679012348
- conditional_mae_reduction: 2.678914665147513
- turb_calm_width_ratio: 0.9959323406219482
- acf_corr: 0.9232927902023736
- kurtosis_ratio: 1.596979644576032
- corr_ratio: 1.1747065280532747
- rank_ratio: 0.49666117733363385
- mr_ratio: 1.0050514891614342
- mr_h1_ratio: 1.0050514891614342
- mr_h30_ratio: -0.46398297893052864
- max_jump_ks: 1.0
- cointegration_ratio: 0.8919354838709679
- token_std_mean: 0.0109908077865839
- teacher_forced_corr_ratio: 1.5556449379128932
- teacher_forced_rank_ratio: 0.2982250307596241
- teacher_forced_mr_ratio: -1.2004791639162615
- teacher_forced_kurtosis_ratio: 0.9392174017539218

## Mechanism Read
- The token bottleneck fixed the collapse pathology from 270a. Teacher-forced token states have real variation (mean std ~0.011 instead of ~0.0014).
- But teacher-forced decoding is still structurally wrong: it already has too-low rank and the wrong mean-reversion sign/profile. So the remaining failure is not latent collapse; it is the decoded state law.
- Sampled rollouts then amplify that mismatch into explosive width and extreme jump scales. This makes the decoded target/object the next clean bottleneck, not the token representation itself.
- The family is still alive. The next minimal change should keep the token bottleneck and autoregressive state feedback, but change the decoded object from next level to next change.

## Decision
- Proceed to 270c-v0: keep the latent-sequence bottleneck and autoregressive latent FM transition, but decode the next normalized change rather than the next normalized level.
