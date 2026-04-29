# 757a Next-Step Ideation

## Context

755a made short-prefix generated exposure a promising active ingredient: train-tail improved to `8/11`, while validation stayed `6/11`. 756a showed the remaining validation gap is mostly shifted-level hard-cell support and per-cell regime/median allocation, not generic path realism.

## Literature Anchors

- Scheduled Sampling, NeurIPS 2015: teacher-forced sequence models can accumulate errors at inference because they are conditioned on generated rather than true previous states. Source: https://papers.nips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html
- Conformal Prediction Under Covariate Shift, NeurIPS 2019: weighted conformal calibration is a principled way to handle train/test covariate shift when test covariates are known or density ratios can be estimated. Source: https://papers.nips.cc/paper/8522-conformal-prediction-under-covariate-shift
- EnbPI for dynamic time series, ICML 2021: conformal-style prediction intervals can be adapted to non-exchangeable time series and wrapped around general predictive models. Source: https://proceedings.mlr.press/v139/xu21h.html

## Options

### Option A: Scheduled Short-Prefix Curriculum

Keep the same normalized-innovation AR flow core and the 755a short-prefix ingredient, but train in stages that gradually increase generated-prefix exposure. This directly addresses the 752a/753a failure mechanism: full generated-prefix FM was too off-manifold when applied statically, while K=5 repaired in-sample path law.

Advantages:
- Model-internal and first-principles aligned.
- No post-hoc calibration layer.
- Directly tests whether late-horizon validation coverage needs more generated-prefix exposure, introduced gradually.

Risk:
- May reintroduce the full-prefix damage seen in 752a if the final stage is too aggressive.

### Option B: State-Local Weighted Support Calibration

Use calibration windows weighted by current history/level similarity to adjust scenario support, inspired by weighted conformal prediction under covariate shift and time-series conformal methods.

Advantages:
- Directly targets validation shifted-level coverage.
- Risk-manager defensible if framed as calibration rather than learned law.

Risk:
- Less Bitter-Lesson aligned.
- More likely to become a calibrated stress system rather than a clean learned conditional law.
- Needs careful overcoverage control and could become another research knob if introduced too early.

## Decision

Try Option A first. The next experiment should be a scheduled prefix-length curriculum:

- stage 1: start from the 674a incumbent with `free_running_fm_prefix_steps=5`;
- stage 2: continue from stage 1 with `free_running_fm_prefix_steps=15`;
- keep `free_running_fm_weight=0.2`, rollout/channel objective, normalized-innovation coordinate, AR flow backend, and one stochastic source fixed;
- evaluate validation and train-tail full suites.

Do not jump to conformal/support calibration yet. Keep it as the next fallback if scheduled exposure fails to improve validation support without sacrificing the 755a train-tail gains.
