# Autoresearch 451a: scaled AR empirical-score transition core

## Context

The 448-450 conditional noise-scale branch is capped. Scale-only learning either
collapsed or stayed at identity, and joint transport/backbone training still
returned the learned scale to the lower clamp while scoring only 7/11.

The prior model-family evidence is also asymmetric:

- Full one-shot empirical-score path laws scored poorly: 339/345/346/354/413/417/421
  were typically 3-4/11 and lost time-series, mean-reversion, correlation, or jump
  realism.
- The empirical normal-score AR transition family reached the deployable frontier:
  377a, 385a, and 392a reached 8/11.
- 392a preserves daily-change distribution, cointegration, cross-cell correlation,
  mean reversion, and pathwise jumps, but still fails coverage, regime coverage,
  and unconditional level KS.

## Hypothesis

The next clean branch should not add another calibration shell around 392a. It
should test whether the same vanilla AR conditional flow core improves when scaled
directly:

- same conditional scenario-generator semantics,
- same empirical normal-score coordinate,
- same rectified-flow transition objective,
- larger history memory / transition velocity capacity,
- no retrieval, oracle validation futures, per-window correction, or evaluator
  specific losses.

This is a Bitter-Lesson-aligned experiment: let a larger learned conditional law
absorb the remaining level/regime structure before adding explicit risk-policy
calibration.

## Proposed Decisive Falsifier

Train a larger `340a`-family empirical normal-score causal-memory transition flow
from scratch with only capacity/training-scale changes:

- increase memory and velocity dimensions,
- keep the existing train/validation framing,
- keep vanilla FM training,
- evaluate with the same full 11-suite.

Expected read:

- If larger vanilla AR improves level KS or regime layer-2 without damaging
  conditionality, the branch is alive and should be fine-tuned with rollout-energy
  scoring like 392a.
- If it remains at or below the 392a frontier, the remaining failures are not
  primarily capacity-limited; the next paradigm must change the likelihood/score
  target rather than add local knobs.

## Deployability

This remains deployable if successful: at inference it only uses history and
random source noise. There is no validation-future oracle, no residual retrieval,
and no posthoc per-window correction.
