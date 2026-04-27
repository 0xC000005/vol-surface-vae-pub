# 637a Coordinate Diagnostic: State-Conditioned Level-Score Transition

## Context

634a and 636a both falsified rollout-loss fine-tuning from 631a. They preserved native 25+13 joint mechanics but damaged IV coverage, conditionality, level occupancy, and terminal path realism. This points at the state evolution coordinate rather than another loss-weight variant.

## What Is Already Falsified

The obvious score-state idea has already been tested:

- 622a trained the generic native joint AR transition in encoded log/diff state coordinates, then decoded generated states for evaluation.
- 622a scored `4/11`, passing surface, block-AR, cross-cell correlation, and mean reversion.
- It failed coverage, conditionality, time-series properties, cointegration, regime coverage, distributional fidelity, and pathwise jump realism.
- 623a widened 622a with temperature `1.5`; coverage improved to `77.1%`, but score fell to `2/11`, daily-change KS collapsed to `3/25`, level KS stayed `1/25`, median-bias stayed `1/25`, and cross-cell geometry failed.

So simply returning to the old 609/622 score-state model is not a principled move.

## What Remains Untried

The untested coordinate is a hybrid in the state-space sense, not a two-branch architecture:

- condition on both recent encoded levels and recent encoded increments, as in 629a/631a;
- generate the next empirical level-score change, not the raw encoded log/diff increment;
- decode the generated next level score back to encoded level value;
- derive the realized encoded increment from the difference between generated next level and previous encoded level;
- feed both generated level score and generated increment score back into the same causal memory.

This keeps the one-model native joint panel interface while changing the generated stochastic object from raw encoded increment to support-valid next level score.

## Why This Is Different From 622a

622a's memory was the old generic score-state memory: current empirical score plus score deltas. It did not use the 629/631 state-conditioned increment memory that explicitly carries both level and increment score histories.

The proposed 638a model keeps the richer 629/631 conditioning signal that improved native joint factor behavior, but changes the sampled coordinate to prevent raw increment accumulation from drifting IV levels out of realistic support.

## Hypothesis

If the main 631a failure is level drift from integrating raw encoded increments, then generating next level scores directly should:

- reduce surface explosions and terminal level occupancy failures;
- preserve mean reversion better than pure increment integration;
- retain more joint factor dependence than the old 622a score-state route because increment history remains in the memory state.

## Decision

Run 638a as a clean coordinate/paradigm experiment:

- new model class: state-conditioned level-score transition flow;
- same panel data and preprocessing as 629a/631a;
- same generic `iv_only`/`joint38` interface;
- no low-rank decoder, no bounded idio path, no IV/factor branch, no post-hoc calibration;
- evaluate with the IV full 11-suite and the joint-panel audit.

Acceptance is not just a higher count. 638a must materially reduce the 631a/636a level-drift symptoms without collapsing joint-panel dependence.
