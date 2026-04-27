# 639a Spec-Driven Mixed Coordinate Ideation

## Context

638a exposed a clean coordinate tradeoff:

- 631a / 629a encoded-increment generation gives the best native 25+13 joint behavior so far, but IV paths drift and explode.
- 638a level-score generation fixes IV support validity, kurtosis, tail scale, and pathwise jump-size ratios, but damages anchor-factor changes and absolute co-movement.

The failure is no longer "the architecture cannot model joint scenarios." It is that a single generated coordinate is not equally appropriate for every financial variable.

## Constraint

The user explicitly does not want separate IV and anchor-factor models or post-hoc gluing. Any next step must remain:

- one shared causal memory;
- one training script and one checkpoint;
- one stochastic rollout;
- no separate IV deck and factor deck;
- no low-rank decoder or bounded idiosyncratic side path.

## Candidate Principle

A mixed generated coordinate can be principled if it is treated as data-coordinate selection, not an architecture branch.

The model can still be one model. The difference is that each channel's generated stochastic variable is chosen by a variable specification:

- support-sensitive, mean-reverting level variables generate next empirical level-score changes;
- random-walk-like traded anchor levels generate encoded increments/log returns/diffs;
- all generated coordinates are concatenated into one vector and passed through the same memory, velocity network, optimizer, and sampler.

This is not "generate IV separately from factors." It is closer to the existing preprocessing rule that some raw columns use `log_level` and others use `diff_level`. The scientific claim would be: choose a stationary/support-valid generated coordinate per variable, then learn the joint law in that coordinate.

## Why This Is Not Fully Settled

The risky part is deciding the coordinate policy.

Hard-coding `iv:* -> level_score` and `factor:* -> increment` is probably effective but scientifically weaker. A cleaner version derives the policy from training data:

- level-score coordinate is preferred when level support validity and mean-reversion dominate;
- increment coordinate is preferred when increments are more stationary and level-score generation produces unrealistic tails;
- the decision is logged in the variable spec and remains fixed during training.

That still introduces a preprocessing rule, but it is a data-coordinate rule, not a model branch or evaluator hack.

## Proposed Next Step

Run 640a as a coordinate-policy audit before implementing a mixed-coordinate model:

- compute per-channel training-only diagnostics for level-score deltas versus encoded increments;
- compare 631a and 638a per-channel IV/factor failure signatures;
- propose a fixed generated-coordinate map based on stationarity/support criteria;
- only then implement 641a mixed-coordinate transition if the map is simple and defensible.

This prevents the loop from adding a research knob under pressure. The objective is to make the next implementation scientifically explainable before writing another model class.
