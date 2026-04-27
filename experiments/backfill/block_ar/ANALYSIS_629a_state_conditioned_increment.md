# 629a State-Conditioned Encoded-Increment Law

## Hypothesis

628a showed that encoded daily changes are the right generated object for a
native 25+13 joint scenario model, but pure increment conditioning loses level
anchoring. 629a tests the canonical state-space form:

`p(delta_{t+1} | recent encoded levels, recent encoded changes)`

The model still generates encoded daily changes for every channel and integrates
them back to raw levels for evaluation. The only architecture change versus 628a
is that the causal memory sees both encoded levels and encoded changes.

## Result

The IV full 11-suite remains `3/11`, but the mechanism improved substantially.

Passed:

- block-AR
- cointegration
- cross-cell correlation

Failed:

- surface validity
- coverage
- conditionality
- time-series properties
- regime coverage
- distributional fidelity
- mean reversion
- pathwise jump realism

Key IV comparison:

| metric | 628a pure increment | 629a state-conditioned increment |
| --- | ---: | ---: |
| suite score | 3/11 | 3/11 |
| surface explosion rate | 76.6% | 45.1% |
| overall 90% coverage | 90.1% | 76.7% |
| conditionality MAE reduction | 8.6% | 7.9% |
| daily-change KS pass | 22/25 | 25/25 |
| level KS pass | 13/25 | 5/25 |
| sample ceiling rate | 2.106% | 0.154% |
| cross-cell corr/rank ratio | 0.719 / 1.397 | 0.967 / 1.167 |
| h1 mean-reversion ratio | -0.149 | 0.976 |
| pathwise max-jump KS | 0.724 | 0.464 |
| pathwise q90/q99 jump ratio | 28.117 / 698.619 | 2.580 / 6.542 |

The joint-panel audit improved again:

| model | factor KS mean | KS pass <0.20 | q99 ratio median | q99 pass [0.5,2.0] | factor-factor corr | factor-factor gen/gt mean abs | IV-factor corr | IV-factor gen/gt mean abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 628a increment path | 0.116 | 12/13 | 1.397 | 12/13 | 0.720 | 0.140 / 0.225 | 0.849 | 0.113 / 0.149 |
| 629a state-conditioned increment | 0.095 | 12/13 | 1.164 | 13/13 | 0.823 | 0.135 / 0.225 | 0.883 | 0.123 / 0.149 |

Artifacts:

- `models/backfill/629a_joint38_statecond_increment_e8_w2048_s629/best_model.pt`
- `results/autoresearch/629a_joint38_statecond_increment_e8_w2048/full11.md`
- `results/autoresearch/629a_joint38_statecond_increment_e8_w2048/joint_panel.md`

## Mechanism Read

The state-conditioned increment formulation is the cleanest native joint
direction so far. It keeps 628a's factor-change benefit while restoring much of
the missing level anchoring:

- IV daily-change realism is excellent (`25/25` KS pass).
- Cross-cell dependence is almost exactly calibrated by mean-correlation ratio.
- h1 mean reversion is correct (`0.976` ratio, active-cell corr `0.964`).
- Pathwise max-jump shape passes the relaxed KS gate.
- Anchor-factor tails and IV-factor co-movement are the best native joint result
  so far.

The remaining failure is no longer "wrong architecture"; it is calibration and
state-dependent scale:

- Overall coverage is too low at `76.7%`.
- Turbulent windows are not widened enough; turb/calm width ratio is `0.894`.
- Level occupancy is biased even though daily changes are good.
- Extreme jump scale is much closer than 628a but still too high in a few cells.

## Decision

Do not change paradigm. 629a is the right current pathology: one model, one
state-space transition, generated changes, deterministic integration, no
IV/factor branch, no deck gluing.

The next most principled move is to scale this same model before adding more
mechanism. Train 629b with the full available training window set instead of the
2048-window recent subset. This follows the Bitter Lesson and directly tests
whether the remaining calibration/regime failures are data-limited rather than
architectural.
