# 628a Encoded-Increment Coordinate Reset

## Hypothesis

The native joint38 branch may be failing because it trains on state-level paths.
Financial risk scenarios are naturally daily changes or returns integrated back
to levels. 628a keeps the same generic AR transition core but changes the model
coordinate to encoded daily changes for every channel:

- IV channels: log daily changes.
- Positive anchor levels: log daily changes.
- Diff-level anchors: arithmetic daily differences.

Generated paths are decoded by cumulatively integrating these changes from the
last observed state.

## Result

The IV full 11-suite fell to `3/11`.

Passed:

- conditionality
- block-AR
- cross-cell correlation

Failed:

- surface validity
- coverage
- time-series properties
- cointegration
- regime coverage
- distributional fidelity
- mean reversion
- pathwise jump realism

Key IV metrics:

- surface explosion rate: `76.6%`
- overall 90% coverage: `90.1%`, but per-cell coverage still failed
- conditionality MAE reduction: `8.6%`
- turbulent/calm width ratio: `1.205` informational
- daily-change KS pass cells: `22/25`
- level KS pass cells: `13/25`
- cross-cell corr/rank ratio: `0.719 / 1.397`
- mean-reversion aggregate ratio: `-0.149`
- pathwise max-jump KS: `0.724`
- pathwise q90/q99 jump ratio: `28.117 / 698.619`

The joint-panel audit improved sharply relative to 610a/612a/625a:

| model | factor KS mean | KS pass <0.20 | q99 ratio median | q99 pass [0.5,2.0] | factor-factor corr | factor-factor gen/gt mean abs | IV-factor corr | IV-factor gen/gt mean abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 610a state path | 0.133 | 11/13 | 2.365 | 2/13 | 0.565 | 0.054 / 0.225 | 0.664 | 0.075 / 0.149 |
| 612a state path + scale | 0.135 | 11/13 | 2.114 | 5/13 | 0.559 | 0.037 / 0.225 | 0.698 | 0.053 / 0.149 |
| 625a state RealNVP | 0.167 | 8/13 | 2.164 | 5/13 | 0.664 | 0.029 / 0.225 | 0.873 | 0.033 / 0.149 |
| 628a increment path | 0.116 | 12/13 | 1.397 | 12/13 | 0.720 | 0.140 / 0.225 | 0.849 | 0.113 / 0.149 |

Artifacts:

- `models/backfill/628a_joint38_ar_increment_e8_w2048_s628/best_model.pt`
- `results/autoresearch/628a_joint38_ar_increment_e8_w2048/full11.md`
- `results/autoresearch/628a_joint38_ar_increment_e8_w2048/joint_panel.md`

## Mechanism Read

628a validates the user's concern that anchor factors should be generated as
changes, not as separate level targets. The joint audit improved on every key
native joint-panel dependence and factor-change metric. This is the first native
joint38 learned model in this branch that looks directionally plausible for
anchor co-movement.

However, pure increment modeling is not enough. Because the model conditions on
recent increments but does not explicitly condition the transition law on the
current level/state, sampled log changes can compound for 30 days without enough
level anchoring. That creates IV explosions, invalid factor ranges for some
nonnegative anchor levels, weak mean reversion, and extreme pathwise jumps.

The failure is therefore clean:

- State-path models have level anchoring but attenuate joint changes and
  cross-factor dependence.
- Pure increment models learn joint changes much better but lose state anchoring
  and path-level realism.

## Decision

Do not ship 628a as a risk model. It is a decisive data-coordinate milestone, not
a deployable model.

The next principled move is a unified state-conditioned increment law:

- The generated variable remains the encoded daily change for every channel.
- The conditioning state includes recent levels and recent changes.
- The path is still reconstructed by integration from the observed state.
- There are no IV/factor-specific branches and no post-hoc deck gluing.

This is the canonical state-space formulation for the problem:
`p(delta_{t+1} | history of states and changes)`, followed by deterministic
state integration. It preserves the first-principles lesson from 628a while
adding the missing level anchor without introducing evaluator-specific knobs.
