# 543a Frontier And Deployability Closure

## Current Frontier
The deployable learned frontier remains:

- `392a`: `models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt`, `8/11`;
- `510a`: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`, `8/11`.

Both fail the same hard suites:

- coverage;
- regime coverage;
- distributional fidelity.

The non-deployable feasibility proof remains:

- `435a`: oracle validation-future construction, `11/11`, not a learned conditional scenario generator.

## Recent Evidence Since 531a
Recent clean branches did not beat the frontier:

| Iteration | Idea | Score | Mechanism Read |
|---|---:|---:|---|
| 532a | coherent Gaussian future score-path density | `3/11` | explicit one-shot Gaussian density broke cross-cell/path structure |
| 535a | 51-variable panel Gaussian score-path law | `3/11` | adding factors as one-shot targets did not fix IV law |
| 537a | AR panel daily Cholesky transition | `4/11` | learned conditional covariance fixed cross-cell structure but not level occupancy |
| 538a | AR panel daily mixture transition | `3/11` | daily KS improved to `20/25`, but level KS stayed `1/25` |
| 540a | factor-conditioned patch-energy synthesis | `5/11` | structural geometry recovered, but level/regime/conditionality remained below frontier |

The result is consistent with earlier evidence:

- daily-change realism alone is not enough;
- one-step teacher-forced likelihood is not enough;
- local factor conditioning is constructive but insufficient;
- patch/rollout proper scoring preserves path geometry but does not break future level/regime occupancy;
- local `392a` repair variants are exhausted.

## Deployability Read
If this had to be shipped today, the honest deployable learned base is `392a` or `510a`, not the newer panel-likelihood or factor-patch variants.

However, it should not be represented as a complete risk scenario generator:

- it fails coverage and regime coverage;
- it does not match enough future IV-level marginal occupancy;
- the remaining failures are exactly the ones a risk manager would care about for conditional stress scenario reliability.

The right paper framing is:

```text
learned conditional scenario generator with strong structural realism,
not yet a fully calibrated risk system
```

If a risk-product framing is required, calibrated-system metrics must be reported separately from base learned-model metrics.

## Test-Suite Read
The 11-suite is not theoretically contradictory based on current evidence:

- oracle 435a reaches `11/11`, proving feasibility under the suite;
- several learned models can pass different subsets of the suite;
- failures are tradeoffs in learned conditional path law, not impossible gates.

The suite is strict, especially:

- per-cell coverage upper bounds;
- regime layer2 per-cell coverage;
- level KS;
- cellwise mean-reversion geometry;
- pathwise per-cell jump scale.

But these strict gates are aligned with the stated risk-manager objective: the generator should be conditional, not merely broad unconditional overcoverage.

## What Would Be Required To Continue
The next principled route is not another local model variant. It requires external data scale:

- multiple liquid option-surface universes;
- stable tenor/moneyness or tenor/delta grid construction;
- no-lookahead factor and underlying features;
- pretraining over many names/regimes;
- SPX held out or evaluated comparably with the current 11-suite.

Candidate sources are documented in `experiments/backfill/block_ar/IDEA_542a_external_option_surface_data_program.md`.

## Closure Decision
Do not continue local autoresearch by mutating losses, temperatures, source noise, factors, mixture heads, or calibration wrappers.

The clean local conclusion is:

- current learned deployable frontier: `8/11`;
- oracle feasibility: `11/11`;
- local data and model search: exhausted under the clean-pathology guard;
- next real progress requires external multi-underlying option-surface data or a revised deployment framing.

