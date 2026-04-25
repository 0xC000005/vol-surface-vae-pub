# 486 Density-Ratio Branch Postmortem

## Context

The 483-485 branch tested whether the current frontier model (`392a`) already
generates enough future-path support and only assigns the wrong probabilities to
candidate paths.

The branch was deliberately narrow:

- freeze `392a`;
- train one conditional density-ratio scorer;
- resample frozen proposal candidates;
- evaluate a small, principled logit-strength ablation.

No new path support was generated, and no validation-future oracle or regime label was
used.

## Results

| model | strength | score | failed suites | read |
| --- | ---: | ---: | --- | --- |
| `392a` | baseline | `8/11` | coverage, regime, distributional | active deployable frontier |
| `483a` | `1.0` | `6/11` | coverage, conditionality, cointegration, regime, distributional | reweighting too strong |
| `484a` | `0.5` | `7/11` | coverage, conditionality, regime, distributional | cointegration recovered; conditionality near gate |
| `485a` | `0.25` | `7/11` | coverage, conditionality, regime, distributional | weaker reweighting still below frontier |

Key metrics:

| model | cond MAE red. | level KS | coverage note | cointegration worst |
| --- | ---: | ---: | --- | ---: |
| `392a` | `5.14%` | `10/25` | fails per-cell coverage | `0.278` |
| `483a` | `3.47%` | `12/25` | nearly passes, one high cap | `0.175` |
| `484a` | `4.8%` | `13/25` | high caps remain | `0.281` |
| `485a` | `4.10%` | `12/25` | high caps remain | `0.361` |

## Mechanism

Density-ratio reweighting answered the support-versus-allocation question:

`392a` has some useful support for the missing outcomes. Reweighting improved level KS
from `10/25` to `12-13/25` and made coverage much closer to passing.

But frozen-proposal reallocation is not enough. It cannot push level KS to the `15/25`
gate, cannot solve regime layer-2 coverage, and tends to weaken the conditional MAE
advantage that made `392a` the frontier.

This means the missing object is not only probability assignment among existing
samples. The generator still needs a learned training signal that changes the
conditional law itself while preserving the local AR geometry.

## Why Not More Density-Ratio Knobs

Further logit strengths would be scalar tuning around a capped mechanism. The observed
frontier is stable:

- stronger weights improve allocation but damage conditionality/cointegration;
- weaker weights preserve structure but drift back toward `392a`;
- none of the tested strengths beats `392a`.

This is now too knob-like to keep pursuing under the clean-pathology rule.

## Next Direction

Return to the `392a` generator and change the proper scoring objective, not the
architecture.

The path-energy score used by `392a` is a multivariate proper score over a 750-dimensional
future path. That helped path realism but is known to be weakly discriminative in high
dimension. The hard remaining failures are marginal/cell-horizon level occupancy and
per-cell interval allocation.

The next clean falsifier is a marginal CRPS fine-tune:

```text
loss = FM anchor + small average CRPS over generated free-run samples,
       computed for every future horizon and cell in empirical score coordinates
```

This is still a proper scoring rule and still model-based. It is not a calibration table,
not a regime rule, and not an evaluator-specific KS loss. It directly trains the
conditional marginal distributions whose occupancy/coverage gates remain failed.

## Decision

Close density-ratio resampling as a primary route. Run one `487a` marginal-CRPS
fine-tune from the active `392a` checkpoint with a small CRPS weight and FM anchor.
If it improves level/coverage without losing conditionality, continue the objective
family. If it regresses, the next move should be broader paradigm ideation rather than
more scalar objective weights.
