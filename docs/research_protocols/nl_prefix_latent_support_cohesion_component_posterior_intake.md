# Support-Cohesion Component-Posterior Bakeoff Intake

Date: 2026-05-27

## Objective

Diagnose and improve risk-manager-visible narrative conditionality by testing
whether the current broad support pool is too heterogeneous. The experiment
compares support-selection policies and component-posterior distributions under
the same fixed start and professional narratives, then validates any promising
candidate against held-out scenario-quality guardrails.

## Method Story

The current support-grounded generator already selects different historical
support components for different professional narratives. The latest
component-aware audit shows that those components have much larger
cross-narrative path separation than the final broad pooled fan. This suggests
that conditionality may be diluted by pooling together support components that
are individually valid but belong to different sub-regimes.

The proposed method keeps the historical support store and frozen SNI rollout.
It changes the support posterior formation step:

```text
narrative + fixed start
-> candidate support bank
-> support-coherence selection
-> component-preserving rollout
-> component-posterior distribution
```

The key hypothesis is that a coherent support family can preserve narrative
signal better than a broad heterogeneous support pool while staying on the
historical market manifold.

## Candidate Support-Selection Policies

1. Current broad diverse support.
   Baseline incumbent: direction-checked, temporally non-overlapping,
   narrative/start-compatible broad support.

2. Most-similar cohesive support.
   Select the top narrative-compatible support, then choose additional
   non-overlapping supports that are most similar to that anchor in text-memory,
   generator-memory, prefix-signature, or a combined support feature space.

3. Cluster or family support.
   Cluster candidate supports into regime families, choose the best family for
   the narrative/start pair, and pool only inside that family.

4. Low-temperature similarity kernel.
   Keep a support set but sharpen weights around the most similar candidates so
   weakly related supports do not dilute the fan.

## Candidate Component-Posterior Distributions

For each support-selection policy, compare:

1. Full broad pooled distribution.
2. Top-1 component posterior.
3. Top-2 or 80% sparse posterior.
4. Top-3 or 90% sparse posterior.

## Evaluation

Use the same fixed-start professional narrative deck first, then extend to
multi-start and held-out backtests only after the mechanism is clean.

Conditionality metrics:

- visual raw-level and standardized fan separation;
- narrative-relevant factor terminal KS;
- path energy;
- portfolio VaR/ES or tail-loss spread;
- factor contribution changes;
- component-to-pooled signal-loss ratio;
- support overlap and support provenance.

Quality guardrails:

- held-out CRPS;
- energy score;
- 80% coverage;
- terminal MAE;
- direction checks;
- same-narrative repeat and start-only/null controls.

## Promotion Criteria

Promote only if a support-cohesion/component-posterior candidate:

- improves fixed-start narrative-visible conditionality versus the broad pooled
  incumbent;
- remains materially competitive on held-out CRPS, energy, coverage, and
  terminal MAE;
- preserves direction checks and support provenance;
- keeps the start-only/null control flat;
- passes independent verification before changing paper/demo defaults.

## Kill Conditions

Stop or downgrade the branch if:

- coherent support produces prettier fans but materially worsens CRPS, energy,
  coverage, or terminal MAE;
- sparse posterior improvement is only a top-1 replay effect with weak support
  diversity;
- support selection becomes a hidden nearest-neighbor replay system;
- the start-only/null control becomes nonzero;
- gains disappear under multi-start or held-out historical backtesting.

## Expected Decision

The bakeoff should decide whether component-aware views are only an explanation
layer or whether a sparse/cohesive component posterior can become the actual
product distribution.
