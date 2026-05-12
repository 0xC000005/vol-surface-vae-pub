# NL Prefix Latent Set-Ranker Method Intake

## Candidate Name

Short name: `support_set_item_ranker`

Workflow lane: `exploration`

Iteration type: `research_ideation` now; `experiment` only after this intake is
accepted.

## Local Bottleneck

Because the pairwise pooled-feature ranker has weak training correlation even
after scaling from `32` to `128` train-query labels, the current pipeline fails
when support-mixture quality depends on which support windows co-occur inside a
candidate set. The pooled summaries average away item identity and
interactions. This candidate should help because it scores a candidate mixture
as a small set of support items conditioned on the query narrative/start,
rather than as a hand-pooled vector.

## Method Story

A risk-manager narrative and fixed start first retrieve a small candidate pool
of historical support windows. The model then scores candidate support mixtures
by looking at each support item and how the items combine. The output is still
an auditable historical support mixture. The frozen SNI generator and native
autoregressive rollout remain unchanged.

```text
narrative/start query
+ candidate support set items
-> permutation-aware support-set ranker
-> selected or softly weighted support mixture
-> frozen SNI rollout
-> historical-backtest CRPS/energy/coverage
```

The full narrative remains a first-class input through the existing text-memory
features. Grounding remains a sidecar audit. The fixed start is already
resolved before candidate mixtures are scored.

## Related Work Basis

- RankNet supports the ranking framing: learn a scoring function from pairwise
  preferences instead of treating candidate rows as independent regression
  targets. What transfers is the within-query preference objective. What does
  not transfer is the application metric: our labels must remain frozen-generator
  scenario quality, not document relevance.
  Source: Microsoft Research, "Learning to Rank using Gradient Descent"
  (https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
- ListNet supports listwise training when the prediction object is an ordered
  list of candidates for one query. What transfers is the list-level candidate
  view. What does not transfer is the IR-specific relevance objective; our
  backtest still decides with energy, CRPS, coverage, and support audits.
  Source: Microsoft Research, "Learning to Rank: From Pairwise Approach to
  Listwise Approach"
  (https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach-2/).
- Deep Sets supports a minimal permutation-invariant architecture for set-valued
  inputs. What transfers is the structure `rho(sum(phi(item)))` for support
  mixtures whose item order should not matter. What does not transfer is a
  guarantee of better financial scenario quality; it only fixes the input
  representation class.
  Source: NeurIPS 2017 Deep Sets
  (https://papers.nips.cc/paper/6931-deep-sets).
- Set Transformer is the heavier alternative if pairwise interactions among
  support windows are essential. What transfers is attention over set elements.
  What does not transfer to the first TestFlight is the heavier architecture;
  use it only if the DeepSets-style model underfits with evidence.
  Source: ICML 2019 Set Transformer
  (https://icml.cc/virtual/2019/oral/4842).

## Novelty Claim

This is not just RankNet, ListNet, or Deep Sets. The local contribution is a
narrative-conditioned, generator-response-trained support-set policy for
financial scenario generation. It uses the project's narrative/start/support
contract to learn which auditable historical support mixtures are most useful
for the frozen scenario generator.

## Elegance Check

This candidate adds one justified component: a support-set item encoder for the
mixture scorer. It does not add OpenAI calls, does not change the frozen
generator, does not introduce a hidden start-selection step, and does not remove
the historical support mixture.

The first TestFlight should deliberately stay small:

- support-set item features from existing candidate rows;
- a shared item encoder plus sum/mean pooling;
- a query-level scorer over pooled support-set representation and query/start
  context;
- pairwise or listwise ranking loss using existing generator-response labels;
- no large hyperparameter sweep.

Out of scope for the first TestFlight:

- Set Transformer attention;
- generator fine-tuning;
- new OpenAI labeling;
- support-free latent generation;
- hand-authored market-rule features.

## Training And Inference Contract

Training data: existing 128-query generator-response mixture labels from
`nl_rollout_response_train_mixture_labels_888a_128q`.

Inference inputs:

- full narrative/text memory and fixed start features already available in the
  bridge;
- candidate support items and their per-item diagnostics;
- candidate support-set membership.

Output object: ranked or softly weighted historical support mixture.

Leakage guard: training labels may use realized futures only inside historical
backtesting. Inference-time scoring may not use realized future paths.

No hidden future information: future-looking narrative language remains
warning-only.

No hidden model-chosen start: the start is selected or supplied before support
mixture scoring.

## Baselines

Required baselines:

- `soft_topk_narrative_start_checked` / `narrative_generator_topk`;
- current 32-query pairwise pooled-feature ranker;
- current 128-query pairwise pooled-feature ranker;
- default candidate top-3 mixture;
- candidate-pool oracle, clearly marked as non-deployable upper bound.

Optional diagnostics:

- raw narrative only;
- implications only;
- narrative plus grounding sidecar;
- no-narrative or shuffled-narrative null when making conditionality claims.

## Backtest Gate

Exploration gate:

- train candidate score correlation with negative energy should clearly exceed
  the pooled-feature ranker (`0.153`) or the branch is likely still
  under-representing the support set;
- held-out candidate-pool correlation with negative energy and CRPS should
  improve over the 128-query pooled-feature ranker (`0.077` and `0.183`);
- exact or near-oracle selection should improve over `3/29`.

Promotion gate:

- held-out energy and CRPS must beat or be clearly competitive with the simple
  mixture;
- 80% coverage cannot improve only through noisy widening;
- support audits and fixed-start behavior must remain valid;
- repeat or seed control is required before paper-facing claims.

## Kill Condition

Stop this branch if:

- the set-aware model still cannot fit the 128-query training label surface;
- held-out candidate-pool alignment remains close to the pooled-feature ranker;
- scenario-level CRPS and energy remain worse than the simple mixture;
- gains require a broad hyperparameter sweep;
- the method becomes a hidden nearest-neighbor selector without auditable
  support weights.

## Independent Verification Trigger

Independent verification is required before any default or paper-facing claim.
It is not required for the first local TestFlight unless the result is
surprisingly strong.

Verifier artifact path: not yet applicable.

## Decision

Decision: `run_testflight` after implementation plan.

Decision rationale: the prior pairwise ranker failed even with 128-query labels,
and the utility analysis ruled out metric conflict as the main explanation.
The smallest justified next method is therefore a richer support-set
representation that remains compatible with the support-mixture contract.

## TestFlight Result

Status: `not_promoted`.

Artifacts:

- policy report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889b_128train_to_fullheldout/learned_mixture_policy_report.json`
- held-out scenario report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889c_128train_fullheldout_scenario_eval/scenario_level_eval_report.json`
- post-analysis:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889d_128train_post_analysis/support_set_ranker_post_analysis.json`

Result:

- train labels: `128` query windows and `1280` candidate mixtures;
- held-out scenario energy: `0.9896` versus simple mixture `0.9907`;
- held-out scenario CRPS: `0.6902` versus simple mixture `0.6882`;
- held-out 80% coverage: `0.580` versus simple mixture `0.565`;
- train score correlation with negative energy: `0.127`;
- held-out candidate-pool score correlation with negative energy: `-0.120`;
- held-out candidate-pool score correlation with negative CRPS: `-0.063`;
- exact held-out energy-oracle selection: `4/29`;
- selected-minus-default candidate-pool deltas: `+0.0147` energy and
  `+0.0197` CRPS.

Mechanism read:
`set_ranker_adds_capacity_but_still_misses_candidate_pool_oracle`.

Decision: do not promote. The support-set representation is more aligned with
the method story than pooled features, but the first DeepSets-style scorer does
not fit the generator-response label surface and does not produce reliable
held-out candidate-pool rankings. The next step should be post-experiment
analysis or a cleaner listwise/prototype objective, not a broad hyperparameter
sweep.

## Post-TestFlight Failure Analysis

Artifact:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_support_set_ranker_889e_failure_analysis/support_set_ranker_failure_analysis.json`

Result:

- default item ranker train correlation: `0.127`;
- default item ranker held-out correlation: `-0.120`;
- default item ranker train pairwise accuracy: `0.604`;
- default item ranker held-out pairwise accuracy: `0.509`;
- overfit stress setting (`hidden_dim=32`, `epochs=2000`, `l2=0`) train
  correlation: `0.279`;
- overfit stress held-out correlation: `-0.011`;
- overfit stress train pairwise accuracy: `0.693`;
- overfit stress held-out pairwise accuracy: `0.452`.

Interpretation: extra item-encoder capacity can fit the train label surface
better, but it does not produce held-out generator-response ranking signal. This
rules out a simple "make the DeepSets scorer larger" next step. The likely
problem is the hard candidate-selection objective or regime heterogeneity, so
the next candidate should be a cleaner listwise/soft-weighting or
prototype-aware support policy rather than another capacity tweak.
