# NL Prefix Latent Soft Listwise Mixture Policy Intake

## Candidate Name

Short name: `soft_listwise_mixture_policy`

Workflow lane: `research_ideation` now; `exploration` after implementation.

## Local Bottleneck

The current learned support-policy branch has found a real oracle signal but has
not converted it into a deployable policy. The best-in-pool top-3 support
mixture is often better than the default cosine top-3, but linear, pairwise, and
support-set hard-selection policies do not generalize. The latest support-set
ranker can fit train preferences better with more capacity, but held-out
candidate-pool correlation remains near zero.

The likely bottleneck is not just item representation. The current policies
force a noisy hard decision: choose exactly one candidate subset. The frozen
generator then samples each selected analogue equally. This discards the
probability mass over nearby plausible support mixtures and makes small ranking
errors expensive.

## Method Story

A risk-manager narrative and fixed start retrieve a small candidate support
pool. Instead of selecting one hard top-3 subset, the policy learns a listwise
distribution over candidate mixtures using generator-response labels. At
inference, candidate probabilities are marginalized into support-window weights.
The frozen generator then samples from the support pool according to those
weights, preserving historical provenance while making the support mixture
softer and less brittle.

```text
narrative/start query
+ candidate support pool
+ generator-response labels during backtest training
-> listwise candidate probabilities
-> marginal support-window weights
-> weighted frozen SNI analogue rollout
-> CRPS / energy / coverage / audit comparison
```

The output remains an auditable support mixture. The method does not generate
free-form scenarios, does not remove the support store, and does not change the
frozen SNI generator.

## Related Work Basis

- RankNet frames ranking as learning a scoring function from within-query
  pairwise preferences. It supports the earlier pairwise attempt but also
  explains its limitation: pairwise comparisons do not model the full candidate
  list as the prediction object.
  Source: Microsoft Research, "Learning to Rank using Gradient Descent"
  (https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
- ListNet explicitly argues that ranking should be trained listwise, with lists
  as the learning instance and probability models over candidate orderings or
  top items. What transfers is the list-level loss over candidate support
  mixtures for one narrative/start query.
  Source: Microsoft Research, "Learning to Rank: From Pairwise Approach to
  Listwise Approach"
  (https://www.microsoft.com/en-us/research/?p=153086).
- Adaptive mixtures of local experts supports soft gating over multiple
  specialists rather than one hard expert. What transfers is the idea that a
  gating network can divide cases across local regimes and combine them softly.
  What does not transfer is the architecture itself: our "experts" are
  auditable historical support windows, not trainable decoders.
  Source: Jacobs et al., Neural Computation 1991
  (https://direct.mit.edu/neco/article/3/1/79/5560/Adaptive-Mixtures-of-Local-Experts).
- Prototypical Networks support a low-data, metric-space prior: simple
  prototypes can generalize better than heavier meta-learning in few-shot
  regimes. What transfers is optional regime/prototype grouping if listwise
  labels are heterogeneous.
  Source: NeurIPS 2017 Prototypical Networks
  (https://papers.nips.cc/paper/6996-prototypical-networks-for-fe).
- Deep Sets and Set Transformer remain relevant only if the candidate support
  pool must be encoded as a set. The first soft-listwise TestFlight should stay
  simpler than Set Transformer unless a local falsifier proves pairwise/listwise
  logits cannot model support weights.
  Sources: NeurIPS 2017 Deep Sets (https://papers.nips.cc/paper/6931-deep-sets)
  and PMLR ICML 2019 Set Transformer
  (https://proceedings.mlr.press/v97/lee19d.html).

## Novelty Claim

The local contribution is a narrative-and-start-conditioned listwise support
mixture policy trained on the frozen generator's own historical backtest
response. Unlike ordinary learning-to-rank, the ranked objects are not
documents; they are auditable historical support mixtures for a multivariate
financial scenario generator. Unlike support-free text-to-latent generation,
the method preserves provenance and lets grounding audits inspect the final
support weights.

## Elegance Check

This candidate is smaller and better aligned with the observed failure than
another ranker-capacity tweak:

- it changes hard subset selection into a soft distribution over existing
  support candidates;
- it uses the same generator-response labels already built;
- it exposes weights that risk managers can audit;
- it requires one product-relevant evaluator improvement: the scenario sampler
  must honor support weights rather than treating all retrieved analogues
  equally.

Out of scope for the first TestFlight:

- broad hyperparameter sweeps;
- new OpenAI labeling;
- generator fine-tuning;
- Set Transformer attention;
- support-free text-to-prefix generation.

## Training And Inference Contract

Training data:

- existing candidate-mixture labels from
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_rollout_response_train_mixture_labels_888a_128q`;
- each query has a list of candidate support subsets and frozen-generator
  energy/CRPS labels.

Training target:

- convert lower generator-response score into higher within-query target
  probability;
- use a listwise cross-entropy or KL loss against candidate probabilities;
- optionally regularize toward the incumbent narrative/start similarity weights
  to avoid unsupported jumps.

Inference:

- score candidate mixtures for the narrative/start query;
- convert scores to candidate probabilities;
- marginalize candidate probabilities into support-window weights;
- sort support windows by weight for display;
- run the frozen generator with weighted analogue sampling or an equivalent
  weighted support allocation.

Leakage guard:

- realized future labels are used only in historical backtesting;
- inference uses only narrative, grounding sidecar, fixed start, and support
  diagnostics available at run time.

## Required Baselines

- same-seed simple equal top-k mixture;
- current linear/pairwise/support-set hard-selection policies;
- incumbent `soft_topk_narrative_start_checked` /
  `narrative_generator_topk`;
- weighted-sampler-only ablation using incumbent similarity weights;
- non-deployable oracle listwise distribution from held-out candidate labels.

## Backtest Gate

Exploration gate:

- weighted sampler tests must prove that the evaluator actually honors support
  weights;
- listwise policy must improve held-out candidate-pool correlation or
  within-query top-probability rank versus the support-set ranker;
- scenario-level CRPS and energy must be competitive with same-seed simple
  equal top-k, not just coverage-wider.

Promotion gate:

- held-out CRPS and energy beat or match the simple mixture floor;
- coverage improvement is not the only win;
- support-weight provenance remains visible;
- fixed-start behavior, direction checks, and null/repeat controls remain valid;
- independent verifier reviews code, artifacts, and metric claims.

## Kill Condition

Stop this branch if:

- weighted analogue sampling itself worsens the incumbent when using incumbent
  weights;
- listwise probabilities still have weak or negative held-out candidate-pool
  alignment;
- scenario-level gains appear only as noisy coverage widening with worse CRPS
  and energy;
- the method requires a broad temperature or architecture sweep to look good;
- support weights become unauditable or collapse to a hidden top-1 analogue.

## Independent Verification Trigger

Independent verification is required before any default, paper-facing, or
production-demo claim. It is not required for the first weighted-sampler or
listwise-policy TestFlight unless the result is surprisingly strong.

Verifier artifact path: not yet applicable.

## Decision

Decision: prepare for a bounded TestFlight after adding TDD coverage for
weighted analogue sampling.

Decision rationale: the last failure analysis points away from larger hard
rankers and toward softer support allocation. This candidate directly targets
the brittleness of hard subset selection while keeping the historical mixture
as the auditable production backbone.

## Weighted-Sampler Baseline Result

Status: `sampler_ready_policy_not_tested`.

Artifacts:

- no-op softmax-cosine sampler:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_weighted_sampler_889g_softmax_cosine_fullheldout/scenario_level_eval_report.json`
- sharper softmax-cosine diagnostic:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_weighted_sampler_889h_softmax_cosine_t002_fullheldout/scenario_level_eval_report.json`
- comparison summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_weighted_sampler_889i_baseline_analysis/weighted_sampler_baseline_analysis.json`

Result:

- equal baseline CRPS/energy/coverage: `0.6882` / `0.9907` / `0.565`;
- softmax-cosine temperature `1.0`: identical to equal sampling because every
  held-out window keeps `[2,2,2]` sample allocation;
- softmax-cosine temperature `0.02`: CRPS `0.6894`, energy `0.9921`, coverage
  `0.570`;
- diagnostic allocation patterns at `0.02`: `[2,2,2]` for `16/29`, `[3,2,1]`
  for `11/29`, `[4,1,1]` for `2/29`.

Decision: weighted sampling is operational and tested, but naive cosine
sharpening is not useful. Keep equal sampling as the default. The next TestFlight
should train listwise weights from generator-response labels rather than assume
retrieval cosine should be sharpened.

## Minimal Listwise Policy Result

Status: `not_promoted_to_scenario_rollout`.

Artifacts:

- policy report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_listwise_mixture_policy_889j_128train_to_fullheldout/learned_mixture_policy_report.json`
- weighted bridge report:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_listwise_mixture_policy_889j_128train_to_fullheldout/learned_mixture_policy_bridge_report.json`
- candidate-pool post-analysis:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_listwise_mixture_policy_889k_candidate_analysis/listwise_policy_candidate_analysis.json`

Result:

- train score correlation with negative energy: `0.270`;
- train pairwise accuracy: `0.553`;
- held-out score correlation with negative energy: `0.093`;
- held-out score correlation with negative CRPS: `0.133`;
- held-out pairwise accuracy: `0.523`;
- exact held-out energy-oracle selection: `3/29`;
- selected-minus-default candidate-pool deltas: `+0.0048` energy and
  `+0.0022` CRPS;
- mean support effective `N`: `4.99`, meaning the marginal weights are almost
  uniform over the top support pool.

Interpretation: the listwise formulation is directionally better than the
support-set ranker because held-out correlations are positive again, but the
minimal linear policy is too weak and too uniform. Do not spend a full
scenario-level rollout on this version. The next mechanism question is whether
the label target is too diffuse, whether candidate features are too weak, or
whether a prototype/regime-conditioned listwise model is needed.

## Target-Diffuseness Analysis

Artifact:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_listwise_mixture_policy_889l_target_diffuseness/listwise_target_diffuseness_analysis.json`

Result:

- train target effective candidate count: `5.67` out of `10`;
- held-out target effective candidate count: `5.36` out of `10`;
- held-out target max probability: `0.330`;
- held-out best-minus-default label: `0.0594`;
- held-out best-minus-second label: `0.0280`;
- minimal listwise predicted effective candidate count: `9.89`;
- minimal listwise predicted max probability: `0.116`;
- strongest individual held-out feature correlation with negative energy:
  `support_cosine_min` at `0.129`.

Interpretation: the target distribution is not fully flat. It is only
moderately sharp, but materially sharper than the minimal listwise model's
predictions. The current bottleneck is therefore feature/model weakness, not a
pure label-diffuseness issue. A next candidate should either add a principled
regime/prototype conditioning layer or improve candidate features before making
the scorer more complex.
