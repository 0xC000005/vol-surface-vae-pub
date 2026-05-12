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
