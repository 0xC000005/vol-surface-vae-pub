# NL Prefix Latent Mixture Policy Method Story

## Context

The current simple mixture is strong and remains the production baseline. The
oracle mixture-label experiments show that better support mixtures exist inside
the same top-5 candidate pool, but the first linear pointwise mixture policy did
not recover those choices out of sample. The next method must therefore be
better justified than "add a stronger model."

## Proposed Candidate

Use a **query-relative listwise or pairwise set-ranker** for support-mixture
selection.

Workflow lane: `research_ideation` now; `experiment` only after this intake is
accepted.

The unit of prediction is not an isolated candidate row. For each
risk-manager narrative plus fixed starting level, the model receives a small
set of candidate support mixtures and ranks them. The selected mixture remains
a transparent historical support set. The frozen SNI generator and native
autoregressive rollout remain unchanged.

```text
narrative/start query
+ top candidate support mixtures
-> set/list ranker over candidate mixtures
-> selected or softly weighted support mixture
-> frozen SNI rollout
-> historical-backtest CRPS/energy/coverage
```

## Method Story

The product problem is naturally a ranking problem: given a narrative/start
query, choose the support mixture whose frozen-generator rollout is most
distributionally useful. The first linear policy treated each candidate mixture
as an independent regression row, but the labels are only meaningful relative
to the other candidate mixtures for the same query. A pairwise/listwise ranker
matches the task better because it learns within-query preferences such as
"mixture A should rank above mixture B for this narrative/start."

The candidate mixture itself is a small set of historical supports. Its score
should not depend on arbitrary item order. A DeepSets-style pooled support
representation is the simplest permutation-aware architecture; a Set
Transformer-style attention scorer is a later upgrade only if the pooled model
shows a clear local bottleneck.

## Related Work Basis

- RankNet frames ranking as learning pairwise preferences with a neural scoring
  function, which matches the within-query support-mixture comparison we need:
  Microsoft Research, "Learning to Rank using Gradient Descent"
  (https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
  What transfers: pairwise preferences are the right supervision when only
  relative candidate quality matters inside a query. What does not transfer:
  web-search ranking metrics do not define financial scenario quality; our
  preference labels must come from frozen-generator rollout response.
- ListNet argues that ranking should be learned over lists rather than only
  isolated pairs, which supports training on all candidate mixtures for a query:
  Microsoft Research, "Learning to Rank: From Pairwise Approach to Listwise
  Approach"
  (https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/).
  What transfers: the candidate mixtures form a list per narrative/start query,
  so listwise losses match the decision object better than pointwise
  regression. What does not transfer: top-k document relevance is not enough;
  the loss must stay tied to energy/CRPS/coverage labels from the scenario
  generator.
- Deep Sets provides the minimal permutation-invariant structure for set-valued
  inputs, appropriate for candidate support mixtures whose order should not
  matter: NeurIPS 2017 Deep Sets
  (https://papers.neurips.cc/paper/6931-deep-sets).
  What transfers: a support mixture is a set of windows, so pooled set features
  avoid arbitrary ordering. What does not transfer: a pooled set encoder alone
  does not solve narrative conditioning; it is only the support-mixture
  representation inside the ranker.
- Set Transformer is the heavier attention-based alternative for set inputs,
  useful only if interactions among support windows matter beyond pooled
  summaries: PMLR ICML 2019 Set Transformer
  (https://proceedings.mlr.press/v97/lee19d.html).
  What transfers: attention can model pairwise interactions among support
  windows. What does not transfer: it is too heavy for the first TestFlight
  unless the DeepSets-style pooled ranker shows a specific interaction
  bottleneck.

## Novelty Claim

This is not a generic text-to-time-series generator and not a pure K-nearest
neighbor system. The local contribution is a narrative-conditioned,
generator-response-trained support-mixture policy:

- language and fixed start define the candidate support region;
- the learned ranker chooses or weights historical support mixtures using
  labels from the frozen generator's own rollout quality;
- provenance remains auditable because the output is still a support mixture;
- historical backtests test whether the language-conditioned mixture improves
  distributional scenario quality.

## Elegance Check

This method replaces the failed pointwise linear policy with a model class that
matches the actual decision structure. It does not add a new generator, does
not remove historical support, does not ask text to hallucinate a 30-day hidden
prefix, and does not add another market-rule feature. The simplest TestFlight
should use:

- the existing candidate-mixture label artifacts;
- within-query pairwise or listwise loss;
- small pooled support-mixture features;
- no OpenAI calls;
- the same held-out historical-backtest gate against the simple mixture.

Added complexity is limited to one component: the candidate-mixture scorer.
There is no new text-labeling prompt, no new market-rule feature, no generator
fine-tuning, and no hidden start-selection policy. The method is elegant only if
it improves the existing support-selection decision. If it requires additional
manual thresholds, a large sweep, or new ad hoc direction features to work, the
candidate should be rejected or returned to ideation.

## Training And Inference Contract

Training data: existing train-window generator-response mixture labels, built
without new OpenAI calls.

Inference inputs: full narrative/text memory, fixed selected or supplied start,
candidate support mixtures, and inference-available support diagnostics.

Output object: ranked or softly weighted support mixture, not a standalone
generated prefix and not a hidden single analogue.

Leakage guard: labels come only from historical backtesting during training and
evaluation; inference-time scoring must not use realized future paths.

No hidden future information: future-looking language remains warning-only.

No hidden model-chosen start: the start must be selected or supplied before the
ranker scores candidate mixtures.

## Backtest Gate

Promotion requires same-seed held-out comparison against
`soft_topk_narrative_start_checked` / `narrative_generator_topk`:

- energy score must improve or be clearly competitive;
- ensemble CRPS must improve or be clearly competitive;
- coverage must not improve by merely widening noisy distributions;
- fixed-start and support-audit behavior must remain valid;
- same method must pass at least one seed-repeat check before paper claims.

The first TestFlight is allowed to be exploratory, but it must still report:

- candidate-pool rank correlation with negative energy and negative CRPS;
- exact-or-near-oracle mixture selection rate;
- held-out scenario-level energy, CRPS, and coverage versus the simple mixture;
- whether any gain survives fixed-start and shuffled-narrative controls if the
  claim is conditionality.

## Kill Condition

Stop this branch if the pooled pairwise/listwise ranker still has weak
held-out candidate-pool alignment with generator-response labels, for example:

- score correlation with negative energy remains below `0.20`;
- exact or near-oracle selection remains close to random;
- held-out scenario-level CRPS/energy remain worse than the simple mixture;
- gains come only from coverage while CRPS/energy and terminal errors regress.
