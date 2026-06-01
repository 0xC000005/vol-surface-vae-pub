# Portfolio-Response-Aware Support Policy Intake

## Candidate Name

Short name: `portfolio_response_kernel_listwise_921a`

Workflow lane: `candidate`

Iteration type: `experiment`

## Local Bottleneck

Because the latest product conditionality audit shows that narrative signal
reaches support selection and decoded prefixes, but final factor distribution
and portfolio-tail separation remain warning layers, the current pipeline fails
when support weights that look semantically plausible do not produce enough
risk-manager-visible portfolio response after frozen SNI rollout. This
candidate should help because it learns support weights from historical
generator-response labels in portfolio-risk space, not merely because it adds a
more complex ranker.

## Method Story

The risk-manager narrative and fixed starting level still define the candidate
support pool. The full narrative remains the primary condition channel;
grounding remains a sidecar check. The candidate does not change the frozen SNI
generator, does not choose a hidden start level, and does not invent scenarios.

For each historical training query, existing candidate support mixtures are
scored by how well their generated paths match the realized future in normalized
portfolio-risk books. A compact kernel-listwise policy then reweights candidate
support mixtures at inference. The output remains an auditable historical
support mixture with explicit weights.

## Related Work Basis

- Learning-to-rank: Listwise ranking treats the whole candidate list as the
  learning instance rather than isolated pairs. This transfers directly because
  each narrative/start query has a list of candidate support mixtures. It does
  not transfer as a document-retrieval objective; our labels come from frozen
  generator backtests rather than clicks or relevance judgments.
  <https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/>
- Retrieval-augmented generation: RAG motivates combining parametric models
  with explicit non-parametric memory and provenance. This transfers to our
  support-store design. It does not mean an LLM is allowed to produce the
  numerical scenario distribution.
  <https://papers.nips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html>
- Ensemble copula coupling and dynamic ECC: these methods preserve scenario
  dependence structures after calibration/postprocessing. This supports the
  component-preserving support-mixture framing. It does not transfer literally
  as a weather forecast copula method; our dependence source is the frozen SNI
  rollout over selected historical prefixes.
  <https://arxiv.org/abs/1305.3445>
  <https://orbit.dtu.dk/en/publications/generation-of-scenarios-from-calibrated-ensemble-forecasts-with-a/>

## Novelty Claim

This is not just learning to rank or retrieval. The local contribution is a
portfolio-risk-response label for narrative-conditioned support mixtures,
because it uses the narrative/start/support/frozen-generator contract to learn
which auditable historical support mixtures produce useful future risk
distributions.

## Elegance Check

- It reuses the existing support-policy machinery and scenario evaluator.
- It adds one label surface, not a new generator, new text model, or new hidden
  start selector.
- The new operational knob is the portfolio book set. For this TestFlight, the
  books are fixed to the previously reliable response books:
  `equity_beta_carry`, `dollar_liquidity`, and `short_volatility`.
- Full per-factor rules, a hand-tuned portfolio sweep, and direct decoder
  fine-tuning are out of scope.
- If it fails, the result is interpretable: either portfolio labels are too
  noisy, or the current deployable features cannot learn the generator-response
  surface.

## Training And Inference Contract

Training data: existing 906e train candidate-mixture rollouts and realized
future paths.

Inference inputs: narrative/start candidate support report, bridge arrays, and
historical support features available before future realization.

Output object: a reranked bridge report with support weights in
`top_train_pool`.

Leakage guard: portfolio labels are built only for training from historical
realized futures; inference uses the learned kernel-listwise policy and does
not access future paths.

No hidden future information: required.

No hidden model-chosen start: required.

## Baselines

Required baselines:

- equal/simple support mixture floor on the same 66 held-out windows;
- previous generator-response kernel-listwise policy;
- persistence and historical replay as reported by the scenario evaluator.

Optional baselines:

- direct text-predicted memory without support;
- oracle soft support weights.

## Backtest Gate

Primary metrics:

- held-out ensemble CRPS and energy score;
- 80% coverage;
- reliable portfolio path score, CRPS, and energy;
- support-weight non-uniformity;
- comparison to the equal/simple mixture floor.

Promotion floor:

The candidate must beat or be clearly competitive with the simple mixture.
Small regressions are allowed only if portfolio response improves materially
and the trade-off is documented. A portfolio-only gain with energy/CRPS
regression remains diagnostic.

## Kill Condition

Kill before promotion if held-out CRPS and energy both worsen versus the equal
mixture, if portfolio gains are below repeat/bootstrap noise, or if gains appear
only in one hand-picked exposure book.

## Independent Verification Trigger

Use independent verifier before changing demo or paper defaults, claiming
production readiness, or promoting this support policy.

Verifier artifact path:
`docs/research_protocols/nl_prefix_latent_verifier_reports/portfolio_response_support_policy_921a.md`

## Decision

Decision: `run_testflight`

Decision rationale: The method is the smallest deployable candidate after the
portfolio-risk response-label audit: it tests whether the risk-manager-facing
response signal can improve support weights without changing the frozen
generator or abandoning auditable historical support.
