# NL Prefix Latent Method Intake: Episode-Level Narrative Retrieval

## Candidate Name

Short name: `episode_level_narrative_retrieval`

Workflow lane: `exploration`

Iteration type: `research_ideation`, then `experiment` after the dry-run design
is accepted.

## Local Bottleneck

```text
Because fixed-start narrative conditionality can be weakened when a rich
professional narrative is compressed into one projected 128-dimensional memory,
the current pipeline can fail when the support-selector must distinguish
episodes that differ in sequencing, mechanism, or cross-asset interpretation
but are close in final-hidden SNI memory.

Because the current self-supervised `training_caption` field is often short and
factor-list-like, it can also fail when a user writes a sparse,
mechanism-driven narrative that mentions only one or two visible channels and
expects the system to infer the rest of the cross-asset regime.

This candidate should help because it first enriches each historical 30-day
episode into multiple risk-manager narrative views, then searches those episode
cards before applying numerical guardrails, instead of relying on the projected
memory or one compact caption as the only retrieval key.
```

## Method Story

A risk manager writes a professional narrative and selects a starting market
state. The system extracts grounded current/recent claims and warnings, then
searches a historical bank of multi-view narrative cards that describe full
30-day market episodes in professional language, sparse user-query language,
mechanism-first language, and factor-list baseline language. Candidate episodes
are scored by text similarity, grounded-claim agreement, SNI memory
compatibility, starting-level fit, and temporal diversity. The selected support
episodes are then fed through the same top3/90 component-preserving frozen SNI
rollout used by the incumbent method.

The full narrative remains the primary story channel. Grounding remains a
sidecar for direction checks and warnings. The fixed start remains upstream of
support selection. The historical support mixture remains visible and
auditable. The frozen SNI rollout path is unchanged.

## Related Work Basis

- Sentence-BERT supports efficient text-to-text semantic retrieval by embedding
  sentences into comparable vectors. This transfers to matching user narratives
  against historical episode captions. It does not by itself guarantee
  financial direction correctness, so direction checks remain required.
- Dense Passage Retrieval supports a dual-encoder retrieval architecture. This
  transfers to large support-bank recall. It does not replace reranking or
  downstream scenario validation.
- ColBERT supports late interaction, preserving token-level query/document
  detail better than a single pooled vector. This transfers to future
  fine-grained matching of phrases such as `VIX elevated`, `carry rebound`, or
  `liquidity-led rally`. It is optional for the first branch because it adds
  operational complexity.
- BEIR shows that retrieval robustness varies by domain and that lexical
  baselines can remain strong. This transfers as a warning to benchmark BM25 or
  lexical overlap against dense retrieval rather than assuming embeddings win.
- Reranking practice supports cheap first-stage retrieval followed by stronger
  pairwise reranking on a small candidate set. This transfers to Codex/LLM
  qualitative reranking on top candidates after local candidate recall. If
  pure semantic text similarity is not enough, this branch may use Codex-style
  agentic adjudication over a shortlist, but the reranker must return
  structured evidence, not just a hidden preference.
- NIST cautions against treating LLM judgments as ground truth relevance
  labels. This transfers directly: Codex can help rank or enrich candidates,
  but deterministic support checks and historical backtests decide promotion.
- Public macro-risk and market-commentary sources support the product premise
  that professional narratives usually emphasize a scenario spine, transmission
  channels, asset implications, confidence, ambiguity, and monitoring
  indicators rather than listing every observed factor move. This transfers to
  the multi-view episode-card schema, not to direct LLM-generated scenarios.

## Novelty Claim

```text
This is not just semantic search. The local contribution is an episode-level
support posterior for financial scenario generation because it combines
professional narrative-card retrieval with grounded-claim checks, fixed-start
compatibility, SNI memory compatibility, temporal non-overlap, and frozen SNI
component-preserving rollout.
```

## Elegance Check

- It does not remove the current support bank; it uses it more directly.
- It adds one retrieval front end because the current bottleneck is information
  loss before support selection.
- The new knobs are limited to interpretable score channels and views: text
  similarity, lexical/claim overlap, grounded direction agreement, SNI memory
  compatibility, start gap, temporal overlap, and explicit narrative-view type.
- Generator retraining and support-free direct text-to-scenario generation are
  deliberately out of scope. Online enrichment is in scope as a bounded later
  phase with leakage and source-audit guardrails.
- If it fails, the result is interpretable: either text-level episode matching
  does not improve support selection, or it improves retrieval but not rollout
  quality.

## Training And Inference Contract

Training data: existing professional RiskManagerCaptionV2 records, support-bank
windows, generated multi-view episode cards, and bounded online-enriched
episode-card variants. Additional caption generation is isolated in new
episode-retrieval scripts.

Inference inputs: user professional narrative, grounded claims/warnings, fixed
starting market state, historical episode cards, SNI memory bank.

Output object: ranked support episodes with component weights, support
provenance, direction-check status, start gap, and top3/90 rollout artifact.

Leakage guard: retrieval text fields describe only the current/recent 30-day
prefix. Online context must be timestamp-safe and must not describe post-window
realized market outcomes.

No hidden future information: realized next-30-day futures are used only for
validation metrics, never retrieval text.

No hidden model-chosen start: the user-selected or supplied start is fixed
before retrieval.

## Baselines

Required baselines:

- incumbent top3/90 projected-memory support selector;
- dense narrative-to-narrative selector;
- lexical or claim-overlap selector;
- hybrid episode selector;
- hybrid episode selector with sparse user-query views;
- online-enriched hybrid episode selector;
- start-only/no-narrative null;
- shuffled-narrative null.

Optional baselines:

- Codex/LLM rerank of top candidates;
- ColBERT-style late interaction;
- single-neighbor support;
- all-regime pooling calibration view.

## Backtest Gate

Primary metrics:

- fixed-start narrative-vs-start-only support lift;
- fixed-start narrative-vs-start-only generated distribution lift;
- fixed-start narrative sensitivity across risk channels;
- same-narrative repeat stability;
- shuffled-narrative or no-narrative null gap;
- support-direction audit pass/warn/fail rate;
- support-family cohesion and temporal diversity.

Guardrail metrics:

- held-out Energy Score;
- held-out ensemble CRPS;
- 80% coverage and interval behavior.

Promotion floor:

```text
The candidate must beat or be clearly competitive with the current top3/90
incumbent and the start-only support baseline. The branch's main product
differentiator is fixed-start conditionality lift over start-only. Historical
quality metrics are guardrails: small regressions are allowed only when
conditionality, provenance quality, or support-direction reliability improves
materially and the tradeoff is documented.
```

## Kill Condition

Kill or demote the branch if the hybrid episode selector fails to improve
support conditionality over the incumbent, or if any apparent gain disappears
under start-only/shuffled-narrative nulls. Demote online enrichment to a
diagnostic lane if it adds leakage risk, untraceable claims, or worse
scenario-level performance than the local multi-view episode cards.

## Independent Verification Trigger

Use independent verifier before:

- changing demo or paper defaults;
- claiming the method improves conditionality;
- scaling online enrichment;
- using Codex/LLM reranking as more than an exploratory sidecar.

Verifier artifact path:

`docs/research_protocols/nl_prefix_latent_verifier_reports/YYYY-MM-DD_episode_level_narrative_retrieval.md`

## Decision

Decision: `run_testflight`.

Decision rationale: the method is isolated, begins with a Phase 0 corpus-quality
and multi-view episode-card audit, includes Codex reranking and online
enrichment as explicit later phases, keeps the incumbent frozen SNI/top3/90
rollout, and has clear falsifiers before any default change.
