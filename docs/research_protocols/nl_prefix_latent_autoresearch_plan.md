# Narrative Prefix-Latent Autoresearch Plan

## Objective

Build a risk-manager-facing narrative-conditioned scenario generator that uses
historical analogue mixtures as an auditable latent support prior, then learns a
narrative-and-start residual refinement before reusing the frozen state-aware
normalized-innovation (SNI) conditional scenario generator and its native
autoregressive rollout.

Current boss-demo runbook:
`docs/research_protocols/nl_prefix_latent_boss_demo_runbook.md`.

Tracked current goal:
`docs/research_protocols/nl_prefix_latent_goal.json`.

Tracked current-truth index:
`docs/research_protocols/nl_prefix_latent_current_truth.md`.

Current text-to-latent prior research note:
`docs/research_protocols/nl_prefix_latent_text_to_latent_prior_plan.md`.

Current structured text/start fusion note:
`docs/research_protocols/nl_prefix_latent_structured_text_start_fusion_plan.md`.

The long-run product contract has one fixed-start requirement with two start
input routes:

1. **Manually selected historical/current start.** The app can expose historical
   or current joint39 start states for browsing, but the risk manager manually
   selects the level. The system must not silently choose the start as a hidden
   conditioning input.
2. **User-specified joint39 start.** The risk manager supplies a narrative plus
   the current or hypothetical starting level for the joint scenario factors.
   The supplied level is treated as the fixed initial condition.

After the initial level is fixed, both modes share the same contract:

```text
narrative + grounding sidecar + fixed initial joint39 level
-> narrative-compatible and start-compatible analogue pool
-> soft mixture of plausible 30-day recent prefixes
-> bounded residual/refinement or decoder condition
-> frozen SNI autoregressive rollout
-> 30-day future scenario distribution
```

Historical analogues are now the main support prior, but not as a single
nearest-neighbor replay engine. The production path should retrieve a set of
narrative-relevant and start-compatible historical regimes, form a soft mixture
in latent or prefix space, and learn a residual refinement from the narrative
plus fixed starting state. The mixture is therefore not upstream of the initial
level. It is conditioned on the level the user supplied or manually selected.
The system must expose the analogue weights, support diagnostics, and
post-rollout implication checks so a risk manager can see whether the generated
distribution is supported, weakly supported, or rejected.

## Workflow Revision: Grounding Is A Sidecar, Not The Narrative

The grounding layer must not collapse the user's narrative into only a few
up/down labels. A risk-manager story can contain regime shape, fragility,
liquidity language, catalyst context, and uncertainty about mechanism. Those
fields can matter for support selection even when they are not directly
observable as one of the joint39 factors.

The production representation is therefore multi-channel:

```text
full narrative text
+ condition-only market implications
+ grounding warnings / unsupported claims / forward-risk sidecars
+ accepted joint39 start
-> support mixture and frozen rollout
```

The explicit market implications are still required because they provide
directionality, temporal discipline, and auditability. But they are not the
whole condition. They are a sidecar that constrains and explains the full
narrative channel. The support mixture should score candidates with at least
three separable channels:

- full-narrative similarity, which preserves story nuance and regime language;
- explicit-implication alignment, which prevents semantic collapse such as
  "rates up" and "rates down" landing in the same bucket;
- fixed-start compatibility, which keeps the autoregressive rollout in the
  generator's valid domain.

Every future bridge or support-mixture experiment should include a grounding
bottleneck ablation:

1. raw narrative only;
2. implications only;
3. current grounded condition text;
4. raw narrative plus explicit implications;
5. raw narrative plus full grounding sidecar.

If implications-only wins, the narrative is probably not adding usable signal.
If raw narrative or narrative-plus-grounding wins, the workflow must preserve
the richer story channel. If raw narrative wins but produces temporal leakage
or poor warnings, the product should keep grounding as a guardrail while using
full narrative for support scoring.

For the production path, the preferred default is stricter than a single
concatenated score:

```text
full narrative embedding + fixed start -> support mixture selection
grounded implications -> selected-support direction check
grounded implications -> final mixed-prefix direction check
```

That means the narrative should be allowed to select nuanced support by itself,
with start compatibility included because the frozen generator needs a valid
initial level. Grounding should then check whether the selected historical
prefixes and their weighted mixed prefix actually move in the extracted
directions. For example, if grounding says volatility is compressing, the
support audit should report the weighted fraction of selected support windows
whose IV/VIX proxy is down, and the final mixed-prefix audit should report
whether the mixture itself has a negative volatility terminal delta. If the
final mixed prefix violates a checkable high-confidence direction, the run is a
reject. If individual support is weak but the mixed prefix passes, the run is a
warning. Embedding-only conditions are therefore insufficient for production
unless they are tied back to an auditable prefix/support object or a validated
decoder/probe.

The current production default for the demo path is
`soft_topk_narrative_start_checked`: rank candidate support by full-narrative
memory similarity plus fixed-start compatibility, then keep only directionally
passing candidates when checkable grounded motion claims exist. This differs
from `soft_topk_combined`, which lets implication alignment continuously change
the score. The checked mode is the preferred trust contract because the
narrative remains the ranking channel and grounding acts as a pass/reject
audit gate.

Warnings and unsupported claims should remain visible audit fields. They should
be embedded only when the experiment is explicitly testing a full-sidecar
channel, because concatenating warning text into the retrieval embedding can
add noise even when the warning is correctly excluded from numerical
conditioning.

## Workflow Revision: Start Before Mixture

The narrative is a description of the current and recent market condition, not
an instruction for what the future must look like. Phrases such as "the risk is
that equities keep falling" or "a volatility reversal could unwind the rally"
are useful risk-manager language, but they are warnings and scenario concerns,
not conditioning targets. The grounding layer must therefore separate:

- current or recent market implications that can be used for support retrieval;
- unsupported causal claims that require warnings;
- forward-looking or desired-future language that must be excluded from the
  conditioning text.

The start level is resolved before any prefix mixture is formed. The app may
show historical/current start candidates for browsing, but a candidate is not a
conditioning input until the risk manager explicitly selects it. If the user
supplies a joint39 level directly, that level is already the fixed start. Only
after this point should the model retrieve and weight historical prefix support:

```text
narrative -> condition-only grounding
user-selected/user-supplied s0 + full narrative + grounding sidecar -> analogue pool and mixture weights
mixture prefix ending at s0 -> frozen SNI rollout -> future distribution
```

This ordering is now part of the trust contract. The same narrative paired with
two different starting levels may legitimately produce different analogue
weights, different recent-prefix mixtures, and different future distributions.
The product must show the fixed start, the start-support diagnostics, and the
analogue weights before presenting the generated scenario fan.

## Production-Readiness Workflow

This protocol now treats research progress and production readiness as separate
but connected obligations. A new model result is not enough. Each iteration
should move at least one of these product gates:

1. **Narrative plus grounded sidecar input.** The risk-manager story is kept as
   full narrative text and also converted into explicit market implications,
   unsupported-claim warnings, forward-risk sidecars, model metadata, and
   cached text conditions.
2. **Fixed-start mixture contract.** The model first accepts either a manually
   selected historical/current start or an explicit user-specified joint39
   start. Only after that level is fixed does it build the
   narrative-conditioned analogue mixture, then decode a recent-prefix object
   that ends exactly at that start.
3. **Native frozen rollout.** The decoded prefix is fed through the frozen SNI
   generator's native encoder and autoregressive rollout, not a stale fixed
   condition vector.
4. **Scenario quality and calibration.** The resulting distribution is compared
   against persistence, replay, analogue-conditioned generation, and true-prefix
   oracle baselines with energy score, CRPS, coverage, and sensitivity metrics.
5. **Trust and validation.** Each run reports endpoint pinning, memory
   compatibility, start support, rollout sensitivity, warning/failure counts,
   and hard cases.
6. **Risk-manager UI.** The demo must show story, implications, provenance,
   validation status, scenario fans, selected IV-cell views, and artifacts in a
   form a non-ML risk manager can inspect.

The next production milestone is a **mixture-supported live story path with
grounding-bottleneck evidence**. It should retrieve several
narrative-consistent analogue prefixes, blend them into a support prior, apply
a bounded latent residual or candidate refinement, and verify that the full
narrative plus grounding sidecar performs at least as well as implication-only
conditioning while retaining auditability and temporal leakage controls.

## Current Problem

The current narrative bridge maps a text embedding to the frozen generator's
last 128-dimensional prefix memory. That is useful for analogue retrieval and
diagnostics, but it is not a native prompt-conditioned rollout contract. The
SNI generator recomputes memory at each autoregressive step from the evolving
prefix. Reusing one fixed memory vector turns the model into repeated
one-step sampling under a stale condition.

The other extreme, directly generating the full 30-day hidden prefix from text,
is too ambitious as the main production path at the current data scale. It asks
language to invent a detailed latent trajectory that the user usually did not
specify. A more defensible contract is to use historical neighbors as a
manifold-supported prior, mix multiple analogues rather than copying one, then
learn the smallest residual adjustment needed to satisfy the story and starting
state.

The direct-memory and residual-memory ablations confirm the mechanics:

- fixed text-predicted memory can drive the decoder mechanically, but it is not
  equivalent to native autoregressive generation;
- residual prompt conditioning is safer because it keeps the evolving-prefix
  encoder alive, but a single retrieved prefix still carries too much of the
  scenario identity;
- implication-aligned start selection and simple rollout reranking did not fix
  directional mismatch, so the next clean formulation is a **mixture-supported
  prefix prior plus learned residual refinement**.

## Main Model Family: Analogue-Mixture Latent Refinement

The central object is a **mixture-supported prefix latent**:

```text
narrative -> user-selected or user-specified starting state s0
narrative + fixed s0 -> start-compatible analogue pool -> soft mixture prior
soft mixture prior + narrative + fixed s0 -> residual-refined prefix latent
residual-refined prefix latent -> synthetic recent-prefix state
synthetic recent-prefix state -> frozen SNI encoder/rollout -> future scenarios
```

The analogue mixture should represent the historically supported region of the
30-day recent-regime manifold. The learned residual should adapt that support
prior to the user's narrative and starting state without pretending that text
alone determines every hidden-state token. The decoded prefix must provide the
fields needed by the frozen SNI encoder, including level history, normalized
innovation or flow-coordinate history, and the history-derived
center/scale/drift conditioning features.

### Stage A: Prefix Dataset

Create a training table from the existing joint39 windows:

- `source_index` and `window_index`;
- 30-day historical prefix in joint39 level coordinates;
- normalized innovations / flow coordinates;
- last-day starting state `s0`;
- center, scale, and drift features used by the frozen SNI generator;
- frozen SNI memory sequence if needed for diagnostics;
- existing OpenAI narrative bundle IDs and text embeddings.

The dataset should preserve manifest train/validation/test splits.

### Stage B: Prefix Autoencoder / Latent Bank

Train a compact prefix autoencoder over the generator-input prefix object:

```text
prefix object -> prefix encoder -> z
z + starting state -> prefix decoder -> reconstructed prefix object
```

The decoder must be start-aware. If the risk manager pins today's starting
point, the decoded prefix should end at that starting point. Internally this can
be implemented by decoding backward increments, residualized level paths, or a
latent trajectory in normalized-innovation space.

First acceptance gate:

- reconstructed prefix reproduces the frozen encoder's last memory with high
  cosine similarity;
- decoded-prefix rollout is close to true-prefix rollout under the frozen
  generator;
- reconstruction respects the final starting state and produces finite
  center/scale/drift features.

If this fails, text conditioning is premature.

The encoder also defines the latent bank used for analogue mixtures: each
historical prefix has a latent `z`, a starting state `s0`, metadata, realized
future, and narrative/text condition when available.

### Stage C: Narrative to Analogue-Mixture Prior

After `s0` is fixed, retrieve a pool of candidate prefixes by combining:

- text-memory similarity to the grounded narrative;
- explicit market-implication alignment in the recent prefix;
- starting-state compatibility with the fixed selected/supplied `s0`;
- support diversity so the pool does not collapse to nearly identical windows.

Convert the pool into a soft prior:

```text
{z_i, prefix_i, support_i}_{i=1..k} -> weights w_i -> z_mix or prefix_mix
```

The system should report the analogue weights and why each analogue was used.
The mixture may be formed in prefix-latent space first; direct prefix-space
mixing is allowed only as a baseline because it may average away regime shape.

### Stage D: Residual Latent Refinement

Train or evaluate a supervised residual bridge:

```text
text embedding + starting state + z_mix + support diagnostics -> delta_z
z_refined = z_mix + delta_z
```

Positive pairs are multiple narratives for the same historical window. Hard
negatives include opposite-direction narratives, nearby but directionally
different windows, and windows with similar starting levels but different recent
regime paths. The residual should be bounded or regularized so it stays near
the support prior unless the validation gate clearly marks the case as
out-of-distribution.

The loss should begin simple:

- latent regression or cosine loss to the target prefix-autoencoder latent;
- residual-size penalty or trust-region constraint around `z_mix`;
- CLIP-style / InfoNCE contrastive alignment between text-start-mixture
  embeddings and prefix latents;
- implication-alignment loss or post-rollout selection metric using explicit
  market implications;
- optional hard-negative margin only if the simple bridge collapses.

The refined latent is decoded to a synthetic prefix and passed through the
unchanged SNI autoregressive sampler.

### Stage E: Starting-Level Contract

For manually selected historical/current starts:

- display searchable or curated historical/current `s0` states to the risk
  manager;
- treat only the user's selected start as the conditioning input;
- build the analogue mixture conditioned on both the narrative and that selected
  `s0`;
- run the mixture-supported residual generator from that fixed start.

For user-specified starts:

- accept the provided joint39 starting state;
- build the analogue mixture and predict the prefix latent conditional on the
  same narrative and that explicit start;
- decode a compatible recent-prefix state;
- run the frozen generator normally.

This makes the analogue mixture a transparent prior rather than a hidden
nearest-neighbor generator. The refined latent is the scenario condition; the
analogue pool is provenance, plausibility support, and a guardrail.

There is no model-chosen start in the production default. A future start
recommendation feature would need its own product gate, user approval step, and
evaluation set; it must not be folded into the core narrative-conditioning
claim.

## Evaluation Plan

Use the existing representative manifest and scenario-level evaluation harness,
then add prefix-latent baselines in this order:

1. **Oracle decoded prefix.** Encode true historical prefix to `z`, decode it,
   and run the frozen generator. This tests whether the prefix autoencoder
   preserves generator-relevant information.
2. **Single-neighbor analogue.** Use the best retrieved analogue prefix to
   measure the KNN-style baseline explicitly.
3. **Analogue-mixture prior.** Blend top-k analogue latents/prefixes without a
   learned residual and roll out.
4. **Mixture plus residual.** Predict a residual from narrative, starting
   state, and mixture diagnostics, then decode and roll out.
5. **Fixed-start narrative sensitivity.** Hold the selected start fixed and vary
   narratives, then hold the narrative fixed and vary starts. The scenario
   distribution should change for the right reason in both directions.
6. **Ablations.** Compare against historical replay, persistence, current
   analogue-top-k generation, direct memory, residual memory, raw text embedding
   retrieval, no-contrastive bridge, contrastive bridge, mixture without
   residual, and residual without mixture support.
7. **Null and repeat controls.** Before promoting a fixed-start conditionality
   claim, include controls that separate narrative signal from sampling noise
   and support-pool geometry:
   - same narrative, same fixed start, different rollout seeds;
   - shuffled narratives assigned to the same fixed starts;
   - no-narrative or memory-only support mixture;
   - same narrative across different fixed starts.

Primary metrics:

- energy score and ensemble CRPS versus persistence;
- 80% coverage and interval score;
- mean-path MAE as a secondary diagnostic, not the publication target;
- memory cosine between true, decoded, and predicted prefixes;
- hard-case narrative validation and directional contrastive rank;
- sensitivity to changing the user-specified starting state under the same
  narrative.

Promotion criterion:

The mixture-supported residual system must beat or match the single-neighbor
analogue-conditioned generator on distributional scenario metrics and reduce
explicit implication mismatch versus the current `0.50` balanced-start and
`0.4706` two-candidate-reranker baselines. Exact historical-window retrieval is
not the target; auditable support plus story-consistent distributions is.

For any paper-facing, demo-facing, or default-promoting fixed-start
conditionality claim, promotion additionally requires:

- a tracked case spec or manifest under
  `docs/research_protocols/nl_prefix_latent_promoted_specs/`;
- a verifier report under
  `docs/research_protocols/nl_prefix_latent_verifier_reports/`;
- a current-truth update in
  `docs/research_protocols/nl_prefix_latent_current_truth.md`;
- fixed-start equality;
- per-start quality floors, not only mean quality across all starts;
- null/repeat controls or a documented reason they are not yet available;
- visible start-damping pass/warn/fail status.

## HEAD Loop Setup

Use the repo's HEAD discipline:

- **Hypothesis:** state one falsifiable claim about prefix-latent conditioning.
- **Execute:** run one focused implementation, analysis, or experiment.
- **Analyze:** compare against saved artifacts and baselines.
- **Decide:** update persistent state, append the research log, and recommend
  the next iteration.

### Autoresearch Guardrails

The workflow should learn from autonomous-research systems without copying
their failure modes. Related systems such as AI Scientist, Agent Laboratory,
and Robin show that autonomous loops are useful when they include literature
review, experiment execution, reporting, and review. Independent evaluations of
AI Scientist also show recurring risks: weak novelty checks, failed or
misleading experiments, stale citations, hallucinated numbers, and structural
report errors.

The current online multi-agent review is tracked at
`docs/research_protocols/nl_prefix_autoresearch_multiagent_review.md`. It is a
standing policy artifact for this workflow, not a one-off note.

Before adopting a new model family, bridge objective, agent workflow, or
deployment architecture, create a short related-work artifact under
`docs/research_protocols/` or the relevant experiment output directory. The
artifact should contain:

- sources checked, preferably primary sources;
- what transfers to this project;
- what does not transfer;
- the mechanism being tested;
- the falsifier;
- the implementation constraint imposed by the literature review.

Do not use web search as decoration after the decision has already been made.
Use it to constrain the next HEAD hypothesis.

### Multi-Agent Sidecar Policy

Keep the narrative prefix-latent workflow as a centralized HEAD orchestrator.
Use sub-agents or independent sidecars only when the task is separable enough to
produce a bounded artifact that can be checked against local evidence. The
default sidecar roles are:

- **Literature Scout:** searches primary sources before adopting a new external
  method analogy or agent workflow.
- **Experiment Critic:** reviews a proposed hypothesis before adding a model
  family, loss, reranker, threshold, or other knob.
- **Artifact Verifier:** checks code, commands, metrics, artifacts, and
  research-log context before promotion.
- **Report Auditor:** checks paper/demo claims for stale text, missing figures,
  unsupported causal language, and mismatch with the current-truth index.
- **Implementation Worker:** edits only a disjoint write scope that the
  orchestrator can later integrate and verify.

Do not use multi-agent workflows for narrow sequential debugging, same-file
edits, metric truth by debate, broad hyperparameter sweeps, or replacing human
approval of product-contract changes. Use them at research decision gates, not
as a default way to make every iteration larger.

### Sweep And Hyperparameter Policy

Sweeping is not a principled default research action. It is allowed only as a
bounded diagnostic or calibration step.

Before a sweep, write:

- the single failure mechanism under test;
- the one hyperparameter axis being varied;
- the baseline and expected movement;
- the trade-off that would still be acceptable;
- the falsifier that stops the branch.

Default sweep budget is at most three values on one axis. Larger sweeps require
a preceding `research_ideation` decision and a saved analysis note explaining
why the mechanism cannot be tested more directly. A sweep cannot promote a new
default by itself; promotion still requires mechanism attribution, held-out
validation, and independent verification.

### Independent Verifier Checklist

Before promoting a bridge, support policy, demo path, production-readiness
claim, or paper-facing result, the verifier should check:

- the novelty claim is real and not just a renamed known method;
- the experiment actually ran and the artifacts match the claimed command;
- reported metrics are current, held out where needed, and not cherry-picked;
- code changes are substantive enough to support the claimed mechanism;
- citations are real, relevant, and current enough for the claim;
- generated reports contain no hallucinated numbers, duplicated figures,
  missing figures, placeholder text, or stale paths presented as evidence;
- the result still satisfies the fixed-start contract and does not reintroduce
  hidden model-chosen starts.
- the tracked current-truth index is updated, or the claim is explicitly marked
  non-promoted;
- promoted case specs/manifests are tracked, not only stored under ignored
  `autoresearch-session/`;
- null/repeat controls and per-start floors exist for fixed-start
  conditionality claims, or their absence is called out as a limitation.

If the verifier cannot answer these from local artifacts, the next step is
`post_experiment_analysis` or `research_ideation`, not promotion.

Verifier outputs must be durable artifacts. Save each promotion verifier report
under `docs/research_protocols/nl_prefix_latent_verifier_reports/` and cite it
from the research-log entry. A prose statement in `RESEARCH_LOG.md` is not
enough for promotion.

### Artifact Lifecycle Policy

Autoresearch creates many scripts, reports, and intermediate branches. Prevent
stale artifacts from becoming current truth:

- maintain one current default artifact path per production gate;
- maintain `docs/research_protocols/nl_prefix_latent_current_truth.md` as the
  tracked index of promoted claims, incumbent methods, and non-promoted results;
- mark superseded branches in local state when possible with `superseded_by`
  and `superseded_reason`;
- cite older reports only as baselines or historical context;
- if two reports conflict, run `post_experiment_analysis` before using either
  as evidence for a product or paper claim.
- copy promoted ignored manifests or case specs into
  `docs/research_protocols/nl_prefix_latent_promoted_specs/` so a clean checkout
  can identify the exact setup behind a claim.

### Commit and Artifact Policy

The workflow should commit verified tracked production changes unless the
current turn explicitly says not to. This applies to scripts, tests, tracked
protocol documentation, demo code, and research-log entries.

Do not commit ignored local control files or generated scratch artifacts by
default:

- `.agents/` local skill files;
- `autoresearch-session/` state and temporary body files;
- large arrays, checkpoints, caches, and one-off experiment outputs;
- private paper drafts or under-review paper material.

If the worktree is mixed, stage only files owned by the current iteration and
leave unrelated changes untouched. The local state should record either the
commit SHA or a concrete reason that no commit was made.

### OpenAI Call Policy

OpenAI calls are allowed when they advance a production-readiness gate. Use a
TestFlight first:

1. run 1-5 narratives/windows;
2. validate schema, grounding quality, embedding shape, and cache artifacts;
3. log model name, prompt version, timestamp, and approximate token/use count;
4. scale only if outputs are not garbage;
5. keep raw and normalized outputs in experiment artifact directories.

No private manuscript text, private paper PDFs, or under-review backup paper
content should be sent or uploaded.

The first iteration after this direction change should be a local
mixture-prior baseline, not a large OpenAI labeling run. It should use existing
casebook artifacts and prefix/latent outputs to test whether top-k mixture
support improves implication alignment before adding a learned residual.

Suggested first HEAD hypothesis:

> A soft mixture of narrative-relevant analogue prefixes/latents can improve
> story implication alignment versus top-1 analogue selection and simple
> rollout reranking, without asking text to generate the whole 30-day hidden
> prefix from scratch.

Suggested first execution:

1. build an offline top-k analogue-mixture prior evaluator using existing
   live/cached casebook artifacts and latent-bank outputs;
2. compare top-1, soft top-k, diverse top-k, and mixture-plus-rerank variants;
3. evaluate explicit implication mismatch, start support, memory compatibility,
   and scenario-level metrics where available;
4. only then train a residual bridge if the mixture prior provides a useful
   support manifold.

## Citation Notes

This plan follows a common pattern in generative modeling: compress complex
sequence objects into a learned latent space, condition that latent space on
language and context, and decode through a domain model rather than asking an
LLM to invent the final distribution.

- **Latent diffusion.** Rombach et al. show the value of doing diffusion in a
  compressed latent space instead of pixel space, which motivates operating on
  prefix latents rather than raw joint39 paths when possible:
  https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
- **VQ-VAE / discrete latent representations.** van den Oord et al. introduce
  discrete learned latent codes, relevant if continuous prefix latents collapse
  and we need tokenized regime codes:
  https://papers.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html
- **Text-to-motion latent tokens.** T2M-GPT maps text to discrete motion tokens,
  a useful analogue for text-to-prefix-latent generation without predicting raw
  hidden states directly:
  https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Generating_Human_Motion_From_Textual_Descriptions_With_Discrete_Representations_CVPR_2023_paper.html
- **Text-to-time-series generation.** T2S explicitly studies text-to-series
  diffusion and alignment of text and time-series latent spaces:
  https://www.ijcai.org/proceedings/2025/580
- **Text-controlled time-series generation.** BRIDGE studies bootstrapping
  text-time-series data and diffusion for text-controlled time-series
  generation:
  https://proceedings.mlr.press/v267/li25ah.html
- **Time-series forecasting with language models.** Time-LLM is useful as a
  reference for combining numerical time-series context with language-model
  representations rather than relying on language alone:
  https://openreview.net/forum?id=Unb5CVPtae
- **Contrastive language alignment.** CLIP is the standard reference for
  contrastive alignment between language and a non-language latent space:
  https://proceedings.mlr.press/v139/radford21a.html
- **Retrieval as support, not the generator.** RAG and retrieval-augmented
  time-series work motivate keeping provenance and analogue support visible
  while not treating retrieved neighbors as the whole generative mechanism:
  https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
  and https://proceedings.mlr.press/v267/han25d.html
- **Autonomous research loops.** AI Scientist, Agent Laboratory, Robin, and
  automated-discovery surveys motivate HEAD-style iteration, literature review,
  experiment execution, reporting, and review, but also require explicit
  guardrails for novelty, failed experiments, hallucinated evidence, and human
  intervention points:
  https://arxiv.org/abs/2408.06292
  https://arxiv.org/abs/2501.04227
  https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system
  https://link.springer.com/article/10.1007/s10994-025-06955-2
  https://arxiv.org/abs/2502.14297
