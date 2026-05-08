# Narrative Prefix-Latent Autoresearch Plan

## Objective

Build a risk-manager-facing narrative-conditioned scenario generator that stays
inside the learned latent geometry as much as possible while reusing the frozen
state-aware normalized-innovation (SNI) conditional scenario generator and its
native autoregressive rollout.

The long-run product contract has two modes:

1. **Model-chosen starting point.** The risk manager supplies only a narrative.
   The system chooses one or more plausible starting states, then generates
   30-day scenario distributions conditioned on the narrative.
2. **User-specified starting point.** The risk manager supplies a narrative plus
   the current or hypothetical starting level for the joint scenario factors.
   The system generates 30-day scenario distributions from that explicit state.

Historical analogues are allowed as audit evidence, support diagnostics, or a
starting-point proposal mechanism. They should not be the hidden engine that
defines the 30-day prefix dynamics.

## Production-Readiness Workflow

This protocol now treats research progress and production readiness as separate
but connected obligations. A new model result is not enough. Each iteration
should move at least one of these product gates:

1. **Grounded narrative input.** The risk-manager story is converted into
   explicit market implications, unsupported-claim warnings, model metadata,
   and cached text conditions.
2. **Text-memory plus start contract.** The model accepts either a proposed
   historical start or an explicit user-specified joint39 start, then decodes a
   recent-prefix object that ends exactly at that start.
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

The next production milestone is a **live prefix-latent story smoke path**. It
should use cached text-memory examples first, then add carefully bounded OpenAI
calls only after the cached path works end to end.

## Current Problem

The current narrative bridge maps a text embedding to the frozen generator's
last 128-dimensional prefix memory. That is useful for analogue retrieval and
diagnostics, but it is not a native prompt-conditioned rollout contract. The
SNI generator recomputes memory at each autoregressive step from the evolving
prefix. Reusing one fixed memory vector turns the model into repeated
one-step sampling under a stale condition, while retrieving a historical prefix
makes the product look like K-nearest-neighbor generation.

The direct-memory and residual-memory ablations confirm the mechanics:

- fixed text-predicted memory can drive the decoder mechanically, but it is not
  equivalent to native autoregressive generation;
- residual prompt conditioning is safer because it keeps the evolving-prefix
  encoder alive, but the retrieved prefix still carries too much of the
  scenario identity;
- the next clean formulation is to infer a **latent representation of the
  recent prefix** from text plus starting state, then let the frozen SNI
  generator roll forward normally.

## Proposed Model Family

The central object is a **prefix latent**:

```text
narrative + starting state -> prefix latent z -> synthetic recent-prefix state
synthetic recent-prefix state -> frozen SNI encoder/rollout -> future scenarios
```

The prefix latent should represent the 30-day recent regime in the generator's
native coordinate system without requiring the final product to copy a historical
window. The decoded prefix must provide the fields needed by the frozen SNI
encoder, including level history, normalized innovation or flow-coordinate
history, and the history-derived center/scale/drift conditioning features.

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

### Stage B: Prefix Autoencoder

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

### Stage C: Text-and-Start to Prefix Latent

Train a supervised bridge:

```text
text embedding + starting state -> predicted prefix latent z_hat
```

Positive pairs are multiple narratives for the same historical window. Hard
negatives include opposite-direction narratives, nearby but directionally
different windows, and windows with similar starting levels but different recent
regime paths.

The loss should begin simple:

- latent regression or cosine loss to the prefix autoencoder latent;
- CLIP-style / InfoNCE contrastive alignment between text-start embeddings and
  prefix latents;
- optional hard-negative margin only if the simple bridge collapses.

The bridge output is decoded to a synthetic prefix and passed through the
unchanged SNI autoregressive sampler.

### Stage D: Product Modes

For model-chosen starts, use retrieval only at the start-state layer:

- retrieve or sample plausible `s0` states consistent with the narrative;
- generate several start candidates and display them to the risk manager;
- run the prefix-latent generator from each start.

For user-specified starts:

- accept the provided joint39 starting state;
- predict prefix latent conditional on the same narrative and that explicit
  start;
- decode a compatible recent-prefix state;
- run the frozen generator normally.

This makes the analogue optional. The latent bridge is the scenario condition;
the analogue is provenance and plausibility support.

## Evaluation Plan

Use the existing representative manifest and scenario-level evaluation harness,
then add prefix-latent baselines in this order:

1. **Oracle decoded prefix.** Encode true historical prefix to `z`, decode it,
   and run the frozen generator. This tests whether the prefix autoencoder
   preserves generator-relevant information.
2. **Text-start predicted prefix.** Predict `z_hat` from narrative plus true
   held-out starting state, decode, and roll out.
3. **Text-only start-selected prefix.** Let the model choose plausible starts,
   then predict prefix latents and roll out.
4. **Ablations.** Compare against historical replay, persistence, current
   analogue-top-k generation, direct memory, residual memory, raw text embedding
   retrieval, no-contrastive bridge, and contrastive bridge.

Primary metrics:

- energy score and ensemble CRPS versus persistence;
- 80% coverage and interval score;
- mean-path MAE as a secondary diagnostic, not the publication target;
- memory cosine between true, decoded, and predicted prefixes;
- hard-case narrative validation and directional contrastive rank;
- sensitivity to changing the user-specified starting state under the same
  narrative.

Promotion criterion:

The prefix-latent system must beat or match the analogue-conditioned narrative
generator on distributional scenario metrics while reducing dependence on
retrieved historical prefixes. Exact historical-window retrieval is not the
target.

## HEAD Loop Setup

Use the repo's HEAD discipline:

- **Hypothesis:** state one falsifiable claim about prefix-latent conditioning.
- **Execute:** run one focused implementation, analysis, or experiment.
- **Analyze:** compare against saved artifacts and baselines.
- **Decide:** update persistent state, append the research log, and recommend
  the next iteration.

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

The first iteration after approval should be a `research_ideation` or
`post_experiment_analysis` cycle, not a large OpenAI labeling run. The cleanest
first execution step is a shape-and-contract audit plus an oracle prefix
autoencoder scaffold that uses already available windows and labels.

Suggested first HEAD hypothesis:

> A low-dimensional prefix latent can reconstruct enough of the 30-day joint39
> recent-prefix object that the frozen SNI encoder produces nearly the same
> final memory and the frozen generator produces similar scenario distributions.

Suggested first execution:

1. build the prefix-latent dataset contract from existing joint39 arrays;
2. train a small local prefix autoencoder without OpenAI calls;
3. run oracle decoded-prefix rollout on the existing representative held-out
   split;
4. compare decoded-prefix generator results against true-prefix oracle and
   analogue-top-k narrative generator.

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
