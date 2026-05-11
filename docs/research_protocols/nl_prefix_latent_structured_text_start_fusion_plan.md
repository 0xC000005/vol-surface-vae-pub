# Structured Text/Start Fusion Plan

Last updated: 2026-05-11

This note records the next design direction after direct text-plus-start
concatenation failed the generator-memory diagnostic.

## Negative Local Result

The cached 182-window representative diagnostics show that appending the
standardized joint39 starting level to OpenAI text embeddings is not sufficient:

- text-only target cosine: `0.8582`;
- naive text-plus-start target cosine: `0.7865`;
- text-only hard-negative gap: `0.8950`;
- naive text-plus-start hard-negative gap: `0.1872`;
- small start weights `0.05`, `0.10`, and `0.25` still materially degrade
  target cosine and hard-negative separation.

The start level matters to the product contract, but direct concatenation
distorts the text geometry that currently carries narrative directionality.

## Related-Work Constraints

- FiLM conditions a network by producing feature-wise affine modulation rather
  than concatenating the condition into the same input vector. This suggests a
  start-conditioned modulation layer can influence the memory bridge while
  preserving the text tower. Source:
  https://ojs.aaai.org/index.php/AAAI/article/view/11671.
- Gated Multimodal Units learn multiplicative gates over separate modality
  representations. This suggests start information should enter through a
  learned gate or residual, not by overwhelming text embeddings. Source:
  https://huggingface.co/papers/1702.01992.
- BLIP-2 bridges frozen unimodal systems using a lightweight Querying
  Transformer rather than full end-to-end retraining. This supports a small
  bridge module between frozen text embeddings and frozen SNI memory targets.
  Source: https://proceedings.mlr.press/v202/li23q.html.
- CLIP/DALL-E/Sora-style analogies remain useful only at the level of paired
  alignment, latent priors, and rich captions. They do not remove the frozen
  SNI autoregressive-prefix constraint.

## Candidate Designs

### Option A: Late-Fusion Residual Adapter

```text
text embedding -> text tower -> base memory
start state -> start tower -> residual memory
output = base memory + bounded residual
```

Pros:

- simplest test after the failed concatenation result;
- preserves the incumbent text-only path;
- residual size can be bounded so start cannot erase hard-negative geometry.

Cons:

- may still learn shortcuts from level state to memory target;
- does not explicitly model interactions between story and start.

### Option B: Start-Gated Text Adapter

```text
text embedding -> text tower -> text features
start state -> gate tower -> sigmoid gate + FiLM scale/shift
modulated text features -> memory head
```

Pros:

- follows FiLM/GMU-style separation;
- lets start modulate text features without replacing them;
- gate statistics are auditable.

Cons:

- introduces one architectural choice and one regularization choice;
- needs diagnostics to ensure gates do not collapse to all-start or no-start.

### Option C: Query-Former-Style Lightweight Cross-Attention

```text
learned query tokens attend to text tokens and start/factor tokens
query outputs -> memory prior
```

Pros:

- closest to BLIP-2-style bridge logic;
- can preserve token-level narrative detail instead of one sentence vector.

Cons:

- requires token embeddings or a richer text encoder artifact than the current
  saved OpenAI sentence embeddings;
- too large for the next TestFlight unless the smaller gates fail.

## Recommended Next Experiment

Implement Option B as a bounded TestFlight:

```text
text tower + start gate/FiLM modulation -> generator-memory target
```

Training target:

- same cached representative OpenAI embeddings;
- same 182 windows and same train/test split;
- same memory target and hard-negative caption groups;
- no new OpenAI calls;
- no broad hyperparameter sweep.

Controls:

- incumbent text-only MLP;
- failed concatenation baseline;
- start-gated adapter with a bounded residual/gate penalty.

Falsifiers:

- held-out target cosine drops by more than `0.01` versus text-only;
- hard-negative gap or margin drops by more than `0.05` versus text-only;
- average gate saturates near `0` or `1`, showing the model either ignored
  start or let start dominate;
- same-start hard negatives no longer separate directionally.

Expected useful outcome:

- equal or better target cosine versus text-only;
- no material hard-negative degradation;
- gate statistics show partial start use rather than collapse;
- if recall improves but memory/direction drops, reject it as retrieval-only.

## Decision

The next executable HEAD iteration should implement a small start-gated
text-memory adapter and run it as a cached TestFlight. Direct concatenation is
closed as a non-promoted diagnostic branch.
