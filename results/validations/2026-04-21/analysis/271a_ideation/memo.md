# 271a Ideation Memo

## Context

`270a` through `270c` established a clean chain:

- `270a`: single latent code collapsed.
- `270b`: latent token sequence fixed the representation collapse.
- `270c`: changing the decoded object from next level to next change did not fix the family.

The common finding is now specific: the latent token state is usable, but the tiny
per-step token-to-observation decoder is too weak. Under teacher forcing, it still
misses the short-horizon conditional law. Under rollout, that mismatch amplifies into
support explosion, flat conditionality, and jump failure.

## Constraint

The next family must keep:

- autoregressive state feedback
- learned bottleneck
- first-principles simplicity

It must not add:

- hard low-rank heads
- bounded idio or EC side paths
- teacher / KL patching
- suite-specific losses
- multiple new branches at once

## Candidate Directions

### 1. Conditional observation decoder

Keep the `270` latent transition, but decode next change from both:

- the transitioned latent token state
- a compact current-history summary from the same encoder

This is the smallest plausible fix. The missing piece in `270` is that the decoder
only sees latent tokens, even though the next-step change law is state-dependent.

### 2. More expressive latent transition

Make the latent transition itself more expressive and keep the tiny decoder.

This is lower priority because teacher-forced probing already showed that the latent
state is not the main failure anymore.

### 3. Direct history-to-next-change bypass

Give the decoder a strong direct history bypass.

This is rejected for now. It would make the model less bottleneck-driven and risks
recreating the same bypass pathologies that collapsed earlier probabilistic lines.

## Decision

Choose `271a-v0`: autoregressive latent-sequence flow matching with a conditional
next-change decoder.

Minimal architectural change:

- reuse the `270c` history encoder and latent FM transition
- expose one compact history summary from the same encoder
- decode next normalized change from:
  - latent token sequence after transition
  - current history summary

This is still a single-core architecture:

- history encoder
- latent transition
- conditional decoder

No extra side branches.

## Why This Is Principled

- It follows the postmortem directly: the decoder needs current state context.
- It does not add finance-specific structure.
- It does not hard-code low rank, EC, or idio behavior.
- It preserves autoregressive feedback and lets the model learn the conditional law.

## Kill Criteria

`271a` is worth keeping only if it materially improves at least one of:

- conditionality
- cross-cell structure
- mean reversion

without making support and jump realism even worse than `270c`.

If teacher-forced probing still shows the decoder law is wrong after adding current
history summary, then this whole latent-token AR line should be closed.
