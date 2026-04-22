# 270b Ideation Memo

## Context

`270a-v0` showed that a generic bottleneck is likely necessary, but a **single-vector** bottleneck is too compressive.
Teacher-forced probing showed the bottleneck itself collapsed before the stochastic transition mattered.

## Constraint Set

Still keep the reset bans:
- no hard low-rank decoder
- no bounded idio / EC side paths
- no KL or teacher-engineering hacks
- no suite-specific loss stack

## Recommended Next Family

**270b-v0: autoregressive latent-sequence bottleneck next-change flow matching**

Minimal change from `270a`:
- replace the single latent vector with a short sequence of learned latent tokens
- keep autoregressive next-change FM in latent space
- decode next level from the token sequence
- keep teacher-forced one-step training and recursive rollout

## Why This Is Principled

- It changes exactly one thing: the bottleneck shape.
- It preserves the core story that the model should learn its own shared representation.
- A short latent sequence is still generic and data-driven, but it gives the bottleneck enough capacity to avoid collapsing into a single common mode.

## Pre-Registered Question

Can a short latent token state:
- avoid the rank-1 collapse seen in `270a`
- preserve autoregressive state feedback
- recover useful structure without reverting to hand-designed factors?

## Decision

Proceed to `270b-v0` when execution resumes.
