# 270c Ideation Memo

## Context

`270b-v0` fixed the single-vector collapse from `270a`, but teacher-forced probing shows the decoded state law is still wrong even before stochastic rollout.

## Clean Lesson

The bottleneck shape is no longer the dominant problem.
The next decoded object is.

## Recommended Next Family

**270c-v0: autoregressive latent-sequence bottleneck with next-change decoding**

Minimal change from `270b`:
- keep the latent token bottleneck
- keep the autoregressive latent FM transition
- change only the decoder target from next level to next normalized change

## Why This Is Principled

- It changes exactly one causal lever.
- The postmortem says the level decoder is the part distorting MR and rank structure.
- Decoding changes is also more aligned with the strong results from direct change-space experiments, but now with autoregressive state feedback preserved.

## Decision

Proceed to `270c-v0` when execution resumes.
