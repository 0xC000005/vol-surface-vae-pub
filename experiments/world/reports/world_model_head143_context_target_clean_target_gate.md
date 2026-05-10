# World Model HEAD143: Context-Target Clean-Target Gate

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

`context_to_target_jepa` design correction.

## Literature Status

`canonical_jepa`.

I-JEPA predicts target-block representations in the same image and stresses
that target blocks are sampled from the target-encoder output, not created by
masking the target input. In the paper's terms, the full input is fed through
the target encoder and target blocks are then selected from the resulting
representations. Source: <https://arxiv.org/pdf/2301.08243>, especially the
method discussion around target representations and masking. V-JEPA keeps the
same broad feature-prediction framing for video and evaluates frozen features
on downstream tasks. Source: <https://arxiv.org/abs/2404.08471>.

## Local Evidence

HEAD142 shows the current HEAD140 implementation does not follow that target
construction cleanly:

- the target encoder receives target-only sparse values plus a target mask
  channel;
- selected target latents are strongly target-family identifiable, with
  accuracy lift `0.467955`;
- predicted target latents are low-rank, with effective rank `5.908201`;
- predicted-to-target cosine is high (`0.979068`), but row retrieval is poor
  (`top10=0.042969`);
- the clean context embedding remains weak for state probes.

This means the low loss is not a valid success signal. The target surface is
partly a mask-family artifact, and the predictor can align to a low-rank
surrogate without learning a better market-state representation.

## Proposed Single Correction

Run exactly one named diagnostic branch:

```text
clean market window
-> sample typed target blocks
-> context encoder receives context-masked values and observed/synthetic masks
-> target encoder receives clean full-window values and observed masks only
-> select the target time rows from the target-encoder output
-> predictor aligns context rows to clean target rows
-> score clean context embeddings on rank, retrieval, mask-family leakage, and
   exact-state probes
```

This is not a new hyperparameter sweep. It is a correction to make the local
context-to-target diagnostic match the canonical JEPA target rule more closely.
It still uses no future targets, no value reconstruction, and no decoder loss.

## Falsifier

Reject this correction as insufficient if the clean-target smoke still shows:

- clean context effective rank below the scaled Barlow candidate;
- high cosine with poor row retrieval;
- exact-state probes worse than scaled Barlow;
- target-family leakage from the target or predicted latent surfaces;
- no improvement in current-IV retention.

## Decision

Proceed to one minimal clean-target context-to-target smoke. Do not add a knob
sweep, do not add value reconstruction, and do not start Part 2. The next report
must compare the corrected branch against HEAD127 scaled Barlow and HEAD140
target-only context-to-target evidence.
