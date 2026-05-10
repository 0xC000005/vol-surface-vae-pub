# World Model HEAD100: Sample-Scale Caveat

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Evidence-boundary audit for `masked_multiview_invariance`.

## Hypothesis

The HEAD070 package should make clear that the reference checkpoint is a
smoke-scale reference candidate, not a full-data convergence result.

## Falsifier

The audit fails if package text allows future work to imply HEAD070 was trained
to full-data convergence.

## Evidence

`results/world/masked_multiview_barlow_head070.json` records:

- epochs: `8`;
- train windows: `384`;
- validation windows: `128`;
- token dimension: `58`;
- latent dimension: `64`;
- device: `cpu`.

This is enough for a reference-candidate smoke package and evidence-chain
guardrails. It is not enough to claim that full-data masked-multiview training
has converged.

## Package Update

- Added `sample_scale_caveat` to
  `experiments/world/part1_jepa_latent/reference_manifest.json`.
- Added the same caveat to
  `experiments/world/part1_jepa_latent/package_summary.md`.

## Decision

HEAD070 remains the current reference candidate, but its claims are explicitly
limited to smoke-scale evidence. Full-data scaling remains future work and
should not be silently assumed.

## Verification

- Parsed `results/world/masked_multiview_barlow_head070.json`.
- Confirmed the existing package already listed `384`/`128` windows but did not
  label this as a caveat.
