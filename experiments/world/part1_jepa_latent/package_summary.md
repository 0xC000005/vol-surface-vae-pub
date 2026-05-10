# World Model Part 1 Package Summary

Date: 2026-05-10

## Status

The current Part 1 world-model reference candidate is the HEAD070 direct
masked-multiview Barlow representation. It replaces the earlier fixed
delta-PCA predictor package as the active objective after the masked-multiview
objective correction.

This package is not a decoder and not a future-prediction pretraining model. It
is a representation-learning package for same-market-state masked views.

## Reference

- Objective family: `masked_multiview_invariance`.
- Literature status:
  `supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview`.
- Checkpoint:
  `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt`.
- Training result:
  `results/world/masked_multiview_barlow_head070.json`.
- Manifest:
  `experiments/world/part1_jepa_latent/reference_manifest.json`.
- Artifact digests:
  `experiments/world/part1_jepa_latent/reference_artifact_digests.json`.
- Package checker:
  `experiments/world/part1_jepa_latent/reference_package_check.py`, covering
  source reports, guardrail-doc caveat terms, and ignored artifact identities.
- Restart checklist:
  `experiments/world/part1_jepa_latent/restart_checklist.md`.
- Part 1 quality gate:
  `experiments/world/part1_jepa_latent/part1_quality_gate.md`.

## Fixed Contract

- Data: `data/vol_surface_with_ret.npz`.
- Windowing: history `30`, future `30`.
- Token dimension: `58`.
- Latent dimension: `64`.
- Reference training windows: `384`.
- Reference validation windows: `128`.
- Positive pair: same market window and same relative index under two
  structured synthetic masks.
- Loss: direct encoder-output Barlow alignment with canonical mean-scaled
  off-diagonal term.

## Evidence

- Same-state retrieval: top1/top5/top10
  `0.321354/0.662500/0.841927`.
- Rank health: effective rank about `14.5` for both views.
- Redundancy: health offdiag about `0.224`, substantially lower than the
  high-retrieval but low-rank HEAD068 branch.
- Mask-artifact audit: mask-family prediction is below majority baselines.
- Stratified audit: no mask family has top10 below `0.826`.
- Mask-policy coverage: HEAD070 trained and was audited on
  `surface_maturity`, `surface_moneyness`, `surface_rectangle`,
  `vol_side_channel`, `factor_family`, and `time_block`; richer protocol ideas
  such as wing, ATM-strip, whole-surface day dropout, and cross-family stress
  masks remain coverage caveats, not validated HEAD070 claims.
- Downstream probe caveat: HEAD070 helps some risk-width/path-shape probes
  relative to raw last-surface features, but raw last-surface features remain
  stronger for mean/terminal deltas and regime-label accuracy. The full-history
  raw surface baseline also beats Barlow on max-absolute-step MSE, so the
  downstream utility claim is not "Barlow beats all raw baselines." These
  downstream probes target IV-surface futures only; factor-panel future targets
  have not been evaluated.

## Caveats

- This is not canonical ImageNet I-JEPA.
- The HEAD070 checkpoint is a smoke-scale reference candidate trained on `384`
  windows and validated on `128`; do not claim full-data convergence from it.
- Future prediction, range estimation, regime labels, and scenario generation
  remain downstream probes or consumers, not pretraining objectives.
- The representation should not be described as a general predictor.
- Regime classification is not ready as an acceptance criterion.
- Part 1 success does not prove Part 2 scenario-generation quality.
- Part B remains gated until the literature-aligned Part 1 quality gate passes:
  representation health, corruption robustness, baseline superiority,
  market-state frozen probes, temporal utility probes, and scale/stability.

## Next Work Requires Direction

Future work should be one of:

- a documented Part 1 quality-gate diagnostic that does not add model knobs by
  default;
- downstream benchmark/probe work that consumes the frozen representation
  without mutating the pretraining objective.
