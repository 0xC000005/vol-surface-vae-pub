# World Model HEAD086: Downstream Probe Interpretation

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe` interpretation for frozen Part 1 representations.

## Hypothesis

HEAD085 is useful if it clarifies what HEAD070 can support as a Part 1
reference candidate without turning downstream forecasting into pretraining.

## Falsifier

If the probe evidence requires claiming that HEAD070 is a general forecasting
representation, or if it shows no clear domain where the representation beats
simple raw-surface baselines, the Part 1 reference claim should be weakened.

## Evidence Read

- HEAD082: HEAD070 has strong same-state masked-view retrieval
  (top1/top5/top10 `0.321354/0.662500/0.841927`) with healthier rank than
  HEAD068 (effective rank about `14.5` vs `4.5`) and lower redundancy than
  HEAD068 (health offdiag about `0.224` vs `0.47`).
- HEAD083: mask-family leakage is below majority baselines, so the embedding
  does not look like it is primarily encoding the synthetic mask artifact.
- HEAD084: no mask-family stratum collapses; every stratum has top10 retrieval
  at least `0.826`.
- HEAD085: downstream probes are mixed. `barlow_clean_last` improves over
  `raw_surface_last` on range, max-absolute-step, and drawdown MSE, but
  `raw_surface_last` remains stronger on future mean delta, terminal delta, and
  regime-label accuracy.

## Interpretation

HEAD070 is a plausible Part 1 reference candidate for the objective actually
being trained: direct masked-multiview market-state invariance with redundancy
control. The strongest evidence is same-window retrieval, healthy rank,
nontrivial variance, limited mask-artifact leakage, and no large stratified
mask-family failure.

The downstream probe evidence should be framed as partial utility, not as a
forecasting result. The representation is comparatively useful on path-width
and risk-shape probes:

| target | HEAD070 Barlow MSE | raw last-surface MSE | interpretation |
| --- | ---: | ---: | --- |
| future_range | 0.047185 | 0.054625 | Barlow better |
| future_max_abs_step | 0.039526 | 0.041928 | Barlow better |
| future_drawdown | 0.042269 | 0.050058 | Barlow better |

It is not currently better for directional/terminal targets:

| target | HEAD070 Barlow R2 | raw last-surface R2 | interpretation |
| --- | ---: | ---: | --- |
| future_mean_delta | 0.162420 | 0.533258 | raw stronger |
| future_terminal_delta | 0.020872 | 0.409125 | raw stronger |

Regime classification is not yet reliable as an acceptance criterion. Accuracy
is below the majority baseline for all tested feature sets, and HEAD070
accuracy is especially weak (`0.109375` vs majority `0.597656`). Macro recall
alone should not be used to promote the representation.

## Acceptance Boundary

HEAD070 can be treated as the current Part 1 reference candidate only for:

- masked-multiview invariance under structured missingness;
- representation health and non-collapse;
- risk-width/path-shape downstream probe utility.

HEAD070 should not be claimed as:

- an ImageNet-level JEPA analogue;
- a general future predictor;
- a regime classifier;
- proof that Barlow features are universally complementary to raw features.

## Decision

Keep HEAD070 as the reference candidate with explicit caveats. The next
autoresearch step should consolidate a Part 1 readiness checklist and artifact
manifest, then decide whether another diagnostic is necessary before any Part 2
decoder work. Do not add a model knob from HEAD085 alone.

## Verification

- Read HEAD082, HEAD084, and HEAD085 reports.
- Parsed `results/world/masked_multiview_downstream_probe_head085.json` with
  `python -m json.tool`.
