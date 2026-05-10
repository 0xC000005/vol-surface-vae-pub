# World Model HEAD071: Canonical Barlow Reference Decision

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

Is HEAD070's residual redundancy a blocker, or is the canonical direct Barlow
model good enough to become the current Part 1 reference candidate for
downstream probes?

## Evidence

HEAD070 validation embeddings are not merely high-retrieval. Their spectrum is
healthier than the raw masked-view baseline:

| representation | top1 share | top3 share | top5 share | top10 share | effective rank | participation ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD070 view A | 0.090855 | 0.242807 | 0.353656 | 0.566354 | 14.501471 | 9.676580 |
| HEAD070 view B | 0.090381 | 0.241693 | 0.353157 | 0.565351 | 14.593816 | 9.777783 |
| raw view A | 0.113880 | 0.265981 | 0.372180 | 0.569724 | 12.820380 | 7.462804 |
| raw view B | 0.113912 | 0.265746 | 0.371881 | 0.571119 | 12.797703 | 7.493438 |

The model still has more same-view correlation than ideal:

| representation | offdiag abs mean | offdiag abs max |
| --- | ---: | ---: |
| HEAD070 view A | 0.224284 | 0.809242 |
| HEAD070 view B | 0.223059 | 0.801243 |
| raw view A | 0.163462 | 0.949041 |
| raw view B | 0.190238 | 0.965464 |

But this is no longer a collapse signature:

- effective rank is above raw;
- top singular concentration is lower than raw;
- same-state retrieval is far above raw and HEAD066;
- the remaining redundancy is moderate and measurable, not catastrophic.

## Decision

Do not add a projection head or another pretraining tweak yet. That would be
premature knob work.

Treat HEAD070 as the current Part 1 reference candidate and run a frozen
downstream probe next. The probe should answer whether the learned
same-state embeddings carry useful market information beyond raw masked-view
features.

## Next Experiment

Build a frozen probe audit for HEAD070 embeddings:

- load `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt`;
- encode train/validation windows with the frozen shared encoder;
- compare ridge probes from HEAD070 embeddings against raw last-day or raw
  flattened-window baselines;
- target simple future summaries only as downstream probes, not pretraining:
  future mean delta and/or range over IV-surface tokens;
- report probe MSE/R2 plus representation health and same-state retrieval.

Falsifier: if HEAD070 embeddings beat raw masked-view retrieval but fail to
support even simple frozen downstream probes relative to raw baselines, then
the representation may be learning mask-invariance without useful market
state content.

## Artifacts

- `results/world/masked_multiview_barlow_head070.json`
- `models/world/checkpoints/part1_jepa_latent/masked_multiview_barlow_head070.pt`
- `experiments/world/reports/world_model_head070_canonical_barlow_scaling.md`
- `experiments/world/reports/world_model_head071_canonical_barlow_reference_decision.md`
