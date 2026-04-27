# 583a 582c Unified Flow Sample-Quality Audit

## Context

582c is the current best unified joint-law prototype. It uses one shared flow over
the cleaned 38-variable increment tensor and a top-k conditional empirical source.
Before changing the model again, the key question was whether 582c is actually a
learned conditional law or mostly a conditional source-replay mechanism.

## Implementation

Added:

- `experiments/backfill/block_ar/audit_583a_unified_flow_sample_quality.py`;
- `test_code/test_583a_unified_flow_sample_quality.py`.

The audit compares:

- source-only samples from the checkpoint source distribution;
- post-flow samples after the learned conditional transport;
- IV/factor ranges;
- train-range exceedance;
- 90% interval coverage;
- sample diversity;
- endpoint-move conditionality proxies;
- effective rank of transformed increment paths.

## Artifact

Command:

```bash
python experiments/backfill/block_ar/audit_583a_unified_flow_sample_quality.py \
  --checkpoint models/backfill/582c_clean_conditional_empirical_source_top4_flow_s586/best_model.pt \
  --sample_windows 441 \
  --n_samples 32 \
  --sample_steps 16 \
  --seed 583 \
  --device cuda \
  --output results/autoresearch/583a_582c_unified_flow_sample_quality/audit.json
```

Output:

- `results/autoresearch/583a_582c_unified_flow_sample_quality/audit.json`.

## Result

Post-flow 582c on full validation, 32 samples per window:

- finite state rate: `1.0`;
- finite increment rate: `1.0`;
- IV min / max: `0.00072` / `10.43`;
- IV q0.1% / q99.9%: `0.00905` / `2.52`;
- factor min / max: `0.249` / `22,943.97`;
- IV above train max rate: `0.00349`;
- IV below train min rate: `0.00144`;
- factor above train max rate: `0.00204`;
- IV 90% interval coverage: `0.521`;
- factor 90% interval coverage: `0.500`;
- full state 90% interval coverage: `0.514`;
- IV sample std / GT std: `2.181`;
- factor sample std / GT std: `0.991`;
- transformed increment effective rank: `69.41`;
- IV endpoint-move correlation: `0.593`;
- factor endpoint-move correlation: `0.057`.

Source-only versus post-flow deltas:

- IV max delta: `+0.123`;
- factor max delta: `-32.84`;
- IV coverage delta: `+0.0045`;
- factor coverage delta: `-0.0076`;
- increment effective-rank delta: `-0.073`.

## Mechanism Read

The audit clarifies the pathology:

- 582c is mostly conditional source selection, not strong learned transport;
- post-flow samples are almost identical to source-only samples;
- IV endpoint conditionality is moderately positive, but factor endpoint
  conditionality is near zero;
- interval coverage is far below the nominal 90% level despite global IV sample
  std being too high, meaning the distribution is misallocated rather than simply
  too narrow;
- the rare IV extremes are source-driven and are not corrected by the flow.

This is not yet a publishable conditional probability law.

## Decision

Do not keep tightening top-k. That would become a retrieval stress deck rather
than a learned generative law.

The next clean move should strengthen the learned transport with a generic
reconstructed-level/path realism objective, while keeping:

- one shared model;
- one unified 38-variable panel;
- no separate IV/factor heads;
- no evaluator-specific metric hacks.

The most direct next experiment is a short fine-tune of 582c with a generic
sampled path-realism loss: penalize train-range exceedance and improve interval
coverage in reconstructed state space across all variables.
