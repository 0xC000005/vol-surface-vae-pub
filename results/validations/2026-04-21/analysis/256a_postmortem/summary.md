# 256a-v0 Postmortem

## Result

- suite score: `2/11`
- passes:
  - `surface`
  - `block_ar`
- still below the deterministic `4/11` frontier

Artifacts:

- eval: `results/block_ar/256a_v0_L8_s42/full11.json`
- spec: `results/validations/2026-04-21/analysis/256a_design/spec.md`
- checkpoint: `models/backfill/256a_v0_L8_s42/best_model.pt`

## High-Signal Metrics

- `change KS`: `0/25`
- `level KS`: `4/25`
- `corr_ratio`: `0.475`
- `rank_ratio`: `0.330`
- `mr_gt_ratio`: `1.235`
- `mr_h30`: `1.080`
- `acf_corr`: `0.860`
- `kurtosis_ratio`: `0.387`
- `cointegration_ratio`: `0.194`
- `max-jump KS`: `0.677`

## Mechanism Read

The hierarchical token backbone did **not** collapse to one token. It failed more
fundamentally: the token-attention mechanism stayed exactly **uniform** for the entire
run, so the model learned around the tokenization rather than through it.

### Token mechanism was dead

- attention entropy mean: `1.6094`
- attention entropy min: `1.6094`
- mean top-1 token weight: `0.2000`
- token utilization:
  - `[0.2000, 0.2000, 0.2000, 0.2000, 0.2000]`
- utilization entropy: `1.6094`

For 5 tokens, `log(5) = 1.6094`. So the attention distribution stayed exactly
uniform across all horizons and all validation windows.

### What actually carried the model

- factor share RMS: `0.978`
- residual share RMS: `0.067`

So the model solved the task almost entirely through the low-rank factor path while
the token mechanism contributed no selective structure.

### Output structure still degraded

- output effective rank: `2.67`
- output PC1 share: `0.573`

This is not the same failure as `254`, but it still drifts toward an over-shared,
under-flexible deterministic path and fails the key temporal-law suites.

## Conclusion

`256a-v0` cleanly failed as a decisive prototype:

- below the deterministic frontier (`2/11`)
- token mechanism inactive
- no basis to continue deterministic token families in this form

The next principled move is another **paradigm shift**:

- leave deterministic tokenized center-path families
- move to a richer latent generative family that can represent conditional future
  structure without relying on dead uniform token attention
