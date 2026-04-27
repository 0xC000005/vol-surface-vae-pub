# 582a Conditional Empirical Source Unified Flow

## Context

581a showed that full-path empirical source sampling is much better than Gaussian
source sampling, but it can still attach a historically extreme increment path to
an unrelated current history. 582a tests source conditioning using a generic
history-state nearest-neighbor rule over the unified 38-variable panel.

## Implementation

Extended `train_577a_unified_increment_flow.py` with:

- `--source_mode empirical_conditional`;
- `--conditional_source_topk`;
- source-bank keys equal to the standardized last history state over all 38
  variables;
- source paths sampled uniformly among the top-k nearest training history states.

This is still one shared source law over the unified panel. It does not add
separate IV or factor heads.

## Runs

Base unconditional empirical source:

- `models/backfill/581b_clean_empirical_source_unified_flow_s583`;
- best val loss: `2.5859909909`;
- IV max: `44.67`;
- factor max: `20,772.86`;
- increment std ratio: `1.0536`.

Conditional top-k 64:

- `models/backfill/582a_clean_conditional_empirical_source_flow_s584`;
- best val loss: `2.5263086557`;
- IV max: `28.98`;
- factor max: `20,707.53`;
- increment std ratio: `1.0186`.

Conditional top-k 16:

- `models/backfill/582b_clean_conditional_empirical_source_top16_flow_s585`;
- best val loss: `2.5485931805`;
- IV max: `21.09`;
- factor max: `19,856.02`;
- increment std ratio: `0.9832`.

Conditional top-k 4:

- `models/backfill/582c_clean_conditional_empirical_source_top4_flow_s586`;
- best val loss: `2.5458627599`;
- IV max: `10.43`;
- factor max: `19,861.31`;
- increment std ratio: `0.9843`.

## Mechanism Read

The monotonic IV-tail improvement is strong evidence that source mismatch was the
dominant remaining problem:

- unconditional empirical source IV max: `44.67`;
- conditional top-k 64: `28.98`;
- conditional top-k 16: `21.09`;
- conditional top-k 4: `10.43`.

Factor realism stays stable while IV tails improve. This is the best unified
joint-law prototype so far.

The tradeoff is methodological:

- top-k conditioning is generic and unified;
- but very small k starts to look like conditional source retrieval;
- top-k 4 is therefore a useful frontier candidate, not yet a final publishable
  probability-law claim.

## Decision

582c is the current best unified joint-law prototype, but still not deployable:

- IV max around `10.43` remains far above historical IV scale;
- the flow is still weak relative to source choice;
- top-k source conditioning must be evaluated for diversity and conditionality
  before being accepted.

Next clean step:

- build a richer sample audit for 582c over more validation windows/samples;
- measure IV/factor ranges, coverage, conditionality proxies, and diversity;
- only then decide whether to train longer, adjust top-k, or add a generic
  reconstructed-level realism objective.
