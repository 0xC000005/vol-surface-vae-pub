# World Model HEAD081: Part 1 Metric-Gap Audit

## Objective Family

`post_experiment_analysis`.

## Hypothesis

After HEAD080, the core Part 1 leaderboard should be reproducible, but the full
Part 1 acceptance contract is likely not yet fully covered by saved artifacts.

## Evidence Read

- `docs/research_protocols/world_model_autoresearch_plan.md`
- `experiments/world/reports/world_model_head080_part1_scorecard.md`
- `results/world/masked_multiview_part1_scorecard_head080.json` (ignored local
  scorecard output)
- HEAD063/064/065 protocol and diagnostic reports
- HEAD070/072/074/077 reference and probe reports

## Gate Coverage

| Part 1 gate | current status | evidence | gap |
| --- | --- | --- | --- |
| Same-state alignment MSE/cosine/cross-correlation | covered | HEAD080 scorecard extracts `mse`, `cosine_mean`, Barlow diag/offdiag | None for aggregate validation. |
| Same-state retrieval top-k/MRR | covered | HEAD080 extracts top1/top5/top10/MRR and median rank | None for aggregate validation. |
| Embedding variance | partial | HEAD080 extracts variance mean per view; source artifacts include richer health blocks | Scorecard does not expose variance min/max or per-dimension worst cases. |
| Effective rank / participation ratio | covered | HEAD080 extracts both views | None for aggregate validation. |
| Singular-value spectrum | partial | source artifacts store `singular_values` in health blocks | Scorecard/report does not summarize spectrum shape or rank concentration. |
| Off-diagonal covariance/correlation norm | covered | HEAD080 extracts Barlow offdiag and health offdiag is available in artifacts | Report only highlights Barlow offdiag; health offdiag can be surfaced. |
| Mask-artifact diagnostics | partial | visibility rates and mask families are saved | No explicit classifier/shortcut test that embeddings fail to encode mask pattern. |
| Geometry-stratified diagnostics | partial | visibility by geometry/family is saved | No geometry-stratified retrieval/alignment failure table by surface/factor/time mask. |
| Frozen downstream probes | partial | HEAD074 covers future mean-delta and future range against raw baselines | Missing regime/state labels, jump/tail, drawdown, correlation/dependence probes. |

## Interpretation

HEAD070 is a defensible **reference candidate**, not a finished Part 1 model.
The scorecard now supports the main reference decision:

- HEAD066 is rejected because predictor-target agreement does not preserve
  retrieval/rank on the evaluated representation surface.
- HEAD068 is useful but low-rank and redundant.
- HEAD070 keeps strong retrieval while repairing the rank/redundancy failure.
- HEAD076 shows that the first geometry-aware mean-pooling encoder is too
  lossy.

The remaining risk is not another Barlow scaling knob. The remaining risk is
measurement coverage: the workflow still needs a mask-artifact shortcut audit
and geometry/task-stratified probe coverage before claiming robust market-state
representation learning.

## Decision

Do not add a new encoder or decoder next. The next bounded iteration should
extend the Part 1 audit surface in one of two safe ways:

1. **Mask-artifact leakage audit:** train a simple probe from saved/frozen
   embeddings to mask-family or visibility summaries. If this is easy, the
   representation may still encode corruption artifacts.
2. **Scorecard health expansion:** surface variance min/max, singular spectrum
   concentration, and health offdiag from the already-saved artifacts.

Prefer the scorecard health expansion first if it can be done without rerunning
training, because it improves the common comparison harness and may reveal
whether a separate leakage audit is urgent.
