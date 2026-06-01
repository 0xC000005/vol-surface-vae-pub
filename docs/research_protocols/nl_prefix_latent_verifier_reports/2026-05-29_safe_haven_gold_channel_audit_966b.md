# Independent Verification: Safe-Haven Gold Channel Audit 966b

Verdict: `AGREE`

## What I Checked

- Audit script:
  `experiments/backfill/block_ar/nl_safe_haven_gold_channel_audit.py`
- Source case artifacts:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_966a_professional_start22_s384_d400/safe_haven_gold/`
- Generated audit artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/safe_haven_gold_channel_audit_966b/safe_haven_gold_channel_audit.json`
- Demo/table implementation:
  `experiments/backfill/block_ar/nl_paper_fixed_start_demo_casebook_tables.py`
  and `experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py`
- Paper appendix table:
  `paper/narrative_grounded_scenarios/generated_tables/table_fixed_start_demo_casebook_readout.tex`

## Confirmed Correct

- The Safe-haven Gold appendix table is generated from the saved `966a`
  professional start-22 reports and reproduces the Gold row exactly.
- `factor:gold` is index `37` in `JOINT39_SPEC_NAMES`, matching the generated
  state array channel used by the audit.
- The grounding sidecar extracts a high-confidence `GOLD up` current/recent
  claim.
- The selected Safe-haven top3/90 support prefixes pass the Gold-up prefix
  direction check.
- The day-30 generated Gold distribution is effectively unchanged versus the
  start-only top3/90 baseline:
  - narrative top3/90: mean `+4.98` points, `57.7%` up paths;
  - start-only top3/90: mean `+5.08` points, `59.3%` up paths.

## Issues Found

- `WARNING`: This audit uses saved generator artifacts and does not rerun the
  frozen SNI rollout from checkpoint. It is sufficient to check the paper/demo
  table and channel interpretation, but it is not a new scenario-generation
  experiment.
- `NOTE`: The support direction check validates the current/recent prefix, not
  the terminal future direction. A Gold-up prefix can be followed by a
  baseline-neutral day-30 Gold forecast distribution.

## Independent Assessment

There is no evidence of a table-generation, factor-index, or top3/90 selection
bug. The weak Gold terminal response is an expected current workflow outcome for
this fixed start: the Safe-haven narrative selects Gold-up support prefixes, but
the frozen SNI rollout produces a Gold terminal distribution similar to the
start-only baseline.

The public interpretation should be: Safe-haven Gold is prefix-supported but
terminal-neutral versus baseline for Gold in this case. The narrative impact
should be discussed through baseline-relative channels and provenance, not as a
promise that Gold must end higher at day 30.

## Recommended Action

Proceed with the current top3/90 workflow. Keep the paper/demo wording
baseline-relative. If stronger factor-specific terminal sensitivity is required,
open a separate candidate method branch for narrative-channel response weighting
or calibration and benchmark it against CRPS/energy and fixed-start controls.
