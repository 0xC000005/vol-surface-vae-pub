# Independent Verification: Paper/Demo Surface Sync to 949a

Date: 2026-05-26

## Verdict

AGREE, scoped to paper/demo synchronization.

The edited paper surface now reflects the latest combined evidence bundle as a
paper/demo candidate rather than a production-ready or direct prompt-to-scenario
claim. The new table and prose are consistent with the current 949a production
readiness audit and the 948d fixed-start live story-deck artifact.

## What I Checked

- `paper/narrative_grounded_scenarios/main.tex`
- `paper/narrative_grounded_scenarios/generated_tables/table_combined_readiness_audit.tex`
- `docs/research_protocols/nl_prefix_latent_current_truth.md`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_production_readiness_audit_949a/production_readiness_audit.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/fixed_start22_calibrated_story_deck_conditionality_summary.json`

## Confirmed Correct

- The paper reports the 949a headline metrics consistently:
  - CRPS improvement vs persistence: `0.216852726562`
  - energy improvement vs persistence: `0.306246897599`
  - 80% coverage: `0.822636622637`
  - start-normalized narrative plus interaction: `0.5239292248283334`
  - fixed-start factor KS: `0.3283035714285714`
  - fixed-start portfolio KS: `0.4110416666666667`
  - live max support Jaccard: `0.0`
- The paper correctly says the live interface validation used six professional
  narratives with one explicit historical start and that calibration was applied
  in all six cases.
- The paper does not promote the method as fully automatic production behavior.
  It explicitly scopes the claim as demo/paper validation.
- The LaTeX source compiles after two `pdflatex` passes, and the log check did
  not find unresolved references or LaTeX warnings after the second pass.

## Issues Found

- NOTE: This verification only checks the paper/demo synchronization. It does
  not rerun the live OpenAI demo path or regenerate the 947b/948d/949a artifacts.
- NOTE: The worktree contains many pre-existing natural-language workflow
  changes. This verifier only covers the paper sync files named above.

## Recommended Action

Proceed with the synchronized paper/demo surface. Keep the active research goal
open because 949a itself still reports `goal_complete=false` and scopes the
current state as a paper/demo candidate.
