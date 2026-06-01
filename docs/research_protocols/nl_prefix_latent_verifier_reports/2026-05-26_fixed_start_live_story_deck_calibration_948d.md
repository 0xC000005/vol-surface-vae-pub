# Independent Verifier: Fixed-Start Live Story-Deck Calibration 948d

Date: 2026-05-26

## Verification Result

Verdict: `PARTIAL`

The 948d live story-deck sweep supports the narrow claim that the user-facing
Gradio API path now runs the calibrated support-grounded narrative workflow
across six professional narratives at the same explicit starting level. It also
supports the support-conditionality claim for this live demo deck: each
narrative selected a disjoint support set under the same start. It does not by
itself prove full production readiness or replace the held-out historical
backtest evidence.

## What I Checked

- Live API smoke implementation:
  `experiments/backfill/block_ar/nl_prefix_latent_gradio_api_smoke.py`
  - checks calibration metadata, support gate, and active directional claims;
  - redraws fan charts using the app-returned default analogue scope rather
    than a stale hard-coded `ALL`;
  - writes per-case JSON/markdown/array snapshots so later casebook runs do not
    overwrite evidence.
- Live API casebook implementation:
  `experiments/backfill/block_ar/nl_prefix_latent_gradio_live_api_casebook.py`
  - aggregates calibration-applied counts and minimum support gate;
  - can run the default professional story deck at one fixed start;
  - writes markdown even when failed cases have partial schemas.
- Reproducible story-deck analysis:
  `experiments/backfill/block_ar/nl_live_story_deck_analysis.py`.
- Tests:
  `test_code/test_808a_nl_prefix_latent_gradio_api_smoke.py` and
  `test_code/test_809a_nl_prefix_latent_gradio_live_api_casebook.py`, plus
  `test_code/test_948d_nl_live_story_deck_analysis.py`.
- Live sweep summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/gradio_live_api_casebook_summary.json`.
- Conditionality summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/fixed_start22_calibrated_story_deck_conditionality_summary.json`.
- Terminal-delta panel:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/fixed_start22_terminal_mean_deltas.png`.
- Per-case snapshots under:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/*/prefix_report_snapshot.json`
  and `*/prefix_arrays_snapshot.npz`.

## Findings

### Confirmed Correct

- The live story-deck run passed `6/6` cases through the public Gradio API
  endpoint at fixed start index `22`.
- Calibration applied in `6/6` cases.
- Minimum calibration support gate was `1.0`.
- Total OpenAI token usage recorded by the run was `11240`.
- Per-case report and array snapshots exist, avoiding the previous issue where
  all case summaries pointed back to the shared app output path.
- Pairwise support Jaccard across the six fixed-start narratives was `0.0`.
  This verifies that different narratives selected different support sets even
  when the start level was held fixed.
- A smaller three-case live casebook also passed before the six-case run:
  `3/3` pass, calibration applied `3/3`, min support gate `1.0`.
- Focused API/casebook regression tests passed:
  `11 passed` for
  `test_code/test_808a_nl_prefix_latent_gradio_api_smoke.py` and
  `test_code/test_809a_nl_prefix_latent_gradio_live_api_casebook.py`.
- The reusable analysis test passed:
  `2 passed` for `test_code/test_948d_nl_live_story_deck_analysis.py`.

### Issues Found

- `WARNING`: This is a live demo-path sweep, not a held-out scenario-quality
  backtest. It should be paired with the 947b held-out quality evidence rather
  than used alone.
- `WARNING`: The result proves support-set conditionality and calibrated live
  wiring. It does not prove that every individual factor fan will show strong
  visual separation for every narrative.
- `WARNING`: The sweep uses one explicit starting level, start index `22`. This
  is the right fixed-start product test, but broader production promotion still
  needs multiple start levels and the existing 947b five-start evidence.
- `NOTE`: Some support sets have only one or two components after the
  non-overlap/diversity filters. That is acceptable for this demo-path gate but
  remains a product-display caveat.

## Alternative Explanations

- Disjoint support sets could partly come from strong narrative direction
  filters rather than full semantic nuance. This is still useful for auditability
  but does not settle the richer text-representation question.
- The bounded directional delta calibration can improve visible response while
  remaining a post-rollout adjustment; it is not direct language-conditioned
  decoding.

## My Independent Assessment

The 948d sweep closes the immediate verifier gap from the prior live smoke: the
calibrated path is no longer shown only on one story. The current demo can now
claim that six professional narratives at the same starting level are routed
through the live Gradio API, retain warning/provenance fields, apply the
support-gated calibration, and select distinct historical support sets.

This strengthens the demo-facing conditionality story. The broader production
objective remains active because full readiness also requires the 947b
held-out quality evidence, multi-start validation, and product-facing visual
readouts to remain synchronized.

## Recommended Action

Use 948d as the live-demo path validation artifact and keep 947b as the
held-out quantitative paper/demo evidence surface. The next iteration should
refresh or verify the public paper/demo figures against these calibrated
artifacts if the UI or paper claims are updated.
