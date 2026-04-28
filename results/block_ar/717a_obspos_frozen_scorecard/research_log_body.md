### Context

717a closed the narrow support-coordinate falsifier after 716a rejected global innovation-score coordinates. The hypothesis was that AAA/BBB OAS failures might be caused by treating strictly positive rate/spread-like factors as diff-level variables. The test kept the same frozen AR flow framework, zero-center normalized innovations, sampler, and fm/rollout/channel loss recipe, while using `positive_level_policy=observed_positive` so observed-positive factors move to log-level coordinates.

### Result

The frozen tri-scope scorecard remained non-deployable: overall `false`, framework gate `true`, IV `false`, anchor `false`, joint `false`.

- IV-only: `6/11`; effective failures were `coverage`, `cointegration`, `regime_coverage`, and `distributional_fidelity`; cov90 `0.715`; risk-state allocation passed; daily-change KS `24/25`; level KS `8/25`; max-jump KS `0.344`.
- Anchor-only: factor delta KS failed `11/13` with mean `0.130`; tails passed `13/13`; factor-factor corr upper `0.903`; conditional panel passed. The failing factors were still `aaa_oas` and `bbb_oas`; generated maxima were too high (`6.13` vs GT `0.99` for AAA, `6.99` vs GT `3.03` for BBB).
- Joint38: factor delta KS failed `11/13` with mean `0.123`; tails passed `13/13`; factor-factor corr upper `0.836`; IV-factor matrix corr `0.908`; conditional panel passed. AAA/BBB OAS remained the only factor-KS blockers.

Artifacts:

- `results/block_ar/717a_obspos_frozen_scorecard/scorecard.json`
- `results/block_ar/717a_obspos_frozen_scorecard/iv_val_full11_s64.json`
- `results/block_ar/717a_obspos_frozen_scorecard/anchor_val_panel_s64.json`
- `results/block_ar/717a_obspos_frozen_scorecard/joint_val_panel_s64.json`

### Mechanism Read

Observed-positive log-level support is not the missing repair. It worsened OAS excursions rather than correcting them, so the OAS failure is not just a positivity/support-coordinate issue. The repeated pattern now points to sticky or zero-inflated spread dynamics and state-dependent spread jumps that the current continuous normalized-innovation coordinate does not represent cleanly. The IV regression also means 717a is not a safe replacement for the normalized-coordinate incumbent.

### Decision

Reject 717a as a frozen candidate. The next HEAD step should be post-experiment analysis, not another model tweak: diagnose AAA/BBB OAS no-change/sticky-increment structure and re-check IV long-horizon coverage attribution before introducing any new coordinate, objective, or architecture change.
