### Context

719a tested the 718a mechanism read with the smallest generic change: keep the incumbent zero-center normalized-innovation AR flow checkpoints and add a deterministic sticky-zero readout for channels whose training data has a large empirical no-change atom. The rule is variable-type based, not OAS-specific: if train no-change rate is at least `0.25`, raw generated moves below the train nonzero-move q10 are decoded as no-change.

### Result

The frozen tri-scope scorecard remains non-deployable: overall `false`, framework `true`, IV `false`, anchor `false`, joint `false`.

- Sticky channel selection was clean: only `factor:aaa_oas` and `factor:bbb_oas` were selected. IV channels were untouched.
- Anchor-only improved from the frozen baseline `11/13` factor-KS pass to `12/13`; mean factor KS improved to `0.093`; conditional panel passed; BBB OAS remained the only factor-KS failure at `0.225`.
- Joint38 improved to `12/13` factor-KS pass; mean factor KS `0.091`; conditional panel and IV-factor corr passed; BBB OAS remained the only factor-KS failure at `0.221`.
- Joint factor-factor correlation narrowly failed the absolute-amplitude ratio gate: generated/GT mean abs corr ratio `0.345` versus required `0.35`.
- IV reused the 674/711 incumbent because the sticky rule selects no IV channels; IV remains `7/11` with effective failures `coverage`, `regime_coverage`, and `distributional_fidelity`.

Artifacts:

- `experiments/backfill/block_ar/audit_719a_sticky_zero_readout.py`
- `results/block_ar/719a_sticky_zero_readout_scorecard/anchor_val_panel_s64.json`
- `results/block_ar/719a_sticky_zero_readout_scorecard/joint_val_panel_s64.json`
- `results/block_ar/719a_sticky_zero_readout_scorecard/scorecard.json`

### Mechanism Read

The sticky/no-change diagnosis is directionally correct: the generic atom readout improves both anchor-only and joint38 factor marginal realism without changing the backend or creating a glued deck. The conservative tick-level threshold is not strong enough for BBB OAS, and snapping small moves slightly attenuates joint factor correlation amplitude.

### Decision

719a is not the deployable candidate, but it validates the sticky-spread failure class. The next HEAD step should analyze the residual BBB error before changing the model again: separate zero-mass mismatch, nonzero-move tail mismatch, and correlation attenuation. If the residual is mainly atom mass, the next experiment should use an explicit mixed discrete-continuous innovation variable rather than a stronger post-hoc threshold.
