### Hypothesis
Adding a small generic marginal CRPS term to the frozen zero-center normalized-innovation AR flow recipe might repair lower-tail/regime under-inclusion without changing the core mechanism. The same recipe was applied to `iv_only`, `anchor_only`, and native `joint38`: same coordinate, AR flow backend, sampler, rollout energy, channel energy, and marginal CRPS weight.

### Execute
- Trained `714a_iv_mcrps010_frozen_e2_s7141`, `714a_anchor_mcrps010_frozen_e2_s7142`, and `714a_joint38_mcrps010_frozen_e2_s7143`.
- Fixed a panel-audit bug where subset scopes such as `anchor_only` compared 13 selected specs against a full 38-channel state block.
- Evaluated the fixed artifacts with the 712a general acceptance scorecard.

### Results
- Overall scorecard: `overall_pass=false`.
- Framework gate: `true`; no disallowed recipe differences across scopes.
- IV gate: `false`; IV remained `7/11`, with effective failed suites `coverage`, `regime_coverage`, and `distributional_fidelity`. Coverage regressed versus the 674a control (`cov90=0.800`, h30 worst-cell coverage about `0.490`, level-KS pass cells `12/25`).
- Anchor gate: `false`; conditional-panel now passes, but `factor_delta_ks` still fails (`11/13` factors pass, mean KS `0.122`).
- Joint gate: `false`; IV-factor co-movement passes (`matrix_corr=0.909`), conditional-panel passes, but `factor_delta_ks` still fails (`11/13` factors pass, mean KS `0.122`).

### Decision
714a falsifies marginal CRPS as the minimal frozen-framework repair. It improves the scorecard shape for anchor/joint conditional-panel checks after the audit fix, but it does not solve per-factor marginal mismatch and worsens IV coverage/level fidelity. The next step should be post-experiment trade-off attribution: identify why the same normalized-innovation recipe still misses two anchor credit-spread factors and why IV lower-tail/regime coverage remains too narrow before adding any new loss term or backend change.
