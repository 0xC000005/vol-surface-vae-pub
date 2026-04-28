### Hypothesis
The existing innovation-score coordinate might be the clean support/coordinate repair suggested by 715a: map normalized innovations through train-fitted empirical normal-score marginals while keeping the same AR flow, sampler, and vanilla FM objective. This tests whether the 698a/698b joint-only gain can become a frozen tri-scope framework candidate.

### Execute
- Trained `716a_iv_innovscore_boundediv_e8_w2048_s7161` and `716a_anchor_innovscore_boundediv_e8_w2048_s7162` with the same base recipe as existing `698a_joint38_innovscore_boundediv_e8_w2048_s6981`.
- Evaluated IV-only with the full 11-suite and anchor/joint with the panel audit.
- Scored the tri-scope recipe with the 712a general acceptance scorecard.

### Results
- Overall scorecard: `overall_pass=false`.
- Framework gate: `true`.
- IV gate: `false`; IV regressed to `5/11`, failing `coverage`, `cointegration`, `regime_coverage`, `distributional_fidelity`, and `pathwise_jump_realism` after replacing the old conditionality gate with risk-state allocation. Key metrics: `cov90=0.766`, level-KS `9/25`, max-jump KS `0.740`.
- Anchor gate: `false`; factor KS improved to `12/13`, but `factor:aaa_oas` still fails. Conditional-panel and factor correlation checks pass.
- Joint gate: `false`; factor KS remains `11/13`, with `factor:aaa_oas` and `factor:bbb_oas` failing. IV-factor co-movement remains healthy (`matrix_corr=0.893`).

### Decision
716a rejects innovation-score as the frozen base tri-scope recipe. It is useful evidence that empirical marginal scoring can help factor daily-shape realism, but it creates unacceptable IV level/path artifacts without additional rollout repair. The next principled repair should return to the normalized-coordinate incumbent and test the more localized support-coordinate change for positive spread-like factors rather than using score-coordinate globally.
