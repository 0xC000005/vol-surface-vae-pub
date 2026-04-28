### Context
714a falsified a global marginal-CRPS repair under the frozen zero-center normalized-innovation recipe. Before adding another loss or changing backend, 715a attributed the remaining failures from the 714a IV, anchor-only, and native joint38 scorecard artifacts.

### Findings
- The scorecard still fails overall: IV, anchor, and joint gates fail while framework consistency passes.
- IV failure is concentrated in long-horizon turbulent coverage: `cov90=0.800`, h30 worst-cell coverage `0.490`, regime layer2 `2/8`, layer3 catastrophic rate `0.059`.
- IV risk-state allocation passes, so the encoder is allocating uncertainty by observable state; the issue is insufficient support/coverage in specific turbulent cells, not total condition blindness.
- Anchor and joint factor failures are the same two factors: `factor:aaa_oas` and `factor:bbb_oas`.
- Native joint38 preserves cross-market co-movement well (`iv_factor_corr.matrix_corr=0.909`) and passes the conditional-panel scorecard check after the subset-audit fix.
- The repeated AAA/BBB OAS failure across anchor-only and joint38 points to a localized coordinate/support/calibration problem for spread-like positive factors, not to joint co-modeling interference.

### Decision
Do not switch backend or add another global proper-score weight yet. The most principled next experiment is a data-coordinate/support repair that remains within the same normalized-innovation AR flow framework: improve positive spread-like factor handling and IV turbulent long-horizon support without changing the core stochastic source, temporal factorization, or tri-scope recipe discipline.
