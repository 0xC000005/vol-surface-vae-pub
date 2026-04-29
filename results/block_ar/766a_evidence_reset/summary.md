# 766a Evidence Reset

- primary decision: Do not promote 766a over the protected 734a/739a deployable incumbent. Use the short-prefix recipe as the active research ingredient only after tri-scope non-regression, because it improves IV-only validation but does not fix sticky anchor channels and regresses native joint IV level support.
- deployable incumbent: `734a/739a real-VIX tri-scope framework`
- active research candidate: `766a short-prefix tri-scope transfer, with 755a remaining the IV-only frontier reference`

## IV Full-Suite Reruns

| run | score | failed suites | cov90 | calerr | level KS | median | risk-state | mean reversion | econ-link | path KS | regime L2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| 734a_iv_only_incumbent_rerun | 5/11 | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, mean_reversion | 0.8393 | 0.0442 | 15/25 | 16/25 | True | False | True | 0.4385 | 1/8 |
| 734a_native_joint_iv_slice_rerun | 6/11 | coverage, conditionality, time_series, regime_coverage, mean_reversion | 0.8143 | 0.0597 | 19/25 | 22/25 | True | False | True | 0.2925 | 1/8 |
| 755a_iv_only_shortprefix_frontier_rerun | 7/11 | coverage, conditionality, regime_coverage, distributional_fidelity | 0.8178 | 0.0632 | 15/25 | 15/25 | True | True | True | 0.3866 | 1/8 |
| 755a_iv_only_shortprefix_train_tail_rerun | 8/11 | conditionality, time_series, regime_coverage | 0.8429 | 0.0324 | 22/25 | 25/25 | False | True | True | 0.4830 | 2/8 |
| 766a_native_joint_shortprefix_iv_slice | 6/11 | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity | 0.8026 | 0.0756 | 13/25 | 17/25 | True | True | True | 0.3710 | 1/8 |

## Anchor And Joint Panel Audits

| run | finite | delta KS mean | KS pass | tail pass | factor corr | IV-factor corr | cond reduction | width rho | failing KS factors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 734a_anchor_only_incumbent_rerun | 1.0000 | 0.1069 | 12/14 | 14/14 | 0.9070 | n/a | 5.111% | 0.9263 | factor:aaa_oas, factor:bbb_oas |
| 734a_native_joint_panel_rerun | 1.0000 | 0.0971 | 12/14 | 14/14 | 0.8628 | 0.9472 | 5.011% | 0.9502 | factor:aaa_oas, factor:bbb_oas |
| 766a_anchor_only_shortprefix_transfer | 1.0000 | 0.1090 | 12/14 | 14/14 | 0.9033 | n/a | 4.893% | 0.9284 | factor:aaa_oas, factor:bbb_oas |
| 766a_native_joint_shortprefix_transfer | 1.0000 | 0.0957 | 12/14 | 14/14 | 0.8658 | 0.9423 | 4.869% | 0.9499 | factor:aaa_oas, factor:bbb_oas |

## Scorecards

- `734a_updated_scorecard` overall pass: `False`; gates: `{'iv': False, 'anchor': False, 'joint': False, 'framework': True}`
- `734a_updated_scorecard` IV effective failures: `['coverage', 'regime_coverage', 'distributional_fidelity', 'mean_reversion']`
- `734a_updated_scorecard` anchor failures: `['factor_delta_ks']`; joint failures: `['factor_delta_ks']`
- `766a_shortprefix_updated_scorecard` overall pass: `False`; gates: `{'iv': False, 'anchor': False, 'joint': False, 'framework': True}`
- `766a_shortprefix_updated_scorecard` IV effective failures: `['coverage', 'regime_coverage', 'distributional_fidelity']`
- `766a_shortprefix_updated_scorecard` anchor failures: `['factor_delta_ks']`; joint failures: `['factor_delta_ks']`

## Remaining Bottlenecks

- IV validation coverage/regime layer-2 and level/median allocation
- native joint IV level support versus IV-only 755a
- sticky low-activity AAA/BBB OAS no-update mass in anchor and joint panels
- single-seed summary sensitivity around borderline old cointegration and mean-reversion gates
