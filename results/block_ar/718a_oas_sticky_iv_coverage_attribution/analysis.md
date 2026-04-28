# 718a OAS Sticky-Increment And IV Coverage Attribution

## Data Object Audit

| policy | factor | transform | train zero | val zero | raw KS | norm KS | val/train raw q99 | val hist/fut activity rho | val zero/zero rho |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reference_based` | `factor:aaa_oas` | `diff_level` | `0.435` | `0.474` | `0.058` | `0.090` | `0.222` | `0.468` | `0.474` |
| `reference_based` | `factor:bbb_oas` | `diff_level` | `0.265` | `0.383` | `0.136` | `0.134` | `0.417` | `0.635` | `0.343` |
| `observed_positive` | `factor:aaa_oas` | `log_level` | `0.435` | `0.474` | `0.058` | `0.093` | `0.222` | `0.468` | `0.474` |
| `observed_positive` | `factor:bbb_oas` | `log_level` | `0.265` | `0.383` | `0.136` | `0.134` | `0.417` | `0.635` | `0.343` |

## Recent Model OAS Behavior

| run | pass | mean KS | AAA KS | BBB KS | AAA gen range | BBB gen range | IV-factor corr | conditional panel |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `688_anchor_reference_baseline` | `11/13` | `0.107` | `0.286` | `0.292` | `[0.418, 1.825]` | `[1.181, 5.454]` | `nan` | `nan` |
| `676_joint_reference_baseline` | `11/13` | `0.101` | `0.240` | `0.260` | `[0.159, 1.758]` | `[1.146, 4.120]` | `0.896` | `nan` |
| `714_anchor_mcrps` | `11/13` | `0.122` | `0.296` | `0.349` | `[0.417, 2.028]` | `[1.217, 4.924]` | `nan` | `4.525` |
| `714_joint_mcrps` | `11/13` | `0.122` | `0.278` | `0.338` | `[0.294, 2.602]` | `[1.232, 4.324]` | `0.909` | `4.057` |
| `716_anchor_innovscore` | `12/13` | `0.108` | `0.244` | `0.194` | `[0.035, 2.183]` | `[1.203, 3.863]` | `nan` | `2.970` |
| `716_joint_innovscore` | `11/13` | `0.114` | `0.203` | `0.265` | `[0.240, 2.276]` | `[1.246, 4.668]` | `0.893` | `3.153` |
| `717_anchor_observed_positive` | `11/13` | `0.130` | `0.313` | `0.407` | `[0.475, 6.135]` | `[1.240, 6.988]` | `nan` | `3.932` |
| `717_joint_observed_positive` | `11/13` | `0.123` | `0.285` | `0.327` | `[0.450, 1.378]` | `[1.244, 3.582]` | `0.908` | `3.842` |

## IV Coverage Side

| run | full 11 | cov90 | h30 worst | level KS | max-jump KS | risk-state | failed suites |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `674_iv_reference_baseline` | `6/11` | `0.829` | `0.522` | `15/25` | `0.451` | `False` | coverage, conditionality, regime_coverage, distributional_fidelity, mean_reversion |
| `711_674_iv_current` | `7/11` | `0.837` | `0.524` | `15/25` | `0.446` | `True` | coverage, conditionality, regime_coverage, distributional_fidelity |
| `714_iv_mcrps` | `7/11` | `0.800` | `0.490` | `12/25` | `0.346` | `True` | coverage, conditionality, regime_coverage, distributional_fidelity |
| `716_iv_innovscore` | `5/11` | `0.766` | `0.408` | `9/25` | `0.740` | `True` | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, pathwise_jump_realism |
| `717_iv_observed_positive` | `6/11` | `0.715` | `0.313` | `8/25` | `0.344` | `True` | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity |

## Mechanism Read

- failure class: `sticky spread coordinate/objective failure plus separate IV long-horizon undercoverage`
- support-policy read: Observed-positive log-level support changes the coordinate but does not remove the OAS failure; the raw OAS series has high no-change mass and large state-dependent jumps, so a continuous-only innovation law smears an atom at zero into small and occasional excessive moves.
- train/validation read: OAS raw-increment drift is not the dominant explanation if raw train/val KS stays moderate while generated OAS KS remains high across trainable variants. The issue is primarily data-object mismatch inside the continuous law.

## Decision

Do not add another global loss or backend switch. The next decisive move is a generic sticky/mixed discrete-continuous innovation coordinate for channels with empirical no-change atoms, frozen across IV, anchor, and joint scopes. It should be formulated as a variable-type data coordinate, not an OAS-only special case.
