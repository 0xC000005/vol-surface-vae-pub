# 711a Normalized-Innovation Active-Family Audit

## Candidate Summary

| Candidate | Full 11 | Stress | Cov90 | h30 worst | Cond MAE | Risk-state | Regime under slices | Level-KS overlap | Key failed suites |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `510a_splitpass` | `7/11` | `2/4` | `0.871` | `0.729` | `5.46%` | `True` | `22` | `0.500` | coverage, cointegration, regime_coverage, distributional_fidelity |
| `674a_iv_norminnov_current` | `7/11` | `1/4` | `0.837` | `0.524` | `10.16%` | `True` | `23` | `0.667` | coverage, conditionality, regime_coverage, distributional_fidelity |
| `698b_joint38_norminnov_current` | `7/11` | `1/4` | `0.771` | `0.583` | `9.77%` | `True` | `39` | `0.333` | coverage, conditionality, time_series, regime_coverage |

## Mechanism Read

- risk-state allocation valid in active family: `True`
- active family risk-manager deployable now: `False`
- common failure: The normalized-innovation family now shows real state-dependent uncertainty allocation, but it is locally too narrow in regime/cell/horizon slices. That is a dispersion/allocation failure, not evidence that the encoder is unused.

## Decision

Stay on the normalized-innovation framework and run a frozen-framework coverage repair experiment that targets lower-tail/regime under-inclusion inside the same state-normalized innovation law. Do not return to 510a except as a control, because 510a is less general even when its IV-only stress score is higher.
