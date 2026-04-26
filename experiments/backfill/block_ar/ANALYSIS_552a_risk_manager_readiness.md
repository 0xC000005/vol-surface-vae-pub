# 552a Risk-Manager Readiness Audit

## Context

The user changed the product framing: overcoverage is no longer a primary concern if
the generator is used for risk stress testing. The relevant question becomes whether
the scenarios are conditionally responsive, individually realistic, and useful as a
stress envelope.

552a therefore re-scored existing candidates under a risk-manager lens:

- ignore high-side overcoverage caps;
- retain lower inclusion requirements;
- retain conditionality;
- retain surface/path/dependence/scenario authenticity;
- expose whether the current generator is IV-only or factor-ready.

Artifacts:

- Script: `experiments/backfill/block_ar/audit_552a_risk_manager_readiness.py`
- Tests: `test_code/test_552a_risk_readiness_audit.py`
- Report: `results/autoresearch/552a_risk_manager_readiness/risk_readiness.md`
- JSON: `results/autoresearch/552a_risk_manager_readiness/risk_readiness.json`

## Result

| Candidate | Original score | Stress score | Stress pass | Main blockers |
| --- | ---: | ---: | --- | --- |
| `510a_patch_energy` | `8/11` | `3/4` | `False` | regime under-inclusion, level KS warning |
| `392a_base` | `8/11` | `2/4` | `False` | coverage under-inclusion, regime under-inclusion, level KS warning |
| `551a_state_stress` | `6/11` | `1/4` | `False` | conditionality borderline, regime under-inclusion, level KS warning |

The best current risk-lens candidate is `510a_patch_energy`, not `551a`. It passes
lower-only coverage, conditionality, and scenario authenticity, but still fails
lower-only regime inclusion:

- lower-only coverage: pass, cov90 `0.873`, worst cell `0.714`;
- conditionality: pass, MAE reduction `5.123%`;
- scenario authenticity: pass;
- lower-only regime inclusion: fail, worst regime cell `0.538`;
- factor readiness: fail, current generator is IV-surface-only.

## Mechanism Read

Removing overcoverage as a failure does not make the current system deployable. The
main blocker shifts from interval calibration to conditional stress allocation:

- `392a` and `510a` produce realistic-looking IV paths and preserve dependence;
- `551a` widened the envelope but damaged conditionality and cointegration;
- all candidates still miss regime-cell under-inclusion;
- none generates or conditions on the available non-IV factor paths.

## Decision

For a risk manager, `510a` can be shown only as an IV-only prototype with explicit
caveats. It is not yet presentable as a complete risk scenario generator.

The next research move should not be overcoverage tuning. It should target regime
under-inclusion and factor conditioning: the model must know when the current market
state implies stress, and the scenario should be coherent with observed return, price,
level, slope, and skew factors.
