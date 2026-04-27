# 607a Support-Aware Path Fallback

## Hypothesis

606a opened a path-location fallback branch after 605a showed the hard failures occur under
low-support histories rather than impossible future amplitudes. 607a tests the first
minimal version: keep frozen 510a as the base generator, but for histories whose IV-history
support distance exceeds a training threshold, replace a bounded fraction of samples with
anchored historical training future-increment paths.

This is not a base learned-law improvement. It is a disclosed low-support risk fallback.

## Implementation

Added `experiments/backfill/block_ar/evaluate_607a_support_aware_path_fallback.py` and
focused tests in `test_code/test_607a_support_aware_path_fallback.py`.

Focused tests passed:

```text
pytest test_code/test_607a_support_aware_path_fallback.py -q
2 passed in 1.31s
```

Run configuration:

- base checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`;
- support trigger: IV-history nearest-neighbor distance above train self-support q95;
- support threshold: `8.244`;
- fallback windows: `127/441`;
- increment pool: training future increments with path severity above q80;
- pool size: `802`;
- fallback fraction: `0.50` of samples in each low-support window.

## Result

607a scored `4/11`, so it is not deployable. But it changed the failure profile in a way
the interval adapters did not:

- overall 90% coverage: `79.8%`;
- h1/h7/h14/h30 coverage: `86.5% / 81.4% / 79.8% / 76.6%`;
- per-cell coverage became more balanced, with no high-side overcoverage in the overall
  per-cell table and only h14/h30 worst cells failing;
- conditional MAE reduction: `6.18%`;
- turbulent/calm width ratio: `1.163`, passing the policy signal threshold;
- regime layer2 improved from `0/8` to `1/8`;
- daily-change KS improved to `25/25`;
- level KS improved to `5/25`;
- mean-reversion core passed, though the full suite still failed due full-horizon active
  mean coverage;
- pathwise max-jump KS: `0.431`;
- per-cell extreme jump scale remained weak at `14/25`.

Risk-readiness comparison:

| Candidate | Full suite | Stress score | Worst lower-only cell | Worst regime cell |
| --- | ---: | ---: | ---: | ---: |
| `510a_base_broad` | `5/11` | `1/4` | `0.478` | `0.213` |
| `603a_cell_horizon_adapter` | `4/11` | `1/4` | `0.556` | `0.360` |
| `604a_asymmetric_tail_adapter` | `4/11` | `1/4` | `0.512` | `0.270` |
| `607a_support_path_fallback` | `4/11` | `1/4` | `0.644` | `0.472` |

Artifacts:

- `results/autoresearch/607a_support_aware_path_fallback/full11.json`
- `results/autoresearch/607a_support_aware_path_fallback/full11.md`
- `results/autoresearch/607a_support_aware_path_fallback/risk_readiness.json`
- `results/autoresearch/607a_support_aware_path_fallback/risk_readiness.md`

## Mechanism Read

Path-location fallback is not enough yet, but it is the first branch to move the sparse
stress-inclusion geometry in the right direction. It improves worst lower-only coverage,
worst regime-cell inclusion, regime width differentiation, daily-change fidelity, and
mean-reversion shape. That supports the 606a diagnosis: the remaining issue is path
location under low support.

The failure is that the fallback is still too blunt. It improves inclusion but does not
clear the lower-only gates, and it creates new authenticity failures in per-cell extreme
jump scale and conditionality edge cases.

## Decision

Keep this branch alive for one controlled follow-up. Do not return to interval scaling.
The next experiment should test a single, more aggressive path-location setting, not a
broad knob sweep: increase the fallback fraction while keeping the same support trigger
and increment-pool definition. If that improves lower-only regime/cell inclusion without
destroying the improved daily-change/mean-reversion behavior, the branch remains viable;
otherwise it should be closed as a risk overlay too blunt for deployment.
