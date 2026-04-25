# 514a Temporal Source-Prior Result

## Context

`513a` selected the last clean source-prior falsifier from the recent literature
route: keep the `392a` AR flow core, but replace white future-step source noise
with AR(1)-correlated Gaussian source increments.

This tested a TSFlow-style source-prior idea without adding a calibration layer,
readout branch, oracle signal, or ensemble.

## Artifacts

- Trainer: `experiments/backfill/block_ar/train_514a_temporal_source_prior_energy_finetune.py`
- Unit test: `test_code/test_empirical_normal_score_source_prior.py`
- Model directory: `models/backfill/514a_temporal_source_ar085_energy_w005_s42/`
- Best checkpoint suite: `results/block_ar/514a_temporal_source_ar085_energy_w005_s42/full11.json`
- Final checkpoint suite: `results/block_ar/514a_temporal_source_ar085_energy_w005_s42/full11_final.json`

## Result

| metric | 392a | 514a best | 514a final |
|---|---:|---:|---:|
| score | `8/11` | `4/11` | `5/11` |
| coverage90 | `0.8675` | `0.9782` | `0.9773` |
| calibration error | `0.0240` | `0.1602` | `0.1734` |
| conditionality MAE reduction | `5.14%` | `5.50%` | `5.61%` |
| daily-change KS pass cells | `25/25` | `15/25` | `16/25` |
| level KS pass cells | `10/25` | `1/25` | `1/25` |
| regime layer2 | `0/8` | `0/8` | `0/8` |
| worst-cell cointegration ratio | `0.278` | `0.105` | `0.193` |
| cross-cell corr ratio | `0.963` | `0.861` | `0.861` |
| rank ratio | `1.495` | `1.915` | `1.897` |
| path max-jump KS | `0.373` | `0.968` | `0.969` |

## Mechanism Read

AR(1)-correlated source increments are too smooth and too broad for this learned
transition map. They preserve the thin conditionality pass, but the official
suite shows a clear failure mode:

- long-horizon intervals become overwide, with overall 90% coverage near `98%`;
- daily-change fidelity drops from `25/25` to `15-16/25`;
- level KS collapses from `10/25` to `1/25`;
- pathwise max-jump KS worsens from about `0.37` to about `0.97`;
- worst-cell cointegration falls below the gate.

This is not a near miss. The source-prior change moves the law away from the
frontier by broadening path occupancy without preserving the local jump and
level geometry that made `392a` deployable.

## Decision

Close temporal-source-prior changes as a primary route. Do not sweep
`path_source_ar`; the one-shot falsifier already shows the mechanism is wrong
for the current frontier. Keep the deployable frontier at the `392a` / `510a`
`8/11` tie.

The next iteration should be post-experiment analysis or paradigm selection,
not another source-prior knob.
