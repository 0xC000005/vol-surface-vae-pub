# 563a TimePFN Synthetic Prior Patch-Energy Closure

## Context

562a showed that synthetic pretraining plus broad real fine-tuning recovered some real IV dynamics but remained far below `510a`. 563a applied the same patch-energy final adaptation pattern that created the `510a` frontier, using the synthetic-pretrained + broad-real checkpoint as initialization.

## Run

Patch-energy adaptation:

- Source: `models/backfill/562a_timepfn_synthetic_broad_real/best_model.pt`
- Output: `models/backfill/563a_timepfn_synthetic_broad_real_patch_energy`
- Recent windows: `441`
- Epochs: `4`
- Objective: FM anchor `1.0` plus patch-energy weight `0.05`, patch length `5`
- Best validation-total epoch: `2`

Official full-suite artifacts:

- Final checkpoint: `results/autoresearch/563a_timepfn_synthetic_broad_real_patch_energy/full11_final.json`
- Validation-best checkpoint: `results/autoresearch/563a_timepfn_synthetic_broad_real_patch_energy/full11_best.json`

## Results

Final checkpoint:

- Score: `4/11`
- Passed: `surface`, `block_ar`, `cointegration`, `cross_cell_correlation`
- Failed: `coverage`, `conditionality`, `time_series`, `regime_coverage`, `distributional_fidelity`, `mean_reversion`, `pathwise_jump_realism`

Validation-best checkpoint:

- Score: `5/11`
- Passed: `surface`, `time_series`, `block_ar`, `cointegration`, `cross_cell_correlation`
- Failed: `coverage`, `conditionality`, `regime_coverage`, `distributional_fidelity`, `mean_reversion`, `pathwise_jump_realism`

Best-checkpoint key metrics:

- coverage90: `0.856`
- h30 worst-cell coverage: `0.344`
- conditionality MAE reduction: `3.9%`
- regime layer2: `0/8`
- daily-change KS: `22/25`
- level KS: `7/25`
- cointegration gen/GT ratio: `0.800`
- cointegration worst-cell ratio: `0.278`
- cross-cell correlation ratio: `0.698`
- mean-reversion aggregate ratio: `0.672`
- pathwise max-jump KS: `0.711`

## Mechanism Read

Patch-energy recovered some local distributional shape:

- daily-change KS reached `22/25`;
- time-series suite passed for the validation-best checkpoint;
- cointegration and cross-cell structure remained passable.

But it did not create a risk-manager-deployable model:

1. conditionality remains below gate (`3.9%`);
2. lower coverage is much worse than `510a` (`h30` worst cell `0.344` versus `510a` around `0.714` worst-cell lower-only coverage);
3. regime layer2 remains `0/8`;
4. level occupancy remains weak (`7/25`);
5. mean reversion is just below gate;
6. pathwise max-jump KS is worse than the relaxed gate.

The decisive result is that synthetic-prior pretraining does not improve the current local-data frontier after broad real fine-tuning and patch-energy adaptation. It recovers pieces of the real law but does not fix the risk-manager blocker: localized stress/regime inclusion under coherent conditional paths.

## Decision

Close the local TimePFN-style synthetic-prior branch for the current data setting.

The branch was worth trying because it directly targeted sparse regime exposure, but the evidence is now clear:

- 560a: scaffold works, tiny smoke model `1/11`;
- 561a: scaled synthetic pretrain plus recent adaptation `4/11`;
- 562a: synthetic pretrain plus broad real fine-tune `4/11`;
- 563a: patch-energy final adaptation `5/11` best.

None approaches the `510a` risk prototype (`8/11`, risk-readiness `3/4`). The next deployability path should not be more synthetic-prior tuning. It should either package `510a` as the base learned law with a clearly separated conservative stress-scenario selection policy, or move to genuinely new external/multi-underlying data.
