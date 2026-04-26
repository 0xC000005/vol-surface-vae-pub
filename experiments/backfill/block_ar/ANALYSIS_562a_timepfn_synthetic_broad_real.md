# 562a TimePFN Synthetic Prior With Broad Real Fine-Tuning

## Context

561a showed that synthetic pretraining plus recent-only real adaptation is below frontier. The likely issue was insufficient real-data adaptation: the model saw only the 441 recent windows after synthetic pretraining, whereas the historical 340c/392a/510a line inherits a broader real IV transition law.

562a reused the 561a synthetic-pretrained checkpoint but fine-tuned on a much larger pre-validation real window set.

## Run

Broad real fine-tuning:

- Source: `models/backfill/561a_timepfn_synthetic_prior_scaled/best_model.pt`
- Output: `models/backfill/562a_timepfn_synthetic_broad_real`
- Adaptation windows: `2400`
- Epochs: `12`
- Best adaptation loss: `0.4246`

Official-size full-suite:

- Artifact: `results/autoresearch/562a_timepfn_synthetic_broad_real/full11.json`
- Score: `4/11`
- Passed: `surface`, `block_ar`, `cross_cell_correlation`, `mean_reversion`
- Failed: `coverage`, `conditionality`, `time_series`, `cointegration`, `regime_coverage`, `distributional_fidelity`, `pathwise_jump_realism`

## Key Metrics

- coverage90: `0.832`
- h30 worst-cell coverage: `0.224`
- conditionality MAE reduction: `4.2%`
- time-series ACF correlation: `0.956`
- time-series kurtosis ratio: `0.892`
- tail-scale pass cells: `16/25`
- cointegration gen/GT ratio: `0.852`
- cointegration worst-cell ratio: `0.083`
- regime layer2: `0/8`
- daily-change KS pass cells: `21/25`
- level KS pass cells: `5/25`
- cross-cell correlation ratio: `0.603`
- effective-rank ratio: `2.485`
- mean-reversion aggregate ratio: `0.743`
- pathwise max-jump KS: `0.673`

## Mechanism Read

Broad real fine-tuning recovered several real IV dynamics that 561a lacked:

- daily-change KS improved from `10/25` to `21/25`;
- mean reversion moved from fail to pass;
- aggregate time-series kurtosis became realistic;
- cross-cell correlation stayed passable.

But the result remains below frontier because the model became too narrow and locally underinclusive:

- aggregate coverage is only `0.832`;
- h30 worst-cell coverage collapses to `0.224`;
- conditionality remains below the `5%` gate;
- cointegration worst-cell ratio falls to `0.083`;
- regime layer2 remains `0/8`;
- pathwise max-jump KS is still above the relaxed `0.50` gate.

The synthetic prior is not yet adding the specific risk-manager value we need. It helps some real dynamics after broad fine-tuning, but it does not improve stress/regime support versus `510a`.

## Decision

Do not promote 562a. It is not risk-manager deployable and is far below the `510a` prototype.

Keep the branch alive for exactly one clean next falsifier: apply the same patch-energy final adaptation that created the `510a` frontier, but initialize from the synthetic-pretrained + broad-real 562a checkpoint. If patch-energy cannot recover lower coverage and pathwise realism without breaking the recovered real dynamics, the TimePFN-style synthetic-prior branch should be closed for the local-data setting.
