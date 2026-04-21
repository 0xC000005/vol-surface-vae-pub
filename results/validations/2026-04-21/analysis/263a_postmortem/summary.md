# 263a Postmortem

- best checkpoint: `models/backfill/263a_v0_s42/best_model.pt`
- best epoch: `12`
- suite score: `3/11`

## Mechanism
- latent sigma mean: `13.611` (p10 `12.711`, p90 `14.059`)
- latent sigma std across path: `3.592`
- target factor std mean: `11.863`
- eps target std mean: `0.863`
- predicted sigma vs vol-of-vol Spearman: `0.330`
- target factor std vs vol-of-vol Spearman: `0.017`

## Read
- the recurrent latent state-space backbone restored common structure much better than 262a/262b (`corr_ratio 0.812`, `rank_ratio 1.192`, PC1 50.3%)
- calibration remained strong (`0.034`) and overall coverage stayed in the 262a range (`0.811`)
- but deterministic mean-reversion remained effectively dead (`0.062` aggregate, `0.259` at h30)
- so dynamic latent state alone is not enough; the family still needs an explicit deterministic error-correction mechanism if it is to attack MR while keeping the stronger 263a structure
