# 262a Postmortem

- best checkpoint: `models/backfill/262a_v0_s42/best_model.pt`
- best epoch: `23`
- suite score: `3/11`

## Mechanism
- latent sigma mean: `0.884` (p10 `0.804`, p90 `0.973`)
- latent sigma std across path: `0.584`
- deterministic idio/common RMS ratio: `0.0000`
- mean common RMS: `0.0168`
- mean idio RMS: `0.0000`
- target factor std mean: `0.881`
- eps target std mean: `0.845`
- predicted sigma vs vol-of-vol Spearman: `0.062`
- target factor std vs vol-of-vol Spearman: `-0.115`

## Read
- the joint family restored a coherent probabilistic decomposition, but the deterministic idio path effectively collapsed to zero
- predicted latent scale is positively aligned with vol-of-vol, so the residual-target misalignment from `261d` is gone
- the remaining failure is severe mean-reversion collapse and over-spiky time-series tails, not a dead scale head
- this first `262a` implementation behaves like a common-factor stochastic generator with too little cross-sectional mean flexibility
