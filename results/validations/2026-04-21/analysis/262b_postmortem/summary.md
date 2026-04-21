# 262b Postmortem

- best checkpoint: `models/backfill/262b_v0_s42/best_model.pt`
- best epoch: `28`
- suite score: `2/11`

## Mechanism
- latent sigma mean: `0.289` (p10 `0.223`, p90 `0.311`)
- latent sigma std across path: `0.060`
- deterministic idio/common RMS ratio: `0.000002`
- mean common RMS: `0.0182`
- mean idio RMS: `0.000000`
- target factor std mean: `0.252`
- eps target std mean: `0.854`
- EC gain mean: `0.096`
- predicted sigma vs vol-of-vol Spearman: `0.116`
- target factor std vs vol-of-vol Spearman: `-0.253`

## Read
- the bounded history-mean EC baseline materially restored deterministic MR shape, but it did not revive the deterministic idio mean path
- idio mean still effectively collapsed, so most of the correction came through the shared mean path plus EC baseline
- uncertainty scale stayed positively aligned with vol-of-vol, but absolute dispersion shrank too far, which drove the large coverage collapse
- this means 262b did not solve the core 262 problem cleanly; it traded 262a's no-MR failure for under-dispersed mean-corrected paths with weakened cross-cell structure
