# 598a IV-Only Cap and Joint-Factor Signal-Audit Shift

## Context

593a-597a tested the clean remaining IV-only routes around the 510a frontier:

- learned common-latent AR wrapper;
- same-frame frontier audit;
- final-path joint-distribution finetuning;
- global temperature/noise calibration.

All routes either underperform same-frame 510a or improve one diagnostic while damaging another. The broad-frame bottleneck is stable: the IV-only model does not know where to allocate future level mass across sparse cells, regimes, and horizons.

## Local Evidence

Available data:

- `data/vol_surface_with_ret.npz`: `surface`, `ret`, `price`, `slopes`, `skews`, `levels`;
- `data/multi_factor_data.npz`: 13 factor levels and 13 corresponding returns/diffs;
- factor list: `spx`, `usdcad`, `usdjpy`, `dxy`, `copper`, `wheat`, `crude_oil`, `us2y`, `us10y`, `aaa_oas`, `bbb_oas`, `nikkei`, `gold`.

Prior relevant findings:

- Exp 100 found SPX returns were mostly redundant after IV history for future IV changes, so adding one obvious factor is not automatically useful.
- 537a trained a native 51-variable panel daily Cholesky transition model and scored only `4/11`; the Gaussian innovation was too broad/symmetric and did not preserve per-cell IV law.
- 573a/574a produced a risk-manager acceptable IV-plus-anchor-factor stress deck, but it is explicitly a stress product, not a calibrated joint probability law.
- 575a/576a already identified the publication path as a native unified increment panel with no duplicate level/return targets and audited the reversible data framing.
- 448a-450a and 597a together close the source-noise calibration family: global or conditional scale is not enough.

## Decision

The correct paradigm shift is not "train another joint model immediately." The correct shift is:

1. Treat IV-only model research as capped for now.
2. Treat the joint-factor route as the only remaining learned-law path that could be publishable.
3. Before training, run a factor signal audit on the actual hard IV failure windows.

The audit should answer one question:

Does factor history add out-of-sample information about the windows/cells where 510a is undercovered or level-biased, beyond IV history alone?

If yes, the next model should be a native unified factor generator. If no, then the remaining gap is not learnable from the current data panel and the honest product path is a risk-policy overlay around the best learned generator.

## Proposed 599 Audit

Use the 441-window same-frame 510a samples as the failure map.

Targets:

- per-window coverage floor failure;
- regime layer2 hard failures;
- persistent undercoverage count;
- median-direction bias;
- large level-location error by cell/horizon.

Feature sets:

- IV-only history summaries;
- joint factor history summaries;
- IV plus factors.

Method:

- simple out-of-sample ridge/logistic models only;
- time-ordered split;
- compare lift over IV-only features;
- no neural model, no new generator, no policy overlay.

This is the most principled next step because it distinguishes an architecture problem from a data/information problem.
