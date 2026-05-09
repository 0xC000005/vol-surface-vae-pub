# World Model HEAD040: Probe Baseline Comparison

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

HEAD039 showed that the fused-context fixed delta-PCA model passes a frozen-probe audit. This analysis asks how that result compares with the older raw-delta context-probe family, and whether the next step should be decoder work, a Barlow/VICReg-style loss, or another Part 1 JEPA validation.

## Literature Status

No new objective is introduced here.

- The active direction remains JEPA-style because it predicts future target representations from past context representations, matching the core I-JEPA/V-JEPA contract.
- The fixed delta-PCA target contract is `supported_adjacent`, not canonical I-JEPA: it is a stabilized target-space diagnostic after learned EMA target collapse.
- Barlow Twins is not the active loss. Its redundancy-reduction idea is adjacent motivation for variance/covariance diagnostics. If explicit variance/covariance regularization is revisited, VJ-VCR or VICReg is the closer framing than claiming a Barlow Twins JEPA.

Primary references checked during this loop:

- I-JEPA: `https://arxiv.org/abs/2301.08243`
- V-JEPA: `https://arxiv.org/abs/2404.08471`
- TS-JEPA: `https://arxiv.org/abs/2406.04853`
- Barlow Twins: `https://arxiv.org/abs/2103.03230`
- VICReg: `https://arxiv.org/abs/2105.04906`
- VJ-VCR: `https://arxiv.org/abs/2412.10925`

## Raw-Delta Probe Comparison

These rows compare frozen context ridge probes to raw horizon deltas. HEAD039 uses the fused fixed-PCA model's frozen context and the raw-delta probe from the HEAD039 audit.

| label | ctx rank | offdiag | ctx var | ridge delta MSE | ridge MRR | top5 | top10 | pred rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HEAD010 unregularized | 3.688405 | 0.484346 | 0.014709 | 0.017107 | 0.111537 | 0.147656 | 0.228906 | 3.594345 |
| HEAD012 corr 0.005 | 5.732591 | 0.332259 | 0.009405 | 0.018033 | 0.106710 | 0.136719 | 0.225000 | 3.767882 |
| HEAD013 corr 0.002 | 5.284253 | 0.343461 | 0.011004 | 0.016903 | 0.107289 | 0.139844 | 0.220312 | 3.642267 |
| HEAD015 headsharp | 4.832314 | 0.385486 | 0.009028 | 0.016111 | 0.111276 | 0.146875 | 0.222656 | 3.584262 |
| HEAD017 mildhead | 5.323299 | 0.352284 | 0.026349 | 0.017022 | 0.110486 | 0.137500 | 0.233594 | 3.501096 |
| HEAD019 top5 | 4.748494 | 0.368346 | 0.012012 | 0.017327 | 0.104012 | 0.141406 | 0.221875 | 3.564088 |
| HEAD020 softhead | 4.495739 | 0.373337 | 0.008195 | 0.017281 | 0.098365 | 0.128906 | 0.202344 | 3.565533 |
| HEAD023 softneighborhood | 5.060637 | 0.361862 | 0.013273 | 0.017022 | 0.111229 | 0.142187 | 0.226562 | 3.540544 |
| HEAD039 fused raw-delta probe | 6.648142 | 0.323006 | 0.153847 | 0.015701 | 0.096147 | 0.128125 | 0.196875 | 3.016013 |

## Horizon Read

HEAD039 raw-delta probe improves MSE most at longer horizons but is weaker on retrieval at early and middle horizons.

| label | h1 MRR/top5 | h5 MRR/top5 | h10 MRR/top5 | h20 MRR/top5 | h30 MRR/top5 |
|---|---:|---:|---:|---:|---:|
| HEAD010 | 0.092428 / 0.121094 | 0.101102 / 0.128906 | 0.121001 / 0.156250 | 0.121566 / 0.175781 | 0.121590 / 0.156250 |
| HEAD013 | 0.083857 / 0.105469 | 0.096643 / 0.125000 | 0.118501 / 0.156250 | 0.119230 / 0.167969 | 0.118215 / 0.144531 |
| HEAD023 | 0.107113 / 0.144531 | 0.093861 / 0.109375 | 0.129368 / 0.160156 | 0.116359 / 0.132812 | 0.109442 / 0.164062 |
| HEAD039 raw delta | 0.082441 / 0.109375 | 0.085236 / 0.097656 | 0.102618 / 0.128906 | 0.105684 / 0.136719 | 0.104755 / 0.167969 |
| HEAD039 fixed PCA | 0.093230 / 0.117188 | 0.086869 / 0.125000 | 0.108226 / 0.148438 | 0.109947 / 0.144531 | 0.120544 / 0.148438 |

## Interpretation

- HEAD039 is the best of this comparison on raw-delta ridge MSE (`0.015701`) and context health (rank `6.648142`, offdiag `0.323006`).
- HEAD039 is not the best raw-delta retrieval representation: older retrieval/contrastive-trained contexts remain stronger on MRR/top5/top10.
- HEAD039's trained fixed-PCA head still has the best decoded-delta MSE seen in this branch (`0.015176`), but its fixed-PCA retrieval is not directly equivalent to raw-delta retrieval from the older context-probe audits.
- The old neighborhood/retrieval branch should not be promoted as the main direction because it previously failed forecast/decode gates and was classified as noncanonical. It remains a useful diagnostic baseline.

## Decision

Do not move to the decoder yet, and do not add a Barlow Twins or retrieval/neighborhood objective.

The next Part 1 risk is robustness of the current fused fixed-target representation:

1. Verify HEAD039-style frozen probes on the supporting HEAD036 seed.
2. If the second seed agrees, run the same audit on the held-out test split or add a reusable train/val/test probe wrapper.
3. Only after cross-seed and split checks should the workflow consider decoder conditioning.

## Artifacts

- `results/world/part1_fused_context_probe_audit_head039.json`
- Prior comparison artifacts: `results/world/part1_context_probe_audit_head010.json`, `results/world/part1_context_probe_audit_head013.json`, `results/world/part1_context_probe_audit_head023.json`
