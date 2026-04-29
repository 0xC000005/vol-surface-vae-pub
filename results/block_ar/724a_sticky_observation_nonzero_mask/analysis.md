# 724a Sticky-Observation Nonzero-Mask Diagnostic

## Context

723a reframed the OAS failure as a mixed observation object: exact no-update prints plus continuous nonzero economic moves. 724a tests the minimal clean repair without changing the core AR flow mechanism: exact-zero future targets on sticky channels are masked out of the continuous flow-matching loss, then an empirical no-update atom is applied at readout for the diagnostic.

## Result

The diagnostic does not satisfy the single-framework gate.

| scope | variant | factor KS pass | mean KS | AAA KS | BBB KS | factor corr abs ratio | IV-factor abs ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor | identity | 9/13 | 0.168 | 0.439 | 0.519 | 0.728 | n/a |
| anchor | global atom | 9/13 | 0.138 | 0.232 | 0.336 | 0.637 | n/a |
| anchor | history-bin atom | 9/13 | 0.138 | 0.231 | 0.335 | 0.634 | n/a |
| joint | identity | 10/13 | 0.132 | 0.265 | 0.349 | 0.508 | 0.641 |
| joint | global atom | 11/13 | 0.116 | 0.155 | 0.254 | 0.461 | 0.604 |
| joint | history-bin atom | 11/13 | 0.116 | 0.154 | 0.255 | 0.461 | 0.603 |

## Mechanism Read

The atom part works mechanically: generated no-update rates move close to validation no-update rates. The remaining failure is the nonzero OAS movement law. In anchor-only, the nonzero OAS tails are still too wide after atom readout: AAA generated q99 is about 0.15 versus GT 0.04, and BBB generated q99 is about 0.13 versus GT 0.05. In joint38, IV context partially constrains the OAS nonzero distribution, so AAA passes and BBB gets closer, but BBB still fails and dependency ratios weaken.

## Decision

Reject masked-zero continuous loss as the next production framework. The clean next step is not another atom probability tweak. The bottleneck is scale allocation for rare nonzero updates on mixed-frequency channels, especially when anchor-only lacks IV context. A principled follow-up should diagnose or repair nonzero-update scale conditioning while keeping one shared framework.

## Artifacts

- `models/backfill/724a_anchor_sticky_obs_nonzero_mask_e8_w2048_s7242/best_model.pt`
- `models/backfill/724a_joint38_sticky_obs_nonzero_mask_e8_w2048_s7243/best_model.pt`
- `results/block_ar/724a_sticky_observation_nonzero_mask/atom_gate_analysis.json`
- `results/block_ar/724a_sticky_observation_nonzero_mask/atom_gate_analysis.md`
