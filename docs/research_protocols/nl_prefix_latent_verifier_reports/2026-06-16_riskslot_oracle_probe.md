# Probe Report: risk_context Oracle-Injection (novel direction-slot paradigm gate)

**Date:** 2026-06-16 · **Author:** Claude (Opus) · **Status:** probe complete, WEAK/leaning NO-GO (caveated)

## Goal
Gate the novel "steer the frozen SNI generator's `risk_context` slot with a narrative-derived
direction" paradigm (from workflow wk4vwo0qe). Test whether the slot, once instantiated + co-trained,
steers the generated joint39 distribution past the 994a-style null band under ORACLE injection.

## Method (built by Claude, not delegated)
- Patched `load_model` (`diffusion/block_ar/generic_state_aware_normalized_innovation_flow_matching.py`)
  with an optional backward-compatible `cfg_overrides` (e.g. `risk_state_dim>0`); patched
  `train_666a` with `--risk_state_dim`. Verified no-override path unchanged.
- **Stage B (instantiate):** documented Stage-2 `train_666a` finetune from the protected
  `734a_joint39` checkpoint (NEW checkpoint `…/734a_riskslot_probe_20260616/`, 734a untouched),
  `--risk_state_dim 4` (matches the built-in 4-d `_future_risk_targets` activity target),
  `--risk_state_weight 1.0 --risk_state_rank_weight 0.5 --conditional_base_noise_scale`,
  documented objective (energy 0.2 / channel-level 0.05 / level coord), 2 epochs. Trained cleanly
  (risk_state_head norm 13.1, risk_context_proj norm 2.0; sample-smoke finite, IV in range).
- **Stage C (verdict):** direct ±-direction `risk_context` injection swept over magnitude vs a
  repeat-sample null band, per-channel separation.

## Result
| k (risk_context norm) | sep / null band | channels > 1× |
|---|---|---|
| 1 | 0.03× | 0/39 |
| 6 | 0.15× | 0/39 |
| 12 | 0.39× | 1/39 |
| 24 | 1.07× | 25/39 |

The generator is highly INSENSITIVE to `risk_context`: separation reaches the null band only at
**k≈24 (~300× the trained slot's natural output norm of 0.08, off-manifold)**; at realistic
magnitudes it is 3–15% of the noise floor. Non-regression clean (probe finite, drift 0.035 vs 734a).

## Caveats (do not over-claim a kill)
1. Tests the built-in 4-d ACTIVITY slot, not per-factor DIRECTION (the synth's actual vision) —
   directional steering would need a custom slot/target, untested.
2. Short 2-epoch finetune may under-activate the zero-init `risk_context_proj` (output norm 0.08).
3. Random injection direction, not optimized.
Note: an initial Stage-C run wrongly reported exactly-zero separation — that was a test artifact
(constant 4-d input degenerates under the proj's input LayerNorm), corrected here.

## Verdict + recommendation
WEAK / leaning NO-GO: limited conditioning leverage via this slot. This is the SECOND independent
ceiling signal (T4 showed the retriever near its fit-gate ceiling). Recommendation: ship **T7**
(cheap, deployable, β=0-safe conditionality floor); pursue the full directional-slot paradigm only if
its upside justifies a bigger build (custom directional slot + stronger activation).

## Artifacts
- Probe checkpoint: `…/734a_riskslot_probe_20260616/best_model.pt`
- Scripts: `nl_riskslot_probe_stageA_smoke.py`, `nl_riskslot_probe_stageC_eval.py`
- Patches (uncommitted): `load_model` cfg_overrides + `train_666a --risk_state_dim`
