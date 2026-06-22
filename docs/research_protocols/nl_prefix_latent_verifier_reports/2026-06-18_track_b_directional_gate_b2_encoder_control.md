# Track B directional/width gate -- 2026-06-18

- Trained checkpoint: `models/backfill/generator_conditioning_probe_b2_encoder_control/best_model.pt`
- Frozen 734a base: `models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt`
- Held-out val windows with narrative: 115 (bank-aligned 4010); embed_mode=openai
- **OVERALL PASS: False**

## Level 1 -- knob activated
- mean context norm (real): 2.095 (bar > 0.001)
- max context norm (present=False): 0 (bar <= 1e-08)
- PASS: True

## Level 2 -- propagated (direction) + width axis
- direction-match (grounding emphasis): {'real': 0.5925603864734298, 'shuffled': 0.5935265700483092, 'zero': 0.5989371980676328}
- direction-match (all-39 robustness): {'real': 0.6268106278864456, 'shuffled': 0.6255906651853975, 'zero': 0.6303272615932126}
- **real - shuffled (HEADLINE): -0.0009661835748794312** (bar > 0.05)
- real - zero: -0.006376811594203002 (bar > 0.05)
- |shuffled - zero|: 0.005410628019323571 (bar <= 0.05)
- width spearman vs intensity: -0.2378284541939557
- PASS: False

## Level 3 -- moved + monotone + fidelity
- alpha-ladder response (context gain): {'0.5': 0.02259754140395671, '1.0': 0.04034485248848796, '2.0': 0.058357668574899435}
- monotone in context gain: True
- CRPS gap vs 734a: 0.001802 | Energy gap: 0.008501 (guardrail +/-0.015)
- fidelity pass: True | PASS: True

> Gate headlines on real-minus-shuffled direction-match DELTA (emphasis bias cancels). real~shuffled~zero => no faithful direction steering (Gap-2 structural on the direction axis); real>>shuffled~zero => narrative steers direction.
> NOTE: alpha scales the adapter final-Linear (context) output, NOT the input embedding (LayerNorm makes input scaling a no-op); reported as monotone in injected-context gain.
