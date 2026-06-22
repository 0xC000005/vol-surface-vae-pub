# Track B directional/width gate -- 2026-06-17

- Trained checkpoint: `models/backfill/generator_conditioning_probe_b1/best_model.pt`
- Frozen 734a base: `models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt`
- Held-out val windows with narrative: 115 (bank-aligned 4010); embed_mode=openai
- **OVERALL PASS: False**

## Level 1 -- knob activated
- mean context norm (real): 5.651 (bar > 0.001)
- max context norm (present=False): 0 (bar <= 1e-08)
- PASS: True

## Level 2 -- propagated (direction) + width axis
- direction-match (grounding emphasis): {'real': 0.5953623188405799, 'shuffled': 0.5938164251207728, 'zero': 0.6015458937198065}
- direction-match (all-39 robustness): {'real': 0.6222659823096939, 'shuffled': 0.6170277568216783, 'zero': 0.6376498947321007}
- **real - shuffled (HEADLINE): 0.0015458937198070677** (bar > 0.05)
- real - zero: -0.006183574879226605 (bar > 0.05)
- |shuffled - zero|: 0.007729468599033673 (bar <= 0.05)
- width spearman vs intensity: -0.3407875009863489
- PASS: False

## Level 3 -- moved + monotone + fidelity
- alpha-ladder response (context gain): {'0.5': 0.03660331575665623, '1.0': 0.08221900309436023, '2.0': 0.1891539502888918}
- monotone in context gain: True
- CRPS gap vs 734a: 0.02765 | Energy gap: 0.04063 (guardrail +/-0.015)
- fidelity pass: False | PASS: False

> Gate headlines on real-minus-shuffled direction-match DELTA (emphasis bias cancels). real~shuffled~zero => no faithful direction steering (Gap-2 structural on the direction axis); real>>shuffled~zero => narrative steers direction.
> NOTE: alpha scales the adapter final-Linear (context) output, NOT the input embedding (LayerNorm makes input scaling a no-op); reported as monotone in injected-context gain.
