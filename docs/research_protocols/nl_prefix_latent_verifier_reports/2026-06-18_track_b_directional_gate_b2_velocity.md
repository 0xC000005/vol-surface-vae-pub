# Track B B2 directional gate — VELOCITY-READOUT arm — 2026-06-18

> CORRECTED 2026-06-18: an earlier copy of this file mistakenly held the ENCODER-CONTROL arm's
> numbers (fixed-path write collision during same-day runs; caught by independent Codex review).
> The numbers below are now the VELOCITY-readout arm, transcribed from the authoritative durable
> JSON: `experiments/backfill/block_ar/nl_scenario_demo_outputs/track_b_directional_gate_b2_velocity/track_b_directional_gate.json`.
> (The encoder-control arm is in `2026-06-18_track_b_directional_gate_b2_encoder_control.md` /
> `..._b2_encoder_control/track_b_directional_gate.json`.)

- Trained checkpoint: `models/backfill/generator_conditioning_probe_b2/best_model.pt` (velocity readout unfrozen + narrative adapter)
- Frozen 734a base: `models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/best_model.pt` (byte-identical, read-only)
- **OVERALL_PASS: False → VERDICT: KILL**

## Level 1 — knob activated
- mean context norm (real): **0.623**; present=False norm = 0 → PASS: True

## Level 2 — propagated (direction), held-out grounding-emphasis factors
- real **0.5891** / shuffled **0.5917** / zero **0.5910**
- **real − shuffled (HEADLINE): −0.0026** (BREAK bar > +0.05) → no narrative-direction sensitivity
- real − zero: −0.0019
- PASS: **False**

## Level 3 — moved + fidelity (vs frozen 734a)
- CRPS conditioned **0.5938** vs base **0.6095** → gap **−0.0157** (sign = fidelity slightly IMPROVED)
- Energy gap **−0.0091**
- fidelity_pass: False — *only* because |CRPS gap| 0.0157 marginally exceeds the ±0.015 guardrail band; the sign is an improvement, not degradation.

## Interpretation (the decisive B2 fact)
Unfreezing the narrative-CONSUMING velocity readout gave it real capacity that **demonstrably improved aggregate fit** (CRPS −0.0157) — yet it added **zero** narrative-direction sensitivity (real ≈ shuffled ≈ zero; real−shuffled −0.0026 ≪ the +0.05 bar). Whatever the readout learned is **narrative-independent**. ⇒ the directional ceiling is **not merely a frozen-backbone transmission limit** under these gates; the learnable narrative-direction signal is **absent or too weak in this setup** (a data-signal limit). 5th independent confirmation (T7 → risk_context oracle → B1-L2 → B1-L3 → B2). CAVEAT: the CRPS gain cannot be decomposed into narrative-independent readout fine-tune vs width; not chased here.
