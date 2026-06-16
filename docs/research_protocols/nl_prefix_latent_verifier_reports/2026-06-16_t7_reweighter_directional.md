# T7 Narrative Reweighter — Directional-Steering Test (DECISIVE)

**Date:** 2026-06-16 · **Author:** Claude (Opus) · **Status:** T7 = live + fidelity-neutral, but
**does NOT steer** narratively. NOT promoted. Third convergent ceiling signal.

## What T7 is
Training-free reweighter: `score += beta * match(narrative_emphasis, analogue_factor_profile)`
before `_apply_top3_90`, over the frozen SNI generator. `beta=0` exact no-op. Module + 13 unit
tests (`nl_narrative_reweighter.py`, `test_t7_nl_narrative_reweighter.py`). Built on a #48-clean
66-query deck (regenerated: `..._embedding_grounded_top3_90_66q_clean48/`, 0/528 gap violations).

## Tests run (clean deck, frozen 734a, field_weight, 48 samples, CRN-by-query)
1. **Free gate** (`nl_t7_beta_sensitivity_gate.py`): beta moves the top3/90 selection on 52% of
   queries (not a structural no-op). Effect saturates by beta≈0.25.
2. **Rollout responsiveness + fidelity** (`nl_t7_select_beta.py`): among movers, terminal shift
   ≈3.9× the reseed-noise floor; fidelity preserved (CRPS median 0.515 unchanged, mean +0.004
   negligible; coverage 0.714→0.701, within tol). **But responsiveness is near-tautological** —
   reselecting an analogue trivially changes the rollout. It establishes "knob live + fidelity-safe",
   not steering.
3. **Directional hit-rate (DECISIVE)** (`nl_t7_directional_hitrate.py`): do scenarios move in the
   narrative's claimed factor directions, and does beta>0 beat beta=0 / chance?

| metric | value |
|---|---|
| (a) selected-analogue HISTORY hit-rate vs claims | **0.849** (retrieval direction-gate works) |
| (b) generated-SCENARIO terminal hit-rate, beta=0 | 0.424 (≤ 0.5 chance) |
| (b) generated-SCENARIO terminal hit-rate, beta=0.25 | 0.428 (Δ +0.004 all; **+0.000 on movers**) |

## Verdict
**T7 reselects narrative-direction-matched analogues (history 0.849) but the frozen SNI does NOT
carry that direction into the forward scenario (0.428 ≈ chance); beta>0 adds zero directional
accuracy over beta=0.** So T7 moves scenarios and preserves fidelity, but achieves **no narrative
directional steering**. `selected_beta = 0` for the steering goal. The earlier "TILT HELPS / β=0.25"
label measured responsiveness only and is corrected here.

## Mechanism + convergence
The frozen generator's forward terminal direction is decorrelated from the seeding analogue's
(matched) historical direction. This is the **same root cause** as the two prior ceilings:
- **risk_context probe** (2026-06-16): generator insensitive to oracle conditioning injection.
- **T4 retriever**: at its fit-gate ceiling on clean multi-seed data.
- **T7 (here)**: retrieval matches direction; generator washes it out forward.
→ Narrative→scenario *directional* steering is not achievable by conditioning a frozen SNI from the
outside; it would require a generator actually sensitive to the conditioning channel.

## Caveats
- Claims are CURRENT-state directions vs a 30-day FORWARD terminal; mean-reversion partly explains
  (b)≤0.5. But the decisive within-comparison (beta=0.25 vs beta=0, Δ≈0 on movers) is robust to this.
- Matched-episode deck. The fix-pool-vary-emphasis cross-narrative test (T7.8) would corroborate;
  given Δ≈0 here, it is expected to confirm divergence-without-steering. Layer the same directional
  check on it (do not score raw cross-narrative divergence — that is also tautological).

## Artifacts
- `nl_t7_select_beta.py`, `nl_t7_beta_sensitivity_gate.py`, `nl_t7_directional_hitrate.py`
- `nl_scenario_demo_outputs/nl_t7_beta_sweep_20260616/beta_sweep_report.json`
- Clean deck: `..._embedding_grounded_top3_90_66q_clean48/embedding_grounded_bridge_report.json`
