# Method Intake — T7: Narrative-Conditioned Analogue Reweighter

**Date:** 2026-06-16 · **Thread:** NL prefix-latent narrative · **Status:** DESIGN (pre-implementation)
**Family:** `support_grounded_latent_scenario_generation` (conditioning-interface upgrade)

## Method story (why this, why now)
The NL pipeline conditions a FROZEN SNI generator on a weighted mixture of retrieved historical
analogue regimes: narrative → retrieve top-k analogues → **weight (today: fixed `top3/90`
nearest-similar)** → blend into prefix latent → frozen rollout → 30-day distribution. The
2026-06-15 Codex PARTIAL verdict + the T4 honest baseline (both old retrievers fail the fit gate on
clean multi-seed data, reproducing 992b's information ceiling) establish that **leverage is in the
conditioning interface (the weighting), not retrieval recall**, and that **trained text→latent heads
re-hit the ceiling**. Promotion target is **conditionality** (narrative distinguishability), not
start-only CRPS (retired 2026-06-11). T7 makes the weighting **narrative-conditioned** by a
structured, training-free reweight — sidestepping the ceiling by construction.

## Goal / success criteria
- Increase narrative **distinguishability** (framework-v1 conditionality gate) vs the fixed `top3/90`,
  holding the start fixed, with **no fidelity-floor regression**.
- **Expose top-k analogue weights** (product requirement; never a hidden single prefix).
- **Independent Codex verifier must AGREE** before any paper/demo default change.
- Bitter-Lesson-clean for this product thread: a single global `β` scalar (no per-factor data-derived
  constants); the only learned input is the existing grounding sidecar.

## Architecture (well-bounded units)
1. **Narrative emphasis extractor** → `e = {factor: (direction, salience∈[0,1])}`.
   - Primary: extend the grounding sidecar to emit a structured `factor_emphasis` field (the LLM
     already interprets the narrative's factor implications).
   - Fallback: deterministic parser over the grounding's existing `current_market_state_implications`
     prose → `e`. (Owner decision 2026-06-16: build **both** — structured emit + parse fallback.)
   - Interface: `narrative_emphasis(grounding_output) -> e`. Unit-testable in isolation.
2. **Analogue factor-profile** → `a_i = {factor: (direction, magnitude)}` per retrieved analogue, via
   the pipeline's existing `_direction_and_magnitude` on the window's 30-day deltas. Reuse (adapter only).
3. **Reweighter** (pure function) → `w_i ∝ base_score_i · exp(β · match(e, a_i))`, renormalized to the
   same `top3/90` mass. `match(e,a_i) = Σ_f salience_f · agree(e_f.dir, a_i_f.dir)`, `agree ∈ {+1,−1,0}`.
   **β=0 ⇒ identity (recovers today's default exactly).** Interface:
   `reweight(top_k_with_scores, e, beta) -> weighted_top_k`.
4. **β selector** → sweep `β` on a HELD-OUT narrative set; pick the β maximizing the framework-v1
   conditionality metric subject to no fidelity regression. One global scalar. `select_beta(...) -> (beta, report)`.
5. **(Conditional) learned correction (option 3)** → a small learned residual on salience/match;
   implemented **only if** the core shows conditionality lift with leftover headroom; gated by the
   held-out check + Codex verifier. **Out of scope for the first build (YAGNI).**

## Data flow
narrative → grounding (emit ∥ parse) → `e` → [frozen retrieval top-k + analogue profiles] →
`reweight(e, β)` → weighted top-k (exposed) → frozen SNI rollout → 30-day distribution → eval.

## Error handling / safe degradation
- Missing factor in `e` or `a_i` → contributes 0 to `match` (treated flat).
- Structured emit fails → prose-parse fallback; parse fails → `e` empty → `match=0` → reweighter is
  identity → **recovers `top3/90` (never worse than default).**
- β selector finds no β with lift → `β=0` → T7 is a no-op.
- Operates on the already-retrieved **causal-clean (#48)** pool; introduces **no new leakage surface.**

## Testing
- Unit: emphasis emit + parse fallback on sample groundings; reweighter (β=0 identity; β>0 upweights
  direction-matching analogues; renormalization to `top3/90` mass; agreement signs).
- Integration: a few narratives end-to-end — different narratives → different weights → distinguishable
  distributions; weights exposed.
- Gate (see below).

## Backtest gate (promotion)
- Framework-v1 conditionality gate: distinguishability ↑ vs fixed `top3/90` across the narrative deck,
  above repeat/bootstrap/start-only controls.
- Matched-episode backtest on the **causal-clean** harness (#48), min query→support gap ≥30.
- Fidelity floor intact (scenarios individually realistic; no CRPS/coverage collapse vs start-only).
- β chosen on held-out; **independent Codex verifier AGREE** before any default change.

## Kill condition
- If the held-out β-sweep shows **no conditionality lift at any β without fidelity regression**, T7 is
  declared a no-op (β=0) and the bottleneck is attributed to **pool composition** (66% zero-overlap) —
  which becomes the next move. Do not escalate to option 3 in that case.

## Incumbent / non-regression
- Frozen generator (734a) + frozen retrieval; only mixture weights change.
- Start-only + `top3/90` remain the defaults until T7 passes the gate + verifier AGREE.

## Out of scope (YAGNI)
- Pool-composition fix (separate follow-on; the reweighter result informs whether it's the bottleneck).
- The learned correction (option 3) — conditional only.
- Any generator/retriever retraining (both frozen).
