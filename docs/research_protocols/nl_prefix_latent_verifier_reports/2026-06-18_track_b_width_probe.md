# Track B B-WIDTH probe (narrative-driven severity dial) — 2026-06-18

**Question.** Can the narrative faithfully dial the *width/severity* of the conditioned ensemble
on the frozen 734a generator — i.e. does fan width track grounded narrative-INTENSITY, vs merely
responding to narrative PRESENCE? Direction is a closed KILL (B1 + B2; 5x ceiling); this probe is
purely WIDTH, the verified "crack" (conditioning DOES widen the fan).

**Frozen backbone, adapter-only, separate checkpoint. 734a never modified** (read-only; sha256
`35b3c416c89d0d928b2943406e64e0b687a0ded91c6fb379c8b78b5fecdfb217`, unchanged before/after).

## Mechanism note (load-bearing)
734a has `conditional_base_noise_scale=False` => `base_noise_log_scale is None` => there is NO
differentiable width knob. Width is produced ONLY by the frozen velocity ODE acting on i.i.d.
base noise, with `narrative_context` entering via `memory_states` (the same path that carries
734a's validated width-conditioning, turb/calm 1.229). So the width objective is trained through
a gradient-enabled mini-rollout (mirrors `sample_batched`, per-step gradient-checkpointed for the
8GB card). No base_noise_scale head was added (different mechanism + non-adapter param + would
break the present=False->734a-identity invariant).

## Measurement discipline (advisor-tightened)
- Correlate intensity vs **width_DELTA** = width(present=True) - width(present=False), CRN-shared
  base noise per window — NOT raw conditioned width (regime-confounded: frozen 734a already widens
  from high-vol history alone, turb/calm 1.229).
- Grounded intensity = prose-parse `magnitude(small .33/med .67/large 1.0) * |salience|` aggregated
  over named factors, **NO OpenAI**, NO realized-future peek (`nl_track_b_grounded_intensity.py`).
- Shuffled control = >=1000-permutation null band (single-shuffle std ~0.09 at N~115 > the margin).
- **Severity-dial (intercept) check** on ABSOLUTE intensity: a faithful dial needs positive OLS
  slope AND near-zero intercept (low-intensity present narratives must NOT uniformly inflate the
  fan). A rank-only spearman/real-minus-shuffled CANNOT catch uniform presence-inflation (a
  constant +offset is rank-invisible) — this guards the task's own KILL pattern from passing.

## STEP 0 — pilot (no training), 115 held-out windows
| ckpt | width cond | base | mean width_delta | spearman(intensity, width_delta) real / shuffled | null p975_abs / exceeds tail |
|---|---|---|---|---|---|
| b2_velocity (best-fidelity, direction-trained) | 0.557 | 0.556 | +0.001 | -0.034 / +0.061 | 0.215 / no |
| b1 (off-manifold direction adapter) | 0.541 | 0.364 | +0.176 | -0.187 / -0.169 | 0.203 / no |

Read: untrained adapters give NO width-intensity signal. b2 doesn't move width (uninformative
null); b1 DOES move width but real≈shuffled => responds to PRESENCE not INTENSITY. GO for STEP 1
(train a width-intensity objective), with the risk posture that b1's widening came with a
fidelity break (B1: CRPS +0.0277).

## STEP 1 — width-objective adapter (separate ckpt `generator_conditioning_probe_bwidth`)
Loss = velocity_matching (B1's `training_loss`, path-fidelity anchor) + lambda_width*(1-corr(
grounded_intensity, differentiable width_delta)) [+ lambda_level intercept anchor]. Adapter-only
(855,680 trainable), frozen probe `feature_proj.weight` max|diff|=0.0.

- Run A (corr-only, flow_steps=2): in-train tercile stable ~low 0.25 / high 0.28 across epochs
  (uniform presence-inflation). **NOTE: flow2-train / flow16-gate is a metric mismatch — its
  gate fidelity number is confounded and not reported as clean.**
- Run B (MATCHED best-shot: flow_steps=16, lambda_level=2.0, K=8) — the channel's best case with
  matched integration + an explicit intercept objective. Pre-registered: verdict stands either way.

## STEP 2 — WIDTH GATE (held-out, pre-registered) — MATCHED run B
PASS = spearman_real>0.30 AND (real-shuffled)>0.15 AND severity-dial (slope>0 & low/high frac<=0.5)
AND fidelity (CRPS & Energy vs 734a within +/-0.015). KILL otherwise.

<!-- FILL: matched-run B gate numbers + VERDICT -->

## Verdict & decision
<!-- FILL -->

## Artifacts
- `experiments/backfill/block_ar/nl_track_b_grounded_intensity.py` (intensity scalar)
- `experiments/backfill/block_ar/nl_track_b_width_pilot.py` + `.../track_b_width_pilot/track_b_width_pilot.json`
- `experiments/backfill/block_ar/train_track_b_width_conditioning.py` (differentiable width trainer)
- `experiments/backfill/block_ar/nl_track_b_width_gate.py` + `.../track_b_width_gate/track_b_width_gate.json`
- `models/backfill/generator_conditioning_probe_bwidth/` (separate checkpoint)
- tests: `test_code/test_track_b_grounded_intensity.py`, `test_code/test_track_b_width_rollout.py`
