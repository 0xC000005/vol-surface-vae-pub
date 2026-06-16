# Independent Verifier Report: Retrieval-Direction Probes (Codex)

**Date:** 2026-06-15
**Verifier:** Codex CLI (gpt-5.5, reasoning xhigh, read-only sandbox) — INDEPENDENT verification.
**Artifact (full reasoning):** `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_retrieval_probe_codex_verification_20260615/codex_verdict.log`
**Subject:** the T6 cheap-probe conclusion used to gate the T4/T7 retrieval-direction pivot.

## Verdict: PARTIAL (operationally lean-AGREE)

Codex agrees with the **operational** pivot but does NOT accept the strong claim "contrastive
retriever training is low-leverage" as proven by these probes.

### Methodology bug found (load-bearing) — Probe 2
`nl_retrieval_probe_2_procrustes_linear_bridge.py` (~line 230) treats
`heldout_examples[].window_index` as a window-id remapped through `window_metadata`, but it is a
**pool position**. Effect: 30 of 66 held-out windows silently dropped; the remaining 36 scored
against wrong ground-truth pool positions. Codex's read-only corrected recompute:

| metric | reported (buggy) | corrected |
|---|---|---|
| n held-out | 36 | 66 |
| Procrustes rank-median (pool 380) | 162 | 181 |
| PCA-Procrustes rank-median | 143 | 178.5 (recall@10 2/66) |
| condition-vector rank-median | 154 | 172 (recall@10 1/66) |
| trained 906b adapter rank-median | 254 (**below chance**) | ~175 (**near chance, NOT below**) |

So the "contrastive trained below random" claim is **not supported**; corrected it is near/slightly-
above chance. `recall@10 = 0/36` was never below-chance evidence (random expects ~0.95 hits;
P(0) ≈ 38%).

### Confirmed
- Probe 1: DBSN "after" numbers are a degenerate Sinkhorn artifact (must not inform any decision);
  pre-DBSN hubness is mild, though "not significant" is mildly overstated without a random
  finite-sample baseline.
- Probe 3: the hard number is the **0.0264** oracle-leakage within-pool CRPS bound (CI excludes 0);
  the "0.0042 realistic" is a heuristic, and it is top-50 within-pool headroom, not a full-bank
  pool-composition ceiling.
- 992b: exact-window retrieval in the 4010-row SNI memory space is intrinsically hard (adjacent
  offset-5 recall@10 0.1117, rank-median 208) — qualitative ceiling only; do not cross-compare to
  Probe 2's 380-pool numbers.

## Decision adopted (2026-06-15)
- **Deprioritize** the old exact-window bridge objective and **blind** contrastive scaling.
- **Prioritize**: conditioning interface (top-k tilt/weighting) + memory-target definition + pool
  composition + the clean-bank honest baseline restamp (with corrected indexing).
- **Do NOT kill contrastive as a class.** The promotion target is **conditionality** (distinguishable
  narrative-conditioned distributions preserving the fidelity floor), not beating start-only CRPS
  (retired). A **conditionality-targeted** contrastive retriever (locality-soft / posterior /
  set-level targets, production `text-embedding-3-large`, NV-Retriever false-negative filtering)
  remains the decisive experiment.

### What would move the verdict (Codex)
- → **AGREE** if the clean 14×14 restamp (corrected indexing + 3-large) shows no fixed-start
  conditionality lift and no support-posterior improvement beyond interface/pool changes.
- → **DISAGREE** if a corrected NV-Retriever run (false-neg filter + locality-soft/posterior target)
  improves fixed-start narrative distinguishability without support-honesty / fidelity regressions.

## Follow-ups opened
- Fix Probe 2 indexing bug + restamp the corrected numbers; audit the matched-eval harness for the
  same `window_index`-as-id confusion (relates to the 14×14 matched-eval causal-gap leakage task).

---

## Addendum (2026-06-15): clean 14×14 bank + manifest COMPLETE

- **Bank**: `stride5_fourteen_view_bank_clean_20260615/` — **802/802 pass**, combined report
  `status=pass, fail_count=0`. Built from clean 972b (corrected joint39 map), sharded 4× then
  cleanup passes.
- **Directional gate**: all 802 targets' `mechanical_summary` is **verbatim-identical** to the
  index-gate-verified 972b support cards (0 mismatches); AAA_OAS claims 71/71 credit-spread-scale,
  0 nikkei-scale. Bank inherits 972b cleanliness by construction.
- **Manifest**: `stride5_self_supervised_training_manifest_clean_20260615/` — 11,228 pairs / 802
  targets, `validation_error_count=0`; new `text_hash_digest=bf328fd8…` (was contaminated
  `c1586f2f…`) → embeddings self-heal on next run.
- **LESSON (future bank rebuilds under the corrected map):** `--negative-candidate-count` has a
  per-window viable band. 200 (old default) STARVES the unique-per-view negative assignment for
  edge windows (ValueError "no usable negative candidate"); ≥~560 OVERFLOWS codex context. Working
  band ≈ 280–480 depending on the window. 793/802 built at 200; the remaining 9 needed 320 (5) and
  480 (4). Recommend defaulting to ~360 with per-window escalation on assignment-starve.

---

## Addendum (2026-06-15): T4 honest baseline restamp on CLEAN manifest — both old methods FAIL fit gate

Multi-seed (0/1/2), `--method both`, mirror-992a protocol (openai 3-large, 5000 steps, val split
610:730,1490:1610,2370:2490,3250:3370,3915:4010, purge-gap 30, holdout view families, patience 8),
clean manifest (hash bf328fd8), embeddings self-healed.
- **projected_memory** heldout_true_memory_rank_median: **980.5 / 1021 / 1009.5**; recall@10:
  **0.0067 / 0.0080 / 0.0053** → FAILS fit gate (<=400 / >=0.10) across ALL seeds. Matches the
  contaminated 992a (969 / 0.0093).
- **text_space** heldout same-label recall@1 ~0.20-0.22; top-10 retrieved temporal distance
  ~946-998 days (near-random for window identity) → no precise window retrieval either.
CONCLUSION: the clean-data restamp REPRODUCES the failure. The contamination was NOT masking a working
retriever; the exact-window / contrastive objective is information-limited (992b confirmed on clean
data). Honest closure: do-not-promote stands. Validates the pivot to T7 (conditioning interface).
Artifacts: stride5_14x14_retrieval_training_clean_restamp_20260615/both_seed{0,1,2}/.
