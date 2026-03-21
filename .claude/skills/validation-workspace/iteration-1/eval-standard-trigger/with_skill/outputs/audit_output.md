# Validation Audit — Experiments from 2026-03-19 to 2026-03-21

**Date**: 2026-03-21
**Lookback window**: 2026-03-19 to 2026-03-21
**Scope**: ~55 research log entries (lines 28683-32334), covering experiments 120b_v2 through RC4 (120b_v5_is_fix)

---

## Phase 1: Scan — Inventory of Recent Work

### Experiments with Training (14)

| ID | Description | Date | Model Path |
|----|-------------|------|-----------|
| 120b_v2 | Noise-Free MLP + Ortho Reg | 03-19 | models/backfill/afcrps_120b_v2 |
| 115a_v4 | 4-Factor + Student-t(df=8) One-Shot | 03-19 | (uses 115a base) |
| 124a | Low-Rank Cell Spread (3 Factors) | 03-19 | models/backfill/afcrps_124a |
| 126a | Curriculum Noise | 03-19 | models/backfill/afcrps_126a |
| 128a | 3-Layer AR MLP | 03-19 | models/backfill/afcrps_128a |
| 129a | noise_dim=64 | 03-19 | models/backfill/afcrps_129a |
| 130a | H1: Log-Det Covariance Penalty | 03-20 | models/backfill/afcrps_130a |
| 132a / 132a_v2 / 132b | H3: 4-Layer CLN MLP (3 variants) | 03-20 | models/backfill/afcrps_132a, 132a_v2, 132b |
| 133a/b/c/d/e/f/f_v2/f_v3 | H4: Joint Transformer (8 variants) | 03-20 | models/backfill/afcrps_133* |
| 134a/b/c/d | Reflecting Boundaries + Bias (4 variants) | 03-20 | models/backfill/afcrps_134* |
| 120b_v5_is_fix | RC4: Fixed Interval Score | 03-21 | models/backfill/afcrps_120b_v5_is_fix |

### Inference-Only Evaluations (16)

| ID | Description | Date |
|----|-------------|------|
| E3 | 3-model multi-checkpoint ensemble | 03-19 |
| E4 | 4-model ensemble (all-time best 68.74) | 03-19 |
| E5/E5a/E5b | Hybrid (133c+99m) and 5-model ensembles | 03-19/20 |
| E6/E7/E8/E9 | Various 3-model ensembles | 03-20 |
| E_best3/E_best4 | Optimal ensemble with joint TF | 03-20 |
| 120b+Gaussian | Noise inference switch | 03-19 |
| Inference-only sweep | 10 configurations (df, fixnorm, ep40) | 03-19 |
| 133f df sweep | df=4,8,12,20 on joint transformer | 03-20 |
| 134a multi-seed | 5 seeds for stochastic variance check | 03-20 |

### Analyses / Investigations (10)

| Task | Description | Date |
|------|-------------|------|
| Investigation Depth Audit | Completeness check across 47 experiments | 03-19 |
| Round 3 deep investigations | 7 parallel analyses (E3/E4, 126a, 120b+Gauss, 120b_v2, 115a_v4, 120b_v3, 120b_v4) | 03-19 |
| 128a investigation | 3-layer MLP mechanism (different from 120a) | 03-19 |
| Post-Round-3 audit | All 50 experiments audited | 03-19 |
| Literature survey | 5 domains, 50+ papers | 03-20 |
| Meta-research | Science of ideation & AI tools | 03-20 |
| Cross-domain lit search | Rank-1 attractor in 5 domains | 03-19 |
| Round 4 follow-up analyses | 8 parallel investigations (A1-A6, B3, B5) | 03-20 |
| Test suite code review | 5 parallel agents, 9 bugs/issues | 03-20 |
| Oracle test | GT fed as predictions | 03-20 |

### Research Compass Entries (4)

| ID | Focus | Date | Hypotheses |
|----|-------|------|-----------|
| RC1 | Breaking 5/8 ceiling | 03-19 | H1 (log-det), H2 (variogram), H3 (CLN), H4 (joint TF) |
| RC2 | Per-cell calibration barrier | 03-20 | H1 (cond cell spread), H2 (higher cell_var), H3 (learned anchor) |
| RC3 | 6/8 via Suite 8 bias | 03-20 | H1 (static bias), H2 (multi-seed), H3 (cond-dep bias) |
| RC4 | Fix interval score | 03-21 | Retrain 120b only |

### Test Suite V2 Development (1 major artifact)

| Task | Description | Date |
|------|-------------|------|
| V2 test suite | 4 bug fixes, 5 methodology fixes, Suite 9 (cross-cell corr) | 03-20 |

---

## Phase 2: Audit Completeness

### Audit Table

| Exp/Task | Results Saved | Script Exists | Depth | Cross-val | Follow-up Done | Compass Tested | Metrics Verified | Gaps |
|----------|:------------:|:------------:|:-----:|:---------:|:--------------:|:--------------:|:----------------:|:----:|
| **120b_v2** | SAVED | NO | Adequate | NO | N/A | N/A | YES (66.92, 5/8) | 1 |
| **E3 ensemble** | SAVED | NO | Deep (Round 3) | YES (LOO) | DONE (E4 built on it) | N/A | YES (68.33, 5/8) | 1 |
| **E4 ensemble** | SAVED | NO | Deep (Round 3) | YES (LOO) | DONE | N/A | YES (68.74, 5/8) | 1 |
| **115a_v4** | SAVED | NO | Deep (Round 3) | YES | N/A | N/A | YES (66.25, 5/8) | 1 |
| **124a** | SAVED | NO | Adequate | NO | BUG NOTED | N/A | Not checked | 2 |
| **126a** | SAVED | NO | Deep (Round 3) | NO | N/A | N/A | YES (65.94, 5/8) | 1 |
| **120b+Gauss** | SAVED | NO | Deep (Round 3) | YES | DONE (sweep) | N/A | YES (67.6, 5/8) | 1 |
| **128a** | SAVED | NO | Deep (Round 3) | NO | N/A | N/A | YES (score verified) | 1 |
| **129a** | SAVED | NO | Adequate | NO | N/A | N/A | YES (66.08, 5/8) | 1 |
| **Inference sweep** | SAVED (14 dirs) | NO | Deep | YES (10 configs) | DONE | N/A | YES (all scores match) | 1 |
| **130a (H1)** | SAVED | PARTIAL (cmd in log) | Deep | NO | N/A | RC1-H1: CLEAN FALSIFICATION | YES (66.90, 5/8) | 1 |
| **132a (H3)** | SAVED | NO | Deep | NO | 132a_v2/132b variants | RC1-H3: CONFIRMED | YES (44.24, 3/8) | 1 |
| **132b** | SAVED | NO | Adequate | NO | N/A | RC1-H3: EXHAUSTED | YES (54.63, 4/8) | 1 |
| **133a/b** | SAVED | NO | Deep | NO | Iterated to 133c | RC1-H4 POC | YES (42.83, 3/8) | 1 |
| **133c** | SAVED | NO | Deep | YES (ensembles) | DONE (133d, E5, 133f) | RC1-H4: PASSES Suite 4 | YES (53.54, 4/8) | 1 |
| **133d** | SAVED | NO | Adequate | NO | Falsified by A1 (ep20) | N/A | YES (40.69, 3/8) | 1 |
| **133e** | SAVED | NO | Shallow | NO | Val-loss-vs-kurtosis noted | N/A | YES (40.34, 3/8) | 2 |
| **E5 hybrid** | SAVED | NO | Adequate | YES (E6 also) | N/A | N/A | YES (65.3, 5/8) | 1 |
| **E6** | SAVED | NO | Adequate | YES | N/A | N/A | YES (65.92, 5/8) | 1 |
| **133f** | SAVED | NO | Deep | YES (ep sweep, df sweep) | Multiple follow-ups | N/A | YES (63.78, 5/8) | 1 |
| **133f_v2** | SAVED | NO | Shallow | NO | N/A | N/A | Not checked | 2 |
| **133f_v3 (RC2-H2)** | SAVED | NO | Shallow | NO | N/A | RC2-H2: FALSIFIED | Not checked | 2 |
| **134a (RC2-H1)** | SAVED | NO | Adequate | YES (5 seeds) | DONE (multi-seed) | RC2-H1: MIXED | YES (64.23, 5/8) | 1 |
| **134b (reflect)** | SAVED | PARTIAL (cmd in log) | Deep | NO | DONE (134c, 134d) | RC3: INFORMED | YES (69.36, 5/8) | 1 |
| **134c (static bias)** | SAVED | NO | Deep | NO | DONE (134d) | RC3-H1: PARTIAL | YES (69.41, 5/8) | 1 |
| **134d (cond bias)** | SAVED | NO | Adequate | NO | N/A | RC3-H3: FALSIFIED | YES (54.21, 4/8) | 1 |
| **A1: 133d ep20** | SAVED | NO | Deep | NO | N/A | Hypothesis falsified | YES (42.1, 3/8) | 1 |
| **A2: 133f ep sweep** | SAVED (4 dirs) | NO | Deep | YES (4 checkpoints) | N/A | Hypothesis falsified | YES (ep14 best) | 1 |
| **A3: PC1 comparison** | NOT SAVED | NO | Deep (in log) | YES (4 models) | N/A | N/A | IN LOG ONLY | 3 |
| **A5: Failure cell map** | NOT SAVED | NO | Deep (in log) | YES (367 models) | Informs all future work | N/A | IN LOG ONLY | 3 |
| **A6: 108a_v2 + Pareto** | PARTIAL | NO | Deep (in log) | YES | N/A | N/A | IN LOG ONLY | 2 |
| **B3: df sweep JT** | SAVED (6 dirs) | NO | Deep | YES (6 configs) | N/A | N/A | YES (all match) | 1 |
| **B5: Long-horizon** | SAVED (results dir) | NO | Deep | YES (3 models) | N/A | N/A | Results on disk | 1 |
| **KS-levels bug fix** | COMMITTED | YES (git commit) | Deep | YES (oracle test) | DONE (test v2) | N/A | YES (oracle 25/25) | 0 |
| **Test suite code review** | COMMITTED (v2 files) | YES (v2 scripts) | Deep | YES (oracle validates) | DONE (v2 implemented) | N/A | YES (v2 results saved) | 0 |
| **Oracle test** | NOT SAVED | YES (test_v2_oracle.py) | Deep | 3 oracle variants | N/A | N/A | IN LOG ONLY | 2 |
| **V2 test results (4 models)** | SAVED (4 dirs) | YES (v2 script) | Deep | YES (4 models) | N/A | N/A | YES (all match) | 0 |
| **Loss function analysis** | IN LOG ONLY | NO | Deep | N/A | RC4 started | N/A | IN LOG ONLY | 2 |
| **Arch principles review** | IN LOG ONLY | NO | Deep | N/A | Untested | N/A | N/A | 2 |
| **Research Compass 2** | IN LOG | N/A | Deep | N/A | H1: DONE, H2: DONE, H3: INCOMPLETE | N/A | N/A | 1 |
| **Research Compass 3** | IN LOG | N/A | Deep | N/A | H1: DONE, H2: DONE, H3: DONE | N/A | N/A | 0 |
| **RC4 (IS fix)** | MODEL EXISTS | PARTIAL (cmd in log) | N/A (just planned) | N/A | NOT STARTED | N/A | NO RESULTS | 3 |
| **120b_v5_is_fix** | MODEL TRAINED | NO | N/A | N/A | NOT EVALUATED | N/A | NO RESULTS | 3 |
| **Lit survey** | IN LOG | NO | Deep | 5 agents | N/A | N/A | N/A | 1 |
| **Meta-research** | IN LOG | NO | Deep | N/A | N/A | N/A | N/A | 1 |
| **Cross-domain search** | IN LOG | NO | Deep | 5 domains | N/A | N/A | N/A | 1 |

---

## Phase 2 Summary: Metrics Verification

All specific numerical claims verified against disk artifacts:

| Claim | Log Value | Disk Value | Status |
|-------|-----------|-----------|--------|
| 134c score | 69.41 | 69.41 | MATCH |
| 134b score | 69.36 | 69.36 | MATCH |
| 134d score | 54.21 | 54.21 | MATCH |
| 134a score | 64.23 | 64.23 | MATCH |
| 133f score | 63.78 | 63.78 | MATCH |
| 133c score | 53.54 | 53.54 | MATCH |
| 132a score | 44.24 | 44.24 | MATCH |
| 132b score | 54.63 | 54.63 | MATCH |
| 130a score | 66.90 | 66.90 | MATCH |
| E3 score | 68.33 | 68.33 | MATCH |
| E4 score | 68.74 | 68.74 | MATCH |
| E5 score | 65.3 | 65.30 | MATCH |
| E6 score | 65.92 | 65.92 | MATCH |
| E9 score | 66.01 | 66.01 | MATCH |
| E_best3 score | 67.83 | 67.83 | MATCH |
| 134b KS daily | 23/25 | 23/25 | MATCH |
| 134c KS daily | 24/25 | 24/25 | MATCH |
| 134b floor explosion | 0.00% | 0.0 | MATCH |
| 134b bias mag | 21/25 | 21/25 | MATCH |
| 134c bias mag | 21/25 | 21/25 | MATCH |
| 120b v2 suites | 7/9 | 7/9 | MATCH |
| 99m_v2 v2 suites | 6/9 | 6/9 | MATCH |
| 108a v2 suites | 6/9 | 6/9 | MATCH |
| 133f v2 suites | 5/9 | 5/9 | MATCH |
| 133f Suite 9 | FAIL | FAIL (corr ratio on disk) | MATCH |

**Zero discrepancies found between log claims and disk artifacts.**

---

## Phase 3: Triage & Plan

### Gap Categorization

| # | Gap Type | Severity | Experiment/Task | Description |
|---|----------|----------|----------------|-------------|
| 1 | **MISSING_RESULTS** | HIGH | **120b_v5_is_fix** | Model trained (3 checkpoints) but NEVER evaluated. RC4 is the active research direction (IS width fix). No test suite results exist. |
| 2 | **MISSING_RESULTS** | HIGH | **A3: PC1 loading comparison** | Cross-architecture PC1 analysis (4 models, cosine similarities, cross-model similarity matrix) exists only in conversation text. No saved JSON or script. Key structural insight (JT learns different factors). |
| 3 | **MISSING_RESULTS** | HIGH | **A5: Per-suite failure cell map** | 367-model failure aggregation exists only in conversation text. Cell (0,3) = worst in 47% of models. No saved data. This is a foundational analysis for future work. |
| 4 | **MISSING_RESULTS** | HIGH | **Oracle test results** | Oracle test script exists (test_v2_oracle.py) but no saved result JSON. The oracle validates all suites — this is test infrastructure verification. |
| 5 | **MISSING_RESULTS** | MEDIUM | **A6: Pareto frontier** | 335-model kurtosis-cointegration Pareto analysis, 9 Pareto-optimal models identified. Exists only in log text. |
| 6 | **MISSING_RESULTS** | MEDIUM | **Loss function analysis** | IS width term bug analysis, CRPS gradient asymmetry simulation. Only in log. Critical finding that drove RC4. |
| 7 | **NO_SCRIPT** | MEDIUM | **All 14 trained experiments** | No reproducible .sh scripts for any training run. Commands are embedded in log text but not saved as executable scripts. |
| 8 | **NO_SCRIPT** | MEDIUM | **All 16 inference evaluations** | No reproducible scripts for ensemble evaluations, df sweeps, or inference-only configs. |
| 9 | **NO_SCRIPT** | MEDIUM | **Round 3 & 4 investigations** | Diagnostic analyses (per-layer rank traces, variance decompositions, etc.) have no saved scripts. |
| 10 | **SHALLOW_ANALYSIS** | MEDIUM | **133e** | Only reports metrics + "val_loss doesn't track kurtosis quality". No investigation of WHY kurtosis overshoots or whether ep9-19 sweet spot is reproducible. |
| 11 | **SHALLOW_ANALYSIS** | MEDIUM | **133f_v2** | Only one line: "Higher IS weight creates high CI from ep 1". No root cause, no comparison with 133f at matched epochs. |
| 12 | **SHALLOW_ANALYSIS** | MEDIUM | **133f_v3 (RC2-H2)** | Single paragraph. No per-cell analysis of where cointegration collapsed. RC2-H2 falsified but mechanism not explored. |
| 13 | **UNTESTED_HYPOTHESIS** | MEDIUM | **RC2-H3: Learned anchor** | Proposed but never tested. Replacing history[-1] with learned anchor is still a viable direction for bias reduction. |
| 14 | **INCOMPLETE_FOLLOWUP** | LOW | **Arch principles review** | "Minimal architecture test" proposed (strip vol_scale, cell_spread, NoiseMLP, frozen encoder) but never executed. |
| 15 | **INCOMPLETE_FOLLOWUP** | LOW | **Student-t normalizer fix for retraining** | Cross-cutting finding: /1.414 wrong for df=6. Fix validated at inference-time but retraining never done. |
| 16 | **INCOMPLETE_FOLLOWUP** | LOW | **124a retry with proper init** | Gradient trap identified, fix known (normal init on expand layer), but retry never done. |
| 17 | **STALE_METRIC** | LOW | **compute_score L2 always 3.0** | V1 composite scores include 3.0 phantom L2 points. All scores in log use v1 scoring. V2 scoring exists but v2 scores are reported separately. Not technically stale (both systems documented) but worth noting. |
| 18 | **DUPLICATE_ENTRY** | LOW | **Arch principles review** | Entry appears TWICE in log (lines 31609-31711 and 31713-31788). Identical content duplicated. |

### Gap Summary

| Severity | Count | Types |
|----------|-------|-------|
| HIGH | 4 | 3 MISSING_RESULTS, 1 MISSING_RESULTS (unevaluated model) |
| MEDIUM | 9 | 3 MISSING_RESULTS, 3 NO_SCRIPT, 3 SHALLOW_ANALYSIS + 1 UNTESTED_HYPOTHESIS |
| LOW | 5 | 3 INCOMPLETE_FOLLOWUP, 1 STALE_METRIC, 1 DUPLICATE_ENTRY |
| **Total** | **18** | |

### Systematic Pattern

**3+ gaps of the same type**: YES — **7 MISSING_RESULTS** gaps (HIGH + MEDIUM). This is a recurring pattern: subagent analyses (A3, A5, A6, oracle test, loss analysis) were performed thoroughly but their outputs were not persisted to disk. The analyses exist only in research log text, not as structured data files. This matches the exact failure mode the validation skill was designed to catch.

Additionally, **NO_SCRIPT** applies to virtually every experiment and investigation. Training commands exist in the log but are not saved as standalone reproducible .sh files.

---

### Proposed Verification Tasks

Given >10 gaps, presenting HIGH severity first (per skill guidelines):

#### Batch 1: HIGH Priority (4 tasks)

| Task | Description | Type | Estimated Time | Dependencies |
|------|-------------|------|---------------|-------------|
| **T1: Evaluate 120b_v5_is_fix** | Run v1 and v2 test suites on the trained model. This is the RC4 result — the active research direction. | GPU (test suite) | 10 min | Model exists at models/backfill/afcrps_120b_v5_is_fix/best_model.pt. CRITICAL: verify model config has noise-free MLP flags (the training command in RC4 entry does NOT include noise-free flags — possible config mismatch). |
| **T2: Persist A3 PC1 analysis** | Write script that runs PCA on daily changes from 4 architectures (133f, 120b, 99m_v2, 111b), computes cosine similarities with GT, saves to JSON. | GPU (sampling) | 10 min | All 4 model checkpoints exist |
| **T3: Persist A5 failure cell map** | Write script that reads all 367 summary.json files, aggregates per-cell failure patterns, saves master table to JSON. | CPU only | 5 min | All result dirs exist |
| **T4: Save oracle test results** | Run test_v2_oracle.py, save results JSON for all 3 oracle variants. | GPU (test suite) | 10 min | test_v2_oracle.py exists |

#### Batch 2: MEDIUM Priority (5 tasks, independent)

| Task | Description | Type | Estimated Time | Dependencies |
|------|-------------|------|---------------|-------------|
| **T5: Persist A6 Pareto analysis** | Write script that reads all summary.json files, extracts kurtosis + cointegration, computes Pareto frontier, saves to JSON. | CPU only | 5 min | All result dirs exist |
| **T6: Persist loss function analysis** | Write script reproducing CRPS gradient asymmetry simulation and IS width analysis. Save results. | CPU only | 5 min | None |
| **T7: Deepen 133e analysis** | Evaluate 133e at checkpoints ep10, ep19, ep27. Check if ep10-19 sweet spot is real. | GPU (test suite) | 15 min | models/backfill/afcrps_133e/ must have epoch checkpoints |
| **T8: Create training script archive** | Extract all training commands from log entries in this window, save as .sh files in results/validations/2026-03-21/scripts/. | CPU only | 10 min | None |
| **T9: Verify 120b_v5 model config** | Check whether 120b_v5_is_fix has correct noise-free MLP configuration. The RC4 training command does NOT include noise-free flags. If config is wrong, the model is not testing what RC4 intended. | CPU only | 2 min | Model exists |

#### Batch 3: LOW Priority (deferred unless user requests)

| Task | Description |
|------|-------------|
| T10: Test RC2-H3 (learned anchor) | Train variant with learned anchor replacing history[-1]. Still viable. |
| T11: Retry 124a with proper init | Fix gradient trap, test low-rank spread concept. |
| T12: Retrain with fixed Student-t normalizer | Cross-cutting improvement for all Student-t models. |
| T13: Fix duplicate log entry | Remove duplicate arch principles review (lines 31713-31788). |
| T14: Run minimal architecture test | Strip crutches from joint transformer, test from scratch. |

---

### Ideation Recommendation

The audit found **7 MISSING_RESULTS gaps** following a systematic pattern: subagent analyses performed in conversation context but not persisted to disk. This is not a research direction problem but a process problem. No `research-ideation` invocation is recommended — the gaps are fill-able through verification, not rethinking.

However, **T1 (evaluate 120b_v5_is_fix) combined with T9 (verify its config)** is the highest-value task. If the model was trained without noise-free MLP flags, the RC4 experiment may need to be re-run, which would be the most important finding from this audit.

---

### Key Findings from the Audit

1. **All 25 numerical claims verified against disk match exactly.** Zero discrepancies between log metrics and summary.json files. The research log is highly trustworthy for metrics.

2. **120b_v5_is_fix exists but was never evaluated.** This is the RC4 interval score width fix model — the current active research direction. Evaluation is the single highest-priority task.

3. **120b_v5_is_fix may have a config problem.** The RC4 training command (line 32320) does NOT include the `--noise_free_mlp` flag that made 120b distinctive. The checkpoint shows `noise_free_mlp: NOT SET`. If this model is a standard AR MLP (not noise-free), it's testing the IS fix on the wrong architecture class, and the results won't be comparable to the 120b baseline (7/9 v2).

4. **Seven analyses from subagent rounds exist only in log text**, not as saved artifacts. The most critical are A3 (PC1 factor comparison) and A5 (367-model failure map) — these inform all future research decisions.

5. **No reproducible scripts exist for any experiment.** Commands are in the log but not saved as .sh files. A full-day research session would be difficult to reproduce from scratch.

6. **Duplicate log entry** at lines 31609-31711 and 31713-31788 (Architectural Principles Review appears twice with identical content).

7. **V2 test suite is fully validated.** test_v2.py, compute_score_v2.py, and test_v2_oracle.py all exist on disk. Oracle passes 8/8 testable suites. V2 results for 4 models are saved and verified.

8. **All 4 Research Compass entries have clean hypothesis tracking.** RC1: H1 falsified, H2 skipped (informed by H1), H3 exhausted, H4 succeeded. RC2: H1 done, H2 falsified, H3 not tested. RC3: H1 partial, H2 falsified, H3 falsified. RC4: just started, not evaluated.

---

**STOP: Awaiting user approval before executing Phase 4-6.**

The user may:
- Reprioritize tasks (e.g., T1+T9 before anything else)
- Skip items (e.g., skip T7-T8 as low value)
- Add tasks (e.g., re-run RC4 training with correct noise-free flags)
- Choose to invoke `research-ideation` instead
