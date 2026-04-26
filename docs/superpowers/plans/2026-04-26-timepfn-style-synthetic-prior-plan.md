# TimePFN-Style Synthetic Prior Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and smoke-test a TimePFN-style synthetic-prior pretraining path for the current empirical-normal-score IV-surface AR flow.

**Architecture:** Add a deterministic synthetic IV-surface sequence generator that emits 5x5 `(history, future)` windows with calm/stress regimes. Add a 560a pretraining script that trains the existing `EmpiricalNormalScoreCausalMemoryTransitionFlowMatching` model on synthetic windows and saves a normal checkpoint that can be adapted/evaluated by existing 340c-family tooling.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing `diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching`.

---

### Task 1: Synthetic IV Prior Generator

**Files:**
- Create: `experiments/backfill/block_ar/synthetic_iv_prior.py`
- Test: `test_code/test_560a_synthetic_iv_prior.py`

- [ ] **Step 1: Write failing tests**

Create tests that assert deterministic generation, bounded 5x5 windows, regime diversity, and train/future shape compatibility.

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest test_code/test_560a_synthetic_iv_prior.py -q`

Expected: import failure because `synthetic_iv_prior.py` does not exist yet.

- [ ] **Step 3: Implement generator**

Implement `SyntheticIVPriorConfig`, `SyntheticIVPriorBatch`, and `generate_synthetic_iv_windows`.

- [ ] **Step 4: Verify tests pass**

Run: `pytest test_code/test_560a_synthetic_iv_prior.py -q`

Expected: all tests pass.

### Task 2: Synthetic Pretraining Script

**Files:**
- Create: `experiments/backfill/block_ar/train_560a_synthetic_prior_pretrain.py`
- Test: `test_code/test_560a_synthetic_pretrain_smoke.py`

- [ ] **Step 1: Write failing smoke test**

Create a small CPU smoke test that calls the training entry point for one epoch on a tiny synthetic dataset and confirms `best_model.pt`, `training_history.json`, and `args.json` are created.

- [ ] **Step 2: Run test and verify failure**

Run: `pytest test_code/test_560a_synthetic_pretrain_smoke.py -q`

Expected: import failure because the training script does not exist yet.

- [ ] **Step 3: Implement training script**

Reuse the 340a model config, standard flow-matching loss, synthetic windows, and existing checkpoint format.

- [ ] **Step 4: Verify smoke test passes**

Run: `pytest test_code/test_560a_synthetic_pretrain_smoke.py -q`

Expected: pass on CPU in under a minute.

### Task 3: 560a HEAD Result

**Files:**
- Create: `experiments/backfill/block_ar/ANALYSIS_560a_timepfn_synthetic_prior_pilot.md`
- Modify: `autoresearch-session/state_11x11.json`
- Append: `RESEARCH_LOG.md`

- [ ] **Step 1: Run smoke pretraining**

Run the training script with a small but real command into `models/backfill/560a_timepfn_synthetic_prior_smoke`.

- [ ] **Step 2: Optionally run short real-data adaptation**

If the smoke checkpoint is valid, adapt with the existing recent-window 377a script into a 560a adaptation directory.

- [ ] **Step 3: Analyze result**

Write a concise analysis stating whether the TimePFN-style scaffold is implemented and whether the next HEAD iteration should scale it to a full 11-suite run.

- [ ] **Step 4: Log and commit**

Update state, append to `RESEARCH_LOG.md` with the helper script, and commit only the 560a files plus intentional state/log updates.
