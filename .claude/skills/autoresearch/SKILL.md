---
name: autoresearch
description: "Theory-driven autonomous research loop with HEDA cycle (Hypothesize-Experiment-Document-Analyze). Extends Karpathy's autoresearch with scientific reasoning: consults research log before each hypothesis, enforces Bitter Lesson constraints, auto-generates new theoretical directions from accumulated evidence when existing ones are exhausted. Use this skill whenever the user wants to run autonomous experiments, iterate on model architecture, improve test suite scores, optimize ML training, or says 'run overnight', 'iterate until done', 'keep improving', 'autoresearch', 'research loop', 'find a way to pass more test suites', or 'improve the model'. Also use for security audits, shipping workflows, debugging, and fixing."
version: 2.0.0
---

# Autoresearch v2 — Theory-Driven Autonomous Research

Extends [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) with a
scientific reasoning layer. The original Modify -> Verify -> Keep/Discard optimizes
greedily. This version thinks before acting and learns deeply from every outcome.

**Core loop:** Hypothesize -> Experiment -> Document -> Analyze WHY -> Repeat (HEDA)

The most important word in the loop is **WHY**. Every experiment — success or failure —
contains information. A failed experiment that you understand deeply is more valuable
than a successful one you can't explain. The breakthroughs come from accumulating
understanding across many experiments, not from any single lucky change.

**Research philosophy:**
- Never discard information. Every experiment is logged permanently in the research log.
- Always investigate WHY something worked or didn't. The "why" often reveals the next
  breakthrough direction.
- Verify your test methodology before trusting results. A premature conclusion based on
  a wrong test can block a promising path for months.
- Allow temporary regression if the understanding it provides is valuable. Like RL
  exploration — short-term cost for long-term reward. Multiple breakthroughs compound.
- Git never reverts knowledge. Code may revert, but findings persist in the research log.

## Subcommands

| Subcommand | Purpose |
|------------|---------|
| `/autoresearch` | Run the HEDA research loop (default) |
| `/autoresearch:plan` | Interactive wizard: Goal + Scope + Metric + Directions |
| `/autoresearch:security` | Autonomous security audit (load `references/security-workflow.md`) |
| `/autoresearch:ship` | Shipping workflow (load `references/ship-workflow.md`) |
| `/autoresearch:debug` | Bug-hunting loop (load `references/debug-workflow.md`) |
| `/autoresearch:fix` | Error fix loop (load `references/fix-workflow.md`) |

For security/ship/debug/fix subcommands, load the referenced file and follow it.
The rest of this document covers the main `/autoresearch` HEDA loop.

## Setup Phase

### Interactive Setup (`/autoresearch:plan`)

**Batch 1 — Research context:**

| # | Question |
|---|----------|
| 1 | **Goal**: What are you trying to achieve? |
| 2 | **Starting point**: Which model/checkpoint to start from? |
| 3 | **Scope**: What can the agent modify? (files/architecture/training) |
| 4 | **Constraints**: Hard rules? (e.g., Bitter Lesson, no post-hoc fixes) |

**Batch 2 — Metric and execution:**

| # | Question |
|---|----------|
| 5 | **Metric**: How to score each iteration? (composite for multi-objective) |
| 6 | **Guard constraints**: What must never regress beyond tolerance? |
| 7 | **Budget**: Quick runs (30 epochs) or full (60 epochs)? |
| 8 | **Initial directions**: Starting theoretical directions (or auto-generate?) |

### Direct Invocation (`/autoresearch`)

If no config exists in `autoresearch-session/config.json`:

1. Read `CLAUDE.md` for project context and standing directives
2. **Use the `research-log` skill** to search for recent experiments and exhausted approaches
3. Propose a configuration based on findings, get user confirmation

If config exists: resume from last iteration.

### Setup Steps

1. Read all in-scope files for full context
2. **Use the `research-log` skill** to search for what's been tried and what failed.
   The research log is the PERSISTENT source of truth — it contains all past experiments,
   root cause analyses, and proven mechanisms. Always use the `research-log` skill
   (MCP semantic search or Grep+Read) to consult it. Never rely on MEMORY.md alone
   (it's a 200-line summary that gets overwritten).
3. Create experiment branch: `git checkout -b autoresearch-session-YYYYMMDD` — all
   experiment commits go here, never on the main working branch
4. Create `autoresearch-session/` with `config.json`, `results-log.md`, `theory_queue.json`
5. Write `compute_score.py` — composite metric script that reads summary.json
6. Establish baseline — run verification, record as iteration #0
7. Initialize theory queue with configured starting directions
8. Confirm with user, then begin

## The HEDA Loop

```
LOOP:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. HYPOTHESIZE                                          │
  │    ⚠️  PRE-CHECK (skip for iteration 1):                 │
  │    Read current_state.json → get last experiment_id.     │
  │    Grep RESEARCH_LOG.md for "### Exp {experiment_id}".   │
  │    If NOT FOUND → STOP. Write the missing entry NOW      │
  │    (follow step 6 below) before doing anything else.     │
  │    If FOUND → proceed.                                   │
  │                                                         │
  │    Read: results-log.md (this session's iterations)     │
  │    INVOKE the `research-log` skill to search for:       │
  │      - what's been tried and failed (exhausted)         │
  │      - root causes of current failures                  │
  │      - what approaches showed partial promise            │
  │    If theory queue empty → SYNTHESIZE (see below)        │
  │    Pick next hypothesis with theoretical justification   │
  │    Write prediction: "Expect X to improve because Y"     │
  │    Write risk: "May regress Z because W"                 │
  ├─────────────────────────────────────────────────────────┤
  │ 2. EXPERIMENT                                           │
  │    Make ONE focused change (architecture or training)    │
  │    Run Bitter Lesson guard (reject if violated)          │
  │    Git commit with hypothesis in commit message          │
  │    SERIALIZE STATE to current_state.json (iteration #,   │
  │      hypothesis, experiment ID, training command, PID)   │
  │    Train (quick 30ep first; full 60ep if promising)      │
  │    Run test suite + compute composite metric             │
  │    VERIFY METHODOLOGY: before trusting results, check:   │
  │      - Is the test actually testing what we think?       │
  │      - Are there confounded variables?                   │
  │      - Is the evaluation budget sufficient? (noise)      │
  ├─────────────────────────────────────────────────────────┤
  │ 3. DOCUMENT (session log only — research log is step 6)  │
  │                                                         │
  │  A. results-log.md (session-local quick reference):      │
  │     - Iteration #, experiment ID, composite score        │
  │     - One-line hypothesis + one-line result              │
  │     - Decision: BUILD ON / VALUABLE FAILURE / etc.       │
  │                                                         │
  │  B. PREPARE research log content (written at step 6):    │
  │     Draft the RESEARCH_LOG.md entry content now (you     │
  │     have all metrics fresh in context), but the actual   │
  │     write + commit happens at step 6 (BLOCKING GATE).   │
  │     Do NOT skip ahead — step 4 (WHY) may add crucial    │
  │     analysis that belongs in the entry.                  │
  │                                                         │
  │     The entry MUST contain (template for step 6):        │
  │     - Experiment ID + based-on lineage                   │
  │     - Hypothesis with theoretical justification          │
  │     - Architecture/code change description               │
  │     - ALL metrics in a comparison table (vs baseline)    │
  │     - Training command used (reproducibility)            │
  │     - ANALYSIS: WHY it worked or didn't                  │
  │       (this is the most valuable part — a future session │
  │       searching "why did Suite 7 fail" needs to find     │
  │       the mechanism, not just "it regressed")            │
  │     - What was LEARNED (insight, not just outcome)       │
  │     - What this suggests trying NEXT                     │
  │                                                         │
  │     Log wins AND losses with EQUAL detail. A well-       │
  │     documented failure prevents repeating that path AND  │
  │     may contain the insight for the next breakthrough.   │
  ├─────────────────────────────────────────────────────────┤
  │ 4. INVESTIGATE WHY (BLOCKING GATE #1)                    │
  │    ⚠️  This is a BLOCKING step. You MUST run at least     │
  │    ONE diagnostic script before proceeding to DECIDE.     │
  │    Narrative-only explanations ("I think because X")      │
  │    are NOT sufficient. You must VERIFY with code.         │
  │                                                         │
  │    For EVERY result (success or failure), answer:         │
  │      - WHY did this metric improve/regress?              │
  │      - What mechanism caused the change?                 │
  │      - Does this reveal something new about the system?  │
  │      - Does this contradict or confirm previous findings?│
  │                                                         │
  │    REQUIRED investigation actions (at least 2 of these): │
  │      □ Per-cell/per-horizon metric breakdown script       │
  │      □ Weight comparison vs baseline (norms, cosine sim) │
  │      □ Training dynamics analysis (loss components, corr) │
  │      □ Variance ratio / ACF analysis on generated samples│
  │      □ Targeted diagnostic for surprising results        │
  │                                                         │
  │    The investigation may reveal:                         │
  │      - A new root cause → add to theory queue            │
  │      - A methodology bug → fix test, re-evaluate         │
  │      - A confounded variable → design cleaner experiment │
  │      - A promising partial result → refine the direction │
  │                                                         │
  │    DO NOT make premature conclusions. If a result seems  │
  │    to "prove" something can't work, verify:              │
  │      - Was the test methodology correct?                 │
  │      - Were there confounded settings?                   │
  │      - Was the training budget sufficient?               │
  │      - Could the implementation have a bug?              │
  │    A wrong conclusion here can block a promising path    │
  │    for months. When in doubt, investigate more.          │
  │                                                         │
  │    SELF-CHECK before moving to DECIDE:                   │
  │      □ Did I run at least 1 diagnostic SCRIPT (not just  │
  │        read metrics from summary.json)?                  │
  │      □ Can I explain the mechanism to a colleague who    │
  │        wasn't in the room?                               │
  │      □ Did the investigation produce any NUMBERS that    │
  │        weren't in the original test output?              │
  ├─────────────────────────────────────────────────────────┤
  │ 5. DECIDE                                               │
  │    Based on the investigation (not just the score):      │
  │      - BUILD ON THIS: score improved, mechanism understood│
  │      - KEEP WITH NOTE: score slightly worse but the      │
  │        investigation revealed valuable understanding     │
  │      - VALUABLE FAILURE: code stays committed (never     │
  │        revert), but next experiment starts from the      │
  │        pre-experiment base. The findings are logged in   │
  │        the research log permanently.                     │
  │    Update theory_queue.json with new insights            │
  │                                                         │
  │    NO TIME LIMIT on investigation. Do not rush to the     │
  │    next experiment. The WHY is more valuable than the     │
  │    score. Dig until you have a mechanistic explanation:   │
  │      - Run per-cell/per-horizon diagnostics               │
  │      - Compare training dynamics (correlation drift,      │
  │        loss components, weight norms)                     │
  │      - If result is surprising, write a diagnostic script │
  │        to trace the specific mechanism                    │
  │      - Only move to DECIDE when you can explain the       │
  │        result to a colleague who wasn't in the room       │
  ├─────────────────────────────────────────────────────────┤
  │ 6. PERSIST TO RESEARCH LOG (BLOCKING GATE)              │
  │    ⚠️  DO NOT proceed to the next iteration until this   │
  │    step is COMPLETE. This is not optional.               │
  │                                                         │
  │    Use the Skill tool to invoke `research-log` skill:     │
  │      Skill(skill="research-log", args="append")          │
  │    This loads the research-log skill which handles        │
  │    appending via heredoc + MCP re-ingestion.             │
  │    Follow the skill's append template (## date: title).  │
  │    Then call mcp__local-rag__ingest_file to re-index.    │
  │    Fallback ONLY if Skill tool fails: direct cat >>      │
  │    to RESEARCH_LOG.md + manual ingest_file call.          │
  │    The entry MUST contain ALL items from step 3B above.  │
  │                                                         │
  │    Then git commit the research log update.              │
  │                                                         │
  │    WHY THIS IS BLOCKING:                                │
  │    - The research log is the ONLY artifact that survives │
  │      context compaction and session boundaries           │
  │    - results-log.md is session-local and disposable      │
  │    - Without research log entries, future sessions WILL  │
  │      repeat failed experiments (proven in this project:  │
  │      autoresearch session 2026-03-17 deferred 4 entries  │
  │      and only wrote them when the user intervened)       │
  │    - A missing entry = wasted GPU hours in the future    │
  │                                                         │
  │    SELF-CHECK before proceeding:                         │
  │      □ Did I call Skill(skill="research-log") to         │
  │        append? (not raw Edit/cat — use the SKILL)        │
  │      □ Did I call mcp__local-rag__ingest_file to         │
  │        re-index? (or did the skill handle it?)           │
  │      □ Does the entry contain metrics table, WHY         │
  │        analysis, and what was learned?                   │
  │      □ Did I git commit the research log?                │
  │      □ Only THEN start the next HYPOTHESIZE step         │
  └─────────────────────────────────────────────────────────┘
  REPEAT (forever or N times)
```

## Decision Logic: Beyond Greedy Keep/Discard

The original autoresearch is greedy: improve = keep, else = revert. Real research
requires exploration and deep understanding. The decision determines what to BUILD ON
next, not what to erase.

All experiments stay on the branch as permanent history. The decision is about which
commit to use as the BASE for the next iteration:

| Score | Understanding | Decision | Next experiment |
|-------|-------------|----------|-----------------|
| Improved + understood why | Deep | **BUILD ON THIS** | New experiment (e.g., 102b) building on 102a's code |
| Improved + don't understand why | Shallow | **INVESTIGATE FIRST** — spend time understanding the mechanism before building further. An unexplained improvement can't be reliably extended. | Investigation, then new experiment |
| Worse + understood why + reveals mechanism | Deep | **VALUABLE FAILURE** — the code didn't work but the investigation revealed WHY. This "why" often points to the next breakthrough. | New experiment (e.g., 102c) using the insight, starting from pre-102a base |
| Worse + don't understand why | None | **INVESTIGATE** — do NOT move on. Dig into per-cell breakdowns, training dynamics, compare predictions to results. The answer may be the key insight. | Investigation first, then decide |
| Any result from wrong methodology | N/A | **FIX METHODOLOGY** — re-run with corrected test. No conclusions until methodology is sound. | Same experiment, fixed test |

There is no "discard and forget" option. Every experiment gets its own ID, its own
checkpoint directory, its own results directory, and its own research log entry.
The question is always: what did we learn, and what does it tell us to try next?

**Comparison with original autoresearch:**

The original skill has only 3 outcomes: keep, discard, crash. Our version adds two
critical categories the original lacks:
- **"Improved but don't understand why"** — the original would just keep it. We pause
  to understand, because building on a mystery improvement leads to fragile stacks
  that collapse when you change something else.
- **"Worse but reveals mechanism"** — the original would just revert and move on.
  We preserve the finding because understanding WHY it failed is often more valuable
  than the failed code. Many breakthroughs come from deeply understanding failures.

## Hypothesis Generation

### From the Theory Queue

Each direction is a structured object:
```json
{
  "name": "Condition-modulated noise amplitude",
  "theory": "Noise amplitude is fixed (unit variance). Encoder can modulate
             per-cell spread via learned sigma: noise_scaled = noise * sigma(cond)",
  "target": "Suite 7 (regime coverage) — per-cell regime response",
  "risk": "May destabilize training if sigma grows unbounded",
  "attempts": 0,
  "max_attempts": 3,
  "status": "active"
}
```

Pick the active direction with fewest attempts. After max_attempts without
improvement, mark "exhausted".

### When Queue is Empty: SYNTHESIZE

This is the critical capability that makes the loop self-sustaining.

**Step 1 — Gather evidence:**
- Read `autoresearch-session/results-log.md` (all iterations this session)
- **Use `research-log` skill** to search for: "exhausted approaches", "what hasn't been
  tried", "root cause analysis", recent experiment results
- Read the latest `summary.json` to understand current failure modes

The research log is the PERSISTENT source of truth. It contains all past experiments,
root cause analyses, and proven mechanisms. MEMORY.md is a summary that gets overwritten
— always go to the research log for comprehensive evidence.

**Step 2 — Pattern recognition:**
- Which changes improved which metrics? (cross-reference from results log)
- Which changes caused unexpected regressions? (surprises = learning opportunities)
- What architectural mechanisms are shared across failures?
- What does the research log say has NOT been tried?

**Step 3 — Generate 2-3 new directions:**
Each new direction must:
- Reference specific experimental results that motivate it
- Explain the theoretical mechanism (why should this work?)
- Predict which metrics will improve and which might regress
- Not repeat any exhausted direction from the research log

**Step 4 — Present or auto-proceed:**
- Interactive: show user the new directions and reasoning
- Overnight: auto-proceed with the highest-confidence direction, log the reasoning

## Bitter Lesson Guard

**Principle**: Everything must be LEARNED from data. The method must generalize to
any conditional scenario generation problem — not just IV surfaces.

Before committing any change, verify:

| Reject if... | Example |
|--------------|---------|
| Per-cell constants indexed by position | `scale[i, j] = precomputed_value` |
| Data-derived lookup tables | `quantile_map.npz`, per-cell statistics |
| Domain heuristics | "smile must be convex", "term structure slopes down" |
| Post-hoc corrections | Quantile mapping, conformal calibration |
| Magic numbers specific to this dataset | Constants that wouldn't transfer to rates/FX/credit |

**Allowed**: `nn.Parameter`, `nn.Linear` (learned from data), hyperparameters
(architecture choices like hidden_dim, lr), loss function design (proper scoring
rules are mathematical, not domain-specific).

If a change violates Bitter Lesson: do NOT commit it. Discard the working directory
changes (`git checkout -- .`), log "rejected (Bitter Lesson: [specific violation])"
in results-log.md, and move to the next hypothesis.

## Experiment Naming Convention

Every experiment gets a unique ID that encodes its lineage. This project uses a
numbering system established across 100+ experiments — follow it exactly.

### The System

```
[Major]  [Letter]  [_vN]
  102       a        _v2
```

**Major number** (90, 91, ..., 102, 103, ...): A new RESEARCH DIRECTION.
Increment the major number when the theoretical hypothesis changes fundamentally.

Examples from project history:
- 90 = Block-AR afCRPS baseline experiments
- 96 = Boundary conditions (logit, reflecting)
- 99 = Skip connections + energy score + cell_spread
- 100 = Return conditioning
- 101 = Noise correlation ablation

**Letter suffix** (a, b, c, ...): VARIANTS within a direction.
Different architectural implementations of the same theoretical idea.

Examples:
- 99a = ES spread-only
- 99b = Skip + ES spread
- 99j = Skip bypass cell_spread
- 99k = Full energy score
- 99m = K=8 + cell_var

**Version suffix** (_v2, _v3, ...): REFINEMENTS of a specific variant.
Same architecture, different hyperparameters or minor tweaks.

Examples:
- 99j_v2 = Skip bypass with noise_dim=32 (was 16)
- 99j_v3 = Same + tanh bounding
- 99m_v2 = Same + moderate λ_cv=1.0 (was 5.0)

### When to Increment What

| Change type | Example | Naming |
|-------------|---------|--------|
| New theoretical direction | "Try attention-based decoder" | **New major**: 102a |
| Different implementation of same theory | "Try spatial attention vs temporal attention" | **New letter**: 102b |
| Hyperparameter tweak on same impl | "Same attention but 4 heads instead of 8" | **New version**: 102a_v2 |
| Building on a previous experiment's insight | "Combine 99m_v2's cell_var with 102a's attention" | **New major**: 103a (new combination = new direction) |

### Creating New Experiments

NEVER overwrite or modify a previous experiment's checkpoint or results.
Each experiment creates:
```
models/backfill/afcrps_{exp_id}/          # e.g., afcrps_102a/
  best_model.pt
  final_model.pt
  training_history.json
  checkpoint_epoch_10.pt
  ...
results/block_ar/{exp_id}_30d/            # e.g., 102a_30d/
  summary.json
```

The old experiment's files remain intact. This is the lab notebook — you need to
compare 102a against 99m_v2 months later.

### Lineage Tracking

In the research log entry, always state the parent:
```
### Exp 102a: Spatial Cross-Attention Decoder
**Based on**: 99m_v2 (base architecture) + insight from 99l_v3 investigation
  (freeze-at-peak shows correlation can reach GT given enough post-freeze epochs)
```

## Composite Metric

Design a weighted sum where every aspect of quality contributes, not just suite count.
Continuous metrics provide gradient signal between pass/fail boundaries.

**Template** (customize in `compute_score.py`):
```python
score = (suites_passed * 10)                        # 0-80, main signal
      + (ks_daily_pass / total_cells) * 5           # 0-5, distributional
      + clamp(kurtosis_ratio, 0.5, 2.0) * 2         # 1-4, time series
      + (ci_90 - 0.85) * 20                          # 0-3, coverage
      + (1 - suite7_failing / total_combos) * 5     # 0-5, regime
      + growing_unc_pass * 3                         # 0/3, temporal
```

## Training Budget

| Mode | Epochs | Time | When |
|------|--------|------|------|
| Quick | 30 | ~30 min | First attempt at a new direction |
| Full | 60 | ~60 min | Direction showed promise in quick mode |

Strategy: quick first, full only if quick shows improvement. This doubles the
number of directions explored per night.

## Session State

All state in `autoresearch-session/`:
```
config.json          — goal, scope, metric, guard, constraints
results-log.md       — per-iteration: hypothesis, metrics, decision
compute_score.py     — composite metric script
baseline.json        — iteration #0 metrics
current_best.json    — best metrics so far
theory_queue.json    — directions (active/exhausted/completed)
```

On context compaction or new session, follow the Context Recovery Protocol below.

## Compaction Survival & Auto-Resume

Context compaction WILL happen during long overnight runs. The skill is designed to
survive this automatically. Here's how:

**What persists across compaction:**
- CLAUDE.md (reloaded into context automatically)
- Tasks API tasks (persist in ~/.claude/tasks/)
- All files in `autoresearch-session/` (on disk)
- Git branch state
- Background training processes (they keep running!)

**What gets lost:**
- Conversation memory of current iteration
- Loaded reference files
- MCP tool definitions (may need re-fetch via ToolSearch)

### Auto-Resume via Tasks API

At setup, create a persistent task:
```
TaskCreate: "AUTORESEARCH ACTIVE — resume loop from autoresearch-session/"
```

After compaction, the agent sees this task in its task list and knows to resume.
This works even without user intervention.

### Pre-Compaction: Serialize State

Before EVERY training run (which takes 30-60 min and may trigger compaction),
write the current state to disk:

```python
# Write to autoresearch-session/current_state.json
{
  "iteration": 5,
  "current_hypothesis": "Condition-modulated noise amplitude",
  "experiment_id": "102a",
  "training_status": "running",  # or "complete", "evaluating"
  "training_pid": 12345,
  "training_command": "PYTHONPATH=. python ...",
  "base_commit": "abc123",
  "started_at": "2026-03-17T22:15:00"
}
```

This file is the primary recovery artifact.

### Context Recovery Protocol

When resuming after compaction (detected by seeing the active task or
`autoresearch-session/` directory):

1. Read `autoresearch-session/current_state.json` — what was happening?
2. Check if training is still running: `ps aux | grep train_afcrps`
   - If running: wait for it, then continue to evaluation
   - If finished: check if output exists in the expected model directory
   - If crashed: follow Crash Recovery below
3. Read `autoresearch-session/config.json` for goal, scope, constraints
4. Read LAST 20 lines of `results-log.md` for recent iterations
5. Read `theory_queue.json` for direction status
6. **Use the `research-log` skill** to find the last documented experiment
7. Re-fetch MCP tools if needed: `ToolSearch("select:mcp__local-rag__query_documents")`
8. Resume from where current_state.json says we left off

## Crash Recovery

If training crashes (OOM, NaN loss, CUDA error):
1. Log the crash in results-log.md with hypothesis and error message
2. Append to RESEARCH_LOG.md via research-log skill (crashes are informative!)
3. Do NOT count as a full iteration
4. If OOM: try reducing batch_size or n_members, then retry same hypothesis
5. If NaN loss: likely architecture bug — investigate before retrying
6. If code bug: fix and retry (max 2 attempts)
7. Then proceed to HYPOTHESIZE for next iteration

## MCP Server Fallback

If `mcp-local-rag` is unavailable for research log search:
- Fall back to Grep-based methods from the `research-log` skill
- `Grep pattern="### Exp {id}" path="RESEARCH_LOG.md"` for known experiment IDs
- `Grep pattern="^## 2026-" path="RESEARCH_LOG.md"` for section headers
- Then `Read` at the found offset with `limit=80`
- The MCP server is a convenience for semantic search, not a hard dependency

## Critical Rules

1. **Theorize before modifying** — every change needs a hypothesis and prediction
2. **Research log is the ONLY persistent truth** — always use the `research-log` skill
   to search it before proposing "new" ideas. MEMORY.md is a volatile summary that gets
   overwritten. The research log contains every experiment, root cause, and exhausted
   approach. If you skip this step, you WILL repeat failed experiments.
3. **One change per iteration** — atomic, so you know what caused what
4. **Bitter Lesson is non-negotiable** — no domain heuristics, no post-hoc fixes
5. **Allow exploration** — temporary regression OK if theoretically motivated
6. **INVESTIGATION IS A BLOCKING GATE** — Step 4 requires running at least ONE diagnostic
   script (not just reading summary.json metrics). Narrative explanations ("I think
   because X") must be VERIFIED with code. The self-check in step 4 must pass before
   proceeding to step 5. This was added after the 2026-03-17 session where 12/18
   experiments got narrative-only investigation with zero diagnostic scripts.
   **Failure mode to avoid**: writing "WHY: the model probably does X because Y" without
   running ANY code to verify. Every WHY claim must have a NUMBER backing it up.
7. **Verify methodology** — before trusting any result, verify the test is actually
   testing what you think. Check for confounded variables, insufficient eval budget,
   or silent test failures (like cointegration being skipped because returns weren't
   loaded). A wrong test → wrong conclusion → blocked promising path.
8. **Synthesize when stuck** — generate new directions from evidence, don't repeat failures
9. **Quick then full** — validate cheaply before investing in full training
10. **RESEARCH LOG IS A BLOCKING GATE** — You MUST invoke the `research-log` skill to
    append to RESEARCH_LOG.md, then git commit, BEFORE starting the next iteration.
    This is step 6 of the HEDA loop. It is NOT optional. It is NOT deferrable.
    Do NOT batch-write multiple entries later — write EACH entry IMMEDIATELY after the
    DECIDE step. The session-local results-log.md is a convenience copy; RESEARCH_LOG.md
    is the permanent record. Missing entries = repeated experiments = wasted GPU hours.
    **Failure mode to avoid**: writing to results-log.md and "planning to update the
    research log later." Later never comes — context compaction or session end erases
    the intent. Write it NOW, every time, no exceptions.
11. **Experiment branch, never revert knowledge** — create `autoresearch-session-YYYYMMDD`
    branch at setup. All commits stay on the branch — NEVER revert commits, even for
    failed experiments. Code that didn't work is still valuable history. Each commit
    message includes the hypothesis. At session end, user reviews the full branch
    history and cherry-picks architectural winners to the main branch. The branch
    is a complete lab notebook.

## Handling Complex Implementations

Some directions require significant code changes (new architectures, new loss functions,
new training loops). DO NOT STOP the loop and ask the user. The whole point of autoresearch
is autonomous execution. Instead:

**For implementations that take <2 hours:**
1. Break the work into sub-steps using the Tasks API
2. Implement step by step, testing compilation after each
3. If stuck on a specific step for >30 min, simplify the approach
4. Always have a MINIMAL VIABLE version that can train, even if incomplete

**For implementations that take 2+ hours:**
1. Start with a SIMPLIFIED PROXY experiment that tests the core hypothesis
   with less code (e.g., test "does one-shot work?" before "does CSDI work?")
2. If the proxy shows promise, implement the full version
3. If the proxy fails, move to the next direction — don't spend hours on
   something the proxy already disproved

**NEVER DO THIS:**
- Stop the loop and say "this is too complex, should I continue?"
- Skip a direction because it requires "significant code work"
- Ask for permission to implement something in the theory queue

**ALWAYS DO THIS:**
- Estimate implementation time before starting
- If >2 hours: find a simpler proxy experiment first
- Use the Agent tool to parallelize independent implementation tasks
- Commit working code frequently (every 30 min of implementation)
- If compilation fails after 3 attempts, simplify and move on

**Example**: Direction "CSDI 2D attention denoiser" (estimated 3h):
- Proxy: Can the existing Conv3D decoder with 2D attention conditioning
  (instead of AdaGN) improve factor structure? (estimated 1h)
- If proxy works: implement full CSDI framework
- If proxy fails: move to next direction

## Controlled Loop Count

Use Claude Code's `/loop` command for bounded runs:
```
/loop 8 /autoresearch     # 8 iterations (~4-8 hours depending on training budget)
```

After N iterations: print summary with baseline -> best, all hypotheses tested,
directions exhausted vs remaining, recommended next steps.
