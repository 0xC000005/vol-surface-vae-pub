---
name: validation
description: "Research gap analysis and reproducible audit — scans recent research log entries, identifies missing follow-up analyses and unsaved results, launches parallel agents to fill gaps with reproducible scripts, and persists all artifacts to disk with git commits. Use this skill whenever the user wants to find what analysis is missing, check what follow-up can be done immediately, identify inference-only experiments to run, review experiment thoroughness, verify results are saved and reproducible, or says things like 'validate experiments', 'check what we missed', 'what analysis is missing', 'what can we do without retraining', 'is everything verified', 'run validation pass', 'audit the research', 'review what we've done', 'any follow-up analysis we can do', 'check reproducibility'. Also use when returning after running experiments to review quality, after merging experiment branches, when prior conversation results may not have been persisted to disk, or when the user says 'fork validation branch' or 'validation conversation'."
---

# Validation — Research Gap Analysis & Reproducible Audit

Two jobs: (1) identify what follow-up analysis is missing and can be done immediately,
and (2) ensure every claim has verifiable evidence on disk with reproducible scripts.

## Why This Exists

**Primary use case**: After days of experiments, you return and ask "what analysis is
missing? what follow-up can we do right now without retraining?" The skill scans the
research log, identifies experiments that lack standard follow-up analyses (factor
structure, per-cell breakdowns, cross-model comparison, long-horizon tests, ensemble
evaluation), and dispatches agents to fill those gaps — all with reproducible scripts.

**Secondary use case**: Subagents run analyses in conversation but don't persist results
to disk. When the conversation ends, results vanish. This has happened:

- PC1 factor analysis across 4 architectures — deep analysis, no saved JSON or script
- 367-model failure cell map — foundational finding, exists only in log text
- Oracle validations — reported in conversation, scripts not saved

The skill fixes both problems: it identifies WHAT analysis is missing (gap analysis)
and ensures everything that runs is SAVED (the Verification Agent Contract).

## Relationship to Other Skills

- **autoresearch**: Forward loop — Hypothesize → Experiment → Document → Analyze
- **validation**: Gap analysis + backward check — What analysis is missing? Are results saved?
- **research-log**: Shared dependency — both skills read/write the research log
- **research-ideation**: Generates Research Compass hypotheses with falsification tests.
  Validation audits whether those hypotheses were tested, whether falsification was clean,
  and whether outcomes updated the evidence summary. Ideation depends on accurate metrics
  in the research log — validation ensures they're trustworthy.

Validation complements autoresearch. Run it periodically (every 1-2 days of active
research) or after merging experiment branches to catch gaps before they compound.

When validation reveals **systematic patterns** in gaps (e.g., all experiments in a
direction have shallow analysis, or multiple contradictions emerge), recommend the user
invoke `research-ideation` to synthesize new directions from the accumulated evidence.
Validation finds the gaps; ideation fills them with principled next steps.

## The Validation Cycle

```
Scan → Audit → Triage → [USER APPROVAL] → Execute → Document → Commit
```

### Phase 1: Scan Recent Work

Determine the lookback window (default: 2 days, override via user instruction).

1. **Calculate date range**:
   ```bash
   START_DATE=$(date -d "2 days ago" +%Y-%m-%d)
   TODAY=$(date +%Y-%m-%d)
   ```

2. **Search research log**: Use both date headers AND experiment IDs to catch everything:
   ```bash
   # Date-based scan (catches entries by date)
   grep -n "^## 20" RESEARCH_LOG.md | tail -50
   # Experiment-based scan (catches cross-day experiments)
   grep -n "^### Exp" RESEARCH_LOG.md | tail -50
   ```
   Read entries whose dates fall within the range using `Read` with offset/limit.
   Also use QMD for semantic search:
   ```
   Skill(skill="research-log", args="search: experiments from last 2 days")
   ```

3. **Extract inventory** from each entry:
   - **Experiment IDs**: patterns like `Exp <id>`, `### Exp <id>`, experiment numbers
   - **Claims**: specific metrics, scores, pass/fail counts, comparisons
   - **Result paths**: any `results/`, `summary.json`, or model paths mentioned
   - **Verifications described**: tests run, analyses performed, scripts executed
   - **Follow-up items**: "next steps", "should try", "needs investigation", "TODO"
   - **Research Compass hypotheses**: entries titled "Research Compass" contain ranked
     hypotheses with falsification tests. Track which hypotheses were tested and whether
     their kill conditions were evaluated cleanly.

4. **Cross-reference with disk**: For each experiment/verification found, check if
   artifacts actually exist:
   ```bash
   # Check for result directories matching each experiment
   ls results/block_ar/<exp_id>*_30d/summary.json 2>/dev/null
   # Check for saved scripts
   ls results/validations/*/scripts/*<exp_id>* 2>/dev/null
   ```

### Phase 2: Audit Completeness

Assess each experiment across 7 dimensions. The first 5 check individual experiments;
the last 2 check the research direction as a whole.

| Dimension | Check | Method |
|-----------|-------|--------|
| **Results saved** | Does summary.json or equivalent exist on disk? | Glob/ls for expected result paths |
| **Script exists** | Can someone reproduce this exact verification? | Check for .sh files, or training commands in the log entry |
| **Depth** | Was root cause explored beyond pass/fail? | See "Depth Standard" below |
| **Cross-validation** | Were claims checked from multiple angles? | Look for different seeds, different metrics, alternative approaches |
| **Follow-up done** | Were noted follow-up items addressed? | Cross-reference "next steps" against later entries |
| **Compass tested** | If a Research Compass hypothesis targeted this, was its falsification test actually run? | Search log for Research Compass entries, cross-reference hypothesis IDs |
| **Metrics verified** | Do specific numbers cited in the log match what's on disk? | Read summary.json, compare against log claims |

#### Depth Standard (from research-ideation)

"Shallow" vs "Deep" is not subjective. Use the ideation skill's post-experiment
checklist as the rubric. A "Deep" entry must show evidence of at least 3 of these:

1. **Nanda Q1**: "Was my prediction correct? Where exactly did it diverge?"
2. **Nanda Q2**: "What would I do differently?"
3. **Nanda Q3**: "What is the MOST INTERESTING thing about this result?"
4. **Falsification cleanliness**: Did the experiment test what was intended, or were
   there confounds (implementation bugs, hyperparameter leaks, dirty controls)?
5. **Mechanism identified**: Can you explain WHY, not just WHAT happened?
6. **Evidence summary updated**: Did the entry change our understanding of the system?

A log entry that only reports metrics and a one-line conclusion is SHALLOW regardless
of how good the numbers are. The WHY is what makes research cumulative.

#### Analysis Playbook (the primary gap-finding tool)

For each experiment with results on disk, check which standard follow-up analyses
exist. This is the forward-looking part of the audit — it identifies what SHOULD be
done but hasn't been. These are all inference-only (no retraining needed).

**For any trained model with test suite results:**

| Analysis | Check | When to flag as missing |
|----------|-------|----------------------|
| **Test suite evaluation** | Does summary.json exist with current test suite? | Always — no results = UNEVALUATED_MODEL |
| **Per-cell/per-horizon breakdown** | Were borderline metrics investigated at cell level? | When any suite is within 1 test of pass/fail threshold |
| **Factor analysis** | PCA on generated samples (eff_rank, PC1 loadings, variance explained) | When model shows unusual cross-cell correlation or passes new suites |
| **Cross-model comparison** | Metrics compared against baseline and recent experiments | When model is claimed as "best" or "breakthrough" |
| **Long-horizon test** | 252-day evaluation | When 30-day results are promising (5+ suites) |
| **Ensemble evaluation** | Combined with complementary models | When model has orthogonal strengths to existing best |
| **Checkpoint comparison** | Best vs intermediate epochs | When training dynamics show non-monotonic quality |
| **Multi-seed verification** | Same config, different seeds | When result is borderline or claimed as significant |

**For a Research Compass hypothesis:**

| Analysis | Check |
|----------|-------|
| **Falsification test run** | Was the specific kill condition evaluated? |
| **Clean falsification** | Was the test free of confounds? |
| **Evidence summary updated** | Did the outcome change our understanding? |

Don't flag MISSING_ANALYSIS for every cell in the playbook — use judgment. A clearly
failed experiment (2/8 suites) doesn't need factor analysis. A breakthrough result
(first to pass a new suite) needs all of them.

Produce an audit table:
```
| Exp/Task   | Results | Script | Depth    | Cross-val | Follow-up | Gaps |
|------------|---------|--------|----------|-----------|-----------|------|
| 120b v2test| MISSING | NONE   | Thorough | YES       | N/A       | 2    |
| 133f oracle| MISSING | NONE   | Shallow  | NO        | 0/1       | 4    |
| 99m factor | SAVED   | NONE   | Deep     | YES       | 1/1       | 1    |
```

### Phase 3: Triage & Plan

Categorize each gap:

| Gap Type | Severity | Description |
|----------|----------|-------------|
| **UNEVALUATED_MODEL** | HIGH | Model checkpoint exists but was never evaluated with test suite |
| **MISSING_RESULTS** | HIGH | Analysis described in log but no results on disk |
| **UNVERIFIED_CLAIM** | HIGH | Log makes a specific numerical claim with no artifact |
| **STALE_METRIC** | HIGH | A metric cited in a Research Compass hypothesis doesn't match disk |
| **MISSING_ANALYSIS** | MEDIUM | Standard follow-up analysis not done (per Analysis Playbook) |
| **SHALLOW_ANALYSIS** | MEDIUM | Only pass/fail, no root cause (fails Depth Standard above) |
| **NO_SCRIPT** | MEDIUM | Results exist but can't be reproduced |
| **UNTESTED_HYPOTHESIS** | MEDIUM | Research Compass hypothesis has a falsification test never run |
| **DIRTY_FALSIFICATION** | MEDIUM | Hypothesis was "falsified" but confounds noted in log |
| **INCOMPLETE_FOLLOWUP** | LOW | Follow-up items noted but never addressed |

**MISSING_ANALYSIS** is the most common gap type in practice. An experiment can have
results saved, claims verified, and thorough log entries — but nobody ran factor
analysis, or tested at long-horizon, or checked different seeds. The Analysis Playbook
above defines what "complete" looks like. Flag experiments where high-value analysis
is missing, using judgment about which playbook items apply.

**UNEVALUATED_MODEL** is the highest-priority gap — a trained model with no test suite
results means the entire training effort is unverified. In eval testing, this was the
#1 finding: the RC4 model (120b_v5_is_fix) was trained but never evaluated.

Group gaps into concrete, independent verification tasks. Each task should be:
- **Independent**: can run in parallel with others
- **Self-contained**: agent has all context needed
- **Bounded**: clear success criteria, won't spiral

#### When to Recommend Ideation

If the audit reveals **3+ gaps of the same type** or **contradictions between experiments**,
this signals the research direction needs principled rethinking, not just more verification.
After presenting the audit, recommend:

> "The audit found [pattern]. Consider invoking `research-ideation` to synthesize
> new directions from the accumulated evidence before running more experiments."

Validation finds WHERE the gaps are. Ideation determines WHAT to do about systematic ones.

**STOP HERE. Present the audit table and proposed tasks to the user. Wait for approval.**
The user may reprioritize, skip items, add tasks, or choose to invoke ideation first.
Do not execute without approval.

### Phase 4: Execute Verifications

After user approval, dispatch parallel agents for each verification task.

**Critical**: Every agent prompt MUST include the Verification Agent Contract (below).
This is the mechanism that prevents the "results lost in conversation" failure mode.

#### Context the Orchestrator MUST Provide

Each agent prompt must include, alongside the contract:
- **Full model path** (e.g., `models/backfill/afcrps_120b/best_model.pt`)
- **Specific claims to verify**, with exact numbers from the research log
- **Test command template** from CLAUDE.md, including `--no_ema` (project invariant —
  EMA destroys conditionality) and any model-specific flags
- **Expected result path** where summary.json should be saved
- **The VAL_DATE** string so all agents use the same validation directory

Steps:
1. Set `VAL_DATE=$(date +%Y-%m-%d)` — all agents use the same date directory
2. Create the validation directory structure:
   ```bash
   mkdir -p results/validations/$VAL_DATE/{scripts,verification_results,analysis}
   ```
3. Check GPU availability: `nvidia-smi` — if >50% GPU memory is in use by a training
   process, mark GPU-dependent tasks as SCRIPT_ONLY. Low usage (<2GB) = proceed.
4. Dispatch agents using the `dispatching-parallel-agents` skill pattern:
   - One agent per independent task
   - Each gets: the contract (verbatim) + orchestrator context (above) + task description
   - Agents save to non-overlapping paths (exp_id in every filename)
5. Track dispatched count: note how many agents were launched (needed for Phase 5)

### Phase 5: Document

After all agents complete:

1. **Collect and reconcile results**: Read all `verification_result.json` files.
   Compare count against number of dispatched agents. For any missing result:
   - Check if the script exists (partial completion → agent crashed after writing script)
   - If nothing was written, log as `AGENT_FAILURE` in the manifest
   - Do NOT silently skip missing agents — each gap is a reproducibility risk
2. **Create manifest.json**: Aggregate with summary statistics
3. **Create audit_report.md**: Human-readable summary:
   - Scope (date range, experiments audited)
   - Gaps found (count by severity)
   - Verifications run (pass/fail/script_only counts)
   - Corrections to prior claims (if any)
   - Outstanding items needing human attention
4. **Update research log**: Use the `research-log` skill to append:
   ```
   ## YYYY-MM-DD: Validation Audit — N experiments, M verifications

   ### Scope
   Audited research log entries from <start> to <end>. <N> experiments reviewed.

   ### Gaps Found
   | Type | Count | Details |
   ...

   ### Verification Results
   | Task | Status | Key Finding |
   ...

   ### Corrections
   <Any prior claims that were corrected, with evidence>

   ### Outstanding
   <Items that need human attention or GPU time>
   ```

#### Self-application: the validation session itself must be reproducible

The Verification Agent Contract applies to dispatched agents, but the orchestrator's
own work (the audit table, grading, timing data) must also be saved. Before committing:

- Save timing data from agent task notifications to `timing.json` in each agent's dir
- Grade agent outputs against assertions, save `grading.json`
- Write reproduction scripts (the exact agent prompts) so evals can be re-run
- Everything the orchestrator produces goes to files, not just conversation text

This prevents the meta-problem discovered in eval: the skill found gaps in its own
eval artifacts because the orchestrator didn't follow the same persistence rules.

### Phase 6: Commit

```bash
git add results/validations/$VAL_DATE/
git add results/block_ar/          # any new test suite results
git add RESEARCH_LOG.md
git commit -m "validation($VAL_DATE): audit N experiments, M verifications

Audited: <exp_id_list>
Gaps: X total (H high, M medium, L low)
Verifications: Y run, Z passed
See results/validations/$VAL_DATE/audit_report.md"
```

---

## The Verification Agent Contract

**Include this VERBATIM in every dispatched agent's prompt.** Adapt only the
placeholders marked with `<ANGLE_BRACKETS>`. This is the core mechanism that
prevents artifacts from being lost.

```
=== VERIFICATION AGENT CONTRACT (MANDATORY) ===

You are running a verification task as part of a validation audit.
You MUST follow this contract. Results that exist only in conversation
text are WORTHLESS — they vanish when the conversation ends.

BEFORE running any verification:

1. CREATE output directories:
   mkdir -p results/validations/<DATE>/scripts
   mkdir -p results/validations/<DATE>/verification_results

2. WRITE a self-contained bash script that reproduces this verification:
   - Path: results/validations/<DATE>/scripts/<EXP_ID>_<TYPE>.sh
   - Must be runnable from repo root with: bash <script_path>
   - Include: #!/bin/bash, set -euo pipefail, PYTHONPATH=., all args
   - Header comment: what this verifies, why, expected outcome

3. RUN the verification (execute the script or equivalent commands)

4. SAVE all outputs to standard paths:
   - Test suite results → results/block_ar/<exp_id>_<suffix>/summary.json
   - Analysis outputs → results/validations/<DATE>/analysis/<exp_id>/
   - Figures/plots → same directory as analysis outputs

5. WRITE verification_result.json:
   Path: results/validations/<DATE>/verification_results/<EXP_ID>_<TYPE>.json

   {
     "experiment_id": "<exp_id>",
     "verification_type": "<test_suite|oracle|analysis|comparison|factor|other>",
     "timestamp": "<ISO 8601>",
     "script_path": "results/validations/<DATE>/scripts/<EXP_ID>_<TYPE>.sh",
     "result_paths": ["<all paths where outputs were saved>"],
     "status": "<PASS|FAIL|PARTIAL|SCRIPT_ONLY>",
     "summary": "<one paragraph: what was found>",
     "claims_verified": [
       {
         "claim": "<specific claim from research log>",
         "verified": true|false,
         "evidence": "<path to file or specific metric value>"
       }
     ],
     "metrics": {}
   }

6. RETURN the verification_result.json content as your final message.

SCOPE RULES:
- Your job is VERIFICATION, not investigation. If you discover a discrepancy,
  DOCUMENT it in the verification_result.json and stop. Do not attempt to fix,
  retrain, debug, or investigate root cause — the orchestrator decides next steps.
- Target completion within 10 minutes wall-clock (excluding GPU wait time).
  If blocked on GPU, write the script and exit with SCRIPT_ONLY status.
- If the model checkpoint doesn't exist, check other branches:
  git log --all --oneline -- <model_path>
  If found on another branch, note the branch. If truly missing, set status
  to "FAIL" with summary explaining the checkpoint is unrecoverable.

PERSISTENCE RULES:
- NEVER report results only in conversation text. Everything goes to files.
- If GPU is unavailable, write the script anyway → status: "SCRIPT_ONLY"
- Use paths relative to repo root in result_paths and script_path fields.
  Bash commands may use absolute paths for reliability. Never use /tmp.
- Do not modify existing result files. Create new ones with distinct names.
- If you discover a contradiction with a prior claim, note it explicitly.

=== END CONTRACT ===
```

## Directory Structure

```
results/validations/
└── YYYY-MM-DD/
    ├── manifest.json              # Aggregated results, summary statistics
    ├── audit_report.md            # Human-readable validation summary
    ├── scripts/                   # Reproducible bash scripts (one per verification)
    │   ├── 120b_v2test.sh
    │   ├── 133f_oracle.sh
    │   └── ensemble_weighted.sh
    ├── analysis/                  # Non-test-suite outputs
    │   ├── 120b_factor/
    │   │   └── pca_loadings.json
    │   └── cell_03_bottleneck/
    │       └── drift_analysis.json
    └── verification_results/      # Structured per-verification records
        ├── 120b_v2test.json
        ├── 133f_oracle.json
        └── ensemble_weighted.json
```

Test suite summary.json files go to the project's standard result path (e.g.,
`results/block_ar/<exp>_<suffix>/`). The validation directory holds only scripts,
metadata, and supplementary analysis outputs.

## Context Recovery

If a validation session is interrupted:

1. Check for existing directory: `ls results/validations/$(date +%Y-%m-%d)/`
2. Read `manifest.json` if it exists — shows completed verifications
3. Check `verification_results/` — each file represents a completed task
4. Resume from where things left off; don't re-run completed verifications

## Edge Cases

**No GPU**: Write all scripts, set status SCRIPT_ONLY, document. User runs later:
`for f in results/validations/<date>/scripts/*.sh; do bash "$f"; done`

**Model checkpoint missing**: If the `.pt` file doesn't exist on disk or the current
branch, check other branches: `git log --all --oneline -- models/backfill/afcrps_<id>/`.
If found on another branch, note the branch in the audit. If truly missing, mark as
UNRECOVERABLE — this experiment cannot be verified without the checkpoint.

**Research log too large**: Never read the full file. Use QMD semantic search or
grep for date headers + targeted Read with offset/limit.

**Conflicting results**: Flag prominently in audit report AND research log. The log
is append-only — add a correction entry, don't edit old entries. When a corrected
metric was cited in a Research Compass hypothesis or `autoresearch-session/theory_queue.json`,
flag the dependent entry for re-evaluation.

**Already verified**: If results exist on disk, metrics match claims, and a script
exists, mark VERIFIED and skip. Don't waste GPU time re-running.

**Large backlog (>10 gaps)**: Prioritize HIGH severity. Present in batches of 5-8
to the user rather than launching 15+ agents simultaneously.

**Verification fails**: If running a saved script produces different results than
the research log claims, this is a CORRECTION — the most valuable validation finding.
Document the discrepancy, the correct numbers, and the likely cause.
