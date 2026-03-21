#!/bin/bash
# Reproduction script for eval-standard-trigger
# Purpose: Re-run the "validate experiments from the last 2 days" eval
# Expected: Audit table with 5+ experiments, gap categorization, stops at approval gate
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

claude -p "You are testing a skill. Read the skill at .claude/skills/validation/SKILL.md, then follow it to execute this user request:

\"validate experiments from the last 2 days\"

IMPORTANT:
1. Follow the skill's Phase 1-3 EXACTLY (Scan → Audit → Triage)
2. STOP at the user approval gate (do NOT execute Phase 4-6)
3. Save your complete output (audit table, gap categorization, proposed tasks) to:
   .claude/skills/validation-workspace/iteration-1/eval-standard-trigger/with_skill/outputs/audit_output.md
4. This is a REAL project — research log at RESEARCH_LOG.md, results in results/block_ar/, models in models/backfill/
5. Read CLAUDE.md for project context.
Do NOT ask questions. Execute Phases 1-3 fully and save the output."
