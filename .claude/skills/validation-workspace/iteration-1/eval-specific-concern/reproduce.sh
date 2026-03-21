#!/bin/bash
# Reproduction script for eval-specific-concern
# Purpose: Re-run the "check if v2 test results are saved" eval
# Expected: Identifies v2 results on disk, verifies metrics match log, flags untracked oracle files
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

claude -p "You are testing a skill. Read the skill at .claude/skills/validation/SKILL.md, then follow it to execute this user request:

\"I'm worried the v2 test suite results weren't saved to disk. check if all models we tested on v2 have summary.json files\"

IMPORTANT:
1. Follow the skill's workflow — focus on v2 results specifically
2. Check which summary.json files exist and whether they contain v2-specific data (cross_cell_correlation key)
3. Cross-reference research log claims vs disk
4. Produce audit table and proposed verification tasks
5. STOP at approval gate
6. Save output to:
   .claude/skills/validation-workspace/iteration-1/eval-specific-concern/with_skill/outputs/audit_output.md

Read CLAUDE.md for project context. Do NOT ask questions."
