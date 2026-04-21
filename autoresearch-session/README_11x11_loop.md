# 11x11 Autoresearch Loop

This repo now has the minimum scaffolding for a persistent Codex-driven autoresearch loop.

## Primary Workflow: Stay In Session

The default workflow is now **in-session**.

You stay in the same Codex session and use commands like:

- `continue autoresearch`
- `run 2 iterations`
- `run until blocked`

Each iteration:

- reads the persistent state,
- chooses the next principled step,
- executes it,
- appends to the actual end of `RESEARCH_LOG.md`,
- and makes one focused commit.

If the session later pauses or ends, the next session resumes from saved state.

Current in-session default:

- `continue autoresearch` means **keep going until manually stopped or a real stop condition occurs**
- manual stop is via `autoresearch-session/STOP`
- the loop must remain scientifically clean:
  - if the pathology becomes unclear, pause experimentation and do postmortem/ideation
  - if the active line starts accumulating too many knobs, do ideation or paradigm shift instead of continuing local patching

## Files

- `goal_11x11.json`
  - declares the objective and hard constraints
- `state_11x11.json`
  - persistent state updated after each iteration
- `check_goal_11x11.py`
  - machine-checkable stop condition
- `driver_prompt_11x11.md`
  - the prompt your outer driver should feed to Codex
- `CONTINUE_AUTORESEARCH.md`
  - the in-session continuation prompt / protocol
- `.agents/skills/autoresearch-head-loop/SKILL.md`
  - the per-iteration protocol

## Required Logging Behavior

Every iteration must append to the actual tail of `RESEARCH_LOG.md` via the `research-log-tail-append` skill.

## Required Commit Behavior

Every iteration ends with one focused git commit.

## Stop Behavior

The loop stops when one of these is true:

- `current_best_n_pass >= 11`
- `state_11x11.json` sets `goal_reached = true`
- `autoresearch-session/STOP` exists

## Optional Outer Driver Pattern

Adapt this pattern to however you invoke Codex non-interactively:

```bash
while ! python autoresearch-session/check_goal_11x11.py; do
  # Replace this with your actual Codex CLI invocation.
  # The only requirement is that it runs Codex once using:
  #   autoresearch-session/driver_prompt_11x11.md
  YOUR_CODEX_COMMAND_HERE
done
```

Examples of acceptable outer-driver environments:

- `tmux` session with a shell loop
- local scheduler / cron
- CI / GitHub Actions style repeat runner
- any wrapper script that can re-invoke Codex until `check_goal_11x11.py` exits `0`

## Recommended Discipline

- one decisive iteration per commit
- no skipped logging
- no skipped commit
- no broad manual staging of local tooling state
- prefer evidence-driven shifts between:
  - analysis
  - ideation
  - paradigm shift
  - experiment
- keep the active model elegant and defensible; reaching `11/11` by knob soup is not an acceptable terminal state

## Why This Works Better

Without this setup, “keep going until 11/11” is only an aspiration.

With this setup:

- the stopping rule is explicit,
- the state is persistent,
- the research log append is enforced,
- and the next in-session continuation has a concrete handoff point.
