Use the `autoresearch-head-loop` skill and the `research-log-tail-append` skill.

Default behavior for this prompt:

- stay **inside the current Codex session**
- read persistent state
- honor the active restart phase in state
- choose the most principled next step
- execute one full HEAD iteration
- update state
- append to the true tail of `RESEARCH_LOG.md`
- make one focused git commit

Important:

- if `state_11x11.json` says the program has restarted from a clean minimal architecture line,
  do **not** resume old-family follow-ups such as `258b`
- use `205-258` only as baselines, negative evidence, and design constraints
- prefer the smallest publishable architecture with:
  - a vanilla generative core
  - dynamic latent state
  - explicit low-rank factor structure
  - bounded idio path

If the user says:

- `continue autoresearch`
  - do exactly one next iteration
- `run 2 iterations`
  - do two full iterations in sequence
- `run until blocked`
  - keep iterating in this same session until:
    - goal reached
    - `autoresearch-session/STOP` exists
    - a hard blocker requires human input
    - or session/runtime/tool limits make further work unreasonable

Always read first:

- `autoresearch-session/goal_11x11.json`
- `autoresearch-session/state_11x11.json`
- tail of `RESEARCH_LOG.md`

Do not use generic append edits for the research log. Always use `research-log-tail-append`.
