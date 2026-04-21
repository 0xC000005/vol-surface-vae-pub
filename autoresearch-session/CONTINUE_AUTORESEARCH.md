Use the `autoresearch-head-loop` skill and the `research-log-tail-append` skill.

Default behavior for this prompt:

- stay **inside the current Codex session**
- read persistent state
- honor the active restart phase in state
- choose the most principled next step
- keep executing full HEAD iterations in sequence
- after each iteration:
  - update state
  - append to the true tail of `RESEARCH_LOG.md`
  - make one focused git commit
- stop only when:
  - `autoresearch-session/STOP` exists
  - the goal is reached
  - a hard blocker requires human input
  - or session/runtime/tool limits make further work unreasonable

Important:

- if `state_11x11.json` says the program has restarted from a clean minimal architecture line,
  do **not** resume old-family follow-ups such as `258b`
- use `205-258` only as baselines, negative evidence, and design constraints
- prefer the smallest publishable architecture with:
  - a vanilla generative core
  - dynamic latent state
  - explicit low-rank factor structure
  - bounded idio path
- if the pathology becomes unclear or the active branch starts accumulating too many knobs,
  do not keep stacking experiments in place:
  - switch to post-experiment analysis
  - or research ideation
  - or paradigm shift
  before continuing experimentation

If the user says:

- `continue autoresearch`
  - keep going until manually stopped or a real stop condition occurs
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
