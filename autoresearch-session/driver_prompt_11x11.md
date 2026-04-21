Use the `autoresearch-head-loop` skill and the `research-log-tail-append` skill.

Continue the persistent autoresearch loop toward `11/11` on the common full 11-suite for a generalizable conditional scenario generator over general financial factors.

Rules for this invocation:

1. Read:
   - `autoresearch-session/goal_11x11.json`
   - `autoresearch-session/state_11x11.json`
   - the tail of `RESEARCH_LOG.md`
2. Decide the single most principled next step:
   - post-experiment analysis
   - research ideation
   - paradigm shift
   - experiment
3. Execute exactly one full HEAD iteration:
   - Hypothesis
   - Execute
   - Analyze
   - Decide
4. Update `autoresearch-session/state_11x11.json`.
5. Append a concise result entry to the actual end of `RESEARCH_LOG.md` using `research-log-tail-append`.
6. Create one focused git commit for this iteration.

Do not stop just because the answer is uncertain. Stop only if:

- the goal is already reached,
- the stop file exists,
- or a hard blocker requires human input.

This invocation should leave the repository in a resumable state for the next invocation.
