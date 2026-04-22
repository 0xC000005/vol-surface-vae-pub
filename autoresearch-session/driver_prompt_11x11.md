Use the `autoresearch-head-loop` skill and the `research-log-tail-append` skill.

Continue the persistent autoresearch loop toward `11/11` on the common full 11-suite for a generalizable conditional scenario generator over general financial factors.

Honor the active restart phase in `state_11x11.json`. If the state says the old
`205-265` program is archived, do not resume `258b`, `265a`, or any other old-family follow-up.
Treat that tree as baselines and falsification history only.

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
3. Keep executing full HEAD iterations in this same session:
   - Hypothesis
   - Execute
   - Analyze
   - Decide
4. After each iteration:
   - update `autoresearch-session/state_11x11.json`
   - append a concise result entry to the actual end of `RESEARCH_LOG.md` using `research-log-tail-append`
   - create one focused git commit for that iteration
5. If the pathology becomes unclear or the active line starts accumulating too many knobs, do not keep stacking local experiments:
   - switch to post-experiment analysis, research ideation, or paradigm shift as appropriate
6. Continue until stopped only by:
   - the goal being reached
   - `autoresearch-session/STOP` existing
   - the configured hard iteration cap being reached
   - or session/runtime/tool limits making further work unreasonable

Do not stop for research/model blockers.
If a blocker appears, convert it into:
- post-experiment analysis
- research ideation
- or paradigm shift
and continue.

Current reset doctrine:
- the only core architectural bias that should be assumed by default is a narrow encoder-decoder bottleneck
- keep the generative core vanilla
- do not assume hard low-rank structure in the core spec
- do not assume bounded idio or EC side paths in the core spec
- if those return later, they must return only as explicit post-failure ablations

This invocation should leave the repository in a resumable state after every iteration.
