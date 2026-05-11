# NLP Prefix-Latent Autoresearch: Multi-Agent Workflow Review

Date: 2026-05-11

This review checks whether the narrative prefix-latent autoresearch workflow
should directly use multi-agent patterns rather than a mostly single-agent HEAD
loop.

## Sources Checked

- AI Scientist: https://arxiv.org/abs/2408.06292
- Agent Laboratory: https://arxiv.org/abs/2501.04227
- FutureHouse Robin: https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system
- Google AI co-scientist: https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/
- Google agent-scaling study: https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/
- Automated scientific discovery survey: https://link.springer.com/article/10.1007/s10994-025-06955-2

## What Transfers

### AI Scientist

AI Scientist is useful as a full-loop template: ideation, code changes,
experiment execution, plotting/reporting, and automated review. The transferable
lesson is not "let the agent publish"; it is that the research loop must produce
code, experiments, plots, papers, and review artifacts as one auditable chain.

For this repo, that means every promoted result needs:

- hypothesis and falsifier;
- command and artifact path;
- metric readout;
- verifier report;
- paper/demo implication.

### Agent Laboratory

Agent Laboratory emphasizes staged work with human feedback at literature
review, experimentation, and report-writing stages. The transferable lesson is
that human checkpoints are valuable at stage boundaries, especially when the
question changes from implementation to interpretation.

For this repo, the matching stages are:

1. literature or related-work check;
2. local TestFlight or cached experiment;
3. report/paper/demo update;
4. independent verification before promotion.

### FutureHouse Robin

Robin's useful pattern is a simple, interpretable orchestration of specialized
agents rather than a large peer swarm. Crow/Falcon/Finch-style specialization
maps well to this repo:

- literature/search sidecar;
- experiment-design or candidate-ranking sidecar;
- data-analysis/verifier sidecar.

The product lesson is that human execution/validation can remain outside the AI
loop while the intellectual workflow is still machine-assisted and auditable.

### Google AI Co-Scientist

AI co-scientist uses specialized agents for generation, reflection, ranking,
evolution, proximity, and meta-review, coordinated by a supervisor. The relevant
pattern is generate-critique-rank-refine with explicit meta-review.

For this repo, use that pattern only for research-ideation decisions:

- generate 2-3 candidate next hypotheses;
- critique each for mechanism clarity, cost, falsifier, and product relevance;
- rank the candidates;
- choose one HEAD iteration;
- save the rationale.

Do not use it to let agents vote on numerical results. Numerical results must
come from local artifacts.

### Google Agent-Scaling Study

The strongest workflow correction comes from Google's agent-scaling study:
multi-agent coordination helps parallelizable tasks but can degrade sequential
tasks. This argues against "more agents by default."

For this repo:

- use sidecar agents for independent literature search, artifact verification,
  code review, report audit, or competing hypothesis critique;
- keep execution of a tightly coupled code change or experiment harness under
  one orchestrator unless the write scopes are disjoint;
- avoid peer-to-peer debate for long sequential debugging;
- do not sweep agent architectures as a research knob.

### Automated Scientific Discovery Survey

The survey emphasizes autonomy levels, closed-loop discovery, interpretability,
and the need for scientists to intervene with goals, hints, and values. This
supports the current HEAD discipline: the system should be autonomous enough to
execute bounded loops, but its discoveries must be communicated in forms the
human risk manager/researcher can inspect.

For this repo, "production ready" should not mean fully autonomous discovery.
It should mean auditable, reproducible, and interruptible research automation.

## Recommended Multi-Agent Policy

Keep the main workflow as a **centralized HEAD orchestrator**. Add sidecar
agents only when the subtask is independent and can produce a bounded artifact.

Default sidecar roles:

1. **Literature Scout:**
   Use before adopting a new external method analogy, such as CLIP, Sora,
   DALL-E, ControlNet, RAG, Q-Former, or autonomous-research workflows.

2. **Experiment Critic:**
   Use during `research_ideation` before adding a new model family or knob.
   Output must include mechanism, expected movement, trade-off, and falsifier.

3. **Artifact Verifier:**
   Use before promotion. It must inspect code, commands, metrics, saved
   artifacts, and research-log context.

4. **Report Auditor:**
   Use before paper/demo promotion. It checks stale claims, missing figures,
   hidden assumptions, unsupported causal language, and whether the paper/demo
   matches the current-truth index.

5. **Implementation Worker:**
   Use only for disjoint write scopes. The orchestrator owns integration and
   final verification.

## Do Not Use Multi-Agent For

- narrow sequential debugging where the next step depends on one trace;
- editing the same file from multiple workers;
- deciding metric truth by debate instead of artifact inspection;
- broad hyperparameter or architecture sweeps;
- replacing human approval at product-contract changes.

## Workflow Change

Before a promoted result, the workflow should save:

- the orchestrator HEAD entry;
- any sidecar/verifier report under
  `docs/research_protocols/nl_prefix_latent_verifier_reports/` or the experiment
  output directory;
- the current-truth update if the result changes the default or paper claim.

This is the right amount of multi-agent architecture for the current project:
specialized sidecars at decision gates, not a full autonomous scientist swarm.
