# NLP Prefix Autoresearch Guardrail Sources

## Purpose

This note records the related-work check behind the May 2026 update to the
narrative prefix-latent autoresearch workflow. The update tightens the loop
around fixed starts, bounded sweeps, independent verification, and stale-artifact
handling.

## Sources Checked

- AI Scientist: https://arxiv.org/abs/2408.06292
- AI Scientist Nature update: https://sakana.ai/ai-scientist-nature/
- Agent Laboratory: https://arxiv.org/abs/2501.04227
- FutureHouse Robin: https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system
- Automated Scientific Discovery survey: https://link.springer.com/article/10.1007/s10994-025-06955-2
- Independent AI Scientist critique: https://arxiv.org/abs/2502.14297
- Google Rules of ML: https://developers.google.com/machine-learning/guides/rules-of-ml/
- Google ML Test Score: https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
- OpenAI eval best practices: https://developers.openai.com/api/docs/guides/evaluation-best-practices

## What Transfers

- Use a structured loop: hypothesis, implementation, experiment, report, review.
- Require literature review before major method changes.
- Use specialized verification rather than one free-form agent judgment.
- Keep human intervention points explicit, especially before promotion.
- Treat evaluation datasets, held-out checks, and regression tests as first-class
  workflow objects.

## What Does Not Transfer

- Open-ended autonomous paper generation is not the objective.
- Automated reviewer scores are not sufficient for financial scenario quality.
- Broad tree search over many model variants would add knobs faster than it adds
  understanding.
- Model-chosen starting levels do not match the current product contract; the
  risk manager should supply or manually select the start before conditioning.

## Mechanism Being Tested

The updated workflow tests whether a narrative-conditioned scenario generator
can improve through mechanism-attributed HEAD iterations instead of accumulating
unexplained loss weights, rerankers, support filters, prompt variants, or UI
options.

## Falsifier

If future iterations repeatedly require new knobs whose causal role cannot be
stated before the run, or if independent verification finds stale artifacts,
unreal citations, hallucinated metrics, or hidden model-chosen starts in
promotion claims, the workflow has failed and must switch to
`post_experiment_analysis` or `research_ideation`.

## Implementation Constraints

- Fixed start first: user-selected historical/current start or user-specified
  joint39 state before mixture.
- Sweep cap: one axis, at most three values by default.
- Related-work artifact before new model families or major workflow changes.
- Independent verifier checklist before promotion.
- One current default artifact path per production gate; older reports are
  baselines unless explicitly revalidated.
