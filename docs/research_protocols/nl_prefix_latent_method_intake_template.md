# NL Prefix Latent Method Intake Template

Use this template before launching a new sophisticated
narrative-to-mixture experiment. The goal is to prevent two bad patterns:
metric-chasing methods with no coherent paper story, and elegant-looking
methods that weaken the working simple-mixture baseline.

## Candidate Name

Short name:

Workflow lane: `exploration`, `candidate`, `promotion`, or `production_demo`

Iteration type: `research_ideation`, `experiment`, `post_experiment_analysis`,
or `paradigm_shift`

## Local Bottleneck

What failure mechanism is this candidate targeting?

Required form:

```text
Because <local evidence>, the current pipeline fails when <mechanism>.
This candidate should help because <causal mechanism>, not merely because it is
more complex.
```

## Method Story

Explain in product language how a risk-manager narrative plus fixed starting
level becomes an auditable support mixture.

Checklist:

- The full narrative remains a first-class input.
- Grounding is a sidecar for checks and warnings.
- The fixed start is resolved before mixture selection.
- The historical support mixture remains visible and auditable.
- The frozen SNI rollout path is unchanged unless this intake explicitly says
  otherwise.

## Related Work Basis

List primary sources or local first-principles evidence that justify the design
family. Each source should include one sentence on what transfers and one
sentence on what does not transfer.

Suggested source classes:

- contrastive alignment, when the method learns text/support compatibility;
- learning-to-rank, when the method ranks candidate mixtures within a query;
- permutation-invariant set models, when the method scores support sets;
- retrieval/RAG, when the method uses memory and provenance;
- text-to-time-series generation, when the method changes the latent prior;
- financial scenario-generation literature, when the method affects risk
  evaluation or conditioning.

## Novelty Claim

State the local contribution relative to known methods.

Required form:

```text
This is not just <known method>. The local contribution is <specific mechanism>
because it uses <our narrative/start/support/generator contract> to improve
<specific support-mixture behavior>.
```

## Elegance Check

Why is this the smallest justified method?

Answer each:

- Does it remove or unify an existing component?
- If it adds a component, why is that component necessary?
- What new knob does it introduce?
- What knob or branch is deliberately out of scope?
- Can the result be interpreted if it fails?

## Training And Inference Contract

Training data:

Inference inputs:

Output object:

Leakage guard:

No hidden future information:

No hidden model-chosen start:

## Baselines

Required baselines:

- `soft_topk_narrative_start_checked` / `narrative_generator_topk`;
- direct text-predicted memory without support;
- raw narrative only;
- implications only;
- narrative plus grounding sidecar;
- no-narrative or shuffled-narrative null when the claim is conditionality.

Optional baselines:

- persistence;
- historical replay;
- single-neighbor support;
- oracle true-prefix generator;
- previous candidate method.

## Backtest Gate

Primary metrics:

- held-out energy score;
- held-out ensemble CRPS;
- 80% coverage and interval behavior;
- per-start quality floor when fixed-start claims are involved;
- fixed-start narrative sensitivity;
- same-narrative repeat stability;
- shuffled-narrative or no-narrative null gap;
- support-direction audit pass/warn/fail rate.

Promotion floor:

```text
The candidate must beat or be clearly competitive with the simple mixture.
Small regressions are allowed only when a named trust, warning, OOD,
provenance, or fixed-start-stability gain is visible and measured.
```

## Kill Condition

State the exact result that stops this branch before it is launched.

Examples:

- held-out CRPS and energy are both worse than the simple mixture without a
  compensating trust metric;
- candidate score has weak alignment with generator-response labels;
- gains appear only under a leakage-prone proxy;
- the method changes support weights but fails the shuffled-narrative null;
- the component adds operational complexity without improving the paper story.

## Independent Verification Trigger

Use independent verifier before:

- changing a default;
- claiming production or boss-demo readiness;
- making a paper-facing result claim;
- scaling expensive OpenAI labeling;
- accepting a surprising result;
- declaring a paradigm shift.

Verifier artifact path:

## Decision

Allowed decisions:

- `run_testflight`;
- `needs_more_ideation`;
- `dead_end_before_experiment`;
- `promotion_review_required`.

Decision rationale:

