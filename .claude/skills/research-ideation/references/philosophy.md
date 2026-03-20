# Research Philosophy — Principles That Guide Hypothesis Formation

These are not decorative quotes. They are decision-making tools. When forming or evaluating
a hypothesis, explicitly apply the relevant principle and state which one you're using.

## The Hamming Test (Richard Hamming, 1986)

"What are the important problems in your field? Why aren't you working on them?"

A problem is worth working on when TWO conditions hold simultaneously:
1. **It is important** — solving it would matter
2. **You have a reasonable attack** — you can see how to make progress

An important problem without an attack is philosophy. A tractable problem that doesn't
matter is busywork. Apply this as a binary gate: if either condition fails, skip it.

## The Bitter Lesson (Richard Sutton, 2019)

General methods that leverage computation (search + learning) ultimately dominate
domain-specific approaches. Human-engineered knowledge helps short-term but plateaus.

**For hypothesis formation**: Prefer solutions that learn from data over solutions that
encode human knowledge. If you find yourself writing per-cell constants, lookup tables,
or domain-specific heuristics, you are violating the Bitter Lesson. The right question
is: "What architecture would LEARN this automatically?"

## Independent Thinking First (Geoffrey Hinton)

"Reading rots the mind." Figure out how YOU would solve the problem first. Write down
your approach. THEN read how others solved it.

**For hypothesis formation**: Before searching arxiv or literature, spend time reasoning
from your own evidence. The gap between your naive approach and the published state of
the art reveals where your understanding is miscalibrated. This gap IS the insight.

## Falsificationism (Karl Popper)

You can never prove a hypothesis correct. You can only prove it wrong. A hypothesis that
cannot be falsified is not scientific — it's a wish.

**For hypothesis formation**: Every hypothesis needs a SPECIFIC falsification test — an
experiment whose outcome would KILL the hypothesis. "If X happens, this idea is dead."
Without this, you're not doing science, you're doing hope.

The corollary: a failed experiment is MORE informative than a successful one. Success
tells you "this works but I don't know why." Failure tells you "my model of reality was
wrong in THIS specific way."

## Research Taste (Neel Nanda, 2025)

Research taste is learned like a neural network — starts poorly initialized, improves
with diverse training data (research experiences). Three domains:
1. **Exploration** — noticing interesting anomalies
2. **Understanding** — designing discriminative experiments
3. **Distillation** — identifying the best narrative

After EVERY experiment, ask three questions:
1. "Was my prediction correct?" (calibration)
2. "What would I do differently?" (counterfactual)
3. "What is the most interesting thing about this result?" (pattern recognition)

## Incremental Complexity (Andrej Karpathy, 2019)

Start with the simplest possible thing that could work. Verify it. Then add one thing
at a time. Each addition should be independently testable.

**For hypothesis formation**: If your hypothesis requires 3 simultaneous changes to test,
you've failed. Decompose it. Each change should be testable alone. If component A works
and component B works but A+B fails, you've learned something profound about interaction
effects. If you test A+B+C together and it fails, you've learned nothing.

## Cross-Pollination (John Schulman + Nova framework)

Work on two things simultaneously for natural cross-pollination. Deliberately search for
solutions to your problem in unrelated fields.

**For hypothesis formation**: Ask "Who else has this problem but calls it something
different?" Time series correlation in finance = spatial correlation in weather = multi-gene
co-expression in biology. The solution may already exist under a different name.

## The Garbage Can (Ethan Mollick, 2025)

Innovation often happens when unsolved problems collide with available techniques. Maintain
two running lists:
1. **Unsolved problems** — things you can't make work
2. **Interesting techniques** — solutions looking for problems

Periodically scan both lists for unexpected matches.

## Problem-Solver vs Problem-Creator (Michael Nielsen, 2004)

Two research styles:
- **Problem-solver**: Works on well-posed technical problems. Produces solid incremental work.
- **Problem-creator**: Asks interesting NEW questions. Technically simpler papers but
  profound reframing. Rarer and often more impactful.

**For hypothesis formation**: Periodically ask "Am I being a problem-solver or a
problem-creator right now?" If all your hypotheses are "how to improve metric X,"
you're solving. Step back and ask "Is metric X even the right thing to measure?"

## Breadth Predicts Disruption (Park et al., Nature 2023)

Analysis of 45 million papers: disruptive work has declined 91-100% since 1945.
Root cause: scientists draw on NARROWER knowledge. The most disruptive papers cite
unusual combinations of references from diverse fields.

**For hypothesis formation**: If all your ideas come from the same 10 papers, your
work will be incremental. Deliberately read outside your field. Budget time for
unrelated reading. The Nova framework operationalizes this.

## LLM Ideas: Novel but Fragile (Si et al., ICLR 2025)

LLM-generated ideas are statistically MORE novel than human expert ideas (p<0.05).
But after 100+ hours of execution, the novelty advantage EVAPORATES — LLM ideas
score worse than human ideas on ALL metrics after implementation.

**For hypothesis formation**: When an LLM (including this skill) generates a hypothesis,
treat it as a CANDIDATE that requires a cheap feasibility probe before commitment.
The 2-hour feasibility checkpoint in the hypothesis template exists because of this
finding. Never commit to a full implementation based on theoretical appeal alone.
