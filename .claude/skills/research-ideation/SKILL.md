---
name: research-ideation
description: Generates principled research hypotheses by synthesizing accumulated experimental evidence, applying structured ideation frameworks (TRIZ, Zwicky Box, Hamming test), and grounding in scientific literature. Use this skill when the user wants new research directions, feels stuck at a performance plateau, asks "what should I try next", "where should I focus", "generate hypotheses", "research compass", "new ideas", "what are the fundamental bottlenecks", or needs to step back from incremental improvements and think about principled architectural changes. Also use when the user says "I've exhausted this direction" or "we need a bigger change". Do NOT use for running experiments (use autoresearch) or logging results (use research-log).
---

# Research Ideation Skill

Generate principled, independently-testable research hypotheses grounded in accumulated
evidence and scientific philosophy. This is NOT brainstorming — it is structured scientific
reasoning that treats hypothesis formation as seriously as experimentation.

The philosophy is the most important part. Every decision — what to investigate, what to
propose, what to abandon — should be traceable to a specific principle. If you can't name
which principle justifies a choice, the choice is ad hoc.

## Core Principles (read `references/philosophy.md` for full details)

These are decision-making tools, not decorations. Apply them explicitly throughout.

- **Hamming**: Only pursue what is both important AND attackable. Binary gate.
- **Popper**: We can only falsify, never verify. A failed experiment teaches more than
  a successful one. Every hypothesis needs a kill condition.
- **Hinton**: Think independently BEFORE reading literature. The gap between your naive
  solution and the published one IS the insight.
- **Sutton (Bitter Lesson)**: General methods that learn > domain heuristics. If you're
  writing per-cell constants, you're violating this.
- **Karpathy**: Each change must be independently testable. No stacked dependencies.
  If A+B+C fails, you learn nothing. Test A alone first.
- **Nanda**: After every experiment, ask: "Was my prediction correct? What would I do
  differently? What's most interesting about this result?"
- **Schulman**: Work on two things for cross-pollination. Read PhD theses, not just papers.

## The Process: Four Phases

### Phase 1: Evidence Synthesis (Parallel Agents)

Ground yourself in what is KNOWN. Launch 3 parallel agents to cover breadth efficiently:

**Agent 1: "Current state and proven root causes"**
- QMD search: `lex: "root cause" "bottleneck" "proven"` + `vec: fundamental architectural
  limits preventing progress`
- QMD search: `lex: "ceiling" "plateau" "exhausted"` + `vec: what has been tried enough
  to confidently abandon`
- Compile: what do we KNOW is true from experiments?

**Agent 2: "Failure landscape and contradictions"**
- QMD search: `lex: FAIL regressed worse` + `vec: experiments that failed and why`
- QMD search: `vec: results that were surprising or contradicted expectations`
- Compile: what patterns emerge from failures? Where do results conflict?

**Agent 3: "What worked and why we don't understand it"**
- QMD search: `lex: PASS improved better` + `vec: experiments that succeeded unexpectedly`
- QMD search: `vec: things that work but we cannot explain why`
- Compile: where is our understanding weakest? (Popper: success without understanding
  is fragile — the next change might break it for unknown reasons)

After agents return, synthesize into the **Evidence Summary**:
1. **Proven root causes** — things we KNOW are true from experiments
2. **Exhausted directions** — things tried enough to confidently abandon, with WHY
3. **Contradictions** — results that conflict with each other or with theory
4. **Fragile successes** — things that work but we don't understand why
5. **Open questions** — things we don't understand yet

**Step 1e: Verify Key Numbers (Trust but Verify)**

Research log entries are snapshots — they capture what was true at the time of writing.
Before building hypotheses on specific numbers:

- **Cross-reference**: If the log says "Exp 120b achieved kurtosis 0.957," check whether
  multiple entries agree. If only one entry mentions it, treat it as unverified.
- **Check the source**: For critical metrics, read the actual results file:
  ```
  mcp__qmd__get({ file: "research/results/block-ar/[experiment]/summary.json" })
  ```
  Or use Grep to find the results directory and Read the summary.json directly.
- **Sanity check**: Does the number make sense given what we know? If an experiment
  claims kurtosis 0.957 but similar experiments get 0.4-0.6, investigate the discrepancy
  before accepting it.
- **Recency**: Was this metric from the current model version or an older one? Check dates.
- **When in doubt, rerun**: If a hypothesis hinges on a specific metric being true, and
  you can't verify it from files, run the evaluation again. 5 minutes of validation is
  cheaper than 8 hours of implementation based on a stale number.

Present the evidence summary to the user. Do NOT proceed until they confirm accuracy.
Wrong evidence poisons everything downstream.

### Phase 2: Hypothesis Generation (Structured, Multi-Agent)

**Step 2a: Independent Reasoning (Hinton principle)**

Before touching any literature, reason from the evidence alone:
- What does the PATTERN of failures tell us? Not individual failures — the pattern.
- If you could change only ONE thing, what would have the highest information value?
  (Not highest expected performance — highest LEARNING regardless of outcome)
- What assumption, if wrong, would change everything? (TRIZ contradiction seed)
- Apply Popper: for each idea, immediately ask "what would kill this?"

Write 2-3 raw hypotheses. These are your independent baselines.

**Step 2b: Framework Application**

Read `references/frameworks.md` and apply EACH framework. The frameworks are complementary —
TRIZ finds contradictions, Zwicky finds unexplored space, Chain of Ideas predicts evolution,
Garbage Can finds unexpected matches.

1. **TRIZ**: For each contradiction from Phase 1, ask: "Is there a design that DECOUPLES
   these objectives so both improve?" Don't optimize the trade-off — resolve it.
2. **Zwicky Box**: What dimension × option combinations are unexplored but viable?
3. **Chain of Ideas**: What's the temporal evolution? What's the logical next link?
4. **Garbage Can**: Scan unsolved problems list against available techniques. Surprising matches?

**Step 2c: Literature Search (Parallel Agents — structured by root cause)**

Launch 4-6 parallel agents. Each agent uses the full tool stack (WebSearch + arxiv MCP +
PaperQA2 via Bash). Structure by ROOT CAUSE, not by tool:

```
Agent A: "[Root cause 1] — principled solutions"
  - mcp__arxiv__search_papers: papers addressing this specific mechanism
  - WebSearch: recent blog posts, conference talks, practitioner discussions
  - Bash: pqa ask "Has [mechanism] been solved in [adjacent field]? What worked?"

Agent B: "[Root cause 2] — principled solutions"
  - Same tool mix, different root cause

Agent C: "[Root cause 3] — principled solutions"
  - Same pattern

Agent D: "Cross-domain analogues (Nova principle)"
  - Search OUTSIDE the field: weather forecasting, drug discovery, robotics, biology
  - Ask: "Who else has this problem but calls it something different?"
  - Translate the problem into each domain's language before searching

Agent E: "Validate or challenge hypotheses from Step 2a"
  - Search for evidence these approaches have been tried before
  - PRIORITIZE failure reports over success reports
  - PaperQA2: "What are known failure modes of [approach X]?"

Agent F (optional): "Foundational theory"
  - Search for mathematical guarantees, information-theoretic bounds, impossibility results
  - If a hypothesis violates an impossibility result, kill it now rather than after implementation
```

**Tool usage within each agent:**
- `mcp__arxiv__search_papers`: Find relevant papers by title/abstract
- `mcp__arxiv__download_paper` + `mcp__arxiv__read_paper`: For the most promising 1-2 papers
  per agent, DOWNLOAD and READ the actual paper — don't rely on abstracts alone. Abstracts
  are marketing; the methods section has the truth.
- `WebSearch`: Recent blog posts, conference talks, practitioner discussions, HN threads
- `Bash: pqa ask "question"`: Deep Q&A with citations when you need synthesis across papers
- `mcp__claude_ai_Hugging_Face__paper_search`: ML-specific papers and model cards
- Schulman's advice: search for PhD theses, not just papers — they contain full context
  including dead ends and motivation that papers omit

After agents return, synthesize:
- Where does literature AGREE with your independent reasoning? (validation)
- Where does it DISAGREE? (most interesting — investigate the gap)
- What solutions exist you didn't think of? (new candidates)
- What failures are documented? (learn from others' mistakes before repeating them)

**Step 2d: Adversarial Critique (Diverse Critics)**

Research shows diverse critics improve ideas more than diverse generators (SIGDIAL 2025).
Before presenting hypotheses, attack each one from 3 adversarial perspectives:

1. **The Skeptical Statistician**: "Your sample size is N experiments. How confident are
   you that this pattern is real and not noise? What's the base rate of this kind of
   improvement happening by chance?"

2. **The Systems Engineer**: "This requires changing X lines of code across Y files.
   What's the blast radius? How many things can go wrong in implementation? Can you
   decompose this into staged checkpoints where each stage is independently valuable?"

3. **The Domain Expert**: "Does this actually make physical/mathematical sense? Is there
   a theoretical reason this mechanism should work, or are you just pattern-matching on
   a few experiments?"

Kill or revise hypotheses that can't survive all three critics.

**Step 2e: Filter and Rank**

Apply these filters IN ORDER:

1. **Hamming Gate** (binary): Important AND attackable? Kill if either fails.
2. **Popper Gate** (binary): Specific falsification test? If you can't state what kills it, kill it.
3. **Independence**: Stands alone? No stacked dependencies? Decompose if needed.
4. **Bitter Lesson**: Learns from data? Flag any domain heuristics.
5. **Staged Checkpoints**: Can the implementation be broken into milestones where each
   milestone produces independent value? A hypothesis that requires a 12-hour monolithic
   implementation before you get ANY signal is risky. Prefer hypotheses with early signal.
6. **Information Value**: Even if this FAILS, do we learn something fundamental?
   Rank higher: hypotheses where failure is almost as informative as success.

### Phase 3: Research Compass (Interactive)

Present surviving hypotheses (typically 3-5) in this template:

```markdown
## Hypothesis N: [One-line description]

**Evidence chain**: What we KNOW from experiments that leads here.
[Cite specific experiment IDs and results]

**Principled argument**: WHY this should work, theoretically.
[Not "let's try this" — explain the mechanism. Name which philosophy principle supports it.]

**The bet**: What specifically to implement.
[Concrete enough to start coding]

**Staged checkpoints**:
1. [2-hour feasibility probe — smallest test that gives signal]
2. [4-hour minimal implementation — tests core mechanism in isolation]
3. [Full implementation — only proceed if checkpoints 1-2 succeed]
[Each checkpoint is independently valuable — if you stop at checkpoint 1, you still learned something]

**Falsification test**: What would KILL this hypothesis at each stage.
[Stage 1: "If X, abort." Stage 2: "If Y, abort." Stage 3: "If Z, the hypothesis is dead."]

**Independence**: What this does NOT depend on.
[Confirm it stands alone. Explicitly list what other hypotheses it is NOT coupled to.]

**If it fails**: What we would learn from failure.
[The Nanda question: what's most interesting about a negative result?]
[How does failure update our model of the system?]

**Effort**: [Stage 1: Xh] [Stage 2: Yh] [Stage 3: Zh]
```

Discuss with the user. They may refine, reject, prioritize, or request deeper literature
dives on specific hypotheses.

### Phase 4: Save the Research Compass

Append to research log (use research-log skill):

```markdown
## YYYY-MM-DD: Research Compass — [Theme]

### Philosophy Applied
[Which principles were most relevant and how they shaped the hypotheses]

### Evidence Summary
[From Phase 1 — proven root causes, exhausted directions, contradictions, open questions]

### Active Hypotheses (ranked by information value)
[The 3-5 hypotheses in template format]

### Exhausted Directions
[What we've confidently ruled out, with the specific mechanism of WHY]

### Open Questions
[What we still don't understand]

### Garbage Can Lists
**Unsolved problems**: [updated list]
**Available techniques**: [updated list]
```

Re-index: `qmd update --collection research && qmd embed`

## After an Experiment Fails

This is the MOST IMPORTANT part. The quality of research is determined not by how you
handle success, but by how you handle failure.

**The Autoresearch Trap**: Try X → X fails → try Y → Y fails → try Z. This is metric
chasing, not science. Each failure should UPDATE YOUR UNDERSTANDING before you move on.

When a hypothesis fails:

1. **Do NOT move to the next hypothesis.** Sit with the failure.

2. **Apply Nanda's three questions**:
   - Was my prediction correct? Where EXACTLY did it diverge from expectation?
   - What would I do differently if I could rerun this?
   - What is the MOST INTERESTING thing about this failure?

3. **Check falsification cleanliness.** Did the experiment test what you thought?
   Or did an implementation bug, hyperparameter, or confound muddy the result?
   A dirty falsification is not a falsification — it's wasted time.

4. **Update the evidence summary.** The failure is new evidence. Does it:
   - Confirm a root cause you suspected?
   - Reveal a NEW root cause you didn't know about?
   - Challenge your model of the system?

5. **Apply Popper explicitly.** You can only move on when you can articulate:
   "This failed because [specific mechanism], and this mechanism is FUNDAMENTAL to the
   approach, not an implementation detail that could be fixed."
   If you can't say this, you haven't learned enough from the failure yet.

6. **Update the research compass.** Remove the falsified hypothesis. Check if the failure
   changes the ranking of remaining hypotheses. Update the Garbage Can lists.

## After an Experiment Succeeds

Success is actually HARDER to handle well than failure. Popper warns: you can never verify,
only falsify. When something works:

1. **Ask WHY it worked.** Can you explain the mechanism? If not, the success is fragile.
2. **Identify what would BREAK it.** Design a stress test that targets the weakest link.
3. **Check if the success is Bitter-Lesson-compatible.** Did the model learn this, or did
   you engineer it? Engineered success won't generalize.
4. **Update the evidence summary.** What does this success teach us about the system?

## Anti-Patterns

- **Metric chasing**: "Score went from 66.3 to 66.5" is not science. Ask WHY.
- **Stacking patches**: "A worked, B worked, let's try A+B" without understanding why
  either worked. If A+B fails, you learn nothing.
- **Premature abandonment**: Regression after 2 hours → move on. The regression is data.
- **Literature worship**: Paper says X works → try X. WHY does X work? Does the mechanism
  apply here?
- **Monolithic implementation**: 12 hours of coding before any signal. Use staged checkpoints.
- **Sunk cost persistence**: Continuing after clean falsification because of invested time.
- **Success complacency**: Something worked → ship it. Without understanding WHY, the next
  change might silently break it.
