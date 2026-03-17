# Theory Engine — Hypothesis Generation Protocol

## Purpose

When the theory queue is empty, this protocol generates new research directions
from accumulated evidence. It prevents the agent from either (a) repeating
exhausted approaches or (b) making random unjustified changes.

## The Synthesis Process

### Step 1: Evidence Gathering (~2 min)

Use the `research-log` skill to search for:
```
query: "exhausted approaches what hasn't been tried root cause"
query: "Suite [N] failure root cause mechanism"
query: "recent experiment results improvement regression"
```

Also read:
- `autoresearch-session/results-log.md` — this session's iterations
- `autoresearch-session/theory_queue.json` — exhausted directions and why

### Step 2: Failure Mode Analysis (~3 min)

For each failing test suite, identify:
1. **The proximate cause** — what metric fails by how much?
2. **The root cause** — what architectural mechanism creates this failure?
3. **What's been tried** — which approaches targeted this root cause?
4. **Why they failed** — what made each approach insufficient?

Build a table:
```
Suite | Proximate cause | Root cause | Tried | Why failed
S2    | worst_cell < 70%| delta ACF  | rho sweep, cell_var | ACF structural
```

### Step 3: Gap Identification (~2 min)

Look for gaps between root causes and what's been tried:
- Root cause identified but no direct fix attempted?
- Fix attempted but with wrong mechanism?
- Two partial fixes that haven't been combined?
- New evidence that invalidates a previous "exhausted" conclusion?

### Step 4: Direction Generation (~3 min)

For each gap, formulate a direction:

```json
{
  "name": "Descriptive name",
  "theory": "Based on [specific evidence from research log], the root cause of
             [suite] failure is [mechanism]. The proposed fix is [change] because
             [theoretical justification].",
  "evidence": ["Exp 99k showed decorrelation works but exposes amplitude uniformity",
               "Investigation found AR noise saturates by h~8"],
  "target": "Suite X — specific metric to improve",
  "risk": "May regress [metric] because [mechanism]",
  "novelty": "This hasn't been tried because [reason] / differs from [exhausted approach] by [difference]"
}
```

### Step 5: Confidence Ranking

Rank directions by:
- **Evidence strength** — how directly does evidence support this? (strong > weak > speculative)
- **Novelty** — how different from exhausted approaches? (must be genuinely new)
- **Risk** — how likely to cause catastrophic regression? (low > medium > high)
- **Scope** — how many suites could this affect? (multi-suite > single-suite)

Pick the highest-confidence direction first.

## Anti-Patterns

**Don't generate these kinds of directions:**
- "Try a larger model" — too vague, not theory-driven
- "Tune hyperparameters" — the research log says this is exhausted
- "Add regularization" — which regularization? For what theoretical reason?
- Anything the research log marks as "exhausted" or "proven ineffective"

**Do generate these kinds of directions:**
- "Replace AR(1) noise with learned dampening f(condition)*z + g(condition)*eps — evidence from Exp 101a shows rho=0 breaks GU but rho=0.8 causes super-diffusive spread. A learned middle ground could fix both."
- "Add spatial cross-attention in the decoder so cells can attend to each other's states — evidence from the decorrelation investigation shows the MLP processes all cells identically through shared weights."
