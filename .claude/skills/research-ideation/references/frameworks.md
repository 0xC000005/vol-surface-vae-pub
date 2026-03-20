# Structured Ideation Frameworks

## TRIZ — Resolve Contradictions, Don't Optimize Trade-offs

When you face a trade-off ("improving A hurts B"), TRIZ says the right move is NOT to
find the optimal balance. The right move is to find an architecture that DECOUPLES A and B
so both can improve independently.

**How to apply:**
1. Identify the contradiction (e.g., "CRPS optimizes marginals but destroys correlation")
2. Ask: "Is there a design where marginals and correlation are handled by SEPARATE mechanisms?"
3. If yes, you've resolved the contradiction. If no, the trade-off is real — document why.

Example from this project: CRPS-vs-correlation is a TRIZ contradiction. CW-Gen resolves it
by giving marginals to CRPS and correlation to a separate JMCE module. TACTiS-2 resolves it
by training in two stages (marginals first, copula second).

## Zwicky Box — Enumerate the Design Space

Break the problem into independent dimensions. List all options for each. Systematically
identify unexplored but viable combinations.

**Dimensions for this project:**
- Encoder: {GRU, Mamba, S5, Transformer, VIB}
- Noise injection: {concat to MLP, skip path, FiLM/AdaGN, conditional LayerNorm, weight perturbation}
- Loss function: {afCRPS, afCRPS+variogram, afCRPS+patched ES, energy score, signature MMD}
- Decoder: {MLP, attention-based, flow matching, Neural SDE}
- Covariance: {implicit from CRPS, learned JMCE, copula layer, whitening transform}

The current best model occupies ONE cell: {GRU, skip path, afCRPS, MLP, implicit}.
Most of the grid is unexplored.

## Chain of Ideas — Predict the Next Link

Organize the evolution of approaches as a temporal chain:
1. MSE pretraining (worked, but encoder collapses to rank 2)
2. DDPM denoiser (anti-collapse, but slow inference)
3. afCRPS single-pass (fast, good marginals, but rank-1 correlation)
4. Loss engineering (variogram, ACF, ortho — all regressed)
5. Noise architecture (skip, noise-free MLP — marginal gains)
6. **??? — what is the logical next step?**

The chain suggests: loss engineering is exhausted, noise architecture is exhausted,
the next step is either decoder architecture or covariance separation.

## Nova — Cross-Domain Search

After generating ideas from your own domain, DELIBERATELY search 3 unrelated fields for
analogous problems:
- Weather forecasting (ECMWF faces the same CRPS correlation problem at 80+ variables)
- Drug discovery (multi-objective optimization with non-differentiable rewards)
- Robotics (policy learning with structured action spaces)

Use arxiv MCP and PaperQA2 for this. Search with domain-translated queries.

## The Garbage Can — Collision-Based Discovery

Maintain two lists and scan for unexpected matches:

**Unsolved problems** (from the research log):
- CRPS correlation agnosticism
- MLP rank collapse
- Encoder information loss (rank 2)
- KS distribution matching (daily changes)
- Regime-specific cell response

**Available techniques** (from literature surveys):
- CW-Gen conditional whitening
- Rotation modulation (rank-preserving)
- Variogram Score
- TACTiS-2 two-stage training
- Neural SDE with FDM training
- FiLM conditioning
- Patched Energy Score

Scan: which technique addresses which problem? Are there surprising matches?

## Boden's Creativity Check — What Kind of Idea Is This?

Three types of creativity (Margaret Boden):
1. **Combinatorial**: Novel combination of familiar ideas. "What if we combine CW-Gen with
   a Mamba encoder?" Most ML research is this. Lowest risk, lowest disruption.
2. **Exploratory**: Systematic traversal of a conceptual space. "What does the full Zwicky
   Box look like? Which cells have we never tried?" More thorough than combinatorial.
3. **Transformational**: Restructuring the space itself. "What if the loss function shouldn't
   be differentiable? What if we shouldn't be generating frames at all?" Riskiest but most
   disruptive.

After generating hypotheses, classify each one. If ALL hypotheses are combinatorial (type 1),
force yourself to generate at least one transformational (type 3) hypothesis. It might not
survive the Hamming filter, but the exercise prevents you from being stuck in incrementalism.

Ask: "What assumption am I making that, if violated, would change the entire problem?"
That's where transformational ideas live.
