# Public Paper Cleanup Plan

Date: 2026-05-30

## Objective

Revise the natural-language conditioned heterogeneous multivariate scenario
generation paper so it reads as a concise public technical paper rather than an
internal autoresearch report.

The scientific claim remains unchanged: professional risk-manager narratives
are converted into auditable support-grounded conditions for a frozen SNI
scenario generator. The paper should not overclaim free-form prompt-following,
but it should clearly show that narrative changes support provenance and
selected risk distributions relative to a same-start baseline.

## Accepted Review Findings

1. The core method is coherent and supported by the current evidence.
2. The manuscript is too long for the intended short technical-paper style.
3. The abstract includes too much implementation state and internal product
   wording.
4. Related work should be tighter: LLM limits in numerical finance, stochastic
   scenario generation, multimodal/text-latent alignment, and provenance.
5. Method and results contain internal autoresearch language that should be
   replaced by public-facing terms.
6. Provider/cost details are implementation notes, not core scientific claims.
7. The Safe-haven Gold discussion is valuable because it clarifies that
   grounding validates the conditioning prefix, not the terminal future sign.
8. The paper should keep a compact main story and push detailed workflow,
   metric explanations, casebook tables, and implementation details to the
   appendix.

## Todo

1. Rewrite the abstract as a short problem-method-evidence-conclusion paragraph.
2. Simplify the introduction and shrink the contribution list.
3. Tighten related work around the actual gap filled by the paper.
4. Keep the method focused on the public algorithmic contract; move excess
   bridge/provider details to appendix.
5. Reframe results around the main evidence: grounding reliability, historical
   backtest, fixed-start conditionality, top3/90 posterior view, and portfolio
   readout.
6. Replace internal language:
   - product-facing candidate -> default reported configuration
   - promotion gate -> evaluation criterion
   - diagnostic -> supplementary analysis or ablation, where appropriate
   - Codex route -> implementation note or appendix provider comparison
7. Ensure all paper-facing narratives remain professional risk-manager
   narratives and that examples match the artifacts used for figures/tables.
8. Rebuild the PDF and run citation, label, stale-language, and LaTeX warning
   checks.

## Acceptance Criteria

The revision is acceptable when:

- the main paper reads as a public method/results paper, not a project log;
- every main claim has a corresponding table, figure, or clearly cited source;
- volatile provider/pricing details are out of the main scientific argument;
- the top3/90 default is described as the default reported configuration under
  an explicit calibration-versus-conditionality tradeoff;
- the Safe-haven Gold discussion is retained but framed as prefix support versus
  terminal response, not a failure of grounding;
- the paper compiles without undefined references or citations.
