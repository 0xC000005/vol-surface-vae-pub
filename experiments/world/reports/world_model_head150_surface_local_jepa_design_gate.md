# World Model HEAD150: Surface-Local JEPA Design Gate

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

Design gate for a possible `token_geometry_level_context_to_target_jepa`.

## Literature Status

`canonical_jepa_translated_to_time_series_geometry`.

I-JEPA predicts representations of target blocks from visible context and uses
target representations selected from the target-encoder output, not masked
target inputs. It also conditions prediction on target position tokens. Source:
<https://arxiv.org/pdf/2301.08243>. V-JEPA supports feature prediction as the
standalone pretraining objective and evaluates frozen representations on
downstream tasks. Source: <https://arxiv.org/abs/2404.08471>.

## Local Evidence

The current failure is now specific enough:

- scaled Barlow is the best learned candidate but is `DO_NOT_PROMOTE`;
- minimal row-level context-to-target JEPA is demoted;
- the remaining exact-state gap is broad (`20/25` IV cells worse than raw), but
  concentrated in wing moneyness and edge maturities;
- the worst cell is `iv_m0_t0`;
- global row-level embeddings compress away local IV geometry.

## Proposed Design Boundary

If we attempt another JEPA branch, it must be token/geometry-level rather than
row-level:

```text
clean same market window
-> tokenize by (relative time, geometry token)
-> add geometry id, factor id/family, moneyness/maturity coord, and time embeddings
-> sample surface-local target blocks: wings, edge maturities, ATM/core strips,
   local rectangles, and factor-family blocks
-> context encoder sees visible tokens and observed/synthetic mask metadata
-> target encoder sees clean full-window tokens
-> predictor receives context tokens plus target time/geometry position tokens
-> predict target token embeddings selected from target-encoder output
-> evaluate both local token embeddings and pooled market-state embeddings
```

This is not the minimal HEAD140/144 route. It is a different architecture class.

## Guardrails

- No future targets in pretraining.
- No raw value reconstruction loss as the default route.
- No decoder or scenario-generation loss.
- No sweep over masks, EMA, hidden sizes, or predictor depth before a data
  contract and quality gate are written.
- Collapse/redundancy controls must apply to the representation surface being
  evaluated.
- The first implementation, if authorized by a later iteration, must start with
  a data object and tests for target-token identity, geometry coordinates, and
  same-window alignment.

## Falsifiers

Demote this route if the first tested token/geometry-level diagnostic:

- improves local target loss but not clean pooled/state probes;
- encodes target mask family more strongly than market state;
- remains worse than scaled Barlow on exact-IV retention and rank;
- improves wings/edges only by damaging core/ATM state;
- needs multiple knobs before showing a clean signal.

## Decision

Do not implement this immediately as a patch. This report only opens a
design-gated route. The next implementation work, if pursued, should be a TDD
data contract for token/geometry-level target blocks. Until then, scaled Barlow
remains the active learned candidate and Part B remains blocked.
