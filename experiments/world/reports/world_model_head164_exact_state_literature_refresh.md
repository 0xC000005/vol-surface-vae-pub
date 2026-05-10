# World Model HEAD164: Exact-State Literature Refresh

Date: 2026-05-10

## Iteration Type

`research_ideation`

## Objective Family

Literature refresh for exact-state retention after demoted context-to-target
routes; no model change.

## Question

After row-level and surface-local context-to-target diagnostics failed, what is
the principled boundary for trying to fix exact-state retention without adding
ad hoc reconstruction or knob tuning?

## Sources

- I-JEPA: <https://huggingface.co/papers/2301.08243>
- V-JEPA: <https://huggingface.co/papers/2404.08471>
- VICReg: <https://huggingface.co/papers/2105.04906>
- MAE: <https://huggingface.co/papers/2111.06377>

## Literature Read

I-JEPA supports context-to-target prediction in representation space, but its
own framing emphasizes target-block and context design: targets need sufficient
semantic scale and context must be informative. That maps poorly onto our failed
surface-local smoke because the target latent was mostly token/factor geometry,
not market-state variation.

V-JEPA reinforces the same point for video: feature prediction can work as a
standalone objective, but the learned target feature surface must be meaningful
enough for downstream frozen-backbone tasks. A predictor matching a weak or
geometry-dominated target latent does not certify the representation.

VICReg/Barlow-style objectives support explicit variance/covariance/redundancy
control for same-view or multiview embeddings, which remains relevant for
collapse prevention. They do not solve the choice of target representation.

MAE is the canonical reconstruction-side alternative: it masks input patches and
reconstructs pixel/value space through a decoder. That can retain detail, but it
is a different objective family from JEPA feature prediction and should not be
introduced as a small auxiliary patch under the JEPA label.

## Local Implication

The current blocker is not simply collapse, target coverage, or insufficient
mask aggression. It is that the learned target surfaces tested so far do not
carry enough exact market-state variation to beat raw current-state features.

Therefore:

- do not add a raw-value reconstruction auxiliary loss as a quiet patch;
- do not tune hidden size, predictor depth, EMA, epochs, mask coverage, or
  Barlow weights on demoted context-to-target branches;
- if a new JEPA route is proposed, it must first define and test a target latent
  surface whose intrinsic nearest-neighbor structure reflects market-state
  variation rather than token/factor identity;
- if a reconstruction route is proposed, label it as a separate MAE-style
  diagnostic and evaluate it against the Part 1 gate separately.

## Decision

No implementation is authorized by this refresh. The next principled model
design, if any, must pass a target-latent state-variation gate before predictor
training. Part 1 remains `DO_NOT_PROMOTE`; Part B remains blocked.
