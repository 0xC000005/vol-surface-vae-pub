# 433a: Post-340 Frontier Audit

## Context

432a caught a duplicate paradigm before implementation: the selected 431a exact-likelihood
future-path flow was already run as `346a` and failed. This audit summarizes the actual
post-340 result frontier before selecting another direction.

## Result Frontier

Only one learned family repeatedly reaches the frontier:

| Family / Mechanism | Best Result | Read |
| --- | ---: | --- |
| Empirical normal-score causal memory transition FM (`340` family) | `7/11` before recent adaptation | Strong structural/local law, weak level/regime coverage. |
| Recent adaptation of same family (`377`, `385`, `392`) | `8/11` | Current frontier; preserves structure, still fails coverage/regime/distribution. |
| Proper-score fine-tunes (`391`, `398`, `410`) | `5-7/11` | Can move level/coverage but breaks conditionality or cointegration. |
| Calibration/oracle system (`403`, `405`, `407`, `426`, `427`, `429`) | `5-7/11` | Repairs some marginal symptoms but trades off structure. |
| Direct full-path FM/diffusion (`339`, `345`, `354`, `413`, `417`) | `3-4/11` | Sometimes improves level occupancy, but loses conditionality/geometry/MR. |
| Full-path exact likelihood (`346`) | `3/11` | Destroys cross-cell geometry and conditionality. |
| Transition exact likelihood (`347-351`) | `3-5/11` | Restores some geometry but remains below frontier. |
| Diffusion or alternate sampler core (`352`) | `3/11` | Rotating the generative sampler does not solve the frontier. |
| Student-forced / off-policy transition repair (`419`) | `6/11` | Acts like a width actuator and worsens level occupancy. |
| Joint transition-path / source-noise variants (`421`, `423`) | `4-6/11` | Adds path coupling/noise but breaks structure or overbroadens. |

## Mechanism Read

The evidence is now consistent:

1. The only robust learned object is the causal transition law in empirical normal-score
   space.
2. Models that learn the whole path directly tend to lose the cross-cell and
   mean-reversion structure.
3. Post-hoc or oracle correction can repair marginals only by damaging that structure.
4. The remaining failed suites are not a missing architecture block; they are a conflict
   between weakly identified future level/regime behavior and the structural law the
   model can actually learn.

The historical regime audits support this: validation history vol-of-vol was weakly
related to realized future movement, and earlier oracle checks did not strongly support a
hard turbulent/calm width expansion rule on this split.

## Decision

Do not select another architecture until the objective is reframed.

The next principled step is not another model implementation. It is a product/test
feasibility decision:

- either accept `392a` as the learned conditional-law frontier and report it honestly at
  `8/11`;
- or define a separately reported risk-policy layer with explicit calibration, knowing
  it is not a learned conditional law;
- or change the suite/product target if `11/11` is intended to certify a learned model
  rather than a calibrated risk system.

If the loop must keep searching for `11/11`, the only honest next experiment is a
feasibility upper-bound diagnostic, not a new learned architecture.

