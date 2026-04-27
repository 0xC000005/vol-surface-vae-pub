# 627a Native Joint-Panel Scenario Quality Audit

## Context

The IV-only full 11-suite does not answer whether a native joint38 model is usable
for a risk manager who wants IV plus anchor-factor scenarios in the same scenario
path. 627a adds a generic joint-panel audit for the native one-model candidates
610a, 612a, and 625a. It checks finite generation, anchor-factor daily-change
KS, anchor-factor tail-change ratios, factor-factor correlation preservation, and
IV-factor correlation preservation.

## Results

| model | native law | factor KS mean | KS pass <0.20 | q99 ratio median | q99 pass [0.5,2.0] | factor-factor corr | factor-factor gen/gt mean abs | IV-factor corr | IV-factor gen/gt mean abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 610a | AR empirical-score transition | 0.133 | 11/13 | 2.365 | 2/13 | 0.565 | 0.054 / 0.225 | 0.664 | 0.075 / 0.149 |
| 612a | 610a + conditional source scale | 0.135 | 11/13 | 2.114 | 5/13 | 0.559 | 0.037 / 0.225 | 0.698 | 0.053 / 0.149 |
| 625a | AR RealNVP transition likelihood | 0.167 | 8/13 | 2.164 | 5/13 | 0.664 | 0.029 / 0.225 | 0.873 | 0.033 / 0.149 |

Artifacts:

- `results/autoresearch/627a_joint_panel_audit/610a_joint_panel.md`
- `results/autoresearch/627a_joint_panel_audit/612a_joint_panel.md`
- `results/autoresearch/627a_joint_panel_audit/625a_joint_panel.md`

## Mechanism Read

All three candidates are finite native joint generators, so the code path can
produce 25 IV channels plus 13 anchor factors without post-hoc deck gluing.
However, none is yet a risk-manager-deployable joint probability law.

The common failure is dependence attenuation. The generated factor-factor and
IV-factor correlation matrices have some correct rank/shape signal, especially
625a on IV-factor matrix correlation, but their absolute correlation magnitudes
are far too small. For example, factor-factor mean absolute correlation falls
from 0.225 in validation truth to 0.029-0.054 in generated paths.

Anchor-factor marginal changes are only partially acceptable. 610a/612a match
daily-change KS for 11/13 factors, but tail-change ratios are too wide. 625a
does not fix this: it improves correlation shape but worsens factor marginal KS
and still has wide tails.

This points to a data-coordinate issue rather than a need for another special
architecture knob. The current native joint family trains and samples in state
levels after empirical-score normalization. That is clean for a generic state
panel, but financial scenarios are naturally laws over encoded daily changes,
with levels reconstructed by integration. The unified panel builder already
constructs `future_increment`; the main training scripts mostly ignore it.

## Decision

Do not declare 610a/612a/625a joint deployable. 610a remains the cleanest native
joint base, 612a remains the highest-count but less clean source-scale variant,
and 625a is a useful likelihood falsifier rather than the next base.

The next principled experiment is a coordinate reset: train a native joint AR
transition law on unified encoded daily changes for all 38 channels, then decode
and integrate those changes back to levels for the same IV and joint audits. This
keeps one model and one transition, avoids hand-engineered IV/factor branches,
and directly tests whether the prior failures came from asking the model to
learn level paths instead of the return/difference process that risk scenarios
actually require.
