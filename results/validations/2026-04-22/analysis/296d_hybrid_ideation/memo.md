## 296d hybrid ideation

### Context
`296b` and `296c` validated the hybrid decomposition:
- frozen `277d` backbone
- zero-mean coarse shell
- paired symmetric sampling

But the scale-allocation branch is now near a local cap.

`296c` tied the `5/11` frontier while leaving the same narrow misses:
- weak h1 coverage
- weak regime-sensitive width
- dead level KS
- weak pathwise jump realism

The live question is no longer scale geometry. It is whether the shell support is too
Gaussian and too smooth in the tails.

### Why the next move should be a support change
The `296c` shell already has:
- broad overall coverage (`0.882`)
- strong calibration (`0.031`)
- preserved structure from `277d`

So the failure is not "more width everywhere".
It is "wrong local mass allocation":
- not enough early-horizon tail mass
- not enough regime-sensitive tail expression
- not enough jump-size mass

That points to the shell support law, not another scale factor.

### Candidate options
1. **Heavy-tailed zero-mean coarse shell**
- keep `296c` geometry
- replace Gaussian control law with Student-t control law
- optionally learn one shared query-conditioned degrees-of-freedom scalar

2. **Short-horizon objective reweighting**
- keep Gaussian shell
- overweight h1 or jump-like residual controls in the loss

3. **Sparse event shell**
- keep the backbone
- replace smooth coarse controls with a small number of event pulses

### Recommendation
Choose **Option 1**.

Reason:
- it is the cleanest first-principles change
- it directly targets tail allocation and jump mass
- it does not introduce heuristic suite-specific weighting
- it stays inside the validated zero-mean shell geometry
- the repo already has strong Student-t precedent from the `169` line

### 296d-v0
Keep fixed:
- frozen `277d` backbone
- zero-mean coarse shell
- paired symmetric sampling
- `296c` scale geometry

Change only:
- coarse residual controls follow a Student-t law instead of a Gaussian law
- add one shared query-conditioned degrees-of-freedom scalar for the whole future

Expected read:
- if h1 coverage and max-jump KS improve without giving back MR / cointegration /
  cross-cell structure, the hybrid line is still alive
- if they do not, then the issue is deeper than shell support tails and the hybrid
  line should move toward a broader paradigm review
