## 296e hybrid ideation

### Context
The `296c -> 296d` comparison narrowed the hybrid bottleneck further:
- global scale allocation tweaks are near a local cap
- global tail-heaviness tweaks are also near a local cap

What still fails is more local:
- h1 coverage
- regime-sensitive width
- pathwise jump timing

That suggests the shell is not missing "more width everywhere".
It is missing **finer early-horizon allocation geometry**.

### Candidate options
1. **Multiresolution zero-mean shell**
- keep the `296c` Gaussian shell law
- keep the zero-mean geometry
- add a small fast basis for days `1-3` or `1-5`
- keep the existing coarse knot basis for the rest of the path

2. **Objective-side horizon weighting**
- keep `296c` geometry
- reweight the loss toward early horizons or stressed windows

3. **Sample-based path scoring**
- keep shell geometry
- add a proper sampled path score on top of control NLL

### Recommendation
Choose **Option 1**.

Reason:
- it directly matches the observed miss: early-horizon allocation
- it stays architectural rather than suite-specific
- it preserves the validated hybrid split
- it remains generalizable to longer horizons as a multiresolution shell, not a
  one-off hand-tuned fix

### 296e-v0
Keep fixed:
- frozen `277d` backbone
- zero-mean shell
- paired symmetric sampling
- Gaussian shell law from `296c`

Change only:
- replace the single coarse shell basis with a **multiresolution basis**
- add a fast local basis covering the first few days
- keep the coarse basis for the rest of the horizon

Expected read:
- if h1 coverage and jump realism improve without giving back MR / cointegration /
  cross-cell structure, the hybrid line is still alive
- if not, then even targeted early-horizon shell geometry is not enough and the
  hybrid line is near a broader cap
