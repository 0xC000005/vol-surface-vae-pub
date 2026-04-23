## 296f hybrid ideation

### Context
`296e` validated the multiresolution shell direction:
- h1 coverage improved sharply
- regime-sensitive width improved
- jump scale improved

But it also showed the fast shell was too unconstrained:
- overall coverage overexpanded
- calibration worsened
- cross-cell structure weakened

So the next move should not add more expressivity.
It should **factorize** shell behavior into:
- total shell budget
- temporal redistribution of that budget

### Candidate options
1. **Budgeted multiresolution shell**
- keep the `296e` multiresolution shell basis
- predict a separate per-cell shell budget
- predict a raw multiresolution redistribution pattern
- normalize the raw pattern so its average daily shell scale matches the learned
  budget

2. **Penalty-based calibration constraint**
- keep `296e`
- add explicit regularization on overall coverage / calibration

3. **Freeze-on-top of 296c budget**
- reuse `296c` shell budget directly
- let the fast basis only reallocate that fixed budget

### Recommendation
Choose **Option 1**.

Reason:
- it is architectural, not suite-specific
- it directly encodes the mechanism the comparison exposed
- it stays generalizable to longer horizons and other factor panels
- it avoids depending on a second frozen shell model

### 296f-v0
Keep fixed:
- frozen `277d` backbone
- zero-mean Gaussian shell law
- multiresolution knot basis from `296e`
- paired symmetric sampling

Change only:
- add an explicit per-cell shell budget head
- predict raw multiresolution knot scales separately
- renormalize the raw shell so the interpolated daily shell scale matches the learned
  budget on average

Expected read:
- if `296f` keeps the local gains from `296e` while pulling overall coverage and
  calibration back toward `296c`, the hybrid line stays alive
- if not, then even constrained multiresolution shell geometry is near a local cap
