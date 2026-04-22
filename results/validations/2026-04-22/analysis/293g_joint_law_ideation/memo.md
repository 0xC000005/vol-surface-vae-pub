## 293g ideation

### Context
The `293d/293e/293f` bracket established that the explicit support-object branch is near a local cap **as a representation search**.

What is alive:
- coverage
- change KS
- cross-cell structure
- some cointegration support

What remains dead:
- level KS
- regime width timing
- jump realism
- strong MR support

And the `293d/293e/293f` comparison makes the likely reason clearer:
- the scaffold is informative
- but the daily decoder is still treating it as soft context
- it is not forced to model the future as a refinement around that scaffold

### Decision
Next step: `293g-v0`

Keep:
- the `293d` monolithic coarse support object
- the same fixed-horizon daily joint-token decoder architecture

Change:
- the daily decoder will predict a **residual next-change law around the scaffold**, not the absolute next-change law

### Mechanism
1. Encode history.
2. Predict the monolithic coarse support code as in `293d`.
3. Decode it into a coarse knot panel and interpolate a daily scaffold.
4. Convert that scaffold into a daily transformed-change baseline.
5. Train the daily token head on the residual transformed change:
   - `target_change_coord - scaffold_change_coord`
6. At sampling time:
   - sample residual token
   - add it to the scaffold change coordinate
   - invert back to the next normalized level

### Why this is the right next move
This is the narrowest direct test of the current diagnosis.

It does not change:
- the local joint-law family
- the support representation
- the history encoder

It changes only:
- how the scaffold is used by the daily law

So if `293g` helps, we learn:
- the scaffold was good enough
- the support-use formulation was the blocker

If `293g` fails, we learn:
- the support branch is closer to capped even when used structurally

### Why this is clean
No:
- new latent branch
- second support object
- retrieval bank
- AR rollout reset
- custom evaluator logic

Just:
- one coarse scaffold
- one residual daily law around it

### Kill criteria
`293g` is alive only if it improves at least one dead path-shape suite:
- level KS
- regime differentiation
- jump realism
- MR support

while preserving the existing live suites:
- coverage
- change KS
- cross-cell structure
- cointegration robustness

If it only preserves the current `293d/293f` pattern, then the support branch should be treated as close to capped beyond local tweaks.
