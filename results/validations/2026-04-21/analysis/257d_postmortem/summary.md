# 257d Postmortem

## Result

`257d` scored `2/11` on the common full 11-suite.

Passes:

- `surface`
- `block_ar`

So the total suite score stayed flat relative to `257c`.

## Core Finding

The structure-preserving objective did **not** rescue long-run structure.

Instead, it pushed the `257` family into an over-coupled common-mode regime:

- correlation rose too high
- rank collapsed further
- cointegration worsened
- aggregate MR became much too strong
- jump realism deteriorated sharply

## What Improved

Relative to `257c`:

- coverage90: `0.332 -> 0.343`
- h30 coverage90: `0.199 -> 0.234`
- calibration error: `0.341 -> 0.335`
- ACF correlation: `0.841 -> 0.930`
- level KS pass: `1 -> 3`
- turb/calm ratio: `0.991 -> 1.013` (still far from passing)

So `257d` did not kill stochasticity. It slightly improved spread, calibration, and
some low-frequency level behavior.

## What Worsened

The long-run shared-structure story got worse:

- `corr_ratio`: `0.792 -> 1.988`
- `rank_ratio`: `0.487 -> 0.286`
- `cointegration_ratio`: `0.234 -> 0.119`
- `mr_ratio`: `1.105 -> 2.774`
- `change KS pass`: `7 -> 4`
- `max-jump KS`: `0.440 -> 0.931`
- `kurtosis_ratio`: `0.407 -> 0.208`

This is not a small regression. It is a regime shift toward overly strong common-mode
coupling and distorted temporal law.

## Mechanistic Read

`257d` preserved and even increased sample identity:

- `post_vs_prior_mean_mae`: `0.0102 -> 0.0342`
- `sample_std_mean`: `0.0086 -> 0.0098`
- `std_to_mean_error_ratio`: `0.153 -> 0.194`
- `kl_mean`: `0.0010 -> 0.0078`

So the failure is **not** that the latent channel collapsed.

The failure is that the new covariance/spectrum anchors on the ensemble mean were too
coarse. They encouraged a stronger shared mode, but did not preserve the right
cross-sectional flexibility or error-correction structure. The result is:

- too much common coupling
- too little effective rank
- too much aggregate reversion

The best-checkpoint training metrics support this interpretation:

- `change_corr_loss` became very small
- `level_corr_loss` remained materially non-zero
- yet suite-level rank / cointegration still worsened

So matching these generic covariance targets inside the current decoder family is not
enough.

## Decision

`257d` is the clean failure that likely caps the `257` family.

`257c` proved the family was partly objective-limited.
`257d` then tested the best objective-only follow-up and still failed to improve the
frontier, while over-coupling the long-run structure.

The next principled step should therefore be a **paradigm shift** to `258a`, not more
objective stacking inside `257`.
