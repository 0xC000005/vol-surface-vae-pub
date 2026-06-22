"""Framework-v1 Gate [B] coherence sign-gate SCORER (Tier-1 item 4).

CPU-only, pure numpy. Validation-side only -- this module is a *scorer* that grades a
conditioned ensemble against the historical Gaussian conditional-completion target; it never
enters the conditioning path and contains no model constants or learned lookups (Bitter-Lesson
clean). The closed-form completion (mu_{-S|S} and the conditional betas) is produced upstream by
``nl_conditional_completion.conditional_completion``; this module only consumes those arrays.

Spec source: ``docs/research_protocols/nl_counterfactual_validation_framework_v1.md`` --
- Section B "Internal coherence" (revised, KEEP) -> concrete spec (1) *Conditional-completion
  check*: "narrative pins factor set S; compute mu_{-S|S} from historical 30-day moves;
  **hard gate = sign agreement on all factors with |conditional beta| above a 994a-bootstrap
  noise floor**; magnitude disagreement = flag-and-investigate, NOT auto-fail (could be learned
  tail dependence); **Red flag = material tilt on near-zero-beta factors**."
- Tier 1 item 4 "Coherence quick-check [B]: conditional-completion **sign gate** +
  **negative-control excess-tilt check**."

Pre-registered noise floor: the default ``noise_floor=0.048`` is the value frozen at
``experiments/backfill/block_ar/nl_996a_n1_tilt_training.py:114`` (``NOISE_FLOOR = 0.048``), the
994a twin-noise band floor. It is reused here as both the conditional-beta materiality threshold
(|conditional beta| must clear it for a free factor to participate in the sign gate) and the
default "material tilt" threshold for the near-zero-beta red flag.

------------------------------------------------------------------------------------------------
EXCESS-TILT PASS DIRECTION (decided from the framework text; documented per task instruction)
------------------------------------------------------------------------------------------------
``excess_tilt_check`` implements the **negative-control excess-tilt check** of Tier-1 item 4,
whose semantics are pinned by Section D item (6) negative controls: "relative null -- **no excess
tilt beyond what the unconditioned support mixture already implies**." A negative-control factor
is one that should NOT respond to the narrative; the *placebo* tilts (from null / scrambled /
off-domain narratives) trace the null distribution of tilt magnitude the unconditioned mixture
already produces. The failure being policed is the REAL narrative producing tilt that is
*anomalously LARGE* relative to that null -- i.e. spurious / over-tilted conditioning.

  per-factor one-sided empirical p = (1 + #{placebo >= real}) / (1 + n_placebo)
    - p SMALL  <=> real tilt exceeds (almost) all placebos  <=> EXCESS tilt -> FLAG / FAIL
    - p LARGE  <=> real tilt sits inside the placebo cloud   <=> no excess  -> PASS

  pass_gate = no free factor has p <= p_floor  (no factor shows anomalous excess tilt).

Note on the task's "indistinguishable from placebo" framing: that phrasing describes the *other*
metamorphic check -- Section D item (2) the **placebo REFUTER**, which is applied to the
narrative-*named* factors and wants the real tilt to be *distinguishable* (p<=0.05) so that the
conditioning channel is provably alive. That is a DIFFERENT gate with the OPPOSITE pass direction
and a different factor scope (named factors, not the free factors graded here). This module is the
negative-control / Section-B coherence side, so a small p is the failure, exactly as the task
signature ("pass_gate = no factor has p < p_floor") encodes. We use ``<=`` rather than ``<`` for
a load-bearing reason: with the default ``p_floor = 1/(n_placebo+1)`` the minimum attainable p is
itself ``1/(1+n_placebo)`` (achieved when real exceeds every placebo), so a strict ``<`` could
never fire and the gate would be an inert no-op. ``<=`` makes "real exceeds all placebos" -- the
maximal certifiable excess, and the most extreme level the placebo resolution can report per
Section D's "Placebo p floor = 1/(N+1)" rule -- trip the gate.

------------------------------------------------------------------------------------------------
sign() convention: ``np.sign`` returns 0 for an exactly-zero argument, so a zero tilt or zero
mu_free counts as sign-disagreement (a factor with a clear conditional direction that the ensemble
left flat is, correctly, a coherence failure). For beta-above-floor factors mu_free is generically
nonzero, so this only matters at exact-zero edges.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Pre-registered 994a twin-noise band floor; frozen at
# experiments/backfill/block_ar/nl_996a_n1_tilt_training.py:114 (NOISE_FLOOR = 0.048).
NOISE_FLOOR = 0.048


def sign_agreement_scorer(
    mu_free: np.ndarray,
    conditioned_tilts: np.ndarray,
    conditional_betas: np.ndarray,
    noise_floor: float = NOISE_FLOOR,
) -> dict[str, Any]:
    """Section-B conditional-completion sign gate over the free (non-pinned) factors.

    Parameters
    ----------
    mu_free : (n_free,)
        Gaussian conditional-completion target mu_{-S|S} for each free factor -- "what this
        factor should do if S moves like the narrative pins it." Sign is the gate target.
    conditioned_tilts : (n_free,)
        The conditioned ensemble's realized tilt (vs start-only) on each free factor.
    conditional_betas : (n_free, n_pinned)
        Sigma_{-S,S} Sigma_{S,S}^{-1} per free factor (one row per free factor, one column per
        pinned factor). Materiality = max over pinned of |beta|.
    noise_floor : float
        994a-bootstrap noise floor; a free factor participates in the hard sign gate only if its
        beta magnitude clears this. Also the default "material tilt" threshold for the
        near-zero-beta red flag. Default = pre-registered 0.048.

    Returns
    -------
    dict with keys:
      sign_agreement            (n_free,) bool   sign(tilt) == sign(mu_free)
      beta_magnitude            (n_free,) float  max_j |conditional_betas[i, j]|
      above_noise_floor         (n_free,) bool   beta_magnitude > noise_floor   (strict)
      material_tilt_on_low_beta (k,)     int     INDICES (into the free-factor axis) where
                                                 |tilt| > noise_floor AND beta below floor
                                                 (Section-B red flag: material tilt where there
                                                 should be ~none -- "material tilt" threshold =
                                                 noise_floor, reusing the one pre-registered
                                                 constant rather than introducing a new one)
      material_tilt_on_low_beta_mask (n_free,) bool  the same red flag as a boolean mask
      n_free, n_gated           ints             total free factors / factors clearing the floor
      pass_gate                 bool             all above-floor free factors sign-agree
                                                 (vacuously True if none clear the floor)
    """
    mu_free = np.asarray(mu_free, dtype=np.float64).reshape(-1)
    tilts = np.asarray(conditioned_tilts, dtype=np.float64).reshape(-1)
    betas = np.asarray(conditional_betas, dtype=np.float64)
    floor = float(noise_floor)

    n_free = mu_free.shape[0]
    if tilts.shape[0] != n_free:
        raise ValueError(
            f"conditioned_tilts has length {tilts.shape[0]}, expected n_free={n_free}"
        )
    if betas.ndim != 2:
        raise ValueError(f"conditional_betas must be 2-D (n_free, n_pinned); got {betas.shape}")
    if betas.shape[0] != n_free:
        raise ValueError(
            f"conditional_betas has {betas.shape[0]} rows, expected n_free={n_free}"
        )

    if betas.shape[1] == 0:
        beta_magnitude = np.zeros(n_free, dtype=np.float64)
    else:
        beta_magnitude = np.max(np.abs(betas), axis=1)

    sign_agreement = np.sign(tilts) == np.sign(mu_free)
    above_noise_floor = beta_magnitude > floor
    material_tilt_low_beta_mask = (np.abs(tilts) > floor) & (~above_noise_floor)
    material_tilt_on_low_beta = np.where(material_tilt_low_beta_mask)[0]

    # Hard gate: every free factor whose conditional beta clears the floor must sign-agree.
    # Vacuously True if no free factor clears the floor.
    pass_gate = bool(np.all(sign_agreement[above_noise_floor])) if n_free else True

    return {
        "sign_agreement": sign_agreement,
        "beta_magnitude": beta_magnitude,
        "above_noise_floor": above_noise_floor,
        "material_tilt_on_low_beta": material_tilt_on_low_beta,
        "material_tilt_on_low_beta_mask": material_tilt_low_beta_mask,
        "n_free": int(n_free),
        "n_gated": int(np.sum(above_noise_floor)),
        "pass_gate": pass_gate,
    }


def excess_tilt_check(
    real_tilt_mag: np.ndarray,
    placebo_tilt_mag: np.ndarray,
    p_floor: float | None = None,
) -> dict[str, Any]:
    """Section-D(6) negative-control excess-tilt check over the free factors.

    Flags free factors where the REAL narrative produces tilt magnitude anomalously LARGE versus
    a placebo (null / scrambled / off-domain narrative) cloud -- i.e. excess tilt beyond what the
    unconditioned support mixture already implies. See the module docstring for the full pass-
    direction derivation; in brief: small one-sided p == excess tilt == FAIL.

    Parameters
    ----------
    real_tilt_mag : (n_free,)
        |tilt| of the real narrative on each free factor.
    placebo_tilt_mag : (n_placebo, n_free)
        |tilt| of each placebo narrative on each free factor (the empirical null cloud).
    p_floor : float | None
        One-sided p threshold; a factor is flagged when p <= p_floor. Defaults to the
        Section-D resolution floor 1/(n_placebo + 1) -- the most extreme level a placebo pool of
        this size can certify ("don't report p < 1/(N+1)"). ``<=`` is used so that "real exceeds
        every placebo" (p == 1/(1+n_placebo) == p_floor) trips the gate rather than being inert.

    Returns
    -------
    dict with keys:
      p_values        (n_free,) float  (1 + #{placebo >= real}) / (1 + n_placebo)
      p_floor         float            threshold used
      excess_flag     (n_free,) bool   p_values <= p_floor  (anomalous excess tilt)
      n_placebo, n_free  ints
      pass_gate       bool             no free factor flagged for excess tilt
    """
    real = np.asarray(real_tilt_mag, dtype=np.float64).reshape(-1)
    placebo = np.asarray(placebo_tilt_mag, dtype=np.float64)
    n_free = real.shape[0]

    if placebo.ndim != 2:
        raise ValueError(
            f"placebo_tilt_mag must be 2-D (n_placebo, n_free); got {placebo.shape}"
        )
    n_placebo = placebo.shape[0]
    if placebo.shape[1] != n_free:
        raise ValueError(
            f"placebo_tilt_mag has {placebo.shape[1]} cols, expected n_free={n_free}"
        )
    if n_placebo == 0:
        raise ValueError("placebo_tilt_mag must have at least one placebo row")

    if p_floor is None:
        p_floor = 1.0 / (n_placebo + 1)
    p_floor = float(p_floor)

    # One-sided: how many placebos are at least as extreme as the real tilt?
    ge_count = np.sum(placebo >= real[None, :], axis=0)  # (n_free,)
    p_values = (1.0 + ge_count) / (1.0 + n_placebo)

    excess_flag = p_values <= p_floor
    pass_gate = not bool(np.any(excess_flag))

    return {
        "p_values": p_values,
        "p_floor": p_floor,
        "excess_flag": excess_flag,
        "n_placebo": int(n_placebo),
        "n_free": int(n_free),
        "pass_gate": pass_gate,
    }
