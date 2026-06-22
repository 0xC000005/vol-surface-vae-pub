"""Framework-v1 Gate [B] coherence sign-gate SCORER -- unit tests (TDD).

Run from repo root: PYTHONPATH=. python -m pytest test_code/test_nl_coherence_sign_gate.py -q

Spec: docs/research_protocols/nl_counterfactual_validation_framework_v1.md, Section B
conditional-completion check + Tier-1 item 4 (sign gate + negative-control excess-tilt check).
Pre-registered noise floor 0.048 = nl_996a_n1_tilt_training.py:114.
"""
import numpy as np

from experiments.backfill.block_ar.nl_coherence_sign_gate import (
    NOISE_FLOOR,
    excess_tilt_check,
    sign_agreement_scorer,
)


def test_noise_floor_is_preregistered_value():
    # Frozen at experiments/backfill/block_ar/nl_996a_n1_tilt_training.py:114.
    assert NOISE_FLOOR == 0.048


# ----------------------------------------------------------------------------------------------
# sign_agreement_scorer
# ----------------------------------------------------------------------------------------------

def test_a_clear_sign_agreement_above_floor_passes():
    # All free factors clear the floor (beta=0.5 > 0.048) and the ensemble tilt sign matches
    # the conditional-completion target sign -> hard gate PASSES.
    mu_free = np.array([1.0, -2.0, 0.5])
    tilts = np.array([0.3, -0.7, 0.2])          # signs: +, -, + == signs of mu_free
    betas = np.array([[0.5], [0.5], [0.5]])      # all above 0.048
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert out["pass_gate"] is True
    assert bool(np.all(out["sign_agreement"])) is True
    assert bool(np.all(out["above_noise_floor"])) is True
    assert out["n_gated"] == 3
    assert out["material_tilt_on_low_beta"].tolist() == []   # no red flags
    assert bool(np.any(out["material_tilt_on_low_beta_mask"])) is False


def test_b_clear_sign_disagreement_above_floor_fails():
    # A factor with beta well above the floor tilts the WRONG way -> hard gate FAILS.
    mu_free = np.array([1.0, -2.0, 0.5])
    tilts = np.array([0.3, +0.7, 0.2])           # second factor sign flipped (+ vs target -)
    betas = np.array([[0.5], [0.9], [0.5]])       # all above floor
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert out["pass_gate"] is False
    assert out["sign_agreement"].tolist() == [True, False, True]
    assert out["n_gated"] == 3


def test_c_disagreement_only_on_below_floor_factor_still_passes():
    # The factor that disagrees has beta BELOW the floor, so the sign gate ignores it.
    mu_free = np.array([1.0, -2.0, 0.5])
    tilts = np.array([0.3, -0.7, -0.9])          # third factor disagrees (- vs target +)
    betas = np.array([[0.5], [0.5], [0.01]])      # third factor beta 0.01 < 0.048 -> sub-floor
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert out["pass_gate"] is True               # gate ignores the sub-floor disagreement
    assert out["sign_agreement"].tolist() == [True, True, False]
    assert out["above_noise_floor"].tolist() == [True, True, False]
    assert out["n_gated"] == 2


def test_c2_material_tilt_on_low_beta_red_flag():
    # Section-B red flag: a near-zero-beta factor that nonetheless shows a MATERIAL tilt
    # (|tilt| > noise_floor) is flagged. It does NOT auto-fail the sign gate (beta sub-floor),
    # but the red-flag mask must light up.
    mu_free = np.array([1.0, 0.5])
    tilts = np.array([0.3, 0.9])                 # second tilt 0.9 is material
    betas = np.array([[0.5], [0.001]])            # second beta near-zero (< floor)
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert out["material_tilt_on_low_beta"].tolist() == [1]          # index of the flagged factor
    assert out["material_tilt_on_low_beta_mask"].tolist() == [False, True]
    # below-floor factor tilt happens to AGREE in sign here, so the gate itself still passes
    assert out["pass_gate"] is True


def test_sign_gate_vacuous_pass_when_no_factor_clears_floor():
    mu_free = np.array([1.0, -1.0])
    tilts = np.array([-1.0, 1.0])                # both disagree...
    betas = np.array([[0.01], [0.02]])            # ...but both betas are sub-floor
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert out["n_gated"] == 0
    assert out["pass_gate"] is True               # vacuously true -- nothing to gate


def test_sign_gate_zero_tilt_counts_as_disagreement():
    # np.sign(0) == 0 != np.sign(mu) for a clearly-directional factor -> disagreement.
    mu_free = np.array([1.0])
    tilts = np.array([0.0])
    betas = np.array([[0.5]])
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert out["sign_agreement"].tolist() == [False]
    assert out["pass_gate"] is False


def test_sign_gate_multi_pinned_uses_max_abs_beta():
    mu_free = np.array([1.0])
    tilts = np.array([0.5])
    betas = np.array([[0.01, 0.9, -0.2]])         # max|.| = 0.9 -> above floor
    out = sign_agreement_scorer(mu_free, tilts, betas)
    assert np.isclose(out["beta_magnitude"][0], 0.9)
    assert out["above_noise_floor"].tolist() == [True]


def test_sign_gate_validates_shapes():
    import pytest

    with pytest.raises(ValueError):
        sign_agreement_scorer(np.array([1.0, 2.0]), np.array([1.0]), np.array([[0.5], [0.5]]))
    with pytest.raises(ValueError):
        sign_agreement_scorer(np.array([1.0]), np.array([1.0]), np.array([0.5]))  # betas 1-D


# ----------------------------------------------------------------------------------------------
# excess_tilt_check  (Section-D(6) negative-control: small p == excess tilt == FAIL)
# ----------------------------------------------------------------------------------------------

def test_d_real_much_larger_than_placebo_small_p_fails():
    # Real tilt dwarfs every placebo -> p hits the resolution floor 1/(n_placebo+1) -> FLAG/FAIL.
    rng = np.random.default_rng(0)
    n_placebo = 99
    placebo = np.abs(rng.normal(0.0, 0.05, size=(n_placebo, 2)))  # null cloud ~ small
    real = np.array([5.0, 6.0])                                   # far outside the null
    out = excess_tilt_check(real, placebo)
    # p == (1 + 0)/(1 + 99) == 0.01 == p_floor -> excess flagged on both factors
    assert np.allclose(out["p_values"], 1.0 / (n_placebo + 1))
    assert out["excess_flag"].tolist() == [True, True]
    assert out["pass_gate"] is False             # <-- the direction decision under test


def test_d_real_similar_to_placebo_large_p_passes():
    # Real tilt sits in the middle of the placebo cloud -> large p -> no excess -> PASS.
    n_placebo = 99
    # placebo magnitudes 0.01 .. 0.99; real == 0.50 sits near the median.
    col = np.linspace(0.01, 0.99, n_placebo)
    placebo = np.stack([col, col], axis=1)        # (99, 2)
    real = np.array([0.50, 0.50])
    out = excess_tilt_check(real, placebo)
    # ~half the placebos are >= 0.50, so p ~ 0.5, well above the 0.01 floor.
    assert np.all(out["p_values"] > out["p_floor"])
    assert out["excess_flag"].tolist() == [False, False]
    assert out["pass_gate"] is True              # <-- the direction decision under test


def test_d_default_p_floor_is_not_inert():
    # Regression guard: with the default floor and a real that exceeds all placebos, the gate
    # MUST fire. A strict `p < p_floor` would make this an inert no-op (min p == p_floor).
    n_placebo = 49
    placebo = np.full((n_placebo, 1), 0.1)
    real = np.array([10.0])
    out = excess_tilt_check(real, placebo)
    assert out["p_floor"] == 1.0 / (n_placebo + 1)
    assert out["p_values"][0] == out["p_floor"]
    assert out["excess_flag"].tolist() == [True]
    assert out["pass_gate"] is False


def test_excess_tilt_custom_p_floor():
    n_placebo = 99
    col = np.linspace(0.01, 0.99, n_placebo)
    placebo = col.reshape(n_placebo, 1)
    real = np.array([0.80])                       # exceeds ~80% of placebos -> p ~ 0.2
    out_loose = excess_tilt_check(real, placebo, p_floor=0.5)   # 0.2 <= 0.5 -> flagged
    out_tight = excess_tilt_check(real, placebo, p_floor=0.05)  # 0.2 > 0.05 -> not flagged
    assert out_loose["excess_flag"].tolist() == [True]
    assert out_tight["excess_flag"].tolist() == [False]


def test_excess_tilt_p_formula_exact():
    # Hand-checked: placebos >= real(0.75) are 0.9 and 0.8 only (0.7 < 0.75) -> count=2 ->
    # p = (1+2)/(1+4) = 0.6.
    placebo = np.array([[0.9], [0.8], [0.7], [0.1]])
    real = np.array([0.75])
    out = excess_tilt_check(real, placebo, p_floor=0.2)
    assert np.isclose(out["p_values"][0], 3.0 / 5.0)
    assert out["pass_gate"] is True   # 0.6 > 0.2 floor -> no excess


def test_excess_tilt_validates_shapes():
    import pytest

    with pytest.raises(ValueError):
        excess_tilt_check(np.array([1.0, 2.0]), np.array([0.1, 0.2]))          # placebo 1-D
    with pytest.raises(ValueError):
        excess_tilt_check(np.array([1.0]), np.zeros((5, 2)))                    # n_free mismatch
    with pytest.raises(ValueError):
        excess_tilt_check(np.array([1.0]), np.zeros((0, 1)))                    # no placebos
