"""Unit tests for the Track B B-width GROUNDED narrative-intensity scalar.

Locks the load-bearing properties: (1) magnitude ordering large>medium>small; (2) multi-
direction clauses ("DXY higher; SPX lower") still detect ALL factors (the shared reweighter's
sentence-level up/down collapse would drop them); (3) NO OpenAI / NO realized-future peek;
(4) normalization spread.
"""

from __future__ import annotations

from experiments.backfill.block_ar.nl_track_b_grounded_intensity import (
    grounded_intensity_for_windows,
    window_intensity_from_text,
)


def test_magnitude_ordering_monotone() -> None:
    big = window_intensity_from_text("Equities crash violently as VIX spikes sharply.")
    mid = window_intensity_from_text("Equities move moderately lower while VIX is notable.")
    small = window_intensity_from_text("Equities edge slightly lower while VIX stays muted.")
    # same factors named (SPX, VIX); magnitude tier drives the per-factor scalar
    assert big["raw_intensity"] > mid["raw_intensity"] > small["raw_intensity"]


def test_multidirection_clause_detects_all_factors() -> None:
    # the shared reweighter drops this whole clause (both "higher" and "lower" present);
    # the intensity parser must still detect every named factor.
    r = window_intensity_from_text(
        "DXY higher large; USDJPY higher large; crude higher large; "
        "SPX lower medium; VIX flat; BBB OAS flat."
    )
    factors = set(r["factors"].keys())
    for f in ("DXY", "USDJPY", "CRUDE_OIL", "SPX", "VIX", "BBB_OAS"):
        assert f in factors, f"missing {f} in {factors}"
    assert r["n_factors"] >= 6


def test_no_factors_for_empty_or_unrelated() -> None:
    assert window_intensity_from_text("")["n_factors"] == 0
    assert window_intensity_from_text("The weather was pleasant today.")["n_factors"] == 0


def test_normalization_spread_and_range() -> None:
    texts = {
        0: "A violent equity crash with VIX spiking sharply and spreads blowing out.",
        1: "Quiet tape: spreads barely moved, equities flat, VIX subdued.",
        2: "DXY moved moderately higher while crude rose notably.",
    }
    norm = grounded_intensity_for_windows(texts)
    assert set(norm) == {0, 1, 2}
    assert all(0.0 <= v <= 1.0 for v in norm.values())
    # the violent window must outrank the quiet window
    assert norm[0] > norm[1]


def test_degenerate_all_equal_maps_to_zero() -> None:
    # identical texts -> no spread -> all-zero (uninformative, by design)
    texts = {0: "Equities fell sharply.", 1: "Equities fell sharply."}
    norm = grounded_intensity_for_windows(texts)
    assert norm == {0: 0.0, 1: 0.0}
