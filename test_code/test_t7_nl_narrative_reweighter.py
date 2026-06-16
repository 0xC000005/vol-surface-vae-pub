"""T7 narrative-conditioned reweighter — unit tests (TDD).

Run from repo root: PYTHONPATH=. python -m pytest test_code/test_t7_nl_narrative_reweighter.py -q
"""
import numpy as np

from experiments.backfill.block_ar.nl_narrative_reweighter import (
    analogue_profile,
    dir_sign,
    match,
    narrative_emphasis,
)


def test_dir_sign():
    assert dir_sign("up") == 1 and dir_sign("wider") == 1
    assert dir_sign("down") == -1 and dir_sign("tighter") == -1
    assert dir_sign("flat") == 0 and dir_sign("") == 0


def test_match_agreement():
    e = {"USDJPY": {"direction": "up", "salience": 1.0},
         "AAA_OAS": {"direction": "wider", "salience": 0.5}}
    # analogue agrees on USDJPY (up), opposes on AAA_OAS (tighter)
    a = {"USDJPY": "up", "AAA_OAS": "tighter"}
    assert match(e, a) == 1.0 * 1 + 0.5 * (-1)   # = 0.5
    # flat emphasis or flat analogue contributes 0
    assert match({"SPX": {"direction": "flat", "salience": 1.0}}, {"SPX": "down"}) == 0.0
    assert match({"SPX": {"direction": "down", "salience": 1.0}}, {"SPX": "flat"}) == 0.0
    # missing factor in analogue contributes 0
    assert match({"VIX": {"direction": "up", "salience": 1.0}}, {}) == 0.0


def test_analogue_profile_directions():
    # synthetic panel: 40 rows, 2 factor cols. col0 rises +5 over [0,29]; col1 falls.
    panel = np.zeros((40, 2), dtype=np.float32)
    panel[:, 0] = np.linspace(0, 10, 40)   # rising
    panel[:, 1] = np.linspace(0, -10, 40)  # falling
    cols = {"USDJPY": 0, "AAA_OAS": 1}
    prof = analogue_profile(window_index=0, panel=panel, factor_cols=cols, horizon=30)
    assert prof["USDJPY"] == "up"
    assert prof["AAA_OAS"] == "tighter"   # OAS spread falling -> tighter


def test_emphasis_prose_fallback():
    grounding = {"condition_only_grounding": {"current_market_state_implications": [
        "USDJPY is up strongly as the dollar firms",
        "AAA credit spreads are widening",
        "equities (SPX) are down",
    ]}}
    e = narrative_emphasis(grounding)   # no structured field -> prose parse
    assert e["USDJPY"]["direction"] == "up"
    assert e["AAA_OAS"]["direction"] == "wider"
    assert e["SPX"]["direction"] == "down"
    assert all(0.0 <= v["salience"] <= 1.0 for v in e.values())


def test_emphasis_empty_safe():
    assert narrative_emphasis({}) == {}   # safe-degrade -> identity reweight


def test_emphasis_prefers_structured():
    grounding = {"condition_only_grounding": {
        "factor_emphasis": {"VIX": {"direction": "up", "salience": 0.8}},
        "current_market_state_implications": ["equities down"],  # should be ignored
    }}
    e = narrative_emphasis(grounding)
    assert e == {"VIX": {"direction": "up", "salience": 0.8}}


def test_emphasis_reads_native_structured_implications():
    # The REAL grounding shape (verified on disk): current_market_state_implications is a list
    # of ConditionMarketImplication dicts with market/direction/magnitude/confidence.
    grounding = {"condition_only_grounding": {"current_market_state_implications": [
        {"market": "SPX", "direction": "down", "magnitude": "medium", "confidence": "high"},
        {"market": "VIX", "direction": "up", "magnitude": "large", "confidence": "high"},
        {"market": "BBB_OAS", "direction": "wider", "magnitude": "small", "confidence": "medium"},
        {"market": "credit spreads", "direction": "wider", "magnitude": "small"},  # alias -> BBB_OAS
        {"market": "IV_SURFACE", "direction": "up", "magnitude": "medium"},         # not an anchor -> dropped
        {"market": "DXY", "direction": "mixed", "magnitude": "medium"},             # mixed -> no tilt
    ]}}
    e = narrative_emphasis(grounding)
    assert e["SPX"]["direction"] == "down"
    assert e["VIX"]["direction"] == "up" and e["VIX"]["salience"] == 1.0   # large
    assert e["BBB_OAS"]["direction"] == "wider"
    # monotonic salience: large > medium > small > 0
    assert e["VIX"]["salience"] > e["SPX"]["salience"] > e["BBB_OAS"]["salience"] > 0.0
    assert "IV_SURFACE" not in e and "DXY" not in e   # non-anchor / non-directional dropped
