"""T7 narrative-conditioned reweighter — unit tests (TDD).

Run from repo root: PYTHONPATH=. python -m pytest test_code/test_t7_nl_narrative_reweighter.py -q
"""
from experiments.backfill.block_ar.nl_narrative_reweighter import dir_sign, match


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
