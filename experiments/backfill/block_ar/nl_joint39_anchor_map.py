"""Single source of truth for joint39 anchor name -> column index.

Derived from the data file's OWN stored `level_columns` (NOT a hardcoded table), so
code and data cannot drift. Guards against the 2026-06-11 factor-mapping contamination
in which the narrative builder read AAA_OAS from col 36 (nikkei) and USDJPY from col 29
(copper). All narrative/support-card code should derive anchor columns from here, and the
regression gate (`test_nl_joint39_anchor_map_grounding.py`) asserts every hardcoded map
equals this canonical one.

joint39 panel layout: cols 0..24 = 25 IV cells; cols 25..38 = the 14 anchors in
`level_columns` order. So anchor at level_columns[i] lives at joint39 column 25 + i.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

IV_CELL_COUNT = 25
DEFAULT_DATA_PATH = "data/multi_factor_data.npz"

# The pre-fix contamination, pinned as an anti-regression target. 36=nikkei, 29=copper.
KNOWN_BAD_COLS = {"AAA_OAS": 36, "USDJPY": 29}


@lru_cache(maxsize=8)
def joint39_anchor_columns(data_path: str = DEFAULT_DATA_PATH) -> dict[str, int]:
    """Map ANCHOR_NAME (upper) -> joint39 column index, read from the data file."""
    with np.load(data_path, allow_pickle=True) as d:
        names = [str(x) for x in d["level_columns"].tolist()]
    return {name.upper(): IV_CELL_COUNT + i for i, name in enumerate(names)}


def canonical_col(name: str, data_path: str = DEFAULT_DATA_PATH) -> int:
    return joint39_anchor_columns(data_path)[name.upper()]


def card_index_violations(
    card: dict[str, Any], data_path: str = DEFAULT_DATA_PATH
) -> list[dict[str, Any]]:
    """Return contamination violations for a support card's move rows.

    A violation = a move row whose recorded `index` does not equal the canonical column
    for its `market`. Since raw_change is computed deterministically as
    end[index]-start[index], a correct `index` for every anchor is necessary and
    sufficient for a clean card. `is_known_bad` marks the exact 2026-06-11 bug signature.
    """
    canon = joint39_anchor_columns(data_path)
    rows = card.get("support_metadata", {}).get("support_move_rows", []) or []
    out: list[dict[str, Any]] = []
    for r in rows:
        market = str(r.get("market", "")).upper()
        if market not in canon:
            continue
        recorded = int(r.get("index", -1))
        cc = canon[market]
        if recorded != cc:
            out.append(
                {
                    "window_index": card.get("support_metadata", {}).get("window_index"),
                    "market": market,
                    "kind": "wrong_index",
                    "recorded_index": recorded,
                    "canonical_index": cc,
                    "is_known_bad": recorded == KNOWN_BAD_COLS.get(market),
                }
            )
    return out


def recompute_delta(history_raw: np.ndarray, row: int, market: str,
                    data_path: str = DEFAULT_DATA_PATH) -> float:
    """Spot-check helper: net 30-day delta for `market` at its canonical column."""
    cc = canonical_col(market, data_path)
    h = np.asarray(history_raw, dtype=np.float64)
    return float(h[row, -1, cc] - h[row, 0, cc])
