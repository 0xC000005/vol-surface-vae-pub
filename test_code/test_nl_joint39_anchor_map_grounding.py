"""Regression gate for the 2026-06-11 joint39 factor-mapping contamination.

Proves (TDD): (1) the canonical map derived from the data file's level_columns is the
ground truth; (2) every hardcoded anchor->column map in the narrative builders equals it;
(3) the grounding gate FLAGS the old bug signature (AAA_OAS@36 / USDJPY@29) and PASSES
correct cards; (4) any regenerated support-card corpus on disk is clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.backfill.block_ar.nl_joint39_anchor_map import (
    KNOWN_BAD_COLS,
    canonical_col,
    card_index_violations,
    joint39_anchor_columns,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_ground_truth():
    # Straight from data/multi_factor_data.npz level_columns (+25). These are the values
    # the contamination got wrong.
    assert canonical_col("USDJPY") == 27
    assert canonical_col("COPPER") == 29
    assert canonical_col("AAA_OAS") == 34
    assert canonical_col("BBB_OAS") == 35
    assert canonical_col("NIKKEI") == 36
    # the bug read AAA_OAS from nikkei's col and USDJPY from copper's col
    assert KNOWN_BAD_COLS["AAA_OAS"] == canonical_col("NIKKEI")
    assert KNOWN_BAD_COLS["USDJPY"] == canonical_col("COPPER")


def test_support_cards_markets_match_canonical():
    from experiments.backfill.block_ar.nl_episode_narrative_support_cards import MARKETS

    canon = joint39_anchor_columns()
    for spec in MARKETS:
        name = str(spec["name"]).upper()
        assert int(spec["index"]) == canon[name], (
            f"MARKETS[{name}] index {spec['index']} != canonical {canon[name]}"
        )


def test_workbench_historical_map_matches_canonical():
    from experiments.backfill.block_ar.nl_scenario_to_narrative_workbench import (
        HISTORICAL_RAW_PATH_FACTORS,
    )

    canon = joint39_anchor_columns()
    for name, idx in HISTORICAL_RAW_PATH_FACTORS:
        assert int(idx) == canon[str(name).upper()], (
            f"HISTORICAL_RAW_PATH_FACTORS[{name}] {idx} != canonical {canon[str(name).upper()]}"
        )


def _card(rows):
    return {"support_metadata": {"window_index": 7, "support_move_rows": rows}}


def test_gate_flags_old_bug_and_passes_correct():
    # contaminated card (the old bug): AAA_OAS recorded at col 36, USDJPY at col 29
    bad = _card([
        {"market": "AAA_OAS", "index": 36, "raw_change": 0.1},
        {"market": "USDJPY", "index": 29, "raw_change": 0.2},
    ])
    v = card_index_violations(bad)
    kinds = {(x["market"], x["is_known_bad"]) for x in v}
    assert ("AAA_OAS", True) in kinds and ("USDJPY", True) in kinds
    assert len(v) == 2

    # correct card: canonical columns -> zero violations
    good = _card([
        {"market": "AAA_OAS", "index": 34, "raw_change": 0.1},
        {"market": "USDJPY", "index": 27, "raw_change": 0.2},
        {"market": "BBB_OAS", "index": 35, "raw_change": 0.0},
    ])
    assert card_index_violations(good) == []


@pytest.mark.xfail(
    reason="legacy support-card corpora (970f etc.) contaminated until R1 regen (#31); "
    "remove this marker after regeneration so it becomes a hard gate",
    strict=False,
)
def test_regenerated_support_cards_on_disk_are_clean():
    """Real gate: every support-card corpus on disk must have zero violations.

    Currently xfail because legacy corpora predate the fix. After R1 regenerates them
    (and the final 27f sign-off), drop the xfail marker so this is a hard pass.
    """
    candidates = list(
        (ROOT / "experiments/backfill/block_ar/nl_scenario_demo_outputs").glob(
            "**/episode_narrative_support_cards.jsonl"
        )
    )
    if not candidates:
        pytest.skip("no regenerated support-card corpus on disk yet (run R1 first)")
    total_violations = []
    for path in candidates:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            total_violations.extend(
                {**v, "corpus": str(path.relative_to(ROOT))}
                for v in card_index_violations(card)
            )
    assert not total_violations, f"contaminated cards found: {total_violations[:10]}"
