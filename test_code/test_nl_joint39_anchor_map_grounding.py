"""Regression gate for the 2026-06-11 joint39 factor-mapping contamination.

Proves (TDD): (1) the canonical map derived from the data file's level_columns is the
ground truth; (2) every hardcoded anchor->column map in the narrative builders equals it;
(3) the grounding gate FLAGS the old bug signature (AAA_OAS@36 / USDJPY@29) and PASSES
correct cards; (4) any regenerated support-card corpus on disk is clean;
(5) a directional spot-check on multiformat_episode_cards.jsonl confirms that the clean
stride-5 corpus is directionally accurate and the contaminated 982g_sharded corpus fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from experiments.backfill.block_ar.nl_joint39_anchor_map import (
    KNOWN_BAD_COLS,
    canonical_col,
    card_index_violations,
    joint39_anchor_columns,
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Directional spot-check constants (Task 5 — multiformat_episode_cards.jsonl)
# ---------------------------------------------------------------------------
# USDJPY: data col 2, joint39 col 27.
# The contaminated builder read COPPER (data col 4 → joint39 col 29) for USDJPY,
# so cards where USDJPY and COPPER moved in opposite directions will have the wrong
# directional text in the contaminated corpus.
#
# Detection strategy: keyword match on the views text vs. the actual 30-day delta
# from data/multi_factor_data.npz.  A card "fails" when the text says USDJPY moved
# in one direction but the real delta exceeds ±USDJPY_THRESHOLD in the OTHER direction.
# Threshold is set conservatively (1.5 JPY) to avoid flagging flat/near-zero windows;
# only flag when both a directional keyword is present AND the real delta clearly
# contradicts it.
#
# Results from calibration run (2026-06-15):
#   Clean stride-5 corpus: 2/802 errors (0.25%) — genuine Codex authoring errors
#   Contaminated 982g_sharded: 846/4010 errors (21.1%)
#
# Test thresholds:
#   CLEAN_MAX_ERROR_RATE = 3%   (> 100× headroom above observed 0.25%)
#   CONTAMINATED_MIN_ERROR_RATE = 10%  (well below observed 21.1%)
#
# The two corpora are separated by >7× on error rate so the thresholds are robust.

USDJPY_THRESHOLD: float = 1.5  # JPY; moves below this are "flat" and not tested

# Unambiguous USDJPY-UP terms (yen weakens vs dollar)
_USDJPY_UP_STRINGS: tuple[str, ...] = (
    "usdjpy higher",
    "usdjpy up ",
    "usdjpy up,",
    "usdjpy strength",
    "dollar-yen strength",
    "dollar-yen higher",
    "dollar-yen surge",
    "dollar-yen firmer",
    "usdjpy firmer",
    "usdjpy up large",
    "usdjpy up medium",
    "usdjpy up small",
)
# 'yen weaker' / 'yen weakness' are ambiguous: 'dollar-yen weakness' means USDJPY DOWN.
# Use negative lookbehind to skip them when part of 'dollar-yen …'.
_USDJPY_UP_REGEX: tuple[str, ...] = (
    r"(?<!dollar-)yen weaker",
    r"(?<!dollar-)yen weakness",
)

# Unambiguous USDJPY-DOWN terms (yen strengthens vs dollar)
_USDJPY_DN_STRINGS: tuple[str, ...] = (
    "usdjpy lower",
    "usdjpy down",
    "lower usdjpy",
    "usdjpy lower large",
    "dollar-yen lower",
    "dollar-yen falls",
    "dollar-yen weakness",
    "dollar-yen decline",
    "dollar-yen sold",
    "lower dollar-yen",
)
_USDJPY_DN_REGEX: tuple[str, ...] = (
    r"(?<!dollar-)yen strength",
    r"(?<!dollar-)yen firmer",
    r"(?<!dollar-)yen stronger",
    r"(?<!dollar-)yen gained",
    r"(?<!dollar-)yen appreciation",
)

CLEAN_CORPUS_PATHS: tuple[str, ...] = (
    "episode_card_v3_codex_multiformat_982g_clean_stride5_20260612",
)
CONTAMINATED_CORPUS_PATHS: tuple[str, ...] = (
    "episode_card_v3_full_codex_multiformat_982g_sharded",
)

CLEAN_MAX_ERROR_RATE: float = 0.03   # 3%
CONTAMINATED_MIN_ERROR_RATE: float = 0.10  # 10%


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


def test_regenerated_support_cards_on_disk_are_clean():
    """Hard gate: every support-card corpus on disk must have zero violations.

    Went green 2026-06-12 after R1 regenerated 970c/970f/972b with the fixed builder
    (was xfail while legacy corpora were contaminated).
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


# ---------------------------------------------------------------------------
# Task 5: directional spot-check on multiformat_episode_cards.jsonl
# ---------------------------------------------------------------------------

def _usdjpy_direction_error(text_lower: str, real_delta: float) -> str | None:
    """Return a description string if the text contradicts the real USDJPY delta.

    Returns None when no directional keyword is present, delta is below threshold,
    or both up and down keywords are present (ambiguous / mixed window).
    Only fires when the contradiction is unambiguous (one direction keyword set, not both).
    """
    if abs(real_delta) < USDJPY_THRESHOLD:
        return None
    says_up = any(t in text_lower for t in _USDJPY_UP_STRINGS) or any(
        bool(re.search(p, text_lower)) for p in _USDJPY_UP_REGEX
    )
    says_dn = any(t in text_lower for t in _USDJPY_DN_STRINGS) or any(
        bool(re.search(p, text_lower)) for p in _USDJPY_DN_REGEX
    )
    # Only flag unambiguous contradictions (not both directions present)
    if says_up and not says_dn and real_delta < -USDJPY_THRESHOLD:
        return f"text=UP but USDJPY_delta={real_delta:.3f}"
    if says_dn and not says_up and real_delta > USDJPY_THRESHOLD:
        return f"text=DOWN but USDJPY_delta={real_delta:.3f}"
    return None


def _directional_error_rate(corpus_path: Path) -> tuple[int, int]:
    """Return (n_errors, n_checked) for USDJPY directional consistency in a corpus.

    Reads multi_factor_data.npz for the ground-truth 30-day USDJPY delta.
    window_id encoding: 'joint39_train_NNNN' -> data start row = NNNN.
    """
    data_path = ROOT / "data" / "multi_factor_data.npz"
    if not data_path.exists():
        pytest.skip(f"data file not found: {data_path}")

    levels = np.load(str(data_path), allow_pickle=True)["levels"]

    n_errors = 0
    n_checked = 0

    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            wid = card.get("window_id", "")
            try:
                start_row = int(wid.split("_")[-1])
            except (ValueError, IndexError):
                continue
            window_data = levels[start_row : start_row + 30]
            if len(window_data) < 30:
                continue
            real_delta = float(window_data[-1, 2] - window_data[0, 2])  # USDJPY data col 2

            # Concatenate all prose views for a broad text scan
            all_text = " ".join(
                v for v in card.get("views", {}).values() if isinstance(v, str)
            ).lower()

            err = _usdjpy_direction_error(all_text, real_delta)
            if err:
                n_errors += 1
            n_checked += 1

    return n_errors, n_checked


def _find_multiformat_corpus(fragment: str) -> Path | None:
    """Return the multiformat_episode_cards.jsonl under a directory matching `fragment`."""
    base = ROOT / "experiments" / "backfill" / "block_ar" / "nl_scenario_demo_outputs"
    for candidate in base.glob("**/multiformat_episode_cards.jsonl"):
        if fragment in str(candidate):
            return candidate
    return None


def test_clean_stride5_corpus_passes_usdjpy_direction_check():
    """Clean stride-5 corpus must have < CLEAN_MAX_ERROR_RATE directional errors.

    The clean corpus was regenerated with the canonical joint39 map (USDJPY→col 27).
    Calibration: 2/802 (0.25%) genuine Codex authoring errors — well within the 3% gate.
    """
    for fragment in CLEAN_CORPUS_PATHS:
        corpus_path = _find_multiformat_corpus(fragment)
        if corpus_path is None:
            pytest.skip(f"clean corpus not found on disk (fragment: {fragment!r})")

        n_errors, n_checked = _directional_error_rate(corpus_path)
        if n_checked == 0:
            pytest.skip(f"corpus has no parseable cards: {corpus_path}")

        error_rate = n_errors / n_checked
        assert error_rate < CLEAN_MAX_ERROR_RATE, (
            f"Clean corpus {corpus_path.relative_to(ROOT)} failed direction check: "
            f"{n_errors}/{n_checked} = {error_rate:.1%} > threshold {CLEAN_MAX_ERROR_RATE:.0%}"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Contaminated 982g_sharded corpus is expected to fail the USDJPY direction "
        "check (observed ~21% error rate vs 10% threshold).  This xfail proves the "
        "detector catches the bug; it must stay xfail as long as the contaminated "
        "corpus is on disk.  If this mark unexpectedly passes, the detector is broken."
    ),
)
def test_contaminated_982g_sharded_fails_usdjpy_direction_check():
    """Contaminated corpus MUST fail the direction check (xfail — expected to be wrong).

    The 982g_sharded corpus was built with USDJPY→col 29 (COPPER's column).
    Calibration: 846/4010 (21.1%) errors, far above the 10% contamination gate.
    This xfail is a sentinel: if it unexpectedly passes, the direction detector
    has regressed and no longer catches the contamination.
    """
    for fragment in CONTAMINATED_CORPUS_PATHS:
        corpus_path = _find_multiformat_corpus(fragment)
        if corpus_path is None:
            pytest.skip(f"contaminated corpus not found on disk (fragment: {fragment!r})")

        n_errors, n_checked = _directional_error_rate(corpus_path)
        if n_checked == 0:
            pytest.skip(f"corpus has no parseable cards: {corpus_path}")

        error_rate = n_errors / n_checked
        # This assertion is expected to FAIL (that's the xfail intent).
        # A contaminated corpus should have error_rate >= CONTAMINATED_MIN_ERROR_RATE.
        # We assert the opposite: that it is BELOW the threshold.
        # Because of xfail, pytest reports "xfailed" (expected failure) when this assert fails.
        assert error_rate < CONTAMINATED_MIN_ERROR_RATE, (
            f"Contaminated corpus {corpus_path.relative_to(ROOT)}: "
            f"{n_errors}/{n_checked} = {error_rate:.1%} >= threshold "
            f"{CONTAMINATED_MIN_ERROR_RATE:.0%} — corpus IS contaminated as expected"
        )
