"""Track B B-WIDTH: a GROUNDED per-window narrative-INTENSITY scalar (prose-parse, NO OpenAI).

The width probe asks whether the conditioned ensemble's fan WIDTH tracks how *severe* the
narrative says the scenario is. To answer that faithfully we need a per-window scalar that
measures narrative intensity WITHOUT peeking at the realized future (no leakage) and WITHOUT
new model/API calls.

Design (matches the B-width task spec):
- aggregate, over the factors a narrative names, ``magnitude(factor) * |salience(factor)|``;
- magnitude words: small=0.33, medium=0.67, large=1.0 (graded from the prose);
- normalize the per-window aggregate to [0, 1] across the window set.

This REUSES the verified prose factor/direction parser in ``nl_narrative_reweighter``
(``_emphasis_from_implications`` -> {factor: {direction, salience}}) for *which* factors are
named, then layers a per-factor MAGNITUDE read on top of it. It deliberately does NOT edit
``nl_narrative_reweighter`` (T7 / Track A / the demo import it); this is additive, new code.

The magnitude read is intentionally simple and monotone (small<medium<large): the absolute
values do not need to be exact, only the ordering, because the width gate scores a rank
(spearman) and the trainer's width term is correlation/rank-based. ``flat`` factors and
factors with no magnitude cue contribute their direction's presence at a baseline magnitude.
"""

from __future__ import annotations

import re
from typing import Any

from experiments.backfill.block_ar.nl_narrative_reweighter import (
    _ALIASES,
    _valid_factors,
)

# magnitude tier -> scalar. small<medium<large (ordering is what matters for the rank gate).
MAGNITUDE_SCALE: dict[str, float] = {"large": 1.0, "medium": 0.67, "small": 0.33}
# a named factor with a clear direction but no explicit magnitude cue gets the medium tier:
# the narrative committed to a move, just did not grade it. (flat factors contribute nothing.)
DEFAULT_MAGNITUDE = MAGNITUDE_SCALE["medium"]

# Prose magnitude cues. LARGE = strong intensity adverbs/verbs; SMALL = hedged/minor moves;
# explicit words "large"/"medium"/"small" map directly. Order of checks: explicit > large >
# small > default. (Matched as word-ish substrings; the regex anchors on word boundaries.)
_LARGE_CUES = (
    "surg", "spik", "plung", "soar", "sharp", "sharply", "strongly", "strong",
    "unusually large", "very large", "large", "severe", "severely", "extreme",
    "extremely", "violent", "collaps", "crater", "skyrocket", "rout", "crash",
    "massiv", "dramatic", "steep", "steeply", "elevated", "heavy", "heavily",
)
_SMALL_CUES = (
    "edg", "slight", "slightly", "modest", "modestly", "mild", "mildly", "marginal",
    "marginally", "small", "minor", "muted", "quiet", "calm", "narrow", "narrowly",
    "incremental", "tepid", "subdued", "contained", "limited", "a touch", "a bit",
)
_MEDIUM_CUES = ("medium", "moderate", "moderately", "notable", "notably", "meaningful")


def _word_present(text: str, cue: str) -> bool:
    """Substring-with-boundary match: ``cue`` appears as the start of a word in ``text``."""
    return re.search(r"(?:^|[^a-z])" + re.escape(cue), text) is not None


def _magnitude_for_factor(text_low: str, factor: str) -> float:
    """Grade the magnitude the prose attaches to ``factor``.

    Heuristic: look in a window of text around the factor's first alias mention for a magnitude
    cue; if none is local, fall back to the strongest cue anywhere in the sentence; else the
    default tier. This stays robust to terse views ("DXY higher large; SPX lower medium; ...")
    AND verbose views ("crude posted an unusually large rise ... SPX fell by a medium amount").
    """
    aliases = _ALIASES.get(factor, ())
    # locate the factor mention(s)
    spans: list[int] = []
    for alias in aliases:
        for m in re.finditer(re.escape(alias), text_low):
            spans.append(m.start())
    # local window around each mention (+-60 chars); pick the strongest cue found locally.
    def _tier_in(segment: str) -> str | None:
        if any(_word_present(segment, c) for c in _LARGE_CUES):
            return "large"
        if any(_word_present(segment, c) for c in _SMALL_CUES):
            return "small"
        if any(_word_present(segment, c) for c in _MEDIUM_CUES):
            return "medium"
        return None

    local_tiers: list[str] = []
    for pos in spans:
        seg = text_low[max(0, pos - 60) : pos + 60]
        tier = _tier_in(seg)
        if tier is not None:
            local_tiers.append(tier)
    if local_tiers:
        # prefer the strongest local cue (large > medium > small)
        for pref in ("large", "medium", "small"):
            if pref in local_tiers:
                return MAGNITUDE_SCALE[pref]
    # fall back to the strongest cue anywhere in the text
    global_tier = _tier_in(text_low)
    if global_tier is not None:
        return MAGNITUDE_SCALE[global_tier]
    return DEFAULT_MAGNITUDE


def window_intensity_from_text(text: str) -> dict[str, Any]:
    """Grounded narrative-intensity for one window's narrative prose.

    Returns a dict with the raw (un-normalized) aggregate plus the per-factor breakdown.
    Aggregate = sum over named, non-flat factors of ``magnitude(factor) * |salience(factor)|``.
    The prose parser supplies which factors are named + their direction/salience; the magnitude
    read is layered here. NO OpenAI / NO realized-future peek.
    """
    text = str(text or "")
    text_low = text.lower()
    valid = _valid_factors()
    # which factors does the narrative NAME? Detect each factor independently by its aliases.
    # Deliberately NOT the shared reweighter's sentence-level up/down collapse: that parser
    # drops a whole clause when both "higher" and "lower" appear (e.g. "DXY higher; SPX lower"),
    # which loses exactly the high-information multi-factor windows. For an INTENSITY/severity
    # scalar the SIGN is irrelevant -- we only need magnitude * |salience| -- so we detect the
    # factor mention and grade its local magnitude, with no direction gating.
    factors: dict[str, dict[str, Any]] = {}
    aggregate = 0.0
    for factor, aliases in _ALIASES.items():
        if factor not in valid:
            continue
        named = any(re.search(r"(?:^|[^a-z])" + re.escape(a), text_low) for a in aliases)
        if not named:
            continue
        sal = 1.0  # salience = the narrative named this factor (relative magnitudes via tier)
        mag = _magnitude_for_factor(text_low, factor)
        contrib = mag * sal
        factors[factor] = {"salience": sal, "magnitude": mag, "contribution": contrib}
        aggregate += contrib
    return {
        "raw_intensity": float(aggregate),
        "n_factors": int(len(factors)),
        "factors": factors,
        "text": text,
    }


def grounded_intensity_for_windows(texts: dict[int, str]) -> dict[int, float]:
    """Map window-index -> NORMALIZED [0,1] grounded intensity over the given window set.

    Normalization is min-max across the supplied windows (so the scalar is comparable within an
    eval frame). A degenerate all-equal set maps to 0.0 everywhere (no spread -> no rank signal,
    correctly reported as uninformative by the caller).
    """
    raw = {int(i): float(window_intensity_from_text(t)["raw_intensity"]) for i, t in texts.items()}
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0.0:
        return {i: 0.0 for i in raw}
    return {i: (v - lo) / span for i, v in raw.items()}


def grounded_intensity_with_raw(texts: dict[int, str]) -> dict[int, dict[str, float]]:
    """Per-window {"raw": absolute intensity, "norm": frame min-max [0,1]}.

    The ABSOLUTE intensity is needed for the intercept/level analysis in the width gate (a
    severity DIAL must have intercept ~ 0 at genuinely-low ABSOLUTE intensity -- few/weak factors
    named -- not merely a positive RANK slope, which is also satisfied by uniform presence-
    inflation with a tilt). Frame min-max alone makes the calmest in-frame window 0 even if it
    names several "large" factors, which is rank-fine but level-wrong.
    """
    raw = {int(i): float(window_intensity_from_text(t)["raw_intensity"]) for i, t in texts.items()}
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo
    norm = (lambda v: 0.0) if span <= 0.0 else (lambda v: (v - lo) / span)
    return {i: {"raw": v, "norm": float(norm(v))} for i, v in raw.items()}
