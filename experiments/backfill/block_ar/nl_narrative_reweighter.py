"""T7: training-free narrative-conditioned reweighter over the retrieved analogue pool.

Adjusts each candidate's score by beta * match(narrative_emphasis, analogue_factor_profile)
BEFORE the existing _apply_top3_90 softmax. beta=0 is an exact no-op. No trained text->latent
head (sidesteps the 992b/T4 information ceiling). See
docs/research_protocols/nl-prefix-latent-t7-narrative-reweighter-intake.md.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from experiments.backfill.block_ar.nl_joint39_anchor_map import joint39_anchor_columns

_POS = {"up", "wider"}
_NEG = {"down", "tighter"}


@lru_cache(maxsize=1)
def _valid_factors() -> frozenset[str]:
    """Canonical joint39 anchor factor names (UPPER); cached to avoid npz reload per call."""
    return frozenset(joint39_anchor_columns().keys())


def dir_sign(direction: str) -> int:
    d = str(direction).strip().lower()
    if d in _POS:
        return 1
    if d in _NEG:
        return -1
    return 0


def match(emphasis: dict[str, dict[str, Any]], analogue: dict[str, str]) -> float:
    """Sum over factors of salience * sign(emphasis_dir) * sign(analogue_dir)."""
    total = 0.0
    for factor, spec in emphasis.items():
        a_dir = analogue.get(factor)
        if a_dir is None:
            continue
        sal = float(spec.get("salience", 0.0))
        total += sal * dir_sign(str(spec.get("direction", ""))) * dir_sign(a_dir)
    return total


_OAS_FACTORS = {"AAA_OAS", "BBB_OAS"}


def _direction(factor: str, delta: float, small: float = 1e-9) -> str:
    if abs(float(delta)) < small:
        return "flat"
    if factor in _OAS_FACTORS:
        return "wider" if delta > 0 else "tighter"
    return "up" if delta > 0 else "down"


def analogue_profile(*, window_index, panel, factor_cols, horizon: int = 30) -> dict[str, str]:
    """Factor-direction profile of the analogue's `horizon`-day window."""
    i = int(window_index)
    end = min(i + int(horizon) - 1, panel.shape[0] - 1)
    out: dict[str, str] = {}
    for factor, col in factor_cols.items():
        delta = float(panel[end, int(col)] - panel[i, int(col)])
        out[factor] = _direction(factor, delta)
    return out


# factor -> aliases that may appear in grounding prose
_ALIASES = {
    "SPX": ("spx", "equit", "s&p", "stock"), "VIX": ("vix", "volatil"),
    "USDJPY": ("usdjpy", "yen", "dollar-yen"), "DXY": ("dxy", "dollar index", "the dollar"),
    "AAA_OAS": ("aaa",), "BBB_OAS": ("bbb", "credit spread", "ig spread"),
    "US2Y": ("2y", "two-year", "front-end"), "US10Y": ("10y", "ten-year", "long-end", "yield"),
    "GOLD": ("gold",), "CRUDE_OIL": ("crude", "oil", "wti", "brent"),
    "COPPER": ("copper",), "WHEAT": ("wheat",), "NIKKEI": ("nikkei",), "USDCAD": ("usdcad", "loonie"),
}
_UP_WORDS = ("up", "rise", "rising", "higher", "firm", "surg", "rally", "rebound", "widen", "elevat", "jump", "gain")
_DOWN_WORDS = ("down", "fall", "falling", "lower", "drop", "selloff", "sell-off", "tighten", "compress", "decline", "weaken")


def _emphasis_from_implications(implications: list[str]) -> dict[str, dict[str, Any]]:
    factors = _valid_factors()
    out: dict[str, dict[str, Any]] = {}
    for line in implications:
        low = str(line).lower()
        up = any(w in low for w in _UP_WORDS)
        down = any(w in low for w in _DOWN_WORDS)
        if up == down:
            continue  # ambiguous / none
        for factor, aliases in _ALIASES.items():
            if factor not in factors:
                continue
            if any(a in low for a in aliases):
                if factor in _OAS_FACTORS:
                    direction = "wider" if up else "tighter"
                else:
                    direction = "up" if up else "down"
                out.setdefault(factor, {"direction": direction, "salience": 1.0})
    return out


# magnitude -> salience: monotonic only (beta absorbs absolute scale, so exact values
# don't matter; large>medium>small and the sign are what count). unclear keeps a small
# nonzero tilt since the direction is still stated; flat contributes nothing.
_MAGNITUDE_SALIENCE = {"large": 1.0, "medium": 0.67, "small": 0.34, "unclear": 0.34, "flat": 0.0}


def _canonical_factor(market: str, valid: frozenset[str]) -> str | None:
    """Map a grounding `market` string to a canonical joint39 anchor factor, or None."""
    m = str(market).strip().upper()
    if m in valid:
        return m
    low = str(market).strip().lower()
    for factor, aliases in _ALIASES.items():
        if factor in valid and any(a in low for a in aliases):
            return factor
    return None  # e.g. IV_SURFACE (not a single anchor) -> dropped


def _emphasis_from_structured(implications: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read the grounding's native ConditionMarketImplication dicts (market/direction/magnitude).

    This is the real grounding shape (verified across 10.8k on-disk implications), strictly
    higher-fidelity than prose-parsing. mixed/unclear/flat directions contribute no tilt.
    On duplicate factors, keep the higher-salience implication.
    """
    valid = _valid_factors()
    out: dict[str, dict[str, Any]] = {}
    for it in implications:
        if not isinstance(it, dict):
            continue
        factor = _canonical_factor(it.get("market", ""), valid)
        if factor is None:
            continue
        direction = str(it.get("direction", "")).strip().lower()
        if dir_sign(direction) == 0:  # flat / mixed / unclear / empty -> no directional tilt
            continue
        sal = _MAGNITUDE_SALIENCE.get(str(it.get("magnitude", "")).strip().lower(), 0.5)
        if sal <= 0.0:
            continue
        prev = out.get(factor)
        if prev is None or sal > prev["salience"]:
            out[factor] = {"direction": direction, "salience": float(sal)}
    return out


def narrative_emphasis(grounding_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Narrative -> {factor: {direction, salience}}. Priority:
    (1) explicit `factor_emphasis` dict (forward-compat; not emitted by current schema);
    (2) the grounding's native structured `current_market_state_implications` dicts (the real,
        verified shape: market+direction+magnitude) -- primary path;
    (3) legacy prose-parse when implications are plain strings;
    (4) {} identity (safe-degrade). Scope: current state only (not recent_regime).
    """
    g = grounding_output or {}
    cog = g.get("condition_only_grounding", g)
    structured = cog.get("factor_emphasis") or g.get("factor_emphasis")
    if isinstance(structured, dict) and structured:
        return {str(k).upper(): {"direction": str(v.get("direction", "")),
                                 "salience": float(v.get("salience", 1.0))}
                for k, v in structured.items()}
    implications = list(cog.get("current_market_state_implications") or [])
    if any(isinstance(it, dict) and "market" in it for it in implications):
        return _emphasis_from_structured(implications)
    return _emphasis_from_implications([str(x) for x in implications])


def reweight_candidate_scores(candidates, emphasis, profiles_by_window, beta: float):
    """Return copies of candidates with score += beta*match(emphasis, analogue_profile).
    beta=0 -> exact no-op. Records the additive tilt for transparency."""
    b = float(beta)
    out = []
    for cand in candidates:
        row = copy.deepcopy(cand)
        prof = profiles_by_window.get(int(cand["window_index"]), {})
        m = match(emphasis, prof)
        row["t7_match"] = float(m)
        row["t7_tilt"] = float(b * m)
        row["score"] = float(cand["score"]) + b * m
        out.append(row)
    return out


def reweight_pool(candidates, *, emphasis, profiles_by_window, beta: float):
    """Narrative-tilt a retrieved pool, ready for _apply_top3_90. beta=0 -> identity."""
    return reweight_candidate_scores(candidates, emphasis, profiles_by_window, beta)
