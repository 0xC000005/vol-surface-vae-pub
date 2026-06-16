"""T7: training-free narrative-conditioned reweighter over the retrieved analogue pool.

Adjusts each candidate's score by beta * match(narrative_emphasis, analogue_factor_profile)
BEFORE the existing _apply_top3_90 softmax. beta=0 is an exact no-op. No trained text->latent
head (sidesteps the 992b/T4 information ceiling). See
docs/research_protocols/nl-prefix-latent-t7-narrative-reweighter-intake.md.
"""
from __future__ import annotations

from typing import Any

from experiments.backfill.block_ar.nl_joint39_anchor_map import joint39_anchor_columns

_POS = {"up", "wider"}
_NEG = {"down", "tighter"}


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
    factors = set(joint39_anchor_columns().keys())
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


def narrative_emphasis(grounding_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Structured `factor_emphasis` if present; else parse grounding prose; else {} (identity)."""
    g = grounding_output or {}
    cog = g.get("condition_only_grounding", g)
    structured = cog.get("factor_emphasis") or g.get("factor_emphasis")
    if isinstance(structured, dict) and structured:
        return {str(k).upper(): {"direction": str(v.get("direction", "")),
                                 "salience": float(v.get("salience", 1.0))}
                for k, v in structured.items()}
    implications = cog.get("current_market_state_implications") or []
    return _emphasis_from_implications(list(implications))
