"""T7: training-free narrative-conditioned reweighter over the retrieved analogue pool.

Adjusts each candidate's score by beta * match(narrative_emphasis, analogue_factor_profile)
BEFORE the existing _apply_top3_90 softmax. beta=0 is an exact no-op. No trained text->latent
head (sidesteps the 992b/T4 information ceiling). See
docs/research_protocols/nl-prefix-latent-t7-narrative-reweighter-intake.md.
"""
from __future__ import annotations

from typing import Any

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
