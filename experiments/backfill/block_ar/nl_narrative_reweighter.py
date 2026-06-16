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
