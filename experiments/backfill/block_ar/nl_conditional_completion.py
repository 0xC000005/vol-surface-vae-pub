"""BJRS conditional-completion core (framework-v1 §B coherence-check + §I hull-gate target).

Given the historical 30-day-move covariance Sigma over the factor panel and a narrative-
pinned factor set S with target values x_S, the Gaussian conditional mean of the OTHER
factors (assuming zero-mean moves) is

    mu_{-S|S} = Sigma_{-S,S} Sigma_{S,S}^{-1} x_S .

This is the closed-form "what every non-targeted factor should do if S moves like this".
Two ratified uses: (1) §B coherence quick-check — the conditioned ensemble's tilt sign on
each free factor must agree with sign(mu_{-S|S}) where |conditional beta| clears a noise
floor; (2) §I hull-honesty gate — the completion target whose representability inside the
analogue pool's convex hull is checked. Validation-side only; never enters the conditioning
path (Bitter-Lesson clean — it is a property of historical data, not a learned constant in
the generator).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def conditional_completion(
    cov: np.ndarray,
    pinned_indices: Sequence[int],
    pinned_values: Sequence[float],
) -> dict[str, Any]:
    """Return the Gaussian conditional completion of the non-pinned factors.

    cov: (D, D) covariance of factor moves (symmetric PSD).
    pinned_indices: indices of the narrative-pinned factor set S.
    pinned_values: target move values x_S (same order as pinned_indices).

    Returns dict with:
      completion        (D,)            x_S at S, mu_{-S|S} at the free factors
      conditional_betas (n_free, n_pin) Sigma_{-S,S} Sigma_{S,S}^{-1}
      pinned_indices, free_indices
    """
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    if cov.ndim != 2 or cov.shape[1] != d:
        raise ValueError(f"cov must be square (D, D); got {cov.shape}")
    s = [int(i) for i in pinned_indices]
    x_s = np.asarray(pinned_values, dtype=np.float64).reshape(-1)
    if len(s) != x_s.shape[0]:
        raise ValueError("pinned_indices and pinned_values must align")
    if len(set(s)) != len(s) or any(i < 0 or i >= d for i in s):
        raise ValueError("pinned_indices must be unique and in range")
    free = [i for i in range(d) if i not in set(s)]

    completion = np.zeros(d, dtype=np.float64)
    completion[s] = x_s

    if free and s:
        cov_ss = cov[np.ix_(s, s)]
        cov_fs = cov[np.ix_(free, s)]
        # betas = cov_fs @ inv(cov_ss); solve cov_ss X = cov_fs^T -> X = inv(cov_ss) cov_fs^T
        x = np.linalg.solve(cov_ss, cov_fs.T)  # (n_pin, n_free)
        betas = x.T  # (n_free, n_pin)
        completion[free] = betas @ x_s
    else:
        betas = np.zeros((len(free), len(s)), dtype=np.float64)

    return {
        "completion": completion,
        "conditional_betas": betas,
        "pinned_indices": s,
        "free_indices": free,
    }
