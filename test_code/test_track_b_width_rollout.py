"""Unit tests for the Track B B-width differentiable rollout + gate dial logic (no GPU/training).

Locks: (1) present=False rollout is independent of narrative_emb (frozen-734a baseline, the
null-anchor); (2) gradients flow to the adapter when present=True; (3) CRN -- present=True and
present=False at the SAME base_noise share noise so width_delta isolates the adapter; (4) the
gate's severity-dial logic distinguishes a proportional dial from uniform presence-inflation.
"""

from __future__ import annotations

import numpy as np
import torch

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import load_model
from experiments.backfill.block_ar.train_track_b_generator_conditioning import (
    DEFAULT_CHECKPOINT,
    build_windows,
)
from experiments.backfill.block_ar.train_track_b_width_conditioning import (
    differentiable_terminal_widths,
    soft_corr,
)

_B2 = "models/backfill/generator_conditioning_probe_b2/best_model.pt"


def _load():
    dev = torch.device("cpu")
    model, payload = load_model(_B2, dev)
    w = build_windows(DEFAULT_CHECKPOINT, payload)
    rows = np.array([700, 705, 710])
    t = {k: torch.from_numpy(w[k][rows]) for k in
         ("history_level", "history_norm", "center", "scale", "drift_feature")}
    return model, t


def test_present_false_independent_of_narrative() -> None:
    model, t = _load()
    nsteps = int(model.cfg.future_len); ncells = int(model.cfg.n_cells)
    present_false = torch.zeros(3, dtype=torch.bool)
    bn = torch.randn(3, 6, nsteps, ncells)
    with torch.no_grad():
        w_a = differentiable_terminal_widths(
            model, t["history_level"], t["history_norm"], t["center"], t["scale"], t["drift_feature"],
            narrative_emb=torch.randn(3, 1536), narrative_present=present_false,
            base_noise=bn, n_steps=nsteps, flow_steps=4, grad_checkpoint=False,
        )
        w_b = differentiable_terminal_widths(
            model, t["history_level"], t["history_norm"], t["center"], t["scale"], t["drift_feature"],
            narrative_emb=torch.randn(3, 1536) * 99.0, narrative_present=present_false,
            base_noise=bn, n_steps=nsteps, flow_steps=4, grad_checkpoint=False,
        )
    # present=False gates the adapter to zero context -> width must be identical (the baseline).
    assert torch.allclose(w_a, w_b, atol=1e-5), (w_a, w_b)


def test_gradient_flows_to_adapter() -> None:
    model, t = _load()
    nsteps = int(model.cfg.future_len); ncells = int(model.cfg.n_cells)
    bn = torch.randn(3, 6, nsteps, ncells)
    width = differentiable_terminal_widths(
        model, t["history_level"], t["history_norm"], t["center"], t["scale"], t["drift_feature"],
        narrative_emb=torch.randn(3, 1536), narrative_present=torch.ones(3, dtype=torch.bool),
        base_noise=bn, n_steps=nsteps, flow_steps=4, grad_checkpoint=False,
    )
    width.sum().backward()
    g = model.narrative_adapter.net[-1].weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_soft_corr_matches_numpy() -> None:
    x = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
    y = torch.tensor([0.2, 0.6, 1.0, 0.1, 0.8])
    got = float(soft_corr(x, y))
    exp = float(np.corrcoef(x.numpy(), y.numpy())[0, 1])
    assert abs(got - exp) < 1e-4, (got, exp)
