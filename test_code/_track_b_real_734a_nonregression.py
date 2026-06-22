"""Real-734a non-regression proof for the Track B narrative-conditioning channel.

Loads the actual frozen 734a checkpoint via ``load_model`` two ways:
  - baseline  : default config (narrative_conditioning=False)
  - conditioned: cfg_overrides={"narrative_conditioning": True} (adapter inits fresh,
                 load_state_dict(strict=False) tolerates the missing adapter keys via the
                 ``narrative_adapter`` new-head marker)
Then runs a fixed-seed ``sample_batched`` with ``narrative_emb=None`` on both and asserts
max|diff| == 0.0 — proving the tri-scope/NL incumbent inference path is untouched.

The checkpoint is only READ. This script never writes under models/.
"""

from __future__ import annotations

import numpy as np
import torch

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    load_model,
)

CKPT = (
    "models/backfill/734a_joint39_realvix_channel_level_alltrain_w005_e3_s7345/"
    "best_model.pt"
)
SEED = 4242


def _inputs(model, device, bsz=2):
    cfg = model.cfg
    g = torch.Generator(device=device).manual_seed(7)
    # level_quantiles is loaded from the checkpoint; draw level values inside its support
    # range so level_values_to_scores stays in-table. Use the loaded table's min/max.
    table = model.level_quantiles  # (n_cells, n_quantiles)
    lo = table.min(dim=1).values
    hi = table.max(dim=1).values
    u = torch.rand(bsz, cfg.history_len, cfg.n_cells, generator=g, device=device)
    history_level = lo[None, None, :] + u * (hi - lo)[None, None, :]
    history_innov = torch.randn(
        bsz, cfg.history_len, cfg.n_cells, generator=g, device=device
    )
    center = lo[None, :] + torch.rand(bsz, cfg.n_cells, generator=g, device=device) * (hi - lo)[None, :]
    scale = torch.rand(bsz, cfg.n_cells, generator=g, device=device) + 0.5
    return history_level, history_innov, center, scale


def _sample(model, inputs):
    history_level, history_innov, center, scale = inputs
    torch.manual_seed(SEED)  # seed immediately before the call (base-noise RNG)
    return model.sample_batched(
        history_level,
        history_innov,
        center,
        scale,
        n_samples=8,
        n_steps=int(model.cfg.future_len),
        chunk_size=4,
        narrative_emb=None,
    )


def main() -> None:
    device = torch.device("cpu")
    baseline, _ = load_model(CKPT, device)  # default => narrative_conditioning=False
    conditioned, _ = load_model(
        CKPT, device, cfg_overrides={"narrative_conditioning": True}
    )

    assert baseline.narrative_adapter is None, "baseline should have no adapter"
    assert conditioned.narrative_adapter is not None, "conditioned should have an adapter"
    print(f"baseline cfg.narrative_conditioning = {baseline.cfg.narrative_conditioning}")
    print(f"conditioned cfg.narrative_conditioning = {conditioned.cfg.narrative_conditioning}")
    print(f"conditioned memory_dim = {conditioned.cfg.memory_dim}, "
          f"narrative_dim = {conditioned.cfg.narrative_dim}, "
          f"narrative_hidden = {conditioned.cfg.narrative_hidden}, "
          f"risk_state_dim = {conditioned.cfg.risk_state_dim}")

    inputs = _inputs(baseline, device)
    out_base = _sample(baseline, inputs)
    out_cond = _sample(conditioned, inputs)

    max_diff = (out_base - out_cond).abs().max().item()
    print(f"output shape = {tuple(out_base.shape)}")
    print(f"REAL-734a non-regression max|diff| = {max_diff}")
    assert max_diff == 0.0, f"NON-REGRESSION FAILED: max|diff|={max_diff}"
    print("PASS: real-734a narrative_emb=None path is byte-identical (max|diff| = 0.0)")


if __name__ == "__main__":
    main()
