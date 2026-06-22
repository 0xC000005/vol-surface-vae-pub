"""Track B: NarrativeConditioningAdapter unit tests (TDD).

The adapter maps a high-dim narrative embedding to an additive memory_dim context,
mirroring the SNI model's existing risk_context channel. Zero-init final layer =>
exact no-op at init (so the conditioned model is byte-identical to frozen 734a until
trained). Presence-gated: an absent narrative => zero context => baseline.
High-dim hidden (>= memory_dim) deliberately avoids the Exp-153a 128-d bottleneck
that killed window-specific steering.
"""

import torch

from diffusion.block_ar.narrative_conditioning_adapter import NarrativeConditioningAdapter


def test_zero_init_is_noop():
    adapter = NarrativeConditioningAdapter(narrative_dim=1536, memory_dim=128, hidden=256)
    emb = torch.randn(4, 1536)
    present = torch.ones(4, dtype=torch.bool)
    ctx = adapter(emb, present)
    assert ctx.shape == (4, 128)
    # zero-init final layer => exact zero context at init (exact 734a no-op)
    assert torch.allclose(ctx, torch.zeros_like(ctx), atol=0)


def test_absent_narrative_is_zero():
    adapter = NarrativeConditioningAdapter(narrative_dim=1536, memory_dim=128, hidden=256)
    # perturb params so a present narrative would produce a nonzero context
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(0.01)
    emb = torch.randn(2, 1536)
    present = torch.tensor([True, False])
    ctx = adapter(emb, present)
    assert torch.count_nonzero(ctx[1]) == 0  # absent => zero context (baseline)
    assert torch.count_nonzero(ctx[0]) > 0  # present => nonzero


def test_no_low_dim_bottleneck_enforced():
    # the 153a mandate: hidden must not compress below memory_dim
    raised = False
    try:
        NarrativeConditioningAdapter(narrative_dim=1536, memory_dim=128, hidden=64)
    except AssertionError:
        raised = True
    assert raised
