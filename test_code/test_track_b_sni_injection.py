"""Track B Task 3 non-regression tests for the optional narrative-conditioning channel.

These tests prove that wiring the additive ``NarrativeConditioningAdapter`` into the
frozen SNI generator (``generic_state_aware_normalized_innovation_flow_matching``) is:

  (a) byte-identical when the feature is OFF (``narrative_conditioning=False``) vs a model
      built with the feature ON but ``narrative_emb=None`` (gated off);
  (b) byte-identical when ON with a NONZERO narrative + ``narrative_present=all True`` while
      the adapter is still zero-init (zero-init context == 0 => exact no-op);
  (c) DIFFERENT once the adapter's final-layer weights are perturbed to nonzero with a
      present narrative (the channel is genuinely live).

The small config uses ``risk_state_dim=0`` on purpose: the real 734a checkpoint has it 0, so
``risk_context`` is ``None`` at runtime. If the narrative injection were nested inside the
``if risk_context is not None:`` guard it would be dead code on 734a; test (c) only passes
when the injection sits at the unconditional level, so it doubles as a placement guard.
"""

from __future__ import annotations

import torch

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (
    GenericStateAwareNormalizedInnovationFMConfig,
    GenericStateAwareNormalizedInnovationFlowMatching,
)

SEED = 1234
N_CELLS = 4
HISTORY_LEN = 5
FUTURE_LEN = 4
MEMORY_DIM = 16
N_QUANTILES = 41
NARRATIVE_DIM = 32
NARRATIVE_HIDDEN = 16
BSZ = 3
N_SAMPLES = 4


def _small_config(**overrides) -> GenericStateAwareNormalizedInnovationFMConfig:
    base = dict(
        n_cells=N_CELLS,
        history_len=HISTORY_LEN,
        future_len=FUTURE_LEN,
        memory_dim=MEMORY_DIM,
        token_dim=MEMORY_DIM,
        hidden_dim=MEMORY_DIM,
        memory_layers=1,
        memory_heads=2,
        memory_ff=32,
        token_layers=1,
        token_heads=2,
        token_ff=32,
        n_quantiles=N_QUANTILES,
        flow_steps=4,
        base_noise_rho=0.0,
        risk_state_dim=0,
        conditional_base_noise_scale=False,
        velocity_mixer_mode="transformer",
        innovation_coordinate="normalized",
        narrative_dim=NARRATIVE_DIM,
        narrative_hidden=NARRATIVE_HIDDEN,
    )
    base.update(overrides)
    return GenericStateAwareNormalizedInnovationFMConfig(**base)


def _prime_quantiles(model: GenericStateAwareNormalizedInnovationFlowMatching) -> None:
    """Install a monotonic per-cell level-quantile table so level_values_to_scores works."""
    levels = (torch.arange(N_QUANTILES, dtype=torch.float32) + 0.5) / float(N_QUANTILES)
    # Monotonic increasing along the quantile axis, distinct per cell.
    table = levels[None, :].repeat(N_CELLS, 1) + torch.arange(N_CELLS)[:, None] * 0.01
    model.set_level_quantiles(table, levels)


def _build_model(narrative_conditioning: bool) -> GenericStateAwareNormalizedInnovationFlowMatching:
    """Deterministic construction: seed first so both models get identical backbone weights."""
    torch.manual_seed(SEED)
    cfg = _small_config(narrative_conditioning=narrative_conditioning)
    model = GenericStateAwareNormalizedInnovationFlowMatching(cfg)
    _prime_quantiles(model)
    model.eval()
    return model


def _copy_backbone(
    src: GenericStateAwareNormalizedInnovationFlowMatching,
    dst: GenericStateAwareNormalizedInnovationFlowMatching,
) -> None:
    """Copy every shared parameter/buffer from ``src`` into ``dst`` so the ONLY difference
    between the two models is the presence (and weights) of the narrative adapter."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    overlap = {k: v for k, v in src_state.items() if k in dst_state}
    missing = dst.load_state_dict(overlap, strict=False)
    # The only keys allowed to be missing on dst are the narrative adapter's (new head).
    assert all(
        "narrative_adapter" in k for k in missing.missing_keys
    ), f"unexpected missing keys: {missing.missing_keys}"


def _inputs(device: torch.device):
    g = torch.Generator(device=device).manual_seed(99)
    history_level = torch.rand(BSZ, HISTORY_LEN, N_CELLS, generator=g, device=device)
    history_innov = torch.randn(BSZ, HISTORY_LEN, N_CELLS, generator=g, device=device)
    center = torch.rand(BSZ, N_CELLS, generator=g, device=device)
    scale = torch.rand(BSZ, N_CELLS, generator=g, device=device) + 0.5
    return history_level, history_innov, center, scale


def _sample(model, inputs, narrative_emb=None, narrative_present=None):
    history_level, history_innov, center, scale = inputs
    torch.manual_seed(SEED)  # seed IMMEDIATELY before the call (controls base-noise RNG)
    return model.sample_batched(
        history_level,
        history_innov,
        center,
        scale,
        n_samples=N_SAMPLES,
        n_steps=FUTURE_LEN,
        chunk_size=2,
        narrative_emb=narrative_emb,
        narrative_present=narrative_present,
    )


def test_off_vs_on_with_none_narrative_byte_identical():
    """(a) Feature OFF == feature ON with narrative_emb=None (the channel is gated off)."""
    device = torch.device("cpu")
    model_off = _build_model(narrative_conditioning=False)
    model_on = _build_model(narrative_conditioning=True)
    _copy_backbone(model_off, model_on)  # identical backbone; only difference = adapter
    inputs = _inputs(device)

    out_off = _sample(model_off, inputs)
    out_on = _sample(model_on, inputs, narrative_emb=None)

    max_diff = (out_off - out_on).abs().max().item()
    assert max_diff == 0.0, f"OFF vs ON(None) differ: max|diff|={max_diff}"


def test_on_zero_init_present_narrative_byte_identical():
    """(b) Feature ON, zero-init adapter, NONZERO narrative + present=True => still no-op."""
    device = torch.device("cpu")
    model_off = _build_model(narrative_conditioning=False)
    model_on = _build_model(narrative_conditioning=True)
    _copy_backbone(model_off, model_on)
    inputs = _inputs(device)

    emb = torch.randn(BSZ, NARRATIVE_DIM)
    present = torch.ones(BSZ, dtype=torch.bool)

    out_off = _sample(model_off, inputs)
    out_on = _sample(model_on, inputs, narrative_emb=emb, narrative_present=present)

    max_diff = (out_off - out_on).abs().max().item()
    assert max_diff == 0.0, f"zero-init present narrative perturbed output: max|diff|={max_diff}"


def _training_loss(model, inputs, narrative_emb=None, narrative_present=None):
    history_level, history_innov, center, scale = inputs
    # Build a matching future tensor (same shape as history minus history_len, future_len).
    g = torch.Generator().manual_seed(55)
    future_level = history_level[:, :FUTURE_LEN].clone()
    future_innov = torch.randn(BSZ, FUTURE_LEN, N_CELLS, generator=g)
    torch.manual_seed(SEED)  # controls the flow-matching base-noise / t sampling RNG
    loss, _aux = model.training_loss(
        history_level,
        history_innov,
        future_level,
        future_innov,
        center,
        scale,
        narrative_emb=narrative_emb,
        narrative_present=narrative_present,
    )
    return loss


def test_training_loss_off_vs_on_none_byte_identical():
    """training_loss injection coverage: OFF == ON with narrative_emb=None (bit-equal loss).

    The (a)/(b)/(c) tests only exercise sample_batched; this gives the training_loss
    injection (line ~1022, unconditional 8-space) its own executable non-regression check."""
    model_off = _build_model(narrative_conditioning=False)
    model_on = _build_model(narrative_conditioning=True)
    _copy_backbone(model_off, model_on)
    inputs = _inputs(torch.device("cpu"))

    loss_off = _training_loss(model_off, inputs)
    loss_on = _training_loss(model_on, inputs, narrative_emb=None)

    max_diff = (loss_off - loss_on).abs().item()
    assert max_diff == 0.0, f"training_loss OFF vs ON(None) differ: max|diff|={max_diff}"


def test_perturbed_adapter_changes_output():
    """(c) Perturb the adapter final layer => present narrative now moves the output.

    This is the live-channel proof AND the unconditional-placement guard (risk_state_dim=0
    means a nested injection would never fire on this config)."""
    device = torch.device("cpu")
    model_off = _build_model(narrative_conditioning=False)
    model_on = _build_model(narrative_conditioning=True)
    _copy_backbone(model_off, model_on)
    inputs = _inputs(device)

    emb = torch.randn(BSZ, NARRATIVE_DIM)
    present = torch.ones(BSZ, dtype=torch.bool)

    baseline = _sample(model_on, inputs, narrative_emb=emb, narrative_present=present)

    with torch.no_grad():
        final = model_on.narrative_adapter.net[-1]
        final.weight.add_(0.1)
        final.bias.add_(0.1)

    perturbed = _sample(model_on, inputs, narrative_emb=emb, narrative_present=present)

    max_diff = (baseline - perturbed).abs().max().item()
    assert max_diff > 0.0, f"live channel did not move output: max|diff|={max_diff}"
