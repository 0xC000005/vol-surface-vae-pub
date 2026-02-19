"""Unit tests for Block-AR Diffusion components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from diffusion.block_ar.bigru_denoiser import BiGRUDenoiser, DenoiserConfig, SpatialStream
from diffusion.block_ar.masking import MCVDTask, sample_mcvd_masks, get_task_types
from diffusion.block_ar.noise_schedules import sample_task_adaptive_noise, sample_batch_task_adaptive_noise


def test_encoder_shape():
    enc = GRUEncoder(EncoderConfig())
    x = torch.randn(4, 30, 5, 5)
    out = enc(x)
    bdim = EncoderConfig().bottleneck_dim
    assert out.shape == (4, bdim), f"Expected (4, {bdim}), got {out.shape}"
    print(f"  PASS: encoder shape (4, 30, 5, 5) -> (4, {bdim})")


def test_encoder_variable_length():
    enc = GRUEncoder(EncoderConfig())
    bdim = EncoderConfig().bottleneck_dim
    for T in [30, 40, 60]:
        x = torch.randn(4, T, 5, 5)
        out = enc(x)
        assert out.shape == (4, bdim), f"T={T}: Expected (4, {bdim}), got {out.shape}"
    print("  PASS: encoder handles variable lengths (30, 40, 60)")


def test_encoder_masking():
    enc = GRUEncoder(EncoderConfig())
    x = torch.randn(4, 30, 5, 5)
    mask = torch.ones(4, dtype=torch.bool)  # all masked
    out = enc(x, mask=mask)
    # All should equal null_embedding
    null = enc.null_embedding.expand(4, -1)
    assert torch.allclose(out, null.detach()), "Masked output should equal null_embedding"
    print("  PASS: encoder masking returns null_embedding")


def test_encoder_per_item_masking():
    enc = GRUEncoder(EncoderConfig())
    enc.eval()  # disable augmentation for deterministic check
    x = torch.randn(4, 30, 5, 5)
    mask = torch.tensor([True, False, True, False])
    out = enc(x, mask=mask)
    null = enc.null_embedding.squeeze(0)
    assert torch.allclose(out[0], null.detach()), "Item 0 (masked) should be null"
    assert torch.allclose(out[2], null.detach()), "Item 2 (masked) should be null"
    assert not torch.allclose(out[1], null.detach()), "Item 1 (unmasked) should differ from null"
    print("  PASS: per-item masking works correctly")


def test_encoder_cond_augmentation():
    enc = GRUEncoder(EncoderConfig())
    x = torch.randn(4, 30, 5, 5)
    # Train mode: stochastic
    enc.train()
    torch.manual_seed(42)
    out1 = enc(x)
    torch.manual_seed(43)
    out2 = enc(x)
    assert not torch.allclose(out1, out2), "Train mode should be stochastic"
    # Eval mode: deterministic
    enc.eval()
    out3 = enc(x)
    out4 = enc(x)
    assert torch.allclose(out3, out4), "Eval mode should be deterministic"
    print("  PASS: conditioning augmentation stochastic in train, deterministic in eval")


def test_denoiser_shape():
    dec = BiGRUDenoiser(DenoiserConfig())
    noisy = torch.randn(4, 10, 25)
    bdim = DenoiserConfig().bottleneck_dim
    cond = torch.randn(4, bdim)
    pos = torch.arange(10).unsqueeze(0).expand(4, -1)
    k = torch.randint(0, 100, (4, 10))
    out = dec(noisy, cond, pos, k)
    assert out.shape == (4, 10, 25), f"Expected (4, 10, 25), got {out.shape}"
    print("  PASS: denoiser shape (4, 10, 25)")


def test_gradient_flow():
    enc = GRUEncoder(EncoderConfig())
    dec = BiGRUDenoiser(DenoiserConfig())
    x = torch.randn(4, 30, 5, 5)
    # Use partial masking so null_embedding gets gradients too
    mask = torch.tensor([True, False, True, False])
    cond = enc(x, mask=mask)
    noisy = torch.randn(4, 10, 25)
    pos = torch.arange(10).unsqueeze(0).expand(4, -1)
    k = torch.randint(0, 100, (4, 10))
    out = dec(noisy, cond, pos, k)
    loss = out.mean()
    loss.backward()
    for name, p in list(enc.named_parameters()) + list(dec.named_parameters()):
        assert p.grad is not None, f"No gradient for {name}"
    print("  PASS: gradient flow through all parameters (including null_embedding)")


def test_param_count():
    enc = GRUEncoder(EncoderConfig())
    dec = BiGRUDenoiser(DenoiserConfig())
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    total = enc_params + dec_params
    print(f"  INFO: Encoder params: {enc_params:,}")
    print(f"  INFO: Denoiser params: {dec_params:,}")
    print(f"  INFO: Total params: {total:,}")
    # Dual-path architecture: ~303K total (BiGRU ~282K + SpatialStream ~20K)
    assert 290_000 < total < 320_000, f"Total params {total:,} outside expected range [290K, 320K]"
    print(f"  PASS: param count {total:,} in expected range [290K, 320K]")


def test_spatial_stream_shape():
    """SpatialStream produces (B, T, mid_ch * H * W) output."""
    cfg = DenoiserConfig()
    cond_dim = cfg.noise_embed_dim + cfg.bottleneck_dim
    stream = SpatialStream(
        h=cfg.surface_h, w=cfg.surface_w,
        mid_channels=cfg.spatial_mid_channels, cond_dim=cond_dim,
    )
    B, T = 4, 10
    frames = torch.randn(B, T, 25)
    noise_emb = torch.randn(B, T, cfg.noise_embed_dim)
    condition = torch.randn(B, cfg.bottleneck_dim)
    out = stream(frames, noise_emb, condition)
    expected_dim = cfg.spatial_mid_channels * cfg.surface_h * cfg.surface_w
    assert out.shape == (B, T, expected_dim), (
        f"Expected ({B}, {T}, {expected_dim}), got {out.shape}"
    )
    print(f"  PASS: spatial stream shape ({B}, {T}, {expected_dim})")


def test_spatial_stream_noise_sensitivity():
    """AdaGN makes spatial stream output depend on noise level after training.

    At init, AdaGN projections are zero (near-identity), so we perturb them
    to simulate a trained model. Different noise embeddings should then
    produce different spatial outputs via the AdaGN scale/shift path.
    """
    cfg = DenoiserConfig()
    cond_dim = cfg.noise_embed_dim + cfg.bottleneck_dim
    stream = SpatialStream(
        h=cfg.surface_h, w=cfg.surface_w,
        mid_channels=cfg.spatial_mid_channels, cond_dim=cond_dim,
    )
    # Perturb AdaGN projections away from zero-init to simulate trained model
    with torch.no_grad():
        stream.ada_proj1.weight.normal_(std=0.1)
        stream.ada_proj2.weight.normal_(std=0.1)
    stream.eval()

    B, T = 2, 10
    frames = torch.randn(B, T, 25)
    condition = torch.randn(B, cfg.bottleneck_dim)

    # Two different noise embeddings (simulating low vs high noise levels)
    noise_emb_low = torch.zeros(B, T, cfg.noise_embed_dim)
    noise_emb_high = torch.ones(B, T, cfg.noise_embed_dim) * 5.0

    out_low = stream(frames, noise_emb_low, condition)
    out_high = stream(frames, noise_emb_high, condition)

    diff = (out_low - out_high).abs().mean().item()
    assert diff > 1e-6, f"Spatial stream output unchanged by noise level (diff={diff:.2e})"
    print(f"  PASS: spatial stream is noise-level-sensitive (mean diff={diff:.4f})")


# Phase 2 tests

def test_task_distribution():
    counts = {t: 0 for t in MCVDTask}
    N = 10000
    for _ in range(N):
        mp, mf = sample_mcvd_masks(1, p_mask=0.5)
        task = get_task_types(mp, mf).item()
        counts[MCVDTask(task)] += 1
    for task, count in counts.items():
        pct = count / N * 100
        assert 20 < pct < 30, f"{task.name}: {pct:.1f}% not in [20%, 30%]"
    print(f"  PASS: task distribution ~25% each ({', '.join(f'{t.name}:{c/N*100:.1f}%' for t, c in counts.items())})")


def test_forward_mean_increasing():
    N = 1000
    k_all = torch.zeros(N, 10)
    for i in range(N):
        k_all[i] = sample_task_adaptive_noise(MCVDTask.FORWARD, 1, 10, 100).float()
    means = k_all.mean(dim=0)
    for i in range(9):
        assert means[i] < means[i + 1], f"Forward: E[k[{i}]]={means[i]:.1f} >= E[k[{i+1}]]={means[i+1]:.1f}"
    print(f"  PASS: forward mean increasing ({means[0]:.1f} -> {means[-1]:.1f})")


def test_backward_mean_decreasing():
    N = 1000
    k_all = torch.zeros(N, 10)
    for i in range(N):
        k_all[i] = sample_task_adaptive_noise(MCVDTask.BACKWARD, 1, 10, 100).float()
    means = k_all.mean(dim=0)
    for i in range(9):
        assert means[i] > means[i + 1], f"Backward: E[k[{i}]]={means[i]:.1f} <= E[k[{i+1}]]={means[i+1]:.1f}"
    print(f"  PASS: backward mean decreasing ({means[0]:.1f} -> {means[-1]:.1f})")


def test_interpolation_tent():
    N = 1000
    k_all = torch.zeros(N, 10)
    for i in range(N):
        k_all[i] = sample_task_adaptive_noise(MCVDTask.INTERPOLATION, 1, 10, 100).float()
    means = k_all.mean(dim=0)
    mid = len(means) // 2
    assert means[0] < means[mid], f"Tent: edges ({means[0]:.1f}) should be < middle ({means[mid]:.1f})"
    assert means[-1] < means[mid], f"Tent: edges ({means[-1]:.1f}) should be < middle ({means[mid]:.1f})"
    print(f"  PASS: interpolation tent shape (edges {means[0]:.1f},{means[-1]:.1f} < middle {means[mid]:.1f})")


def test_noise_valid_range():
    for task in MCVDTask:
        k = sample_task_adaptive_noise(task, 100, 10, 100)
        assert k.min() >= 0, f"{task.name}: min k = {k.min()}"
        assert k.max() < 100, f"{task.name}: max k = {k.max()}"
    print("  PASS: all noise levels in [0, 100)")


def test_jitter_nonzero():
    for task in [MCVDTask.FORWARD, MCVDTask.BACKWARD, MCVDTask.INTERPOLATION]:
        k = sample_task_adaptive_noise(task, 1000, 10, 100)
        stds = k.float().std(dim=0)
        assert stds.min() > 0, f"{task.name}: some frames have zero jitter"
    print("  PASS: jitter nonzero for structured tasks")


def test_batch_task_adaptive():
    mp = torch.tensor([False, True, False, True])  # [FORWARD/INTERP, BACKWARD/UNCOND, ...]
    mf = torch.tensor([True, False, False, True])
    tasks = get_task_types(mp, mf)
    k = sample_batch_task_adaptive_noise(mp, mf, 10, 100)
    assert k.shape == (4, 10)
    assert k.min() >= 0 and k.max() < 100
    print(f"  PASS: batch task-adaptive noise (tasks: {[MCVDTask(t.item()).name for t in tasks]})")


# Phase 3 tests — ConditionalBlockARDDPM wrapper

from diffusion.block_ar.block_ar_ddpm import ConditionalBlockARDDPM, BlockARConfig
from diffusion.block_ar.conv3d_denoiser import Conv3DBlockDenoiser, Conv3DDenoiserConfig


def test_forward_produces_scalar_loss():
    config = BlockARConfig(n_steps=20)
    model = ConditionalBlockARDDPM(config)
    model.train()
    history = torch.randn(2, 30, 5, 5)
    future = torch.randn(2, 30, 5, 5)
    result = model(history, future)
    assert "loss" in result, "Forward should return dict with 'loss'"
    assert result["loss"].shape == (), f"Loss should be scalar, got {result['loss'].shape}"
    assert result["loss"].item() > 0, "Loss should be positive"
    print(f"  PASS: forward produces scalar loss ({result['loss'].item():.4f})")


def test_forward_all_params_have_grad():
    config = BlockARConfig(n_steps=20)
    model = ConditionalBlockARDDPM(config)
    model.train()
    history = torch.randn(2, 30, 5, 5)
    future = torch.randn(2, 30, 5, 5)
    result = model(history, future)
    result["loss"].backward()
    no_grad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(name)
    assert len(no_grad) == 0, f"Parameters without gradient: {no_grad}"
    print("  PASS: all model parameters have gradients after forward")


def test_synthetic_training_loss_decreases():
    """50-step synthetic training — loss should decrease >10%.

    Block-AR has MCVD random masking each step, so convergence is slower.
    Use p_mask=0 (always FORWARD) for deterministic training signal.
    """
    config = BlockARConfig(n_steps=20, bigru_hidden_dim=64, gru_hidden_dim=32, p_mask=0.0)
    model = ConditionalBlockARDDPM(config)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Fixed synthetic data (so model can memorize)
    torch.manual_seed(42)
    history = torch.randn(4, 30, 5, 5)
    future = torch.randn(4, 30, 5, 5)

    losses = []
    for step in range(50):
        result = model(history, future)
        loss = result["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    pct_decrease = (losses[0] - losses[-1]) / losses[0] * 100
    assert pct_decrease > 10, f"Loss decreased only {pct_decrease:.1f}% (need >10%)"
    print(f"  PASS: synthetic training loss decreased {pct_decrease:.1f}% ({losses[0]:.4f} -> {losses[-1]:.4f})")


def test_sample_shape():
    """Sampling produces correct output shape."""
    config = BlockARConfig(n_steps=10, bigru_hidden_dim=32, gru_hidden_dim=32)
    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 30, 5, 5)
    samples = model.sample(history, n_samples=3, max_residual=5)
    expected = (2, 3, 30, 5, 5)
    assert samples.shape == expected, f"Expected {expected}, got {samples.shape}"
    print(f"  PASS: sample shape {samples.shape}")


def test_sample_in_valid_range():
    """Samples should be denormalized to [0, 1] and clamped."""
    config = BlockARConfig(n_steps=10, bigru_hidden_dim=32, gru_hidden_dim=32)
    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 30, 5, 5)
    samples = model.sample(history, n_samples=2, max_residual=5)
    assert samples.min() >= 0.0, f"Samples below 0: min={samples.min():.4f}"
    assert samples.max() <= 1.0, f"Samples above 1: max={samples.max():.4f}"
    print(f"  PASS: samples in [0, 1] (min={samples.min():.4f}, max={samples.max():.4f})")


def test_sample_diversity():
    """Different samples from same history should differ."""
    config = BlockARConfig(n_steps=10, bigru_hidden_dim=32, gru_hidden_dim=32)
    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 30, 5, 5)
    samples = model.sample(history, n_samples=5, max_residual=5)
    diversity = samples.std(dim=1).mean().item()
    assert diversity > 0.01, f"Diversity too low: {diversity:.6f}"
    print(f"  PASS: sample diversity {diversity:.4f} (> 0.01)")


# Phase 4 tests — Conv3D denoiser

def test_conv3d_denoiser_shape():
    """Conv3DBlockDenoiser matches BiGRU interface: (B, T, 25) -> (B, T, 25)."""
    cfg = Conv3DDenoiserConfig()
    dec = Conv3DBlockDenoiser(cfg)
    noisy = torch.randn(4, 10, 25)
    cond = torch.randn(4, cfg.bottleneck_dim)
    pos = torch.arange(10).unsqueeze(0).expand(4, -1)
    k = torch.randint(0, 100, (4, 10))
    out = dec(noisy, cond, pos, k)
    assert out.shape == (4, 10, 25), f"Expected (4, 10, 25), got {out.shape}"
    print("  PASS: conv3d denoiser shape (4, 10, 25)")


def test_conv3d_gradient_flow():
    """All Conv3D denoiser params get gradients."""
    cfg = Conv3DDenoiserConfig()
    dec = Conv3DBlockDenoiser(cfg)
    noisy = torch.randn(4, 10, 25)
    cond = torch.randn(4, cfg.bottleneck_dim)
    pos = torch.arange(10).unsqueeze(0).expand(4, -1)
    k = torch.randint(0, 100, (4, 10))
    out = dec(noisy, cond, pos, k)
    loss = out.mean()
    loss.backward()
    no_grad = [name for name, p in dec.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Parameters without gradient: {no_grad}"
    print("  PASS: conv3d gradient flow through all parameters")


def test_conv3d_noise_differentiation():
    """Different noise levels should produce different predictions.

    At init, conv_out is zero-initialized (near-identity residual), so we
    perturb it to simulate a trained model, same pattern as
    test_spatial_stream_noise_sensitivity.
    """
    cfg = Conv3DDenoiserConfig()
    dec = Conv3DBlockDenoiser(cfg)
    # Perturb conv_out away from zero-init to simulate trained model
    with torch.no_grad():
        dec.conv_out.weight.normal_(std=0.1)
    dec.eval()
    noisy = torch.randn(2, 10, 25)
    cond = torch.randn(2, cfg.bottleneck_dim)
    pos = torch.arange(10).unsqueeze(0).expand(2, -1)
    k_low = torch.ones(2, 10, dtype=torch.long) * 5
    k_high = torch.ones(2, 10, dtype=torch.long) * 90
    out_low = dec(noisy, cond, pos, k_low)
    out_high = dec(noisy, cond, pos, k_high)
    diff = (out_low - out_high).abs().mean().item()
    assert diff > 1e-4, f"Outputs unchanged by noise level (diff={diff:.2e})"
    print(f"  PASS: conv3d noise differentiation (mean diff={diff:.4f})")


def test_conv3d_forward_loss():
    """ConditionalBlockARDDPM with denoiser_type='conv3d' produces scalar loss."""
    config = BlockARConfig(n_steps=20, denoiser_type="conv3d")
    model = ConditionalBlockARDDPM(config)
    model.train()
    history = torch.randn(2, 30, 5, 5)
    future = torch.randn(2, 30, 5, 5)
    result = model(history, future)
    assert "loss" in result, "Forward should return dict with 'loss'"
    assert result["loss"].shape == (), f"Loss should be scalar, got {result['loss'].shape}"
    assert result["loss"].item() > 0, "Loss should be positive"
    result["loss"].backward()
    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Parameters without gradient: {no_grad}"
    print(f"  PASS: conv3d forward loss ({result['loss'].item():.4f}), all params have grads")


def test_conv3d_sample_shape():
    """Sampling with Conv3D denoiser produces correct output shape."""
    config = BlockARConfig(n_steps=10, denoiser_type="conv3d", gru_hidden_dim=32)
    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 30, 5, 5)
    samples = model.sample(history, n_samples=2, max_residual=5)
    expected = (2, 2, 30, 5, 5)
    assert samples.shape == expected, f"Expected {expected}, got {samples.shape}"
    assert samples.min() >= 0.0 and samples.max() <= 1.0, "Samples should be in [0, 1]"
    print(f"  PASS: conv3d sample shape {samples.shape}, range [{samples.min():.3f}, {samples.max():.3f}]")


def test_conv3d_param_count():
    """Conv3D denoiser should be ~250K-300K params (comparable to BiGRU)."""
    cfg = Conv3DDenoiserConfig()
    dec = Conv3DBlockDenoiser(cfg)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"  INFO: Conv3D denoiser params: {n_params:,}")
    assert 200_000 < n_params < 350_000, f"Param count {n_params:,} outside expected range [200K, 350K]"
    print(f"  PASS: conv3d param count {n_params:,} in expected range [200K, 350K]")


# ============================================================
# Phase 5: Regime Conditioning Tests
# ============================================================

def _make_regime_config(**overrides):
    """Create a small BlockARConfig with regime conditioning enabled."""
    defaults = dict(
        history_len=10, future_len=10, block_size=5,
        surface_h=5, surface_w=5,
        gru_hidden_dim=32, bottleneck_dim=32,
        bigru_hidden_dim=64, pos_embed_dim=8, noise_embed_dim=8,
        n_steps=20, noise_rho=0.0,
        use_regime_conditioning=True, n_regimes=5, regime_embed_dim=16,
        regime_loss_weight=1.0,
    )
    defaults.update(overrides)
    return BlockARConfig(**defaults)


def test_regime_forward_loss():
    """forward() with regime_ids produces loss + regime_loss + regime_acc."""
    config = _make_regime_config()
    model = ConditionalBlockARDDPM(config)
    history = torch.randn(4, 10, 5, 5)
    future = torch.randn(4, 10, 5, 5)
    regime_ids = torch.randint(0, 5, (4,))

    result = model(history, future, regime_ids=regime_ids)
    assert "loss" in result, "Missing 'loss'"
    assert "regime_loss" in result, "Missing 'regime_loss'"
    assert "regime_acc" in result, "Missing 'regime_acc'"
    assert result["loss"].ndim == 0, f"Loss not scalar: {result['loss'].shape}"
    result["loss"].backward()
    print("  PASS: regime forward produces loss + regime_loss + regime_acc")


def test_regime_augment_condition():
    """_augment_condition returns (B, bottleneck_dim) unchanged shape."""
    config = _make_regime_config()
    model = ConditionalBlockARDDPM(config)
    condition = torch.randn(4, config.bottleneck_dim)
    regime_ids = torch.randint(0, 5, (4,))

    augmented = model._augment_condition(condition, regime_ids)
    assert augmented.shape == (4, config.bottleneck_dim), \
        f"Expected (4, {config.bottleneck_dim}), got {augmented.shape}"

    # Without regime_ids, should return condition unchanged
    same = model._augment_condition(condition, None)
    assert torch.equal(same, condition), "None regime_ids should return condition unchanged"
    print("  PASS: _augment_condition shape (B, bottleneck_dim), None passthrough")


def test_regime_sample_shape():
    """sample() with regime conditioning produces correct output shape."""
    config = _make_regime_config()
    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 10, 5, 5)

    samples = model.sample(history, n_samples=3)
    expected = (2, 3, 10, 5, 5)
    assert samples.shape == expected, f"Expected {expected}, got {samples.shape}"
    assert samples.min() >= 0.0, f"Samples below 0: {samples.min()}"
    assert samples.max() <= 1.0, f"Samples above 1: {samples.max()}"
    print(f"  PASS: regime sample shape {expected}, values in [0, 1]")


def test_regime_disabled_backward_compat():
    """use_regime_conditioning=False works identically to before."""
    config = BlockARConfig(
        history_len=10, future_len=10, block_size=5,
        surface_h=5, surface_w=5,
        gru_hidden_dim=32, bottleneck_dim=32,
        bigru_hidden_dim=64, pos_embed_dim=8, noise_embed_dim=8,
        n_steps=20, noise_rho=0.0,
        use_regime_conditioning=False,
    )
    model = ConditionalBlockARDDPM(config)
    assert model.regime_classifier is None, "regime_classifier should be None when disabled"
    assert model.regime_embed is None, "regime_embed should be None when disabled"
    assert model.regime_proj is None, "regime_proj should be None when disabled"

    # forward() without regime_ids should work
    history = torch.randn(4, 10, 5, 5)
    future = torch.randn(4, 10, 5, 5)
    result = model(history, future)
    assert "regime_loss" not in result, "regime_loss should not appear when disabled"
    result["loss"].backward()
    print("  PASS: regime disabled backward compat — no regime modules, no regime_loss")


def _make_uniform_config(**overrides):
    """Create a small BlockARConfig with uniform noise enabled."""
    defaults = dict(
        history_len=10, future_len=10, block_size=5,
        surface_h=5, surface_w=5,
        gru_hidden_dim=32, bottleneck_dim=32,
        bigru_hidden_dim=64, pos_embed_dim=8, noise_embed_dim=8,
        n_steps=20, noise_rho=0.0,
        use_uniform_noise=True, sampling_mode="uniform",
    )
    defaults.update(overrides)
    return BlockARConfig(**defaults)


def test_uniform_noise_forward():
    """forward() with use_uniform_noise=True produces valid loss."""
    config = _make_uniform_config()
    model = ConditionalBlockARDDPM(config)
    history = torch.randn(4, 10, 5, 5)
    future = torch.randn(4, 10, 5, 5)

    result = model(history, future)
    assert "loss" in result, "Missing 'loss'"
    assert result["loss"].ndim == 0, f"Loss not scalar: {result['loss'].shape}"
    assert not torch.isnan(result["loss"]), "Loss is NaN"
    result["loss"].backward()
    print("  PASS: uniform noise forward produces valid scalar loss")


def test_uniform_sampling_shape():
    """sample() with sampling_mode='uniform' produces correct output shape."""
    config = _make_uniform_config()
    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 10, 5, 5)

    samples = model.sample(history, n_samples=3)
    expected = (2, 3, 10, 5, 5)
    assert samples.shape == expected, f"Expected {expected}, got {samples.shape}"
    assert samples.min() >= 0.0, f"Samples below 0: {samples.min()}"
    assert samples.max() <= 1.0, f"Samples above 1: {samples.max()}"

    # Also test sample_batched
    samples_b = model.sample_batched(history, n_samples=3)
    assert samples_b.shape == expected, f"Batched: Expected {expected}, got {samples_b.shape}"
    print(f"  PASS: uniform sampling shape {expected}, values in [0, 1]")


def test_uniform_vs_pyramid_different():
    """Verify uniform and pyramid sampling produce different outputs."""
    torch.manual_seed(42)
    config_u = _make_uniform_config(sampling_mode="uniform")
    config_p = _make_uniform_config(sampling_mode="pyramid")

    model_u = ConditionalBlockARDDPM(config_u)
    model_p = ConditionalBlockARDDPM(config_p)
    # Share weights
    model_p.load_state_dict(model_u.state_dict())
    model_u.eval()
    model_p.eval()

    history = torch.randn(2, 10, 5, 5)
    torch.manual_seed(123)
    s_u = model_u.sample(history, n_samples=2)
    torch.manual_seed(123)
    s_p = model_p.sample(history, n_samples=2)

    # Different schedules should produce different outputs
    assert not torch.allclose(s_u, s_p, atol=1e-3), "Uniform and pyramid should differ"
    print("  PASS: uniform vs pyramid produce different outputs")


def test_uniform_backward_compat_defaults():
    """Default config uses task-adaptive noise and pyramid sampling."""
    config = BlockARConfig(
        history_len=10, future_len=10, block_size=5,
        surface_h=5, surface_w=5,
        gru_hidden_dim=32, bottleneck_dim=32,
        bigru_hidden_dim=64, pos_embed_dim=8, noise_embed_dim=8,
        n_steps=20, noise_rho=0.0,
    )
    assert config.use_uniform_noise is False, "Default should be task-adaptive"
    assert config.sampling_mode == "pyramid", "Default should be pyramid"

    model = ConditionalBlockARDDPM(config)
    model.eval()
    history = torch.randn(2, 10, 5, 5)
    future = torch.randn(2, 10, 5, 5)

    # forward and sample should work with defaults
    result = model(history, future)
    result["loss"].backward()
    samples = model.sample(history, n_samples=2)
    assert samples.shape == (2, 2, 10, 5, 5)
    print("  PASS: backward compat — defaults use task-adaptive + pyramid")


if __name__ == '__main__':
    print("=" * 60)
    print("Block-AR Diffusion Unit Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    tests = [
        # Phase 1
        test_encoder_shape, test_encoder_variable_length,
        test_encoder_masking, test_encoder_per_item_masking,
        test_encoder_cond_augmentation, test_denoiser_shape,
        test_gradient_flow, test_param_count,
        test_spatial_stream_shape, test_spatial_stream_noise_sensitivity,
        # Phase 2
        test_task_distribution, test_forward_mean_increasing,
        test_backward_mean_decreasing, test_interpolation_tent,
        test_noise_valid_range, test_jitter_nonzero,
        test_batch_task_adaptive,
        # Phase 3
        test_forward_produces_scalar_loss, test_forward_all_params_have_grad,
        test_synthetic_training_loss_decreases,
        test_sample_shape, test_sample_in_valid_range,
        test_sample_diversity,
        # Phase 4 — Conv3D
        test_conv3d_denoiser_shape, test_conv3d_gradient_flow,
        test_conv3d_noise_differentiation, test_conv3d_forward_loss,
        test_conv3d_sample_shape, test_conv3d_param_count,
        # Phase 5 — Regime Conditioning
        test_regime_forward_loss, test_regime_augment_condition,
        test_regime_sample_shape, test_regime_disabled_backward_compat,
        # Phase 6 — Uniform Noise
        test_uniform_noise_forward, test_uniform_sampling_shape,
        test_uniform_vs_pyramid_different, test_uniform_backward_compat_defaults,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("ALL TESTS PASSED")
