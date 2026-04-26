from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticIVPriorConfig:
    n_windows: int = 4096
    history_len: int = 30
    future_len: int = 30
    grid_h: int = 5
    grid_w: int = 5
    seed: int = 560
    min_iv: float = 0.03
    max_iv: float = 0.95
    stress_prob: float = 0.30
    jump_prob: float = 0.06
    idio_noise: float = 0.006


@dataclass(frozen=True)
class SyntheticIVPriorBatch:
    history: torch.Tensor
    future: torch.Tensor
    full_sequence: torch.Tensor
    regime_labels: torch.Tensor


def _make_surface_loadings(
    grid_h: int,
    grid_w: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tenor = torch.linspace(-1.0, 1.0, grid_h, dtype=dtype).view(grid_h, 1)
    moneyness = torch.linspace(-1.0, 1.0, grid_w, dtype=dtype).view(1, grid_w)
    level = torch.ones(grid_h, grid_w, dtype=dtype)
    term = tenor.expand(grid_h, grid_w)
    skew = moneyness.expand(grid_h, grid_w)
    curvature = (moneyness.square() - moneyness.square().mean()).expand(grid_h, grid_w)
    return level, term, skew, curvature


def _smooth_surface_noise(
    noise: torch.Tensor,
) -> torch.Tensor:
    """Cheap local averaging that avoids cell-wise white-noise surfaces."""
    padded = torch.nn.functional.pad(noise[:, None], (1, 1, 1, 1), mode="reflect")
    smoothed = torch.nn.functional.avg_pool2d(padded, kernel_size=3, stride=1)
    return smoothed[:, 0]


def generate_synthetic_iv_windows(
    cfg: SyntheticIVPriorConfig,
) -> SyntheticIVPriorBatch:
    if cfg.n_windows <= 0:
        raise ValueError("n_windows must be positive")
    if cfg.history_len <= 0 or cfg.future_len <= 0:
        raise ValueError("history_len and future_len must be positive")
    if cfg.grid_h != 5 or cfg.grid_w != 5:
        raise ValueError("The current IV-surface backbone expects a 5x5 grid")
    if not 0.0 <= cfg.stress_prob <= 1.0:
        raise ValueError("stress_prob must be in [0,1]")

    dtype = torch.float32
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed))
    total_len = cfg.history_len + cfg.future_len
    n = cfg.n_windows

    future_stress = torch.bernoulli(
        torch.full((n,), float(cfg.stress_prob), dtype=dtype),
        generator=generator,
    ).long()
    if n >= 2 and int(future_stress.sum()) in {0, n}:
        future_stress[0] = 0
        future_stress[1] = 1

    level_load, term_load, skew_load, curve_load = _make_surface_loadings(
        cfg.grid_h,
        cfg.grid_w,
        dtype=dtype,
    )

    calm_level = 0.17 + 0.035 * torch.randn(n, generator=generator, dtype=dtype)
    stress_level = 0.34 + 0.060 * torch.randn(n, generator=generator, dtype=dtype)
    target_level = torch.where(future_stress.bool(), stress_level, calm_level)
    initial_level = 0.18 + 0.045 * torch.randn(n, generator=generator, dtype=dtype)

    level = torch.empty(n, total_len, dtype=dtype)
    slope = torch.empty(n, total_len, dtype=dtype)
    skew = torch.empty(n, total_len, dtype=dtype)
    curve = torch.empty(n, total_len, dtype=dtype)

    level[:, 0] = initial_level
    slope[:, 0] = 0.015 * torch.randn(n, generator=generator, dtype=dtype)
    skew[:, 0] = -0.035 + 0.025 * torch.randn(n, generator=generator, dtype=dtype)
    curve[:, 0] = 0.025 + 0.012 * torch.randn(n, generator=generator, dtype=dtype)

    transition_start = max(1, cfg.history_len // 2)
    for t in range(1, total_len):
        if t < transition_start:
            target_mix = initial_level
        else:
            frac = min(1.0, (t - transition_start + 1) / max(1, total_len - transition_start))
            target_mix = (1.0 - frac) * initial_level + frac * target_level

        is_stress_phase = future_stress.bool() & (t >= cfg.history_len // 2)
        level_vol = torch.where(is_stress_phase, 0.030, 0.012)
        factor_shock = torch.randn(n, generator=generator, dtype=dtype)
        jump = (
            torch.bernoulli(
                torch.full((n,), float(cfg.jump_prob), dtype=dtype),
                generator=generator,
            )
            * torch.relu(torch.randn(n, generator=generator, dtype=dtype) * 0.030 + 0.035)
            * future_stress.float()
        )
        level[:, t] = (
            0.86 * level[:, t - 1]
            + 0.14 * target_mix
            + level_vol * factor_shock
            + jump
        )
        slope[:, t] = (
            0.90 * slope[:, t - 1]
            + 0.006 * torch.randn(n, generator=generator, dtype=dtype)
            + 0.010 * future_stress.float()
        )
        skew[:, t] = (
            0.88 * skew[:, t - 1]
            + 0.010 * torch.randn(n, generator=generator, dtype=dtype)
            - 0.018 * future_stress.float()
            - 0.020 * jump
        )
        curve[:, t] = (
            0.90 * curve[:, t - 1]
            + 0.004 * torch.randn(n, generator=generator, dtype=dtype)
            + 0.010 * future_stress.float()
        )

    surfaces = (
        level[:, :, None, None] * level_load
        + slope[:, :, None, None] * term_load
        + skew[:, :, None, None] * skew_load
        + curve[:, :, None, None] * curve_load
    )
    idio = cfg.idio_noise * torch.randn(
        n * total_len,
        cfg.grid_h,
        cfg.grid_w,
        generator=generator,
        dtype=dtype,
    )
    idio = _smooth_surface_noise(idio).view(n, total_len, cfg.grid_h, cfg.grid_w)
    surfaces = torch.clamp(surfaces + idio, float(cfg.min_iv), float(cfg.max_iv))

    history = surfaces[:, : cfg.history_len].contiguous()
    future = surfaces[:, cfg.history_len :].contiguous()
    return SyntheticIVPriorBatch(
        history=history,
        future=future,
        full_sequence=surfaces.contiguous(),
        regime_labels=future_stress,
    )
