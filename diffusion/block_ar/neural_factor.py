"""
Neural Factor Model (250 series) — explicit low-rank non-AR joint generator.

Stack-agnostic by construction: model internals are strictly (B, K, T, D). Callers
do any factor-family-specific reshape (5x5 for IV, curve-length for rates, etc.)
outside the model.

Stage A (NeuralFactorModel):
    history -> encoder -> h
    h -> (mu_z, sigma_z)     (LatentEncoder)
    h -> Lambda(h)           (LoadingHead)       explicit low-rank factor loadings
    h -> D(h)                (IdiosyncraticScaleHead)
    z  = mu_z + sigma_z * eta_z,  eta_z ~ N(0, I_L)  (K draws)
    surface_change = einsum("btdl, bkl -> bktd", Lambda(h), z) + D(h) * eta_idio
    surface_level  = history[:, -1] + cumsum(surface_change, dim=2)
    return (B, K, T, D)

Stage B (LearnedMarginalHead):  conditional monotone spline head, knot positions
and output values are both learnable (no precomputed empirical quantiles).
Added later via extend_with_marginal_head().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import EncoderConfig, GRUEncoder


@dataclass
class NeuralFactorConfig:
    # Shape (stack-agnostic — D is whatever the caller passes)
    history_len: int = 30
    future_len: int = 30
    n_cells: int = 25
    latent_dim: int = 8

    # Encoder
    encoder_hidden: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1
    cond_aug_sigma: float = 0.0

    # Heads
    head_hidden: int = 256
    head_layers: int = 2
    head_dropout: float = 0.1

    # Idiosyncratic scale floor/ceiling (softplus-parameterised)
    idio_softplus_shift: float = -3.0
    idio_scale_clip: float = 0.20

    # Latent prior sigma regularisation (KL-free — just a soft bound)
    latent_logsigma_min: float = -4.0
    latent_logsigma_max: float = 1.0

    # Orthogonality regularisation on Lambda (optional, enabled via regulariser flag)
    ortho_reg_weight: float = 0.0

    # Support clamp for emitted surface levels (IV convention; callers can override)
    support_lo: float = 0.01
    support_hi: float = 1.0

    # Stage B knob: attach learned marginal head (default off; enabled in 250b)
    use_marginal_head: bool = False
    marginal_knots: int = 12

    # Stage C knob: replace Gaussian z with flow-matched posterior
    use_latent_fm: bool = False
    latent_fm_steps: int = 4
    latent_fm_hidden: int = 256
    latent_fm_time_embed: int = 32


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    mods: list[nn.Module] = []
    last = in_dim
    for _ in range(layers):
        mods += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = hidden
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class LatentEncoder(nn.Module):
    """h -> (mu_z, log_sigma_z) for the Gaussian latent factor posterior."""

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=2 * cfg.latent_dim,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(h)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(
            self.cfg.latent_logsigma_min, self.cfg.latent_logsigma_max
        )
        return mu, log_sigma


class LoadingHead(nn.Module):
    """h -> Lambda(h) in R^{B, T, D, L} — explicit factor loadings per (t, d)."""

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.future_len
        self.D = cfg.n_cells
        self.L = cfg.latent_dim
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=self.T * self.D * self.L,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        # Small-init: factor path starts near zero so idio path dominates early.
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        flat = self.net(h)
        return flat.view(B, self.T, self.D, self.L)


class IdiosyncraticScaleHead(nn.Module):
    """h -> D(h) in R^{B, T, D} — positive scales via shifted softplus."""

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.future_len
        self.D = cfg.n_cells
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=self.T * self.D,
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        raw = self.net(h).view(B, self.T, self.D)
        raw = raw + self.cfg.idio_softplus_shift
        scale = F.softplus(raw)
        # Soft cap to keep idio path from dominating early training.
        scale = scale.clamp(max=self.cfg.idio_scale_clip)
        return scale


class LatentFM(nn.Module):
    """Stage C — latent flow-matching posterior.

    Replaces the Gaussian reparameterisation `z = mu_z + sigma_z * eta` with an
    ODE-solved trajectory starting from η ~ N(0, I_L):

        z(t=0) = η ~ N(0, I_L)
        dz/dt  = v_θ(z_t, t, h)        (history-conditioned velocity)
        z(t=1) = posterior sample

    4-step Euler integration by default. Small MLP head; all shapes preserve L.
    Trained end-to-end on the same proper-scoring-rule loss as Stage A/B — if the
    optimal posterior is multimodal (tail-driven), LatentFM learns to deform the
    N(0, I) prior into that distribution.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.L = cfg.latent_dim
        self.n_steps = cfg.latent_fm_steps
        self.time_embed = nn.Linear(1, cfg.latent_fm_time_embed)
        in_dim = self.L + cfg.latent_fm_time_embed + cfg.bottleneck_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.latent_fm_hidden), nn.GELU(),
            nn.Linear(cfg.latent_fm_hidden, cfg.latent_fm_hidden), nn.GELU(),
            nn.Linear(cfg.latent_fm_hidden, self.L),
        )
        # Small-init velocity so early training is ≈ identity flow (η → η).
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.1)
                last.bias.zero_()

    def velocity(self, z_t: torch.Tensor, t_val: float, h: torch.Tensor) -> torch.Tensor:
        """z_t: (B, K, L). t_val: scalar in [0, 1]. h: (B, bottleneck). -> (B, K, L)."""
        B, K, _ = z_t.shape
        t_tensor = torch.full(
            (B, K, 1), float(t_val), device=z_t.device, dtype=z_t.dtype
        )
        t_emb = self.time_embed(t_tensor)             # (B, K, time_embed)
        h_exp = h.unsqueeze(1).expand(-1, K, -1)      # (B, K, bottleneck)
        inp = torch.cat([z_t, t_emb, h_exp], dim=-1)  # (B, K, L + time_embed + bottleneck)
        return self.net(inp)

    def sample(self, eta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """eta: (B, K, L) ~ N(0, I_L). h: (B, bottleneck). -> z: (B, K, L)."""
        dt = 1.0 / self.n_steps
        z = eta
        for i in range(self.n_steps):
            v = self.velocity(z, i * dt, h)
            z = z + dt * v
        return z


class LearnedMarginalHead(nn.Module):
    """Stage B — monotone spline head with fully learnable knot positions and values.

    Parameterisation (Choice A, conditional):
        input  knot offsets: delta_x(h, t)   via softplus -> positive
        output knot offsets: delta_y(h, t)   via softplus -> positive
        knot_x[k] = x_start + cumsum_k(delta_x)
        knot_y[k] = y_start + cumsum_k(delta_y)
    Monotone by construction; both axes learned end-to-end. No precomputed empirical
    quantiles anywhere — Bitter-Lesson compliant.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.marginal_knots
        self.T = cfg.future_len
        self.D = cfg.n_cells
        # Shared head that produces per-(t, d) knot offsets conditioned on h.
        # Output: 2 * (K - 1) delta values per (t, d) slot -> split into dx, dy.
        self.net = _mlp(
            in_dim=cfg.bottleneck_dim,
            out_dim=self.T * self.D * 2 * (self.K - 1),
            hidden=cfg.head_hidden,
            layers=cfg.head_layers,
            dropout=cfg.head_dropout,
        )
        # Anchor points — small learned bias; does NOT depend on training-set quantiles.
        self.x_start = nn.Parameter(torch.tensor(-4.0))
        self.y_start = nn.Parameter(torch.tensor(-4.0))
        # Identity init: start with x = y on the spline.
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.zero_()
                last.bias.zero_()

    def _build_knots(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        raw = self.net(h).view(B, self.T, self.D, 2, self.K - 1)
        # Small bias so softplus produces a reasonable initial spacing (~1.0).
        delta_x = F.softplus(raw[..., 0, :] + 0.5413)  # softplus(0.5413) ~= 1.0
        delta_y = F.softplus(raw[..., 1, :] + 0.5413)
        # Cumulative knot positions
        knot_x_inner = torch.cumsum(delta_x, dim=-1)
        knot_y_inner = torch.cumsum(delta_y, dim=-1)
        # Prepend the start anchor
        start_x = self.x_start.expand(B, self.T, self.D, 1)
        start_y = self.y_start.expand(B, self.T, self.D, 1)
        knot_x = torch.cat([start_x, start_x + knot_x_inner], dim=-1)
        knot_y = torch.cat([start_y, start_y + knot_y_inner], dim=-1)
        return knot_x, knot_y  # (B, T, D, K)

    def forward(self, y_raw: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        y_raw: (B, K_ens, T, D)
        h:     (B, bottleneck_dim)
        returns: (B, K_ens, T, D) — monotone-mapped
        """
        knot_x, knot_y = self._build_knots(h)  # (B, T, D, K)
        B, K_ens, T, D = y_raw.shape
        # Broadcast knots across ensemble: (B, 1, T, D, K) vs (B, K_ens, T, D, 1)
        kx = knot_x.unsqueeze(1)
        ky = knot_y.unsqueeze(1)
        y = y_raw.unsqueeze(-1)

        # Find bucket index: largest k where y >= kx[k]
        # cmp shape: (B, K_ens, T, D, K) — True where y >= knot_x
        cmp = y >= kx
        # Count of True along last dim -> bucket index in [0, K]
        idx = cmp.to(torch.long).sum(dim=-1)
        # Clamp so we always have a valid pair (idx-1, idx)
        idx_lo = (idx - 1).clamp(0, self.K - 2)
        idx_hi = idx_lo + 1

        # Gather knot positions/values at the bucket endpoints
        x_lo = torch.gather(kx.expand(B, K_ens, T, D, self.K), -1, idx_lo.unsqueeze(-1)).squeeze(-1)
        x_hi = torch.gather(kx.expand(B, K_ens, T, D, self.K), -1, idx_hi.unsqueeze(-1)).squeeze(-1)
        y_lo = torch.gather(ky.expand(B, K_ens, T, D, self.K), -1, idx_lo.unsqueeze(-1)).squeeze(-1)
        y_hi = torch.gather(ky.expand(B, K_ens, T, D, self.K), -1, idx_hi.unsqueeze(-1)).squeeze(-1)

        # Piecewise-linear interp with linear extrapolation (slope at endpoints).
        denom = (x_hi - x_lo).clamp(min=1e-6)
        slope = (y_hi - y_lo) / denom
        y_mapped = y_lo + slope * (y_raw - x_lo)
        return y_mapped


class NeuralFactorModel(nn.Module):
    """Stage A non-AR neural factor generator.

    Call convention:
        history: (B, T_hist, D)         -- caller pre-flattens any spatial layout
        returns (B, K, T_future, D)

    For IV use with history shape (B, T_hist, 5, 5), callers should reshape to
    (B, T_hist, D=25) BEFORE calling this model, and reshape returned (B, K, T, D)
    to (B, K, T, 5, 5) for IV-grid scoring. No 5x5 semantics live inside the model.
    """

    def __init__(self, cfg: NeuralFactorConfig):
        super().__init__()
        self.cfg = cfg
        encoder_cfg = EncoderConfig(
            input_dim=cfg.n_cells,
            extra_features=0,
            gru_hidden_dim=cfg.encoder_hidden,
            bottleneck_dim=cfg.bottleneck_dim,
            dropout=cfg.encoder_dropout,
            cond_aug_sigma=cfg.cond_aug_sigma,
        )
        self.encoder = GRUEncoder(encoder_cfg)
        self.latent_encoder = LatentEncoder(cfg)
        self.loading_head = LoadingHead(cfg)
        self.idio_head = IdiosyncraticScaleHead(cfg)
        if cfg.use_marginal_head:
            self.marginal_head: Optional[LearnedMarginalHead] = LearnedMarginalHead(cfg)
        else:
            self.marginal_head = None
        if cfg.use_latent_fm:
            self.latent_fm: Optional[LatentFM] = LatentFM(cfg)
        else:
            self.latent_fm = None

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, T_hist, D) -> h: (B, bottleneck_dim)."""
        # GRUEncoder's reshape (B, T, -1) is a no-op when last dim is already D.
        return self.encoder(history)

    def decode(
        self,
        history: torch.Tensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        history: (B, T_hist, D) in normalized [-1, 1] convention (same as 183c family).
        Returns:
          samples: (B, K, T_future, D) in IV [0, 1] space.
          aux:     dict with intermediate tensors for diagnostics / losses.
        """
        B = history.shape[0]
        D = history.shape[-1]
        K = n_samples
        assert D == self.cfg.n_cells, (
            f"history last dim {D} != cfg.n_cells {self.cfg.n_cells}"
        )

        h = self.encode_history(history)  # (B, bottleneck_dim)
        mu_z, log_sigma_z = self.latent_encoder(h)  # (B, L), (B, L)
        Lambda = self.loading_head(h)  # (B, T, D, L)
        D_scale = self.idio_head(h)    # (B, T, D)

        # Draw K latent samples: eta_z ~ N(0, I_L)
        sigma_z = torch.exp(log_sigma_z).unsqueeze(1)  # (B, 1, L)
        eta_z = torch.randn(
            B, K, self.cfg.latent_dim, device=history.device, dtype=history.dtype
        )
        if self.latent_fm is not None:
            # Stage C: ODE-solve from η ~ N(0, I) to posterior sample, history-conditioned.
            z = self.latent_fm.sample(eta_z, h)
        else:
            # Stage A/B: parametric Gaussian reparameterisation
            z = mu_z.unsqueeze(1) + sigma_z * eta_z  # (B, K, L)

        # Factor path: (B, T, D, L) x (B, K, L) -> (B, K, T, D)
        factor = torch.einsum("btdl, bkl -> bktd", Lambda, z)

        # Idiosyncratic residual
        eta_idio = torch.randn(
            B, K, self.cfg.future_len, D, device=history.device, dtype=history.dtype
        )
        idio = D_scale.unsqueeze(1) * eta_idio  # (B, K, T, D)

        surface_change = factor + idio  # (B, K, T, D)
        # Denormalize history's last frame to [0, 1] IV space for cumsum base.
        last_iv = 0.5 * (history[:, -1:, :] + 1.0)
        last_iv = last_iv.clamp(self.cfg.support_lo, self.cfg.support_hi)
        last = last_iv.unsqueeze(1)  # (B, 1, 1, D)
        surface_level = last + torch.cumsum(surface_change, dim=2)
        pre_head = surface_level.clamp(self.cfg.support_lo, self.cfg.support_hi)

        if self.marginal_head is not None:
            surface_level = self.marginal_head(surface_level, h)

        surface_level = surface_level.clamp(self.cfg.support_lo, self.cfg.support_hi)

        aux = {
            "h": h,
            "mu_z": mu_z,
            "log_sigma_z": log_sigma_z,
            "Lambda": Lambda,
            "D_scale": D_scale,
            "z": z,
            "factor": factor,
            "idio": idio,
            "pre_head": pre_head,  # clamped pre-marginal-head samples (B, K, T, D)
        }
        return surface_level, aux

    def forward(
        self,
        history: torch.Tensor,
        n_samples: int = 8,
        **_ignored_kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.decode(history, n_samples)

    # ------------------------------------------------------------------
    # Inference convenience — signature-compatible with sample_batched callers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        **_ignored_kwargs,
    ) -> torch.Tensor:
        """Inference sampler.

        history must be in normalized [-1, 1] convention (same as 183c family).
        Accepts either (B, T_hist, 5, 5) or (B, T_hist, D) shape.
        Returns (B, K, T_future, 5, 5) if input was 4-D, else (B, K, T_future, D).

        n_steps / chunk_size / max_residual / max_global_residual / extra_hist are
        accepted for API compatibility with the 11-suite evaluator and the
        conditionality harness but are unused (non-AR single-pass generator).
        """
        was_5d = (history.dim() == 4)
        if was_5d:
            B, T_hist, H, W = history.shape
            history_flat = history.reshape(B, T_hist, H * W)
        else:
            history_flat = history
            B, T_hist, D = history.shape

        samples, _ = self.decode(history_flat, n_samples=n_samples)  # (B, K, T, D)

        if was_5d:
            B, K, T, D = samples.shape
            samples = samples.reshape(B, K, T, H, W)
        return samples

    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 48,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Alias — matches the 220h evaluator's native-path interface.

        Absorbs max_residual / max_global_residual / extra_hist from the
        conditionality test, which assumes an AR kernel signature.
        """
        return self.sample(
            history, n_samples=n_samples, n_steps=n_steps, chunk_size=chunk_size,
        )

    # ------------------------------------------------------------------
    # Regularisers (optional)
    # ------------------------------------------------------------------
    def orthogonality_penalty(self, Lambda: torch.Tensor) -> torch.Tensor:
        """|| Lambda^T Lambda - diag ||_F^2 averaged over (B, T), for use if
        full-rank leak is detected. Shape: (B, T, D, L) -> scalar.
        """
        B, T, D, L = Lambda.shape
        Lm = Lambda.reshape(B * T, D, L)
        gram = torch.einsum("bdl, bdm -> blm", Lm, Lm)  # (B*T, L, L)
        eye = torch.eye(L, device=Lambda.device, dtype=Lambda.dtype).expand(B * T, L, L)
        diag_target = gram * eye  # keep diag, zero off-diag
        off = gram - diag_target
        return off.pow(2).mean()


# ----------------------------------------------------------------------
# Checkpoint helpers — parallels the load_one_day_kernel contract
# ----------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[NeuralFactorModel, dict]:
    """Load a 250-series checkpoint. Returns (model, payload)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = payload["config"]
    cfg = NeuralFactorConfig(**cfg_dict)
    model = NeuralFactorModel(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def config_to_dict(cfg: NeuralFactorConfig) -> dict:
    return asdict(cfg)
