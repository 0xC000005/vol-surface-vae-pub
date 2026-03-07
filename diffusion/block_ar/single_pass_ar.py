"""
Single-Pass Block-AR model with afCRPS training.

Replaces the DDPM 100-step diffusion loop with a single forward pass.
Keeps: GRU encoder, Conv3D ResBlocks, AdaGN, exp(z × vol_scale) transform.
Removes: noise schedule, timestep embedding, reverse process.
Changes: loss from MSE-on-epsilon to afCRPS-on-IV.

Noise injection via learned MLP that replaces the timestep embedding in AdaGN.
Each ensemble member draws a different noise vector z ~ N(0, I_d), producing
diverse z-space outputs that denormalize through exp() to IV space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from diffusion.block_ar.conv3d_denoiser import (
    Conv3DDenoiserConfig,
    ResBlock3D,
)
from diffusion.time_embedding import (
    SinusoidalTimeEmbedding,
    AdaptiveGroupNorm,
)


def denormalize_iv(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] normalized space to [0, 1] IV space."""
    return (x + 1.0) / 2.0


def normalize_iv(x: torch.Tensor) -> torch.Tensor:
    """Convert from [0, 1] IV space to [-1, 1] normalized space."""
    return x * 2.0 - 1.0


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SinglePassConfig:
    """Configuration for SinglePassBlockAR."""

    # Data
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5
    block_size: int = 10

    # Encoder (reused from DDPM)
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 128
    encoder_dropout: float = 0.1

    # Decoder (Conv3D backbone)
    conv3d_base_channels: int = 32
    conv3d_n_res_blocks: int = 6
    conv3d_groups: int = 8
    pos_embed_dim: int = 16

    # Noise injection
    noise_dim: int = 16          # input noise vector dimension
    noise_embed_dim: int = 64    # must match conv3d_noise_embed_dim for weight transfer
    shared_noise_input: bool = False  # inject first element of z as shared spatial input
    cond_noise_mlp: bool = False  # feed condition into noise MLP for regime-dependent diversity
    noise_dist: str = "gaussian"  # "gaussian" or "student_t"
    student_t_df: float = 4.0     # degrees of freedom for Student-t noise

    # Vol-scaled denormalization
    global_mean_vol: float = 0.0187
    vol_scale_min: float = 0.5
    vol_scale_max: float = 2.0
    vol_scale_power: float = 1.0

    # Direct IV mode: decoder outputs normalized IV directly (no exp/baseline)
    direct_iv: bool = False
    no_tanh: bool = False  # remove tanh bounding (let loss learn output range)
    learned_vol_scale: bool = False  # per-cell vol_scale from condition MLP
    twcrps_beta: float = 0.0  # threshold-weighted CRPS beta (0 = standard CRPS)

    # AR frame decoder: per-frame autoregressive generation (replaces Conv3D blocks)
    ar_frame: bool = False
    ar_frame_rho: float = 0.8        # temporal noise correlation
    ar_frame_hidden: int = 128       # MLP hidden dim
    ar_frame_cell_spread: bool = False  # learned per-cell spread scaling (condition-dependent)
    ar_frame_static_cell_scale: bool = False  # static per-cell scale (nn.Parameter)
    ar_frame_bias_lambda: float = 0.0  # delta zero-mean loss weight
    ar_frame_percell_vol_scale: bool = False  # per-cell vol_scale from history std
    ar_dual_pos: bool = False
    ar_local_pos_period: int = 30
    ar_horizon_bucket_size: int = 30
    ar_horizon_max_buckets: int = 9
    ar_horizon_embed_dim: int = 8
    ar_dynamic_vs: bool = False
    ar_dynamic_vs_mode: str = "scalar"  # "scalar" or "tenor"
    ar_dynamic_vs_scale: float = 0.15
    ar_frame_log_space: bool = False  # multiplicative dynamics: iv = prev * exp(vs * delta)
    ar_factor_noise: bool = False     # factor model noise: z_factors @ loadings.T → per-cell
    ar_n_factors: int = 5             # number of latent noise factors
    ar_factor_noise_norm: bool = False  # normalize cell_noise to unit variance per cell
    ar_factor_init_scale: float = 0.1  # factor loadings init std
    ar_frame_floor_clamp: float = 0.001  # lower clamp for IV values in AR frame generation
    ar_cell_embed: bool = False          # per-cell processing with learned cell embedding
    ar_cell_embed_dim: int = 8           # cell embedding dimension
    ar_cell_cond_offset: bool = False    # per-cell condition offsets (Exp 93d)
    ar_freeze_gru_state: bool = False    # freeze GRU state during generation (Exp 93e)
    ar_input_noise_std: float = 0.0      # noise std on GRU input during training (Exp 93f)
    ar_percell_bias: bool = False         # per-cell conditional bias loss (Exp 94a)

    # Output
    output_dir: str = "models/backfill/afcrps"
    device: str = "cuda"


# ──────────────────────────────────────────────────────────────────────
# Noise MLP (replaces TimeEmbedding)
# ──────────────────────────────────────────────────────────────────────

class NoiseMLP(nn.Module):
    """Projects a noise vector z ~ N(0, I) to the noise embedding dimension.

    Replaces TimeEmbedding in the denoiser. Output has same shape and
    dimension as the timestep embedding, so downstream AdaGN is unchanged.

    When cond_dim > 0, takes concat(z, condition_proj) as input so the
    noise embedding is regime-dependent. A small projection of condition
    (128→16 dim) prevents the condition from drowning out the noise signal.
    """

    def __init__(self, noise_dim: int, embed_dim: int, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = cond_dim

        # Optional condition projection: keep noise and condition at equal scale
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, noise_dim)  # 128 → 16
            input_dim = noise_dim + noise_dim  # cat(z_16, cond_proj_16) = 32
        else:
            self.cond_proj = None
            input_dim = noise_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        # Small random init on output layer (NOT zero) to avoid dead start.
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z: (B, noise_dim)
            condition: (B, cond_dim) — only used if cond_dim > 0
        Returns:
            emb: (B, embed_dim)
        """
        if self.cond_proj is not None and condition is not None:
            cond_small = self.cond_proj(condition)  # (B, noise_dim)
            x = torch.cat([z, cond_small], dim=-1)  # (B, 2*noise_dim)
        else:
            x = z
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────────────
# Frame Decoder (per-frame MLP for AR generation)
# ──────────────────────────────────────────────────────────────────────

class FrameDecoder(nn.Module):
    """Per-frame MLP: predicts delta from prev_frame + condition + noise + position.

    Two modes:
      - Shared (default): prev_frame(25) → MLP → delta(25). All cells processed together.
      - Per-cell (cell_embed=True): prev_cell(1) + cell_emb(8) → MLP → delta(1).
        Each cell processed independently with a learned cell identity embedding.
        Batched as (B*25, input_dim) for efficiency. Enables per-cell differentiation.
    """

    def __init__(self, frame_dim: int, cond_dim: int, noise_dim: int,
                 pos_dim: int, hidden_dim: int,
                 horizon_embed_dim: int = 0, n_horizon_buckets: int = 0,
                 cell_embed: bool = False, cell_embed_dim: int = 8,
                 cell_cond_offset: bool = False,
                 n_cells: int = 25):
        super().__init__()
        self.cell_embed_active = cell_embed
        self.cell_cond_offset_active = cell_cond_offset
        self.n_cells = n_cells
        self.pos_embed = SinusoidalTimeEmbedding(dim=pos_dim)
        self.horizon_embed = None
        if horizon_embed_dim > 0 and n_horizon_buckets > 0:
            self.horizon_embed = nn.Embedding(n_horizon_buckets, horizon_embed_dim)
            nn.init.normal_(self.horizon_embed.weight, std=0.02)

        if cell_cond_offset:
            self.cond_offsets = nn.Parameter(torch.randn(n_cells, cond_dim) * 0.01)
            self.cell_emb = None
            input_dim = 1 + cond_dim + noise_dim + pos_dim + horizon_embed_dim
            out_dim = 1
        elif cell_embed:
            self.cell_emb = nn.Embedding(n_cells, cell_embed_dim)
            nn.init.normal_(self.cell_emb.weight, std=0.02)
            self.cond_offsets = None
            input_dim = 1 + cond_dim + noise_dim + pos_dim + cell_embed_dim + horizon_embed_dim
            out_dim = 1
        else:
            self.cell_emb = None
            self.cond_offsets = None
            input_dim = frame_dim + cond_dim + noise_dim + pos_dim + horizon_embed_dim
            out_dim = frame_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Zero-init last layer → delta=0 at init → prev_frame unchanged
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, prev_frame: torch.Tensor, condition: torch.Tensor,
                noise_t: torch.Tensor, local_position: torch.Tensor,
                horizon_bucket: torch.Tensor | None = None) -> torch.Tensor:
        """
        prev_frame: (B, frame_dim) flattened 5×5 IV [0,1]
        condition:  (B, cond_dim) GRU-encoded context
        noise_t:    (B, noise_dim) AR noise for this frame
        local_position: (B,) local frame index, usually t or t % period
        horizon_bucket: (B,) optional coarse absolute-horizon bucket
        Returns:    (B, frame_dim) delta, tanh-bounded [-1,1]
        """
        if self.cell_cond_offset_active:
            return self._forward_cond_offset(prev_frame, condition, noise_t,
                                             local_position, horizon_bucket)
        if self.cell_embed_active:
            return self._forward_per_cell(prev_frame, condition, noise_t,
                                          local_position, horizon_bucket)
        pos_emb = self.pos_embed(local_position)  # (B, pos_dim)
        pieces = [prev_frame, condition, noise_t, pos_emb]
        if self.horizon_embed is not None:
            if horizon_bucket is None:
                horizon_bucket = torch.zeros_like(local_position)
            pieces.append(self.horizon_embed(horizon_bucket))
        x = torch.cat(pieces, dim=-1)
        return torch.tanh(self.mlp(x))

    def _forward_per_cell(self, prev_frame: torch.Tensor, condition: torch.Tensor,
                          noise_t: torch.Tensor, local_position: torch.Tensor,
                          horizon_bucket: torch.Tensor | None = None) -> torch.Tensor:
        """Per-cell forward: (B, 25) → expand to (B*25, input_dim) → MLP → (B, 25)."""
        B = prev_frame.shape[0]
        C = self.n_cells  # 25

        pos_emb = self.pos_embed(local_position)  # (B, pos_dim)
        # Expand shared inputs: (B, dim) → (B*C, dim)
        cond_exp = condition.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
        noise_exp = noise_t.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
        pos_exp = pos_emb.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)

        # Per-cell inputs: raw IV value + learned cell identity
        prev_cell = prev_frame.reshape(B, C, 1).reshape(B * C, 1)  # (B*C, 1)
        cell_ids = torch.arange(C, device=prev_frame.device).unsqueeze(0).expand(B, C).reshape(B * C)
        cell_emb = self.cell_emb(cell_ids)  # (B*C, cell_embed_dim)

        pieces = [prev_cell, cond_exp, noise_exp, pos_exp, cell_emb]
        if self.horizon_embed is not None:
            if horizon_bucket is None:
                horizon_bucket = torch.zeros_like(local_position)
            hb_exp = horizon_bucket.unsqueeze(1).expand(B, C).reshape(B * C)
            pieces.append(self.horizon_embed(hb_exp))

        x = torch.cat(pieces, dim=-1)       # (B*C, input_dim)
        delta = torch.tanh(self.mlp(x))      # (B*C, 1)
        return delta.reshape(B, C)            # (B, 25)

    def _forward_cond_offset(self, prev_frame: torch.Tensor, condition: torch.Tensor,
                             noise_t: torch.Tensor, local_position: torch.Tensor,
                             horizon_bucket: torch.Tensor | None = None) -> torch.Tensor:
        """Per-cell forward with condition offsets: each cell sees condition + offset_i."""
        B = prev_frame.shape[0]
        C = self.n_cells  # 25

        pos_emb = self.pos_embed(local_position)  # (B, pos_dim)
        # Per-cell condition: (B, 128) + (25, 128) → (B, 25, 128) → (B*25, 128)
        cond_shifted = condition.unsqueeze(1) + self.cond_offsets.unsqueeze(0)
        cond_exp = cond_shifted.reshape(B * C, -1)
        noise_exp = noise_t.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
        pos_exp = pos_emb.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
        prev_cell = prev_frame.reshape(B * C, 1)

        pieces = [prev_cell, cond_exp, noise_exp, pos_exp]
        if self.horizon_embed is not None:
            if horizon_bucket is None:
                horizon_bucket = torch.zeros_like(local_position)
            hb_exp = horizon_bucket.unsqueeze(1).expand(B, C).reshape(B * C)
            pieces.append(self.horizon_embed(hb_exp))

        x = torch.cat(pieces, dim=-1)       # (B*C, input_dim)
        delta = torch.tanh(self.mlp(x))      # (B*C, 1)
        return delta.reshape(B, C)            # (B, 25)


# ──────────────────────────────────────────────────────────────────────
# Single-Pass Decoder (modified Conv3D backbone)
# ──────────────────────────────────────────────────────────────────────

class SinglePassDecoder(nn.Module):
    """Conv3D decoder for single-pass generation.

    Architecture is identical to Conv3DBlockDenoiser but:
    - Takes zeros as spatial input (not noisy x_t)
    - Receives noise embedding instead of timestep embedding
    - Output goes through tanh for bounded z-space output
    - Final conv initialized to near-zero (starts at baseline prediction)
    """

    def __init__(self, config: SinglePassConfig):
        super().__init__()
        self.config = config
        C = config.conv3d_base_channels

        # Position embedding (same as DDPM — frame indices 0-9)
        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)

        # LayerNorm each component before concatenation — prevents scale mismatch
        # where noise (mean|x|=0.72) swamps condition (mean|x|=0.05) in cond_proj.
        # After LN, each component has zero mean / unit variance per element.
        self.noise_ln = nn.LayerNorm(config.noise_embed_dim)
        self.cond_ln = nn.LayerNorm(config.bottleneck_dim)
        self.pos_ln = nn.LayerNorm(config.pos_embed_dim)

        # Conditioning projection: cat(noise_emb, pos_emb, condition) -> C
        cond_input_dim = config.noise_embed_dim + config.pos_embed_dim + config.bottleneck_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, C * 4),
            nn.SiLU(),
            nn.Linear(C * 4, C),
        )

        # Conv3D backbone (same as Conv3DBlockDenoiser)
        self.conv_in = nn.Conv3d(1, C, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList()
        self.ada_norms = nn.ModuleList()
        for _ in range(config.conv3d_n_res_blocks):
            self.res_blocks.append(ResBlock3D(C, groups=config.conv3d_groups))
            self.ada_norms.append(AdaptiveGroupNorm(C, config.conv3d_groups, embed_dim=C))

        self.final_norm = nn.GroupNorm(config.conv3d_groups, C)
        self.final_act = nn.SiLU()
        self.conv_out = nn.Conv3d(C, 1, kernel_size=3, padding=1)

        # Small random init on output conv: z_out ≈ 0 in expectation (baseline
        # prediction) but nonzero per-sample → members produce slightly different
        # z_out → spread > 0 → CRPS diversity gradient has signal from step 1.
        # Strict zero-init would kill ALL gradient flow through conv_out (dead start).
        nn.init.normal_(self.conv_out.weight, std=0.01)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        condition: torch.Tensor,        # (B, bottleneck_dim)
        noise_emb: torch.Tensor,         # (B, noise_embed_dim)
        positions: torch.Tensor,         # (B, T)
        shared_noise_scalar: torch.Tensor = None,  # (B,) shared noise for cross-cell correlation
    ) -> torch.Tensor:
        """
        Args:
            condition: Encoder output (B, bottleneck_dim)
            noise_emb: Noise MLP output (B, noise_embed_dim)
            positions: Frame indices (B, T) — e.g. [0, 1, ..., 9]
            shared_noise_scalar: (B,) noise injected as spatial input (same for all cells)

        Returns:
            z_out: (B, T, H*W) in [-1, 1] via tanh
        """
        B = condition.shape[0]
        T = positions.shape[1]
        H, W = self.config.surface_h, self.config.surface_w

        # Position embedding (frame indices)
        pos_emb = self.pos_embed(positions)  # (B, T, pos_embed_dim)

        # Expand noise embedding to per-frame
        noise_emb_expanded = noise_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, noise_embed_dim)

        # Expand condition to per-frame
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)

        # LayerNorm each component to equalize scales before cond_proj
        noise_normed = self.noise_ln(noise_emb_expanded)
        pos_normed = self.pos_ln(pos_emb)
        cond_normed = self.cond_ln(cond_expanded)

        # Combined conditioning
        cond_cat = torch.cat([noise_normed, pos_normed, cond_normed], dim=-1)
        cond = self.cond_proj(cond_cat.reshape(B * T, -1)).reshape(B, T, -1)  # (B, T, C)

        # Input: shared noise for cross-cell correlation, or zeros
        if shared_noise_scalar is not None:
            # Inject shared noise as spatial input — same value for ALL cells
            # Forces cross-cell correlation (all cells start from same perturbation)
            x = shared_noise_scalar.view(B, 1, 1, 1, 1).expand(B, 1, T, H, W)
        else:
            x = torch.zeros(B, 1, T, H, W, device=condition.device)

        # Conv3D backbone
        x = self.conv_in(x)  # (B, C, T, H, W)

        for res_block, ada_norm in zip(self.res_blocks, self.ada_norms):
            x = res_block(x)
            x = ada_norm(x, cond)

        x = self.final_act(self.final_norm(x))
        x = self.conv_out(x)  # (B, 1, T, H, W)

        # Squeeze channel dim and reshape
        x = x.squeeze(1)  # (B, T, H, W)

        if not self.config.no_tanh:
            x = torch.tanh(x)

        return x.reshape(B, T, H * W)


# ──────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────

class SinglePassBlockAR(nn.Module):
    """Single-pass Block-AR model trained with afCRPS.

    Generates vol surface futures in one forward pass per block.
    Diversity comes from noise injection via NoiseMLP → AdaGN,
    not from denoiser imperfection across 100 reverse steps.
    """

    def __init__(self, config: SinglePassConfig):
        super().__init__()
        self.config = config

        # Encoder (same as DDPM)
        enc_config = EncoderConfig(
            input_dim=config.surface_h * config.surface_w,
            gru_hidden_dim=config.gru_hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
            dropout=config.encoder_dropout,
        )
        self.encoder = GRUEncoder(enc_config)

        if config.ar_frame:
            # AR frame decoder: per-frame MLP (no Conv3D, no NoiseMLP)
            frame_dim = config.surface_h * config.surface_w
            # Factor noise: noise input is per-cell (frame_dim) instead of shared (noise_dim)
            fd_noise_dim = frame_dim if config.ar_factor_noise else config.noise_dim
            self.frame_decoder = FrameDecoder(
                frame_dim=frame_dim,
                cond_dim=config.bottleneck_dim,
                noise_dim=fd_noise_dim,
                pos_dim=config.pos_embed_dim,
                hidden_dim=config.ar_frame_hidden,
                horizon_embed_dim=(
                    config.ar_horizon_embed_dim if config.ar_dual_pos else 0
                ),
                n_horizon_buckets=(
                    config.ar_horizon_max_buckets if config.ar_dual_pos else 0
                ),
                cell_embed=config.ar_cell_embed,
                cell_embed_dim=config.ar_cell_embed_dim,
                cell_cond_offset=config.ar_cell_cond_offset,
                n_cells=frame_dim,
            )
            # Factor noise loadings: (frame_dim, n_factors) — learned spatial correlation
            if config.ar_factor_noise:
                self.factor_loadings = nn.Parameter(
                    torch.randn(frame_dim, config.ar_n_factors) * config.ar_factor_init_scale
                )
            # Per-cell spread: condition → 25 positive scalars ≈ 1.0
            if config.ar_frame_cell_spread:
                self.cell_spread_linear = nn.Linear(config.bottleneck_dim, frame_dim)
                nn.init.zeros_(self.cell_spread_linear.weight)
                nn.init.constant_(self.cell_spread_linear.bias, 0.541)  # softplus(0.541) ≈ 1.0
            # Static per-cell scale: nn.Parameter(ones(25)), clamped [0.3, 3.0]
            if config.ar_frame_static_cell_scale:
                self.cell_scale = nn.Parameter(torch.ones(frame_dim))
            if config.ar_dynamic_vs:
                if config.ar_dynamic_vs_mode not in {"scalar", "tenor"}:
                    raise ValueError(
                        f"Unsupported ar_dynamic_vs_mode={config.ar_dynamic_vs_mode!r}"
                    )
                out_dim = 1 if config.ar_dynamic_vs_mode == "scalar" else config.surface_h
                self.dynamic_vs_head = nn.Linear(config.bottleneck_dim, out_dim)
                nn.init.zeros_(self.dynamic_vs_head.weight)
                nn.init.zeros_(self.dynamic_vs_head.bias)
        else:
            # Noise MLP (replaces TimeEmbedding)
            cond_dim = config.bottleneck_dim if config.cond_noise_mlp else 0
            self.noise_mlp = NoiseMLP(config.noise_dim, config.noise_embed_dim, cond_dim=cond_dim)

            # Decoder (modified Conv3D)
            self.decoder = SinglePassDecoder(config)

        # Per-cell spread scaling (post-exp, kurtosis-invariant)
        # Linear: condition (128) → 25, Softplus → positive spread ≈ 1.0
        if config.learned_vol_scale:
            cond_dim = config.bottleneck_dim
            self.cell_spread_mlp = nn.Sequential(
                nn.Linear(cond_dim, config.surface_h * config.surface_w),
                nn.Softplus(),
            )
            # Init so output starts at 1.0: softplus(0.541) ≈ 1.0
            nn.init.zeros_(self.cell_spread_mlp[0].weight)
            nn.init.constant_(self.cell_spread_mlp[0].bias, 0.541)

        # twCRPS per-cell statistics (populated from training data before training)
        self.register_buffer('cell_median', torch.zeros(config.surface_h, config.surface_w))
        self.register_buffer('cell_iqr', torch.ones(config.surface_h, config.surface_w))

        # Kurtosis matching target (populated from training data before training)
        self.register_buffer('target_kurt', torch.full((config.surface_h, config.surface_w), 3.0))

    def _sample_noise(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample noise vector z ~ N(0,I) or StudentT(df)."""
        # Factor noise uses n_factors dim; shared noise uses noise_dim
        ndim = self.config.ar_n_factors if self.config.ar_factor_noise else self.config.noise_dim
        if self.config.noise_dist == "student_t":
            dist = torch.distributions.StudentT(df=self.config.student_t_df)
            z = dist.rsample((B, ndim)).to(device).clamp(-5, 5)
            z = z / 1.414  # scale so pretrained noise_mlp sees similar magnitude
        else:
            z = torch.randn(B, ndim, device=device)
        return z

    def _compute_vol_scale(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute baseline and vol_scale from history.

        Args:
            history: (B, T_hist, 5, 5) in [-1, 1]

        Returns:
            baseline: (B, 1, 5, 5) in [0, 1]
            vol_scale: (B, 1, 1, 1)
        """
        past_abs = denormalize_iv(history)  # (B, T, 5, 5)
        baseline = past_abs[:, -1:].clamp(min=0.01)  # (B, 1, 5, 5)

        mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T)
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
        vol_scale = (vol / self.config.global_mean_vol).clamp(
            self.config.vol_scale_min, self.config.vol_scale_max
        )
        vol_scale = vol_scale.pow(self.config.vol_scale_power)
        vol_scale = vol_scale.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        return baseline, vol_scale

    def _compute_percell_vol_scale(self, history: torch.Tensor) -> torch.Tensor:
        """Compute per-cell vol_scale from history daily change std.

        Args:
            history: (B, T_hist, 5, 5) in [-1, 1]

        Returns:
            vol_scale_cell: (B, 5, 5) per-cell scaling factor
        """
        past_abs = denormalize_iv(history)  # (B, T, 5, 5)
        daily_chg = past_abs[:, 1:] - past_abs[:, :-1]  # (B, T-1, 5, 5)
        cell_std = daily_chg.std(dim=1)  # (B, 5, 5)
        vol_scale_cell = (cell_std / self.config.global_mean_vol).clamp(
            self.config.vol_scale_min, self.config.vol_scale_max
        )
        return vol_scale_cell.pow(self.config.vol_scale_power)

    # ── GRU step-update helpers (for AR frame mode) ──

    def _init_gru_state(self, history: torch.Tensor):
        """Run GRU on history, return all hidden outputs + last hidden state."""
        B = history.shape[0]
        x = history.reshape(B, history.shape[1], -1)  # (B, T, 25)
        with torch.no_grad():
            all_outputs, h_last = self.encoder.gru(x)  # (B, T, H_gru), (1, B, H_gru)
        return all_outputs, h_last

    def _gru_step(self, frame_iv: torch.Tensor, all_outputs: torch.Tensor,
                  h_last: torch.Tensor):
        """Feed one generated frame into GRU, recompute attention-pooled condition."""
        B = frame_iv.shape[0]
        x = normalize_iv(frame_iv).reshape(B, 1, -1)  # (B, 1, 25)
        with torch.no_grad():
            new_output, h_last = self.encoder.gru(x, h_last)  # (B, 1, H_gru)
            all_outputs = torch.cat([all_outputs, new_output], dim=1)
            # Attention pooling (same as GRUEncoder.forward)
            attn_logits = self.encoder.attn_proj(all_outputs).squeeze(-1)  # (B, T')
            attn_weights = torch.softmax(attn_logits, dim=1)
            h = (attn_weights.unsqueeze(-1) * all_outputs).sum(dim=1)  # (B, H_gru)
            condition = self.encoder.bottleneck(h)  # (B, bottleneck_dim)
        return condition, all_outputs, h_last

    def _get_ar_frame_positions(
        self,
        step_idx: int,
        batch_size: int,
        device: torch.device,
        position_mode: str = "native",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Resolve local position and optional horizon bucket for AR-frame decoding."""
        if position_mode not in {"native", "raw", "cyclic"}:
            raise ValueError(f"Unsupported position_mode={position_mode!r}")

        if position_mode == "raw":
            local_pos_val = step_idx
            bucket_val = None
        elif position_mode == "cyclic":
            local_pos_val = step_idx % max(self.config.ar_local_pos_period, 1)
            bucket_val = None
        elif self.config.ar_dual_pos:
            local_pos_val = step_idx % max(self.config.ar_local_pos_period, 1)
            bucket_val = min(
                step_idx // max(self.config.ar_horizon_bucket_size, 1),
                self.config.ar_horizon_max_buckets - 1,
            )
        else:
            local_pos_val = step_idx
            bucket_val = None

        local_pos = torch.full(
            (batch_size,), local_pos_val, device=device, dtype=torch.long
        )
        horizon_bucket = None
        if bucket_val is not None:
            horizon_bucket = torch.full(
                (batch_size,), bucket_val, device=device, dtype=torch.long
            )
        return local_pos, horizon_bucket

    def _get_ar_frame_vol_scale(
        self,
        condition: torch.Tensor,
        base_vol_scale: torch.Tensor | None,
        percell_vol_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resolve per-frame vol scale, optionally modulated by a bounded dynamic head."""
        B = condition.shape[0]
        H, W = self.config.surface_h, self.config.surface_w

        if percell_vol_scale is not None:
            vs = percell_vol_scale
        elif base_vol_scale is not None:
            vs = base_vol_scale.view(B, 1, 1)
        else:
            vs = torch.ones(B, 1, 1, device=condition.device, dtype=condition.dtype)

        if not self.config.ar_dynamic_vs:
            return vs

        raw = self.dynamic_vs_head(condition)
        multiplier = torch.exp(self.config.ar_dynamic_vs_scale * torch.tanh(raw))
        multiplier = multiplier.clamp(0.75, 1.33)
        if self.config.ar_dynamic_vs_mode == "scalar":
            multiplier = multiplier.view(B, 1, 1)
        else:
            multiplier = multiplier.view(B, H, 1).expand(B, H, W)
        return vs * multiplier

    def _get_noise_for_decoder(self, z_t: torch.Tensor) -> torch.Tensor:
        """Convert raw noise z_t to FrameDecoder input (shared or factor model)."""
        if self.config.ar_factor_noise:
            # z_t: (B, n_factors) → cell_noise: (B, frame_dim) via learned loadings
            cell_noise = z_t @ self.factor_loadings.T
            if self.config.ar_factor_noise_norm:
                # Normalize to unit variance per cell — loadings only control correlation,
                # not scale. Prevents CRPS from shrinking loadings to reduce noise.
                cell_noise = cell_noise / (cell_noise.std(dim=0, keepdim=True) + 1e-6)
            return cell_noise
        return z_t

    @torch.no_grad()
    def _sample_ar_frame_trajectory(
        self,
        history: torch.Tensor,
        n_frames: int,
        position_mode: str = "native",
    ) -> torch.Tensor:
        """Sample one AR-frame trajectory in IV space."""
        B = history.shape[0]
        device = history.device
        H, W = self.config.surface_h, self.config.surface_w
        rho = self.config.ar_frame_rho
        log_space = self.config.ar_frame_log_space
        floor = self.config.ar_frame_floor_clamp

        _, vol_scale = self._compute_vol_scale(history)
        vol_scale_cell = None
        if self.config.ar_frame_percell_vol_scale:
            vol_scale_cell = self._compute_percell_vol_scale(history)

        z = self._sample_noise(B, device)
        z_t = z
        gru_outputs, h_last = self._init_gru_state(history)
        condition = self.encoder(history, mask=None)
        prev_frame = denormalize_iv(history[:, -1])  # (B, H, W)

        frames = []
        for step_idx in range(n_frames):
            if step_idx > 0:
                eps_t = torch.randn_like(z_t)
                z_t = rho * z_t + math.sqrt(1 - rho**2) * eps_t

            local_pos, horizon_bucket = self._get_ar_frame_positions(
                step_idx=step_idx,
                batch_size=B,
                device=device,
                position_mode=position_mode,
            )

            prev_flat = prev_frame.reshape(B, H * W)
            noise_input = self._get_noise_for_decoder(z_t)
            delta = self.frame_decoder(
                prev_flat, condition, noise_input, local_pos, horizon_bucket
            ).reshape(B, H, W)
            if hasattr(self, "cell_scale"):
                cs = self.cell_scale.clamp(0.3, 3.0).view(H, W)
                delta = cs * delta
            if hasattr(self, "cell_spread_linear"):
                cs = F.softplus(self.cell_spread_linear(condition)).view(B, H, W)
                delta = cs * delta

            vs = self._get_ar_frame_vol_scale(condition, vol_scale, vol_scale_cell)
            if log_space:
                iv_t = (prev_frame * torch.exp(vs * delta)).clamp(floor, 1.0)
            else:
                iv_t = (prev_frame + vs * delta).clamp(floor, 1.0)
            frames.append(iv_t)
            prev_frame = iv_t
            if not self.config.ar_freeze_gru_state:
                condition, gru_outputs, h_last = self._gru_step(
                    iv_t, gru_outputs, h_last
                )

        return torch.stack(frames, dim=1)

    def generate_block(
        self,
        condition: torch.Tensor,     # (B, bottleneck_dim)
        noise_z: torch.Tensor,       # (B, noise_dim)
        positions: torch.Tensor,     # (B, T)
        baseline: torch.Tensor = None,   # (B, 1, 5, 5) — not used in direct_iv mode
        vol_scale: torch.Tensor = None,  # (B, 1, 1, 1) — not used in direct_iv mode
    ) -> torch.Tensor:
        """Generate one block of IV surfaces.

        Returns:
            iv_block: (B, T, 5, 5) in [0, 1] IV space
        """
        H, W = self.config.surface_h, self.config.surface_w

        # Noise embedding (regime-dependent if cond_noise_mlp enabled)
        noise_emb = self.noise_mlp(noise_z, condition=condition)  # (B, noise_embed_dim)

        # Shared noise: first element of z vector → spatial input for cross-cell correlation
        shared_noise = noise_z[:, 0] if self.config.shared_noise_input else None

        # Decoder forward
        z_out = self.decoder(condition, noise_emb, positions, shared_noise_scalar=shared_noise)  # (B, T, 25)
        z_out = z_out.reshape(-1, positions.shape[1], H, W)  # (B, T, 5, 5)

        if self.config.direct_iv:
            # Direct IV: decoder output is normalized IV, denormalize to [0, 1]
            iv_block = denormalize_iv(z_out).clamp(0.001, 1.0)
        elif self.config.learned_vol_scale:
            # 1. Base samples with SCALAR vol_scale (preserves uniform kurtosis)
            base_iv = torch.exp(z_out * vol_scale) * baseline
            # 2. Condition-dependent per-cell spread (kurtosis-invariant)
            B = condition.shape[0]
            cell_spread = self.cell_spread_mlp(condition)  # (B, 25)
            cell_spread = cell_spread.clamp(0.85, 1.15)
            cell_spread = cell_spread.view(B, 1, H, W)  # (B, 1, 5, 5) broadcasts over T
            iv_block = (baseline + (base_iv - baseline) * cell_spread).clamp(0.001, 1.0)
        else:
            # Vol-scaled: IV = baseline × exp(z_out × vol_scale)
            iv_block = (torch.exp(z_out * vol_scale) * baseline).clamp(0.001, 1.0)

        return iv_block

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        n_members: int = 4,
        lambda_vs: float = 0.0,
        lambda_is: float = 0.0,
        lambda_cs_reg: float = 0.0,
        lambda_kurt: float = 0.0,
        n_train_blocks: int = 1,
        n_frames: int = 0,
    ) -> dict:
        """Training forward: generate K members, compute afCRPS.

        In ar_frame mode: per-frame autoregressive generation with GRU step updates.
        Otherwise: block-based generation with detached conditioning.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            future: (B, future_len, 5, 5) in [-1, 1]
            n_members: K ensemble members per sample
            lambda_vs: Variogram score weight (0 = disabled)
            n_train_blocks: Number of blocks to generate (block mode only)
            n_frames: Number of frames to generate (ar_frame mode; 0 = use blocks)

        Returns:
            dict with "loss" (with grad), plus detached diagnostics
        """
        B = history.shape[0]
        device = history.device
        H, W = self.config.surface_h, self.config.surface_w
        bs = self.config.block_size

        # Determine number of frames
        if self.config.ar_frame:
            if n_frames <= 0:
                n_frames = self.config.future_len
        else:
            n_frames = min(n_train_blocks * bs, self.config.future_len)

        # GT in IV space
        gt_iv = denormalize_iv(future[:, :n_frames])  # (B, n_frames, 5, 5)

        # Vol_scale from ORIGINAL history
        vol_scale = None
        vol_scale_cell = None
        if not self.config.direct_iv:
            with torch.no_grad():
                _, vol_scale = self._compute_vol_scale(history)
                if self.config.ar_frame_percell_vol_scale:
                    vol_scale_cell = self._compute_percell_vol_scale(history)

        # Generate K member trajectories
        all_member_trajectories = []
        all_deltas = []  # for bias loss (AR frame mode)

        if self.config.ar_frame:
            # ── AR frame mode: per-frame generation with GRU step updates ──
            rho = self.config.ar_frame_rho
            floor = self.config.ar_frame_floor_clamp

            for _ in range(n_members):
                z = self._sample_noise(B, device)
                z_t = z

                # Init GRU state from history
                gru_outputs, h_last = self._init_gru_state(history)

                # Initial condition from full history
                with torch.no_grad():
                    condition = self.encoder(history, mask=None)

                # prev_frame = last history frame in IV space
                prev_frame = denormalize_iv(history[:, -1])  # (B, 5, 5)

                frames = []
                for t in range(n_frames):
                    # AR noise update (skip first frame)
                    if t > 0:
                        eps_t = torch.randn_like(z_t)
                        z_t = rho * z_t + math.sqrt(1 - rho**2) * eps_t

                    local_pos, horizon_bucket = self._get_ar_frame_positions(
                        step_idx=t,
                        batch_size=B,
                        device=device,
                        position_mode="native",
                    )
                    prev_flat = prev_frame.reshape(B, H * W)

                    # Generate delta (condition detached, prev_frame has gradient)
                    cond_t = condition.detach()
                    noise_input = self._get_noise_for_decoder(z_t)
                    delta = self.frame_decoder(
                        prev_flat, cond_t, noise_input, local_pos, horizon_bucket
                    )
                    delta = delta.reshape(B, H, W)

                    # Static per-cell scale (clamped [0.3, 3.0])
                    if hasattr(self, 'cell_scale'):
                        cs = self.cell_scale.clamp(0.3, 3.0).view(H, W)
                        delta = cs * delta

                    # Residual: iv_t = prev + vol_scale * [cell_spread *] delta
                    if hasattr(self, 'cell_spread_linear'):
                        cs = F.softplus(self.cell_spread_linear(cond_t))
                        cs = cs.view(B, H, W)
                        delta = cs * delta
                    vs = self._get_ar_frame_vol_scale(cond_t, vol_scale, vol_scale_cell)
                    all_deltas.append(delta)
                    if self.config.ar_frame_log_space:
                        iv_t = (prev_frame * torch.exp(vs * delta)).clamp(floor, 1.0)
                    else:
                        iv_t = (prev_frame + vs * delta).clamp(floor, 1.0)
                    frames.append(iv_t)

                    # Update prev_frame — NOT detached (BPTT through frame chain)
                    prev_frame = iv_t

                    # Update GRU condition (no grad, frozen encoder)
                    gru_input = iv_t.detach()
                    if self.training and self.config.ar_input_noise_std > 0:
                        gru_input = gru_input + self.config.ar_input_noise_std * torch.randn_like(gru_input)
                    condition, gru_outputs, h_last = self._gru_step(
                        gru_input, gru_outputs, h_last
                    )

                trajectory = torch.stack(frames, dim=1)  # (B, n_frames, 5, 5)
                all_member_trajectories.append(trajectory)

        else:
            # ── Block mode: existing multi-block generation ──
            for _ in range(n_members):
                current_cond = history
                member_blocks = []
                z = self._sample_noise(B, device)

                for block_idx in range(n_train_blocks):
                    with torch.no_grad():
                        condition = self.encoder(current_cond, mask=None)
                        if hasattr(self.encoder, 'null_embedding'):
                            condition = condition + self.encoder.null_embedding.expand(B, -1)
                        baseline = None
                        if not self.config.direct_iv:
                            baseline = self._compute_vol_scale(current_cond)[0]
                    condition = condition.detach()

                    positions = (
                        torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                        + block_idx * bs
                    )

                    iv_block = self.generate_block(condition, z, positions, baseline, vol_scale)
                    member_blocks.append(iv_block)

                    block_norm = normalize_iv(iv_block.detach())
                    current_cond = torch.cat([current_cond, block_norm], dim=1)

                trajectory = torch.cat(member_blocks, dim=1)
                all_member_trajectories.append(trajectory)

        iv_samples = torch.stack(all_member_trajectories, dim=1)  # (B, K, n_frames, 5, 5)

        # afCRPS loss over full trajectory
        # frame_sum: sum over T/H/W, mean over B — each frame gets same gradient
        # magnitude as single-block. Multi-block adds h=30 gradient, doesn't dilute h=1.
        crps, mae, spread = afcrps_loss(
            iv_samples, gt_iv, alpha=0.95, reduction="frame_sum",
            cell_median=self.cell_median if self.config.twcrps_beta > 0 else None,
            cell_iqr=self.cell_iqr if self.config.twcrps_beta > 0 else None,
            twcrps_beta=self.config.twcrps_beta,
        )

        # Total loss (CRPS + variogram + interval score)
        loss = crps
        vs_val = torch.tensor(0.0, device=device)
        is_val = torch.tensor(0.0, device=device)
        if lambda_vs > 0:
            vs_val = variogram_score(iv_samples, gt_iv)
            loss = loss + lambda_vs * vs_val
        if lambda_is > 0:
            is_val = interval_score(iv_samples, gt_iv, alpha=0.9)
            loss = loss + lambda_is * is_val

        kurt_val = torch.tensor(0.0, device=device)
        raw_kurt_mean = torch.tensor(0.0, device=device)
        if lambda_kurt > 0:
            with torch.no_grad():
                ensemble_mean = iv_samples.mean(dim=1)  # (B, T, H, W)
            residuals = gt_iv - ensemble_mean.detach()
            m2 = residuals.pow(2).mean(dim=(0, 1))  # (H, W)
            m4 = residuals.pow(4).mean(dim=(0, 1))  # (H, W)
            raw_kurt = m4 / m2.pow(2).clamp(min=1e-8)  # (H, W)
            kurt_val = (raw_kurt - self.target_kurt).pow(2).mean()
            raw_kurt_mean = raw_kurt.mean().detach()
            loss = loss + lambda_kurt * kurt_val

        # Delta zero-mean bias loss (AR frame mode only)
        bias_loss = torch.tensor(0.0, device=device)
        if self.config.ar_frame_bias_lambda > 0 and len(all_deltas) > 0:
            # all_deltas: list of (B, H, W) tensors, ordered [m0_t0..tT, m1_t0..tT, ...]
            deltas = torch.stack(all_deltas, dim=0)  # (K*T, B, H, W)
            if self.config.ar_percell_bias:
                # Per-cell unconditional bias: average out members AND conditions
                T_actual = len(all_deltas) // n_members
                deltas_by_member = deltas.reshape(n_members, T_actual, B, H, W)
                member_mean = deltas_by_member.mean(dim=0)  # (T, B, 5, 5)
                percell_bias = member_mean.mean(dim=(0, 1))  # (H, W) — unconditional
                bias_loss = percell_bias.pow(2).mean()
            else:
                # Original: penalize cross-cell average per sample
                deltas_flat = deltas.reshape(deltas.shape[0] * deltas.shape[1], -1)  # (K*T*B, 25)
                bias_loss = deltas_flat.mean(dim=-1).pow(2).mean()
            loss = loss + self.config.ar_frame_bias_lambda * bias_loss

        return {
            "loss": loss,
            "crps": crps.detach(),
            "mae": mae.detach(),
            "spread": spread.detach(),
            "variogram": vs_val.detach(),
            "interval_score": is_val.detach(),
            "kurt_loss": kurt_val.detach(),
            "raw_kurt": raw_kurt_mean,
            "spread_mae_ratio": (spread / mae.clamp(min=1e-8)).detach(),
            "bias_loss": bias_loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 50,
        **kwargs,  # ignore DDPM-specific args for compatibility
    ) -> torch.Tensor:
        """Generate ensemble of futures (compatible with test suite interface).

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            n_samples: Number of ensemble members

        Returns:
            samples: (B, n_samples, future_len, 5, 5) in [0, 1]
        """
        B = history.shape[0]
        device = history.device

        if self.config.ar_frame:
            # ── AR frame mode: per-frame generation ──
            n_frames = int(kwargs.get("n_frames", self.config.future_len))
            position_mode = kwargs.get("position_mode", "native")

            all_samples = []
            for _ in range(n_samples):
                all_samples.append(
                    self._sample_ar_frame_trajectory(
                        history, n_frames=n_frames, position_mode=position_mode
                    )
                )

            # Stack and return in [0, 1] (already IV space, no denormalize needed)
            samples = torch.stack(all_samples, dim=1)  # (B, n_samples, future_len, 5, 5)
            return samples.clamp(0.0, 1.0)

        else:
            # ── Block mode: existing multi-block generation ──
            bs = self.config.block_size
            n_blocks = self.config.future_len // bs

            vol_scale = None
            if not self.config.direct_iv:
                _, vol_scale = self._compute_vol_scale(history)

            all_samples = []
            for _ in range(n_samples):
                current_cond = history
                blocks = []
                z = self._sample_noise(B, device)

                for block_idx in range(n_blocks):
                    condition = self.encoder(current_cond, mask=None)
                    if hasattr(self.encoder, 'null_embedding'):
                        condition = condition + self.encoder.null_embedding.expand(B, -1)

                    baseline = None
                    if not self.config.direct_iv:
                        baseline = self._compute_vol_scale(current_cond)[0]

                    positions = (
                        torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                        + block_idx * bs
                    )

                    iv_block = self.generate_block(condition, z, positions, baseline, vol_scale)
                    block_norm = normalize_iv(iv_block)
                    blocks.append(block_norm)
                    current_cond = torch.cat([current_cond, block_norm], dim=1)

                trajectory = torch.cat(blocks, dim=1)
                all_samples.append(trajectory)

            samples = torch.stack(all_samples, dim=1)
            samples = denormalize_iv(samples)
            samples = samples.clamp(0.0, 1.0)
            return samples

    def sample_batched(self, *args, **kwargs):
        """Alias for sample() — compatibility with test suite."""
        return self.sample(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────

def afcrps_loss(
    samples: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 0.95,
    reduction: str = "mean",
    cell_median: torch.Tensor = None,
    cell_iqr: torch.Tensor = None,
    twcrps_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Almost-Fair CRPS loss (ECMWF AIFS-CRPS, α=0.95).

    When twcrps_beta > 0, applies threshold-weighted CRPS (Taillardat et al. 2022):
    w(x) = 1 + beta * ((x - median) / IQR)^2. Upweights tail regions quadratically.

    Args:
        samples: (B, K, T, H, W) ensemble members in IV space
        gt: (B, T, H, W) ground truth in IV space
        alpha: interpolation between fair CRPS and standard CRPS
        reduction: "mean" averages over all dims (dilutes with more frames),
                   "frame_sum" sums over T/H/W and means over B (each frame
                   gets same gradient regardless of n_frames). Use "frame_sum"
                   for multi-block training to avoid gradient dilution.
        cell_median: (H, W) per-cell median from training data (for twCRPS)
        cell_iqr: (H, W) per-cell IQR from training data (for twCRPS)
        twcrps_beta: tail weight strength (0 = standard CRPS)

    Returns:
        (loss, mae_term, spread_term) — all scalars
    """
    K = samples.shape[1]

    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)

    if reduction == "per_window":
        # Scale-normalized per-window CRPS: divide each window's CRPS by its mean IV.
        # A calm window (mean IV=0.15) and turb window (mean IV=0.30) contribute equal
        # RELATIVE CRPS. Without this, turb dominates the gradient because its absolute
        # IV movements are 3-5x larger.
        mae_per_window = (samples - gt.unsqueeze(1)).abs().mean(dim=(1, 2, 3, 4))  # (B,)
        spread_per_window = (samples[:, idx_i] - samples[:, idx_j]).abs().mean(dim=(1, 2, 3, 4))  # (B,)
        window_scale = gt.mean(dim=(1, 2, 3)).clamp(min=0.01)  # (B,) mean IV per window
        mae = (mae_per_window / window_scale).mean()
        spread = (spread_per_window / window_scale).mean()
    elif reduction == "frame_sum":
        # Sum over T, H, W; mean over B and K — prevents gradient dilution
        # with more frames. Each frame contributes same gradient as in 1-block.
        mae_abs = (samples - gt.unsqueeze(1)).abs()  # (B, K, T, H, W)
        spread_abs = (samples[:, idx_i] - samples[:, idx_j]).abs()  # (B, n_pairs, T, H, W)

        if twcrps_beta > 0 and cell_median is not None:
            # twCRPS: weight only the MAE term by GT tail distance.
            # w(y) = 1 + beta * ((y - median) / IQR)^2
            # Penalizes errors in the tails more without rewarding spread in the tails.
            # Spread term stays unweighted — no feedback loop.
            deviation = (gt - cell_median) / cell_iqr  # (B, T, H, W)
            w_gt = 1.0 + twcrps_beta * deviation.pow(2)  # (B, T, H, W)
            mae_abs = mae_abs * w_gt.unsqueeze(1)  # broadcast over K

        mae_per_batch = mae_abs.mean(dim=1).sum(dim=(-3, -2, -1))  # (B,)
        mae = mae_per_batch.mean()
        spread_per_batch = spread_abs.mean(dim=1).sum(dim=(-3, -2, -1))
        spread = spread_per_batch.mean()
    else:
        # Standard: mean over everything (turb-dominated)
        mae = (samples - gt.unsqueeze(1)).abs().mean()
        spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()

    fcrps = mae - 0.5 * spread
    loss = alpha * fcrps + (1 - alpha) * mae

    return loss, mae, spread


def interval_score(
    samples: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 0.9,
) -> torch.Tensor:
    """Interval score for CI calibration.

    Penalizes wide intervals AND missed coverage. Has steep gradient for
    undercoverage — when GT falls outside the CI, penalty is proportional
    to distance scaled by 2/alpha. Much stronger spread signal than CRPS.

    Args:
        samples: (B, K, T, H, W) ensemble members in IV space
        gt: (B, T, H, W) ground truth in IV space
        alpha: CI level (0.9 = 90% CI)

    Returns:
        Scalar interval score loss
    """
    q_lo = 0.5 * (1 - alpha)  # 0.05 for 90% CI
    q_hi = 1 - q_lo            # 0.95
    lower = torch.quantile(samples, q_lo, dim=1)  # (B, T, H, W)
    upper = torch.quantile(samples, q_hi, dim=1)
    # Only penalize misses (GT outside CI), no width penalty.
    # Standard IS has `width + miss_low + miss_high` which rewards narrowing.
    # We want pure "widen where undercovered" signal.
    miss_low = (2.0 / alpha) * torch.relu(lower - gt)
    miss_high = (2.0 / alpha) * torch.relu(gt - upper)
    # Sum over T/H/W, mean over B — consistent with frame_sum CRPS
    return (miss_low + miss_high).sum(dim=(-3, -2, -1)).mean()


def variogram_score(
    samples: torch.Tensor,
    gt: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Vectorized variogram score for spatial coherence.

    Computes over all 300 unique pairs of 25 cells.

    Args:
        samples: (B, K, T, H, W) ensemble members
        gt: (B, T, H, W) ground truth

    Returns:
        Scalar variogram score loss
    """
    B, K, T, H, W = samples.shape
    D = H * W  # 25

    s = samples.reshape(B, K, T, D)  # (B, K, T, 25)
    g = gt.reshape(B, T, D)          # (B, T, 25)

    # All pairwise differences: (B, K, T, 25, 1) - (B, K, T, 1, 25)
    # Add eps before pow(p) to avoid NaN gradient at zero (d/dx x^0.5 = inf at x=0)
    eps = 1e-8
    s_diff = (s.unsqueeze(-1) - s.unsqueeze(-2)).abs().clamp(min=eps).pow(p)  # (B, K, T, 25, 25)
    g_diff = (g.unsqueeze(-1) - g.unsqueeze(-2)).abs().clamp(min=eps).pow(p)  # (B, T, 25, 25)

    # Average over K members
    s_mean = s_diff.mean(dim=1)  # (B, T, 25, 25)

    # Squared difference
    loss = (g_diff - s_mean).pow(2)

    # Upper triangle only (300 pairs)
    mask = torch.triu(torch.ones(D, D, device=samples.device), diagonal=1).bool()
    # Sum over T and pairs, mean over B — consistent with frame_sum CRPS
    return loss[:, :, mask].sum(dim=(-2, -1)).mean()


# ──────────────────────────────────────────────────────────────────────
# Weight Transfer
# ──────────────────────────────────────────────────────────────────────

def load_pretrained_weights(
    model: SinglePassBlockAR,
    checkpoint_path: str,
    device: str = "cpu",
    use_ema: bool = True,
) -> dict:
    """Load pretrained DDPM weights into SinglePassBlockAR.

    Transfers: encoder (exact), Conv3D backbone (exact), cond_proj (exact).
    Does NOT transfer: noise_embed (replaced by noise_mlp), conv_out (zero-init).

    Args:
        model: Target SinglePassBlockAR model
        checkpoint_path: Path to DDPM checkpoint
        device: Device to load on
        use_ema: Use EMA weights if available

    Returns:
        dict with transfer statistics
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if use_ema and "ema_params" in ckpt and ckpt["ema_params"]:
        src_state = ckpt["ema_params"]
    else:
        src_state = ckpt.get("model_state_dict", ckpt)

    stats = {"transferred": 0, "skipped": 0, "missing": 0}
    target_state = model.state_dict()

    # Mapping: DDPM key prefix → SinglePassBlockAR key prefix
    mappings = {
        "encoder.": "encoder.",           # exact copy
        "denoiser.pos_embed.": "decoder.pos_embed.",
        "denoiser.cond_proj.": "decoder.cond_proj.",
        "denoiser.conv_in.": "decoder.conv_in.",
        "denoiser.res_blocks.": "decoder.res_blocks.",
        "denoiser.ada_norms.": "decoder.ada_norms.",
        "denoiser.final_norm.": "decoder.final_norm.",
        "denoiser.final_act.": "decoder.final_act.",
        # conv_out deliberately NOT transferred (zero-init in target)
        # noise_embed deliberately NOT transferred (replaced by noise_mlp)
    }

    for src_key, src_val in src_state.items():
        matched = False
        for src_prefix, tgt_prefix in mappings.items():
            if src_key.startswith(src_prefix):
                tgt_key = src_key.replace(src_prefix, tgt_prefix, 1)
                if tgt_key in target_state:
                    if target_state[tgt_key].shape == src_val.shape:
                        target_state[tgt_key] = src_val
                        stats["transferred"] += 1
                    else:
                        stats["skipped"] += 1
                else:
                    stats["missing"] += 1
                matched = True
                break
        if not matched:
            stats["skipped"] += 1

    model.load_state_dict(target_state)
    return stats
