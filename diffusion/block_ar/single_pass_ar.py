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

    # Vol-scaled denormalization
    global_mean_vol: float = 0.0187
    vol_scale_min: float = 0.5
    vol_scale_max: float = 2.0
    vol_scale_power: float = 1.0

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
    """

    def __init__(self, noise_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        # Small random init on output layer (NOT zero) to avoid dead start.
        # With zero init, all K members produce identical output, spread=0,
        # and CRPS gradient has no signal to break the symmetry.
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, noise_dim)
        Returns:
            emb: (B, embed_dim)
        """
        return self.mlp(z)


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

        # Combined conditioning: same as Conv3DBlockDenoiser line 259
        cond_cat = torch.cat([noise_emb_expanded, pos_emb, cond_expanded], dim=-1)
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

        # tanh clamping: bounds z_out to [-1, 1]
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

        # Noise MLP (replaces TimeEmbedding)
        self.noise_mlp = NoiseMLP(config.noise_dim, config.noise_embed_dim)

        # Decoder (modified Conv3D)
        self.decoder = SinglePassDecoder(config)

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

    def generate_block(
        self,
        condition: torch.Tensor,     # (B, bottleneck_dim)
        noise_z: torch.Tensor,       # (B, noise_dim)
        positions: torch.Tensor,     # (B, T)
        baseline: torch.Tensor,      # (B, 1, 5, 5)
        vol_scale: torch.Tensor,     # (B, 1, 1, 1)
    ) -> torch.Tensor:
        """Generate one block of IV surfaces.

        Args:
            condition: Encoded history
            noise_z: Noise vector for this ensemble member
            positions: Frame indices for this block
            baseline: Baseline IV from history
            vol_scale: Regime-dependent scale factor

        Returns:
            iv_block: (B, T, 5, 5) in [0, 1] IV space
        """
        H, W = self.config.surface_h, self.config.surface_w

        # Noise embedding
        noise_emb = self.noise_mlp(noise_z)  # (B, noise_embed_dim)

        # Shared noise: first element of z vector → spatial input for cross-cell correlation
        shared_noise = noise_z[:, 0] if self.config.shared_noise_input else None

        # Decoder forward
        z_out = self.decoder(condition, noise_emb, positions, shared_noise_scalar=shared_noise)  # (B, T, 25)
        z_out = z_out.reshape(-1, positions.shape[1], H, W)  # (B, T, 5, 5)

        # Vol-scaled denormalization: IV = baseline × exp(z_out × vol_scale)
        iv_block = (torch.exp(z_out * vol_scale) * baseline).clamp(0.001, 1.0)

        return iv_block

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        n_members: int = 4,
        lambda_vs: float = 0.0,
        n_train_blocks: int = 1,
    ) -> dict:
        """Training forward: generate K members over n_train_blocks, compute afCRPS.

        With n_train_blocks=3, generates the full 30-frame trajectory via
        autoregressive chaining with detached conditioning. Each block gets
        direct CRPS gradient, but gradients don't flow across block boundaries.
        The model sees the full horizon and learns that calm windows need
        growing uncertainty even at h=30.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            future: (B, future_len, 5, 5) in [-1, 1]
            n_members: K ensemble members per sample
            lambda_vs: Variogram score weight (0 = disabled)
            n_train_blocks: Number of blocks to generate (1=block1 only, 3=full 30 frames)

        Returns:
            dict with "loss" (with grad), plus detached diagnostics
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size

        # GT in IV space (all blocks we're training on)
        n_frames = min(n_train_blocks * bs, self.config.future_len)
        gt_iv = denormalize_iv(future[:, :n_frames])  # (B, n_frames, 5, 5)

        # Generate K member trajectories
        all_member_trajectories = []

        for _ in range(n_members):
            current_cond = history  # (B, T_hist, 5, 5) — grows with generated blocks
            member_blocks = []

            for block_idx in range(n_train_blocks):
                # Encode condition from growing context (detached for blocks > 0)
                with torch.no_grad():
                    condition = self.encoder(current_cond, mask=None)
                    if hasattr(self.encoder, 'null_embedding'):
                        condition = condition + self.encoder.null_embedding.expand(B, -1)
                    baseline, vol_scale = self._compute_vol_scale(current_cond)
                # Detach condition so gradients only flow through this block's decoder
                condition = condition.detach()

                # Positions for this block
                positions = (
                    torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                    + block_idx * bs
                )

                # Fresh noise per block per member
                z = torch.randn(B, self.config.noise_dim, device=device)

                # Generate block (has gradient through decoder)
                iv_block = self.generate_block(condition, z, positions, baseline, vol_scale)
                member_blocks.append(iv_block)

                # Grow context with DETACHED generated block (no cross-block gradient)
                block_norm = normalize_iv(iv_block.detach())
                current_cond = torch.cat([current_cond, block_norm], dim=1)

            # Concatenate blocks into full trajectory
            trajectory = torch.cat(member_blocks, dim=1)  # (B, n_frames, 5, 5)
            all_member_trajectories.append(trajectory)

        iv_samples = torch.stack(all_member_trajectories, dim=1)  # (B, K, n_frames, 5, 5)

        # afCRPS loss over full trajectory
        crps, mae, spread = afcrps_loss(iv_samples, gt_iv, alpha=0.95)

        # Total loss (CRPS + variogram)
        loss = crps
        vs_val = torch.tensor(0.0, device=device)
        if lambda_vs > 0:
            vs_val = variogram_score(iv_samples, gt_iv)
            loss = loss + lambda_vs * vs_val

        return {
            "loss": loss,
            "crps": crps.detach(),
            "mae": mae.detach(),
            "spread": spread.detach(),
            "variogram": vs_val.detach(),
            "spread_mae_ratio": (spread / mae.clamp(min=1e-8)).detach(),
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
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs

        all_samples = []
        for _ in range(n_samples):
            current_cond = history
            blocks = []

            for block_idx in range(n_blocks):
                # Encode growing context
                condition = self.encoder(current_cond, mask=None)
                if hasattr(self.encoder, 'null_embedding'):
                    condition = condition + self.encoder.null_embedding.expand(B, -1)

                # Baseline and vol_scale from current context
                baseline, vol_scale = self._compute_vol_scale(current_cond)

                # Positions for this block
                positions = (
                    torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                    + block_idx * bs
                )

                # Fresh noise for this block
                z = torch.randn(B, self.config.noise_dim, device=device)

                # Generate block in IV space
                iv_block = self.generate_block(condition, z, positions, baseline, vol_scale)

                # Convert back to normalized space for context growing
                block_norm = normalize_iv(iv_block)
                blocks.append(block_norm)

                # Grow context
                current_cond = torch.cat([current_cond, block_norm], dim=1)

            trajectory = torch.cat(blocks, dim=1)  # (B, future_len, 5, 5)
            all_samples.append(trajectory)

        samples = torch.stack(all_samples, dim=1)  # (B, n_samples, future_len, 5, 5)
        # Denormalize to [0, 1]
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Almost-Fair CRPS loss (ECMWF AIFS-CRPS, α=0.95).

    Args:
        samples: (B, K, T, H, W) ensemble members in IV space
        gt: (B, T, H, W) ground truth in IV space
        alpha: interpolation between fair CRPS and standard CRPS

    Returns:
        (loss, mae_term, spread_term) — all scalars
    """
    K = samples.shape[1]

    # Reliability: mean absolute error across members
    mae = (samples - gt.unsqueeze(1)).abs().mean()

    # Sharpness: vectorized pairwise spread
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=samples.device)
    spread = (samples[:, idx_i] - samples[:, idx_j]).abs().mean()

    # fair CRPS = MAE - 0.5 * spread
    fcrps = mae - 0.5 * spread
    # almost-fair: interpolate to avoid degeneracy
    loss = alpha * fcrps + (1 - alpha) * mae

    return loss, mae, spread


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
    return loss[:, :, mask].mean()


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
