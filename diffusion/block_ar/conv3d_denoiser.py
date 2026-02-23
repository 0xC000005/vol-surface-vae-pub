"""
Conv3D Denoiser for Block-AR Diffusion.

3D convolutional noise predictor that processes blocks as (B, 1, T, 5, 5) volumes.

Two variants:
- Conv3DBlockDenoiser: Standard (non-causal) Conv3d for MCVD bidirectional tasks
- CausalConv3DBlockDenoiser: CausalConv3d for forward-only, enables skewness

Architecture: Conv3d(1, C) -> N x [ResBlock3D + AdaptiveGroupNorm] -> Conv3d(C, 1)
Conditioning via AdaptiveGroupNorm (GroupNorm + FiLM scale/shift from embeddings).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from vae.causal_3d_blocks import CausalConv3d, ResnetBlockCausal3D
from diffusion.time_embedding import (
    AdaptiveGroupNorm,
    SinusoidalTimeEmbedding,
    TimeEmbedding,
)


@dataclass
class Conv3DDenoiserConfig:
    frame_dim: int = 25
    surface_h: int = 5
    surface_w: int = 5
    bottleneck_dim: int = 64
    pos_embed_dim: int = 16
    noise_embed_dim: int = 64  # TimeEmbedding with MLP (matches SimpleDenoiser3D)
    n_steps: int = 100
    base_channels: int = 32
    n_res_blocks: int = 4
    groups: int = 8


class ResBlock3D(nn.Module):
    """Standard 3D residual block with non-causal Conv3d."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        # Initialize conv2 near-zero for residual near-identity at init
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, C, T, H, W)"""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class Conv3DBlockDenoiser(nn.Module):
    """Conv3D denoiser for Block-AR diffusion.

    Drop-in replacement for BiGRUDenoiser with identical forward interface.
    """

    def __init__(self, config: Conv3DDenoiserConfig):
        super().__init__()
        self.config = config
        C = config.base_channels

        # Noise level embedding: TimeEmbedding with MLP (64-dim)
        self.noise_embed = TimeEmbedding(
            n_steps=config.n_steps, embed_dim=config.noise_embed_dim
        )

        # Position embedding: lightweight sinusoidal (16-dim)
        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)

        # Conditioning projection: concat(noise_emb, pos_emb, condition) -> C
        cond_input_dim = config.noise_embed_dim + config.pos_embed_dim + config.bottleneck_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, C * 4),
            nn.SiLU(),
            nn.Linear(C * 4, C),
        )

        # Conv3D backbone
        self.conv_in = nn.Conv3d(1, C, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList()
        self.ada_norms = nn.ModuleList()
        for _ in range(config.n_res_blocks):
            self.res_blocks.append(ResBlock3D(C, groups=config.groups))
            self.ada_norms.append(AdaptiveGroupNorm(C, config.groups, embed_dim=C))

        self.final_norm = nn.GroupNorm(config.groups, C)
        self.final_act = nn.SiLU()
        self.conv_out = nn.Conv3d(C, 1, kernel_size=3, padding=1)

        # Initialize conv_out near-zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        noisy_frames: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise for each frame.

        Args:
            noisy_frames: (B, T_block, 25) flattened noisy IV surfaces
            condition: (B, bottleneck_dim) conditioning from encoder
            positions: (B, T_block) long - absolute frame positions
            noise_levels: (B, T_block) long - per-frame diffusion timesteps

        Returns:
            noise_pred: (B, T_block, 25) predicted noise
        """
        B, T, D = noisy_frames.shape
        H, W = self.config.surface_h, self.config.surface_w

        # Embeddings
        noise_emb = self.noise_embed(noise_levels)  # (B, T, noise_embed_dim)
        pos_emb = self.pos_embed(positions)          # (B, T, pos_embed_dim)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)

        # Combined conditioning: (B, T, C)
        cond_cat = torch.cat([noise_emb, pos_emb, cond_expanded], dim=-1)  # (B, T, cond_input_dim)
        cond = self.cond_proj(cond_cat.reshape(B * T, -1)).reshape(B, T, -1)  # (B, T, C)

        # Reshape input to volume: (B, T, 25) -> (B, 1, T, 5, 5)
        x = noisy_frames.reshape(B, T, H, W).unsqueeze(1)  # (B, 1, T, H, W)

        # Conv3D backbone
        x = self.conv_in(x)  # (B, C, T, H, W)

        for res_block, ada_norm in zip(self.res_blocks, self.ada_norms):
            x = res_block(x)           # (B, C, T, H, W)
            x = ada_norm(x, cond)      # AdaGN with per-frame conditioning

        x = self.final_act(self.final_norm(x))
        x = self.conv_out(x)  # (B, 1, T, H, W)

        # Reshape back: (B, 1, T, H, W) -> (B, T, 25)
        noise_pred = x.squeeze(1).reshape(B, T, D)

        return noise_pred


class CausalConv3DBlockDenoiser(nn.Module):
    """CausalConv3D denoiser for Block-AR diffusion.

    Uses CausalConv3d (asymmetric temporal padding) to preserve temporal
    causality. This creates structural asymmetry in the reverse diffusion
    process that preserves skewness — matching DDPM POC's SimpleDenoiser3D.

    Drop-in replacement for Conv3DBlockDenoiser with identical forward interface.
    """

    def __init__(self, config: Conv3DDenoiserConfig):
        super().__init__()
        self.config = config
        C = config.base_channels

        # Noise level embedding: TimeEmbedding with MLP (64-dim)
        self.noise_embed = TimeEmbedding(
            n_steps=config.n_steps, embed_dim=config.noise_embed_dim
        )

        # Position embedding: lightweight sinusoidal (16-dim)
        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)

        # Conditioning projection: concat(noise_emb, pos_emb, condition) -> C
        cond_input_dim = config.noise_embed_dim + config.pos_embed_dim + config.bottleneck_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_input_dim, C * 4),
            nn.SiLU(),
            nn.Linear(C * 4, C),
        )

        # CausalConv3D backbone (matches SimpleDenoiser3D architecture)
        self.conv_in = CausalConv3d(1, C, kernel_size=3)

        self.res_blocks = nn.ModuleList()
        self.ada_norms = nn.ModuleList()
        for _ in range(config.n_res_blocks):
            self.res_blocks.append(
                ResnetBlockCausal3D(C, C, groups=config.groups)
            )
            self.ada_norms.append(
                AdaptiveGroupNorm(C, config.groups, embed_dim=C)
            )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(config.groups, C),
            nn.SiLU(),
            CausalConv3d(C, 1, kernel_size=3),
        )

    def forward(
        self,
        noisy_frames: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise for each frame (same interface as Conv3DBlockDenoiser).

        Args:
            noisy_frames: (B, T_block, 25) flattened noisy IV surfaces
            condition: (B, bottleneck_dim) conditioning from encoder
            positions: (B, T_block) long - absolute frame positions
            noise_levels: (B, T_block) long - per-frame diffusion timesteps

        Returns:
            noise_pred: (B, T_block, 25) predicted noise
        """
        B, T, D = noisy_frames.shape
        H, W = self.config.surface_h, self.config.surface_w

        # Embeddings
        noise_emb = self.noise_embed(noise_levels)  # (B, T, noise_embed_dim)
        pos_emb = self.pos_embed(positions)          # (B, T, pos_embed_dim)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1)  # (B, T, bottleneck_dim)

        # Combined conditioning: (B, T, C)
        cond_cat = torch.cat([noise_emb, pos_emb, cond_expanded], dim=-1)
        cond = self.cond_proj(cond_cat.reshape(B * T, -1)).reshape(B, T, -1)

        # Reshape input to volume: (B, T, 25) -> (B, 1, T, 5, 5)
        x = noisy_frames.reshape(B, T, H, W).unsqueeze(1)

        # CausalConv3D backbone
        x = self.conv_in(x)  # (B, C, T, H, W)

        for res_block, ada_norm in zip(self.res_blocks, self.ada_norms):
            x = res_block(x)           # (B, C, T, H, W)
            x = ada_norm(x, cond)      # AdaGN with per-frame conditioning

        x = self.conv_out(x)  # (B, 1, T, H, W)

        # Reshape back: (B, 1, T, H, W) -> (B, T, 25)
        noise_pred = x.squeeze(1).reshape(B, T, D)

        return noise_pred
