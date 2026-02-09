"""
BiGRU Denoiser for Block-AR Diffusion.

Bidirectional GRU noise predictor that takes noisy frames, conditioning,
positional embeddings, and per-frame noise level embeddings to predict
the noise component for each frame.

Uses FiLM conditioning (scale + shift) to modulate features via the
conditioning vector, preventing the optimizer from ignoring conditioning.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusion.time_embedding import SinusoidalTimeEmbedding


@dataclass
class DenoiserConfig:
    frame_dim: int = 25
    surface_h: int = 5
    surface_w: int = 5
    bottleneck_dim: int = 64
    pos_embed_dim: int = 16
    noise_embed_dim: int = 16
    gru_hidden_dim: int = 128
    spatial_conv_channels: int = 8
    dropout: float = 0.1


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale and shift features by conditioning."""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        # Initialize near-identity: scale=1, shift=0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) features to modulate
            cond: (B, cond_dim) conditioning vector
        Returns:
            (B, T, D) modulated features
        """
        params = self.proj(cond)  # (B, 2*D)
        scale, shift = params.unsqueeze(1).chunk(2, dim=-1)  # each (B, 1, D)
        return x * (1 + scale) + shift


class SpatialConvBlock(nn.Module):
    """Lightweight spatial conv: flat 25-dim -> 5x5 -> conv -> 5x5 -> flat 25-dim."""

    def __init__(self, h: int, w: int, mid_channels: int = 8):
        super().__init__()
        self.h = h
        self.w = w
        self.conv = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H*W) flattened spatial frames
        Returns:
            (B, T, H*W) spatially processed frames
        """
        B, T, D = x.shape
        # Fold B*T into batch for per-frame conv
        x = x.reshape(B * T, 1, self.h, self.w)
        x = self.conv(x)  # (B*T, 1, H, W)
        return x.reshape(B, T, D)


class BiGRUDenoiser(nn.Module):
    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config

        # Spatial conv: pre-process noisy frames and post-process noise prediction
        self.spatial_pre = SpatialConvBlock(
            config.surface_h, config.surface_w, config.spatial_conv_channels,
        )
        self.spatial_post = SpatialConvBlock(
            config.surface_h, config.surface_w, config.spatial_conv_channels,
        )

        self.pos_embed = SinusoidalTimeEmbedding(dim=config.pos_embed_dim)
        self.noise_embed = SinusoidalTimeEmbedding(dim=config.noise_embed_dim)

        # Input projection WITHOUT conditioning (FiLM applied after)
        input_dim = (
            config.frame_dim
            + config.pos_embed_dim
            + config.noise_embed_dim
        )
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.gru_hidden_dim),
            nn.SiLU(),
        )
        self.input_dropout = nn.Dropout(config.dropout)

        # FiLM conditioning: applied before and after GRU
        self.film_pre = FiLMLayer(config.bottleneck_dim, config.gru_hidden_dim)
        self.film_post = FiLMLayer(config.bottleneck_dim, config.gru_hidden_dim * 2)

        self.bigru = nn.GRU(
            input_size=config.gru_hidden_dim,
            hidden_size=config.gru_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

        self.output_proj = nn.Linear(config.gru_hidden_dim * 2, config.frame_dim)

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
        # Spatial pre-processing: learn local spatial structure per frame
        frames = noisy_frames + self.spatial_pre(noisy_frames)  # residual connection

        # Embeddings
        pos_emb = self.pos_embed(positions)  # (B, T_block, pos_embed_dim)
        noise_emb = self.noise_embed(noise_levels)  # (B, T_block, noise_embed_dim)

        # Concatenate spatially-processed frame + position + noise
        x = torch.cat(
            [frames, pos_emb, noise_emb], dim=-1
        )  # (B, T_block, input_dim)

        # Project and apply dropout
        x = self.input_proj(x)  # (B, T_block, gru_hidden_dim)
        x = self.input_dropout(x)

        # FiLM conditioning before GRU
        x = self.film_pre(x, condition)

        # Bidirectional GRU
        x, _ = self.bigru(x)  # (B, T_block, gru_hidden_dim * 2)

        # FiLM conditioning after GRU
        x = self.film_post(x, condition)

        # Output projection to noise prediction
        noise_pred = self.output_proj(x)  # (B, T_block, frame_dim)

        # Spatial post-processing: enforce spatial coherence in noise prediction
        noise_pred = noise_pred + self.spatial_post(noise_pred)  # residual connection

        return noise_pred
