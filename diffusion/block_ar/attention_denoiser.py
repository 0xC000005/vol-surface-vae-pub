"""
CSDI-style 2D Attention Denoiser for IV Surface Forecasting (Exp 118a).

Replaces Conv3D ResBlocks with temporal + feature self-attention.
Each ResidualBlock applies:
  1. Temporal attention: each cell attends across all 30 timesteps
  2. Feature attention: each timestep attends across all 25 cells
  3. Conditioning: noise embedding + encoder condition injected via gated projection

Input: noisy future (B, K, L) where K=25 cells, L=30 timesteps
Output: predicted noise (B, K, L)

Trained with DDPM score matching loss, sampled via DDIM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AttentionResidualBlock(nn.Module):
    """CSDI-style residual block with temporal + feature self-attention."""

    def __init__(self, channels: int, cond_dim: int, diffusion_dim: int,
                 n_heads: int = 4, n_cells: int = 25, n_steps: int = 30):
        super().__init__()
        self.n_cells = n_cells
        self.n_steps = n_steps

        # Diffusion timestep projection
        self.diff_proj = nn.Linear(diffusion_dim, channels)

        # Temporal self-attention: each cell attends across timesteps
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels * 2,
            activation="gelu", batch_first=True, dropout=0.0,
        )

        # Feature self-attention: each timestep attends across cells
        self.feat_attn = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels * 2,
            activation="gelu", batch_first=True, dropout=0.0,
        )

        # Conditioning projection (gated)
        self.cond_proj = nn.Conv1d(cond_dim, 2 * channels, 1)

        # Mid + output projections
        self.mid_proj = nn.Conv1d(channels, 2 * channels, 1)
        self.out_proj = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                diff_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, K*L) hidden state
            cond: (B, cond_dim, K*L) conditioning
            diff_emb: (B, diff_dim) diffusion timestep embedding
        Returns:
            (residual_out, skip_connection)
        """
        B, C, KL = x.shape
        K, L = self.n_cells, self.n_steps

        # Add diffusion embedding
        y = x + self.diff_proj(diff_emb).unsqueeze(-1)  # (B, C, K*L)

        # Temporal attention: reshape to (B*K, L, C), attend across L
        y_t = y.reshape(B, C, K, L).permute(0, 2, 3, 1).reshape(B * K, L, C)
        y_t = self.time_attn(y_t)  # (B*K, L, C)
        y = y_t.reshape(B, K, L, C).permute(0, 3, 1, 2).reshape(B, C, KL)

        # Feature attention: reshape to (B*L, K, C), attend across K
        y_f = y.reshape(B, C, K, L).permute(0, 3, 2, 1).reshape(B * L, K, C)
        y_f = self.feat_attn(y_f)  # (B*L, K, C)
        y = y_f.reshape(B, L, K, C).permute(0, 3, 2, 1).reshape(B, C, KL)

        # Mid projection
        y = self.mid_proj(y)  # (B, 2C, K*L)

        # Add conditioning (gated)
        y = y + self.cond_proj(cond)  # (B, 2C, K*L)

        # Gated activation
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)  # (B, C, K*L)

        # Output projection → residual + skip
        y = self.out_proj(y)  # (B, 2C, K*L)
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class AttentionDenoiser(nn.Module):
    """CSDI-style 2D attention denoiser for IV surface generation.

    Operates on flattened (K=25 cells, L=30 steps) with alternating
    temporal and feature self-attention layers.
    """

    def __init__(self, n_cells: int = 25, n_steps: int = 30,
                 channels: int = 64, n_layers: int = 4, n_heads: int = 4,
                 cond_dim: int = 128, n_diffusion_steps: int = 200):
        super().__init__()
        self.n_cells = n_cells
        self.n_steps = n_steps
        self.channels = channels

        # Diffusion timestep embedding
        diff_dim = 128
        self.diff_embed = nn.Sequential(
            SinusoidalEmbedding(diff_dim),
            nn.Linear(diff_dim, diff_dim),
            nn.SiLU(),
            nn.Linear(diff_dim, diff_dim),
            nn.SiLU(),
        )

        # Input projection: 2 channels (noisy_target + mask) → C
        # For forecasting: channel 0 = noisy future, channel 1 = history indicator
        self.input_proj = nn.Conv1d(2, channels, 1)

        # Condition projection: encoder output → per-cell-per-step conditioning
        # Expand (B, cond_dim) → (B, cond_dim, K*L) via learned projection
        self.cond_expand = nn.Sequential(
            nn.Linear(cond_dim, n_cells * n_steps),
            nn.SiLU(),
        )

        # Residual blocks with 2D attention
        self.blocks = nn.ModuleList([
            AttentionResidualBlock(
                channels=channels, cond_dim=cond_dim, diffusion_dim=diff_dim,
                n_heads=n_heads, n_cells=n_cells, n_steps=n_steps,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.out_proj1 = nn.Conv1d(channels, channels, 1)
        self.out_proj2 = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.out_proj2.weight)
        nn.init.zeros_(self.out_proj2.bias)

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor,
                diffusion_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_target: (B, K, L) noisy future IV surfaces in [0, 1]
            condition: (B, cond_dim) GRU encoder output
            diffusion_step: (B,) integer timestep [0, T)
        Returns:
            noise_pred: (B, K, L) predicted noise
        """
        B = noisy_target.shape[0]
        K, L = self.n_cells, self.n_steps
        KL = K * L

        # Diffusion embedding
        diff_emb = self.diff_embed(diffusion_step)  # (B, 128)

        # Input: noisy target + zeros (no mask for forecasting)
        x_flat = noisy_target.reshape(B, 1, KL)  # (B, 1, K*L)
        mask = torch.zeros_like(x_flat)  # (B, 1, K*L)
        x = torch.cat([x_flat, mask], dim=1)  # (B, 2, K*L)
        x = self.input_proj(x)  # (B, C, K*L)
        x = F.relu(x)

        # Conditioning: expand encoder output to per-cell-per-step
        cond_expanded = self.cond_expand(condition)  # (B, K*L)
        cond_spatial = condition.unsqueeze(-1).expand(B, -1, KL)  # (B, cond_dim, K*L)

        # Residual blocks
        skip_sum = torch.zeros(B, self.channels, KL, device=x.device)
        for block in self.blocks:
            x_reshaped = x.reshape(B, self.channels, K, L)
            x, skip = block(x_reshaped.reshape(B, self.channels, KL),
                          cond_spatial, diff_emb)
            skip_sum = skip_sum + skip

        # Output
        x = skip_sum / math.sqrt(len(self.blocks))
        x = self.out_proj1(x)
        x = F.relu(x)
        x = self.out_proj2(x)  # (B, 1, K*L)

        return x.reshape(B, K, L)
