"""
GRU Encoder for Block-AR Diffusion.

Encodes variable-length IV surface sequences into a fixed-size bottleneck
conditioning vector. Supports per-item masking for MCVD task types and
conditioning augmentation to mitigate exposure bias in autoregressive generation.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    input_dim: int = 25
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 64
    cond_aug_sigma: float = 0.0
    dropout: float = 0.1


class GRUEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.gru_hidden_dim,
            batch_first=True,
        )
        # Attention pooling over all GRU hidden states (instead of last-only)
        self.attn_proj = nn.Linear(config.gru_hidden_dim, 1)
        self.bottleneck = nn.Linear(config.gru_hidden_dim, config.bottleneck_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.null_embedding = nn.Parameter(torch.zeros(1, config.bottleneck_dim))
        self.cond_aug_sigma = config.cond_aug_sigma

    def forward(
        self, surfaces: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode IV surface sequence into bottleneck conditioning vector.

        Uses attention pooling over ALL GRU hidden states, preventing
        the exponential information decay of last-hidden-state extraction.

        Args:
            surfaces: (B, T, 5, 5) IV surfaces in [-1, 1]
            mask: (B,) bool tensor or None. True = replace with null_embedding.

        Returns:
            (B, bottleneck_dim) conditioning vector
        """
        B = surfaces.shape[0]

        # Flatten spatial dims: (B, T, 5, 5) -> (B, T, 25)
        x = surfaces.reshape(B, surfaces.shape[1], -1)

        # GRU forward — use ALL hidden states, not just final
        output, _ = self.gru(x)  # output: (B, T, gru_hidden_dim)

        # Attention pooling: learn which timesteps matter most
        attn_logits = self.attn_proj(output).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T)
        h = (attn_weights.unsqueeze(-1) * output).sum(dim=1)  # (B, gru_hidden_dim)

        # Bottleneck projection
        output = self.bottleneck(h)  # (B, bottleneck_dim)

        # Dropout
        output = self.dropout(output)

        # Conditioning augmentation (training only)
        if self.training:
            output = output + self.cond_aug_sigma * torch.randn_like(output)

        # Apply mask: replace masked items with null_embedding
        if mask is not None:
            output = torch.where(
                mask.unsqueeze(-1),
                self.null_embedding.expand(B, -1),
                output,
            )

        return output
