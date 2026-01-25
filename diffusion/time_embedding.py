"""
Time Embedding for Diffusion Models.

Provides sinusoidal position encoding for diffusion timesteps,
following the standard approach from transformer literature adapted for diffusion.

Reference: Vaswani et al., "Attention Is All You Need" (2017)
Adapted for diffusion in: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding.

    Maps integer timesteps to continuous embeddings using sin/cos encoding.
    This gives the model a smooth, continuous representation of time.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (should be even)
            max_period: Maximum period for sin/cos (higher = slower frequency variation)
        """
        super().__init__()
        assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embedding for timesteps.

        Args:
            t: Timesteps (B,) - integers or floats

        Returns:
            emb: Embeddings (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Compute frequencies: exp(-log(max_period) * i / (half_dim - 1))
        # This gives frequencies ranging from 1 to 1/max_period
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / (half_dim - 1)
        )

        # Compute arguments: t * freq
        # t: (B,) -> (B, 1)
        # freqs: (half_dim,) -> (1, half_dim)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        return emb


class TimeEmbedding(nn.Module):
    """
    Full time embedding with learnable MLP projection.

    Maps timesteps to embeddings via sinusoidal encoding + MLP.
    This is the standard approach in diffusion U-Nets.
    """

    def __init__(
        self,
        n_steps: int,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            n_steps: Number of diffusion timesteps (for reference, not used directly)
            embed_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
        """
        super().__init__()
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or 4 * embed_dim

        # Sinusoidal embedding (outputs embed_dim)
        self.sinusoidal = SinusoidalTimeEmbedding(embed_dim)

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding with MLP projection.

        Args:
            t: Timesteps (B,) - integers in [0, n_steps)

        Returns:
            emb: Embeddings (B, embed_dim)
        """
        # Sinusoidal encoding
        emb = self.sinusoidal(t)

        # MLP projection
        emb = self.mlp(emb)

        return emb


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization conditioned on time embedding.

    Applies GroupNorm then scales/shifts based on time embedding.
    This is a lightweight version of FiLM (Feature-wise Linear Modulation).

    Used in diffusion U-Nets to condition on timestep.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        embed_dim: int,
        eps: float = 1e-6,
    ):
        """
        Args:
            num_channels: Number of input channels
            num_groups: Number of groups for GroupNorm
            embed_dim: Time embedding dimension
            eps: Small constant for numerical stability
        """
        super().__init__()

        # Adjust groups to be compatible with channel count
        num_groups = min(num_groups, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        num_groups = max(1, num_groups)

        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)

        # Project time embedding to scale and shift
        self.proj = nn.Linear(embed_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalization.

        Args:
            x: Input tensor (B, C, ...) - any number of spatial dims
            t_emb: Time embedding (B, embed_dim)

        Returns:
            Normalized and scaled/shifted tensor (B, C, ...)
        """
        # Normalize
        x = self.norm(x)

        # Get scale and shift from time embedding
        params = self.proj(t_emb)  # (B, 2*C)
        scale, shift = params.chunk(2, dim=1)  # (B, C), (B, C)

        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1, ...) for spatial dims
        n_spatial = x.dim() - 2
        scale = scale.view(scale.shape[0], scale.shape[1], *([1] * n_spatial))
        shift = shift.view(shift.shape[0], shift.shape[1], *([1] * n_spatial))

        # Apply scale and shift
        return x * (1 + scale) + shift


def test_time_embedding():
    """Unit tests for time embedding modules."""
    print("Testing Time Embedding...")

    device = 'cpu'
    B = 4
    embed_dim = 64

    # Test sinusoidal embedding
    sin_emb = SinusoidalTimeEmbedding(embed_dim)
    t = torch.tensor([0, 25, 50, 99])
    emb = sin_emb(t)
    assert emb.shape == (B, embed_dim), f"Wrong shape: {emb.shape}"
    print(f"  Sinusoidal embedding: OK (shape {emb.shape})")

    # Check that different timesteps give different embeddings
    assert not torch.allclose(emb[0], emb[1]), "Different timesteps should give different embeddings"
    print("  Embedding uniqueness: OK")

    # Check smoothness: adjacent timesteps should have similar embeddings
    t_seq = torch.arange(100)
    emb_seq = sin_emb(t_seq)
    diffs = (emb_seq[1:] - emb_seq[:-1]).norm(dim=1)
    assert diffs.std() < diffs.mean(), "Embeddings should vary smoothly"
    print(f"  Embedding smoothness: OK (diff std={diffs.std():.4f} < mean={diffs.mean():.4f})")

    # Test full time embedding with MLP
    time_emb = TimeEmbedding(n_steps=100, embed_dim=embed_dim)
    t = torch.randint(0, 100, (B,))
    emb = time_emb(t)
    assert emb.shape == (B, embed_dim), f"Wrong shape: {emb.shape}"
    print(f"  TimeEmbedding with MLP: OK (shape {emb.shape})")

    # Test adaptive group norm
    C, T, H, W = 32, 10, 5, 5
    agn = AdaptiveGroupNorm(C, num_groups=8, embed_dim=embed_dim)
    x = torch.randn(B, C, T, H, W)
    t_emb = time_emb(torch.randint(0, 100, (B,)))
    out = agn(x, t_emb)
    assert out.shape == x.shape, f"AdaptiveGroupNorm changed shape: {out.shape} vs {x.shape}"
    print(f"  AdaptiveGroupNorm: OK (shape {out.shape})")

    print("Time Embedding tests passed!\n")


if __name__ == "__main__":
    test_time_embedding()
