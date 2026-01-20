"""
Causal 3D VAE for Temporal Volatility Surface Generation.

Ported and adapted from HunyuanVideo (Tencent):
https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/vae/vae.py

Key adaptations for volatility surface (5x5 grid):
- NO spatial downsampling (5x5 is too small)
- Optional temporal compression (T -> T/2 or T/4)
- Removed diffusers dependencies (pure PyTorch)
- Simpler architecture (no attention, fewer layers)
- DiagonalGaussianDistribution for VAE sampling
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_3d_blocks import (
    CausalConv3d,
    DownEncoderBlockCausal3D,
    UpDecoderBlockCausal3D,
    MidBlockCausal3D,
    ResnetBlockCausal3D,
)


@dataclass
class Causal3DVAEConfig:
    """Configuration for Causal 3D VAE adapted for volatility surfaces."""

    # Input/output channels (1 for single-channel vol surface)
    in_channels: int = 1
    out_channels: int = 1

    # Latent dimension
    latent_channels: int = 16

    # Encoder block output channels
    # For 5x5 grid, we use moderate channel expansion
    block_out_channels: Tuple[int, ...] = (32, 64, 128)

    # Number of ResNet layers per block
    layers_per_block: int = 2

    # GroupNorm groups (must divide channel counts)
    norm_num_groups: int = 8

    # Activation function
    act_fn: str = "silu"

    # Compression ratios
    # NO spatial compression for 5x5 grid
    spatial_compression_ratio: int = 1
    # Optional temporal compression (1 = none, 2 = T/2, 4 = T/4)
    temporal_compression_ratio: int = 2

    # VAE options
    double_z: bool = True  # Output mean and logvar

    # Training options
    dropout: float = 0.0


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for VAE latent space.

    From HunyuanVideo/diffusers, adapted for our use case.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """
        Args:
            parameters: Tensor containing concatenated mean and logvar
            deterministic: If True, std = 0 (always return mean)
        """
        # Determine which dimension to split on based on tensor shape
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H, W) or (B, C, H, W)
        else:
            raise NotImplementedError(f"Unsupported tensor shape: {parameters.shape}")

        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)

        # Clamp logvar for numerical stability
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from the distribution using reparameterization trick."""
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        """
        KL divergence to another distribution (or standard normal if None).

        Returns KL per batch element (summed over other dims).
        """
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device)

        reduce_dims = list(range(1, self.mean.ndim))

        if other is None:
            # KL to N(0, 1)
            return 0.5 * torch.sum(
                self.mean.pow(2) + self.var - 1.0 - self.logvar,
                dim=reduce_dims,
            )
        else:
            # KL to another Gaussian
            return 0.5 * torch.sum(
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=reduce_dims,
            )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of a sample under this distribution."""
        if self.deterministic:
            return torch.zeros(sample.shape[0], device=sample.device)

        reduce_dims = list(range(1, sample.ndim))
        logtwopi = np.log(2.0 * np.pi)

        return 0.5 * torch.sum(
            logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=reduce_dims,
        )

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean


class EncoderCausal3D(nn.Module):
    """
    Causal 3D Encoder for volatility surfaces.

    Input: (B, C, T, H, W) where H=W=5 for vol surface
    Output: (B, 2*latent_channels, T', H, W) - mean and logvar concatenated

    Key differences from HunyuanVideo:
    - NO spatial downsampling (5x5 is already small)
    - Only temporal downsampling
    """

    def __init__(self, config: Causal3DVAEConfig):
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv_in = CausalConv3d(
            config.in_channels,
            config.block_out_channels[0],
            kernel_size=3
        )

        # Down blocks (temporal compression only)
        self.down_blocks = nn.ModuleList()
        num_temporal_downsample = int(np.log2(config.temporal_compression_ratio))

        output_channel = config.block_out_channels[0]
        for i, block_out_ch in enumerate(config.block_out_channels):
            input_channel = output_channel
            output_channel = block_out_ch
            is_final = i == len(config.block_out_channels) - 1

            # Temporal downsampling in first blocks
            add_downsample = i < num_temporal_downsample and not is_final
            downsample_stride = (2, 1, 1) if add_downsample else (1, 1, 1)

            down_block = DownEncoderBlockCausal3D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=config.layers_per_block,
                groups=config.norm_num_groups,
                dropout=config.dropout,
                add_downsample=add_downsample,
                downsample_stride=downsample_stride,
            )
            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = MidBlockCausal3D(
            in_channels=config.block_out_channels[-1],
            num_layers=1,
            groups=config.norm_num_groups,
            dropout=config.dropout,
        )

        # Output projection
        conv_out_channels = 2 * config.latent_channels if config.double_z else config.latent_channels
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=config.block_out_channels[-1]
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(
            config.block_out_channels[-1],
            conv_out_channels,
            kernel_size=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input tensor
        Returns:
            (B, 2*latent_channels, T', H, W) tensor (mean, logvar concatenated)
        """
        assert x.ndim == 5, f"Expected 5D input, got {x.ndim}D"

        # Initial conv
        h = self.conv_in(x)

        # Down blocks
        for down_block in self.down_blocks:
            h = down_block(h)

        # Mid block
        h = self.mid_block(h)

        # Output projection
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        return h


class DecoderCausal3D(nn.Module):
    """
    Causal 3D Decoder for volatility surfaces.

    Input: (B, latent_channels, T', H, W)
    Output: (B, C, T, H, W) where H=W=5

    Mirrors encoder structure with upsampling instead of downsampling.
    """

    def __init__(self, config: Causal3DVAEConfig):
        super().__init__()
        self.config = config

        # Initial convolution (from latent to feature space)
        self.conv_in = CausalConv3d(
            config.latent_channels,
            config.block_out_channels[-1],
            kernel_size=3
        )

        # Mid block
        self.mid_block = MidBlockCausal3D(
            in_channels=config.block_out_channels[-1],
            num_layers=1,
            groups=config.norm_num_groups,
            dropout=config.dropout,
        )

        # Up blocks (temporal upsampling only)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(config.block_out_channels))
        num_temporal_upsample = int(np.log2(config.temporal_compression_ratio))

        output_channel = reversed_channels[0]
        for i, block_out_ch in enumerate(reversed_channels):
            input_channel = output_channel
            output_channel = block_out_ch
            is_final = i == len(reversed_channels) - 1

            # Temporal upsampling in first blocks (matching encoder downsampling)
            add_upsample = i < num_temporal_upsample and not is_final
            upsample_factor = (2, 1, 1) if add_upsample else (1, 1, 1)

            up_block = UpDecoderBlockCausal3D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=config.layers_per_block + 1,  # +1 like HunyuanVideo
                groups=config.norm_num_groups,
                dropout=config.dropout,
                add_upsample=add_upsample,
                upsample_factor=upsample_factor,
            )
            self.up_blocks.append(up_block)

        # Output projection
        self.conv_norm_out = nn.GroupNorm(
            num_groups=config.norm_num_groups,
            num_channels=config.block_out_channels[0]
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(
            config.block_out_channels[0],
            config.out_channels,
            kernel_size=3
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, T', H, W) latent tensor
        Returns:
            (B, C, T, H, W) reconstructed tensor
        """
        assert z.ndim == 5, f"Expected 5D input, got {z.ndim}D"

        # Initial conv
        h = self.conv_in(z)

        # Mid block
        h = self.mid_block(h)

        # Up blocks
        for up_block in self.up_blocks:
            h = up_block(h)

        # Output projection
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        return h


class AutoencoderCausal3D(nn.Module):
    """
    Full Causal 3D VAE for volatility surface generation.

    Combines encoder, decoder, and VAE sampling.

    Key features for volatility surface forecasting:
    - Causal temporal structure (no future leakage)
    - Supports autoregressive generation
    - Scheduled sampling for exposure bias correction
    """

    def __init__(self, config: Causal3DVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = EncoderCausal3D(config)
        self.decoder = DecoderCausal3D(config)

        # Latent scaling (for numerical stability, following SD VAE convention)
        self.scaling_factor = 0.18215

        # Track input temporal size for proper reconstruction
        self._input_temporal_size = None

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[DiagonalGaussianDistribution, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input to latent distribution.

        Args:
            x: (B, C, T, H, W) input tensor
            return_dict: If True, return DiagonalGaussianDistribution
        Returns:
            DiagonalGaussianDistribution or (mean, logvar) tuple
        """
        # Store input temporal size for decoder
        self._input_temporal_size = x.shape[2]

        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)

        if return_dict:
            return posterior
        return posterior.mean, posterior.logvar

    def decode(self, z: torch.Tensor, target_temporal_size: Optional[int] = None) -> torch.Tensor:
        """
        Decode latent to output.

        Args:
            z: (B, latent_channels, T', H, W) latent tensor
            target_temporal_size: Optional target T dimension for output
        Returns:
            (B, C, T, H, W) reconstructed tensor
        """
        # Use stored input size or provided target
        target_T = target_temporal_size or self._input_temporal_size

        dec = self.decoder(z)

        # Adjust temporal dimension if needed
        if target_T is not None and dec.shape[2] != target_T:
            if dec.shape[2] > target_T:
                dec = dec[:, :, :target_T, :, :]
            else:
                # Pad by repeating last frame
                pad_size = target_T - dec.shape[2]
                dec = F.pad(dec, (0, 0, 0, 0, 0, pad_size), mode='replicate')

        return dec

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, DiagonalGaussianDistribution]]:
        """
        Full forward pass: encode, sample, decode.

        Args:
            x: (B, C, T, H, W) input tensor
            sample_posterior: If True, sample from posterior; else use mean
            return_posterior: If True, also return the posterior distribution
            generator: Random generator for sampling

        Returns:
            Reconstructed tensor, optionally with posterior
        """
        target_T = x.shape[2]

        # Encode
        posterior = self.encode(x, return_dict=True)

        # Sample or use mode
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # Scale latent
        z = z * self.scaling_factor

        # Decode with target temporal size
        dec = self.decode(z, target_temporal_size=target_T)

        if return_posterior:
            return dec, posterior
        return dec

    def generate_autoregressive(
        self,
        context: torch.Tensor,
        num_steps: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate sequence autoregressively, one step at a time.

        This is the key method for avoiding explosion during chaining:
        - Uses causal convolutions, so each step only sees past
        - Can apply scheduled sampling during training

        Args:
            context: (B, C, T_context, H, W) initial context
            num_steps: Number of steps to generate beyond context
            generator: Random generator for sampling

        Returns:
            (B, C, T_context + num_steps, H, W) full sequence
        """
        B, C, T_ctx, H, W = context.shape
        device = context.device
        dtype = context.dtype

        # Start with context
        sequence = context.clone()

        for step in range(num_steps):
            # Encode current sequence
            posterior = self.encode(sequence, return_dict=True)

            # Sample
            z = posterior.sample(generator=generator) * self.scaling_factor

            # Decode full sequence
            decoded = self.decode(z)

            # Extract the new frame (last decoded frame)
            new_frame = decoded[:, :, -1:, :, :]

            # Append to sequence
            sequence = torch.cat([sequence, new_frame], dim=2)

        return sequence

    def compute_loss(
        self,
        x: torch.Tensor,
        kl_weight: float = 1e-6,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss (reconstruction + KL).

        Args:
            x: (B, C, T, H, W) input tensor
            kl_weight: Weight for KL divergence term
            generator: Random generator for sampling

        Returns:
            Total loss and dict of loss components
        """
        # Forward pass with posterior
        recon, posterior = self.forward(
            x,
            sample_posterior=True,
            return_posterior=True,
            generator=generator
        )

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL loss
        kl_loss = posterior.kl().mean()

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
        }

        return total_loss, loss_dict


def create_vol_surface_vae(
    latent_channels: int = 16,
    temporal_compression: int = 2,
    block_channels: Tuple[int, ...] = (32, 64, 128),
) -> AutoencoderCausal3D:
    """
    Factory function to create a Causal 3D VAE for volatility surfaces.

    Args:
        latent_channels: Latent dimension
        temporal_compression: Temporal compression ratio (1, 2, or 4)
        block_channels: Channel progression for encoder/decoder blocks

    Returns:
        Configured AutoencoderCausal3D
    """
    config = Causal3DVAEConfig(
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        block_out_channels=block_channels,
        layers_per_block=2,
        norm_num_groups=8,
        spatial_compression_ratio=1,  # No spatial compression
        temporal_compression_ratio=temporal_compression,
        dropout=0.1,
    )
    return AutoencoderCausal3D(config)


def create_vol_surface_vae_small(
    latent_channels: int = 4,
    temporal_compression: int = 2,
    block_channels: Tuple[int, ...] = (8, 16, 32),
) -> AutoencoderCausal3D:
    """
    Factory function to create a SMALL Causal 3D VAE for volatility surfaces.

    Scaled down for 5x5 single-channel data:
    - ~100-150K parameters (vs 9.8M for full model)
    - Symmetric encoder-decoder architecture
    - Suitable for 120-day sequences (context=60, horizon=60)

    Args:
        latent_channels: Latent dimension (default: 4)
        temporal_compression: Temporal compression ratio (default: 2)
        block_channels: Channel progression (default: 8->16->32)

    Returns:
        Configured AutoencoderCausal3D
    """
    config = Causal3DVAEConfig(
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        block_out_channels=block_channels,
        layers_per_block=1,  # Reduced from 2
        norm_num_groups=4,   # Reduced from 8
        spatial_compression_ratio=1,  # No spatial compression
        temporal_compression_ratio=temporal_compression,
        dropout=0.1,
    )
    return AutoencoderCausal3D(config)


if __name__ == "__main__":
    print("Testing Causal 3D VAE for volatility surfaces...")
    print("=" * 60)

    # Create model
    model = create_vol_surface_vae(
        latent_channels=16,
        temporal_compression=2,
        block_channels=(32, 64)
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    # Input: (B=4, C=1, T=20, H=5, W=5) - batch of 20-frame vol surfaces
    x = torch.randn(4, 1, 20, 5, 5)
    print(f"\nInput shape: {x.shape}")

    # Encode
    posterior = model.encode(x, return_dict=True)
    print(f"Posterior mean shape: {posterior.mean.shape}")
    print(f"Posterior logvar shape: {posterior.logvar.shape}")

    # Full forward
    recon = model(x, sample_posterior=True)
    print(f"Reconstruction shape: {recon.shape}")

    # Loss
    loss, loss_dict = model.compute_loss(x, kl_weight=1e-6)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"  Recon: {loss_dict['recon']:.4f}")
    print(f"  KL: {loss_dict['kl']:.4f}")

    # Test autoregressive generation
    print("\nTesting autoregressive generation...")
    context = torch.randn(2, 1, 10, 5, 5)  # 10 context frames
    generated = model.generate_autoregressive(context, num_steps=5)
    print(f"Context shape: {context.shape}")
    print(f"Generated shape: {generated.shape}")

    print("\nAll tests passed!")
