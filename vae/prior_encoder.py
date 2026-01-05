"""
Prior Encoder for Ablation Experiment

This module implements Prior Encoders that process raw context independently from
the context encoder, avoiding confounded gradient flow.

Key Differences from Current FullCovariancePrior:
1. Processes RAW context (B, C, 5, 5), not context summary (B, 12)
2. Independent weights from context encoder
3. Receives gradients only from KL loss (clean gradient flow)
4. Architecture mirrors context encoder but scaled down (~50% capacity)

Classes:
- PriorEncoderBase: Base class with Conv2D + LSTM architecture
- PriorEncoderDiagonal: Outputs per-timestep (mu, log_var)
- PriorEncoderFullCov: Outputs mu + AR(1) covariance structure
"""

import torch
import torch.nn as nn
import math
from collections import OrderedDict
from vae.full_covariance_prior import build_ar1_covariance, build_ar1_cholesky_direct


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding for temporal awareness."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, horizon, batch_size, device):
        """
        Generate position encodings for a batch.

        Args:
            horizon: Number of future timesteps
            batch_size: Batch size
            device: Device for tensor

        Returns:
            Position encodings (B, H, d_model)
        """
        pos_enc = self.pe[:horizon, :].unsqueeze(0).expand(batch_size, -1, -1)
        return pos_enc.to(device)


class PriorEncoderBase(nn.Module):
    """
    Base Prior Encoder that processes raw context independently.

    Architecture mirrors CVAECtxMemRandEncoder but scaled down:
    - Conv2D surface embedding (fewer channels)
    - LSTM temporal encoding (smaller hidden size)
    - No weight sharing with context encoder

    Args:
        config: Configuration dictionary with keys:
            - prior_surface_hidden: List of Conv2D channels (default: [16, 32, 64])
            - prior_mem_hidden: LSTM hidden size (default: 64)
            - prior_mem_layers: LSTM num layers (default: 1)
            - prior_dropout: Dropout rate (default: 0.1)
            - latent_dim: Latent dimension (default: 12)
            - feat_dim: Surface grid size (default: [5, 5])
            - padding: Conv2D padding (default: 1)
            - use_dense_surface: Whether to use dense layers (default: False)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.latent_dim = config.get("latent_dim", 12)
        self.max_horizon = config.get("max_horizon", 90)

        # Get prior-specific config (scaled down from context encoder)
        self.prior_surface_hidden = config.get("prior_surface_hidden", [16, 32, 64])
        self.prior_mem_hidden = config.get("prior_mem_hidden", 64)
        self.prior_mem_layers = config.get("prior_mem_layers", 1)
        self.prior_dropout = config.get("prior_dropout", 0.1)

        # Build architecture components
        self.surface_embedding_dim = self._build_surface_embedding(config)
        self.temporal_dim = self._build_temporal_encoder(config)

    def _build_surface_embedding(self, config):
        """Build Conv2D embedding for surfaces (scaled down from context encoder)."""
        feat_dim = config.get("feat_dim", [5, 5])
        use_dense_surface = config.get("use_dense_surface", False)

        surface_embedding = OrderedDict()

        if use_dense_surface:
            # Dense version (not commonly used)
            in_feats = feat_dim[0] * feat_dim[1]
            surface_embedding["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(self.prior_surface_hidden):
                surface_embedding[f"prior_enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                surface_embedding[f"prior_enc_activation_{i}"] = nn.ReLU()
                surface_embedding[f"prior_enc_dropout_{i}"] = nn.Dropout(self.prior_dropout)
                in_feats = out_feats

            self.surface_embedding = nn.Sequential(surface_embedding)
            final_dim = in_feats
        else:
            # Conv2D version (standard)
            padding = config.get("padding", 1)
            in_channels = 1
            for i, out_channels in enumerate(self.prior_surface_hidden):
                surface_embedding[f"prior_enc_conv_{i}"] = nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=1, padding=padding
                )
                surface_embedding[f"prior_enc_activation_{i}"] = nn.ReLU()
                in_channels = out_channels

            surface_embedding["flatten"] = nn.Flatten()
            self.surface_embedding = nn.Sequential(surface_embedding)

            # Calculate flattened dimension
            final_dim = in_channels * feat_dim[0] * feat_dim[1]

        return final_dim

    def _build_temporal_encoder(self, config):
        """Build LSTM for temporal encoding (scaled down from context encoder)."""
        mem_type = config.get("mem_type", "lstm")

        mem_args = {
            "input_size": self.surface_embedding_dim,
            "hidden_size": self.prior_mem_hidden,
            "num_layers": self.prior_mem_layers,
            "batch_first": True,
            "dropout": self.prior_dropout if self.prior_mem_layers > 1 else 0.0,
        }

        if mem_type == "lstm":
            self.temporal_encoder = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.temporal_encoder = nn.GRU(**mem_args)
        else:
            self.temporal_encoder = nn.RNN(**mem_args)

        return self.prior_mem_hidden

    def encode_context(self, context: dict):
        """
        Encode raw context surfaces to temporal representation.

        Args:
            context: Dictionary with "surface" (B, C, H, W) and optionally "ex_feats"

        Returns:
            Temporal embeddings (B, C, temporal_dim)
        """
        surface = context["surface"]  # (B, C, H, W)
        B, C, H, W = surface.shape

        # Embed each timestep's surface independently
        surface_flat = surface.reshape(B * C, 1, H, W)  # (B*C, 1, H, W)
        surface_emb = self.surface_embedding(surface_flat)  # (B*C, embedding_dim)
        surface_emb = surface_emb.reshape(B, C, -1)  # (B, C, embedding_dim)

        # Temporal encoding with LSTM
        temporal_emb, _ = self.temporal_encoder(surface_emb)  # (B, C, temporal_dim)

        return temporal_emb


class PriorEncoderDiagonal(PriorEncoderBase):
    """
    Prior Encoder with diagonal (per-timestep) output.

    Output Format:
        - mu_p: (B, H, latent_dim) - Different mean per future timestep
        - log_var_p: (B, H, latent_dim) - Different log-variance per timestep

    Architecture:
        raw_context -> surface_embed -> LSTM -> position_encoding -> mu, log_var

    This variant models each future timestep independently with diagonal covariance,
    similar to the main encoder's posterior distribution.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Position encoding for horizon-aware prior
        pos_dim = config.get("prior_pos_dim", 32)
        self.pos_encoder = SinusoidalPositionEncoding(d_model=pos_dim, max_len=self.max_horizon)

        # Use last LSTM hidden state as context summary
        combined_dim = self.temporal_dim + pos_dim

        # Output MLPs for mu and log_var
        self.mu_network = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.prior_dropout),
            nn.Linear(64, self.latent_dim)
        )

        self.logvar_network = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.prior_dropout),
            nn.Linear(64, self.latent_dim)
        )

    def forward(self, context: dict, horizon: int):
        """
        Compute prior distribution parameters.

        Args:
            context: Dictionary with "surface" (B, C, H, W)
            horizon: Number of future timesteps

        Returns:
            mu_p: (B, H, latent_dim) - Prior means
            log_var_p: (B, H, latent_dim) - Prior log-variances
        """
        # Encode context
        temporal_emb = self.encode_context(context)  # (B, C, temporal_dim)
        B = temporal_emb.shape[0]
        device = temporal_emb.device

        # Use last timestep as context summary
        context_summary = temporal_emb[:, -1, :]  # (B, temporal_dim)

        # Generate position encodings
        pos_enc = self.pos_encoder(horizon, B, device)  # (B, H, pos_dim)

        # Expand context summary for each future timestep
        context_expanded = context_summary.unsqueeze(1).expand(B, horizon, -1)  # (B, H, temporal_dim)

        # Concatenate context + position
        combined = torch.cat([context_expanded, pos_enc], dim=-1)  # (B, H, combined_dim)

        # Compute mu and log_var for each timestep
        mu_p = self.mu_network(combined)  # (B, H, latent_dim)
        log_var_p_raw = self.logvar_network(combined)  # (B, H, latent_dim)

        # Apply variance floor to prevent collapse
        min_var = 0.01
        var_p = torch.exp(log_var_p_raw).clamp(min=min_var)
        log_var_p = torch.log(var_p)

        return mu_p, log_var_p

    def sample(self, context: dict, horizon: int, num_samples: int = 1):
        """
        Sample from prior distribution using reparameterization trick.

        Args:
            context: Dictionary with "surface" (B, C, H, W)
            horizon: Number of future timesteps
            num_samples: Number of samples per context

        Returns:
            z: (B, H, latent_dim) if num_samples=1
               (B, num_samples, H, latent_dim) if num_samples>1
        """
        mu_p, log_var_p = self.forward(context, horizon)
        B = mu_p.shape[0]
        device = mu_p.device
        dtype = mu_p.dtype

        if num_samples == 1:
            # Single sample
            eps = torch.randn(B, horizon, self.latent_dim, device=device, dtype=dtype)
            z = mu_p + torch.exp(0.5 * log_var_p) * eps
            return z
        else:
            # Multiple samples
            eps = torch.randn(B, num_samples, horizon, self.latent_dim, device=device, dtype=dtype)
            mu_expanded = mu_p.unsqueeze(1)  # (B, 1, H, D)
            std_expanded = torch.exp(0.5 * log_var_p).unsqueeze(1)  # (B, 1, H, D)
            z = mu_expanded + std_expanded * eps
            return z


class PriorEncoderFullCov(PriorEncoderBase):
    """
    Prior Encoder with full covariance (AR(1)) output.

    Output Format:
        - mu_p: (B, H, latent_dim) - Different mean per future timestep
        - Sigma_p: (H, H) - Shared AR(1) covariance across batch

    Architecture:
        raw_context -> surface_embed -> LSTM -> position_encoding -> mu
        + learnable AR(1) parameters (φ, σ²)

    This variant uses AR(1) temporal covariance structure for smoother trajectories,
    similar to the current FullCovariancePrior but with full context processing.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Position encoding for horizon-aware prior
        pos_dim = config.get("prior_pos_dim", 32)
        self.pos_encoder = SinusoidalPositionEncoding(d_model=pos_dim, max_len=self.max_horizon)

        # Use last LSTM hidden state as context summary
        combined_dim = self.temporal_dim + pos_dim

        # Output MLP for mu (same as diagonal variant)
        self.mu_network = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.prior_dropout),
            nn.Linear(64, self.latent_dim)
        )

        # Learnable AR(1) covariance parameters (only 2 scalars!)
        init_phi = config.get("full_cov_init_phi", 0.5)
        init_sigma_sq = config.get("full_cov_init_sigma_sq", 1.0)

        self.log_phi = nn.Parameter(torch.tensor(math.log(init_phi / (1 - init_phi))))  # logit(phi)
        self.log_sigma_sq = nn.Parameter(torch.tensor(math.log(init_sigma_sq)))

        # Cache for Cholesky decomposition
        self._cholesky_cache = {}

    def get_phi(self):
        """Get φ parameter (constrained to (0, 1))"""
        return torch.sigmoid(self.log_phi)

    def get_sigma_sq(self):
        """Get σ² parameter (positive, with floor to prevent collapse)"""
        min_sigma_sq = 0.01  # Prevent variance collapse
        return torch.exp(self.log_sigma_sq).clamp(min=min_sigma_sq)

    def get_cholesky(self, horizon, device, dtype=None, use_cache=True):
        """
        Get Cholesky factor L with optional caching.

        Args:
            horizon: Forecast horizon
            device: Device for tensor
            dtype: Data type (if None, uses default)
            use_cache: If True, use cached L (for inference). If False, recompute (for training)

        Returns:
            L: (H, H) Lower triangular Cholesky factor
        """
        phi = self.get_phi()
        sigma_sq = self.get_sigma_sq()
        sigma = torch.sqrt(sigma_sq)

        if dtype is None:
            dtype = torch.get_default_dtype()

        # During training, always recompute for gradients
        if not use_cache or self.training:
            return build_ar1_cholesky_direct(phi, sigma, horizon, device, dtype=dtype)

        # During inference, use cache
        phi_val = phi.item()
        sigma_sq_val = sigma_sq.item()
        cache_key = (horizon, round(phi_val, 6), round(sigma_sq_val, 6), dtype)

        if cache_key not in self._cholesky_cache:
            L = build_ar1_cholesky_direct(phi_val, math.sqrt(sigma_sq_val), horizon, device, dtype=dtype)
            self._cholesky_cache[cache_key] = L

        return self._cholesky_cache[cache_key]

    def forward(self, context: dict, horizon: int):
        """
        Compute prior distribution parameters.

        Args:
            context: Dictionary with "surface" (B, C, H, W)
            horizon: Number of future timesteps

        Returns:
            mu_p: (B, H, latent_dim) - Prior means
            Sigma_p: (H, H) - AR(1) covariance matrix
        """
        # Encode context
        temporal_emb = self.encode_context(context)  # (B, C, temporal_dim)
        B = temporal_emb.shape[0]
        device = temporal_emb.device
        dtype = temporal_emb.dtype

        # Use last timestep as context summary
        context_summary = temporal_emb[:, -1, :]  # (B, temporal_dim)

        # Generate position encodings
        pos_enc = self.pos_encoder(horizon, B, device)  # (B, H, pos_dim)

        # Expand context summary for each future timestep
        context_expanded = context_summary.unsqueeze(1).expand(B, horizon, -1)  # (B, H, temporal_dim)

        # Concatenate context + position
        combined = torch.cat([context_expanded, pos_enc], dim=-1)  # (B, H, combined_dim)

        # Compute mu for each timestep
        mu_p = self.mu_network(combined)  # (B, H, latent_dim)

        # Build AR(1) covariance matrix
        phi = self.get_phi()
        sigma_sq = self.get_sigma_sq()
        Sigma_p = build_ar1_covariance(phi, sigma_sq, horizon, device, dtype=dtype)

        return mu_p, Sigma_p

    def sample(self, context: dict, horizon: int, num_samples: int = 1):
        """
        Sample from prior distribution using Cholesky decomposition.

        z = μ + L @ ε, where ε ~ N(0, I)

        Args:
            context: Dictionary with "surface" (B, C, H, W)
            horizon: Number of future timesteps
            num_samples: Number of samples per context

        Returns:
            z: (B, H, latent_dim) if num_samples=1
               (B, num_samples, H, latent_dim) if num_samples>1
        """
        mu_p, _ = self.forward(context, horizon)
        B = mu_p.shape[0]
        device = mu_p.device
        dtype = mu_p.dtype

        # Get Cholesky factor
        L = self.get_cholesky(horizon, device, dtype=dtype)  # (H, H)

        if num_samples == 1:
            # Sample ε ~ N(0, I)
            eps = torch.randn(B, horizon, self.latent_dim, device=device, dtype=dtype)

            # Correlated samples: z = μ + L @ ε
            z_centered = torch.einsum('hk,bkd->bhd', L, eps)
            z = mu_p + z_centered

            return z
        else:
            # Multiple samples
            eps = torch.randn(B, num_samples, horizon, self.latent_dim, device=device, dtype=dtype)

            # Correlated samples: z = μ + L @ ε
            z_centered = torch.einsum('hk,bnkd->bnhd', L, eps)
            mu_expanded = mu_p.unsqueeze(1)  # (B, 1, H, D)
            z = mu_expanded + z_centered

            return z
