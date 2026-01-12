"""
Two-Stage CVAE with Tiny Context Bottleneck

Goal: Fix variance collapse by:
1. Tiny context bottleneck (2-3 dims) - forces coarse encoding like k-means
2. True autoencoder training - decoder sees actual ctx_emb for ALL positions
3. Two-stage training support - Stage 1 autoencoder, Stage 2 predictors

Key changes from CVAEMemRand:
- ctx_embedding_dim: Separate parameter (tiny, like 3)
- ctx_encoder: Uses separate config params (ctx_surface_hidden, ctx_mem_hidden, etc.)
- No zero-padding: ctx_emb computed for ALL positions
- Full sequence loss: MSE on all positions, not just horizon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict

from vae.base import BaseVAE, BaseDecoder, BaseEncoder


class TwoStageCtxEncoder(BaseEncoder):
    """
    Tiny context encoder that produces coarse embeddings.

    Key features:
    - Small hidden layers (forces coarse representation)
    - Output dim = ctx_embedding_dim (tiny, like 3)
    - Processes FULL sequence (not just context)
    """

    def __init__(self, config: dict):
        super(TwoStageCtxEncoder, self).__init__(config)

        # Get context-specific parameters (or fall back to defaults)
        ctx_surface_hidden = config.get("ctx_surface_hidden", [16, 32])
        ctx_mem_type = config.get("ctx_mem_type", config.get("mem_type", "lstm"))
        ctx_mem_hidden = config.get("ctx_mem_hidden", 32)
        ctx_mem_layers = config.get("ctx_mem_layers", 1)
        ctx_mem_dropout = config.get("ctx_mem_dropout", 0.1)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)

        # Build surface embedding
        surface_dim = self._build_surface_embedding(config, ctx_surface_hidden)

        # Build extra features embedding (if needed)
        if config.get("ex_feats_dim", 0) > 0:
            ex_feats_dim = self._build_ex_feats_embedding(config)
        else:
            ex_feats_dim = 0

        # Build LSTM memory
        input_dim = surface_dim + ex_feats_dim
        self._build_memory(ctx_mem_type, input_dim, ctx_mem_hidden,
                          ctx_mem_layers, ctx_mem_dropout)

        # Final compression to tiny bottleneck
        self.compress = nn.Linear(ctx_mem_hidden, ctx_embedding_dim)

    def _build_surface_embedding(self, config, hidden_layers):
        """Build surface embedding layers."""
        feat_dim = config["feat_dim"]
        use_dense = config.get("use_dense_surface", True)

        layers = OrderedDict()
        if use_dense:
            in_feats = feat_dim[0] * feat_dim[1]
            layers["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(hidden_layers):
                layers[f"ctx_enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                layers[f"ctx_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            self.surface_embedding = nn.Sequential(layers)
            return in_feats
        else:
            padding = config.get("padding", 1)
            in_feats = 1
            for i, out_feats in enumerate(hidden_layers):
                layers[f"ctx_enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats, kernel_size=3, stride=1, padding=padding
                )
                layers[f"ctx_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            layers["flatten"] = nn.Flatten()
            self.surface_embedding = nn.Sequential(layers)
            return in_feats * feat_dim[0] * feat_dim[1]

    def _build_ex_feats_embedding(self, config):
        """Build extra features embedding."""
        ex_feats_dim = config["ex_feats_dim"]
        ctx_ex_feats_hidden = config.get("ctx_ex_feats_hidden")

        if ctx_ex_feats_hidden is None:
            self.ex_feats_embedding = nn.Identity()
            return ex_feats_dim

        layers = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ctx_ex_feats_hidden):
            layers[f"ctx_ex_enc_{i}"] = nn.Linear(in_feats, out_feats)
            layers[f"ctx_ex_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ex_feats_embedding = nn.Sequential(layers)
        return in_feats

    def _build_memory(self, mem_type, input_size, hidden_size, num_layers, dropout):
        """Build LSTM/GRU/RNN memory."""
        mem_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout if num_layers > 1 else 0,
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)

    def forward(self, x):
        """
        Process full sequence and output CAUSAL ctx_embedding for ALL positions.

        CAUSAL: ctx_emb_t depends on x_{1:t-1} only, NOT x_t.
        This prevents information leakage from target to context embedding.

        Input:
            x: dict with "surface" (B, T, H, W) and optional "ex_feats" (B, T, D)

        Output:
            ctx_embedding: (B, T, ctx_embedding_dim)
                - ctx_emb[:, 0, :] = zeros (no prior context)
                - ctx_emb[:, t, :] = f(x_{1:t-1}) for t > 0
        """
        surface = x["surface"]  # (B, T, H, W)
        B, T = surface.shape[:2]

        # Embed each surface independently
        surface_flat = surface.reshape(B * T, 1, surface.shape[2], surface.shape[3])
        surface_emb = self.surface_embedding(surface_flat)  # (B*T, dim)
        surface_emb = surface_emb.reshape(B, T, -1)  # (B, T, dim)

        # Handle extra features
        if "ex_feats" in x:
            ex_feats = x["ex_feats"]  # (B, T, D)
            ex_flat = ex_feats.reshape(B * T, -1)
            ex_emb = self.ex_feats_embedding(ex_flat)
            ex_emb = ex_emb.reshape(B, T, -1)
            embeddings = torch.cat([surface_emb, ex_emb], dim=-1)
        else:
            embeddings = surface_emb

        # Process through memory
        # mem_out[:, t, :] = f(x_{0:t}) - includes x_t
        mem_out, _ = self.mem(embeddings)  # (B, T, hidden)

        # CAUSAL SHIFT: ctx_emb_t should be f(x_{0:t-1}), NOT f(x_{0:t})
        # Shift right by 1: use mem_out[:, t-1] for ctx_emb[:, t]
        ctx_embedding_dim = self.compress.out_features
        ctx_emb = torch.zeros(B, T, ctx_embedding_dim, device=surface.device)

        # Position 0: no context (zeros) - already initialized
        # Position t>0: use compressed LSTM output from position t-1
        if T > 1:
            ctx_emb[:, 1:, :] = self.compress(mem_out[:, :-1, :])

        return ctx_emb  # ctx_emb_t = f(x_{0:t-1}) ✓ NO x_t!


class TwoStageMainEncoder(BaseEncoder):
    """
    Main encoder that produces z (latent variables).

    Similar to CVAEMemRandEncoder but uses main encoder config params.
    """

    def __init__(self, config: dict):
        super(TwoStageMainEncoder, self).__init__(config)

        latent_dim = config["latent_dim"]
        surface_hidden = config["surface_hidden"]
        mem_type = config.get("mem_type", "lstm")
        mem_hidden = config["mem_hidden"]
        mem_layers = config["mem_layers"]
        mem_dropout = config.get("mem_dropout", 0.1)

        # Build surface embedding
        surface_dim = self._build_surface_embedding(config, surface_hidden)

        # Build extra features embedding (if needed)
        if config.get("ex_feats_dim", 0) > 0:
            ex_feats_dim = self._build_ex_feats_embedding(config)
        else:
            ex_feats_dim = 0

        # Build LSTM memory
        input_dim = surface_dim + ex_feats_dim
        self._build_memory(mem_type, input_dim, mem_hidden, mem_layers, mem_dropout)

        # Latent distribution parameters
        self.z_mean = nn.Linear(mem_hidden, latent_dim)
        self.z_logvar = nn.Linear(mem_hidden, latent_dim)

        # z_logvar floor to prevent variance collapse
        self.z_logvar_floor = config.get("z_logvar_floor", -4.0)

    def _build_surface_embedding(self, config, hidden_layers):
        """Build surface embedding layers."""
        feat_dim = config["feat_dim"]
        use_dense = config.get("use_dense_surface", True)

        layers = OrderedDict()
        if use_dense:
            in_feats = feat_dim[0] * feat_dim[1]
            layers["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(hidden_layers):
                layers[f"enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                layers[f"enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            self.surface_embedding = nn.Sequential(layers)
            return in_feats
        else:
            padding = config.get("padding", 1)
            in_feats = 1
            for i, out_feats in enumerate(hidden_layers):
                layers[f"enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats, kernel_size=3, stride=1, padding=padding
                )
                layers[f"enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            layers["flatten"] = nn.Flatten()
            self.surface_embedding = nn.Sequential(layers)
            return in_feats * feat_dim[0] * feat_dim[1]

    def _build_ex_feats_embedding(self, config):
        """Build extra features embedding."""
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_hidden = config.get("ex_feats_hidden")

        if ex_feats_hidden is None:
            self.ex_feats_embedding = nn.Identity()
            return ex_feats_dim

        layers = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ex_feats_hidden):
            layers[f"ex_enc_{i}"] = nn.Linear(in_feats, out_feats)
            layers[f"ex_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ex_feats_embedding = nn.Sequential(layers)
        return in_feats

    def _build_memory(self, mem_type, input_size, hidden_size, num_layers, dropout):
        """Build LSTM/GRU/RNN memory."""
        mem_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout if num_layers > 1 else 0,
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)

    def forward(self, x):
        """
        Encode full sequence to latent distribution.

        Input:
            x: dict with "surface" (B, T, H, W) and optional "ex_feats" (B, T, D)

        Output:
            z_mean: (B, T, latent_dim)
            z_logvar: (B, T, latent_dim)
            z: (B, T, latent_dim) - sampled via reparameterization
        """
        surface = x["surface"]  # (B, T, H, W)
        B, T = surface.shape[:2]
        latent_dim = self.config["latent_dim"]

        # Embed each surface independently
        surface_flat = surface.reshape(B * T, 1, surface.shape[2], surface.shape[3])
        surface_emb = self.surface_embedding(surface_flat)
        surface_emb = surface_emb.reshape(B, T, -1)

        # Handle extra features
        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            ex_flat = ex_feats.reshape(B * T, -1)
            ex_emb = self.ex_feats_embedding(ex_flat)
            ex_emb = ex_emb.reshape(B, T, -1)
            embeddings = torch.cat([surface_emb, ex_emb], dim=-1)
        else:
            embeddings = surface_emb

        # Process through memory
        mem_out, _ = self.mem(embeddings)  # (B, T, hidden)

        # Compute latent distribution
        mem_flat = mem_out.reshape(B * T, -1)
        z_mean = self.z_mean(mem_flat).reshape(B, T, latent_dim)
        z_logvar = self.z_logvar(mem_flat).reshape(B, T, latent_dim)

        # Apply z_logvar floor to prevent variance collapse
        # Without floor: MSE gradient pushes z_logvar → -∞, making z deterministic
        if self.z_logvar_floor is not None:
            z_logvar = torch.clamp(z_logvar, min=self.z_logvar_floor)

        # Reparameterization trick
        eps = torch.randn_like(z_logvar)
        z = z_mean + torch.exp(0.5 * z_logvar) * eps

        return z_mean, z_logvar, z


class TwoStageDecoder(BaseDecoder):
    """
    Decoder that takes [ctx_embedding, z] and reconstructs surfaces.

    Input: (B, T, ctx_embedding_dim + latent_dim)
    Output: (B, T, H, W) reconstructed surfaces
    """

    def __init__(self, config: dict):
        super(TwoStageDecoder, self).__init__(config)

        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        latent_dim = config["latent_dim"]
        surface_hidden = config["surface_hidden"]
        feat_dim = config["feat_dim"]

        # Decoder-specific LSTM parameters (decoupled from encoder)
        mem_type = config.get("mem_type", "lstm")
        decoder_mem_hidden = config.get("decoder_mem_hidden", 8)
        decoder_mem_layers = config.get("decoder_mem_layers", 1)
        decoder_mem_dropout = config.get("decoder_mem_dropout", 0.1)

        # Compute n_surface for decoder output
        if config.get("use_dense_surface", True):
            n_surface = surface_hidden[-1]
        else:
            n_surface = surface_hidden[-1] * feat_dim[0] * feat_dim[1]

        # Handle extra features
        ex_feats_dim = config.get("ex_feats_dim", 0)
        ex_feats_hidden = config.get("ex_feats_hidden")
        if ex_feats_hidden is not None:
            n_info = ex_feats_hidden[-1]
        else:
            n_info = ex_feats_dim

        self.n_surface = n_surface
        self.n_info = n_info
        self.surface_final_hidden = surface_hidden[-1]
        self.decoder_mem_hidden = decoder_mem_hidden

        # Input: [ctx_embedding, z]
        input_dim = ctx_embedding_dim + latent_dim

        # Build LSTM memory (decoupled hidden size)
        self._build_memory(mem_type, input_dim, decoder_mem_hidden,
                          decoder_mem_layers, decoder_mem_dropout)

        # FiLM modulation layers - z modulates LSTM output
        # This gives z a direct path to output (cannot be ignored)
        self.gamma_net = nn.Linear(latent_dim, decoder_mem_hidden)
        self.beta_net = nn.Linear(latent_dim, decoder_mem_hidden)

        # Optional compression layer
        decoder_compress = config.get("decoder_compress", False)
        decoder_compress_dim = config.get("decoder_compress_dim", 4)
        self.use_compress = decoder_compress

        if decoder_compress:
            self.compress = nn.Linear(decoder_mem_hidden, decoder_compress_dim)
            project_dim = decoder_compress_dim
        else:
            project_dim = decoder_mem_hidden

        # Surface decoder (project to surface features)
        self.surface_input = nn.Linear(project_dim, n_surface)
        self._build_surface_decoder(config, surface_hidden, feat_dim)

        # Extra features decoder (if needed)
        if n_info > 0:
            self.ex_feats_input = nn.Linear(project_dim, n_info)
            self._build_ex_feats_decoder(config, ex_feats_dim)

    def _build_memory(self, mem_type, input_size, hidden_size, num_layers, dropout):
        """Build LSTM/GRU/RNN memory."""
        mem_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout if num_layers > 1 else 0,
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)

    def _build_surface_decoder(self, config, surface_hidden, feat_dim):
        """Build surface decoder layers."""
        use_dense = config.get("use_dense_surface", True)

        layers = OrderedDict()
        if use_dense:
            in_feats = surface_hidden[-1]
            for i, out_feats in enumerate(reversed(surface_hidden[:-1])):
                layers[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
                layers[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            # Final layer to surface size
            final_size = feat_dim[0] * feat_dim[1]
            layers["dec_final"] = nn.Linear(in_feats, final_size)
            layers["dec_final_activation"] = nn.ReLU()
            layers["dec_output"] = nn.Linear(final_size, final_size)
        else:
            padding = config.get("padding", 1)
            deconv_output_padding = config.get("deconv_output_padding", 0)
            in_feats = surface_hidden[-1]
            for i, out_feats in enumerate(reversed(surface_hidden[:-1])):
                layers[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                    in_feats, out_feats, kernel_size=3, stride=1,
                    padding=padding, output_padding=deconv_output_padding
                )
                layers[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            layers["dec_final"] = nn.ConvTranspose2d(
                in_feats, in_feats, kernel_size=3, stride=1,
                padding=padding, output_padding=deconv_output_padding
            )
            layers["dec_final_activation"] = nn.ReLU()
            layers["dec_output"] = nn.Conv2d(in_feats, 1, kernel_size=3, padding="same")

        self.surface_decoder = nn.Sequential(layers)

    def _build_ex_feats_decoder(self, config, ex_feats_dim):
        """Build extra features decoder."""
        ex_feats_hidden = config.get("ex_feats_hidden")

        if ex_feats_hidden is None:
            self.ex_feats_decoder = nn.Linear(ex_feats_dim, ex_feats_dim)
            return

        layers = OrderedDict()
        in_feats = ex_feats_hidden[-1]
        for i, out_feats in enumerate(reversed(ex_feats_hidden[:-1])):
            layers[f"ex_dec_{i}"] = nn.Linear(in_feats, out_feats)
            layers[f"ex_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        layers["ex_output"] = nn.Linear(in_feats, ex_feats_dim)
        self.ex_feats_decoder = nn.Sequential(layers)

    def forward(self, ctx_emb, z):
        """
        Decode [ctx_emb, z] to surfaces, with z modulating via FiLM.

        Args:
            ctx_emb: (B, T, ctx_embedding_dim) - context embedding
            z: (B, T, latent_dim) - latent variable

        Returns:
            If ex_feats_dim > 0:
                (decoded_surface, decoded_ex_feats)
            Else:
                decoded_surface
        """
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config.get("ex_feats_dim", 0)
        use_dense = self.config.get("use_dense_surface", True)

        # LSTM processes [ctx_emb, z] - temporal dynamics preserved
        x = torch.cat([ctx_emb, z], dim=-1)  # (B, T, ctx_embedding_dim + latent_dim)
        mem_out, _ = self.mem(x)  # (B, T, decoder_mem_hidden)
        B, T = mem_out.shape[:2]

        # FiLM: z modulates LSTM output (z has direct path, can't be ignored)
        gamma = self.gamma_net(z)  # (B, T, decoder_mem_hidden)
        beta = self.beta_net(z)    # (B, T, decoder_mem_hidden)
        features = gamma * mem_out + beta  # Modulated features

        # Optional compression
        if self.use_compress:
            features = self.compress(features)  # (B, T, decoder_compress_dim)

        # Decode surface
        surface_in = self.surface_input(features)  # (B, T, n_surface)

        if use_dense:
            surface_flat = surface_in.reshape(B * T, self.surface_final_hidden)
            decoded = self.surface_decoder(surface_flat)
            decoded_surface = decoded.reshape(B, T, feat_dim[0], feat_dim[1])
        else:
            surface_flat = surface_in.reshape(
                B * T, self.surface_final_hidden, feat_dim[0], feat_dim[1]
            )
            decoded = self.surface_decoder(surface_flat)  # (B*T, 1, H, W)
            decoded_surface = decoded.reshape(B, T, feat_dim[0], feat_dim[1])

        # Decode extra features (if any)
        if ex_feats_dim > 0:
            ex_in = self.ex_feats_input(features)  # (B, T, n_info)
            ex_flat = ex_in.reshape(B * T, self.n_info)
            decoded_ex = self.ex_feats_decoder(ex_flat)
            decoded_ex = decoded_ex.reshape(B, T, ex_feats_dim)
            return decoded_surface, decoded_ex

        return decoded_surface


class TwoStageHeteroscedasticDecoder(TwoStageDecoder):
    """
    Heteroscedastic decoder that outputs (mean, log_var) for calibrated uncertainty.

    Key differences from TwoStageDecoder:
    - Two decoder heads: mean and log_var
    - Log_var is clamped to prevent numerical issues
    - Enables NLL loss training for proper variance calibration
    """

    def __init__(self, config: dict):
        super(TwoStageHeteroscedasticDecoder, self).__init__(config)

        feat_dim = config["feat_dim"]
        surface_hidden = config["surface_hidden"]

        # Log-variance decoder (mirrors surface_decoder architecture)
        self._build_logvar_decoder(config, surface_hidden, feat_dim)

        # Logvar bounds for numerical stability
        self.logvar_min = config.get("logvar_min", -10.0)  # var >= exp(-10) ~ 0.00005
        self.logvar_max = config.get("logvar_max", 2.0)    # var <= exp(2) ~ 7.4
        self.logvar_init = config.get("logvar_init", -4.0)  # var ~ 0.018 (initial)

        # Initialize logvar bias to reasonable starting value
        self._init_logvar_bias()

    def _build_logvar_decoder(self, config, surface_hidden, feat_dim):
        """Build log-variance decoder (parallel to surface_decoder)."""
        use_dense = config.get("use_dense_surface", True)

        # Get project_dim from parent class
        decoder_compress = config.get("decoder_compress", False)
        decoder_compress_dim = config.get("decoder_compress_dim", 4)
        decoder_mem_hidden = config.get("decoder_mem_hidden", 8)

        if decoder_compress:
            project_dim = decoder_compress_dim
        else:
            project_dim = decoder_mem_hidden

        # Input projection for logvar - must match n_surface from parent
        # For dense: n_surface = surface_hidden[-1]
        # For conv: n_surface = surface_hidden[-1] * feat_dim[0] * feat_dim[1]
        if use_dense:
            n_surface = surface_hidden[-1]
        else:
            n_surface = surface_hidden[-1] * feat_dim[0] * feat_dim[1]

        self.logvar_input = nn.Linear(project_dim, n_surface)

        # Build logvar decoder layers
        logvar_layers = OrderedDict()
        if use_dense:
            in_feats = surface_hidden[-1]
            for i, out_feats in enumerate(reversed(surface_hidden[:-1])):
                logvar_layers[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
                logvar_layers[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            # Final layer to surface size
            final_size = feat_dim[0] * feat_dim[1]
            logvar_layers["dec_final"] = nn.Linear(in_feats, final_size)
            logvar_layers["dec_final_activation"] = nn.ReLU()
            logvar_layers["dec_output"] = nn.Linear(final_size, final_size)
        else:
            padding = config.get("padding", 1)
            deconv_output_padding = config.get("deconv_output_padding", 0)
            in_feats = surface_hidden[-1]
            for i, out_feats in enumerate(reversed(surface_hidden[:-1])):
                logvar_layers[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                    in_feats, out_feats, kernel_size=3, stride=1,
                    padding=padding, output_padding=deconv_output_padding
                )
                logvar_layers[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            logvar_layers["dec_final"] = nn.ConvTranspose2d(
                in_feats, in_feats, kernel_size=3, stride=1,
                padding=padding, output_padding=deconv_output_padding
            )
            logvar_layers["dec_final_activation"] = nn.ReLU()
            logvar_layers["dec_output"] = nn.Conv2d(in_feats, 1, kernel_size=3, padding="same")

        self.logvar_decoder = nn.Sequential(logvar_layers)

    def _init_logvar_bias(self):
        """Initialize log-variance output to reasonable starting value."""
        # Initialize the final layer bias so initial variance ~ exp(logvar_init)
        if hasattr(self.logvar_decoder, 'dec_output'):
            nn.init.constant_(self.logvar_decoder.dec_output.bias, self.logvar_init)

    def forward(self, ctx_emb, z):
        """
        Decode [ctx_emb, z] to (mean, log_var), with z modulating via FiLM.

        Args:
            ctx_emb: (B, T, ctx_embedding_dim) - context embedding
            z: (B, T, latent_dim) - latent variable

        Returns:
            decoded_mean: (B, T, H, W) - mean surface prediction
            decoded_logvar: (B, T, H, W) - log-variance per grid point
            [decoded_ex_feats]: optional if ex_feats_dim > 0
        """
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config.get("ex_feats_dim", 0)
        use_dense = self.config.get("use_dense_surface", True)

        # LSTM processes [ctx_emb, z] - temporal dynamics preserved
        x = torch.cat([ctx_emb, z], dim=-1)  # (B, T, ctx_embedding_dim + latent_dim)
        mem_out, _ = self.mem(x)  # (B, T, decoder_mem_hidden)
        B, T = mem_out.shape[:2]

        # FiLM: z modulates LSTM output
        gamma = self.gamma_net(z)  # (B, T, decoder_mem_hidden)
        beta = self.beta_net(z)    # (B, T, decoder_mem_hidden)
        features = gamma * mem_out + beta  # Modulated features

        # Optional compression
        if self.use_compress:
            features = self.compress(features)  # (B, T, decoder_compress_dim)

        # Decode mean surface (same as parent)
        surface_in = self.surface_input(features)  # (B, T, n_surface)

        if use_dense:
            surface_flat = surface_in.reshape(B * T, self.surface_final_hidden)
            decoded_mean = self.surface_decoder(surface_flat)
            decoded_mean = decoded_mean.reshape(B, T, feat_dim[0], feat_dim[1])
        else:
            surface_flat = surface_in.reshape(
                B * T, self.surface_final_hidden, feat_dim[0], feat_dim[1]
            )
            decoded_mean = self.surface_decoder(surface_flat)  # (B*T, 1, H, W)
            decoded_mean = decoded_mean.reshape(B, T, feat_dim[0], feat_dim[1])

        # Decode log-variance
        logvar_in = self.logvar_input(features)  # (B, T, n_surface)

        if use_dense:
            logvar_flat = logvar_in.reshape(B * T, self.surface_final_hidden)
            decoded_logvar = self.logvar_decoder(logvar_flat)
            decoded_logvar = decoded_logvar.reshape(B, T, feat_dim[0], feat_dim[1])
        else:
            # For conv mode, n_surface = surface_final_hidden * H * W
            logvar_flat = logvar_in.reshape(
                B * T, self.surface_final_hidden, feat_dim[0], feat_dim[1]
            )
            decoded_logvar = self.logvar_decoder(logvar_flat)  # (B*T, 1, H, W)
            decoded_logvar = decoded_logvar.reshape(B, T, feat_dim[0], feat_dim[1])

        # Clamp log-variance for numerical stability
        decoded_logvar = torch.clamp(decoded_logvar, min=self.logvar_min, max=self.logvar_max)

        # Decode extra features (if any)
        if ex_feats_dim > 0:
            ex_in = self.ex_feats_input(features)
            ex_flat = ex_in.reshape(B * T, self.n_info)
            decoded_ex = self.ex_feats_decoder(ex_flat)
            decoded_ex = decoded_ex.reshape(B, T, ex_feats_dim)
            return decoded_mean, decoded_logvar, decoded_ex

        return decoded_mean, decoded_logvar


class TwoStageFullCovarianceDecoder(TwoStageDecoder):
    """
    Full covariance decoder using Cholesky parameterization.

    Instead of predicting independent variance per grid point, this decoder
    predicts a full 25x25 covariance matrix via its Cholesky factor L.

    Outputs:
    - mean: (B, T, H, W) - mean surface prediction
    - L: (B, T, 25, 25) - lower triangular Cholesky factor

    Covariance: Σ = L @ L.T
    Sampling: x = μ + L @ ε, where ε ~ N(0, I)

    This enables correlated sampling across grid points, preserving the
    spatial correlation structure of volatility surfaces.
    """

    def __init__(self, config: dict):
        super(TwoStageFullCovarianceDecoder, self).__init__(config)

        feat_dim = config["feat_dim"]
        self.grid_size = feat_dim[0] * feat_dim[1]  # 25 for 5x5
        self.n_cholesky = self.grid_size * (self.grid_size + 1) // 2  # 325

        # Get projection dimension from parent
        decoder_compress = config.get("decoder_compress", False)
        decoder_compress_dim = config.get("decoder_compress_dim", 4)
        decoder_mem_hidden = config.get("decoder_mem_hidden", 8)

        if decoder_compress:
            project_dim = decoder_compress_dim
        else:
            project_dim = decoder_mem_hidden

        # Cholesky output head
        self.cholesky_head = nn.Linear(project_dim, self.n_cholesky)

        # Numerical stability parameters
        self.diag_floor = config.get("cholesky_diag_floor", 1e-3)
        self.diag_init = config.get("cholesky_diag_init", -2.0)  # softplus(-2) ≈ 0.13

        # Initialize Cholesky head for stability
        self._init_cholesky_head()

        # Cache tril indices for efficiency
        self.register_buffer(
            '_tril_indices_row',
            torch.tril_indices(self.grid_size, self.grid_size)[0]
        )
        self.register_buffer(
            '_tril_indices_col',
            torch.tril_indices(self.grid_size, self.grid_size)[1]
        )
        self.register_buffer(
            '_diag_indices',
            torch.arange(self.grid_size)
        )

    def _init_cholesky_head(self):
        """Initialize Cholesky head for training stability."""
        # Small weights for off-diagonal (start near independent)
        nn.init.normal_(self.cholesky_head.weight, mean=0.0, std=0.01)

        # Bias: diagonal elements get diag_init, off-diagonal get 0
        with torch.no_grad():
            bias = torch.zeros(self.n_cholesky)
            # Diagonal indices in the flattened lower triangular
            diag_positions = []
            idx = 0
            for i in range(self.grid_size):
                diag_positions.append(idx + i)
                idx += i + 1
            for pos in diag_positions:
                bias[pos] = self.diag_init
            self.cholesky_head.bias.copy_(bias)

    def _construct_cholesky(self, chol_params: torch.Tensor) -> torch.Tensor:
        """
        Construct lower triangular L from flat parameters.

        Args:
            chol_params: (B, T, 325) - flat Cholesky parameters

        Returns:
            L: (B, T, 25, 25) - lower triangular with positive diagonal
        """
        B, T, _ = chol_params.shape
        device = chol_params.device
        dtype = chol_params.dtype

        # Initialize L as zeros with same dtype as input
        L = torch.zeros(B, T, self.grid_size, self.grid_size, device=device, dtype=dtype)

        # Fill lower triangular (including diagonal)
        L[:, :, self._tril_indices_row, self._tril_indices_col] = chol_params

        # Ensure positive diagonal via softplus + floor
        # Note: F.softplus always returns float32, so we cast back to original dtype
        diag_vals = L[:, :, self._diag_indices, self._diag_indices]
        diag_positive = (F.softplus(diag_vals) + self.diag_floor).to(dtype)

        # Clone to avoid in-place modification, then update diagonal
        L = L.clone()
        L[:, :, self._diag_indices, self._diag_indices] = diag_positive

        return L

    def forward(self, ctx_emb: torch.Tensor, z: torch.Tensor):
        """
        Decode [ctx_emb, z] to (mean, L), with z modulating via FiLM.

        Args:
            ctx_emb: (B, T, ctx_embedding_dim) - context embedding
            z: (B, T, latent_dim) - latent variable

        Returns:
            decoded_mean: (B, T, H, W) - mean surface prediction
            L: (B, T, 25, 25) - Cholesky factor of covariance
        """
        feat_dim = self.config["feat_dim"]
        use_dense = self.config.get("use_dense_surface", True)

        # LSTM processes [ctx_emb, z]
        x = torch.cat([ctx_emb, z], dim=-1)
        mem_out, _ = self.mem(x)
        B, T = mem_out.shape[:2]

        # FiLM: z modulates LSTM output
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        features = gamma * mem_out + beta

        # Optional compression
        if self.use_compress:
            features = self.compress(features)

        # Decode mean surface (same as parent)
        surface_in = self.surface_input(features)

        if use_dense:
            surface_flat = surface_in.reshape(B * T, self.surface_final_hidden)
            decoded_mean = self.surface_decoder(surface_flat)
            decoded_mean = decoded_mean.reshape(B, T, feat_dim[0], feat_dim[1])
        else:
            surface_flat = surface_in.reshape(
                B * T, self.surface_final_hidden, feat_dim[0], feat_dim[1]
            )
            decoded_mean = self.surface_decoder(surface_flat)
            decoded_mean = decoded_mean.reshape(B, T, feat_dim[0], feat_dim[1])

        # Decode Cholesky factor
        chol_params = self.cholesky_head(features)  # (B, T, 325)
        L = self._construct_cholesky(chol_params)  # (B, T, 25, 25)

        return decoded_mean, L

    def sample(self, mean: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Sample from N(mean, L @ L.T) with correlated noise.

        Args:
            mean: (B, T, H, W) - mean prediction
            L: (B, T, 25, 25) - Cholesky factor

        Returns:
            sample: (B, T, H, W) - correlated sample
        """
        B, T, H, W = mean.shape
        mean_flat = mean.reshape(B, T, -1)  # (B, T, 25)

        # Sample standard normal
        eps = torch.randn_like(mean_flat)  # (B, T, 25)

        # Correlated sample: x = μ + L @ ε
        # Using einsum for batched matrix-vector multiply
        sample_flat = mean_flat + torch.einsum('btij,btj->bti', L, eps)

        return sample_flat.reshape(B, T, H, W)


class TwoStageStudentTDecoder(TwoStageFullCovarianceDecoder):
    """
    Multivariate Student-t decoder for fat-tailed distributions.

    Extends TwoStageFullCovarianceDecoder by adding:
    - Learnable degrees of freedom (nu) parameter - per grid point (25 params)
    - Student-t sampling via Gamma scale mixture
    - Student-t NLL loss

    This addresses:
    1. Fat tails missing (kurtosis 1.4 vs GT 21.3)
    2. Better correlation learning via higher NLL weight
    3. Heterogeneous kurtosis across grid (GT varies 3-180)

    Sampling:
        eps ~ N(0, I)
        u_i ~ Gamma(nu_i/2, nu_i/2) for each grid point i
        x = mu + L @ eps / sqrt(u)

    The 1/sqrt(u) scaling produces occasional large deviations when u is small,
    naturally creating fat tails while preserving the correlation structure from L.
    Per-grid-point nu allows different tail heaviness across the volatility surface.
    """

    def __init__(self, config: dict):
        super(TwoStageStudentTDecoder, self).__init__(config)

        # Student-t degrees of freedom parameters
        self.nu_floor = config.get("nu_floor", 2.1)  # nu > 2 for finite variance
        self.nu_max = config.get("nu_max", 100.0)    # Prevent collapse to Gaussian
        nu_init = config.get("nu_init", 5.0)         # Initial df, kurtosis ~ 9

        # Option to fix nu from GT kurtosis (literature-recommended approach)
        # Learning nu via gradient descent is known to be difficult
        self.learn_nu = config.get("learn_nu", True)

        # Per-grid-point nu: 25 parameters (one per grid point)
        # Parameterize nu via softplus: nu = softplus(nu_raw) + nu_floor
        # Inverse softplus to get nu_raw from nu_init
        nu_raw_init = np.log(np.exp(nu_init - self.nu_floor) - 1)

        if self.learn_nu:
            # Learnable nu (may not differentiate - gradient issues)
            self.nu_raw = nn.Parameter(torch.full((25,), nu_raw_init, dtype=torch.float32))
        else:
            # Fixed nu - use register_buffer so it's not a Parameter
            # Will be set later via set_nu_from_kurtosis()
            self.register_buffer('nu_raw', torch.full((25,), nu_raw_init, dtype=torch.float32))

    def set_nu_from_kurtosis(self, gt_excess_kurtosis: np.ndarray):
        """
        Fix nu values from ground truth excess kurtosis via method of moments.

        For Student-t with nu > 4: excess_kurtosis = 6 / (nu - 4)
        Inverting: nu = 4 + 6 / excess_kurtosis

        Literature shows learning nu via gradient descent is fundamentally difficult
        (multiple local maxima, weak gradients). This method sets nu directly from data.

        Args:
            gt_excess_kurtosis: (25,) array of excess kurtosis per grid point
        """
        # Method of moments: nu = 4 + 6 / excess_kurtosis
        gt_kurt_clipped = np.clip(gt_excess_kurtosis, 0.1, 1000)  # Avoid div-by-zero
        nu_from_kurt = 4.0 + 6.0 / gt_kurt_clipped

        # Clamp to valid range
        nu_from_kurt = np.clip(nu_from_kurt, self.nu_floor + 0.01, self.nu_max)

        # Convert to nu_raw (inverse softplus)
        # nu = softplus(nu_raw) + nu_floor
        # nu - nu_floor = softplus(nu_raw)
        # nu_raw = inverse_softplus(nu - nu_floor)
        nu_minus_floor = nu_from_kurt - self.nu_floor
        nu_raw_values = np.log(np.exp(nu_minus_floor) - 1)

        # Handle numerical issues for small values
        nu_raw_values = np.clip(nu_raw_values, -10, 10)

        # Update the buffer (not a parameter when learn_nu=False)
        self.nu_raw.data.copy_(torch.tensor(nu_raw_values, dtype=torch.float32))

        # Verify the resulting nu values
        nu_result = F.softplus(self.nu_raw) + self.nu_floor
        nu_result = torch.clamp(nu_result, max=self.nu_max)

        print(f"Fixed nu from GT kurtosis:")
        print(f"  GT excess kurtosis: min={gt_excess_kurtosis.min():.2f}, "
              f"max={gt_excess_kurtosis.max():.2f}, mean={gt_excess_kurtosis.mean():.2f}")
        print(f"  Resulting nu: min={nu_result.min():.3f}, max={nu_result.max():.3f}, "
              f"mean={nu_result.mean():.3f}, std={nu_result.std():.4f}")
        print(f"  learn_nu={self.learn_nu} (nu {'IS' if self.learn_nu else 'is NOT'} trainable)")

    def forward(self, ctx_emb: torch.Tensor, z: torch.Tensor):
        """
        Decode [ctx_emb, z] to (mean, L, nu).

        Args:
            ctx_emb: (B, T, ctx_embedding_dim) - context embedding
            z: (B, T, latent_dim) - latent variable

        Returns:
            decoded_mean: (B, T, H, W) - mean surface prediction
            L: (B, T, 25, 25) - Cholesky factor of covariance
            nu: (25,) - per-grid-point degrees of freedom
        """
        # Get mean and L from parent
        decoded_mean, L = super().forward(ctx_emb, z)

        # Transform nu: softplus + floor, clamp to max
        # nu is (25,) - one per grid point
        nu = F.softplus(self.nu_raw) + self.nu_floor
        nu = torch.clamp(nu, max=self.nu_max)

        return decoded_mean, L, nu

    def sample(self, mean: torch.Tensor, L: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        Sample from multivariate Student-t distribution with per-grid-point nu.

        Uses Gamma scale mixture representation:
            x = mu + L @ eps / sqrt(u)
        where eps ~ N(0, I) and u_i ~ Gamma(nu_i/2, nu_i/2) for each grid point i

        Args:
            mean: (B, T, H, W) - mean prediction
            L: (B, T, 25, 25) - Cholesky factor
            nu: (25,) - per-grid-point degrees of freedom

        Returns:
            sample: (B, T, H, W) - Student-t sample with fat tails
        """
        B, T, H, W = mean.shape
        device = mean.device
        dtype = mean.dtype
        mean_flat = mean.reshape(B, T, -1)  # (B, T, 25)

        # Sample standard normal
        eps = torch.randn_like(mean_flat)  # (B, T, 25)

        # Sample scaling from Gamma(nu_i/2, nu_i/2) for each grid point
        # This gives u_i with mean=1 and variance=2/nu_i
        # Per-grid-point sampling allows different tail heaviness across the surface
        nu_float = nu.float().to(device)  # (25,)

        # Sample u independently for each of the 25 grid points
        u = torch.zeros(B, T, 25, device=device, dtype=torch.float32)
        for i in range(25):
            gamma_dist = torch.distributions.Gamma(nu_float[i] / 2, nu_float[i] / 2)
            u[:, :, i] = gamma_dist.sample((B, T)).squeeze(-1)

        # Clamp to prevent numerical issues with very small u
        u = torch.clamp(u, min=1e-6)

        # Student-t sample: x = mu + L @ eps / sqrt(u)
        # The 1/sqrt(u) scaling creates fat tails, with different tail heaviness per point
        correlated_noise = torch.einsum('btij,btj->bti', L, eps)
        sample_flat = mean_flat + correlated_noise * torch.rsqrt(u)

        return sample_flat.reshape(B, T, H, W).to(dtype)


class CVAETwoStage(BaseVAE):
    """
    Two-Stage CVAE with Tiny Context Bottleneck.

    Key differences from CVAEMemRand:
    1. ctx_embedding_dim is separate from latent_dim (tiny, like 3)
    2. ctx_encoder uses smaller architecture (forces coarse representation)
    3. No zero-padding: ctx_emb computed for ALL positions
    4. Full sequence loss in train_step_autoencoder

    Stage 1: Train autoencoder (encoder + ctx_encoder + decoder)
    Stage 2: Train predictors (frozen autoencoder) - see predictors.py
    """

    def __init__(self, config: dict):
        super(CVAETwoStage, self).__init__(config)
        self._check_config(config)

        # Build components
        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.encoder = TwoStageMainEncoder(config)
        self.decoder = TwoStageDecoder(config)

        # z dropout to prevent z from becoming a lookup key
        # Forces decoder to be robust to missing z information
        z_dropout_rate = config.get("z_dropout", 0.0)
        self.z_dropout = nn.Dropout(p=z_dropout_rate) if z_dropout_rate > 0 else None

        # Loss function for extra features
        if config.get("ex_feats_loss_type", "l1") == "l2":
            self.ex_feats_loss_fn = nn.MSELoss()
        else:
            self.ex_feats_loss_fn = nn.L1Loss()

        # Move to device
        self.to(self.device)

        # Store horizon for inference
        self.horizon = config.get("horizon", 30)

    def _check_config(self, config: dict):
        """Validate and set default config values."""
        required = ["feat_dim", "latent_dim", "surface_hidden", "mem_hidden", "mem_layers"]
        for key in required:
            if key not in config:
                raise ValueError(f"Config missing required key: {key}")

        # Set defaults
        defaults = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "kl_weight": 0.001,
            "ctx_embedding_dim": 3,
            "ctx_surface_hidden": [16, 32],
            "ctx_mem_hidden": 32,
            "ctx_mem_layers": 1,
            "ctx_mem_dropout": 0.1,
            "ctx_mem_type": "lstm",
            "mem_type": "lstm",
            "mem_dropout": 0.1,
            "use_dense_surface": True,
            "ex_feats_dim": 0,
            "ex_feats_hidden": None,
            "ctx_ex_feats_hidden": None,
            "re_feat_weight": 0.0,
            "ex_loss_on_ret_only": False,
            "ex_feats_loss_type": "l1",
            "horizon": 30,
            "context_len": 60,
        }
        for key, value in defaults.items():
            if key not in config:
                config[key] = value

        # Compute padding for conv layers
        if not config["use_dense_surface"]:
            feat_dim = config["feat_dim"]
            stride = 1
            padding = ((feat_dim[-1] - 1) * stride + 3 - feat_dim[-1]) // 2
            if ((feat_dim[-1] - 1) * stride + 3 - feat_dim[-1]) % 2 == 1:
                padding += 1
                config["deconv_output_padding"] = 1
            else:
                config["deconv_output_padding"] = 0
            config["padding"] = padding

    def forward(self, x: Dict[str, torch.Tensor], return_full_sequence: bool = False):
        """
        Forward pass for autoencoder training.

        Key difference from CVAEMemRand: ctx_emb computed for ALL positions (no zero-padding).

        Args:
            x: dict with "surface" (B, T, H, W) and optional "ex_feats" (B, T, D)
            return_full_sequence: If True, return reconstruction for all T positions.
                                  If False, return only horizon positions (for compatibility).

        Returns:
            If ex_feats:
                (surface_recon, ex_feats_recon, z_mean, z_logvar, z)
            Else:
                (surface_recon, z_mean, z_logvar, z)
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]

        # Build encoder input
        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            encoder_input["ex_feats"] = ex_feats

        # Encode context embedding for ALL positions (no zero-padding!)
        ctx_emb = self.ctx_encoder(encoder_input)  # (B, T, ctx_embedding_dim)

        # Encode latent for all positions
        z_mean, z_logvar, z = self.encoder(encoder_input)  # (B, T, latent_dim)

        # Apply z dropout during training to prevent z from becoming a lookup key
        # This forces decoder to be robust to missing z information
        if self.z_dropout is not None and self.training:
            z = self.z_dropout(z)

        # Decode with full ctx_emb (true autoencoder)
        # FiLM decoder: pass ctx_emb and z separately so z can modulate output

        if "ex_feats" in x:
            decoded_surface, decoded_ex = self.decoder(ctx_emb, z)
            if return_full_sequence:
                return decoded_surface, decoded_ex, z_mean, z_logvar, z
            else:
                C = T - self.horizon
                return decoded_surface[:, C:], decoded_ex[:, C:], z_mean, z_logvar, z
        else:
            decoded_surface = self.decoder(ctx_emb, z)
            if return_full_sequence:
                return decoded_surface, z_mean, z_logvar, z
            else:
                C = T - self.horizon
                return decoded_surface[:, C:], z_mean, z_logvar, z

    def train_step_autoencoder(self, x: Dict[str, torch.Tensor],
                                optimizer: torch.optim.Optimizer,
                                loss_mode: str = None):
        """
        Stage 1 training: True autoencoder with configurable loss.

        Key differences from CVAEMemRand.train_step:
        1. ctx_emb computed for ALL positions (no zero-padding)
        2. Configurable loss mode (full, horizon, weighted)
        3. Weak KL weight for variance preservation
        4. CAUSAL ctx_encoder (ctx_emb_t depends on x_{1:t-1} only)

        Args:
            x: dict with "surface" (B, T, H, W) and optional "ex_feats" (B, T, D)
            optimizer: PyTorch optimizer
            loss_mode: Loss computation mode:
                - "full": MSE on all positions (standard autoencoder)
                - "horizon": MSE only on horizon positions (forecasting focus)
                - "weighted": 0.3 * context + 1.0 * horizon (balanced)

        Returns:
            dict with loss components
        """
        # Use config default if loss_mode not specified
        if loss_mode is None:
            loss_mode = self.config.get("loss_mode", "horizon")

        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            # Forward with full sequence
            if "ex_feats" in x:
                recon_surface, recon_ex, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
                # Compute reconstruction loss based on mode
                if loss_mode == "horizon":
                    re_surface = F.mse_loss(recon_surface[:, C:], surface[:, C:])
                elif loss_mode == "weighted":
                    ctx_loss = F.mse_loss(recon_surface[:, :C], surface[:, :C])
                    hor_loss = F.mse_loss(recon_surface[:, C:], surface[:, C:])
                    re_surface = 0.3 * ctx_loss + 1.0 * hor_loss
                else:  # "full"
                    re_surface = F.mse_loss(recon_surface, surface)
                ex_feats = x["ex_feats"].to(self.device)
                if len(ex_feats.shape) == 2:
                    ex_feats = ex_feats.unsqueeze(0)
                if self.config.get("ex_loss_on_ret_only", False):
                    recon_ex = recon_ex[:, :, :1]
                    ex_feats = ex_feats[:, :, :1]
                re_ex = self.ex_feats_loss_fn(recon_ex, ex_feats)
                recon_loss = re_surface + self.config.get("re_feat_weight", 0) * re_ex
            else:
                recon_surface, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
                # Compute reconstruction loss based on mode
                if loss_mode == "horizon":
                    re_surface = F.mse_loss(recon_surface[:, C:], surface[:, C:])
                elif loss_mode == "weighted":
                    ctx_loss = F.mse_loss(recon_surface[:, :C], surface[:, :C])
                    hor_loss = F.mse_loss(recon_surface[:, C:], surface[:, C:])
                    re_surface = 0.3 * ctx_loss + 1.0 * hor_loss
                else:  # "full"
                    re_surface = F.mse_loss(recon_surface, surface)
                recon_loss = re_surface
                re_ex = torch.zeros(1)

            # KL divergence (weak weight)
            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            # Total loss
            total_loss = recon_loss + self.kl_weight * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex if "ex_feats" in x else torch.zeros(1),
            "kl_loss": kl_loss,
        }

    def train_step(self, x, optimizer):
        """Alias for train_step_autoencoder for compatibility."""
        return self.train_step_autoencoder(x, optimizer)

    def train_step_with_p1_loss(self, x: Dict[str, torch.Tensor],
                                 optimizer: torch.optim.Optimizer,
                                 p1_weight: float = 0.1,
                                 n_z_samples: int = 5):
        """
        Training step with P1 loss to force decoder to respond to z variation.

        P1 loss = -log(coefficient_of_variation) where CV = std/|mean|
        This forces the decoder to produce output variance when z varies,
        without targeting a specific variance level.

        Args:
            x: dict with "surface" (B, T, H, W)
            optimizer: PyTorch optimizer
            p1_weight: Weight for P1 loss term
            n_z_samples: Number of z samples for variance estimation

        Returns:
            dict with loss components including p1_loss
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        optimizer.zero_grad()

        # Get context embedding and z distribution
        ctx_emb = self.ctx_encoder(x)
        z_mean, z_logvar, z = self.encoder(x)

        # Standard reconstruction with one z sample
        recon = self.decoder(ctx_emb, z)

        # MSE loss (horizon only)
        mse_loss = F.mse_loss(recon[:, C:], surface[:, C:])

        # KL divergence
        kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
        kl_loss = kl_loss.sum(dim=-1).mean()

        # P1 loss: sample multiple z, measure output variance
        if p1_weight > 0:
            samples = []
            for _ in range(n_z_samples):
                eps = torch.randn_like(z_logvar)
                z_sample = z_mean + torch.exp(0.5 * z_logvar) * eps
                # Apply z_dropout if present
                if self.z_dropout is not None and self.training:
                    z_sample = self.z_dropout(z_sample)
                decoded = self.decoder(ctx_emb, z_sample)
                samples.append(decoded[:, C:])  # Only horizon positions

            samples = torch.stack(samples)  # (n_samples, B, horizon, 5, 5)
            output_var = samples.var(dim=0)  # (B, horizon, 5, 5)
            output_mean = samples.mean(dim=0)

            # Scale-normalized P1 (coefficient of variation) to prevent shrinking
            output_std = output_var.sqrt()
            cv = output_std / (output_mean.abs() + 1e-6)

            # Clamp cv to prevent numerical instability
            cv_mean = cv.mean().clamp(min=1e-4, max=1e4)

            # P1 loss: -log(cv), clamped to prevent gradient explosion
            p1_loss = -torch.log(cv_mean)
            p1_loss = p1_loss.clamp(min=-10, max=10)  # Prevent extreme gradients
        else:
            p1_loss = torch.tensor(0.0, device=self.device)
            output_var = torch.tensor(0.0)
            cv = torch.tensor(0.0)

        # Total loss
        total_loss = mse_loss + self.kl_weight * kl_loss + p1_weight * p1_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "re_surface": mse_loss,  # Alias for compatibility
            "kl_loss": kl_loss,
            "p1_loss": p1_loss,
            "output_var": output_var.mean().item() if torch.is_tensor(output_var) else 0.0,
            "cv": cv.mean().item() if torch.is_tensor(cv) else 0.0,
            "z_logvar_mean": z_logvar.mean().item(),
        }

    def test_step(self, x: Dict[str, torch.Tensor]):
        """Evaluate model on test data."""
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        with torch.no_grad():
            if "ex_feats" in x:
                recon_surface, recon_ex, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
                re_surface = F.mse_loss(recon_surface, surface)
                ex_feats = x["ex_feats"].to(self.device)
                if len(ex_feats.shape) == 2:
                    ex_feats = ex_feats.unsqueeze(0)
                re_ex = self.ex_feats_loss_fn(recon_ex, ex_feats)
                recon_loss = re_surface + self.config.get("re_feat_weight", 0) * re_ex
            else:
                recon_surface, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
                re_surface = F.mse_loss(recon_surface, surface)
                recon_loss = re_surface
                re_ex = torch.zeros(1)

            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()
            total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex if "ex_feats" in x else torch.zeros(1),
            "kl_loss": kl_loss,
        }

    # ==================== Helper Methods for Stage 2 ====================

    def get_ctx_embedding(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get context embedding for all positions (for Stage 2 training).

        Args:
            x: dict with "surface" (B, T, H, W)

        Returns:
            ctx_emb: (B, T, ctx_embedding_dim)
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            encoder_input["ex_feats"] = ex_feats

        with torch.no_grad():
            ctx_emb = self.ctx_encoder(encoder_input)

        return ctx_emb

    def get_z_mean(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get z_mean for all positions (for Stage 2 training).

        Args:
            x: dict with "surface" (B, T, H, W)

        Returns:
            z_mean: (B, T, latent_dim)
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            encoder_input["ex_feats"] = ex_feats

        with torch.no_grad():
            z_mean, _, _ = self.encoder(encoder_input)

        return z_mean

    def decode_with_predicted(self, ctx_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decode using predicted ctx_emb and z (for Stage 2 inference).

        Args:
            ctx_emb: (B, T, ctx_embedding_dim) - from ctx_encoder or predictor
            z: (B, T, latent_dim) - from encoder or predictor

        Returns:
            decoded_surface: (B, T, H, W)
        """
        # FiLM decoder: pass ctx_emb and z separately
        return self.decoder(ctx_emb, z)

    def get_surface_given_conditions(self, c: Dict[str, torch.Tensor],
                                      z: Optional[torch.Tensor] = None,
                                      horizon: Optional[int] = None,
                                      predictors: Optional[Dict] = None):
        """
        Generate surface given context (for inference).

        Args:
            c: dict with "surface" (B, C, H, W) - context only
            z: Optional pre-sampled latents
            horizon: Number of days to forecast
            predictors: Optional dict with "latent" and "context" predictors for Stage 2

        Returns:
            decoded_surface: (B, horizon, H, W)
        """
        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)

        B, C = ctx_surface.shape[:2]
        if horizon is None:
            horizon = self.horizon
        T = C + horizon

        ctx = {"surface": ctx_surface}
        if "ex_feats" in c:
            ctx_ex = c["ex_feats"].to(self.device)
            if len(ctx_ex.shape) == 2:
                ctx_ex = ctx_ex.unsqueeze(0)
            ctx["ex_feats"] = ctx_ex

        # Get ctx_embedding for context positions
        ctx_emb_context = self.ctx_encoder(ctx)  # (B, C, ctx_dim)

        if predictors is not None:
            # Stage 2: Use predictors for future positions
            ctx_emb_future = predictors["context"](ctx_surface, horizon)  # (B, H, ctx_dim)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            z_mean_pred, z_logvar_pred = predictors["latent"](ctx_surface, horizon)
            z_future = z_mean_pred + torch.exp(0.5 * z_logvar_pred) * torch.randn_like(z_mean_pred)

            # Context z from encoder
            z_mean_ctx, _, _ = self.encoder(ctx)
            z = torch.cat([z_mean_ctx, z_future], dim=1)  # (B, T, latent_dim)
        else:
            # No predictors: use zeros for future ctx_emb, N(0,1) for z
            ctx_embedding_dim = self.config.get("ctx_embedding_dim", 3)
            ctx_emb_future = torch.zeros(B, horizon, ctx_embedding_dim, device=self.device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            if z is None:
                z = torch.randn(B, T, self.config["latent_dim"], device=self.device)

            # Use encoder mean for context positions
            z_mean_ctx, _, _ = self.encoder(ctx)
            z[:, :C, :] = z_mean_ctx

        # Decode (FiLM: pass ctx_emb and z separately)
        decoded = self.decoder(ctx_emb, z)

        if isinstance(decoded, tuple):
            return decoded[0][:, C:], decoded[1][:, C:]
        return decoded[:, C:]


class CVAETwoStageHeteroscedastic(CVAETwoStage):
    """
    Two-Stage CVAE with Heteroscedastic Decoder for calibrated uncertainty.

    Key differences from CVAETwoStage:
    - Uses TwoStageHeteroscedasticDecoder (outputs mean + log_var)
    - Uses combined MSE + NLL loss for variance calibration
    - Supports sampling from decoded distribution
    """

    def __init__(self, config: dict):
        # Set heteroscedastic defaults before parent init
        config.setdefault("heteroscedastic", True)
        config.setdefault("mse_weight", 1.0)
        config.setdefault("nll_weight", 0.1)
        config.setdefault("min_var", 1e-6)
        config.setdefault("target_var", 0.1)  # ~log-return variance
        config.setdefault("var_reg_weight", 0.01)
        config.setdefault("decoder_mem_hidden", 32)  # Increase from 8 to reduce bottleneck

        super(CVAETwoStageHeteroscedastic, self).__init__(config)

        # Replace decoder with heteroscedastic version
        self.decoder = TwoStageHeteroscedasticDecoder(config)
        self.decoder.to(self.device)

        # Store heteroscedastic config
        self.mse_weight = config["mse_weight"]
        self.nll_weight = config["nll_weight"]
        self.min_var = config["min_var"]
        self.target_var = config["target_var"]
        self.var_reg_weight = config["var_reg_weight"]

    def gaussian_nll_loss(self, mean, log_var, target):
        """
        Gaussian Negative Log-Likelihood loss.

        NLL = 0.5 * (log(var) + (x - mu)^2 / var)
            = 0.5 * (log_var + (x - mu)^2 / exp(log_var))

        Args:
            mean: (B, T, H, W) - predicted mean
            log_var: (B, T, H, W) - predicted log-variance
            target: (B, T, H, W) - ground truth

        Returns:
            scalar NLL loss
        """
        # Apply minimum variance floor
        log_var = torch.clamp(log_var, min=np.log(self.min_var))
        variance = torch.exp(log_var)
        nll = 0.5 * (log_var + (target - mean) ** 2 / variance)
        return nll.mean()

    def variance_regularization_loss(self, log_var):
        """
        Regularization to prevent variance collapse or explosion.

        Penalizes deviation from target average variance.
        """
        pred_var = torch.exp(log_var).mean()
        # L2 penalty on log-scale (more stable)
        log_pred_var = torch.log(pred_var + 1e-8)
        log_target_var = np.log(self.target_var)
        return (log_pred_var - log_target_var) ** 2

    def forward(self, x: Dict[str, torch.Tensor], return_full_sequence: bool = False):
        """
        Forward pass with heteroscedastic decoder.

        Returns:
            (surface_mean, surface_logvar, z_mean, z_logvar, z)
            or with ex_feats:
            (surface_mean, surface_logvar, ex_feats_recon, z_mean, z_logvar, z)
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]

        # Build encoder input
        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            encoder_input["ex_feats"] = ex_feats

        # Encode context embedding for ALL positions
        ctx_emb = self.ctx_encoder(encoder_input)  # (B, T, ctx_embedding_dim)

        # Encode latent for all positions
        z_mean, z_logvar, z = self.encoder(encoder_input)  # (B, T, latent_dim)

        # Decode with heteroscedastic decoder
        if "ex_feats" in x:
            decoded_mean, decoded_logvar, decoded_ex = self.decoder(ctx_emb, z)
            if return_full_sequence:
                return decoded_mean, decoded_logvar, decoded_ex, z_mean, z_logvar, z
            else:
                C = T - self.horizon
                return decoded_mean[:, C:], decoded_logvar[:, C:], decoded_ex[:, C:], z_mean, z_logvar, z
        else:
            decoded_mean, decoded_logvar = self.decoder(ctx_emb, z)
            if return_full_sequence:
                return decoded_mean, decoded_logvar, z_mean, z_logvar, z
            else:
                C = T - self.horizon
                return decoded_mean[:, C:], decoded_logvar[:, C:], z_mean, z_logvar, z

    def train_step_autoencoder(self, x: Dict[str, torch.Tensor],
                                optimizer: torch.optim.Optimizer,
                                loss_mode: str = None):
        """
        Training step with combined MSE + NLL loss.

        Key differences from parent:
        - MSE on mean ensures mean accuracy
        - NLL on (detached mean, logvar) calibrates variance
        - Variance regularization prevents collapse/explosion
        """
        if loss_mode is None:
            loss_mode = self.config.get("loss_mode", "horizon")

        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            # Forward with full sequence
            if "ex_feats" in x:
                recon_mean, recon_logvar, recon_ex, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
            else:
                recon_mean, recon_logvar, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )

            # Get target surface based on loss_mode
            if loss_mode == "horizon":
                target_surface = surface[:, C:]
                pred_mean = recon_mean[:, C:]
                pred_logvar = recon_logvar[:, C:]
            elif loss_mode == "weighted":
                target_surface = surface
                pred_mean = recon_mean
                pred_logvar = recon_logvar
            else:  # "full"
                target_surface = surface
                pred_mean = recon_mean
                pred_logvar = recon_logvar

            # MSE loss for mean accuracy
            mse_loss = F.mse_loss(pred_mean, target_surface)

            # NLL loss for variance calibration
            # CRITICAL: detach mean so NLL only affects logvar head
            nll_loss = self.gaussian_nll_loss(pred_mean.detach(), pred_logvar, target_surface)

            # Variance regularization
            var_reg_loss = self.variance_regularization_loss(pred_logvar)

            # Combined reconstruction loss
            re_surface = self.mse_weight * mse_loss + self.nll_weight * nll_loss

            # Handle extra features
            if "ex_feats" in x:
                ex_feats = x["ex_feats"].to(self.device)
                if len(ex_feats.shape) == 2:
                    ex_feats = ex_feats.unsqueeze(0)
                if self.config.get("ex_loss_on_ret_only", False):
                    recon_ex = recon_ex[:, :, :1]
                    ex_feats = ex_feats[:, :, :1]
                re_ex = self.ex_feats_loss_fn(recon_ex, ex_feats)
                recon_loss = re_surface + self.config.get("re_feat_weight", 0) * re_ex
            else:
                recon_loss = re_surface
                re_ex = torch.zeros(1, device=self.device)

            # KL divergence
            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            # Total loss
            total_loss = recon_loss + self.kl_weight * kl_loss + self.var_reg_weight * var_reg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        # Monitor predicted variance
        with torch.no_grad():
            pred_var = torch.exp(pred_logvar).mean()
            pred_std = torch.sqrt(pred_var)

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "var_reg_loss": var_reg_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex if "ex_feats" in x else torch.zeros(1),
            "kl_loss": kl_loss,
            "pred_var": pred_var,
            "pred_std": pred_std,
        }

    def test_step(self, x: Dict[str, torch.Tensor]):
        """Evaluate model on test data."""
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        with torch.no_grad():
            if "ex_feats" in x:
                recon_mean, recon_logvar, recon_ex, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
                re_ex = self.ex_feats_loss_fn(recon_ex, x["ex_feats"].to(self.device))
            else:
                recon_mean, recon_logvar, z_mean, z_logvar, z = self.forward(
                    x, return_full_sequence=True
                )
                re_ex = torch.zeros(1, device=self.device)

            # Use horizon for test
            target_surface = surface[:, C:]
            pred_mean = recon_mean[:, C:]
            pred_logvar = recon_logvar[:, C:]

            mse_loss = F.mse_loss(pred_mean, target_surface)
            nll_loss = self.gaussian_nll_loss(pred_mean, pred_logvar, target_surface)
            var_reg_loss = self.variance_regularization_loss(pred_logvar)

            re_surface = self.mse_weight * mse_loss + self.nll_weight * nll_loss
            recon_loss = re_surface + self.config.get("re_feat_weight", 0) * re_ex

            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            total_loss = recon_loss + self.kl_weight * kl_loss + self.var_reg_weight * var_reg_loss

            pred_var = torch.exp(pred_logvar).mean()
            pred_std = torch.sqrt(pred_var)

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "var_reg_loss": var_reg_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex if "ex_feats" in x else torch.zeros(1),
            "kl_loss": kl_loss,
            "pred_var": pred_var,
            "pred_std": pred_std,
        }

    def sample_from_decoder(self, mean, logvar):
        """Sample from decoded distribution N(mean, exp(logvar))."""
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    def get_surface_given_conditions(self, c: Dict[str, torch.Tensor],
                                      z: Optional[torch.Tensor] = None,
                                      horizon: Optional[int] = None,
                                      predictors: Optional[Dict] = None,
                                      sample_from_decoder: bool = True):
        """
        Generate surface given context with heteroscedastic uncertainty.

        Args:
            c: Context dict with "surface" (B, C, H, W)
            z: Optional pre-sampled latents
            horizon: Forecast horizon
            predictors: Optional Stage 2 predictors
            sample_from_decoder: If True, sample from N(mean, var). If False, return mean.

        Returns:
            If sample_from_decoder: sampled surface (B, H, 5, 5)
            If not sample_from_decoder: (mean, std) tuple
        """
        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)

        B, C = ctx_surface.shape[:2]
        if horizon is None:
            horizon = self.horizon
        T = C + horizon

        ctx = {"surface": ctx_surface}
        if "ex_feats" in c:
            ctx_ex = c["ex_feats"].to(self.device)
            if len(ctx_ex.shape) == 2:
                ctx_ex = ctx_ex.unsqueeze(0)
            ctx["ex_feats"] = ctx_ex

        # Get ctx_embedding for context positions
        ctx_emb_context = self.ctx_encoder(ctx)  # (B, C, ctx_dim)

        if predictors is not None:
            # Stage 2: Use predictors
            ctx_emb_future = predictors["context"](ctx_surface, horizon)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            z_mean_pred, z_logvar_pred = predictors["latent"](ctx_surface, horizon)
            z_future = z_mean_pred + torch.exp(0.5 * z_logvar_pred) * torch.randn_like(z_mean_pred)

            z_mean_ctx, _, _ = self.encoder(ctx)
            z = torch.cat([z_mean_ctx, z_future], dim=1)
        else:
            ctx_embedding_dim = self.config.get("ctx_embedding_dim", 3)
            ctx_emb_future = torch.zeros(B, horizon, ctx_embedding_dim, device=self.device)
            ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

            if z is None:
                z = torch.randn(B, T, self.config["latent_dim"], device=self.device)

            z_mean_ctx, _, _ = self.encoder(ctx)
            z[:, :C, :] = z_mean_ctx

        # Decode to (mean, logvar)
        decoded = self.decoder(ctx_emb, z)

        if isinstance(decoded, tuple) and len(decoded) >= 2:
            mean = decoded[0][:, C:]
            logvar = decoded[1][:, C:]

            if sample_from_decoder:
                return self.sample_from_decoder(mean, logvar)
            else:
                std = torch.exp(0.5 * logvar)
                return mean, std
        else:
            return decoded[:, C:]


class CVAETwoStageFullCovariance(CVAETwoStage):
    """
    Two-Stage CVAE with Full Covariance Decoder.

    Key differences from CVAETwoStageHeteroscedastic:
    - Decoder outputs Cholesky factor L (25x25) instead of diagonal variance
    - Samples are correlated across grid points via x = μ + L @ ε
    - Uses multivariate Gaussian NLL loss for proper covariance learning

    This addresses Issue #3: Cross-grid correlation destroyed (0.02 vs GT 0.32)
    by enabling the model to learn and sample from correlated distributions.
    """

    def __init__(self, config: dict):
        # Set full covariance defaults before parent init
        config.setdefault("full_covariance", True)
        config.setdefault("cholesky_diag_floor", 1e-3)
        config.setdefault("cholesky_diag_init", -2.0)
        config.setdefault("mse_weight", 1.0)
        config.setdefault("nll_weight", 0.1)
        config.setdefault("decoder_mem_hidden", 32)  # Need more capacity for 325 outputs

        super(CVAETwoStageFullCovariance, self).__init__(config)

        # Replace decoder with full covariance version
        self.decoder = TwoStageFullCovarianceDecoder(config)
        self.decoder.to(self.device)

        # Store config
        self.mse_weight = config["mse_weight"]
        self.nll_weight = config["nll_weight"]

    def multivariate_gaussian_nll(self, mean: torch.Tensor, L: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        """
        Multivariate Gaussian NLL with Cholesky parameterization.

        NLL = 0.5 * (k*log(2π) + log|Σ| + (x-μ)^T Σ^{-1} (x-μ))
            = 0.5 * (k*log(2π) + 2*sum(log(diag(L))) + ||L^{-1}(x-μ)||²)

        Args:
            mean: (B, T, H, W) - predicted mean
            L: (B, T, 25, 25) - Cholesky factor
            target: (B, T, H, W) - ground truth

        Returns:
            scalar NLL loss (averaged over batch and time)
        """
        B, T = mean.shape[:2]
        original_dtype = mean.dtype

        # Flatten spatial dimensions
        mean_flat = mean.reshape(B, T, -1)  # (B, T, 25)
        target_flat = target.reshape(B, T, -1)  # (B, T, 25)

        residual = target_flat - mean_flat  # (B, T, 25)

        # solve_triangular doesn't support bfloat16, so upcast if needed
        if L.dtype == torch.bfloat16:
            L_f32 = L.float()
            residual_f32 = residual.float()
            z = torch.linalg.solve_triangular(
                L_f32, residual_f32.unsqueeze(-1), upper=False
            ).squeeze(-1)
            z = z.to(original_dtype)
            L_diag = L_f32.diagonal(dim1=-2, dim2=-1)
        else:
            z = torch.linalg.solve_triangular(
                L, residual.unsqueeze(-1), upper=False
            ).squeeze(-1)
            L_diag = L.diagonal(dim1=-2, dim2=-1)

        # Mahalanobis term: ||z||² = ||L^{-1}(x-μ)||²
        mahal = (z ** 2).sum(dim=-1)  # (B, T)

        # Log determinant: log|Σ| = 2 * sum(log(diag(L)))
        log_det = 2 * torch.log(L_diag).sum(dim=-1)  # (B, T)

        # NLL (ignoring constant k*log(2π))
        nll = 0.5 * (log_det.to(original_dtype) + mahal)

        return nll.mean()

    def forward(self, x: Dict[str, torch.Tensor], return_full_sequence: bool = False):
        """
        Forward pass with full covariance decoder.

        Returns:
            (surface_mean, L, z_mean, z_logvar, z)
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]

        # Build encoder input
        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            encoder_input["ex_feats"] = ex_feats

        # Encode context embedding for ALL positions
        ctx_emb = self.ctx_encoder(encoder_input)  # (B, T, ctx_embedding_dim)

        # Encode latent for all positions
        z_mean, z_logvar, z = self.encoder(encoder_input)  # (B, T, latent_dim)

        # Decode with full covariance decoder
        decoded_mean, L = self.decoder(ctx_emb, z)

        if return_full_sequence:
            return decoded_mean, L, z_mean, z_logvar, z
        else:
            C = T - self.horizon
            return decoded_mean[:, C:], L[:, C:], z_mean, z_logvar, z

    def train_step_autoencoder(self, x: Dict[str, torch.Tensor],
                                optimizer: torch.optim.Optimizer,
                                loss_mode: str = None):
        """
        Training step with MSE + Multivariate NLL loss.

        MSE ensures mean accuracy, NLL calibrates full covariance.
        """
        if loss_mode is None:
            loss_mode = self.config.get("loss_mode", "horizon")

        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            # Forward with full sequence
            recon_mean, L, z_mean, z_logvar, z = self.forward(
                x, return_full_sequence=True
            )

            # Get target surface based on loss_mode
            if loss_mode == "horizon":
                target_surface = surface[:, C:]
                pred_mean = recon_mean[:, C:]
                pred_L = L[:, C:]
            else:  # "full"
                target_surface = surface
                pred_mean = recon_mean
                pred_L = L

            # MSE loss for mean accuracy
            mse_loss = F.mse_loss(pred_mean, target_surface)

            # Multivariate NLL loss for covariance calibration
            # Detach mean so NLL only affects Cholesky head
            nll_loss = self.multivariate_gaussian_nll(
                pred_mean.detach(), pred_L, target_surface
            )

            # Combined reconstruction loss
            re_surface = self.mse_weight * mse_loss + self.nll_weight * nll_loss

            # KL divergence
            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            # Total loss
            total_loss = re_surface + self.kl_weight * kl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        # Monitor Cholesky statistics
        with torch.no_grad():
            L_diag = torch.diagonal(pred_L, dim1=-2, dim2=-1)
            diag_mean = L_diag.mean()
            diag_std = L_diag.std()

            # Off-diagonal magnitude
            mask = torch.ones_like(pred_L[0, 0], dtype=torch.bool)
            mask.fill_diagonal_(False)
            off_diag_mean = pred_L[:, :, mask].abs().mean()

        return {
            "loss": total_loss,
            "reconstruction_loss": re_surface,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "re_surface": re_surface,
            "kl_loss": kl_loss,
            "L_diag_mean": diag_mean,
            "L_diag_std": diag_std,
            "L_offdiag_mean": off_diag_mean,
        }

    def test_step(self, x: Dict[str, torch.Tensor]):
        """Evaluate model on test data."""
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        with torch.no_grad():
            recon_mean, L, z_mean, z_logvar, z = self.forward(
                x, return_full_sequence=True
            )

            target_surface = surface[:, C:]
            pred_mean = recon_mean[:, C:]
            pred_L = L[:, C:]

            mse_loss = F.mse_loss(pred_mean, target_surface)
            nll_loss = self.multivariate_gaussian_nll(pred_mean, pred_L, target_surface)
            re_surface = self.mse_weight * mse_loss + self.nll_weight * nll_loss

            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            total_loss = re_surface + self.kl_weight * kl_loss

            L_diag = torch.diagonal(pred_L, dim1=-2, dim2=-1)
            diag_mean = L_diag.mean()

        return {
            "loss": total_loss,
            "reconstruction_loss": re_surface,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "re_surface": re_surface,
            "kl_loss": kl_loss,
            "L_diag_mean": diag_mean,
        }

    def sample_from_decoder(self, mean: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Sample from N(mean, L @ L.T) with correlated noise."""
        return self.decoder.sample(mean, L)

    def get_surface_given_conditions(self, c: Dict[str, torch.Tensor],
                                      z: Optional[torch.Tensor] = None,
                                      horizon: Optional[int] = None,
                                      sample_from_decoder: bool = True):
        """
        Generate surface given context with full covariance.

        Args:
            c: Context dict with "surface" (B, C, H, W)
            z: Optional pre-sampled latents
            horizon: Forecast horizon
            sample_from_decoder: If True, sample from N(mean, Σ). If False, return (mean, L).

        Returns:
            If sample_from_decoder: sampled surface (B, H, 5, 5)
            If not: (mean, L) tuple
        """
        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)

        B, C = ctx_surface.shape[:2]
        if horizon is None:
            horizon = self.horizon
        T = C + horizon

        ctx = {"surface": ctx_surface}
        if "ex_feats" in c:
            ctx_ex = c["ex_feats"].to(self.device)
            if len(ctx_ex.shape) == 2:
                ctx_ex = ctx_ex.unsqueeze(0)
            ctx["ex_feats"] = ctx_ex

        # Get ctx_embedding for context positions
        ctx_emb_context = self.ctx_encoder(ctx)  # (B, C, ctx_dim)

        # No predictors: use zeros for future ctx_emb, N(0,1) for z
        ctx_embedding_dim = self.config.get("ctx_embedding_dim", 3)
        ctx_emb_future = torch.zeros(B, horizon, ctx_embedding_dim, device=self.device)
        ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

        if z is None:
            z = torch.randn(B, T, self.config["latent_dim"], device=self.device)

        # Use encoder mean for context positions
        z_mean_ctx, _, _ = self.encoder(ctx)
        z[:, :C, :] = z_mean_ctx

        # Decode to (mean, L)
        mean, L = self.decoder(ctx_emb, z)
        mean = mean[:, C:]
        L = L[:, C:]

        if sample_from_decoder:
            return self.sample_from_decoder(mean, L)
        else:
            return mean, L


class CVAETwoStageStudentT(CVAETwoStage):
    """
    Two-Stage CVAE with Multivariate Student-t Decoder.

    Key differences from CVAETwoStageFullCovariance:
    - Decoder outputs (mean, L, nu) where nu is degrees of freedom
    - Samples have fat tails via Student-t distribution
    - Uses multivariate Student-t NLL loss

    This addresses:
    - Issue #1: Fat tails missing (kurtosis 1.4 vs GT 21.3)
    - Issue #3: Cross-grid correlation (with higher NLL weight)

    Mathematical foundation:
    - Sampling: x = μ + L @ ε / √u where u ~ Gamma(ν/2, ν/2)
    - The 1/√u scaling produces fat tails
    - Cholesky L preserves correlation structure
    """

    def __init__(self, config: dict):
        # Set Student-t defaults before parent init
        config.setdefault("student_t", True)
        config.setdefault("nu_floor", 2.1)      # nu > 2 for finite variance
        config.setdefault("nu_max", 100.0)      # Prevent collapse to Gaussian
        config.setdefault("nu_init", 5.0)       # Initial df, kurtosis ~ 9
        config.setdefault("mse_weight", 1.0)
        config.setdefault("nll_weight", 1.0)    # HIGHER than Full Cov (0.1) for correlation
        config.setdefault("cholesky_diag_floor", 1e-3)
        config.setdefault("cholesky_diag_init", -2.0)
        config.setdefault("decoder_mem_hidden", 32)

        super(CVAETwoStageStudentT, self).__init__(config)

        # Replace decoder with Student-t version
        self.decoder = TwoStageStudentTDecoder(config)
        self.decoder.to(self.device)

        # Store config
        self.mse_weight = config["mse_weight"]
        self.nll_weight = config["nll_weight"]
        self.kurtosis_loss_weight = config.get("kurtosis_loss_weight", 0.0)

        # GT excess kurtosis for theoretical kurtosis loss supervision
        # Set via set_gt_excess_kurtosis() before training
        self.gt_excess_kurtosis = None

    def set_gt_excess_kurtosis(self, gt_excess_kurtosis: np.ndarray):
        """
        Set ground truth excess kurtosis for theoretical kurtosis loss.

        Args:
            gt_excess_kurtosis: (25,) array of excess kurtosis per grid point
        """
        self.gt_excess_kurtosis = torch.tensor(
            gt_excess_kurtosis, dtype=torch.float32, device=self.device
        )
        print(f"Set GT excess kurtosis: mean={self.gt_excess_kurtosis.mean():.2f}, "
              f"range=[{self.gt_excess_kurtosis.min():.2f}, {self.gt_excess_kurtosis.max():.2f}]")

    def fix_nu_from_kurtosis(self, gt_excess_kurtosis: np.ndarray):
        """
        Fix nu values from GT kurtosis using method of moments (not gradient-learned).

        This is the literature-recommended approach since learning nu via gradient
        descent is fundamentally difficult (multiple local maxima, weak gradients).

        Also sets gt_excess_kurtosis for potential kurtosis loss supervision.

        Args:
            gt_excess_kurtosis: (25,) array of excess kurtosis per grid point
        """
        # Also set for kurtosis loss (in case it's used)
        self.set_gt_excess_kurtosis(gt_excess_kurtosis)

        # Fix nu via method of moments on decoder
        self.decoder.set_nu_from_kurtosis(gt_excess_kurtosis)

    def theoretical_kurtosis_loss(self, nu: torch.Tensor) -> torch.Tensor:
        """
        Compute loss to match nu to GT excess kurtosis via theoretical formula.

        For Student-t with nu > 4, excess kurtosis = 6 / (nu - 4).
        This loss directly supervises nu to match GT kurtosis.

        Args:
            nu: (25,) - current degrees of freedom per grid point

        Returns:
            scalar L1 loss between theoretical and GT excess kurtosis
        """
        if self.gt_excess_kurtosis is None:
            return torch.tensor(0.0, device=self.device)

        # Theoretical excess kurtosis: 6 / (nu - 4) for nu > 4
        # Clamp (nu - 4) to avoid division by zero when nu is close to 4
        theoretical_excess_kurt = 6.0 / torch.clamp(nu - 4.0, min=0.1)

        # L1 loss is more robust for extreme GT values (some >100)
        return F.l1_loss(theoretical_excess_kurt, self.gt_excess_kurtosis)

    def multivariate_student_t_nll(self, mean: torch.Tensor, L: torch.Tensor,
                                    target: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        Multivariate Student-t NLL with per-grid-point degrees of freedom.

        Since each grid point has its own nu, we compute the NLL as a sum of
        univariate Student-t NLLs after decorrelating via L^{-1}.

        For univariate Student-t with nu_i degrees of freedom:
            NLL_i = -log Γ((ν_i+1)/2) + log Γ(ν_i/2) + 0.5*log(ν_i*π)
                    + 0.5*(ν_i+1)*log(1 + z_i²/ν_i)

        Total NLL = log|L| + sum_i NLL_i

        Args:
            mean: (B, T, H, W) - predicted mean
            L: (B, T, 25, 25) - Cholesky factor
            target: (B, T, H, W) - ground truth
            nu: (25,) - per-grid-point degrees of freedom

        Returns:
            scalar NLL loss (averaged over batch and time)
        """
        B, T = mean.shape[:2]
        original_dtype = mean.dtype
        device = mean.device

        # Flatten spatial dimensions
        mean_flat = mean.reshape(B, T, -1)  # (B, T, 25)
        target_flat = target.reshape(B, T, -1)  # (B, T, 25)

        residual = target_flat - mean_flat  # (B, T, 25)

        # solve_triangular doesn't support bfloat16, so upcast if needed
        if L.dtype == torch.bfloat16:
            L_f32 = L.float()
            residual_f32 = residual.float()
            z = torch.linalg.solve_triangular(
                L_f32, residual_f32.unsqueeze(-1), upper=False
            ).squeeze(-1)
            L_diag = L_f32.diagonal(dim1=-2, dim2=-1)
        else:
            z = torch.linalg.solve_triangular(
                L, residual.unsqueeze(-1), upper=False
            ).squeeze(-1)
            L_diag = L.diagonal(dim1=-2, dim2=-1)

        # Log determinant: log|L| (from Jacobian of decorrelation)
        log_det = torch.log(L_diag).sum(dim=-1)  # (B, T)

        # Ensure nu is float32 for lgamma and on correct device
        nu_f32 = nu.float().to(device)  # (25,)

        # Per-point univariate Student-t NLL
        # NLL_i = -lgamma((nu_i+1)/2) + lgamma(nu_i/2) + 0.5*log(nu_i*pi)
        #         + 0.5*(nu_i+1)*log(1 + z_i^2/nu_i)
        nll_per_point = torch.zeros(B, T, 25, device=device, dtype=torch.float32)

        for i in range(25):
            nu_i = nu_f32[i]
            z_i = z[:, :, i]  # (B, T)

            # Constant terms
            const_i = (torch.lgamma((nu_i + 1) / 2)
                      - torch.lgamma(nu_i / 2)
                      - 0.5 * torch.log(nu_i * torch.tensor(np.pi, device=device)))

            # Student-t NLL for point i
            nll_per_point[:, :, i] = (-const_i
                                      + 0.5 * (nu_i + 1) * torch.log(1 + z_i ** 2 / nu_i))

        # Sum over grid points and add log determinant
        nll = log_det.to(original_dtype) + nll_per_point.sum(dim=-1)

        return nll.mean()

    def forward(self, x: Dict[str, torch.Tensor], return_full_sequence: bool = False):
        """
        Forward pass with Student-t decoder.

        Returns:
            (surface_mean, L, nu, z_mean, z_logvar, z)
        """
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]

        # Build encoder input
        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            encoder_input["ex_feats"] = ex_feats

        # Encode context embedding for ALL positions
        ctx_emb = self.ctx_encoder(encoder_input)  # (B, T, ctx_embedding_dim)

        # Encode latent for all positions
        z_mean, z_logvar, z = self.encoder(encoder_input)  # (B, T, latent_dim)

        # Decode with Student-t decoder
        decoded_mean, L, nu = self.decoder(ctx_emb, z)

        if return_full_sequence:
            return decoded_mean, L, nu, z_mean, z_logvar, z
        else:
            C = T - self.horizon
            return decoded_mean[:, C:], L[:, C:], nu, z_mean, z_logvar, z

    def train_step_autoencoder(self, x: Dict[str, torch.Tensor],
                                optimizer: torch.optim.Optimizer,
                                loss_mode: str = None):
        """
        Training step with MSE + Multivariate Student-t NLL loss.

        MSE ensures mean accuracy, Student-t NLL calibrates covariance AND tails.
        """
        if loss_mode is None:
            loss_mode = self.config.get("loss_mode", "horizon")

        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            # Forward with full sequence
            recon_mean, L, nu, z_mean, z_logvar, z = self.forward(
                x, return_full_sequence=True
            )

            # Get target surface based on loss_mode
            if loss_mode == "horizon":
                target_surface = surface[:, C:]
                pred_mean = recon_mean[:, C:]
                pred_L = L[:, C:]
            else:  # "full"
                target_surface = surface
                pred_mean = recon_mean
                pred_L = L

            # MSE loss for mean accuracy
            mse_loss = F.mse_loss(pred_mean, target_surface)

            # Multivariate Student-t NLL loss for covariance AND tail calibration
            # Detach mean so NLL only affects Cholesky head and nu
            nll_loss = self.multivariate_student_t_nll(
                pred_mean.detach(), pred_L, target_surface, nu
            )

            # Combined reconstruction loss
            re_surface = self.mse_weight * mse_loss + self.nll_weight * nll_loss

            # Theoretical kurtosis loss - supervises nu via closed-form formula
            # excess_kurt = 6/(nu-4), so we match nu to GT kurtosis directly
            kurt_loss = self.theoretical_kurtosis_loss(nu)

            # KL divergence
            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            # Total loss
            total_loss = (re_surface + self.kl_weight * kl_loss
                          + self.kurtosis_loss_weight * kurt_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        # Monitor Cholesky and nu statistics
        with torch.no_grad():
            L_diag = torch.diagonal(pred_L, dim1=-2, dim2=-1)
            diag_mean = L_diag.mean()
            diag_std = L_diag.std()

            # Off-diagonal magnitude
            mask = torch.ones_like(pred_L[0, 0], dtype=torch.bool)
            mask.fill_diagonal_(False)
            off_diag_mean = pred_L[:, :, mask].abs().mean()

        return {
            "loss": total_loss,
            "reconstruction_loss": re_surface,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "kurt_loss": kurt_loss,
            "re_surface": re_surface,
            "kl_loss": kl_loss,
            # Per-grid-point nu statistics
            "nu_mean": nu.mean().item(),
            "nu_min": nu.min().item(),
            "nu_max": nu.max().item(),
            "nu_std": nu.std().item(),
            "L_diag_mean": diag_mean,
            "L_diag_std": diag_std,
            "L_offdiag_mean": off_diag_mean,
        }

    def test_step(self, x: Dict[str, torch.Tensor]):
        """Evaluate model on test data."""
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = T - self.horizon

        with torch.no_grad():
            recon_mean, L, nu, z_mean, z_logvar, z = self.forward(
                x, return_full_sequence=True
            )

            target_surface = surface[:, C:]
            pred_mean = recon_mean[:, C:]
            pred_L = L[:, C:]

            mse_loss = F.mse_loss(pred_mean, target_surface)
            nll_loss = self.multivariate_student_t_nll(pred_mean, pred_L, target_surface, nu)
            re_surface = self.mse_weight * mse_loss + self.nll_weight * nll_loss

            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - z_mean.pow(2))
            kl_loss = kl_loss.sum(dim=-1).mean()

            total_loss = re_surface + self.kl_weight * kl_loss

            L_diag = torch.diagonal(pred_L, dim1=-2, dim2=-1)
            diag_mean = L_diag.mean()

        return {
            "loss": total_loss,
            "reconstruction_loss": re_surface,
            "mse_loss": mse_loss,
            "nll_loss": nll_loss,
            "re_surface": re_surface,
            "kl_loss": kl_loss,
            # Per-grid-point nu statistics
            "nu_mean": nu.mean().item(),
            "nu_min": nu.min().item(),
            "nu_max": nu.max().item(),
            "nu_std": nu.std().item(),
            "L_diag_mean": diag_mean,
        }

    def sample_from_decoder(self, mean: torch.Tensor, L: torch.Tensor,
                             nu: torch.Tensor = None) -> torch.Tensor:
        """Sample from Student-t(mean, L @ L.T, nu) with fat tails."""
        if nu is None:
            # Get nu from decoder parameter
            nu = F.softplus(self.decoder.nu_raw) + self.decoder.nu_floor
            nu = torch.clamp(nu, max=self.decoder.nu_max)
        return self.decoder.sample(mean, L, nu)

    def get_surface_given_conditions(self, c: Dict[str, torch.Tensor],
                                      z: Optional[torch.Tensor] = None,
                                      horizon: Optional[int] = None,
                                      sample_from_decoder: bool = True):
        """
        Generate surface given context with Student-t decoder.

        Args:
            c: Context dict with "surface" (B, C, H, W)
            z: Optional pre-sampled latents
            horizon: Forecast horizon
            sample_from_decoder: If True, sample from Student-t. If False, return (mean, L, nu).

        Returns:
            If sample_from_decoder: sampled surface (B, H, 5, 5)
            If not: (mean, L, nu) tuple
        """
        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)

        B, C = ctx_surface.shape[:2]
        if horizon is None:
            horizon = self.horizon
        T = C + horizon

        ctx = {"surface": ctx_surface}
        if "ex_feats" in c:
            ctx_ex = c["ex_feats"].to(self.device)
            if len(ctx_ex.shape) == 2:
                ctx_ex = ctx_ex.unsqueeze(0)
            ctx["ex_feats"] = ctx_ex

        # Get ctx_embedding for context positions
        ctx_emb_context = self.ctx_encoder(ctx)  # (B, C, ctx_dim)

        # No predictors: use zeros for future ctx_emb, N(0,1) for z
        ctx_embedding_dim = self.config.get("ctx_embedding_dim", 3)
        ctx_emb_future = torch.zeros(B, horizon, ctx_embedding_dim, device=self.device)
        ctx_emb = torch.cat([ctx_emb_context, ctx_emb_future], dim=1)

        if z is None:
            z = torch.randn(B, T, self.config["latent_dim"], device=self.device)

        # Use encoder mean for context positions
        z_mean_ctx, _, _ = self.encoder(ctx)
        z[:, :C, :] = z_mean_ctx

        # Decode to (mean, L, nu)
        mean, L, nu = self.decoder(ctx_emb, z)
        mean = mean[:, C:]
        L = L[:, C:]

        if sample_from_decoder:
            return self.sample_from_decoder(mean, L, nu)
        else:
            return mean, L, nu


# =============================================================================
# Simple MLP Student-t Decoder (ported from experiment)
# =============================================================================
# These classes achieve 135.8% kurtosis recovery and 39.5% z contribution
# by using a direct z→mean MLP path instead of LSTM+FiLM.


# GT kurtosis values from shape_diagnostics.json (Fisher/excess kurtosis)
GT_KURTOSIS_MLP = np.array([
    [5.53, 2.75, 2.21, 15.33, 19.66],    # Row 0 (short-term moneyness)
    [21.96, 3.24, 2.51, 84.76, 18.65],   # Row 1
    [65.47, 9.98, 5.02, 2.82, 12.75],    # Row 2 (ATM row)
    [147.78, 47.95, 25.18, 8.50, 143.95],# Row 3
    [75.01, 73.92, 69.44, 44.23, 244.48] # Row 4 (long-term)
])


def compute_nu_from_kurtosis_mlp(gt_kurtosis: np.ndarray) -> np.ndarray:
    """
    Compute fixed ν for Student-t from GT kurtosis.

    For Student-t: kurtosis = 3 + 6/(ν-4) for ν>4
    Excess kurtosis = 6/(ν-4)
    Solving: ν = 4 + 6/excess_kurtosis

    Args:
        gt_kurtosis: Ground truth kurtosis (Fisher definition, normal=0)

    Returns:
        nu: Degrees of freedom, clamped to [4.1, 100]
    """
    excess_kurtosis = np.maximum(gt_kurtosis, 0.1)  # Ensure positive
    nu = 4.0 + 6.0 / excess_kurtosis
    nu = np.clip(nu, 4.1, 100.0)
    return nu


# Precomputed fixed ν values
GT_NU_MLP = compute_nu_from_kurtosis_mlp(GT_KURTOSIS_MLP)


class StudentTMLPDecoder(nn.Module):
    """
    Simple MLP decoder with Student-t output distribution for fat tails.

    This is ported from the experiment that achieved:
    - Kurtosis recovery: 135.8%
    - Z contribution: 39.5%

    Key architecture:
    - Direct z → mean MLP path (no LSTM/FiLM)
    - Per-grid fixed ν from GT kurtosis
    - Full covariance (FF^T + D)
    - Z-dependent covariance
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        self.rank = config.get("cov_rank", 4)

        # Fixed ν from GT kurtosis (not learned)
        self.register_buffer("nu", torch.tensor(GT_NU_MLP, dtype=torch.float32).view(25))

        # Mean decoder: z → mean (direct MLP)
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Z-dependent covariance
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Initialize log_diag to reasonable variance
        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with Student-t sampling.

        Note: ctx_emb is ignored - this decoder uses only z for direct path.
        """
        B, T, _ = z.shape
        device = z.device

        # Mean prediction (z-dependent only)
        z_flat = z.view(B * T, -1)
        mean = self.mean_net(z_flat)
        mean = mean.view(B, T, 5, 5)

        # Z-dependent covariance parameters
        z_pooled = z.mean(dim=1)  # (B, latent_dim)

        factor_flat = self.factor_net(z_pooled)
        factor = factor_flat.view(B, 25, self.rank)

        log_diag = self.log_diag_net(z_pooled)
        log_diag = torch.clamp(log_diag, min=-10, max=2)

        if not sample:
            return mean, mean, factor, log_diag

        # Student-t sampling
        eps_rank = torch.randn(B, T, self.rank, device=device)

        # Create Gaussian samples, then scale by Student-t factor
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)  # (B, T, 25)

        eps_diag = torch.randn(B, T, 25, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag  # (B, T, 25)

        total_gauss = correlated_gauss + independent_gauss  # (B, T, 25)

        # Convert to Student-t by dividing by sqrt(chi2/nu)
        chi2_samples = torch.zeros(B, T, 25, device=device)
        for i in range(25):
            nu_i = self.nu[i].item()
            alpha = nu_i / 2
            beta = nu_i / 2
            gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
            chi2_samples[:, :, i] = gamma_samples

        # t = gaussian / sqrt(chi2) has Student-t distribution
        student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
        total_t = total_gauss * student_t_factor  # (B, T, 25)

        samples = mean + total_t.view(B, T, 5, 5)

        return mean, samples, factor, log_diag

    def compute_student_t_nll(self, pred_mean, target, factor, log_diag):
        """Student-t NLL with per-grid ν and full covariance."""
        B, T = pred_mean.shape[:2]
        device = pred_mean.device

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat  # (B, T, 25)

        D = torch.exp(log_diag)  # (B, 25)
        D = torch.clamp(D, min=1e-8)

        # Total variance per grid point: Σ_ii = D_i + Σ_k F_ik^2
        factor_sq = (factor ** 2).sum(dim=-1)  # (B, 25)
        total_var = D + factor_sq  # (B, 25)

        # Per-grid univariate Student-t NLL
        nu = self.nu.view(1, 1, 25)  # (1, 1, 25)
        sigma_sq = total_var.view(B, 1, 25)  # (B, 1, 25)

        z_sq = residual ** 2 / sigma_sq  # (B, T, 25)

        # Student-t NLL (ignoring normalizing constant)
        nll = 0.5 * (nu + 1) * torch.log(1 + z_sq / nu) + 0.5 * torch.log(sigma_sq)

        return nll.mean()


class CVAETwoStageStudentTMLP(nn.Module):
    """
    Two-Stage CVAE with simple MLP Student-t decoder.

    This is ported from the experiment that achieved:
    - Kurtosis recovery: 135.8%
    - Z contribution: 39.5%
    - Direction accuracy: 48.7%

    Key differences from CVAETwoStageStudentT:
    - Uses StudentTMLPDecoder (simple MLP) instead of TwoStageStudentTDecoder (LSTM+FiLM)
    - Has prior_net for future z prediction
    - Uses main_encoder naming (for checkpoint compatibility)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = StudentTMLPDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with Student-t decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        mean, samples, factor, log_diag = self.decoder(ctx_emb, z, sample=True)

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with Student-t decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        samples = []
        for _ in range(n_samples):
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z
            _, sample, _, _ = self.decoder(ctx_emb, z, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)


# =============================================================================
# Dual-Path Student-t Decoder (new architecture)
# =============================================================================
# This decoder uses ADDITIVE combination of context and z pathways:
#   mean = ctx_mean + z_residual
#
# This forces both pathways to contribute (unlike FiLM which can be bypassed).
# Key insight: decompose into predictable (ctx) + unpredictable (z) components.


class StudentTDualPathDecoder(nn.Module):
    """
    Dual-path decoder with additive combination of context and z.

    Architecture:
        ctx_emb -> [Context MLP] -> ctx_mean (expected drift)
        z       -> [Residual MLP] -> z_residual (innovation)
        mean = ctx_mean + z_residual

    Key insight: Additive combination forces both pathways to contribute.
    Unlike FiLM (multiplicative), neither pathway can be "turned off".

    This preserves:
    - Kurtosis recovery (Student-t sampling unchanged)
    - Z contribution (z_residual pathway)
    - Context usage (ctx_mean pathway)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        ctx_dim = config.get("ctx_embedding_dim", 3)
        self.rank = config.get("cov_rank", 4)

        # Fixed ν from GT kurtosis (not learned)
        self.register_buffer("nu", torch.tensor(GT_NU_MLP, dtype=torch.float32).view(25))

        # Context pathway: ctx_emb -> expected mean
        # Smaller network than z pathway (context is coarse)
        self.ctx_mean_net = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 25),
        )

        # Z pathway: z -> residual mean
        # Same architecture as original StudentTMLPDecoder
        self.z_residual_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Z-dependent covariance (unchanged from StudentTMLPDecoder)
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Initialize log_diag to reasonable variance
        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)

        # Initialize z_residual to small values so ctx dominates initially
        # This helps with training stability
        nn.init.zeros_(self.z_residual_net[-1].bias)
        nn.init.normal_(self.z_residual_net[-1].weight, std=0.01)

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with dual-path mean and Student-t sampling.

        Args:
            ctx_emb: Context embedding (B, T, ctx_dim)
            z: Latent variable (B, T, latent_dim)
            sample: Whether to sample from Student-t

        Returns:
            mean: Predicted mean (B, T, 5, 5)
            samples: Sampled surface (B, T, 5, 5) if sample=True, else mean
            factor: Low-rank covariance factor (B, 25, rank)
            log_diag: Log diagonal covariance (B, 25)
        """
        B, T, _ = z.shape
        device = z.device

        # Context pathway: ctx_emb -> expected mean
        ctx_flat = ctx_emb.view(B * T, -1)
        ctx_mean = self.ctx_mean_net(ctx_flat)  # (B*T, 25)
        ctx_mean = ctx_mean.view(B, T, 5, 5)

        # Z pathway: z -> residual mean
        z_flat = z.view(B * T, -1)
        z_residual = self.z_residual_net(z_flat)  # (B*T, 25)
        z_residual = z_residual.view(B, T, 5, 5)

        # ADDITIVE combination (key difference from FiLM)
        mean = ctx_mean + z_residual

        # Z-dependent covariance parameters
        z_pooled = z.mean(dim=1)  # (B, latent_dim)

        factor_flat = self.factor_net(z_pooled)
        factor = factor_flat.view(B, 25, self.rank)

        log_diag = self.log_diag_net(z_pooled)
        log_diag = torch.clamp(log_diag, min=-10, max=2)

        if not sample:
            return mean, mean, factor, log_diag

        # Student-t sampling (same as StudentTMLPDecoder)
        eps_rank = torch.randn(B, T, self.rank, device=device)

        # Create Gaussian samples, then scale by Student-t factor
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)  # (B, T, 25)

        eps_diag = torch.randn(B, T, 25, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag  # (B, T, 25)

        total_gauss = correlated_gauss + independent_gauss  # (B, T, 25)

        # Convert to Student-t by dividing by sqrt(chi2/nu)
        chi2_samples = torch.zeros(B, T, 25, device=device)
        for i in range(25):
            nu_i = self.nu[i].item()
            alpha = nu_i / 2
            beta = nu_i / 2
            gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
            chi2_samples[:, :, i] = gamma_samples

        # t = gaussian / sqrt(chi2) has Student-t distribution
        student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
        total_t = total_gauss * student_t_factor  # (B, T, 25)

        samples = mean + total_t.view(B, T, 5, 5)

        return mean, samples, factor, log_diag

    def compute_student_t_nll(self, pred_mean, target, factor, log_diag):
        """Student-t NLL with per-grid ν and full covariance."""
        B, T = pred_mean.shape[:2]

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat  # (B, T, 25)

        D = torch.exp(log_diag)  # (B, 25)
        D = torch.clamp(D, min=1e-8)

        # Total variance per grid point: Σ_ii = D_i + Σ_k F_ik^2
        factor_sq = (factor ** 2).sum(dim=-1)  # (B, 25)
        total_var = D + factor_sq  # (B, 25)

        # Per-grid univariate Student-t NLL
        nu = self.nu.view(1, 1, 25)  # (1, 1, 25)
        sigma_sq = total_var.view(B, 1, 25)  # (B, 1, 25)

        z_sq = residual ** 2 / sigma_sq  # (B, T, 25)

        # Student-t NLL (ignoring normalizing constant)
        nll = 0.5 * (nu + 1) * torch.log(1 + z_sq / nu) + 0.5 * torch.log(sigma_sq)

        return nll.mean()


class CVAETwoStageDualPath(nn.Module):
    """
    Two-Stage CVAE with dual-path additive decoder.

    This addresses the fundamental trade-off:
    - MLP decoder: 39.5% z contribution, 0% ctx contribution (ignores context)
    - LSTM+FiLM: 3.8% z contribution, ~60% ctx contribution (washes out z)

    Solution: Additive combination forces both pathways to contribute:
        mean = f(ctx_emb) + g(z)

    Expected improvements:
    - ctx_contribution > 20%
    - z_contribution > 20%
    - Kurtosis recovery preserved (>100%)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = StudentTDualPathDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with dual-path decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        mean, samples, factor, log_diag = self.decoder(ctx_emb, z, sample=True)

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with dual-path decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        samples = []
        for _ in range(n_samples):
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z
            _, sample, _, _ = self.decoder(ctx_emb, z, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)

    def get_pathway_contributions(self, batch):
        """
        Compute contribution of each pathway to the mean prediction.

        Returns dict with:
        - ctx_mean: Mean from context pathway only
        - z_residual: Mean from z pathway only
        - full_mean: ctx_mean + z_residual
        """
        surface = batch["surface"]
        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        B, T, _ = z.shape
        device = z.device

        # Context pathway only
        ctx_flat = ctx_emb.view(B * T, -1)
        ctx_mean = self.decoder.ctx_mean_net(ctx_flat).view(B, T, 5, 5)

        # Z pathway only
        z_flat = z.view(B * T, -1)
        z_residual = self.decoder.z_residual_net(z_flat).view(B, T, 5, 5)

        return {
            "ctx_mean": ctx_mean,
            "z_residual": z_residual,
            "full_mean": ctx_mean + z_residual,
        }


# =============================================================================
# Gated Residual Decoder (variant 2)
# =============================================================================
# This decoder uses z as the primary prediction, with context providing
# a small gated correction: mean = z_pred + gate * ctx_correction
#
# Key insight: Keep z dominant (preserve kurtosis) but add context signal.


class StudentTGatedResidualDecoder(nn.Module):
    """
    Decoder with gated residual connection from context.

    Architecture:
        z       -> [Z MLP] ---------> z_pred (main prediction, unchanged from StudentTMLPDecoder)
        ctx_emb -> [Ctx MLP] -> ctx_correction
        gate    -> [Gate Net] -> gate (0-1)
        mean = z_pred + gate * ctx_correction

    Key insight: Z provides main prediction (preserves kurtosis),
    context provides small correction (enables context usage without bypass).
    Gate prevents context from dominating.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        ctx_dim = config.get("ctx_embedding_dim", 3)
        self.rank = config.get("cov_rank", 4)
        self.gate_scale = config.get("gate_scale", 0.3)  # Max gate value

        # Fixed ν from GT kurtosis (not learned)
        self.register_buffer("nu", torch.tensor(GT_NU_MLP, dtype=torch.float32).view(25))

        # Z pathway: z -> main prediction (UNCHANGED from StudentTMLPDecoder)
        self.z_pred_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Context pathway: ctx_emb -> small correction
        self.ctx_correction_net = nn.Sequential(
            nn.Linear(ctx_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 25),
        )

        # Gate network: combines z and ctx to produce gate (0-1)
        # Gate controls how much context correction is applied
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim + ctx_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 25),
        )

        # Z-dependent covariance (unchanged from StudentTMLPDecoder)
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Initialize log_diag to reasonable variance
        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)

        # Initialize ctx_correction to small values
        nn.init.zeros_(self.ctx_correction_net[-1].bias)
        nn.init.normal_(self.ctx_correction_net[-1].weight, std=0.01)

        # Initialize gate to start closed (near 0)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with gated residual context correction.

        mean = z_pred + gate * ctx_correction
        where gate ∈ [0, gate_scale]
        """
        B, T, _ = z.shape
        device = z.device

        # Z pathway: main prediction
        z_flat = z.view(B * T, -1)
        z_pred = self.z_pred_net(z_flat)  # (B*T, 25)
        z_pred = z_pred.view(B, T, 5, 5)

        # Context pathway: small correction
        ctx_flat = ctx_emb.view(B * T, -1)
        ctx_correction = self.ctx_correction_net(ctx_flat)  # (B*T, 25)
        ctx_correction = ctx_correction.view(B, T, 5, 5)

        # Gate: controls how much context is applied
        combined = torch.cat([z_flat, ctx_flat], dim=-1)  # (B*T, latent_dim + ctx_dim)
        gate_logits = self.gate_net(combined)  # (B*T, 25)
        gate = torch.sigmoid(gate_logits) * self.gate_scale  # (B*T, 25), range [0, gate_scale]
        gate = gate.view(B, T, 5, 5)

        # Gated residual: mean = z_pred + gate * ctx_correction
        mean = z_pred + gate * ctx_correction

        # Z-dependent covariance parameters
        z_pooled = z.mean(dim=1)  # (B, latent_dim)

        factor_flat = self.factor_net(z_pooled)
        factor = factor_flat.view(B, 25, self.rank)

        log_diag = self.log_diag_net(z_pooled)
        log_diag = torch.clamp(log_diag, min=-10, max=2)

        if not sample:
            return mean, mean, factor, log_diag

        # Student-t sampling (same as StudentTMLPDecoder)
        eps_rank = torch.randn(B, T, self.rank, device=device)
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)

        eps_diag = torch.randn(B, T, 25, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag

        total_gauss = correlated_gauss + independent_gauss

        chi2_samples = torch.zeros(B, T, 25, device=device)
        for i in range(25):
            nu_i = self.nu[i].item()
            alpha = nu_i / 2
            beta = nu_i / 2
            gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
            chi2_samples[:, :, i] = gamma_samples

        student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
        total_t = total_gauss * student_t_factor

        samples = mean + total_t.view(B, T, 5, 5)

        return mean, samples, factor, log_diag

    def compute_student_t_nll(self, pred_mean, target, factor, log_diag):
        """Student-t NLL with per-grid ν and full covariance."""
        B, T = pred_mean.shape[:2]

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat

        D = torch.exp(log_diag)
        D = torch.clamp(D, min=1e-8)

        factor_sq = (factor ** 2).sum(dim=-1)
        total_var = D + factor_sq

        nu = self.nu.view(1, 1, 25)
        sigma_sq = total_var.view(B, 1, 25)

        z_sq = residual ** 2 / sigma_sq

        nll = 0.5 * (nu + 1) * torch.log(1 + z_sq / nu) + 0.5 * torch.log(sigma_sq)

        return nll.mean()


class CVAETwoStageGatedResidual(nn.Module):
    """
    Two-Stage CVAE with gated residual decoder.

    Uses z as primary prediction with gated context correction.
    Expected: preserve kurtosis (z dominant) while adding some context usage.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = StudentTGatedResidualDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with gated residual decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        mean, samples, factor, log_diag = self.decoder(ctx_emb, z, sample=True)

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with gated residual decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        samples = []
        for _ in range(n_samples):
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z
            _, sample, _, _ = self.decoder(ctx_emb, z, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)


# =============================================================================
# Dual-Path AR(1) Decoder (ACF preservation variant)
# =============================================================================
# This decoder adds autoregressive mean-reversion to the dual-path decoder:
#   mean_t = ctx_mean_t + z_residual_t + φ * (x_{t-1} - μ)
#
# Key insight: Explicit AR(1) structure enforces temporal coherence and
# mean-reversion, while still allowing both ctx and z to contribute.


class StudentTDualPathARDecoder(nn.Module):
    """
    Dual-path decoder with AR(1) mean-reversion component.

    Architecture:
        ctx_emb -> [Context MLP] -> ctx_mean (expected drift)
        z       -> [Residual MLP] -> z_residual (innovation)
        AR(1)   -> φ * (x_{t-1} - μ) (mean-reversion correction)
        mean = ctx_mean + z_residual + ar_correction

    Key insight: Combines additive dual-path (for ctx+z contribution) with
    explicit AR(1) structure (for ACF preservation / mean-reversion).

    The AR(1) coefficient φ can be learned or fixed to GT value (-0.35).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        ctx_dim = config.get("ctx_embedding_dim", 3)
        self.rank = config.get("cov_rank", 4)

        # AR(1) parameters
        self.learn_phi = config.get("learn_ar_phi", True)
        self.target_phi = config.get("target_ar_phi", -0.35)  # GT value

        if self.learn_phi:
            # Learnable AR(1) coefficient, initialized near target
            self.phi_logit = nn.Parameter(torch.tensor(0.0))
        else:
            # Fixed AR(1) coefficient
            self.register_buffer("phi", torch.tensor(self.target_phi))

        # Long-run mean (learned per grid point)
        self.mu = nn.Parameter(torch.zeros(25))

        # Fixed ν from GT kurtosis (not learned)
        self.register_buffer("nu", torch.tensor(GT_NU_MLP, dtype=torch.float32).view(25))

        # Context pathway: ctx_emb -> expected mean
        self.ctx_mean_net = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 25),
        )

        # Z pathway: z -> residual mean
        self.z_residual_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Z-dependent covariance (unchanged)
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)
        nn.init.zeros_(self.z_residual_net[-1].bias)
        nn.init.normal_(self.z_residual_net[-1].weight, std=0.01)

    def get_phi(self):
        """Get AR(1) coefficient, constrained to (-1, 1) for stationarity."""
        if self.learn_phi:
            # tanh constrains to (-1, 1)
            return torch.tanh(self.phi_logit)
        return self.phi

    def forward(self, ctx_emb, z, prev_x=None, sample=True):
        """
        Forward pass with dual-path mean + AR(1) correction.

        Args:
            ctx_emb: Context embedding (B, T, ctx_dim)
            z: Latent variable (B, T, latent_dim)
            prev_x: Previous timestep surface (B, T, 5, 5) for AR correction
                    If None, AR correction is skipped (first timestep)
            sample: Whether to sample from Student-t

        Returns:
            mean: Predicted mean (B, T, 5, 5)
            samples: Sampled surface (B, T, 5, 5)
            factor: Low-rank covariance factor (B, 25, rank)
            log_diag: Log diagonal covariance (B, 25)
        """
        B, T, _ = z.shape
        device = z.device

        # Context pathway
        ctx_flat = ctx_emb.view(B * T, -1)
        ctx_mean = self.ctx_mean_net(ctx_flat).view(B, T, 5, 5)

        # Z pathway
        z_flat = z.view(B * T, -1)
        z_residual = self.z_residual_net(z_flat).view(B, T, 5, 5)

        # Base mean (without AR correction)
        mean_base = ctx_mean + z_residual

        # AR(1) correction
        if prev_x is not None:
            phi = self.get_phi()
            mu = self.mu.view(1, 1, 5, 5)
            ar_correction = phi * (prev_x - mu)
            mean = mean_base + ar_correction
        else:
            mean = mean_base

        # Z-dependent covariance
        z_pooled = z.mean(dim=1)
        factor_flat = self.factor_net(z_pooled)
        factor = factor_flat.view(B, 25, self.rank)

        log_diag = self.log_diag_net(z_pooled)
        log_diag = torch.clamp(log_diag, min=-10, max=2)

        if not sample:
            return mean, mean, factor, log_diag

        # Student-t sampling (unchanged)
        eps_rank = torch.randn(B, T, self.rank, device=device)
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)

        eps_diag = torch.randn(B, T, 25, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag

        total_gauss = correlated_gauss + independent_gauss

        chi2_samples = torch.zeros(B, T, 25, device=device)
        for i in range(25):
            nu_i = self.nu[i].item()
            alpha = nu_i / 2
            beta = nu_i / 2
            gamma_samples = torch._standard_gamma(
                torch.full((B, T), alpha, device=device)
            ) / beta
            chi2_samples[:, :, i] = gamma_samples

        student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
        total_t = total_gauss * student_t_factor

        samples = mean + total_t.view(B, T, 5, 5)

        return mean, samples, factor, log_diag

    def compute_student_t_nll(self, pred_mean, target, factor, log_diag):
        """Student-t NLL with per-grid ν and full covariance."""
        B, T = pred_mean.shape[:2]

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat

        D = torch.exp(log_diag)
        D = torch.clamp(D, min=1e-8)

        factor_sq = (factor ** 2).sum(dim=-1)
        total_var = D + factor_sq

        nu = self.nu.view(1, 1, 25)
        sigma_sq = total_var.view(B, 1, 25)

        z_sq = residual ** 2 / sigma_sq

        nll = 0.5 * (nu + 1) * torch.log(1 + z_sq / nu) + 0.5 * torch.log(sigma_sq)

        return nll.mean()


class CVAETwoStageDualPathAR(nn.Module):
    """
    Two-Stage CVAE with dual-path AR(1) decoder for ACF preservation.

    Combines:
    - Dual-path additive decoder (ctx + z contribution)
    - AR(1) mean-reversion (temporal coherence / ACF preservation)
    - Student-t sampling (kurtosis recovery)

    Expected improvements:
    - ACF preservation: >30% (up from 15-19%)
    - Kurtosis recovery: preserved (>100%)
    - ctx_contribution: preserved (>5%)
    - z_contribution: preserved (>20%)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = StudentTDualPathARDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with AR(1) correction."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        # Construct prev_x for AR(1) correction
        # prev_x[:, t] = surface[:, t] (the actual previous surface)
        # For t=0, we don't have prev_x, so we use the first surface as placeholder
        prev_x = surface.clone()

        mean, samples, factor, log_diag = self.decoder(
            ctx_emb, z, prev_x=prev_x, sample=True
        )

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with AR(1) decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        prev_x = surface.clone()

        samples = []
        for _ in range(n_samples):
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z
            _, sample, _, _ = self.decoder(ctx_emb, z, prev_x=prev_x, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)

    def get_ar_phi(self):
        """Get the AR(1) coefficient for monitoring."""
        return self.decoder.get_phi()

    def get_pathway_contributions(self, batch):
        """
        Compute contribution of each pathway.

        Returns dict with ctx_mean, z_residual, ar_correction, and full_mean.
        """
        surface = batch["surface"]
        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        B, T, _ = z.shape

        # Context pathway
        ctx_flat = ctx_emb.view(B * T, -1)
        ctx_mean = self.decoder.ctx_mean_net(ctx_flat).view(B, T, 5, 5)

        # Z pathway
        z_flat = z.view(B * T, -1)
        z_residual = self.decoder.z_residual_net(z_flat).view(B, T, 5, 5)

        # AR(1) correction
        phi = self.decoder.get_phi()
        mu = self.decoder.mu.view(1, 1, 5, 5)
        ar_correction = phi * (surface - mu)

        return {
            "ctx_mean": ctx_mean,
            "z_residual": z_residual,
            "ar_correction": ar_correction,
            "full_mean": ctx_mean + z_residual + ar_correction,
        }


# =============================================================================
# SKEW STUDENT-T DECODER (Sinh-Arcsinh Transformation)
# =============================================================================


class StudentTSkewDecoder(nn.Module):
    """
    Student-t decoder with learnable skewness via sinh-arcsinh transformation.

    Extends StudentTMLPDecoder by adding:
    - epsilon (ε): Per-grid skewness parameter (learned from data)
    - delta (δ): Per-grid tailweight parameter (optional, can fix to 1)

    The distribution is:
        Y = μ + L @ sinh((arcsinh(X) + ε) * δ)
    where X ~ StudentT(ν)

    This follows the "bitter lesson" - let the model learn skewness
    from data rather than hand-engineering it.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        self.rank = config.get("cov_rank", 4)
        self.learn_delta = config.get("learn_delta", False)  # Can fix delta=1

        # Fixed ν from GT kurtosis (not learned)
        self.register_buffer("nu", torch.tensor(GT_NU_MLP, dtype=torch.float32).view(25))

        # Mean decoder: z → mean (direct MLP)
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Z-dependent covariance
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Skewness network: z → epsilon (per-grid skewness)
        # Initialize to 0 (symmetric) and let it learn
        self.epsilon_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),  # Soft constraint on range
            nn.Linear(32, 25),
        )
        # Initialize last layer to near-zero
        nn.init.zeros_(self.epsilon_net[-1].weight)
        nn.init.zeros_(self.epsilon_net[-1].bias)

        # Optional: tailweight network
        if self.learn_delta:
            self.log_delta_net = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 25),
            )
            nn.init.zeros_(self.log_delta_net[-1].bias)  # delta=1 at init
        else:
            # Fixed delta=1
            self.register_buffer("delta", torch.ones(25))

        # Initialize log_diag to reasonable variance
        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)

    def get_epsilon(self, z_pooled: torch.Tensor) -> torch.Tensor:
        """Get per-grid skewness parameters."""
        epsilon = self.epsilon_net(z_pooled)  # (B, 25)
        # Clamp to reasonable range
        epsilon = torch.clamp(epsilon, min=-3.0, max=3.0)
        return epsilon

    def get_delta(self, z_pooled: torch.Tensor) -> torch.Tensor:
        """Get per-grid tailweight parameters."""
        if self.learn_delta:
            log_delta = self.log_delta_net(z_pooled)
            delta = torch.exp(torch.clamp(log_delta, min=-1.0, max=1.0))  # (0.37, 2.7)
        else:
            B = z_pooled.shape[0]
            delta = self.delta.unsqueeze(0).expand(B, -1)
        return delta

    def sinh_arcsinh_forward(
        self, x: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Apply sinh-arcsinh transformation: X → Y (add skewness)."""
        return torch.sinh((torch.asinh(x) + epsilon) * delta)

    def sinh_arcsinh_inverse(
        self, y: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Inverse sinh-arcsinh transformation: Y → X (remove skewness)."""
        return torch.sinh(torch.asinh(y) / delta - epsilon)

    def sinh_arcsinh_log_jacobian(
        self, y: torch.Tensor, epsilon: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Log |dX/dY| for density transformation."""
        asinh_y = torch.asinh(y)
        inner = asinh_y / delta - epsilon

        # Numerically stable log(cosh(x))
        log_cosh = torch.logaddexp(inner, -inner) - np.log(2.0)

        log_jacobian = log_cosh - torch.log(delta) - 0.5 * torch.log1p(y ** 2)
        return log_jacobian

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with skewed Student-t sampling.

        Note: ctx_emb is ignored - this decoder uses only z for direct path.

        Returns:
            mean: Predicted mean (B, T, 5, 5)
            samples: Samples from skewed Student-t (B, T, 5, 5)
            factor: Covariance factor (B, 25, rank)
            log_diag: Log diagonal variance (B, 25)
            epsilon: Skewness parameters (B, 25)
            delta: Tailweight parameters (B, 25)
        """
        B, T, _ = z.shape
        device = z.device

        # Mean prediction (z-dependent only)
        z_flat = z.view(B * T, -1)
        mean = self.mean_net(z_flat)
        mean = mean.view(B, T, 5, 5)

        # Z-dependent parameters (pooled across time)
        z_pooled = z.mean(dim=1)  # (B, latent_dim)

        factor_flat = self.factor_net(z_pooled)
        factor = factor_flat.view(B, 25, self.rank)

        log_diag = self.log_diag_net(z_pooled)
        log_diag = torch.clamp(log_diag, min=-10, max=2)

        # Skewness parameters
        epsilon = self.get_epsilon(z_pooled)  # (B, 25)
        delta = self.get_delta(z_pooled)  # (B, 25)

        if not sample:
            return mean, mean, factor, log_diag, epsilon, delta

        # Skewed Student-t sampling
        # Step 1: Sample from standard Student-t (symmetric)
        eps_rank = torch.randn(B, T, self.rank, device=device)

        # Create Gaussian samples, then scale by Student-t factor
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)  # (B, T, 25)

        eps_diag = torch.randn(B, T, 25, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag  # (B, T, 25)

        total_gauss = correlated_gauss + independent_gauss  # (B, T, 25)

        # Convert to Student-t by dividing by sqrt(chi2/nu)
        chi2_samples = torch.zeros(B, T, 25, device=device)
        for i in range(25):
            nu_i = self.nu[i].item()
            alpha = nu_i / 2
            beta = nu_i / 2
            gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
            chi2_samples[:, :, i] = gamma_samples

        # t = gaussian / sqrt(chi2) has Student-t distribution
        student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
        total_t = total_gauss * student_t_factor  # (B, T, 25) - symmetric Student-t

        # Step 2: Apply sinh-arcsinh transformation to add skewness
        epsilon_expanded = epsilon.view(B, 1, 25)  # (B, 1, 25)
        delta_expanded = delta.view(B, 1, 25)  # (B, 1, 25)
        total_skewed = self.sinh_arcsinh_forward(total_t, epsilon_expanded, delta_expanded)

        samples = mean + total_skewed.view(B, T, 5, 5)

        return mean, samples, factor, log_diag, epsilon, delta

    def compute_skew_student_t_nll(self, pred_mean, target, factor, log_diag, epsilon, delta):
        """
        Skewed Student-t NLL with per-grid ν, ε, δ and full covariance.

        The NLL includes:
        1. Student-t log prob of transformed residuals
        2. Jacobian of sinh-arcsinh transformation
        3. Jacobian of covariance (log det)
        """
        B, T = pred_mean.shape[:2]
        device = pred_mean.device

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat  # (B, T, 25)

        D = torch.exp(log_diag)  # (B, 25)
        D = torch.clamp(D, min=1e-8)

        # Total variance per grid point: Σ_ii = D_i + Σ_k F_ik^2
        factor_sq = (factor ** 2).sum(dim=-1)  # (B, 25)
        total_var = D + factor_sq  # (B, 25)
        sigma = torch.sqrt(total_var)  # (B, 25)

        # Standardize residuals
        sigma_expanded = sigma.view(B, 1, 25)
        y_std = residual / sigma_expanded  # (B, T, 25)

        # Transform back to symmetric space via inverse sinh-arcsinh
        epsilon_expanded = epsilon.view(B, 1, 25)
        delta_expanded = delta.view(B, 1, 25)
        x = self.sinh_arcsinh_inverse(y_std, epsilon_expanded, delta_expanded)  # (B, T, 25)

        # Student-t NLL for symmetric variable x
        nu = self.nu.view(1, 1, 25)  # (1, 1, 25)
        log_prob_base = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * np.pi)
            - ((nu + 1) / 2) * torch.log1p(x ** 2 / nu)
        )

        # Jacobian for sinh-arcsinh transform
        log_jacobian_sa = self.sinh_arcsinh_log_jacobian(y_std, epsilon_expanded, delta_expanded)

        # Jacobian for scale (sigma)
        log_jacobian_scale = -torch.log(sigma_expanded)

        # Total log prob
        log_prob = log_prob_base + log_jacobian_sa + log_jacobian_scale

        # NLL = -log_prob
        nll = -log_prob.mean()

        return nll


class CVAETwoStageStudentTSkew(nn.Module):
    """
    Two-Stage CVAE with skewed Student-t decoder (sinh-arcsinh transformation).

    This model learns per-grid skewness from data following the "bitter lesson":
    - Don't hand-engineer which grids should have positive/negative skew
    - Let the model learn epsilon (skewness) through NLL optimization
    - When epsilon=0, reduces to symmetric Student-t (CVAETwoStageStudentTMLP)

    Architecture:
    - Context encoder: Historical surfaces → context embedding
    - Main encoder: Full sequence → latent z (VAE)
    - Skew decoder: z → μ, Σ, ν, ε, δ → skewed Student-t samples
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = StudentTSkewDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with skewed Student-t decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        mean, samples, factor, log_diag, epsilon, delta = self.decoder(ctx_emb, z, sample=True)

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag, epsilon, delta

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with skewed Student-t decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        all_samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps

            _, samples, _, _, _, _ = self.decoder(ctx_emb, z, sample=True)
            all_samples.append(samples)

        return torch.stack(all_samples, dim=0)  # (n_samples, B, T, 5, 5)

    def compute_loss(self, batch, kl_weight=1.0):
        """Compute skewed Student-t NLL + KL loss."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        mean, z_mean, z_logvar, factor, log_diag, epsilon, delta = self.forward(
            batch, return_full_sequence=True
        )

        # Skewed Student-t NLL
        nll = self.decoder.compute_skew_student_t_nll(
            mean, surface, factor, log_diag, epsilon, delta
        )

        # KL divergence
        kl = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

        total_loss = nll + kl_weight * kl

        return {
            "loss": total_loss,
            "nll": nll,
            "kl": kl,
            "epsilon_mean": epsilon.mean().item(),
            "epsilon_std": epsilon.std().item(),
            "delta_mean": delta.mean().item() if self.decoder.learn_delta else 1.0,
        }

    def get_learned_skewness(self, batch):
        """Extract the learned skewness parameters for analysis."""
        surface = batch["surface"]
        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        z_pooled = z.mean(dim=1)
        epsilon = self.decoder.get_epsilon(z_pooled)
        delta = self.decoder.get_delta(z_pooled)

        return {
            "epsilon": epsilon,  # (B, 25)
            "delta": delta,  # (B, 25)
            "epsilon_grid": epsilon.view(-1, 5, 5),  # (B, 5, 5)
        }
