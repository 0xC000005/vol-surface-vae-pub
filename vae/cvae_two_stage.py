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
