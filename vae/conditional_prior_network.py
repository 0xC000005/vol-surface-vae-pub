"""
Conditional Prior Network for VAE

This module implements a learnable prior p(z|context) that replaces the fixed N(0,1)
prior in standard VAEs. The network learns to predict appropriate latent distributions
conditioned on context, eliminating the VAE prior mismatch issue.

Key Concept:
-----------
Instead of:    q(z|context, target)  vs  N(0,1)  ← mismatch!
We learn:      q(z|context, target)  vs  p(z|context)  ← learned conditional prior

During training, KL divergence is computed as:
    KL(q(z|context, target) || p(z|context))

During inference, we sample:
    z ~ p(z|context)  (adapts to each specific context)
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class ConditionalPriorNetwork(nn.Module):
    """
    Learns context-conditioned prior distribution p(z|context).

    Architecture:
        context → Context Encoder → prior_mean, prior_logvar

    The network shares the same architecture as CVAECtxMemRandEncoder to ensure
    it processes context in a compatible way.
    """

    def __init__(self, config: dict):
        """
        Initialize conditional prior network.

        Args:
            config: Model configuration dict containing:
                - latent_dim: Dimension of latent space
                - mem_hidden: Hidden size of LSTM/GRU/RNN
                - ctx_surface_hidden: Context surface encoder layers
                - ctx_ex_feats_hidden: Context extra features encoder layers (optional)
                - mem_type: Type of memory (lstm/gru/rnn)
                - mem_layers: Number of memory layers
                - mem_dropout: Dropout rate for memory
                - ex_feats_dim: Dimension of extra features
                - feat_dim: Dimension of surface features (H, W)
                - use_dense_surface: Whether to use dense layers for surface
                - padding: Padding for conv layers (computed by model)
                - interaction_layers: Number of interaction layers
                - compress_context: Whether to compress to latent_dim
        """
        super(ConditionalPriorNetwork, self).__init__()
        self.config = config
        self.device = config["device"]

        latent_dim = config["latent_dim"]

        # Build context encoding layers (shared architecture with ctx_encoder)
        surface_embedding_dim = self.__get_surface_embedding(config)
        if config["ex_feats_dim"] > 0:
            ex_feats_embedding_dim = self.__get_ex_feats_embedding(config)
        else:
            ex_feats_embedding_dim = 0

        self.__get_interaction_layers(config, surface_embedding_dim + ex_feats_embedding_dim)
        mem_hidden = self.__get_mem(config, surface_embedding_dim + ex_feats_embedding_dim)

        # Compression layer if needed
        if config["compress_context"]:
            self.compression = nn.Linear(mem_hidden, latent_dim)
            hidden_dim = latent_dim
        else:
            self.compression = nn.Identity()
            hidden_dim = mem_hidden

        # Prior prediction heads
        self.prior_mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def __get_surface_embedding(self, config):
        """Build surface embedding layers (same as ctx_encoder)."""
        feat_dim = config["feat_dim"]
        ctx_surface_embedding_layers = config["ctx_surface_hidden"]

        ctx_surface_embedding = OrderedDict()
        if config["use_dense_surface"]:
            in_feats = feat_dim[0] * feat_dim[1]
            ctx_surface_embedding["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(ctx_surface_embedding_layers):
                ctx_surface_embedding[f"prior_enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                ctx_surface_embedding[f"prior_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats

            self.surface_embedding = nn.Sequential(ctx_surface_embedding)
            final_dim = in_feats
        else:
            padding = config["padding"]
            in_feats = 1  # one channel per surface
            for i, out_feats in enumerate(ctx_surface_embedding_layers):
                ctx_surface_embedding[f"prior_enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding,
                )
                ctx_surface_embedding[f"prior_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            ctx_surface_embedding["flatten"] = nn.Flatten()
            self.surface_embedding = nn.Sequential(ctx_surface_embedding)

            final_dim = in_feats * feat_dim[0] * feat_dim[1]
        return final_dim

    def __get_ex_feats_embedding(self, config):
        """Build extra features embedding layers."""
        ex_feats_dim = config["ex_feats_dim"]
        ctx_ex_feats_embedding_layers = config["ctx_ex_feats_hidden"]
        if ctx_ex_feats_embedding_layers is None:
            self.ex_feats_embedding = nn.Identity()
            return ex_feats_dim

        ctx_ex_feats_embedding = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ctx_ex_feats_embedding_layers):
            ctx_ex_feats_embedding[f"prior_ex_enc_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ctx_ex_feats_embedding[f"prior_ex_enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ex_feats_embedding = nn.Sequential(ctx_ex_feats_embedding)
        return in_feats

    def __get_interaction_layers(self, config, input_size):
        """Build interaction layers for feature fusion."""
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"prior_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"prior_interaction_activation_{i}"] = nn.ReLU()

            interaction["prior_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size):
        """Build LSTM/GRU/RNN memory module."""
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": config["mem_hidden"],
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
        return config["mem_hidden"]

    def forward(self, context: dict):
        """
        Predict prior distribution conditioned on context.

        Args:
            context: Dictionary with keys:
                - "surface": (B, C, H, W) - Context surfaces
                - "ex_feats": (B, C, D) - Extra features (optional)

        Returns:
            tuple of (prior_mean, prior_logvar):
                - prior_mean: (B, C, latent_dim) - Mean of p(z|context)
                - prior_logvar: (B, C, latent_dim) - Log-variance of p(z|context)
        """
        surface = context["surface"]  # (B, C, H, W)
        C = surface.shape[1]

        # Embed surface of each day individually
        surface = surface.reshape((surface.shape[0] * surface.shape[1], 1,
                                   surface.shape[2], surface.shape[3]))  # (B*C, 1, H, W)
        surface_embedding = self.surface_embedding(surface)
        surface_embedding = surface_embedding.reshape((-1, C, surface_embedding.shape[1]))

        if "ex_feats" in context:
            ex_feats = context["ex_feats"]  # (B, C, D)
            # Embed features of each day individually
            ex_feats = ex_feats.reshape((ex_feats.shape[0] * ex_feats.shape[1],
                                        ex_feats.shape[2]))  # (B*C, D)
            ex_feats_embedding = self.ex_feats_embedding(ex_feats)
            ex_feats_embedding = ex_feats_embedding.reshape((-1, C, ex_feats_embedding.shape[1]))

            embeddings = torch.cat([surface_embedding, ex_feats_embedding], dim=-1)
        else:
            embeddings = surface_embedding

        # Process through interaction layers and memory
        embeddings = self.interaction(embeddings)  # (B, C, embedding_dim)
        embeddings, _ = self.mem(embeddings)       # (B, C, mem_hidden)
        embeddings = self.compression(embeddings)  # (B, C, hidden_dim)

        # Predict prior parameters
        prior_mean = self.prior_mean_layer(embeddings)      # (B, C, latent_dim)
        prior_logvar = self.prior_logvar_layer(embeddings)  # (B, C, latent_dim)

        return prior_mean, prior_logvar


def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """
    Compute KL divergence between two diagonal Gaussians.

    KL(q || p) = 0.5 * Σ[log(σ²_p/σ²_q) + σ²_q/σ²_p + (μ_q - μ_p)²/σ²_p - 1]

    Args:
        mu_q: Posterior mean (B, T, latent_dim)
        logvar_q: Posterior log-variance (B, T, latent_dim)
        mu_p: Prior mean (B, T, latent_dim)
        logvar_p: Prior log-variance (B, T, latent_dim)

    Returns:
        Scalar KL divergence (averaged over batch and time)
    """
    kl = 0.5 * (
        logvar_p - logvar_q                            # log(σ²_p / σ²_q)
        + torch.exp(logvar_q - logvar_p)               # σ²_q / σ²_p
        + ((mu_q - mu_p) ** 2) / torch.exp(logvar_p)   # (μ_q - μ_p)² / σ²_p
        - 1                                            # constant
    )
    return kl.sum(dim=-1).mean()  # Sum over latent dims, mean over batch/time
