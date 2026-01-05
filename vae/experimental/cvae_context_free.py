"""
CVAEContextFree: CVAEMemRand with context removed from decoder.

Key difference: Decoder receives only z, not concat(ctx_embedding, z).
This forces z to carry all information, potentially solving the
"decoder ignores z" problem.

The hypothesis is that when the decoder can use context directly,
it learns to ignore z (measured decoder gain = 1.2e-7). By removing
this shortcut, we force z to carry all information about the target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from collections import OrderedDict

from vae.cvae_with_mem_randomized import (
    CVAEMemRand,
    CVAEMemRandEncoder,
    CVAEMemRandDecoder,
)


class CVAEContextFreeDecoder(nn.Module):
    """
    Modified decoder that receives only z (no context embedding).

    Same architecture as CVAEMemRandDecoder but with input_size = latent_dim
    instead of latent_dim + ctx_embedding_dim.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.device = config["device"]

        surface_hidden = config["surface_hidden"]
        ex_feats_hidden = config["ex_feats_hidden"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]

        # Record dimensions
        self.surface_final_hidden_size = surface_hidden[-1]
        if ex_feats_hidden is not None:
            self.n_info = self.ex_feats_final_hidden_size = ex_feats_hidden[-1]
        else:
            self.n_info = self.ex_feats_final_hidden_size = config["ex_feats_dim"]

        if config["use_dense_surface"]:
            self.n_surface = surface_hidden[-1]
        else:
            self.n_surface = surface_hidden[-1] * feat_dim[0] * feat_dim[1]

        # LSTM: input is ONLY z (no context embedding!)
        # This is the key change from CVAEMemRandDecoder
        self._build_mem(config, input_size=latent_dim)

        self._build_interaction_layers(config, self.n_surface + self.n_info)
        self.surface_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_surface)
        self._build_surface_decoder(config)

        if self.n_info > 0:
            self.ex_feats_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_info)
            self._build_ex_feats_decoder(config)

    def _build_mem(self, config, input_size):
        """Build LSTM with input_size = latent_dim only."""
        mem_type = config["mem_type"]
        hidden_size = self.n_surface + self.n_info

        mem_args = {
            "input_size": input_size,  # Just latent_dim, no ctx_embedding!
            "hidden_size": hidden_size,
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

    def _build_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"dec_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"dec_interaction_activation_{i}"] = nn.ReLU()
            interaction["dec_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def _build_surface_decoder(self, config):
        surface_hidden = config["surface_hidden"]
        surface_decoder = OrderedDict()

        if config["use_dense_surface"]:
            feat_dim = config["feat_dim"]
            in_feats = surface_hidden[-1]
            for i, out_feats in enumerate(reversed(surface_hidden[:-1])):
                surface_decoder[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
                surface_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            final_size = feat_dim[0] * feat_dim[1]
            surface_decoder["dec_final"] = nn.Linear(in_feats, final_size)
            surface_decoder["dec_final_activation"] = nn.ReLU()
            surface_decoder["dec_output"] = nn.Linear(final_size, final_size)
        else:
            padding = config["padding"]
            deconv_output_padding = config["deconv_output_padding"]
            in_feats = surface_hidden[-1]
            for i, out_feats in enumerate(reversed(surface_hidden[:-1])):
                surface_decoder[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
                )
                surface_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            surface_decoder["dec_final"] = nn.ConvTranspose2d(
                in_feats, in_feats,
                kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
            )
            surface_decoder["dec_final_activation"] = nn.ReLU()
            surface_decoder["dec_output"] = nn.Conv2d(in_feats, 1, kernel_size=3, padding="same")

        self.surface_decoder = nn.Sequential(surface_decoder)

    def _build_ex_feats_decoder(self, config):
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_hidden = config["ex_feats_hidden"]

        if ex_feats_hidden is None:
            self.ex_feats_decoder = nn.Linear(ex_feats_dim, ex_feats_dim)
            return

        ex_feats_decoder = OrderedDict()
        in_feats = ex_feats_hidden[-1]
        for i, out_feats in enumerate(reversed(ex_feats_hidden[:-1])):
            ex_feats_decoder[f"ctx_ex_dec_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ex_feats_decoder[f"ctx_ex_dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        ex_feats_decoder["ctx_dec_output"] = nn.Linear(in_feats, ex_feats_dim)
        self.ex_feats_decoder = nn.Sequential(ex_feats_decoder)

    def forward(self, z):
        """
        Decode z to surface (and optionally ex_feats).

        Input:
            z: (B, T, latent_dim) - just the latent, no context embedding!

        Returns:
            decoded_surface: (B, T, H, W)
            decoded_ex_feat: (B, T, ex_feats_dim) if ex_feats_dim > 0
        """
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config["ex_feats_dim"]

        x, _ = self.mem(z)  # (B, T, n_surface + n_info)
        x = self.interaction(x)
        B, T = x.shape[0], x.shape[1]

        surface_x = self.surface_decoder_input(x)  # (B, T, n_surface)

        if self.config["use_dense_surface"]:
            surface_x = surface_x.reshape(-1, self.surface_final_hidden_size)
            decoded_surface = self.surface_decoder(surface_x)
            decoded_surface = decoded_surface.reshape((B, T, feat_dim[0], feat_dim[1]))
        else:
            surface_x = surface_x.reshape(-1, self.surface_final_hidden_size, feat_dim[0], feat_dim[1])
            decoded_surface = self.surface_decoder(surface_x)
            decoded_surface = decoded_surface.reshape((B, T, feat_dim[0], feat_dim[1]))

        if ex_feats_dim > 0:
            info_x = self.ex_feats_decoder_input(x)
            info_x = info_x.reshape(B * T, self.n_info)
            decoded_ex_feat = self.ex_feats_decoder(info_x)
            decoded_ex_feat = decoded_ex_feat.reshape((B, T, ex_feats_dim))
            return decoded_surface, decoded_ex_feat
        else:
            return decoded_surface


class CVAEContextFree(CVAEMemRand):
    """
    CVAEMemRand with context removed from decoder input.

    Same architecture, same config, same training - just decoder
    receives only z instead of concat(ctx_embedding, z).

    This forces z to carry all information about the target, potentially
    solving the "decoder ignores z" problem observed in the original model.
    """

    def __init__(self, config: dict):
        # Call parent init (creates encoder, ctx_encoder, decoder)
        super().__init__(config)

        # Replace decoder with context-free version
        self.decoder = CVAEContextFreeDecoder(config)
        self.decoder.to(self.device)

        # Create prior encoder (processes context only → μ, logvar)
        # Uses same architecture as main encoder
        self.prior_encoder = CVAEMemRandEncoder(config)
        self.prior_encoder.to(self.device)

        # Optional: full covariance for prior
        self.covariance_type = config.get("covariance_type", "diagonal")
        if self.covariance_type == "full":
            self._init_full_covariance(config["latent_dim"])

    def _init_full_covariance(self, latent_dim):
        """Initialize learnable Cholesky factor for full covariance prior."""
        self.L_diag = nn.Parameter(torch.ones(latent_dim, device=self.device))
        self.L_lower = nn.Parameter(torch.zeros(
            latent_dim * (latent_dim - 1) // 2, device=self.device
        ))

    def get_cholesky(self):
        """Construct Cholesky factor L such that Σ = L @ L.T"""
        d = len(self.L_diag)
        L = torch.diag(F.softplus(self.L_diag))
        tril_idx = torch.tril_indices(d, d, offset=-1, device=self.device)
        L[tril_idx[0], tril_idx[1]] = self.L_lower
        return L

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        Forward pass with context-free decoder.

        Same as CVAEMemRand.forward but:
        1. Compute prior from context only
        2. Don't concatenate ctx_embedding to decoder input

        Returns:
            Tuple of (decoded_surface, [decoded_ex_feat], z_mean, z_log_var, z,
                     prior_z_mean, prior_z_log_var)
        """
        surface = x["surface"]
        T = surface.shape[1]
        B = surface.shape[0]

        # Context = all but last timestep(s)
        context_len = T - self.horizon
        context_dict = {"surface": surface[:, :context_len]}
        if "ex_feats" in x:
            context_dict["ex_feats"] = x["ex_feats"][:, :context_len]

        # Posterior: encode full sequence (existing encoder)
        z_mean, z_log_var, z = self.encoder(x)

        # Prior: encode context only (new prior encoder)
        prior_z_mean, prior_z_log_var, _ = self.prior_encoder(context_dict)

        # Decode with z only (NO ctx_embedding!)
        if "ex_feats" in x and self.config["ex_feats_dim"] > 0:
            decoded_surface, decoded_ex_feat = self.decoder(z)
            return (decoded_surface, decoded_ex_feat, z_mean, z_log_var, z,
                    prior_z_mean, prior_z_log_var)
        else:
            decoded_surface = self.decoder(z)
            return (decoded_surface, z_mean, z_log_var, z,
                    prior_z_mean, prior_z_log_var)

    def train_step(self, x: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
        """Training step with prior-posterior KL."""
        x = {k: v.to(self.device) for k, v in x.items()}
        optimizer.zero_grad()

        forward_output = self.forward(x)
        losses = self._compute_loss(x, forward_output)

        losses["loss"].backward()
        optimizer.step()

        return losses

    def test_step(self, x: Dict[str, torch.Tensor]):
        """Test step with prior-posterior KL."""
        x = {k: v.to(self.device) for k, v in x.items()}

        forward_output = self.forward(x)
        losses = self._compute_loss(x, forward_output)

        return losses

    def _compute_loss(self, x, forward_output):
        """
        Compute loss with prior-posterior KL instead of posterior-N(0,1).

        KL(q(z|x) || p(z|context)) instead of KL(q(z|x) || N(0,1))
        """
        # Unpack forward output
        if len(forward_output) == 7:  # With ex_feats
            decoded_surface, decoded_ex_feat, z_mean, z_log_var, z, prior_mu, prior_logvar = forward_output
        else:
            decoded_surface, z_mean, z_log_var, z, prior_mu, prior_logvar = forward_output
            decoded_ex_feat = None

        # Reconstruction loss
        target_surface = x["surface"]
        mse_loss = F.mse_loss(decoded_surface, target_surface)

        # KL divergence: q(z|x) || p(z|context)
        if self.covariance_type == "full":
            kl_loss = self._kl_full_cov(z_mean, z_log_var, prior_mu)
        else:
            kl_loss = self._kl_diagonal(z_mean, z_log_var, prior_mu, prior_logvar)

        total_loss = mse_loss + self.kl_weight * kl_loss

        # Ex feats loss if applicable
        if decoded_ex_feat is not None and "ex_feats" in x:
            if self.config.get("ex_loss_on_ret_only", False):
                # Only compute loss on returns (first feature)
                ex_feat_loss = self.ex_feats_loss_fn(
                    decoded_ex_feat[:, :, 0], x["ex_feats"][:, :, 0]
                )
            else:
                ex_feat_loss = self.ex_feats_loss_fn(decoded_ex_feat, x["ex_feats"])
            total_loss = total_loss + self.config.get("re_feat_weight", 1.0) * ex_feat_loss
        else:
            ex_feat_loss = torch.tensor(0.0, device=self.device)

        return {
            "loss": total_loss,
            "reconstruction_loss": mse_loss,
            "kl_loss": kl_loss,
            "ex_feats_loss": ex_feat_loss,
        }

    def _kl_diagonal(self, post_mu, post_logvar, prior_mu, prior_logvar):
        """KL divergence between two diagonal Gaussians."""
        # KL(N(μ_q, σ²_q) || N(μ_p, σ²_p))
        # = 0.5 * sum(log(σ²_p/σ²_q) + (σ²_q + (μ_q - μ_p)²)/σ²_p - 1)

        # Handle shape mismatch: prior has shape (B, C, d), posterior has (B, T, d)
        # We need to align them. Prior is computed from context only.
        # Expand prior to match posterior timesteps for comparison
        T = post_mu.shape[1]
        C = prior_mu.shape[1]

        # For KL, we compare posterior at each timestep with the prior
        # The prior is the same for all future timesteps
        # Take the last prior timestep as the reference
        prior_mu_expanded = prior_mu[:, -1:, :].expand(-1, T, -1)
        prior_logvar_expanded = prior_logvar[:, -1:, :].expand(-1, T, -1)

        kl = 0.5 * (
            prior_logvar_expanded - post_logvar
            + (torch.exp(post_logvar) + (post_mu - prior_mu_expanded)**2) / torch.exp(prior_logvar_expanded)
            - 1
        )
        return kl.sum(dim=-1).mean()

    def _kl_full_cov(self, post_mu, post_logvar, prior_mu):
        """KL divergence: diagonal posterior vs full covariance prior."""
        L = self.get_cholesky()  # (d, d)
        d = L.shape[0]

        # Σ_p = L @ L.T
        Sigma_p = L @ L.T
        Sigma_p_inv = torch.inverse(Sigma_p)

        # Handle shape: prior_mu is (B, C, d), post_mu is (B, T, d)
        T = post_mu.shape[1]
        prior_mu_expanded = prior_mu[:, -1:, :].expand(-1, T, -1)  # (B, T, d)

        post_var = torch.exp(post_logvar)  # (B, T, d)
        diff = post_mu - prior_mu_expanded  # (B, T, d)

        # trace(Σ_p^{-1} @ diag(σ²_q)) = sum(diag(Σ_p^{-1}) * σ²_q)
        trace_term = (Sigma_p_inv.diag() * post_var).sum(dim=-1)  # (B, T)

        # (μ_q - μ_p)^T Σ_p^{-1} (μ_q - μ_p)
        quad_term = (diff @ Sigma_p_inv * diff).sum(dim=-1)  # (B, T)

        # log|Σ_p| = 2 * sum(log(diag(L)))
        log_det_p = 2 * torch.log(F.softplus(self.L_diag) + 1e-6).sum()

        # log|diag(σ²_q)| = sum(log_var)
        log_det_q = post_logvar.sum(dim=-1)  # (B, T)

        kl = 0.5 * (trace_term + quad_term - d + log_det_p - log_det_q)
        return kl.mean()

    def get_surface_given_conditions(
        self,
        c: Dict[str, torch.Tensor],
        z: torch.Tensor = None,
        mu: float = 0,
        std: float = 1,
        horizon: int = None,
        prior_mode: str = "standard"
    ):
        """
        Generate surface given context only.

        Modified to NOT use ctx_encoder - sample from prior and decode directly.

        Args:
            c: context dictionary with "surface" (B, C, H, W) and optional "ex_feats"
            z: pre-generated latent samples. If None, samples from prior.
            mu: mean for standard prior (unused in this model)
            std: std for standard prior (unused in this model)
            horizon: number of days to forecast
            prior_mode: "standard" uses learned prior, "fitted" uses fitted GMM if available

        Returns:
            decoded_surface: (B, horizon, H, W)
            decoded_ex_feat: (B, horizon, ex_feats_dim) if ex_feats present
        """
        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)
        C = ctx_surface.shape[1]
        B = ctx_surface.shape[0]

        if horizon is None:
            horizon = self.horizon
        T = C + horizon

        ctx = {"surface": ctx_surface}
        if "ex_feats" in c:
            ctx_ex_feats = c["ex_feats"].to(self.device)
            if len(ctx_ex_feats.shape) == 2:
                ctx_ex_feats = ctx_ex_feats.unsqueeze(0)
            ctx["ex_feats"] = ctx_ex_feats

        # Get prior distribution from context
        prior_mu, prior_logvar, _ = self.prior_encoder(ctx)

        # Sample z for full sequence
        if z is None:
            z = torch.zeros((B, T, self.config["latent_dim"]), device=self.device)

            # Context positions: use posterior mean (deterministic encoding)
            ctx_z_mean, _, _ = self.encoder(ctx)
            z[:, :C, :] = ctx_z_mean

            # Future positions: sample from prior
            # Use last timestep of prior as the distribution for future
            future_prior_mu = prior_mu[:, -1, :]  # (B, d)
            future_prior_logvar = prior_logvar[:, -1, :]  # (B, d)

            if self.covariance_type == "full":
                L = self.get_cholesky()
                for h in range(horizon):
                    eps = torch.randn((B, self.config["latent_dim"]), device=self.device)
                    z[:, C + h, :] = future_prior_mu + (eps @ L.T)
            else:
                for h in range(horizon):
                    eps = torch.randn((B, self.config["latent_dim"]), device=self.device)
                    z[:, C + h, :] = future_prior_mu + eps * torch.exp(0.5 * future_prior_logvar)

        # Decode with z only (NO ctx_embedding!)
        if "ex_feats" in c and self.config["ex_feats_dim"] > 0:
            decoded_surface, decoded_ex_feat = self.decoder(z)
            return decoded_surface[:, C:, :, :], decoded_ex_feat[:, C:, :]
        else:
            decoded_surface = self.decoder(z)
            return decoded_surface[:, C:, :, :]

    def generate_samples(
        self,
        context: Dict[str, torch.Tensor],
        n_samples: int = 100,
        horizon: int = None
    ) -> torch.Tensor:
        """
        Generate multiple samples for uncertainty quantification.

        Args:
            context: dict with "surface" (B, C, H, W) or (C, H, W)
            n_samples: number of samples to generate
            horizon: forecast horizon (default: self.horizon)

        Returns:
            samples: (B, n_samples, horizon, H, W)
        """
        ctx_surface = context["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)
        B = ctx_surface.shape[0]

        if horizon is None:
            horizon = self.horizon

        samples = []
        for _ in range(n_samples):
            sample = self.get_surface_given_conditions(context, horizon=horizon)
            samples.append(sample)

        return torch.stack(samples, dim=1)  # (B, n_samples, horizon, H, W)
