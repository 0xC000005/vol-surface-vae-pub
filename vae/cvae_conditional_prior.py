"""
CVAE with Conditional Prior Network

This module extends CVAEMemRand to support conditional prior p(z|context) instead
of the fixed N(0,1) prior, eliminating VAE prior mismatch.

Key Changes:
-----------
1. Adds ConditionalPriorNetwork to predict p(z|context)
2. Modifies KL loss: KL(q(z|x) || N(0,1)) → KL(q(z|x) || p(z|context))
3. Modifies sampling: z ~ N(0,1) → z ~ p(z|context)

Usage:
------
# In config file:
model_config = {
    ...
    "use_conditional_prior": True,  # Enable conditional prior
    ...
}

# Training automatically uses conditional prior KL loss
# Inference automatically samples from p(z|context)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Dict, Tuple, Optional, Union
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.conditional_prior_network import ConditionalPriorNetwork, kl_divergence_gaussians


class CVAEMemRandConditionalPrior(CVAEMemRand):
    """
    CVAE with learnable conditional prior network.

    Extends CVAEMemRand by adding a conditional prior network p(z|context)
    that learns appropriate prior distributions for each context, eliminating
    the systematic bias from VAE prior mismatch.
    """

    def __init__(self, config: dict):
        """
        Initialize CVAE with conditional prior.

        Args:
            config: Model configuration dict. Must include:
                - use_conditional_prior: True to enable (otherwise behaves like CVAEMemRand)
                - All standard CVAEMemRand config parameters
        """
        # Initialize base VAE
        super(CVAEMemRandConditionalPrior, self).__init__(config)

        # Add conditional prior network if enabled
        self.use_conditional_prior = config.get("use_conditional_prior", False)

        if self.use_conditional_prior:
            print("Initializing Conditional Prior Network...")
            self.prior_network = ConditionalPriorNetwork(config)
            self.prior_network.to(self.device)
            print("✓ Conditional Prior Network initialized")
        else:
            self.prior_network = None
            print("Note: use_conditional_prior=False, using standard N(0,1) prior")

    def compute_kl_loss(self, z_mean, z_logvar, context):
        """
        Compute KL divergence loss.

        If conditional prior enabled:
            KL(q(z|context, target) || p(z|context))
        Else:
            KL(q(z|context, target) || N(0,1))  [standard VAE]

        Args:
            z_mean: Posterior mean (B, T, latent_dim)
            z_logvar: Posterior log-variance (B, T, latent_dim)
            context: Context dict (only used if conditional prior enabled)

        Returns:
            Scalar KL loss
        """
        if self.use_conditional_prior and self.prior_network is not None:
            # Conditional prior: KL(q(z|x) || p(z|context))
            with torch.no_grad() if not self.training else torch.enable_grad():
                # Get prior parameters from context
                # Extract context portion from full sequence
                T = context["surface"].shape[1]
                C = T - self.horizon  # Actual context length
                ctx_only = {
                    "surface": context["surface"][:, :C, :, :]
                }
                if "ex_feats" in context:
                    ctx_only["ex_feats"] = context["ex_feats"][:, :C, :]

                prior_mean, prior_logvar = self.prior_network(ctx_only)

            # KL between two Gaussians
            kl_loss = kl_divergence_gaussians(
                z_mean[:, :C, :],      # Posterior mean (context portion)
                z_logvar[:, :C, :],    # Posterior logvar (context portion)
                prior_mean,            # Conditional prior mean
                prior_logvar           # Conditional prior logvar
            )
        else:
            # Standard prior: KL(q(z|x) || N(0,1))
            kl_loss = -0.5 * (1 + z_logvar - torch.exp(z_logvar) - torch.square(z_mean))
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))

        return kl_loss

    def get_surface_given_conditions(
        self,
        c: dict[str, torch.Tensor],
        z: torch.Tensor = None,
        mu=0,
        std=1,
        horizon=None,
        prior_mode="standard",
        sample_context: bool = False,
        ar_phi: float = 0.0
    ):
        """
        Generate surface given context only.

        If conditional prior enabled, automatically uses p(z|context) for sampling.

        Args:
            c: Context dictionary
            z: Pre-specified latents (optional)
            mu: Mean for standard prior (ignored if conditional prior used)
            std: Std for standard prior (ignored if conditional prior used)
            horizon: Forecast horizon
            prior_mode: "standard" or "fitted" (ignored if conditional prior used)
            sample_context: If True, sample context latents instead of using posterior mean (Exp 2)
            ar_phi: AR(1) persistence parameter for future latents. 0.0 = iid noise (default), >0 = correlated (Exp 4)

        Returns:
            Generated surfaces (and ex_feats if present)
        """
        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)
        C = ctx_surface.shape[1]
        B = ctx_surface.shape[0]

        # Use provided horizon or model's default
        if horizon is None:
            horizon = self.horizon

        T = C + horizon
        ctx = {"surface": ctx_surface}

        if "ex_feats" in c:
            ctx_ex_feats = c["ex_feats"].to(self.device)
            if len(ctx_ex_feats.shape) == 2:
                ctx_ex_feats = ctx_ex_feats.unsqueeze(0)
            assert ctx_ex_feats.shape[1] == C, "context length mismatch"
            ctx["ex_feats"] = ctx_ex_feats

        # Sample latent z
        if z is not None:
            if len(z.shape) == 2:
                z = z.unsqueeze(0)
        else:
            # Use conditional prior if available
            if self.use_conditional_prior and self.prior_network is not None:
                # Sample from p(z|context)
                with torch.no_grad():
                    prior_mean, prior_logvar = self.prior_network(ctx)

                # Sample future latents from conditional prior
                # Extend prior_mean and prior_logvar for horizon
                # Use last context timestep's prior for all future timesteps
                future_prior_mean = prior_mean[:, -1:, :].expand(B, horizon, -1)
                future_prior_logvar = prior_logvar[:, -1:, :].expand(B, horizon, -1)

                # Experiment 4: AR(1) correlated noise
                if ar_phi > 0:
                    # Generate AR(1) autocorrelated noise: eps_t = phi * eps_{t-1} + sqrt(1-phi^2) * innovation
                    import math
                    ar_noise_list = [torch.randn(B, 1, self.config["latent_dim"], device=self.device, dtype=torch.float32)]
                    for t in range(1, horizon):
                        innovation = torch.randn(B, 1, self.config["latent_dim"], device=self.device, dtype=torch.float32)
                        ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
                        ar_noise_list.append(ar_noise_t)
                    eps = torch.cat(ar_noise_list, dim=1)  # (B, horizon, latent_dim)
                else:
                    # Standard iid noise
                    eps = torch.randn((B, horizon, self.config["latent_dim"]),
                                     device=self.device, dtype=torch.float32)

                z_future = future_prior_mean + torch.exp(0.5 * future_prior_logvar) * eps

                # Combine context and future latents
                z = torch.zeros((B, T, self.config["latent_dim"]),
                               device=self.device, dtype=torch.float32)
                z[:, C:, :] = z_future
            else:
                # Use standard/fitted prior (same as parent class)
                if prior_mode == "fitted" and hasattr(self, 'fitted_prior'):
                    z = self._sample_from_fitted_prior(B, T)
                else:
                    z = mu + torch.randn((B, T, self.config["latent_dim"]),
                                        device=self.device, dtype=torch.float32) * std

        # Encode context to get posterior mean for context positions
        ctx_latent_mean, ctx_latent_log_var, ctx_latent = self.encoder(ctx)

        # Experiment 2: Sample context latents instead of using deterministic mean
        if sample_context:
            # Sample from posterior: z = mean + exp(0.5 * logvar) * eps
            eps_ctx = torch.randn_like(ctx_latent_mean)
            z[:, :C, ...] = ctx_latent_mean + torch.exp(0.5 * ctx_latent_log_var) * eps_ctx
        else:
            # Deterministic posterior mean (original behavior)
            z[:, :C, ...] = ctx_latent_mean

        # Decode
        ctx_embedding = self.ctx_encoder(ctx)
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(self.device)
        ctx_embedding_padded[:, :C, :] = ctx_embedding
        z = z.to(self.device)
        decoder_input = torch.cat([ctx_embedding_padded, z], dim=-1)

        if "ex_feats" in c:
            decoded_surface, decoded_ex_feat = self.decoder(decoder_input)
            return decoded_surface[:, C:, :, :, :], decoded_ex_feat[:, C:, :]
        else:
            decoded_surface = self.decoder(decoder_input)
            return decoded_surface[:, C:, :, :, :]

    def train_step(self, x, optimizer: torch.optim.Optimizer, scaler=None):
        """
        Training step with conditional prior KL loss.

        Overrides parent to use compute_kl_loss() instead of standard KL.

        Args:
            x: Input batch
            optimizer: Optimizer
            scaler: Optional GradScaler for mixed precision training
        """
        surface = x["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - self.horizon
        surface_real = surface[:, C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :].to(self.device)

        optimizer.zero_grad()

        # Mixed precision training with BF16
        with autocast(dtype=torch.bfloat16):
            if "ex_feats" in x:
                surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
            else:
                surface_reconstruction, z_mean, z_log_var, z = self.forward(x)

            # Reconstruction loss
            re_surface = self.quantile_loss_fn(surface_reconstruction, surface_real)

            if "ex_feats" in x:
                if self.config["ex_loss_on_ret_only"]:
                    ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                    ex_feats_real = ex_feats_real[:, :, :1]
                re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
            else:
                reconstruction_error = re_surface
                re_ex_feats = torch.zeros(1)

            # KL loss - uses conditional prior if enabled
            context = {"surface": surface.to(self.device)}
            if "ex_feats" in x:
                context["ex_feats"] = ex_feats.to(self.device)
            kl_loss = self.compute_kl_loss(z_mean, z_log_var, context)

            total_loss = reconstruction_error + self.kl_weight * kl_loss

        # Use GradScaler if provided (AMP), otherwise regular backward
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

        return {
            "loss": total_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }

    def train_step_multihorizon(self, x, optimizer: torch.optim.Optimizer,
                                horizons=[1, 7, 14, 30], scaler=None):
        """
        Multi-horizon training with conditional prior KL loss.

        If cache_encoder_multihorizon=True (default): Optimized path - encode once,
        resample z per horizon. ~2x faster (reduced from 2.7x due to prior recompute).
        If False: Legacy path - full forward pass per horizon.

        Args:
            x: Input batch
            optimizer: Optimizer
            horizons: List of horizon lengths to train on
            scaler: Optional GradScaler for mixed precision training
        """
        surface = x["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        surface = surface.to(self.device)

        has_ex_feats = "ex_feats" in x
        if has_ex_feats:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats = ex_feats.to(self.device)

        optimizer.zero_grad()

        use_cache = self.config.get("cache_encoder_multihorizon", True)

        # Mixed precision training with BF16
        with autocast(dtype=torch.bfloat16):
            if use_cache:
                # === OPTIMIZED PATH: Encode once, resample z per horizon ===

                # Encode ONCE (full sequence) - get deterministic z_mean, z_log_var
                full_input = {"surface": surface}
                if has_ex_feats:
                    full_input["ex_feats"] = ex_feats
                z_mean_full, z_logvar_full, _ = self.encoder(full_input)  # Ignore sampled z

                # Loop horizons - compute ctx_encoder and prior per horizon
                total_loss = 0
                horizon_losses = {}

                for h in horizons:
                    C = T - h
                    if C < 1:
                        continue

                    # Ctx_encoder for this horizon's context length
                    ctx_input = {"surface": surface[:, :C, :, :]}
                    if has_ex_feats:
                        ctx_input["ex_feats"] = ex_feats[:, :C, :]
                    ctx_embedding = self.ctx_encoder(ctx_input)

                    # Prior network for this horizon's context length
                    prior_mean, prior_logvar = self.prior_network(ctx_input)

                    # Pad ctx_embedding to full sequence length
                    ctx_embedding_dim = ctx_embedding.shape[2]
                    ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim), device=self.device)
                    ctx_embedding_padded[:, :C, :] = ctx_embedding

                    # Resample z for this horizon (fresh stochasticity)
                    eps = torch.randn_like(z_mean_full)
                    z = z_mean_full + eps * torch.exp(0.5 * z_logvar_full)

                    # Decode
                    decoder_input = torch.cat([ctx_embedding_padded, z], dim=-1)
                    if has_ex_feats:
                        surface_reconstruction, ex_feats_reconstruction = self.decoder(decoder_input)
                    else:
                        surface_reconstruction = self.decoder(decoder_input)

                    # Reconstruction loss
                    surface_real = surface[:, C:, :, :]
                    re_surface = self.quantile_loss_fn(surface_reconstruction[:, C:, :, :, :], surface_real)

                    if has_ex_feats:
                        if self.config["ex_loss_on_ret_only"]:
                            ex_feats_pred = ex_feats_reconstruction[:, C:, :1]
                            ex_feats_real = ex_feats[:, C:, :1]
                        else:
                            ex_feats_pred = ex_feats_reconstruction[:, C:, :]
                            ex_feats_real = ex_feats[:, C:, :]
                        re_ex_feats = self.ex_feats_loss_fn(ex_feats_pred, ex_feats_real)
                        reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
                    else:
                        reconstruction_error = re_surface
                        re_ex_feats = torch.zeros(1)

                    # KL loss using prior computed for this horizon
                    kl_loss = kl_divergence_gaussians(
                        z_mean_full[:, :C, :], z_logvar_full[:, :C, :],
                        prior_mean, prior_logvar
                    )

                    horizon_loss = reconstruction_error + self.kl_weight * kl_loss
                    total_loss = total_loss + horizon_loss
                    horizon_losses[h] = horizon_loss.item()

            else:
                # === LEGACY PATH: Full forward pass per horizon ===
                total_loss = 0
                horizon_losses = {}

                for h in horizons:
                    original_horizon = self.horizon
                    self.horizon = h

                    C = T - h
                    if C < 1:
                        self.horizon = original_horizon
                        continue

                    surface_real = surface[:, C:, :, :]

                    if has_ex_feats:
                        surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
                        if self.config["ex_loss_on_ret_only"]:
                            ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                            ex_feats_real = ex_feats[:, C:, :1]
                        else:
                            ex_feats_real = ex_feats[:, C:, :]
                        re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                    else:
                        surface_reconstruction, z_mean, z_log_var, z = self.forward(x)
                        re_ex_feats = torch.zeros(1)

                    re_surface = self.quantile_loss_fn(surface_reconstruction, surface_real)

                    if has_ex_feats:
                        reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
                    else:
                        reconstruction_error = re_surface

                    context = {"surface": surface}
                    if has_ex_feats:
                        context["ex_feats"] = ex_feats
                    kl_loss = self.compute_kl_loss(z_mean, z_log_var, context)

                    horizon_loss = reconstruction_error + self.kl_weight * kl_loss
                    total_loss = total_loss + horizon_loss
                    horizon_losses[h] = horizon_loss.item()

                    self.horizon = original_horizon

            total_loss = total_loss / len(horizons)

        # Use GradScaler if provided (AMP), otherwise regular backward
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

        return {
            "loss": total_loss,
            "horizon_losses": horizon_losses,
            "kl_loss": kl_loss,
        }

    def test_step(self, x):
        """
        Test step with conditional prior KL loss.

        Overrides parent to use compute_kl_loss() instead of standard KL.
        """
        surface = x["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)
        T = surface.shape[1]
        C = T - self.horizon
        surface_real = surface[:, C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :].to(self.device)

        with torch.no_grad():
            if "ex_feats" in x:
                surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
            else:
                surface_reconstruction, z_mean, z_log_var, z = self.forward(x)

            # Reconstruction loss
            re_surface = self.quantile_loss_fn(surface_reconstruction, surface_real)

            if "ex_feats" in x:
                if self.config["ex_loss_on_ret_only"]:
                    ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                    ex_feats_real = ex_feats_real[:, :, :1]
                re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
            else:
                reconstruction_error = re_surface
                re_ex_feats = torch.zeros(1)

            # KL loss - uses conditional prior if enabled
            context = {"surface": surface.to(self.device)}
            if "ex_feats" in x:
                context["ex_feats"] = ex_feats.to(self.device)
            kl_loss = self.compute_kl_loss(z_mean, z_log_var, context)

            total_loss = reconstruction_error + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }

    def save_model(self, path: str):
        """
        Save model with conditional prior network.

        Args:
            path: Path to save model
        """
        save_dict = {
            'model_config': self.config,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'ctx_encoder_state_dict': self.ctx_encoder.state_dict(),
        }

        # Save prior network if it exists
        if self.use_conditional_prior and self.prior_network is not None:
            save_dict['prior_network_state_dict'] = self.prior_network.state_dict()
            save_dict['use_conditional_prior'] = True
        else:
            save_dict['use_conditional_prior'] = False

        torch.save(save_dict, path)

    def load_weights(self, dict_to_load: dict):
        """
        Load model weights including conditional prior network.

        Args:
            dict_to_load: Dictionary with state dicts
        """
        # Load base model weights
        self.encoder.load_state_dict(dict_to_load['encoder_state_dict'])
        self.decoder.load_state_dict(dict_to_load['decoder_state_dict'])
        self.ctx_encoder.load_state_dict(dict_to_load['ctx_encoder_state_dict'])

        # Load prior network if it exists
        if dict_to_load.get('use_conditional_prior', False):
            if 'prior_network_state_dict' in dict_to_load:
                if self.prior_network is not None:
                    self.prior_network.load_state_dict(dict_to_load['prior_network_state_dict'])
                    print("✓ Loaded conditional prior network weights")
                else:
                    print("Warning: Checkpoint has prior network but model doesn't. Skipping.")
            else:
                print("Warning: use_conditional_prior=True but no prior_network_state_dict found")
