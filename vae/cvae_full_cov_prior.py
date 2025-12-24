"""
CVAE with Full Covariance Prior

This model extends CVAEMemRand with a full covariance prior that addresses
the parameter reuse problem. Key differences:

1. **Prior**: Full covariance AR(1) instead of diagonal N(0,1)
2. **KL Loss**: KL(diagonal posterior || full cov prior) instead of KL(q || N(0,1))
3. **Sampling**: Correlated Cholesky sampling instead of IID

This is a STRICT implementation - no backward compatibility, will raise errors
if configuration is incorrect.
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.full_covariance_prior import FullCovariancePrior, kl_divergence_full_covariance


class CVAEFullCovPrior(CVAEMemRand):
    """
    CVAE with Full Covariance Prior.

    This model REQUIRES full covariance prior - no fallback to diagonal prior.
    Will raise errors if misconfigured.

    Key Modifications:
    - Adds FullCovariancePrior network in __init__
    - Overrides train_step to use full covariance KL
    - Overrides get_surface_given_conditions for correlated sampling
    """

    def __init__(self, config: dict):
        # Validate REQUIRED config keys for full covariance prior
        required_keys = ["latent_dim", "context_len"]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing required config key for Full Covariance Prior: '{key}'\n"
                    f"Required keys: {required_keys}"
                )

        # Set max_horizon if not provided (default to horizon)
        if "max_horizon" not in config:
            config["max_horizon"] = config.get("horizon", 90)

        # Initialize parent class (CVAEMemRand)
        super().__init__(config)

        # Initialize full covariance prior (REQUIRED, not optional)
        print("Initializing Full Covariance Prior...")
        self.full_cov_prior = FullCovariancePrior(
            context_dim=config["latent_dim"],  # compress_context=True
            max_horizon=config["max_horizon"],
            latent_dim=config["latent_dim"],
            pos_dim=config.get("full_cov_pos_dim", 64),
            hidden_dims=config.get("full_cov_hidden_dims", [128, 128]),  # NEW: configurable hidden dims
            dropout=config.get("full_cov_dropout", 0.1),  # NEW: dropout for generalization
            init_phi=config.get("full_cov_init_phi", 0.5),
            init_sigma_sq=config.get("full_cov_init_sigma_sq", 1.0)
        )
        self.full_cov_prior = self.full_cov_prior.to(self.device)
        print(f"✓ Full Covariance Prior initialized (φ={self.full_cov_prior.get_phi().item():.4f}, σ²={self.full_cov_prior.get_sigma_sq().item():.4f})")

    def train_step(self, x, optimizer: torch.optim.Optimizer, scaler=None):
        '''
        Training step with Full Covariance Prior KL loss.

        Computes KL(diagonal posterior || full covariance prior) instead of
        the standard KL(q || N(0,1)).

        Args:
            x: Dictionary with "surface" and optionally "ex_feats"
            optimizer: PyTorch optimizer
            scaler: Optional GradScaler for mixed precision training

        Returns:
            Dictionary with loss components
        '''
        surface = x["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - self.horizon  # Context length
        surface_real = surface[:, C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :].to(self.device)

        # Move input data to device for forward pass
        x_device = {"surface": surface.to(self.device)}
        if "ex_feats" in x:
            x_device["ex_feats"] = ex_feats.to(self.device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training with BF16
        with autocast('cuda', dtype=torch.bfloat16):
            if "ex_feats" in x:
                surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x_device)
            else:
                surface_reconstruction, z_mean, z_log_var, z = self.forward(x_device)

            # Reconstruction loss
            re_surface = nn.functional.mse_loss(surface_reconstruction, surface_real)
            if "ex_feats" in x:
                if self.config["ex_loss_on_ret_only"]:
                    ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                    ex_feats_real = ex_feats_real[:, :, :1]
                re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
            else:
                reconstruction_error = re_surface
                re_ex_feats = torch.tensor(0.0, device=self.device)

            # ============================================================================
            # FULL COVARIANCE PRIOR KL LOSS
            # ============================================================================
            # Get context summary for prior
            ctx_surface = x_device["surface"][:, :C, :, :]
            ctx_encoder_input = {"surface": ctx_surface}
            if "ex_feats" in x:
                ctx_encoder_input["ex_feats"] = x_device["ex_feats"][:, :C, :]

            ctx_embedding = self.ctx_encoder(ctx_encoder_input)  # (B, C, latent_dim)
            context_summary = ctx_embedding[:, -1, :]  # (B, latent_dim) - last timestep

            # Get prior distribution parameters
            mu_p, Sigma_p = self.full_cov_prior.get_prior_params(
                context_summary, horizon=self.horizon
            )

            # Compute KL(diagonal posterior || full covariance prior)
            # Only compute KL for future timesteps (C:C+horizon)
            kl_loss = kl_divergence_full_covariance(
                z_mean[:, C:, :],    # (B, H, latent_dim)
                z_log_var[:, C:, :], # (B, H, latent_dim)
                mu_p,                 # (B, H, latent_dim)
                Sigma_p               # (H, H)
            )

            total_loss = reconstruction_error + self.kl_weight * kl_loss

        # Backward pass with gradient scaling
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
                                horizons=[1, 7, 14, 30, 60, 90], scaler=None):
        '''
        Multi-horizon training with Full Covariance Prior.

        Trains on multiple horizons simultaneously using full covariance prior
        for each horizon's KL loss.

        Args:
            x: Dictionary with "surface" and optionally "ex_feats"
            optimizer: PyTorch optimizer
            horizons: List of horizons to train on
            scaler: Optional GradScaler for mixed precision

        Returns:
            Dictionary with loss components and per-horizon losses
        '''
        surface = x["surface"]
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)

        # Uniform weighting across all horizons
        weights = {h: 1.0 / len(horizons) for h in horizons}

        total_loss = None
        total_reconstruction = None
        total_kl = None
        horizon_losses = {}

        with autocast('cuda', dtype=torch.bfloat16):
            for horizon in horizons:
                # Temporarily set horizon
                original_horizon = self.horizon
                self.horizon = horizon

                C = T - horizon
                surface_real = surface[:, C:C+horizon, :, :].to(self.device)

                if "ex_feats" in x:
                    ex_feats_real = ex_feats[:, C:C+horizon, :].to(self.device)

                # Forward pass
                if "ex_feats" in x:
                    surface_reconstruction, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
                else:
                    surface_reconstruction, z_mean, z_log_var, z = self.forward(x)

                # Reconstruction loss
                re_surface = nn.functional.mse_loss(surface_reconstruction, surface_real)
                if "ex_feats" in x:
                    if self.config["ex_loss_on_ret_only"]:
                        ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                        ex_feats_real = ex_feats_real[:, :, :1]
                    re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                    reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
                else:
                    reconstruction_error = re_surface

                # Full covariance prior KL loss
                ctx_surface = surface[:, :C, :, :]
                ctx_encoder_input = {"surface": ctx_surface}
                if "ex_feats" in x:
                    ctx_encoder_input["ex_feats"] = ex_feats[:, :C, :]

                ctx_embedding = self.ctx_encoder(ctx_encoder_input)
                context_summary = ctx_embedding[:, -1, :]

                mu_p, Sigma_p = self.full_cov_prior.get_prior_params(context_summary, horizon=horizon)

                kl_loss = kl_divergence_full_covariance(
                    z_mean[:, C:, :],
                    z_log_var[:, C:, :],
                    mu_p,
                    Sigma_p
                )

                horizon_loss = reconstruction_error + self.kl_weight * kl_loss
                horizon_losses[f"h{horizon}"] = horizon_loss.item()

                # Accumulate weighted loss
                if total_loss is None:
                    total_loss = weights[horizon] * horizon_loss
                    total_reconstruction = weights[horizon] * reconstruction_error
                    total_kl = weights[horizon] * kl_loss
                else:
                    total_loss += weights[horizon] * horizon_loss
                    total_reconstruction += weights[horizon] * reconstruction_error
                    total_kl += weights[horizon] * kl_loss

                # Restore original horizon
                self.horizon = original_horizon

        # Backward pass
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
            "reconstruction_loss": total_reconstruction,
            "kl_loss": total_kl,
            "horizon_losses": horizon_losses,
        }

    def get_surface_given_conditions(self, c: dict, z: torch.Tensor = None,
                                    mu=0, std=1, horizon=None, prior_mode="full_cov"):
        '''
        Generate surfaces using Full Covariance Prior sampling.

        NO FALLBACK - always uses correlated Cholesky sampling from full covariance prior.

        Args:
            c: Context dictionary with "surface" and optionally "ex_feats"
            z: Pre-generated latents (not used with full_cov prior - for compatibility only)
            mu: Not used (for backward compatibility)
            std: Not used (for backward compatibility)
            horizon: Number of days to forecast (default: self.horizon)
            prior_mode: Must be "full_cov" (other modes not supported)

        Returns:
            If ex_feats present: (surf_pred, ex_pred)
            Otherwise: surf_pred only
        '''
        if prior_mode != "full_cov":
            raise ValueError(
                f"CVAEFullCovPrior only supports prior_mode='full_cov', got '{prior_mode}'\n"
                f"This model uses full covariance prior exclusively - no fallback modes."
            )

        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)
        B = ctx_surface.shape[0]
        C = ctx_surface.shape[1]

        if horizon is None:
            horizon = self.horizon

        # Encode context
        ctx_encoder_input = {"surface": ctx_surface}
        if "ex_feats" in c:
            ctx_ex_feats = c["ex_feats"].to(self.device)
            if len(ctx_ex_feats.shape) == 2:
                ctx_ex_feats = ctx_ex_feats.unsqueeze(0)
            ctx_encoder_input["ex_feats"] = ctx_ex_feats

        ctx_embedding = self.ctx_encoder(ctx_encoder_input)  # (B, C, latent_dim)
        context_summary = ctx_embedding[:, -1, :]  # (B, latent_dim)

        # Sample from full covariance prior - CORRELATED samples!
        z_future = self.full_cov_prior.sample(context_summary, horizon)  # (B, H, latent_dim)

        # Pad context embeddings
        ctx_embedding_dim = ctx_embedding.shape[2]
        decoder_ctx = torch.zeros((B, horizon, ctx_embedding_dim), device=self.device, dtype=ctx_embedding.dtype)

        # Concatenate context and latents for decoder
        decoder_input = torch.cat([decoder_ctx, z_future], dim=-1)  # (B, H, ctx_dim + latent_dim)

        # Decode
        if "ex_feats" in c:
            surf_pred, ex_pred = self.decoder(decoder_input)
            return surf_pred, ex_pred
        else:
            surf_pred = self.decoder(decoder_input)
            return surf_pred

    def get_prior_params_summary(self):
        """
        Get summary of current full covariance prior parameters.

        Returns:
            Dictionary with φ, σ², and total parameter count
        """
        phi = self.full_cov_prior.get_phi().item()
        sigma_sq = self.full_cov_prior.get_sigma_sq().item()
        num_params = sum(p.numel() for p in self.full_cov_prior.parameters())

        return {
            "phi": phi,
            "sigma_sq": sigma_sq,
            "num_params": num_params,
            "mean_network_params": sum(p.numel() for p in self.full_cov_prior.mean_network.parameters()),
            "covariance_params": 2  # Only log_phi and log_sigma_sq
        }
