"""
CVAE with Prior Encoder (Ablation Experiment)

This model replaces FullCovariancePrior with PriorEncoder classes that process
raw context independently, avoiding confounded gradient flow.

Key Differences from CVAEFullCovPrior:
1. Prior encoder processes RAW context (B, C, 5, 5), not summary (B, 12)
2. Context encoder receives gradients only from reconstruction loss
3. Prior encoder receives gradients only from KL loss
4. Independent weights between context encoder and prior encoder

Two variants:
- CVAEWithPriorEncoderDiagonal: Per-timestep diagonal variance
- CVAEWithPriorEncoderFullCov: AR(1) temporal covariance
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.prior_encoder import PriorEncoderDiagonal, PriorEncoderFullCov
from vae.full_covariance_prior import kl_divergence_full_covariance


def kl_divergence_diagonal(z_mean_post, z_log_var_post, z_mean_prior, z_log_var_prior):
    """
    KL divergence between two diagonal Gaussian distributions.

    KL(q || p) = 0.5 * sum(σ_p^2 / σ_q^2 + (μ_q - μ_p)^2 / σ_p^2 - 1 + log(σ_p^2 / σ_q^2))

    Args:
        z_mean_post: (B, H, D) posterior means
        z_log_var_post: (B, H, D) posterior log-variances
        z_mean_prior: (B, H, D) prior means
        z_log_var_prior: (B, H, D) prior log-variances

    Returns:
        KL divergence scalar (averaged over batch and dimensions)
    """
    var_post = torch.exp(z_log_var_post)
    var_prior = torch.exp(z_log_var_prior)

    kl = 0.5 * (
        var_post / var_prior
        + (z_mean_post - z_mean_prior) ** 2 / var_prior
        - 1.0
        + z_log_var_prior - z_log_var_post
    )

    return kl.sum() / z_mean_post.shape[0]  # Average over batch


class CVAEWithPriorEncoderDiagonal(CVAEMemRand):
    """
    CVAE with Diagonal Prior Encoder.

    The prior encoder processes raw context independently and outputs
    per-timestep (mu, log_var) for each future timestep.

    Clean gradient flow:
    - Context encoder: gradients from reconstruction loss only
    - Prior encoder: gradients from KL loss only
    """

    def __init__(self, config: dict):
        # Validate required config keys
        required_keys = ["latent_dim", "context_len"]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing required config key for Prior Encoder: '{key}'\n"
                    f"Required keys: {required_keys}"
                )

        # Set max_horizon if not provided
        if "max_horizon" not in config:
            config["max_horizon"] = config.get("horizon", 90)

        # Initialize parent class (CVAEMemRand)
        super().__init__(config)

        # Initialize prior encoder (processes raw context)
        print("Initializing Prior Encoder (Diagonal)...")
        self.prior_encoder = PriorEncoderDiagonal(config)
        self.prior_encoder = self.prior_encoder.to(self.device)
        print("✓ Prior Encoder (Diagonal) initialized")

    def train_step(self, x, optimizer: torch.optim.Optimizer, scaler=None):
        """
        Training step with Prior Encoder.

        Key difference: Prior encoder processes raw context, receives gradients
        only from KL loss. Context encoder receives gradients only from reconstruction.

        Args:
            x: Dictionary with "surface" and optionally "ex_feats"
            optimizer: PyTorch optimizer
            scaler: Optional GradScaler (not needed for BF16)

        Returns:
            Dictionary with loss components
        """
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
            # Forward pass through main model
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

            # Prior encoder processes RAW context (not summary)
            ctx_input = {"surface": x_device["surface"][:, :C, :, :]}
            if "ex_feats" in x:
                ctx_input["ex_feats"] = x_device["ex_feats"][:, :C, :]

            # Get prior parameters from prior encoder
            mu_p, log_var_p = self.prior_encoder(ctx_input, horizon=self.horizon)

            # Compute KL(diagonal posterior || diagonal prior)
            kl_loss = kl_divergence_diagonal(
                z_mean[:, C:, :],
                z_log_var[:, C:, :],
                mu_p,
                log_var_p
            )

        total_loss = reconstruction_error + self.kl_weight * kl_loss

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
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }

    def get_surface_given_conditions(self, c: dict, z: torch.Tensor = None,
                                    mu=0, std=1, horizon=None, num_samples=1):
        """
        Generate surfaces using Prior Encoder sampling.

        Args:
            c: Context dictionary with "surface" and optionally "ex_feats"
            z: Not used (prior encoder samples internally)
            mu, std: Not used (prior encoder samples internally)
            horizon: Forecast horizon (default: self.horizon)
            num_samples: Number of samples to generate

        Returns:
            Generated surfaces (B, num_samples, H, 5, 5) if num_samples>1
            Generated surfaces (B, H, 5, 5) if num_samples=1
        """
        if horizon is None:
            horizon = self.horizon

        # Sample from prior encoder
        z_future = self.prior_encoder.sample(c, horizon, num_samples=num_samples)

        # Prepare context for decoder
        if num_samples == 1:
            # z_future: (B, H, latent_dim)
            return self._decode_samples(c, z_future, horizon)
        else:
            # z_future: (B, num_samples, H, latent_dim)
            B, N, H, D = z_future.shape
            z_flat = z_future.view(B * N, H, D)

            # Expand context for all samples
            c_expanded = {
                "surface": c["surface"].repeat_interleave(N, dim=0)
            }
            if "ex_feats" in c:
                c_expanded["ex_feats"] = c["ex_feats"].repeat_interleave(N, dim=0)

            surfaces_flat = self._decode_samples(c_expanded, z_flat, horizon)
            # Reshape back to (B, num_samples, H, 5, 5)
            return surfaces_flat.view(B, N, H, 5, 5)

    def _decode_samples(self, c, z_future, horizon):
        """Helper method to decode samples (shared between num_samples=1 and >1)."""
        C = c["surface"].shape[1]
        B = z_future.shape[0]

        # Get context embedding
        ctx_embedding = self.ctx_encoder(c)
        ctx_embedding_dim = ctx_embedding.shape[-1]

        # Prepare decoder input
        decoder_ctx = torch.zeros((B, horizon, ctx_embedding_dim), device=self.device, dtype=ctx_embedding.dtype)
        decoder_input = torch.cat([decoder_ctx, z_future], dim=-1)

        # Decode
        if self.config["ex_feats_dim"] > 0:
            surface_reconstruction, _ = self.decoder(decoder_input)
        else:
            surface_reconstruction = self.decoder(decoder_input)

        return surface_reconstruction


class CVAEWithPriorEncoderFullCov(CVAEMemRand):
    """
    CVAE with Full Covariance Prior Encoder.

    The prior encoder processes raw context independently and outputs
    mu + AR(1) covariance structure for temporal correlation.

    Clean gradient flow:
    - Context encoder: gradients from reconstruction loss only
    - Prior encoder: gradients from KL loss only
    """

    def __init__(self, config: dict):
        # Validate required config keys
        required_keys = ["latent_dim", "context_len"]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing required config key for Prior Encoder: '{key}'\n"
                    f"Required keys: {required_keys}"
                )

        # Set max_horizon if not provided
        if "max_horizon" not in config:
            config["max_horizon"] = config.get("horizon", 90)

        # Initialize parent class (CVAEMemRand)
        super().__init__(config)

        # Initialize prior encoder (processes raw context)
        print("Initializing Prior Encoder (Full Covariance)...")
        self.prior_encoder = PriorEncoderFullCov(config)
        self.prior_encoder = self.prior_encoder.to(self.device)
        print(f"✓ Prior Encoder (Full Cov) initialized (φ={self.prior_encoder.get_phi().item():.4f}, σ²={self.prior_encoder.get_sigma_sq().item():.4f})")

    def train_step(self, x, optimizer: torch.optim.Optimizer, scaler=None):
        """
        Training step with Full Covariance Prior Encoder.

        Args:
            x: Dictionary with "surface" and optionally "ex_feats"
            optimizer: PyTorch optimizer
            scaler: Optional GradScaler (not needed for BF16)

        Returns:
            Dictionary with loss components
        """
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
            # Forward pass through main model
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

            # Prior encoder processes RAW context (not summary)
            ctx_input = {"surface": x_device["surface"][:, :C, :, :]}
            if "ex_feats" in x:
                ctx_input["ex_feats"] = x_device["ex_feats"][:, :C, :]

            # Get prior parameters from prior encoder
            mu_p, Sigma_p = self.prior_encoder(ctx_input, horizon=self.horizon)

            # Compute KL(diagonal posterior || full cov prior)
            kl_loss = kl_divergence_full_covariance(
                z_mean[:, C:, :],
                z_log_var[:, C:, :],
                mu_p,
                Sigma_p
            )

        total_loss = reconstruction_error + self.kl_weight * kl_loss

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
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }

    def get_surface_given_conditions(self, c: dict, z: torch.Tensor = None,
                                    mu=0, std=1, horizon=None, num_samples=1):
        """
        Generate surfaces using Prior Encoder sampling (with Cholesky).

        Args:
            c: Context dictionary with "surface" and optionally "ex_feats"
            z: Not used (prior encoder samples internally)
            mu, std: Not used (prior encoder samples internally)
            horizon: Forecast horizon (default: self.horizon)
            num_samples: Number of samples to generate

        Returns:
            Generated surfaces (B, num_samples, H, 5, 5) if num_samples>1
            Generated surfaces (B, H, 5, 5) if num_samples=1
        """
        if horizon is None:
            horizon = self.horizon

        # Sample from prior encoder (uses Cholesky for correlated samples)
        z_future = self.prior_encoder.sample(c, horizon, num_samples=num_samples)

        # Prepare context for decoder
        if num_samples == 1:
            # z_future: (B, H, latent_dim)
            return self._decode_samples(c, z_future, horizon)
        else:
            # z_future: (B, num_samples, H, latent_dim)
            B, N, H, D = z_future.shape
            z_flat = z_future.view(B * N, H, D)

            # Expand context for all samples
            c_expanded = {
                "surface": c["surface"].repeat_interleave(N, dim=0)
            }
            if "ex_feats" in c:
                c_expanded["ex_feats"] = c["ex_feats"].repeat_interleave(N, dim=0)

            surfaces_flat = self._decode_samples(c_expanded, z_flat, horizon)
            # Reshape back to (B, num_samples, H, 5, 5)
            return surfaces_flat.view(B, N, H, 5, 5)

    def _decode_samples(self, c, z_future, horizon):
        """Helper method to decode samples (shared between num_samples=1 and >1)."""
        C = c["surface"].shape[1]
        B = z_future.shape[0]

        # Get context embedding
        ctx_embedding = self.ctx_encoder(c)
        ctx_embedding_dim = ctx_embedding.shape[-1]

        # Prepare decoder input
        decoder_ctx = torch.zeros((B, horizon, ctx_embedding_dim), device=self.device, dtype=ctx_embedding.dtype)
        decoder_input = torch.cat([decoder_ctx, z_future], dim=-1)

        # Decode
        if self.config["ex_feats_dim"] > 0:
            surface_reconstruction, _ = self.decoder(decoder_input)
        else:
            surface_reconstruction = self.decoder(decoder_input)

        return surface_reconstruction
