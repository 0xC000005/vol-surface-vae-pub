"""
CVAE with Heteroscedastic Decoder

This model extends CVAEFullCovPrior with a heteroscedastic decoder that outputs
both mean and log-variance for each grid point. Key differences:

1. **Decoder**: Outputs (mean, log_var) instead of just surface
2. **Loss**: Gaussian NLL instead of MSE for reconstruction
3. **Generation**: Sample from N(mean, exp(log_var)) for calibrated uncertainty

The model learns context-dependent variance end-to-end:
- High vol regimes → higher predicted variance
- Low vol regimes → lower predicted variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast
from collections import OrderedDict
from vae.cvae_full_cov_prior import CVAEFullCovPrior
from vae.full_covariance_prior import kl_divergence_full_covariance
from vae.base import BaseDecoder


class HeteroscedasticDecoder(BaseDecoder):
    """
    Decoder that outputs both mean and log-variance for each surface grid point.

    Architecture:
    - Shared LSTM memory (from input)
    - Shared hidden layers
    - Two heads: mean_head and logvar_head

    The log_var is clamped to prevent numerical issues.
    """

    def __init__(self, config: dict):
        super(HeteroscedasticDecoder, self).__init__(config)

        surface_embedding_layers = config["surface_hidden"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]

        if config["compress_context"]:
            ctx_embedding_dim = config["latent_dim"]
        else:
            ctx_embedding_dim = config["mem_hidden"]

        # Record sizes
        self.surface_final_hidden_size = surface_embedding_layers[-1]
        if ex_feats_embedding_layers is not None:
            self.n_info = self.ex_feats_final_hidden_size = ex_feats_embedding_layers[-1]
        else:
            self.n_info = self.ex_feats_final_hidden_size = config["ex_feats_dim"]

        if config["use_dense_surface"]:
            self.n_surface = surface_embedding_layers[-1]
        else:
            self.n_surface = surface_embedding_layers[-1] * feat_dim[0] * feat_dim[1]

        # LSTM memory
        self.__get_mem(config, latent_dim + ctx_embedding_dim, self.n_surface + self.n_info)

        # Interaction layers (shared)
        self.__get_interaction_layers(config, self.n_surface + self.n_info)

        # Surface input projection
        self.surface_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_surface)

        # Mean decoder (same as original)
        self.__get_surface_decoder(config)

        # Log-variance decoder (parallel head)
        self.__get_logvar_decoder(config)

        # Extra features decoder (if needed)
        if self.n_info > 0:
            self.ex_feats_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_info)
            self.__get_ex_feats_decoder(config)

        # Logvar initialization and bounds
        self.logvar_min = -10.0  # var >= exp(-10) ≈ 0.00005
        self.logvar_max = 2.0    # var <= exp(2) ≈ 7.4
        self.logvar_init = -5.0  # var ≈ 0.007 (similar to GT variance)

        # Initialize logvar bias
        self._init_logvar_bias()

    def _init_logvar_bias(self):
        """Initialize log-variance output to reasonable starting value."""
        if hasattr(self, 'logvar_decoder') and hasattr(self.logvar_decoder, 'dec_output'):
            nn.init.constant_(self.logvar_decoder.dec_output.bias, self.logvar_init)

    def __get_surface_decoder(self, config):
        """Build mean surface decoder (same as original)."""
        surface_embedding_layers = config["surface_hidden"]

        surface_decoder = OrderedDict()
        if config["use_dense_surface"]:
            feat_dim = config["feat_dim"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
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
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
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
            surface_decoder["dec_output"] = nn.Conv2d(
                in_feats, 1,
                kernel_size=3, padding="same"
            )
        self.surface_decoder = nn.Sequential(surface_decoder)

    def __get_logvar_decoder(self, config):
        """Build log-variance decoder (parallel to mean decoder)."""
        surface_embedding_layers = config["surface_hidden"]

        logvar_decoder = OrderedDict()
        if config["use_dense_surface"]:
            feat_dim = config["feat_dim"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
                logvar_decoder[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
                logvar_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            final_size = feat_dim[0] * feat_dim[1]
            logvar_decoder["dec_final"] = nn.Linear(in_feats, final_size)
            logvar_decoder["dec_final_activation"] = nn.ReLU()
            logvar_decoder["dec_output"] = nn.Linear(final_size, final_size)
        else:
            padding = config["padding"]
            deconv_output_padding = config["deconv_output_padding"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
                logvar_decoder[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
                )
                logvar_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            logvar_decoder["dec_final"] = nn.ConvTranspose2d(
                in_feats, in_feats,
                kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
            )
            logvar_decoder["dec_final_activation"] = nn.ReLU()
            logvar_decoder["dec_output"] = nn.Conv2d(
                in_feats, 1,
                kernel_size=3, padding="same"
            )
        self.logvar_decoder = nn.Sequential(logvar_decoder)

    def __get_ex_feats_decoder(self, config):
        """Build extra features decoder."""
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        if ex_feats_embedding_layers is None:
            self.ex_feats_decoder = nn.Linear(ex_feats_dim, ex_feats_dim)
            return

        ex_feats_decoder = OrderedDict()
        in_feats = ex_feats_embedding_layers[-1]
        for i, out_feats in enumerate(reversed(ex_feats_embedding_layers[:-1])):
            ex_feats_decoder[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
            ex_feats_decoder[f"dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        ex_feats_decoder["dec_final"] = nn.Linear(in_feats, ex_feats_dim)
        self.ex_feats_decoder = nn.Sequential(ex_feats_decoder)

    def __get_interaction_layers(self, config, hidden_size):
        """Build interaction layers."""
        interaction_layers = config.get("interaction_layers", 2)
        if interaction_layers > 0:
            interaction = OrderedDict()
            for i in range(interaction_layers):
                interaction[f"interaction_dense_{i}"] = nn.Linear(hidden_size, hidden_size)
                interaction[f"interaction_activation_{i}"] = nn.ReLU()
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size, hidden_size):
        """Build memory (LSTM/GRU/RNN)."""
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
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

    def forward(self, x):
        """
        Forward pass returning (mean, log_var) for surface.

        Input:
            x: (B, T, latent_dim + ctx_embedding_dim)

        Returns:
            decoded_mean: (B, T, H, W) - mean surface prediction
            decoded_logvar: (B, T, H, W) - log-variance per grid point
            decoded_ex_feat: (B, T, ex_feats_dim) - optional, if ex_feats_dim > 0
        """
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config["ex_feats_dim"]

        # Shared LSTM memory
        x, _ = self.mem(x)
        x = self.interaction(x)
        B, T = x.shape[0], x.shape[1]

        # Surface projection
        surface_x = self.surface_decoder_input(x)

        if self.config["use_dense_surface"]:
            surface_x_flat = surface_x.reshape(-1, self.surface_final_hidden_size)

            # Mean head
            decoded_mean = self.surface_decoder(surface_x_flat)
            decoded_mean = decoded_mean.reshape((B, T, feat_dim[0], feat_dim[1]))

            # Log-variance head
            decoded_logvar = self.logvar_decoder(surface_x_flat)
            decoded_logvar = decoded_logvar.reshape((B, T, feat_dim[0], feat_dim[1]))
        else:
            surface_x_conv = surface_x.reshape(-1, self.surface_final_hidden_size, feat_dim[0], feat_dim[1])

            # Mean head
            decoded_mean = self.surface_decoder(surface_x_conv)
            decoded_mean = decoded_mean.reshape((B, T, feat_dim[0], feat_dim[1]))

            # Log-variance head
            decoded_logvar = self.logvar_decoder(surface_x_conv)
            decoded_logvar = decoded_logvar.reshape((B, T, feat_dim[0], feat_dim[1]))

        # Clamp log-variance for numerical stability
        decoded_logvar = torch.clamp(decoded_logvar, min=self.logvar_min, max=self.logvar_max)

        if ex_feats_dim > 0:
            info_x = self.ex_feats_decoder_input(x)
            info_x = info_x.reshape(B * T, self.n_info)
            decoded_ex_feat = self.ex_feats_decoder(info_x)
            decoded_ex_feat = decoded_ex_feat.reshape((B, T, ex_feats_dim))
            return decoded_mean, decoded_logvar, decoded_ex_feat
        else:
            return decoded_mean, decoded_logvar


class CVAEHeteroscedastic(CVAEFullCovPrior):
    """
    CVAE with Heteroscedastic Decoder.

    Extends CVAEFullCovPrior with:
    - Heteroscedastic decoder that outputs (mean, log_var)
    - Gaussian NLL reconstruction loss
    - Sampling from decoded distribution for generation
    """

    def __init__(self, config: dict):
        # Initialize parent (sets up encoder, context encoder, prior)
        super().__init__(config)

        # Replace decoder with heteroscedastic version
        print("Replacing decoder with HeteroscedasticDecoder...")
        self.decoder = HeteroscedasticDecoder(config).to(self.device)
        print("✓ HeteroscedasticDecoder initialized")

        # Variance regularization settings
        self.var_reg_weight = config.get("var_reg_weight", 0.1)  # Weight for variance matching
        self.target_var = config.get("target_var", 0.006)  # Target average variance (GT variance)
        self.min_var = config.get("min_var", 0.001)  # Minimum allowed variance

        # Variance ratio regularization (encourages regime-dependent variance)
        self.var_ratio_weight = config.get("var_ratio_weight", 0.0)  # Weight for ratio loss
        self.target_var_ratio = config.get("target_var_ratio", 2.0)  # Target high/low vol ratio

        # Separate context-based variance network (independent of z)
        # This allows variance to be regime-dependent at inference time
        self.use_context_variance = config.get("use_context_variance", True)
        if self.use_context_variance:
            latent_dim = config["latent_dim"]
            feat_dim = config["feat_dim"]
            output_size = feat_dim[0] * feat_dim[1]  # 25 for 5x5 grid
            self.context_variance_net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_size),
            ).to(self.device)
            # Initialize to output reasonable log_var (~GT variance)
            # GT var ≈ 0.01, so log(0.01) ≈ -4.6
            # For 90% CI coverage, we need ~10x GT var, so log(0.1) ≈ -2.3
            # Start small and let the model learn to increase variance
            nn.init.zeros_(self.context_variance_net[-1].weight)
            nn.init.constant_(self.context_variance_net[-1].bias, -4.0)  # var ≈ 0.018
            print("✓ Context-based variance network initialized")

    def gaussian_nll_loss(self, mean, log_var, target):
        """
        Gaussian Negative Log-Likelihood loss with minimum variance floor.

        NLL = 0.5 * (log(var) + (x - mu)^2 / var)
            = 0.5 * (log_var + (x - mu)^2 / exp(log_var))
        """
        # Apply minimum variance floor
        log_var = torch.clamp(log_var, min=np.log(self.min_var))
        variance = torch.exp(log_var)
        nll = 0.5 * (log_var + (target - mean) ** 2 / variance)
        return nll.mean()

    def variance_regularization_loss(self, log_var):
        """
        Regularization to prevent variance collapse.

        Penalizes deviation from target average variance.
        """
        pred_var = torch.exp(log_var).mean()
        # L2 penalty on log-scale (more stable)
        log_pred_var = torch.log(pred_var)
        log_target_var = np.log(self.target_var)
        return (log_pred_var - log_target_var) ** 2

    def variance_ratio_loss(self, log_var, surface_context):
        """
        Loss that encourages regime-dependent variance.

        Uses ATM IV (center grid point of last context day) to split samples
        into high/low vol regimes, then penalizes if variance ratio is too low.

        Args:
            log_var: (B, H, 5, 5) - predicted log variance
            surface_context: (B, C, 5, 5) - context surfaces

        Returns:
            ratio_loss: scalar loss that pushes high_var/low_var toward target
        """
        B = log_var.shape[0]
        if B < 4:  # Need minimum batch size for splitting
            return torch.tensor(0.0, device=log_var.device)

        # Get ATM IV from last context day (center point = [2,2])
        atm_iv = surface_context[:, -1, 2, 2]  # (B,)

        # Compute per-sample mean variance
        pred_var = torch.exp(log_var)  # (B, H, 5, 5)
        sample_var = pred_var.mean(dim=(1, 2, 3))  # (B,)

        # Split by median ATM IV
        median_iv = atm_iv.median()
        high_mask = atm_iv >= median_iv
        low_mask = atm_iv < median_iv

        # Compute variance for each regime
        high_var = sample_var[high_mask].mean() if high_mask.sum() > 0 else sample_var.mean()
        low_var = sample_var[low_mask].mean() if low_mask.sum() > 0 else sample_var.mean()

        # Avoid division by zero
        low_var = torch.clamp(low_var, min=1e-8)

        # Current ratio
        current_ratio = high_var / low_var

        # Loss: penalize if ratio is below target
        # Using max(0, target - current) to only push ratio UP, not down
        target = torch.tensor(self.target_var_ratio, device=log_var.device)
        ratio_loss = torch.relu(target - current_ratio) ** 2

        return ratio_loss, current_ratio.detach()

    def forward(self, x):
        """
        Forward pass with heteroscedastic decoder.

        Returns:
            surface_mean: (B, H, 5, 5) - mean prediction
            surface_logvar: (B, H, 5, 5) - log-variance
            z_mean: (B, T, latent_dim) - posterior mean
            z_log_var: (B, T, latent_dim) - posterior log-variance
            z: (B, T, latent_dim) - sampled latent
            [ex_feats_reconstruction]: optional if ex_feats present
        """
        surface = x["surface"]
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - self.horizon

        # Encode full sequence
        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            encoder_input["ex_feats"] = x["ex_feats"]

        z_mean, z_log_var, z = self.encoder(encoder_input)

        # Context encoding
        ctx_input = {"surface": surface[:, :C, :, :]}
        if "ex_feats" in x:
            ctx_input["ex_feats"] = x["ex_feats"][:, :C, :]
        ctx_embedding = self.ctx_encoder(ctx_input)

        # Pad context embedding for decoder
        if self.config["compress_context"]:
            ctx_embedding_dim = self.config["latent_dim"]
        else:
            ctx_embedding_dim = self.config["mem_hidden"]

        # Create decoder input: [z, ctx_embedding]
        # For context steps: use actual context embedding
        # For prediction steps: replicate last context embedding (for variance conditioning)
        ctx_padded = torch.zeros((B, T, ctx_embedding_dim), device=self.device, dtype=z.dtype)
        ctx_padded[:, :C, :] = ctx_embedding
        # Replicate last context embedding for prediction steps
        # This allows decoder to learn context-dependent variance
        ctx_padded[:, C:, :] = ctx_embedding[:, -1:, :].expand(-1, T - C, -1)
        decoder_input = torch.cat([z, ctx_padded], dim=-1)

        # Decode to (mean, log_var)
        if self.config["ex_feats_dim"] > 0:
            surface_mean, surface_logvar, ex_feats_reconstruction = self.decoder(decoder_input)
            # Only return future predictions
            return (surface_mean[:, C:, :, :], surface_logvar[:, C:, :, :],
                    ex_feats_reconstruction[:, C:, :], z_mean, z_log_var, z)
        else:
            surface_mean, surface_logvar = self.decoder(decoder_input)

            # Override log_var with context-based variance (independent of z)
            if self.use_context_variance:
                # Get context summary (last context embedding)
                context_summary = ctx_embedding[:, -1, :]  # (B, latent_dim)
                # Compute variance from context only
                ctx_logvar = self.context_variance_net(context_summary)  # (B, 25)
                ctx_logvar = ctx_logvar.view(B, 1, self.config["feat_dim"][0], self.config["feat_dim"][1])
                # Expand to horizon (already the right shape, no need to slice)
                ctx_logvar = ctx_logvar.expand(-1, self.horizon, -1, -1)
                # Return directly - ctx_logvar already has the right shape (B, H, 5, 5)
                return surface_mean[:, C:, :, :], ctx_logvar, z_mean, z_log_var, z
            else:
                return surface_mean[:, C:, :, :], surface_logvar[:, C:, :, :], z_mean, z_log_var, z

    def train_step(self, x, optimizer: torch.optim.Optimizer, scaler=None):
        """
        Training step with Gaussian NLL loss.
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

        x_device = {"surface": surface.to(self.device)}
        if "ex_feats" in x:
            x_device["ex_feats"] = ex_feats.to(self.device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', dtype=torch.bfloat16):
            if "ex_feats" in x:
                surface_mean, surface_logvar, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x_device)
            else:
                surface_mean, surface_logvar, z_mean, z_log_var, z = self.forward(x_device)

            # MSE loss for mean accuracy (essential - prevents variance from masking errors)
            mse_loss = F.mse_loss(surface_mean, surface_real)

            # Gaussian NLL for variance learning (teaches variance to match errors)
            nll_loss = self.gaussian_nll_loss(surface_mean.detach(), surface_logvar, surface_real)

            # Combined: MSE ensures mean accuracy, NLL calibrates variance
            # Note: detach mean in NLL so variance learns from actual errors, not from gradient flow
            mse_weight = self.config.get("mse_weight", 1.0)
            nll_weight = self.config.get("nll_weight", 0.1)
            recon_loss = mse_weight * mse_loss + nll_weight * nll_loss

            if "ex_feats" in x:
                if self.config["ex_loss_on_ret_only"]:
                    ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                    ex_feats_real = ex_feats_real[:, :, :1]
                re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
                reconstruction_error = recon_loss + self.config["re_feat_weight"] * re_ex_feats
            else:
                reconstruction_error = recon_loss
                re_ex_feats = torch.tensor(0.0, device=self.device)

            # KL loss with full covariance prior
            ctx_surface = x_device["surface"][:, :C, :, :]
            ctx_encoder_input = {"surface": ctx_surface}
            if "ex_feats" in x:
                ctx_encoder_input["ex_feats"] = x_device["ex_feats"][:, :C, :]

            ctx_embedding = self.ctx_encoder(ctx_encoder_input)
            context_summary = ctx_embedding[:, -1, :]

            mu_p, Sigma_p = self.full_cov_prior.get_prior_params(context_summary, horizon=self.horizon)

            kl_loss = kl_divergence_full_covariance(
                z_mean[:, C:, :],
                z_log_var[:, C:, :],
                mu_p,
                Sigma_p
            )

            # Variance regularization to prevent collapse
            var_reg_loss = self.variance_regularization_loss(surface_logvar)

            # Variance ratio loss to encourage regime-dependent variance
            if self.var_ratio_weight > 0:
                ratio_loss, current_ratio = self.variance_ratio_loss(surface_logvar, x_device["surface"][:, :C, :, :])
            else:
                ratio_loss = torch.tensor(0.0, device=self.device)
                current_ratio = torch.tensor(1.0)

            total_loss = (reconstruction_error + self.kl_weight * kl_loss +
                          self.var_reg_weight * var_reg_loss + self.var_ratio_weight * ratio_loss)

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

        # Compute mean predicted variance for monitoring
        with torch.no_grad():
            pred_var = torch.exp(surface_logvar).mean()

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,  # Mean accuracy loss
            "nll_loss": nll_loss,  # Variance calibration loss
            "re_ex_feats": re_ex_feats,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
            "var_reg_loss": var_reg_loss,  # Variance regularization
            "var_ratio_loss": ratio_loss,  # Variance ratio regularization
            "pred_var": pred_var,  # Monitor learned variance
            "pred_var_ratio": current_ratio,  # Monitor high/low var ratio
        }

    def get_surface_given_conditions(self, c: dict, z: torch.Tensor = None,
                                     mu=0, std=1, horizon=None, prior_mode="full_cov",
                                     sample_from_decoder=True):
        """
        Generate surfaces using heteroscedastic decoder.

        Args:
            c: Context dictionary
            z: Pre-generated latents (optional)
            horizon: Forecast horizon
            prior_mode: Must be "full_cov"
            sample_from_decoder: If True, sample from N(mean, var). If False, return mean only.

        Returns:
            If sample_from_decoder=True: sampled surface
            If sample_from_decoder=False: (mean, std) tuple
        """
        if prior_mode != "full_cov":
            raise ValueError(f"CVAEHeteroscedastic only supports prior_mode='full_cov'")

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

        ctx_embedding = self.ctx_encoder(ctx_encoder_input)
        context_summary = ctx_embedding[:, -1, :]

        # Sample z from prior
        z_future = self.full_cov_prior.sample(context_summary, horizon)

        # Prepare decoder input
        ctx_embedding_dim = ctx_embedding.shape[2]
        decoder_ctx = torch.zeros((B, horizon, ctx_embedding_dim), device=self.device, dtype=ctx_embedding.dtype)
        decoder_input = torch.cat([z_future, decoder_ctx], dim=-1)

        # Decode to (mean, log_var)
        if self.config["ex_feats_dim"] > 0:
            surface_mean, surface_logvar, ex_pred = self.decoder(decoder_input)
        else:
            surface_mean, surface_logvar = self.decoder(decoder_input)

        if sample_from_decoder:
            # Sample from decoded distribution
            std = torch.exp(0.5 * surface_logvar)
            surface_sample = surface_mean + std * torch.randn_like(surface_mean)

            if "ex_feats" in c:
                return surface_sample, ex_pred
            else:
                return surface_sample
        else:
            # Return mean and std (for CI computation)
            std = torch.exp(0.5 * surface_logvar)
            return surface_mean, std

    def generate_samples(self, context: dict, n_samples: int = 1000, horizon: int = None):
        """
        Generate multiple samples for uncertainty quantification.

        Returns:
            samples: (n_samples, horizon, 5, 5)
        """
        if horizon is None:
            horizon = self.horizon

        samples = []
        for _ in range(n_samples):
            sample = self.get_surface_given_conditions(
                context, horizon=horizon, sample_from_decoder=True
            )
            if isinstance(sample, tuple):
                sample = sample[0]  # Surface only
            samples.append(sample.squeeze(0))  # Remove batch dim

        return torch.stack(samples)  # (n_samples, horizon, 5, 5)

    def get_prediction_interval(self, context: dict, horizon: int = None, alpha: float = 0.10):
        """
        Get prediction interval from decoded distribution.

        For Gaussian: CI = mean ± z_{1-alpha/2} * std

        Args:
            context: Context dictionary
            horizon: Forecast horizon
            alpha: 1 - coverage level (e.g., 0.10 for 90% CI)

        Returns:
            mean: (horizon, 5, 5) - point prediction
            lower: (horizon, 5, 5) - lower bound
            upper: (horizon, 5, 5) - upper bound
        """
        import scipy.stats as stats

        if horizon is None:
            horizon = self.horizon

        # Get mean and std from decoder (deterministic, z from prior mean)
        ctx_surface = context["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)

        ctx_encoder_input = {"surface": ctx_surface}
        if "ex_feats" in context:
            ctx_ex_feats = context["ex_feats"].to(self.device)
            if len(ctx_ex_feats.shape) == 2:
                ctx_ex_feats = ctx_ex_feats.unsqueeze(0)
            ctx_encoder_input["ex_feats"] = ctx_ex_feats

        ctx_embedding = self.ctx_encoder(ctx_encoder_input)
        context_summary = ctx_embedding[:, -1, :]

        # Use prior mean (deterministic)
        mu_p, _ = self.full_cov_prior.get_prior_params(context_summary, horizon=horizon)

        ctx_embedding_dim = ctx_embedding.shape[2]
        decoder_ctx = torch.zeros((1, horizon, ctx_embedding_dim), device=self.device, dtype=ctx_embedding.dtype)
        decoder_input = torch.cat([mu_p, decoder_ctx], dim=-1)

        if self.config["ex_feats_dim"] > 0:
            surface_mean, surface_logvar, _ = self.decoder(decoder_input)
        else:
            surface_mean, surface_logvar = self.decoder(decoder_input)

        std = torch.exp(0.5 * surface_logvar)

        # Compute CI bounds
        z_score = stats.norm.ppf(1 - alpha / 2)  # e.g., 1.645 for 90% CI
        lower = surface_mean - z_score * std
        upper = surface_mean + z_score * std

        return (
            surface_mean.squeeze(0).cpu(),
            lower.squeeze(0).cpu(),
            upper.squeeze(0).cpu()
        )
