"""
Predictor modules for Two-Stage CVAE

These modules are trained in Stage 2 with frozen autoencoder.
They predict future ctx_emb and z from context surfaces only.

Architecture: Single LSTM with autoregressive generation
- Encode context: Conv → Flatten → LSTM
- Autoregressive: Continue LSTM for H steps, feeding output back via feedback_proj
- Per-step projection: Linear(8→output_dim)

LatentPredictor: context → (z_mean, z_logvar) for horizon positions
ContextPredictor: context → ctx_emb for horizon positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple


class BasePredictor(nn.Module):
    """Base class for predictors with shared surface embedding."""

    def __init__(self, config: dict):
        super(BasePredictor, self).__init__()
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def _build_surface_embedding(self, hidden_layers, feat_dim):
        """Build surface embedding layers using Conv2D (mirrors main model).

        Args:
            hidden_layers: List of channel sizes, e.g., [2, 4, 2]
            feat_dim: Tuple of (H, W), e.g., (5, 5)

        Returns:
            conv_layers: nn.Sequential of Conv2D layers
            flatten_dim: Output dimension after flatten (last_channel * H * W)
        """
        layers = []
        in_channels = 1
        for i, out_channels in enumerate(hidden_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels

        conv_layers = nn.Sequential(*layers)
        # Output: (B, last_channel, H, W) → flatten to (B, last_channel * H * W)
        flatten_dim = hidden_layers[-1] * feat_dim[0] * feat_dim[1]
        return conv_layers, flatten_dim

    def _build_lstm(self, input_size, hidden_size, num_layers, dropout=0.1):
        """Build LSTM memory."""
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )


class LatentPredictor(BasePredictor):
    """
    Predicts z distribution (mean and logvar) for future positions from context.

    Input: raw surfaces[:C] (B, C, H, W)
    Output: (z_mean, z_logvar) for positions C:C+H (B, H, latent_dim)

    Architecture:
    - Conv surface embedding: Conv(1→2→4→2) + Flatten → 50 dims
    - Single LSTM for encoding and autoregressive generation
    - Feedback projection: 8 → 50 (to feed LSTM output back as input)
    - Per-step projection: Linear(8→16) for z_mean and z_logvar
    """

    def __init__(self, config: dict):
        super(LatentPredictor, self).__init__(config)

        feat_dim = config.get("feat_dim", (5, 5))
        latent_dim = config["latent_dim"]
        hidden_size = config.get("hidden_size", 50)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.2)
        surface_hidden = config.get("surface_hidden", [2, 4, 2])
        max_horizon = config.get("max_horizon", 30)

        # Surface embedding: Conv layers
        self.surface_conv, embed_dim = self._build_surface_embedding(
            surface_hidden, feat_dim
        )

        # Single LSTM for both encoding and autoregressive generation
        self.lstm = self._build_lstm(embed_dim, hidden_size, num_layers, dropout)

        # Automatic feedback projection (only if hidden != embed)
        if hidden_size != embed_dim:
            self.feedback_proj = nn.Linear(hidden_size, embed_dim)
            self.use_feedback_proj = True
        else:
            self.use_feedback_proj = False

        # Per-step output projection
        self.z_mean_linear = nn.Linear(hidden_size, latent_dim)
        self.z_logvar_linear = nn.Linear(hidden_size, latent_dim)

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.max_horizon = max_horizon
        self.to(self.device)

    def forward(self, context_surfaces: torch.Tensor, horizon: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict z distribution for future positions.

        Args:
            context_surfaces: (B, C, H, W) - context surfaces
            horizon: Number of future positions to predict. If None, uses max_horizon.

        Returns:
            z_mean: (B, horizon, latent_dim)
            z_logvar: (B, horizon, latent_dim)
        """
        if horizon is None:
            horizon = self.max_horizon

        context_surfaces = context_surfaces.to(self.device)
        B, C = context_surfaces.shape[:2]
        H_surf, W_surf = context_surfaces.shape[2], context_surfaces.shape[3]

        # === ENCODE CONTEXT ===
        # Embed each surface through Conv layers
        surfaces_flat = context_surfaces.reshape(B * C, 1, H_surf, W_surf)
        conv_out = self.surface_conv(surfaces_flat)  # (B*C, last_channel, H, W)
        embeddings = conv_out.reshape(B * C, -1)  # (B*C, embed_dim)
        embeddings = embeddings.reshape(B, C, -1)  # (B, C, embed_dim)

        # Process context through LSTM
        ctx_outputs, (h_n, c_n) = self.lstm(embeddings)  # ctx_outputs: (B, C, hidden_size)

        # === AUTOREGRESSIVE GENERATION ===
        outputs = []
        # Start with last context output
        prev_output = ctx_outputs[:, -1, :]  # (B, hidden_size)

        for t in range(horizon):
            # Project output back to LSTM input space (if needed)
            if self.use_feedback_proj:
                lstm_input = self.feedback_proj(prev_output).unsqueeze(1)  # (B, 1, embed_dim)
            else:
                lstm_input = prev_output.unsqueeze(1)  # Direct feedback

            # Run one LSTM step
            output, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))  # output: (B, 1, hidden_size)
            prev_output = output.squeeze(1)  # (B, hidden_size)
            outputs.append(prev_output)

        # Stack outputs: (B, H, hidden_size)
        horizon_outputs = torch.stack(outputs, dim=1)

        # Per-step projection
        z_mean = self.z_mean_linear(horizon_outputs)      # (B, H, latent_dim)
        z_logvar = self.z_logvar_linear(horizon_outputs)  # (B, H, latent_dim)

        return z_mean, z_logvar


class LatentPredictorCov(BasePredictor):
    """
    Predicts z with AR(1) covariance structure for temporally correlated sampling.

    Key difference from LatentPredictor:
    - Learns global AR(1) parameters (rho, sigma) for covariance structure
    - sample_z() produces correlated z via Cholesky factorization
    - Addresses factorized prior issue that causes ACF mismatch

    Input: raw surfaces[:C] (B, C, H, W)
    Output: (z_mean, z_logvar) for positions C:C+H (B, H, latent_dim)
            + sample_z() for correlated sampling

    AR(1) Covariance Structure:
        Sigma[i,j] = sigma^2 * rho^|i-j|

    Sampling via Cholesky:
        L = cholesky(Sigma)
        z = mu + L @ eps, where eps ~ N(0, I)
    """

    def __init__(self, config: dict):
        super(LatentPredictorCov, self).__init__(config)

        feat_dim = config.get("feat_dim", (5, 5))
        latent_dim = config["latent_dim"]
        hidden_size = config.get("hidden_size", 50)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.2)
        surface_hidden = config.get("surface_hidden", [2, 4, 2])
        max_horizon = config.get("max_horizon", 30)

        # AR(1) covariance parameters (learnable)
        # rho_raw is passed through sigmoid to get rho in (0, 0.99)
        init_rho = config.get("init_rho", 0.8)
        init_sigma = config.get("init_sigma", 1.0)
        # Transform init_rho to raw space: rho = sigmoid(rho_raw) * 0.99
        # So rho_raw = logit(rho / 0.99)
        rho_scaled = init_rho / 0.99
        rho_raw_init = torch.log(torch.tensor(rho_scaled / (1 - rho_scaled + 1e-8)))
        self.rho_raw = nn.Parameter(rho_raw_init)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma)))

        # Surface embedding: Conv layers
        self.surface_conv, embed_dim = self._build_surface_embedding(
            surface_hidden, feat_dim
        )

        # Single LSTM for both encoding and autoregressive generation
        self.lstm = self._build_lstm(embed_dim, hidden_size, num_layers, dropout)

        # Automatic feedback projection (only if hidden != embed)
        if hidden_size != embed_dim:
            self.feedback_proj = nn.Linear(hidden_size, embed_dim)
            self.use_feedback_proj = True
        else:
            self.use_feedback_proj = False

        # Per-step output projection
        self.z_mean_linear = nn.Linear(hidden_size, latent_dim)
        self.z_logvar_linear = nn.Linear(hidden_size, latent_dim)

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.max_horizon = max_horizon
        self._cholesky_cache = {}  # Cache Cholesky factors by horizon
        self.to(self.device)

    @property
    def rho(self) -> torch.Tensor:
        """Get AR(1) correlation coefficient in (0, 0.99)."""
        return torch.sigmoid(self.rho_raw) * 0.99

    @property
    def sigma(self) -> torch.Tensor:
        """Get innovation standard deviation."""
        return torch.exp(self.log_sigma)

    def _build_ar1_covariance(self, horizon: int) -> torch.Tensor:
        """
        Build AR(1) covariance matrix.

        Args:
            horizon: Number of timesteps

        Returns:
            Sigma: (H, H) covariance matrix where Sigma[i,j] = sigma^2 * rho^|i-j|
        """
        idx = torch.arange(horizon, device=self.device, dtype=torch.float32)
        # |i - j| matrix
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        # Sigma[i,j] = sigma^2 * rho^|i-j|
        Sigma = (self.sigma ** 2) * (self.rho ** diff)
        return Sigma

    def _build_ar1_cholesky(self, horizon: int) -> torch.Tensor:
        """
        Build Cholesky factor of AR(1) covariance matrix.

        Uses caching for efficiency during generation.

        Args:
            horizon: Number of timesteps

        Returns:
            L: (H, H) lower triangular Cholesky factor
        """
        # Build covariance and compute Cholesky
        Sigma = self._build_ar1_covariance(horizon)
        # Add small jitter for numerical stability
        Sigma = Sigma + 1e-6 * torch.eye(horizon, device=self.device)
        L = torch.linalg.cholesky(Sigma)
        return L

    def forward(self, context_surfaces: torch.Tensor, horizon: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict z distribution for future positions.

        Args:
            context_surfaces: (B, C, H, W) - context surfaces
            horizon: Number of future positions to predict. If None, uses max_horizon.

        Returns:
            z_mean: (B, horizon, latent_dim)
            z_logvar: (B, horizon, latent_dim) - per-step marginal variance (for compatibility)
        """
        if horizon is None:
            horizon = self.max_horizon

        context_surfaces = context_surfaces.to(self.device)
        B, C = context_surfaces.shape[:2]
        H_surf, W_surf = context_surfaces.shape[2], context_surfaces.shape[3]

        # === ENCODE CONTEXT ===
        surfaces_flat = context_surfaces.reshape(B * C, 1, H_surf, W_surf)
        conv_out = self.surface_conv(surfaces_flat)
        embeddings = conv_out.reshape(B * C, -1)
        embeddings = embeddings.reshape(B, C, -1)

        # Process context through LSTM
        ctx_outputs, (h_n, c_n) = self.lstm(embeddings)

        # === AUTOREGRESSIVE GENERATION ===
        outputs = []
        prev_output = ctx_outputs[:, -1, :]

        for t in range(horizon):
            if self.use_feedback_proj:
                lstm_input = self.feedback_proj(prev_output).unsqueeze(1)
            else:
                lstm_input = prev_output.unsqueeze(1)

            output, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
            prev_output = output.squeeze(1)
            outputs.append(prev_output)

        horizon_outputs = torch.stack(outputs, dim=1)

        # Per-step projection
        z_mean = self.z_mean_linear(horizon_outputs)
        z_logvar = self.z_logvar_linear(horizon_outputs)

        return z_mean, z_logvar

    def sample_z(self, z_mean: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample correlated z using Cholesky factorization of AR(1) covariance.

        This is the key method that produces temporally correlated latent samples,
        addressing the factorized prior issue.

        Args:
            z_mean: (B, H, D) predicted mean from forward()
            n_samples: Number of samples to generate per batch element

        Returns:
            z_samples: (n_samples, B, H, D) correlated latent samples
        """
        B, H, D = z_mean.shape

        # Get Cholesky factor
        L = self._build_ar1_cholesky(H)  # (H, H)

        samples = []
        for _ in range(n_samples):
            # Independent standard normal
            eps = torch.randn(B, H, D, device=self.device)

            # Apply Cholesky transformation per latent dimension
            # z_d = mu_d + L @ eps_d
            # Using einsum: 'ij,bjd->bid' means L[i,j] @ eps[b,j,d] -> z[b,i,d]
            z = z_mean + torch.einsum('ij,bjd->bid', L, eps)
            samples.append(z)

        # Stack: (n_samples, B, H, D)
        return torch.stack(samples, dim=0)

    def sample_z_single(self, z_mean: torch.Tensor) -> torch.Tensor:
        """
        Sample single correlated z (convenience method).

        Args:
            z_mean: (B, H, D) predicted mean

        Returns:
            z: (B, H, D) single correlated sample
        """
        return self.sample_z(z_mean, n_samples=1)[0]

    def get_covariance_params(self) -> dict:
        """Return current AR(1) covariance parameters."""
        return {
            "rho": self.rho.item(),
            "sigma": self.sigma.item(),
        }


class ContextPredictor(BasePredictor):
    """
    Predicts ctx_embedding for future positions from context.

    Input: raw surfaces[:C] (B, C, H, W)
    Output: ctx_emb for positions C:C+H (B, H, ctx_embedding_dim)

    Architecture:
    - Conv surface embedding: Conv(1→2→4→2) + Flatten → 50 dims
    - Single LSTM for encoding and autoregressive generation
    - Feedback projection: 8 → 50 (to feed LSTM output back as input)
    - Per-step projection: Linear(8→3) for ctx_emb
    """

    def __init__(self, config: dict):
        super(ContextPredictor, self).__init__(config)

        feat_dim = config.get("feat_dim", (5, 5))
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        hidden_size = config.get("hidden_size", 50)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.2)
        surface_hidden = config.get("surface_hidden", [2, 4, 2])
        max_horizon = config.get("max_horizon", 30)

        # Surface embedding: Conv layers
        self.surface_conv, embed_dim = self._build_surface_embedding(
            surface_hidden, feat_dim
        )

        # Single LSTM for both encoding and autoregressive generation
        self.lstm = self._build_lstm(embed_dim, hidden_size, num_layers, dropout)

        # Automatic feedback projection (only if hidden != embed)
        if hidden_size != embed_dim:
            self.feedback_proj = nn.Linear(hidden_size, embed_dim)
            self.use_feedback_proj = True
        else:
            self.use_feedback_proj = False

        # Per-step output projection
        self.ctx_linear = nn.Linear(hidden_size, ctx_embedding_dim)

        self.ctx_embedding_dim = ctx_embedding_dim
        self.hidden_size = hidden_size
        self.max_horizon = max_horizon
        self.to(self.device)

    def forward(self, context_surfaces: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """
        Predict ctx_embedding for future positions.

        Args:
            context_surfaces: (B, C, H, W) - context surfaces
            horizon: Number of future positions to predict. If None, uses max_horizon.

        Returns:
            ctx_emb: (B, horizon, ctx_embedding_dim)
        """
        if horizon is None:
            horizon = self.max_horizon

        context_surfaces = context_surfaces.to(self.device)
        B, C = context_surfaces.shape[:2]
        H_surf, W_surf = context_surfaces.shape[2], context_surfaces.shape[3]

        # === ENCODE CONTEXT ===
        # Embed each surface through Conv layers
        surfaces_flat = context_surfaces.reshape(B * C, 1, H_surf, W_surf)
        conv_out = self.surface_conv(surfaces_flat)  # (B*C, last_channel, H, W)
        embeddings = conv_out.reshape(B * C, -1)  # (B*C, embed_dim)
        embeddings = embeddings.reshape(B, C, -1)  # (B, C, embed_dim)

        # Process context through LSTM
        ctx_outputs, (h_n, c_n) = self.lstm(embeddings)

        # === AUTOREGRESSIVE GENERATION ===
        outputs = []
        prev_output = ctx_outputs[:, -1, :]  # (B, hidden_size)

        for t in range(horizon):
            # Project output back to LSTM input space (if needed)
            if self.use_feedback_proj:
                lstm_input = self.feedback_proj(prev_output).unsqueeze(1)  # (B, 1, embed_dim)
            else:
                lstm_input = prev_output.unsqueeze(1)  # Direct feedback
            output, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
            prev_output = output.squeeze(1)  # (B, hidden_size)
            outputs.append(prev_output)

        horizon_outputs = torch.stack(outputs, dim=1)  # (B, H, hidden_size)
        ctx_emb = self.ctx_linear(horizon_outputs)  # (B, H, ctx_embedding_dim)

        return ctx_emb


class PredictorPair(nn.Module):
    """
    Wrapper that combines LatentPredictor and ContextPredictor.

    Provides convenient interface for Stage 2 training and inference.
    """

    def __init__(self, config: dict):
        super(PredictorPair, self).__init__()
        self.latent_predictor = LatentPredictor(config)
        self.context_predictor = ContextPredictor(config)
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, context_surfaces: torch.Tensor, horizon: int = None):
        """
        Predict both z distribution and ctx_emb for future positions.

        Args:
            context_surfaces: (B, C, H, W)
            horizon: Number of future positions

        Returns:
            z_mean: (B, H, latent_dim)
            z_logvar: (B, H, latent_dim)
            ctx_emb: (B, H, ctx_embedding_dim)
        """
        z_mean, z_logvar = self.latent_predictor(context_surfaces, horizon)
        ctx_emb = self.context_predictor(context_surfaces, horizon)
        return z_mean, z_logvar, ctx_emb

    def train_step(self, model, batch: dict, optimizer: torch.optim.Optimizer):
        """
        Train predictors with frozen autoencoder.

        Args:
            model: CVAETwoStage model (frozen)
            batch: dict with "surface" (B, T, H, W)
            optimizer: Optimizer for predictors only

        Returns:
            dict with loss components
        """
        surface = batch["surface"].to(self.device)
        if len(surface.shape) == 3:
            surface = surface.unsqueeze(0)

        B, T = surface.shape[:2]
        C = self.config.get("context_len", 30)
        H = T - C  # Horizon

        # Get targets from frozen autoencoder
        with torch.no_grad():
            ctx_emb_target = model.get_ctx_embedding(batch)  # (B, T, ctx_dim)
            z_mean_target = model.get_z_mean(batch)  # (B, T, latent_dim)

        # Predict from context only
        context = surface[:, :C]
        z_pred_mean, z_pred_logvar = self.latent_predictor(context, H)
        ctx_pred = self.context_predictor(context, H)

        # Losses on future positions (MSE with detached targets)
        loss_z = F.mse_loss(z_pred_mean, z_mean_target[:, C:].detach())
        loss_ctx = F.mse_loss(ctx_pred, ctx_emb_target[:, C:].detach())

        # Optional: NLL loss for z variance (encourages learning uncertainty)
        # loss_z_nll = gaussian_nll(z_mean_target[:, C:], z_pred_mean, z_pred_logvar)

        total_loss = loss_z + loss_ctx

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss,
            "loss_z": loss_z,
            "loss_ctx": loss_ctx,
        }

    def get_predictors_dict(self):
        """Return dict for use with model.get_surface_given_conditions()."""
        return {
            "latent": self.latent_predictor,
            "context": self.context_predictor,
        }


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.

    NLL = 0.5 * (logvar + (target - mean)^2 / exp(logvar))

    Args:
        target: Ground truth values
        mean: Predicted mean
        logvar: Predicted log variance

    Returns:
        Scalar loss
    """
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean).pow(2) / var)
    return nll.mean()
