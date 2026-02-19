"""
Block-AR Conditional DDPM with MCVD Multi-Task Training + Diffusion Forcing.

Combines:
- GRU encoder for variable-length conditioning
- BiGRU denoiser for per-frame noise prediction
- MCVD 4-task masking (forward/backward/interpolation/unconditional)
- Task-adaptive per-frame noise schedules (Diffusion Forcing)
- PYoCo correlated noise for temporal coherence (ICCV 2023)
- Pyramid sampling schedule following the actual DF paper
- Block-autoregressive generation
"""

from dataclasses import dataclass
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.block_ar.gru_encoder import GRUEncoder, EncoderConfig
from diffusion.block_ar.bigru_denoiser import BiGRUDenoiser, DenoiserConfig
from diffusion.block_ar.masking import sample_mcvd_masks
from diffusion.block_ar.noise_schedules import sample_batch_task_adaptive_noise
from diffusion.ddpm_scheduler import DDPMScheduler


# IV normalization constants — must match train_ddpm_poc.py
IV_MIN = 0.0
IV_MAX = 1.0


def denormalize_iv(iv_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize IV from [-1, 1] to [0, 1]."""
    return (iv_norm + 1.0) / 2.0 * (IV_MAX - IV_MIN) + IV_MIN


def sample_pyoco_noise(shape: tuple, rho: float, device: torch.device) -> torch.Tensor:
    """Sample PYoCo temporally-correlated noise (Ge et al., ICCV 2023).

    Each frame's noise is marginal N(0, I), but adjacent frames share a
    common noise component controlled by rho:
        epsilon_t = rho * epsilon_shared + sqrt(1 - rho^2) * epsilon_t_indep

    Args:
        shape: (B, T, H, W) — target noise shape
        rho: correlation coefficient in [0, 1]. 0 = independent, 1 = fully shared.
        device: torch device

    Returns:
        noise: (B, T, H, W) with each frame marginal N(0, I)
    """
    B, T, H, W = shape
    if rho == 0.0:
        return torch.randn(shape, device=device)
    eps_shared = torch.randn(B, 1, H, W, device=device)
    eps_indep = torch.randn(B, T, H, W, device=device)
    return rho * eps_shared + math.sqrt(1.0 - rho ** 2) * eps_indep


@dataclass
class BlockARConfig:
    # Data
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5
    block_size: int = 10

    # Encoder
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 64
    cond_aug_sigma: float = 0.0
    encoder_dropout: float = 0.1

    # Denoiser
    denoiser_type: str = "bigru"  # "bigru" or "conv3d"
    bigru_hidden_dim: int = 128
    pos_embed_dim: int = 16
    noise_embed_dim: int = 16
    denoiser_dropout: float = 0.1

    # Conv3D denoiser params (only used when denoiser_type="conv3d")
    conv3d_base_channels: int = 32
    conv3d_n_res_blocks: int = 4
    conv3d_groups: int = 8
    conv3d_noise_embed_dim: int = 64

    # Diffusion
    n_steps: int = 100
    schedule: str = "cosine"

    # MCVD
    p_mask: float = 0.2
    jitter_std: float = 0.15

    # Uniform-t noise (one scalar t per block instead of per-frame task-adaptive)
    use_uniform_noise: bool = False

    # Sampling mode: "pyramid" (DF staggered) or "uniform" (standard DDPM reverse)
    sampling_mode: str = "pyramid"

    # PYoCo correlated noise (0.0 = independent, 1.0 = fully shared)
    noise_rho: float = 0.5

    # Loss function: "mse" or "huber" (Huber/SmoothL1 preserves tails better)
    loss_type: str = "mse"
    huber_delta: float = 0.1  # Huber threshold — smaller = more L1-like for large errors

    # Regime conditioning (hierarchical sampling)
    n_regimes: int = 5
    regime_embed_dim: int = 32
    regime_loss_weight: float = 1.0
    use_regime_conditioning: bool = False

    # Sampling
    max_residual_timestep: int = 20

    # Global horizon-dependent residual noise for growing uncertainty.
    # Frame at global horizon h stops denoising at t_min = max_global_residual * h / (future_len - 1).
    # 0 = denoise all frames to t=0 (no residual, backward compat).
    # 10 = recommended starting point (~0.09 IV std residual at h=29).
    max_global_residual: int = 0


class _DenoiserAdapter(nn.Module):
    """Wraps denoiser to match DDPMScheduler's model(x_t, t, condition) interface."""

    def __init__(
        self,
        denoiser: nn.Module,
        condition: torch.Tensor,
        positions: torch.Tensor,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.condition = condition
        self.positions = positions

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition_unused: torch.Tensor
    ) -> torch.Tensor:
        # x_t: (B, T, 5, 5) -> flatten to (B, T, 25)
        B, T, H, W = x_t.shape
        x_flat = x_t.reshape(B, T, H * W)

        # Expand scalar t (B,) to per-frame (B, T) for denoiser interface
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(B, T)

        # Call denoiser
        noise_pred_flat = self.denoiser(
            x_flat, self.condition, self.positions, t
        )  # (B, T, 25)

        # Reshape back to (B, T, 5, 5)
        return noise_pred_flat.reshape(B, T, H, W)


class ConditionalBlockARDDPM(nn.Module):
    """Block-AR Conditional DDPM with MCVD multi-task training."""

    def __init__(self, config: BlockARConfig):
        super().__init__()
        self.config = config

        self.encoder = GRUEncoder(
            EncoderConfig(
                input_dim=config.surface_h * config.surface_w,
                gru_hidden_dim=config.gru_hidden_dim,
                bottleneck_dim=config.bottleneck_dim,
                cond_aug_sigma=config.cond_aug_sigma,
                dropout=config.encoder_dropout,
            )
        )

        if getattr(config, 'denoiser_type', 'bigru') == "conv3d":
            from diffusion.block_ar.conv3d_denoiser import Conv3DBlockDenoiser, Conv3DDenoiserConfig
            self.denoiser = Conv3DBlockDenoiser(
                Conv3DDenoiserConfig(
                    frame_dim=config.surface_h * config.surface_w,
                    surface_h=config.surface_h,
                    surface_w=config.surface_w,
                    bottleneck_dim=config.bottleneck_dim,
                    pos_embed_dim=config.pos_embed_dim,
                    noise_embed_dim=config.conv3d_noise_embed_dim,
                    n_steps=config.n_steps,
                    base_channels=config.conv3d_base_channels,
                    n_res_blocks=config.conv3d_n_res_blocks,
                    groups=config.conv3d_groups,
                )
            )
        else:
            self.denoiser = BiGRUDenoiser(
                DenoiserConfig(
                    frame_dim=config.surface_h * config.surface_w,
                    surface_h=config.surface_h,
                    surface_w=config.surface_w,
                    bottleneck_dim=config.bottleneck_dim,
                    pos_embed_dim=config.pos_embed_dim,
                    noise_embed_dim=config.noise_embed_dim,
                    gru_hidden_dim=config.bigru_hidden_dim,
                    dropout=config.denoiser_dropout,
                )
            )

        self.scheduler = DDPMScheduler(
            n_steps=config.n_steps,
            schedule=config.schedule,
            device="cpu",
        )

        # Regime conditioning (hierarchical sampling)
        if config.use_regime_conditioning:
            self.regime_classifier = nn.Sequential(
                nn.Linear(config.bottleneck_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.n_regimes),
            )
            self.regime_embed = nn.Embedding(config.n_regimes, config.regime_embed_dim)
            self.regime_proj = nn.Sequential(
                nn.Linear(config.bottleneck_dim + config.regime_embed_dim, config.bottleneck_dim),
                nn.SiLU(),
            )
        else:
            self.regime_classifier = None
            self.regime_embed = None
            self.regime_proj = None

    def _augment_condition(
        self, condition: torch.Tensor, regime_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Augment condition with regime embedding if enabled.

        Projects concat(condition, regime_embed) back to bottleneck_dim
        so denoiser interface stays unchanged.
        """
        if self.regime_proj is not None and regime_ids is not None:
            r_emb = self.regime_embed(regime_ids)  # (B, regime_embed_dim)
            return self.regime_proj(torch.cat([condition, r_emb], dim=-1))
        return condition

    def _ensure_scheduler_device(self, device: torch.device) -> None:
        """Recreate scheduler on the correct device if needed."""
        if str(self.scheduler.device) != str(device):
            self.scheduler = DDPMScheduler(
                n_steps=self.config.n_steps,
                schedule=self.config.schedule,
                device=device,
            )

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        regime_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass with MCVD multi-task block-AR training.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            future: (B, future_len, 5, 5) in [-1, 1]
            regime_ids: (B,) regime labels for hierarchical conditioning

        Returns:
            dict with 'loss' scalar, optionally 'regime_loss', 'regime_acc'
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs

        self._ensure_scheduler_device(device)

        total_loss = 0.0

        for block_idx in range(n_blocks):
            start = block_idx * bs
            end = start + bs

            # Target block
            target_block = future[:, start:end]  # (B, bs, 5, 5)

            # Build past context: history + previous GT blocks (teacher forcing)
            if block_idx > 0:
                past_ctx = torch.cat(
                    [history, future[:, :start]], dim=1
                )
            else:
                past_ctx = history

            # Build future context: GT blocks after current
            if end < self.config.future_len:
                future_ctx = future[:, end:]
            else:
                future_ctx = None

            # Sample MCVD masks
            mask_past, mask_future = sample_mcvd_masks(
                B, self.config.p_mask, device=device
            )

            # Last block: force mask_future=True (no future context available)
            if future_ctx is None:
                mask_future = torch.ones(B, dtype=torch.bool, device=device)

            # Encode past and future separately
            past_cond = self.encoder(past_ctx, mask=mask_past)  # (B, bottleneck_dim)

            if future_ctx is not None and future_ctx.shape[1] > 0:
                future_cond = self.encoder(future_ctx, mask=mask_future)
            else:
                # No future context (last block) — use learned null embedding
                # to match what encoder returns when mask=True
                future_cond = self.encoder.null_embedding.expand(B, -1)

            # Additive conditioning
            condition = past_cond + future_cond  # (B, bottleneck_dim)

            # Regime conditioning: augment with regime embedding
            condition = self._augment_condition(condition, regime_ids)

            # Sample noise levels
            if self.config.use_uniform_noise:
                # One scalar t per sample, replicated across block frames
                k = torch.randint(0, self.config.n_steps, (B, 1), device=device)
                k = k.expand(B, bs).contiguous()  # (B, bs)
            else:
                # Per-frame task-adaptive noise (DF)
                k = sample_batch_task_adaptive_noise(
                    mask_past, mask_future, bs, self.config.n_steps, self.config.jitter_std
                )  # (B, bs)

            # Forward diffusion with PYoCo correlated noise
            noise = sample_pyoco_noise(
                target_block.shape, self.config.noise_rho, device
            )
            noisy_block, _ = self.scheduler.q_sample_per_frame(
                target_block, k, noise
            )  # (B, bs, 5, 5)

            # Denoise
            noisy_flat = noisy_block.reshape(B, bs, -1)  # (B, bs, 25)
            positions = (
                torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                + block_idx * bs
            )  # (B, bs)
            noise_pred = self.denoiser(
                noisy_flat, condition, positions, k
            )  # (B, bs, 25)

            # Loss
            noise_flat = noise.reshape(B, bs, -1)  # (B, bs, 25)
            if self.config.loss_type == "huber":
                block_loss = F.smooth_l1_loss(
                    noise_pred, noise_flat, beta=self.config.huber_delta
                )
            else:
                block_loss = F.mse_loss(noise_pred, noise_flat)
            total_loss = total_loss + block_loss

        result = {"loss": total_loss / n_blocks}

        # Regime classification loss (if labels provided and classifier exists)
        if regime_ids is not None and self.regime_classifier is not None:
            # no_grad on encoder: classifier adapts to encoder features,
            # encoder is trained only by diffusion loss
            with torch.no_grad():
                regime_cond = self.encoder(history, mask=None)
            regime_logits = self.regime_classifier(regime_cond)
            regime_loss = F.cross_entropy(regime_logits, regime_ids)
            result["loss"] = result["loss"] + self.config.regime_loss_weight * regime_loss
            result["regime_loss"] = regime_loss.item()
            result["regime_acc"] = (
                (regime_logits.argmax(dim=1) == regime_ids).float().mean().item()
            )

        return result

    def _pyramid_timesteps(
        self, T: int, n_steps: int, device: torch.device,
        t_min: Optional[torch.Tensor] = None,
    ) -> list:
        """Build the DF pyramid scheduling matrix.

        Returns a list of (B, T)-broadcastable per-frame timestep tensors,
        one per denoising iteration. Earlier frames are denoised faster.

        The pyramid has (n_steps + T - 1) total iterations:
          - Iteration 0: all frames at t = n_steps - 1  (fully noisy)
          - Each iteration: frame i's timestep decreases by 1 once the
            "denoising wave" reaches it (wave front arrives at frame i
            after i iterations)
          - Final iteration: frame i at t = t_min[i]

        Args:
            T: number of frames
            n_steps: number of diffusion timesteps
            device: torch device
            t_min: (T,) per-frame minimum timestep. Frames stop denoising
                   at their t_min instead of 0. None = all zeros.

        Returns:
            List of (T,) long tensors, one per iteration where at least
            one frame's timestep changes.
        """
        if t_min is None:
            t_min = torch.zeros(T, dtype=torch.long, device=device)

        # Build schedule: iteration k, frame i has t = clamp(n_steps-1-k+i, t_min[i], n_steps-1)
        # Total raw iterations: n_steps + T - 1, but many are duplicates
        # when frames are clamped. We deduplicate to avoid wasted compute.
        n_iters = n_steps + T - 1
        frame_idx = torch.arange(T, device=device)  # (T,)
        schedule = []
        prev = None
        for k in range(n_iters):
            t_frame = torch.max((n_steps - 1 - k + frame_idx), t_min).clamp(max=n_steps - 1)  # (T,)
            if prev is None or not torch.equal(t_frame, prev):
                schedule.append(t_frame)
                prev = t_frame
        return schedule

    @torch.no_grad()
    def _sample_block_pyramid(
        self,
        condition: torch.Tensor,
        positions: torch.Tensor,
        shape: tuple,
        t_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one block using the DF pyramid schedule.

        All frames are denoised jointly across iterations. Earlier frames
        clean up faster, so the BiGRU hidden state propagates clean info
        from early frames to noisy later frames — exactly the DF paper's
        "soft causality" mechanism.

        Args:
            condition: (B, bottleneck_dim) encoder output
            positions: (B, T) absolute frame positions
            shape: (B, T, H, W)
            t_min: (T,) per-frame minimum timestep. Frames stop denoising
                   at their t_min, retaining residual noise at that level.
                   None = all zeros (fully denoise, backward compat).

        Returns:
            block: (B, T, H, W) in [-1, 1]
        """
        B, T, H, W = shape
        device = condition.device
        n_steps = self.config.n_steps

        if t_min is None:
            t_min = torch.zeros(T, dtype=torch.long, device=device)

        # (T,) -> (B, T) for broadcasting
        t_min_expanded = t_min.unsqueeze(0).expand(B, -1)

        # Build pyramid schedule (frames clamp at their t_min instead of 0)
        schedule = self._pyramid_timesteps(T, n_steps, device, t_min=t_min)

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Run pyramid denoising
        for iter_idx in range(len(schedule) - 1):
            t_current = schedule[iter_idx].unsqueeze(0).expand(B, -1)    # (B, T)
            t_next = schedule[iter_idx + 1].unsqueeze(0).expand(B, -1)   # (B, T)

            # Skip if no frame needs updating (all at their t_min)
            active = (t_current > t_min_expanded)  # (B, T)
            if not active.any():
                break

            # Flatten spatial dims for denoiser: (B, T, H, W) -> (B, T, H*W)
            x_flat = x_t.reshape(B, T, H * W)

            # Predict noise (denoiser sees per-frame noise levels)
            noise_pred_flat = self.denoiser(
                x_flat, condition, positions, t_current
            )  # (B, T, H*W)
            noise_pred = noise_pred_flat.reshape(B, T, H, W)

            # DDPM posterior step: for each frame, go from t_current to t_next
            # Predict x_0 from noise prediction
            t_flat = t_current.flatten()
            sqrt_recip = self.scheduler.sqrt_recip_alpha_bar[t_flat].view(B, T, 1, 1)
            sqrt_recip_m1 = self.scheduler.sqrt_recip_alpha_bar_minus_one[t_flat].view(B, T, 1, 1)
            x_0_pred = sqrt_recip * x_t - sqrt_recip_m1 * noise_pred
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            # Compute posterior mean: mu = coef_x0 * x_0_pred + coef_xt * x_t
            alpha_t = self.scheduler.alphas[t_flat].view(B, T, 1, 1)
            alpha_bar_t = self.scheduler.alpha_bar[t_flat].view(B, T, 1, 1)
            alpha_bar_prev_t = self.scheduler.alpha_bar_prev[t_flat].view(B, T, 1, 1)
            beta_t = self.scheduler.betas[t_flat].view(B, T, 1, 1)

            coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            mean = coef_x0 * x_0_pred + coef_xt * x_t

            # Add noise (except when t_current is at absolute 0 where posterior_var=0)
            posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
            z = torch.randn_like(x_t)
            nonzero = (t_current > 0).float().unsqueeze(-1).unsqueeze(-1)
            x_new = mean + nonzero * torch.sqrt(posterior_var) * z

            # Only update active frames (those above their t_min)
            active_mask = active.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            x_t = active_mask * x_new + (1 - active_mask) * x_t

        return x_t

    @torch.no_grad()
    def _sample_block_uniform(
        self,
        condition: torch.Tensor,
        positions: torch.Tensor,
        shape: tuple,
        t_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample one block using uniform DDPM reverse (all frames at same t).

        Standard DDPM reverse diffusion where all frames share the same
        global timestep at each step. Supports per-frame t_min for growing
        uncertainty (frames stop denoising at their t_min).

        Args:
            condition: (B, bottleneck_dim) encoder output
            positions: (B, T) absolute frame positions
            shape: (B, T, H, W)
            t_min: (T,) per-frame minimum timestep, or None (all zeros)

        Returns:
            block: (B, T, H, W) in [-1, 1]
        """
        B, T, H, W = shape
        device = condition.device
        n_steps = self.config.n_steps

        if t_min is None:
            t_min = torch.zeros(T, dtype=torch.long, device=device)

        # (T,) -> (B, T) for broadcasting
        t_min_expanded = t_min.unsqueeze(0).expand(B, -1)

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Standard DDPM reverse: t = n_steps-1, n_steps-2, ..., 0
        for t_global in reversed(range(n_steps)):
            # All frames at same t, clamped to per-frame t_min
            t_current = torch.full((B, T), t_global, device=device, dtype=torch.long)
            t_current = torch.maximum(t_current, t_min_expanded)

            # Skip if all frames have reached their t_min
            active = (t_current > t_min_expanded)  # (B, T)
            if not active.any():
                break

            # Flatten spatial dims for denoiser: (B, T, H, W) -> (B, T, H*W)
            x_flat = x_t.reshape(B, T, H * W)

            # Predict noise
            noise_pred_flat = self.denoiser(
                x_flat, condition, positions, t_current
            )  # (B, T, H*W)
            noise_pred = noise_pred_flat.reshape(B, T, H, W)

            # DDPM posterior step (same math as _sample_block_pyramid)
            t_flat = t_current.flatten()
            sqrt_recip = self.scheduler.sqrt_recip_alpha_bar[t_flat].view(B, T, 1, 1)
            sqrt_recip_m1 = self.scheduler.sqrt_recip_alpha_bar_minus_one[t_flat].view(B, T, 1, 1)
            x_0_pred = sqrt_recip * x_t - sqrt_recip_m1 * noise_pred
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            alpha_t = self.scheduler.alphas[t_flat].view(B, T, 1, 1)
            alpha_bar_t = self.scheduler.alpha_bar[t_flat].view(B, T, 1, 1)
            alpha_bar_prev_t = self.scheduler.alpha_bar_prev[t_flat].view(B, T, 1, 1)
            beta_t = self.scheduler.betas[t_flat].view(B, T, 1, 1)

            coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            mean = coef_x0 * x_0_pred + coef_xt * x_t

            posterior_var = self.scheduler.posterior_variance[t_flat].view(B, T, 1, 1)
            z = torch.randn_like(x_t)
            nonzero = (t_current > 0).float().unsqueeze(-1).unsqueeze(-1)
            x_new = mean + nonzero * torch.sqrt(posterior_var) * z

            # Only update active frames (those above their t_min)
            active_mask = active.float().unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
            x_t = active_mask * x_new + (1 - active_mask) * x_t

        return x_t

    def _compute_block_t_min(
        self, block_idx: int, block_size: int, future_len: int,
        max_global_residual: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Compute per-frame t_min for a block based on global horizon.

        t_min(h) = max_global_residual * h / (future_len - 1)

        where h is the global forecast horizon (0 to future_len-1).

        Args:
            block_idx: which block (0-indexed)
            block_size: frames per block
            future_len: total future frames
            max_global_residual: max residual timestep at h=future_len-1
            device: torch device

        Returns:
            (block_size,) long tensor, or None if max_global_residual == 0
        """
        if max_global_residual == 0:
            return None
        frame_indices = torch.arange(block_size, device=device)
        global_horizons = block_idx * block_size + frame_indices  # 0..future_len-1
        t_min = (max_global_residual * global_horizons.float() / (future_len - 1)).long()
        return t_min

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        max_residual: int = 20,
        max_global_residual: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Block-AR generation using DF pyramid sampling.

        Each block is denoised using the pyramid schedule where earlier
        frames clean up faster, enabling the BiGRU to propagate clean
        information to noisy later frames (DF "soft causality").

        With max_global_residual > 0, frames retain horizon-dependent
        residual noise: frame at global horizon h stops denoising at
        t_min = max_global_residual * h / (future_len - 1). This creates
        naturally growing uncertainty with forecast horizon.

        When regime conditioning is enabled, samples regime once per
        trajectory and uses it consistently across all blocks.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            n_samples: number of independent samples per history
            max_residual: unused (kept for API compatibility)
            max_global_residual: override config.max_global_residual.
                None = use config value.
            temperature: regime sampling temperature (higher = more diverse)

        Returns:
            (B, n_samples, future_len, 5, 5) denormalized to [0, 1]
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs
        mgr = max_global_residual if max_global_residual is not None else self.config.max_global_residual

        self._ensure_scheduler_device(device)

        all_samples = []

        for _ in range(n_samples):
            current_cond_surfaces = history  # (B, history_len, 5, 5) in [-1, 1]
            blocks = []

            # Sample regime once per trajectory (consistent across blocks)
            regime_ids = None
            if self.regime_classifier is not None:
                init_cond = self.encoder(history, mask=None)
                regime_logits = self.regime_classifier(init_cond)
                regime_probs = F.softmax(regime_logits / temperature, dim=-1)
                regime_ids = torch.multinomial(regime_probs, num_samples=1).squeeze(-1)

            for block_idx in range(n_blocks):
                # Encode growing context (no masking at inference)
                condition = self.encoder(
                    current_cond_surfaces, mask=None
                )  # (B, bottleneck_dim)

                # Augment with regime embedding
                condition = self._augment_condition(condition, regime_ids)

                # Create positions for this block
                positions = (
                    torch.arange(bs, device=device).unsqueeze(0).expand(B, -1)
                    + block_idx * bs
                )

                # Compute per-frame t_min for this block
                t_min = self._compute_block_t_min(
                    block_idx, bs, self.config.future_len, mgr, device
                )

                # Generate block
                shape = (B, bs, self.config.surface_h, self.config.surface_w)
                if self.config.sampling_mode == "uniform":
                    block = self._sample_block_uniform(
                        condition, positions, shape, t_min=t_min
                    )
                else:
                    block = self._sample_block_pyramid(
                        condition, positions, shape, t_min=t_min
                    )
                # block: (B, bs, 5, 5) in [-1, 1]

                blocks.append(block)

                # Grow conditioning surfaces
                current_cond_surfaces = torch.cat(
                    [current_cond_surfaces, block], dim=1
                )

            # Concatenate all blocks
            full_trajectory = torch.cat(blocks, dim=1)  # (B, future_len, 5, 5)
            all_samples.append(full_trajectory)

        # Stack samples: (B, n_samples, future_len, 5, 5)
        samples = torch.stack(all_samples, dim=1)

        # Denormalize to [0, 1] and clamp
        samples = denormalize_iv(samples)
        samples = samples.clamp(0.0, 1.0)

        return samples

    @torch.no_grad()
    def sample_batched(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        max_residual: int = 20,
        max_global_residual: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Block-AR generation with all samples batched in parallel.

        Folds n_samples into the batch dimension so the GPU processes
        B*n_samples items per denoiser call instead of B. Same algorithm
        as sample(), just parallelized.

        Args:
            history: (B, history_len, 5, 5) in [-1, 1]
            n_samples: number of independent samples per history
            max_residual: unused (kept for API compatibility)
            max_global_residual: override config.max_global_residual.
                None = use config value.
            temperature: regime sampling temperature (higher = more diverse)

        Returns:
            (B, n_samples, future_len, 5, 5) denormalized to [0, 1]
        """
        B = history.shape[0]
        device = history.device
        bs = self.config.block_size
        n_blocks = self.config.future_len // bs
        S = n_samples
        B_eff = B * S
        mgr = max_global_residual if max_global_residual is not None else self.config.max_global_residual

        self._ensure_scheduler_device(device)

        # Expand history: (B, T, H, W) → (B*S, T, H, W)
        # repeat_interleave gives [h0,h0,...,h0, h1,h1,...,h1, ...]
        # so .view(B, S, ...) later correctly groups samples per history
        current_cond_surfaces = history.repeat_interleave(S, dim=0)

        # Sample regimes for all B*S items (once, reuse across blocks)
        regime_ids = None
        if self.regime_classifier is not None:
            init_cond = self.encoder(current_cond_surfaces, mask=None)
            regime_logits = self.regime_classifier(init_cond)
            regime_probs = F.softmax(regime_logits / temperature, dim=-1)
            regime_ids = torch.multinomial(regime_probs, num_samples=1).squeeze(-1)

        blocks = []

        for block_idx in range(n_blocks):
            # Encode growing context: (B*S, T_growing, 5, 5) → (B*S, bottleneck_dim)
            condition = self.encoder(
                current_cond_surfaces, mask=None
            )

            # Augment with regime embedding
            condition = self._augment_condition(condition, regime_ids)

            # Positions: (B*S, bs)
            positions = (
                torch.arange(bs, device=device).unsqueeze(0).expand(B_eff, -1)
                + block_idx * bs
            )

            # Compute per-frame t_min for this block
            t_min = self._compute_block_t_min(
                block_idx, bs, self.config.future_len, mgr, device
            )

            # Generate block for all samples in parallel
            shape = (B_eff, bs, self.config.surface_h, self.config.surface_w)
            if self.config.sampling_mode == "uniform":
                block = self._sample_block_uniform(
                    condition, positions, shape, t_min=t_min
                )
            else:
                block = self._sample_block_pyramid(
                    condition, positions, shape, t_min=t_min
                )
            # block: (B*S, bs, 5, 5)

            blocks.append(block)

            # Grow conditioning surfaces
            current_cond_surfaces = torch.cat(
                [current_cond_surfaces, block], dim=1
            )

        # Concatenate blocks: (B*S, future_len, 5, 5)
        full_trajectory = torch.cat(blocks, dim=1)

        # Reshape: (B*S, future_len, 5, 5) → (B, S, future_len, 5, 5)
        samples = full_trajectory.view(
            B, S, self.config.future_len,
            self.config.surface_h, self.config.surface_w,
        )

        # Denormalize to [0, 1] and clamp
        samples = denormalize_iv(samples)
        samples = samples.clamp(0.0, 1.0)

        return samples
