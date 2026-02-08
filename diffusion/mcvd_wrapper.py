"""
MCVD Wrapper for Volatility Surface Forecasting.

Adapts MCVD's 2D U-Net (UNetMore_DDPM) for our 5×5 IV surface data by:
1. Padding 5×5 → 8×8 for proper U-Net hierarchy (8→4→2)
2. Framing history as channel-concatenated conditioning
3. Implementing masking training (prob_mask_cond)
4. Providing the same interface as ConditionalDDPM

The MCVD model uses frames-as-channels: (B, T*C, H, W) instead of 3D conv.
Conditioning is via frame concatenation (not FiLM), preserving full spatial info.
"""

import sys
from pathlib import Path

# Add MCVD directory to path so its internal relative imports work
_MCVD_DIR = str(Path(__file__).parent / "mcvd")
if _MCVD_DIR not in sys.path:
    sys.path.insert(0, _MCVD_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.better.ncsnpp_more import UNetMore_DDPM

# IV normalization constants (same as simple_denoiser.py)
IV_MIN = 0.0
IV_MAX = 1.0


def denormalize_iv(iv_norm: torch.Tensor) -> torch.Tensor:
    """Denormalize IV from [-1, 1] to [0, 1]."""
    return (iv_norm + 1.0) / 2.0 * (IV_MAX - IV_MIN) + IV_MIN


def pad_surface(x: torch.Tensor) -> torch.Tensor:
    """Pad (..., 5, 5) → (..., 8, 8) with reflect padding.

    Adds 1 left + 2 right (width), 1 top + 2 bottom (height).
    """
    return F.pad(x, (1, 2, 1, 2), mode='reflect')


def crop_surface(x: torch.Tensor) -> torch.Tensor:
    """Crop (..., 8, 8) → (..., 5, 5) removing padding."""
    return x[..., 1:6, 1:6]


class MCVDModel(nn.Module):
    """MCVD-based conditional diffusion model for IV surface forecasting.

    Wraps MCVD's UNetMore_DDPM with:
    - Automatic pad (5×5 → 8×8) and crop (8×8 → 5×5)
    - Frame-concatenation conditioning with masking
    - DDIM sampling

    The model's internal alphas/betas define the noise schedule (cosine).
    """

    def __init__(self, mcvd_config):
        """Initialize MCVD model.

        Args:
            mcvd_config: Namespace config with config.data and config.model
                         (built by config_mcvd_poc.build_mcvd_config)
        """
        super().__init__()
        self.config = mcvd_config  # EMAHelper.ema_copy() uses module.config
        self.mcvd_config = mcvd_config
        self.num_frames = mcvd_config.data.num_frames
        self.num_frames_cond = mcvd_config.data.num_frames_cond
        self.channels = mcvd_config.data.channels
        self.prob_mask_cond = getattr(mcvd_config.data, 'prob_mask_cond', 0.0)

        # Build MCVD model (contains its own noise schedule)
        self.scorenet = UNetMore_DDPM(mcvd_config)

    @property
    def alphas(self):
        """Cumulative alpha schedule from the MCVD model.

        MCVD convention: index 0 = most noisy (alpha ≈ 0),
                         index T-1 = least noisy (alpha ≈ 1).
        """
        return self.scorenet.alphas

    @property
    def n_steps(self):
        return len(self.alphas)

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
    ) -> dict:
        """Training forward pass.

        Args:
            history: (B, T_hist, 5, 5) past surfaces, normalized to [-1, 1]
            future: (B, T_fut, 5, 5) future surfaces, normalized to [-1, 1]

        Returns:
            dict with 'loss', 'noise_pred', 'noise'
        """
        B = history.shape[0]
        device = history.device

        # --- Pad 5×5 → 8×8 ---
        history_pad = pad_surface(history)  # (B, T_hist, 8, 8)
        future_pad = pad_surface(future)    # (B, T_fut, 8, 8)

        # --- Reshape to MCVD format: (B, T*C, H, W) ---
        # With C=1, this is just (B, T, 8, 8) already
        cond_frames = history_pad  # (B, 30, 8, 8) = (B, num_frames_cond * C, H, W)
        clean_frames = future_pad  # (B, 30, 8, 8) = (B, num_frames * C, H, W)

        # --- Apply conditioning mask ---
        # Per-sample binary mask: 1 = conditioning present, 0 = masked
        if self.training and self.prob_mask_cond > 0:
            cond_mask = (torch.rand(B, device=device) > self.prob_mask_cond).to(torch.int32)
            # Zero out conditioning for masked samples
            cond_frames = cond_frames * cond_mask.reshape(B, 1, 1, 1).float()
        else:
            cond_mask = torch.ones(B, device=device, dtype=torch.int32)

        # --- Add noise (DDPM forward process) ---
        # MCVD convention: t=0 is most noisy, t=T-1 is least noisy
        t = torch.randint(0, self.n_steps, (B,), device=device)
        alpha_t = self.alphas[t]  # (B,)
        alpha_t = alpha_t.reshape(B, 1, 1, 1)  # broadcast

        noise = torch.randn_like(clean_frames)
        x_noisy = alpha_t.sqrt() * clean_frames + (1 - alpha_t).sqrt() * noise

        # --- Forward through model ---
        # model(x_noisy, timestep, cond=conditioning, cond_mask=mask)
        noise_pred = self.scorenet(x_noisy, t, cond=cond_frames, cond_mask=cond_mask)

        # --- Compute loss on full 8×8 ---
        loss = F.mse_loss(noise_pred, noise)

        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }

    def _sample_block(
        self,
        cond_frames_pad: torch.Tensor,
        cond_mask: torch.Tensor,
        B: int, T: int, H: int, W: int,
        device: torch.device,
        sampler: str = 'ddim',
        n_inference_steps: int = 20,
    ) -> torch.Tensor:
        """Generate one block of T frames in [-1, 1], padded space (8×8).

        Returns:
            (B, T, H, W) in [-1, 1] range, still in padded 8×8 space.
        """
        if sampler == 'ddim':
            return self._sample_ddim(
                cond_frames_pad, cond_mask, B, T, H, W, device,
                n_inference_steps=n_inference_steps,
            )
        else:
            return self._sample_ddpm(
                cond_frames_pad, cond_mask, B, T, H, W, device,
            )

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        sampler: str = 'ddim',
        n_inference_steps: int = 20,
        **kwargs,
    ) -> torch.Tensor:
        """Generate future surface samples.

        Args:
            history: (B, T_hist, 5, 5) past surfaces, normalized to [-1, 1]
            n_samples: Number of samples per history
            sampler: 'ddim' (recommended) or 'ddpm'
            n_inference_steps: Steps for DDIM (default: 20)

        Returns:
            samples: (B, n_samples, T_fut, 5, 5) denormalized to [0, 1]
        """
        B = history.shape[0]
        device = history.device
        T = self.num_frames
        H, W = 8, 8  # Padded size

        # Pad history
        history_pad = pad_surface(history)  # (B, T_hist, 8, 8)
        cond_frames = history_pad
        cond_mask = torch.ones(B, device=device, dtype=torch.int32)

        samples = []
        for _ in range(n_samples):
            x_0 = self._sample_block(
                cond_frames, cond_mask, B, T, H, W, device,
                sampler=sampler, n_inference_steps=n_inference_steps,
            )
            # Crop 8×8 → 5×5
            x_0_cropped = crop_surface(x_0)  # (B, T, 5, 5)
            samples.append(x_0_cropped)

        # Stack: (B, n_samples, T, 5, 5)
        samples = torch.stack(samples, dim=1)

        # Denormalize from [-1, 1] to [0, 1]
        samples = denormalize_iv(samples)

        return samples

    @torch.no_grad()
    def sample_autoregressive(
        self,
        history: torch.Tensor,
        n_samples: int = 1,
        n_blocks: int = 6,
        sampler: str = 'ddim',
        n_inference_steps: int = 100,
    ) -> torch.Tensor:
        """Generate long sequences via autoregressive rollout.

        Generates n_blocks * future_len frames by repeatedly:
        1. Generate one block of future_len frames
        2. Use generated frames as conditioning for the next block

        This matches the MCVD paper's autoregressive prediction protocol.

        Args:
            history: (B, T_hist, 5, 5) initial conditioning, normalized [-1, 1]
            n_samples: Number of independent sample trajectories
            n_blocks: Number of blocks to generate (6 × 5 = 30 frames)
            sampler: 'ddim' or 'ddpm'
            n_inference_steps: Steps per block for DDIM

        Returns:
            samples: (B, n_samples, n_blocks * T_fut, 5, 5) denormalized [0, 1]
        """
        B = history.shape[0]
        device = history.device
        T = self.num_frames
        H, W = 8, 8

        all_sample_runs = []

        for _ in range(n_samples):
            # Start with real history (padded)
            cond_pad = pad_surface(history)  # (B, T_hist, 8, 8)
            cond_mask = torch.ones(B, device=device, dtype=torch.int32)
            blocks = []

            for block_idx in range(n_blocks):
                # Generate one block
                x_block = self._sample_block(
                    cond_pad, cond_mask, B, T, H, W, device,
                    sampler=sampler, n_inference_steps=n_inference_steps,
                )
                # x_block: (B, T, 8, 8) in [-1, 1]

                # Crop and store
                blocks.append(crop_surface(x_block))  # (B, T, 5, 5)

                # Shift conditioning: generated block becomes next history
                # Clip before using as conditioning (paper: clip_before=True)
                cond_pad = x_block.clamp(-1, 1)  # (B, T, 8, 8)

            # Concatenate all blocks: (B, n_blocks * T, 5, 5)
            full_seq = torch.cat(blocks, dim=1)
            all_sample_runs.append(full_seq)

        # Stack: (B, n_samples, n_blocks * T, 5, 5)
        result = torch.stack(all_sample_runs, dim=1)
        return denormalize_iv(result)

    def _sample_ddim(
        self,
        cond: torch.Tensor,
        cond_mask: torch.Tensor,
        B: int, T: int, H: int, W: int,
        device: torch.device,
        n_inference_steps: int = 20,
    ) -> torch.Tensor:
        """DDIM sampling using MCVD's internal noise schedule.

        MCVD convention: alphas[0] ≈ 0 (noisy), alphas[-1] ≈ 1 (clean).
        Sampling iterates from index 0 (noisy) toward index T-1 (clean).
        """
        alphas = self.alphas

        # Subsample steps for DDIM
        n_total = len(alphas)
        if n_inference_steps < n_total:
            skip = n_total // n_inference_steps
            step_indices = list(range(0, n_total, skip))
        else:
            step_indices = list(range(n_total))

        # Get subsampled alphas
        steps = torch.tensor(step_indices, device=device)
        sub_alphas = alphas.index_select(0, steps)
        sub_alphas_prev = torch.cat([sub_alphas[1:], torch.tensor([1.0], device=device)])

        # Start from pure noise
        x = torch.randn(B, T * self.channels, H, W, device=device)

        for i, step_idx in enumerate(step_indices):
            alpha_t = sub_alphas[i]
            alpha_prev = sub_alphas_prev[i]

            # Timestep label for the model
            t_label = torch.full((B,), step_idx, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.scorenet(x, t_label, cond=cond, cond_mask=cond_mask)

            # DDIM step: predict x0, then step toward clean
            x0_pred = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t).sqrt() * noise_pred)
            x0_pred = x0_pred.clamp(-1, 1)
            x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred

        # Final denoising step
        last_t = torch.full((B,), len(step_indices) - 1, device=device, dtype=torch.long)
        final_noise = self.scorenet(x, last_t, cond=cond, cond_mask=cond_mask)
        x = x - (1 - alphas[step_indices[-1]]).sqrt() * final_noise
        x = x.clamp(-1, 1)

        # Reshape from (B, T*C, H, W) to (B, T, H, W) for C=1
        x = x.reshape(B, T, H, W)
        return x

    def _sample_ddpm(
        self,
        cond: torch.Tensor,
        cond_mask: torch.Tensor,
        B: int, T: int, H: int, W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Full DDPM sampling (all steps)."""
        alphas = self.alphas
        alphas_prev = self.scorenet.alphas_prev
        betas = self.scorenet.betas

        x = torch.randn(B, T * self.channels, H, W, device=device)

        # Iterate from step 0 (most noisy) to step T-1 (least noisy)
        for i in range(len(alphas)):
            alpha_t = alphas[i]
            alpha_prev = alphas_prev[i]
            beta_t = betas[i]

            t_label = torch.full((B,), i, device=device, dtype=torch.long)
            noise_pred = self.scorenet(x, t_label, cond=cond, cond_mask=cond_mask)

            # DDPM reverse step
            x0_pred = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t).sqrt() * noise_pred)
            x0_pred = x0_pred.clamp(-1, 1)

            # Mean
            mean = (alpha_prev.sqrt() * beta_t / (1 - alpha_t)) * x0_pred + \
                   ((1 - beta_t).sqrt() * (1 - alpha_prev) / (1 - alpha_t)) * x

            # Add noise (except at last step)
            if i < len(alphas) - 1:
                noise = torch.randn_like(x)
                variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
                x = mean + variance.sqrt() * noise
            else:
                x = mean

        x = x.reshape(B, T, H, W)
        return x
