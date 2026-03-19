"""
Dual Decoder: AR + One-Shot joint training with shared CRPS (Exp 121a).

Wrapper that holds two SinglePassBlockAR instances:
  - AR path: FrameDecoder with noise_skip, cell_spread, reflecting boundaries
  - One-shot path: Conv3D with oneshot_additive, AR noise overlay

Both share the same frozen encoder. Each generates K/2 members.
Combined K members are evaluated with a single afCRPS loss.

This forces the two decoders to be complementary — AR provides
kurtosis/temporal structure, one-shot provides factor diversity.
"""

import torch
import torch.nn as nn

from diffusion.block_ar.single_pass_ar import (
    SinglePassBlockAR,
    SinglePassConfig,
    afcrps_loss,
    denormalize_iv,
    energy_score,
    interval_score,
)


class DualDecoder(nn.Module):
    """Joint AR + One-Shot model with shared encoder and combined CRPS."""

    def __init__(self, ar_config: SinglePassConfig, os_config: SinglePassConfig):
        super().__init__()
        self.ar_model = SinglePassBlockAR(ar_config)
        self.os_model = SinglePassBlockAR(os_config)
        # Share encoder: point os_model.encoder to ar_model.encoder
        self.os_model.encoder = self.ar_model.encoder
        self.ar_config = ar_config
        self.os_config = os_config

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        n_members: int = 8,
        lambda_vs: float = 0.0,
        lambda_es: float = 0.0,
        lambda_is: float = 0.0,
        lambda_cell_var: float = 0.0,
        **kwargs,
    ) -> dict:
        """Generate K/2 from AR + K/2 from one-shot, combine, compute CRPS."""
        B = history.shape[0]
        device = history.device
        H, W = self.ar_config.surface_h, self.ar_config.surface_w

        n_ar = n_members // 2
        n_os = n_members - n_ar

        # GT
        n_frames = self.ar_config.future_len
        gt_iv = denormalize_iv(future[:, :n_frames])

        # Generate AR members
        ar_samples = self.ar_model.sample(
            history, n_samples=n_ar, n_frames=n_frames
        )  # (B, n_ar, T, H, W)

        # Generate one-shot members
        os_samples = self.os_model.sample(
            history, n_samples=n_os, n_frames=n_frames
        )  # (B, n_os, T, H, W)

        # Combine
        iv_samples = torch.cat([ar_samples, os_samples], dim=1)  # (B, K, T, H, W)

        # Compute afCRPS on combined ensemble
        crps, mae, spread = afcrps_loss(
            iv_samples, gt_iv, alpha=0.95, reduction="frame_sum"
        )
        loss = crps

        # Energy score
        es_val = torch.tensor(0.0, device=device)
        if lambda_es > 0:
            es_val = energy_score(iv_samples, gt_iv)
            loss = loss + lambda_es * es_val

        # Interval score
        is_val = torch.tensor(0.0, device=device)
        if lambda_is > 0:
            is_val = interval_score(iv_samples, gt_iv, alpha=0.9)
            loss = loss + lambda_is * is_val

        return {
            "loss": loss,
            "crps": crps.detach(),
            "mae": mae.detach(),
            "spread": spread.detach(),
            "energy_score": es_val.detach(),
            "interval_score": is_val.detach(),
            "spread_mae_ratio": (spread / mae.clamp(min=1e-8)).detach(),
        }

    @torch.no_grad()
    def sample(self, history, n_samples=50, n_frames=30):
        """Generate samples from both decoders."""
        n_ar = n_samples // 2
        n_os = n_samples - n_ar
        ar_samples = self.ar_model.sample(history, n_samples=n_ar, n_frames=n_frames)
        os_samples = self.os_model.sample(history, n_samples=n_os, n_frames=n_frames)
        return torch.cat([ar_samples, os_samples], dim=1)
