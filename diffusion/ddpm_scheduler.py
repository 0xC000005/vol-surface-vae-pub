"""
DDPM Scheduler for Volatility Surface Diffusion.

Implements the forward and reverse diffusion processes following:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)

The scheduler handles:
1. Forward process q(x_t | x_0): adding noise according to a schedule
2. Reverse process p(x_{t-1} | x_t): denoising given model predictions
3. Full sampling loop from x_T ~ N(0,I) to x_0
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DDPMScheduler:
    """
    DDPM diffusion scheduler.

    Supports linear and cosine noise schedules.
    """

    def __init__(
        self,
        n_steps: int = 100,
        schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        device: str = 'cpu',
    ):
        """
        Args:
            n_steps: Number of diffusion timesteps
            schedule: 'linear' or 'cosine' beta schedule
            beta_start: Starting beta for linear schedule
            beta_end: Ending beta for linear schedule
            clip_sample: Whether to clip samples during sampling
            clip_sample_range: Clip range (for normalized IV surfaces, use ~1.0)
            device: Device to place tensors on
        """
        self.n_steps = n_steps
        self.schedule = schedule
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.device = device

        # Compute beta schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == 'cosine':
            betas = self._cosine_beta_schedule(n_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(device)

        # Pre-compute alpha and related quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]])

        # Pre-compute quantities for q(x_t | x_0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # Pre-compute quantities for p(x_{t-1} | x_t)
        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        # Clamp to avoid log(0)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance = torch.log(self.posterior_variance)

        # Coefficients for x_0 prediction from noise prediction
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = torch.sqrt(1.0 / self.alpha_bar - 1.0)

    def _cosine_beta_schedule(self, n_steps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine beta schedule from Nichol & Dhariwal (2021).

        More gradual noise addition than linear schedule, especially at the start.
        """
        steps = n_steps + 1
        t = torch.linspace(0, n_steps, steps)
        f_t = torch.cos(((t / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, 0.0001, 0.999)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: sample x_t given x_0.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0: Clean samples (B, ...)
            t: Timesteps (B,) integers in [0, n_steps)
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy samples (B, ...)
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get coefficients for each sample in batch
        sqrt_alpha_bar_t = self._gather(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._gather(self.sqrt_one_minus_alpha_bar, t, x_0.shape)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy samples (B, ...)
            t: Timesteps (B,)
            noise_pred: Predicted noise (B, ...)

        Returns:
            x_0_pred: Predicted clean samples (B, ...)
        """
        sqrt_recip_alpha_bar_t = self._gather(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recip_alpha_bar_minus_one_t = self._gather(
            self.sqrt_recip_alpha_bar_minus_one, t, x_t.shape
        )

        x_0_pred = sqrt_recip_alpha_bar_t * x_t - sqrt_recip_alpha_bar_minus_one_t * noise_pred

        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -self.clip_sample_range, self.clip_sample_range)

        return x_0_pred

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step: sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model: Denoiser network that predicts noise
            x_t: Current noisy samples (B, ...)
            t: Current timesteps (B,)
            condition: Conditioning context (e.g., history encoding)

        Returns:
            x_{t-1}: Slightly less noisy samples (B, ...)
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)

        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Compute mean of p(x_{t-1} | x_t, x_0)
        # mu = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0
        #    + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
        alpha_t = self._gather(self.alphas, t, x_t.shape)
        alpha_bar_t = self._gather(self.alpha_bar, t, x_t.shape)
        alpha_bar_prev_t = self._gather(self.alpha_bar_prev, t, x_t.shape)
        beta_t = self._gather(self.betas, t, x_t.shape)

        coef_x0 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)

        mean = coef_x0 * x_0_pred + coef_xt * x_t

        # Sample (except at t=0, where we return the mean)
        noise = torch.randn_like(x_t)
        posterior_variance_t = self._gather(self.posterior_variance, t, x_t.shape)

        # Mask: at t=0, we don't add noise
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))

        x_prev = mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: sample from x_T ~ N(0,I) to x_0.

        Args:
            model: Denoiser network
            condition: Conditioning context (B, ...)
            shape: Shape of samples to generate (B, T, H, W) for vol surfaces
            return_intermediates: If True, return all intermediate x_t

        Returns:
            x_0: Generated clean samples (B, ...)
            intermediates: (optional) List of intermediate samples
        """
        device = condition.device
        B = condition.shape[0]

        # Start from pure noise
        x_t = torch.randn(B, *shape[1:], device=device)

        intermediates = [] if return_intermediates else None

        # Reverse diffusion: t = T-1, T-2, ..., 0
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_batch, condition)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t

    def _gather(self, values: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Gather values at timestep t and reshape for broadcasting.

        Args:
            values: 1D tensor of values indexed by timestep
            t: Timestep indices (B,)
            shape: Target tensor shape for broadcasting

        Returns:
            values_t: Values at timesteps t, reshaped for broadcasting (B, 1, 1, ...)
        """
        out = values.gather(0, t)
        # Reshape to (B, 1, 1, ...) for broadcasting
        return out.view(-1, *([1] * (len(shape) - 1)))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio at timestep t.

        SNR(t) = alpha_bar_t / (1 - alpha_bar_t)

        Useful for weighting losses (e.g., arbitrage penalty at high SNR).

        Args:
            t: Timesteps (B,)

        Returns:
            snr_t: SNR values (B,)
        """
        alpha_bar_t = self.alpha_bar[t]
        return alpha_bar_t / (1.0 - alpha_bar_t)

    def ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDIM denoising step from timestep t to t_prev.

        DDIM is deterministic given the same initial noise, and can skip
        multiple timesteps (e.g., go from t=80 to t=60 directly).

        Args:
            model: Denoiser network that predicts noise
            x_t: Current noisy samples (B, ...)
            t: Current timesteps (B,)
            t_prev: Target timesteps (B,), can be -1 for final step
            condition: Conditioning context

        Returns:
            x_{t_prev}: Less noisy samples (B, ...)
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)

        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        # Get alpha_bar values
        alpha_bar_t = self._gather(self.alpha_bar, t, x_t.shape)

        # Handle t_prev = -1 case (final step to x_0)
        # When t_prev < 0, alpha_bar_prev = 1.0 (no noise)
        t_prev_clamped = t_prev.clamp(min=0)
        alpha_bar_t_prev = self._gather(self.alpha_bar, t_prev_clamped, x_t.shape)
        # Set alpha_bar_prev = 1.0 where t_prev < 0
        alpha_bar_t_prev = torch.where(
            t_prev.view(-1, *([1] * (x_t.dim() - 1))) >= 0,
            alpha_bar_t_prev,
            torch.ones_like(alpha_bar_t_prev)
        )

        # DDIM update (deterministic)
        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1 - alpha_bar_{t-1}) * noise_pred
        x_prev = (
            torch.sqrt(alpha_bar_t_prev) * x_0_pred +
            torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred
        )

        return x_prev

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        shape: Tuple[int, ...],
        n_inference_steps: int = 20,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        DDIM sampling with step-skipping for faster inference.

        Unlike DDPM which requires all n_steps, DDIM can use fewer steps
        (e.g., 20 instead of 100) by skipping intermediate timesteps.

        Args:
            model: Denoiser network
            condition: Conditioning context (B, ...)
            shape: Shape of samples to generate (B, T, H, W)
            n_inference_steps: Number of denoising steps (can be << n_steps)
            return_intermediates: If True, return all intermediate x_t

        Returns:
            x_0: Generated clean samples (B, ...)
        """
        device = condition.device
        B = condition.shape[0]

        # Create evenly-spaced timestep schedule
        # e.g., if n_steps=100 and n_inference_steps=20: [95, 90, 85, ..., 5, 0]
        step_ratio = self.n_steps / n_inference_steps
        timesteps = [int((self.n_steps - 1) - i * step_ratio) for i in range(n_inference_steps)]
        timesteps = [max(0, t) for t in timesteps]  # Ensure non-negative

        # Start from pure noise
        x_t = torch.randn(B, *shape[1:], device=device)

        intermediates = [] if return_intermediates else None

        # DDIM loop
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Next timestep (or -1 if this is the last step)
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
            else:
                t_prev_val = -1  # Signal final step
            t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)

            x_t = self.ddim_sample(model, x_t, t, t_prev, condition)

            if return_intermediates:
                intermediates.append(x_t.clone())

        if return_intermediates:
            return x_t, intermediates
        return x_t


def test_ddpm_scheduler():
    """Unit tests for DDPM scheduler."""
    print("Testing DDPM Scheduler...")

    device = 'cpu'
    scheduler = DDPMScheduler(n_steps=100, schedule='cosine', device=device)

    # Test q_sample (forward process)
    B, T, H, W = 4, 30, 5, 5
    x_0 = torch.randn(B, T, H, W)
    t = torch.randint(0, 100, (B,))

    x_t, noise = scheduler.q_sample(x_0, t)
    assert x_t.shape == x_0.shape, f"q_sample shape mismatch: {x_t.shape} vs {x_0.shape}"
    print(f"  q_sample: OK (shape {x_t.shape})")

    # Test noise recovery
    x_0_recovered = scheduler.predict_x0_from_noise(x_t, t, noise)
    mse = ((x_0 - x_0_recovered) ** 2).mean()
    # Note: clipping may affect this
    print(f"  x_0 recovery MSE: {mse:.6f} (should be small if no clipping)")

    # Test SNR
    snr_0 = scheduler.snr(torch.tensor([0]))
    snr_50 = scheduler.snr(torch.tensor([50]))
    snr_99 = scheduler.snr(torch.tensor([99]))
    print(f"  SNR at t=0: {snr_0.item():.2f} (high - clean)")
    print(f"  SNR at t=50: {snr_50.item():.4f}")
    print(f"  SNR at t=99: {snr_99.item():.6f} (low - noisy)")
    assert snr_0 > snr_50 > snr_99, "SNR should decrease with t"
    print("  SNR monotonicity: OK")

    # Test alpha_bar bounds
    assert scheduler.alpha_bar[0] > 0.9, f"alpha_bar_0 should be close to 1, got {scheduler.alpha_bar[0]}"
    assert scheduler.alpha_bar[-1] < 0.1, f"alpha_bar_T should be close to 0, got {scheduler.alpha_bar[-1]}"
    print(f"  alpha_bar bounds: OK (alpha_bar_0={scheduler.alpha_bar[0]:.4f}, alpha_bar_T={scheduler.alpha_bar[-1]:.4f})")

    print("DDPM Scheduler tests passed!\n")


if __name__ == "__main__":
    test_ddpm_scheduler()
