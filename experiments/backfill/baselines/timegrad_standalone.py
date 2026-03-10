"""
Standalone TimeGrad implementation for IV surface scenario generation.

Reimplements TimeGrad (Rasul et al., ICML 2021) as a self-contained PyTorch module,
avoiding the GluonTS dependency which is incompatible with Python 3.13.

Architecture:
  GRU encoder processes history → hidden state h_0
  For each future step t (autoregressive):
    1. DDPM reverse process generates x_t conditioned on h_t
    2. Update h_{t+1} = GRU(x_t, h_t)

Reference: "Autoregressive Denoising Diffusion Models for Multivariate
Probabilistic Time Series Forecasting" (Rasul et al., ICML 2021)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeGradDenoiser(nn.Module):
    """Small MLP denoiser conditioned on GRU hidden state and diffusion step."""
    def __init__(self, input_dim=25, hidden_dim=256, cond_dim=128, diff_emb_dim=64):
        super().__init__()
        self.time_emb = SinusoidalPositionEmbeddings(diff_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(diff_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_noisy, t, condition):
        """
        Args:
            x_noisy: (B, input_dim) noisy input
            t: (B,) diffusion timestep
            condition: (B, cond_dim) GRU hidden state
        Returns:
            predicted noise: (B, input_dim)
        """
        t_emb = self.time_proj(self.time_emb(t))  # (B, hidden)
        c_emb = self.cond_proj(condition)  # (B, hidden)
        h = torch.cat([x_noisy, t_emb, c_emb], dim=-1)
        return self.net(h)


class TimeGradModel(nn.Module):
    """
    TimeGrad: autoregressive diffusion model for multivariate time series.

    At each timestep, a small DDPM generates the next observation conditioned
    on the GRU hidden state. The generated observation is then fed back into
    the GRU to update the state for the next step.
    """
    def __init__(self, input_dim=25, gru_hidden=128, n_diffusion_steps=100,
                 beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.input_dim = input_dim
        self.gru_hidden = gru_hidden
        self.n_steps = n_diffusion_steps

        # GRU encoder/state tracker
        self.gru = nn.GRU(input_dim, gru_hidden, batch_first=True)

        # Denoiser
        self.denoiser = TimeGradDenoiser(
            input_dim=input_dim, hidden_dim=256,
            cond_dim=gru_hidden, diff_emb_dim=64,
        )

        # Noise schedule (linear)
        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def forward(self, history, future):
        """
        Training forward pass with teacher forcing.

        Args:
            history: (B, H, input_dim) history sequence
            future: (B, T_f, input_dim) future sequence (teacher forcing)
        Returns:
            loss: scalar diffusion loss
        """
        B, T_f, D = future.shape

        # Encode history
        _, h = self.gru(history)  # h: (1, B, gru_hidden)
        h = h.squeeze(0)  # (B, gru_hidden)

        total_loss = 0.0
        for step in range(T_f):
            x_0 = future[:, step, :]  # (B, D) ground truth

            # Sample random diffusion timestep
            t = torch.randint(0, self.n_steps, (B,), device=x_0.device)

            # Forward diffusion: add noise
            noise = torch.randn_like(x_0)
            sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            x_noisy = sqrt_alpha * x_0 + sqrt_one_minus * noise

            # Predict noise
            noise_pred = self.denoiser(x_noisy, t, h)
            total_loss += F.mse_loss(noise_pred, noise)

            # Teacher forcing: update GRU with ground truth
            _, h_new = self.gru(x_0.unsqueeze(1), h.unsqueeze(0))
            h = h_new.squeeze(0)

        return total_loss / T_f

    @torch.no_grad()
    def sample_trajectory(self, history, n_future=30, n_ddim_steps=20,
                          use_ddpm=False):
        """
        Generate future trajectory using DDIM or full DDPM sampling.

        Args:
            history: (B, H, input_dim)
            n_future: number of future steps
            n_ddim_steps: number of DDIM denoising steps (ignored if use_ddpm=True)
            use_ddpm: if True, use full DDPM reverse process (all n_steps steps)
        Returns:
            trajectory: (B, n_future, input_dim)
        """
        B = history.shape[0]
        device = history.device

        # Encode history
        _, h = self.gru(history)
        h = h.squeeze(0)

        trajectory = []
        for step in range(n_future):
            # Start from noise
            x = torch.randn(B, self.input_dim, device=device)

            if use_ddpm:
                # Full DDPM reverse process (matches original TimeGrad paper)
                for t_val in range(self.n_steps - 1, -1, -1):
                    t = torch.full((B,), t_val, device=device, dtype=torch.long)
                    noise_pred = self.denoiser(x, t, h)

                    alpha_t = self.alphas[t_val]
                    alpha_bar_t = self.alphas_cumprod[t_val]

                    # DDPM reverse step: x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ) + σ_t·z
                    coeff1 = 1.0 / alpha_t.sqrt()
                    coeff2 = self.betas[t_val] / (1 - alpha_bar_t).sqrt()
                    x = coeff1 * (x - coeff2 * noise_pred)

                    if t_val > 0:
                        # Posterior variance σ²_t = β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
                        alpha_bar_prev = self.alphas_cumprod[t_val - 1]
                        sigma = ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * self.betas[t_val]).sqrt()
                        x = x + sigma * torch.randn_like(x)
            else:
                # DDIM accelerated sampling
                step_indices = torch.linspace(0, self.n_steps - 1, n_ddim_steps + 1).long()
                timesteps = step_indices.flip(0)[:-1]

                for i, t_val in enumerate(timesteps):
                    t = torch.full((B,), t_val.item(), device=device, dtype=torch.long)
                    noise_pred = self.denoiser(x, t, h)

                    alpha_t = self.alphas_cumprod[t_val]
                    if i < len(timesteps) - 1:
                        alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
                    else:
                        alpha_prev = torch.tensor(1.0, device=device)

                    # DDIM update (deterministic, σ=0)
                    x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                    x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * noise_pred

            trajectory.append(x)

            # Update GRU with generated sample
            _, h_new = self.gru(x.unsqueeze(1), h.unsqueeze(0))
            h = h_new.squeeze(0)

        return torch.stack(trajectory, dim=1)  # (B, n_future, input_dim)


class TimeGradBaseline:
    """Wraps TimeGrad to match our BaselineModel interface."""

    def __init__(self, model, mean, std, device="cuda", ddim_steps=20,
                 use_ddpm=False):
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device
        self.ddim_steps = ddim_steps
        self.use_ddpm = use_ddpm

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def sample(self, history, n_samples=50, **kwargs):
        """
        Args:
            history: (B, 30, 5, 5) in [-1, 1] normalized space
        Returns:
            samples: (B, n_samples, 30, 5, 5) in [0, 1]
        """
        B = history.shape[0]
        device = self.device

        # Convert [-1,1] → [0,1] → z-score → flat (all on CPU first)
        history_cpu = history.cpu()
        history_01 = (history_cpu + 1) / 2  # [0,1]
        history_flat = history_01.reshape(B, 30, 25)
        history_z = (history_flat - self.mean.cpu()) / self.std.cpu()
        history_z = history_z.to(device)

        all_samples = []
        for _ in range(n_samples):
            traj = self.model.sample_trajectory(history_z, n_future=30,
                                                 n_ddim_steps=self.ddim_steps,
                                                 use_ddpm=self.use_ddpm)
            # traj: (B, 30, 25) z-scored
            traj_01 = traj * self.std + self.mean
            traj_01 = traj_01.clamp(0, 1)
            traj_55 = traj_01.reshape(B, 30, 5, 5)
            all_samples.append(traj_55)

        result = torch.stack(all_samples, dim=1)  # (B, n_samples, 30, 5, 5)
        return result.cpu()

    def sample_batched(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


def load_timegrad_model(checkpoint_path, device="cuda", use_ddpm=True):
    """Load a trained TimeGrad model from checkpoint.

    Args:
        use_ddpm: if True, use full DDPM reverse process (100 steps, matches paper).
                  if False, use DDIM acceleration (20 steps, faster but lower quality).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = TimeGradModel(
        input_dim=ckpt.get("input_dim", 25),
        gru_hidden=ckpt.get("gru_hidden", 128),
        n_diffusion_steps=ckpt.get("n_diffusion_steps", 100),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return TimeGradBaseline(
        model, ckpt["mean"], ckpt["std"], device,
        ddim_steps=ckpt.get("ddim_steps", 20),
        use_ddpm=use_ddpm,
    )
