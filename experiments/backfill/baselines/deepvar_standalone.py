"""
Standalone DeepVAR implementation for IV surface scenario generation.

Reimplements the core of Salinas et al. (NeurIPS 2019), "High-Dimensional
Multivariate Forecasting with Low-Rank Gaussian Copula Processes".

Architecture:
  LSTM encoder processes history → hidden state h_0
  For each future step t (autoregressive):
    1. h_t → MLP → (μ, D, V) parametrizing N(μ, diag(D) + V·Vᵀ)
    2. Sample x_t ~ N(μ, Σ)
    3. Update h_{t+1} = LSTM(x_t, h_t)

Differences from full GluonTS DeepVAREstimator:
  - No copula transformation (we use the low-rank Gaussian variant directly,
    which is the most commonly benchmarked version in the time series literature)
  - No lag features or time features (IV surfaces lack calendar effects like
    day-of-week or holidays that these features target)
  - No input scaling (z-score normalization applied externally in the adapter)
  - Standalone PyTorch (GluonTS requires MXNet, incompatible with Python 3.13)
  - rank=5 matches GluonTS default for LowrankMultivariateGaussianOutput

Reference: Salinas et al., NeurIPS 2019 (arXiv:1910.03002)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal


class DeepVARModel(nn.Module):
    """
    DeepVAR: autoregressive LSTM with low-rank multivariate Gaussian output.

    At each timestep, the LSTM hidden state is projected to parameters of a
    low-rank multivariate Gaussian: mean μ, diagonal D, and low-rank factor V,
    giving covariance Σ = diag(D) + V·Vᵀ.
    """

    def __init__(self, input_dim=25, lstm_hidden=128, lstm_layers=2,
                 rank=5, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.rank = rank

        # LSTM encoder/state tracker
        self.lstm = nn.LSTM(
            input_dim, lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Project hidden state → distribution parameters
        self.mu_head = nn.Linear(lstm_hidden, input_dim)
        self.diag_head = nn.Linear(lstm_hidden, input_dim)  # log-space diagonal
        self.lowrank_head = nn.Linear(lstm_hidden, input_dim * rank)

    def _get_distribution(self, h_last):
        """Map LSTM hidden state to a LowRankMultivariateNormal distribution.

        Args:
            h_last: (B, lstm_hidden) last-layer hidden state
        Returns:
            dist: LowRankMultivariateNormal with batch shape (B,)
        """
        mu = self.mu_head(h_last)  # (B, input_dim)
        log_diag = self.diag_head(h_last)  # (B, input_dim)
        diag = torch.nn.functional.softplus(log_diag) + 0.01  # min variance floor
        V = self.lowrank_head(h_last).reshape(-1, self.input_dim, self.rank)  # (B, D, rank)

        return LowRankMultivariateNormal(mu, V, diag)

    def forward(self, history, future):
        """
        Training forward pass with teacher forcing.

        Args:
            history: (B, H, input_dim) history sequence
            future: (B, T_f, input_dim) future sequence (teacher forcing)
        Returns:
            loss: scalar negative log-likelihood
        """
        B, T_f, D = future.shape

        # Encode history
        _, (h, c) = self.lstm(history)  # h: (n_layers, B, hidden)

        total_nll = 0.0
        # Feed last history step to get first prediction context
        prev_x = history[:, -1:, :]  # (B, 1, D)

        for step in range(T_f):
            # Single LSTM step
            _, (h, c) = self.lstm(prev_x, (h, c))

            # Get distribution from last-layer hidden state
            h_last = h[-1]  # (B, hidden) — last LSTM layer
            dist = self._get_distribution(h_last)

            # NLL of ground truth under predicted distribution
            x_true = future[:, step, :]  # (B, D)
            total_nll -= dist.log_prob(x_true).mean()

            # Teacher forcing: feed ground truth
            prev_x = x_true.unsqueeze(1)  # (B, 1, D)

        return total_nll / T_f

    @torch.no_grad()
    def sample_trajectory(self, history, n_future=30):
        """
        Generate future trajectory by sampling autoregressively.

        Args:
            history: (B, H, input_dim)
            n_future: number of future steps
        Returns:
            trajectory: (B, n_future, input_dim)
        """
        B = history.shape[0]

        # Encode history
        _, (h, c) = self.lstm(history)

        trajectory = []
        prev_x = history[:, -1:, :]

        for step in range(n_future):
            _, (h, c) = self.lstm(prev_x, (h, c))

            h_last = h[-1]
            dist = self._get_distribution(h_last)

            # Sample from low-rank multivariate Gaussian
            x = dist.rsample()  # (B, input_dim)
            trajectory.append(x)

            prev_x = x.unsqueeze(1)

        return torch.stack(trajectory, dim=1)  # (B, n_future, input_dim)


class DeepVARBaseline:
    """Wraps DeepVAR to match our BaselineModel interface."""

    def __init__(self, model, mean, std, device="cuda"):
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device

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

        # Convert [-1,1] → [0,1] → z-score → flat
        history_cpu = history.cpu()
        history_01 = (history_cpu + 1) / 2  # [0,1]
        history_flat = history_01.reshape(B, 30, 25)
        history_z = (history_flat - self.mean.cpu()) / self.std.cpu()
        history_z = history_z.to(device)

        all_samples = []
        for _ in range(n_samples):
            traj = self.model.sample_trajectory(history_z, n_future=30)
            # traj: (B, 30, 25) z-scored
            traj_01 = traj * self.std + self.mean
            traj_01 = traj_01.clamp(0, 1)
            traj_55 = traj_01.reshape(B, 30, 5, 5)
            all_samples.append(traj_55)

        result = torch.stack(all_samples, dim=1)  # (B, n_samples, 30, 5, 5)
        return result.cpu()

    def sample_batched(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


def load_deepvar_model(checkpoint_path, device="cuda"):
    """Load a trained DeepVAR model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = DeepVARModel(
        input_dim=ckpt.get("input_dim", 25),
        lstm_hidden=ckpt.get("lstm_hidden", 128),
        lstm_layers=ckpt.get("lstm_layers", 2),
        rank=ckpt.get("rank", 5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return DeepVARBaseline(model, ckpt["mean"], ckpt["std"], device)
