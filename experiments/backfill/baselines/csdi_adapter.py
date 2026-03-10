"""
CSDI adapter for IV surface scenario generation.

Wraps the CSDI (Conditional Score-based Diffusion) model from Tashiro et al. (NeurIPS 2021)
for our IV surface forecasting task: 30-day history → 30-day future on 5×5 grids.

CSDI operates on (B, K, L) tensors where K=features, L=timesteps.
Our data: (B, T, 5, 5) surfaces → flatten to (B, T, 25) → transpose to (B, 25, T).

Reference: https://github.com/ermongroup/CSDI
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add CSDI to path — patch out linear_attention_transformer dependency
# (we use is_linear=False so standard PyTorch Transformer is used instead)
import types
_lat = types.ModuleType("linear_attention_transformer")
_lat.LinearAttentionTransformer = None
sys.modules["linear_attention_transformer"] = _lat

CSDI_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "external" / "csdi")
if CSDI_DIR not in sys.path:
    sys.path.insert(0, CSDI_DIR)

from main_model import CSDI_Forecasting


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IVSurfaceForecastingDataset(Dataset):
    """Adapts our IV surface data to CSDI's expected format."""

    def __init__(self, surfaces, start_idx=0, end_idx=None,
                 history_len=30, future_len=30, mean=None, std=None):
        """
        Args:
            surfaces: (N, 5, 5) array of IV surfaces in [0, 1]
            start_idx, end_idx: slice indices for train/val/test
            history_len, future_len: window sizes
            mean, std: per-feature normalization (computed from train if None)
        """
        end_idx = end_idx or len(surfaces)
        self.surfaces = surfaces[start_idx:end_idx]  # (N, 5, 5)
        self.history_len = history_len
        self.future_len = future_len
        self.seq_len = history_len + future_len

        # Flatten to (N, 25)
        flat = self.surfaces.reshape(-1, 25)

        if mean is None:
            self.mean = flat.mean(axis=0)  # (25,)
            self.std = flat.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        # Z-score normalize
        self.data = (flat - self.mean) / self.std  # (N, 25)

        self.n_windows = len(self.data) - self.seq_len + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]  # (60, 25)

        observed_mask = np.ones_like(seq)  # all observed
        gt_mask = observed_mask.copy()
        gt_mask[self.history_len:] = 0.0  # mask future for forecasting

        return {
            "observed_data": seq.astype(np.float32),
            "observed_mask": observed_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            "timepoints": np.arange(self.seq_len, dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_csdi_config():
    """CSDI config adapted for IV surface forecasting (25 features, 60 timesteps)."""
    return {
        "train": {
            "epochs": 200,
            "batch_size": 16,
            "lr": 1.0e-3,
            "itr_per_epoch": int(1e8),  # effectively unlimited
        },
        "diffusion": {
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "num_steps": 50,
            "schedule": "quad",
            "is_linear": False,  # Use standard PyTorch Transformer (no linear_attention_transformer dep)
        },
        "model": {
            "is_unconditional": 0,
            "timeemb": 128,
            "featureemb": 16,
            "target_strategy": "test",  # forecasting: condition=history, target=future
            "num_sample_features": 25,  # all 25 features (5×5 grid), no subsampling
        },
    }


# ---------------------------------------------------------------------------
# Wrapper for our evaluation pipeline
# ---------------------------------------------------------------------------

class CSDIBaseline:
    """Wraps CSDI model to match our BaselineModel interface."""

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

        # Convert from [-1,1] to [0,1] then z-score (all on CPU first)
        history_cpu = history.cpu()
        history_01 = (history_cpu + 1) / 2  # (B, 30, 5, 5) in [0,1]
        history_flat = history_01.reshape(B, 30, 25)  # (B, 30, 25)
        mean_cpu = self.mean.cpu()
        std_cpu = self.std.cpu()
        history_z = (history_flat - mean_cpu) / std_cpu

        # Create dummy future (zeros, will be generated)
        future_z = torch.zeros(B, 30, 25)
        full_seq = torch.cat([history_z, future_z], dim=1)  # (B, 60, 25)

        # Masks
        observed_mask = torch.ones(B, 60, 25)
        gt_mask = observed_mask.clone()
        gt_mask[:, 30:, :] = 0.0  # mask future

        timepoints = torch.arange(60, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        batch = {
            "observed_data": full_seq.to(device),
            "observed_mask": observed_mask.to(device),
            "gt_mask": gt_mask.to(device),
            "timepoints": timepoints.to(device),
        }

        with torch.no_grad():
            # CSDI_Forecasting.evaluate returns:
            # (samples, observed_data, target_mask, observed_mask, observed_tp)
            # samples: (B, n_samples, K, L) where K=25, L=60
            output = self.model.evaluate(batch, n_samples)
            samples = output[0]  # (B, n_samples, K=25, L=60)

            # Extract future portion (last 30 timesteps)
            future_samples = samples[:, :, :, 30:]  # (B, n_samples, 25, 30)
            future_samples = future_samples.permute(0, 1, 3, 2)  # (B, n_samples, 30, 25)

            # Inverse z-score → [0,1]
            future_01 = future_samples * self.std + self.mean  # (B, n_samples, 30, 25)
            future_01 = future_01.clamp(0, 1)

            # Reshape to (B, n_samples, 30, 5, 5)
            result = future_01.reshape(B, n_samples, 30, 5, 5)

        return result.cpu()

    def sample_batched(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csdi_model(checkpoint_path, device="cuda"):
    """Load a trained CSDI model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    mean = ckpt["mean"]
    std = ckpt["std"]

    model = CSDI_Forecasting(config, device, target_dim=25).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return CSDIBaseline(model, mean, std, device)
