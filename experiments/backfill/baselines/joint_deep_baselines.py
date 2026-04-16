"""Joint 38-d wrappers for deep baselines.

Wraps DeepVAR, TimeGrad, and CSDI for the multi-factor benchmark.
All operate on z-scored daily changes (not IV surface levels).

Interface:
    model.sample_joint(history_changes_z, n_samples) -> (B, n_samples, T, D) z-scored changes
    Caller inverse-z-scores and reconstructs IV.
"""

import numpy as np
import torch
from .classical_baselines import BaselineModel


class JointDeepVARBaseline(BaselineModel):
    """DeepVAR wrapper for 38-d joint forecasting."""

    def __init__(self, model, mean, std, device="cuda"):
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device

    def eval(self):
        self.model.eval()
        return self

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        """
        Args:
            history_changes: (B, T_hist, D) raw daily changes
        Returns:
            changes: (B, n_samples, T_fut, D) raw daily changes
        """
        B, T, D = history_changes.shape

        # Z-score
        hist_z = torch.tensor(
            (history_changes - self.mean.cpu().numpy()) / self.std.cpu().numpy(),
            dtype=torch.float32, device=self.device,
        )

        all_samples = []
        for _ in range(n_samples):
            traj = self.model.sample_trajectory(hist_z, n_future=30)
            # Inverse z-score
            traj_raw = traj * self.std + self.mean  # (B, 30, D)
            all_samples.append(traj_raw.cpu().numpy())

        result = np.stack(all_samples, axis=1)  # (B, n_samples, 30, D)
        return result.astype(np.float32)


class JointTimeGradBaseline(BaselineModel):
    """TimeGrad wrapper for 38-d joint forecasting."""

    def __init__(self, model, mean, std, device="cuda",
                 ddim_steps=20, use_ddpm=True):
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device
        self.ddim_steps = ddim_steps
        self.use_ddpm = use_ddpm

    def eval(self):
        self.model.eval()
        return self

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        B, T, D = history_changes.shape

        hist_z = torch.tensor(
            (history_changes - self.mean.cpu().numpy()) / self.std.cpu().numpy(),
            dtype=torch.float32, device=self.device,
        )

        all_samples = []
        for _ in range(n_samples):
            traj = self.model.sample_trajectory(
                hist_z, n_future=30,
                n_ddim_steps=self.ddim_steps, use_ddpm=self.use_ddpm,
            )
            traj_raw = traj * self.std + self.mean
            all_samples.append(traj_raw.cpu().numpy())

        result = np.stack(all_samples, axis=1)
        return result.astype(np.float32)


class JointCSDIBaseline(BaselineModel):
    """CSDI wrapper for 38-d joint forecasting.

    All 38 future dims are UNOBSERVED (mask=0). CSDI generates all dims
    at future timesteps. No oracle factor information.
    """

    def __init__(self, model, mean, std, device="cuda"):
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device
        self.D = len(mean)

    def eval(self):
        self.model.eval()
        return self

    def sample_joint(
        self, history_changes: np.ndarray, n_samples: int = 50, **kwargs
    ) -> np.ndarray:
        B, T_hist, D = history_changes.shape
        T_total = T_hist + 30  # 60
        device = self.device

        # Z-score history
        mean_np = self.mean.cpu().numpy()
        std_np = self.std.cpu().numpy()
        hist_z = (history_changes - mean_np) / std_np

        # Build CSDI input: (B, T_total, D)
        future_z = np.zeros((B, 30, D), dtype=np.float32)
        full_seq = np.concatenate([hist_z, future_z], axis=1)  # (B, 60, D)

        # Masks: history observed, ALL future dims unobserved
        observed_mask = np.ones((B, T_total, D), dtype=np.float32)
        gt_mask = observed_mask.copy()
        gt_mask[:, T_hist:, :] = 0.0  # ALL 38 future dims unobserved

        timepoints = np.broadcast_to(
            np.arange(T_total, dtype=np.float32)[None, :],
            (B, T_total),
        ).copy()

        batch = {
            "observed_data": torch.tensor(full_seq, dtype=torch.float32, device=device),
            "observed_mask": torch.tensor(observed_mask, dtype=torch.float32, device=device),
            "gt_mask": torch.tensor(gt_mask, dtype=torch.float32, device=device),
            "timepoints": torch.tensor(timepoints, dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            # CSDI returns (samples, observed_data, target_mask, observed_mask, observed_tp)
            # samples: (B, n_samples, K=D, L=T_total)
            output = self.model.evaluate(batch, n_samples)
            samples = output[0]  # (B, n_samples, D, T_total)

            # Extract future portion
            future_samples = samples[:, :, :, T_hist:]  # (B, n_samples, D, 30)
            future_samples = future_samples.permute(0, 1, 3, 2)  # (B, n_samples, 30, D)

            # Inverse z-score
            future_raw = future_samples * self.std + self.mean

        return future_raw.cpu().numpy().astype(np.float32)


# --- Loading functions ---

def load_joint_deepvar(checkpoint_path, device="cuda"):
    """Load a 38-d DeepVAR from checkpoint."""
    from .deepvar_standalone import DeepVARModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeepVARModel(
        input_dim=ckpt.get("input_dim", 38),
        lstm_hidden=ckpt.get("lstm_hidden", 128),
        lstm_layers=ckpt.get("lstm_layers", 2),
        rank=ckpt.get("rank", 5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return JointDeepVARBaseline(model, ckpt["mean"], ckpt["std"], device)


def load_joint_timegrad(checkpoint_path, device="cuda", use_ddpm=True):
    """Load a 38-d TimeGrad from checkpoint."""
    from .timegrad_standalone import TimeGradModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TimeGradModel(
        input_dim=ckpt.get("input_dim", 38),
        gru_hidden=ckpt.get("gru_hidden", 128),
        n_diffusion_steps=ckpt.get("n_diffusion_steps", 100),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return JointTimeGradBaseline(
        model, ckpt["mean"], ckpt["std"], device,
        ddim_steps=ckpt.get("ddim_steps", 20), use_ddpm=use_ddpm,
    )


def load_joint_csdi(checkpoint_path, device="cuda"):
    """Load a 38-d CSDI from checkpoint."""
    import sys, types
    from pathlib import Path

    # Patch linear_attention_transformer (same as csdi_adapter.py)
    if "linear_attention_transformer" not in sys.modules:
        _lat = types.ModuleType("linear_attention_transformer")
        _lat.LinearAttentionTransformer = None
        sys.modules["linear_attention_transformer"] = _lat

    CSDI_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "external" / "csdi")
    if CSDI_DIR not in sys.path:
        sys.path.insert(0, CSDI_DIR)

    from main_model import CSDI_Forecasting

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = CSDI_Forecasting(config, device, target_dim=ckpt.get("target_dim", 38)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return JointCSDIBaseline(model, ckpt["mean"], ckpt["std"], device)
