"""
VAE (CVAEMemRand) adapter for unified baseline evaluation.

Wraps the CVAEMemRand model to match the BaselineModel .sample() interface
used by evaluate_baselines.py. The VAE generates one-step-ahead predictions
autoregressively to produce 30-day future trajectories.

The VAE requires extra features (ret, skew, slope) alongside surfaces.
These are precomputed from the data file.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vae.cvae_with_mem_randomized import CVAEMemRand


class VAEBaseline:
    """Wraps CVAEMemRand to match our BaselineModel interface."""

    def __init__(self, model, context_len=20, device="cpu"):
        """
        Args:
            model: trained CVAEMemRand instance
            context_len: context length the model was trained with
            device: torch device
        """
        self.model = model
        self.context_len = context_len
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

        # Convert [-1,1] → [0,1]
        history_01 = (history.cpu() + 1) / 2  # (B, 30, 5, 5) in [0,1]

        # Use last context_len days of history for VAE context
        C = min(self.context_len, 30)
        ctx_surfaces = history_01[:, -C:, :, :]  # (B, C, 5, 5)

        # Use zeros for ex_feats (passive conditioning — conservative baseline)
        ctx_ex_feats = torch.zeros(B, C, 3, dtype=torch.float64)

        all_samples = []
        for _ in range(n_samples):
            context = {
                "surface": ctx_surfaces.double().to(self.device),
                "ex_feats": ctx_ex_feats.to(self.device),
            }

            with torch.no_grad():
                result = self.model.generate_autoregressive_sequence(
                    context, horizon=30
                )
                if isinstance(result, tuple):
                    surfaces_out = result[0]  # (B, 30, Q, 5, 5) or (B, 30, 5, 5)
                else:
                    surfaces_out = result

            # Clamp to [0, 1]
            surfaces_out = surfaces_out.float().clamp(0, 1)
            all_samples.append(surfaces_out)

        result = torch.stack(all_samples, dim=1)  # (B, n_samples, 30, 5, 5)
        return result.cpu()

    def sample_batched(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


def _parse_config(raw):
    """Convert string-valued config dict to proper Python types."""
    config = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                config[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                config[k] = v
        else:
            config[k] = v
    return config


def _extract_median_from_quantile_checkpoint(model, state_dict):
    """Extract median channel weights from a quantile regression checkpoint.

    The checkpoint was trained with quantile regression (output 3 channels:
    q05, q50, q95) but the current CVAEMemRand code creates 1-channel output.
    We extract only the median channel (index 1) weights so the model produces
    standard single-surface output.
    """
    dec_output_weight_key = "decoder.surface_decoder.dec_output.weight"
    dec_output_bias_key = "decoder.surface_decoder.dec_output.bias"

    if dec_output_weight_key not in state_dict:
        return False

    ckpt_out_channels = state_dict[dec_output_weight_key].shape[0]
    if ckpt_out_channels <= 1:
        return False

    # Extract median channel (index 1 of [q05, q50, q95])
    median_idx = 1
    state_dict[dec_output_weight_key] = state_dict[dec_output_weight_key][median_idx:median_idx+1]
    if dec_output_bias_key in state_dict:
        state_dict[dec_output_bias_key] = state_dict[dec_output_bias_key][median_idx:median_idx+1]

    return True


def load_vae_model(checkpoint_path, device="cpu"):
    """Load a trained CVAEMemRand from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint
        device: torch device
    Returns:
        VAEBaseline wrapper
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config — checkpoint stores values as strings, need to convert
    raw_config = ckpt.get("model_config", ckpt.get("config", {}))
    config = _parse_config(raw_config)
    config["device"] = device

    # Build model
    model = CVAEMemRand(config).to(device)

    # Load state dict (checkpoint stores it under "model" key)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))

    # Handle quantile regression checkpoint: extract median channel weights
    _extract_median_from_quantile_checkpoint(model, state_dict)

    # Filter out quantile_loss_fn buffers (not part of model architecture)
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("quantile_loss_fn")}

    model.load_state_dict(state_dict)
    model.double()  # VAE was trained in float64
    model.eval()

    context_len = config.get("context_len", 20)

    return VAEBaseline(model, context_len=context_len, device=device)
