"""
Configuration for MCVD POC on Volatility Surfaces.

Adapts MCVD (Masked Conditional Video Diffusion) for IV surface forecasting:
- 2D U-Net with frames as channels (not 3D conv)
- Frame concatenation conditioning (history frames as input channels)
- Masking training (zero out history with probability)
- Pad 5×5 surfaces to 8×8 for proper U-Net hierarchy (8→4→2)
"""

import argparse
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class MCVDPOCConfig:
    """POC configuration for MCVD on volatility surfaces."""

    # === Data Configuration ===
    data_path: str = "data/vol_surface_with_ret.npz"
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5  # Original spatial size
    surface_w: int = 5
    padded_h: int = 8   # Padded spatial size for U-Net
    padded_w: int = 8

    # === MCVD Model Architecture ===
    arch: str = "unetmore"       # Pure 2D U-Net (frames as channels)
    ngf: int = 24                # Base feature channels
    num_res_blocks: int = 2      # ResBlocks per U-Net level
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2])  # 3 levels: 8→4→2
    attn_resolutions: List[int] = field(default_factory=lambda: [8])
    n_head_channels: int = 8   # Must divide all channel counts: 32/8=4, 64/8=8 heads
    dropout: float = 0.1

    # === Diffusion Process ===
    n_steps: int = 100           # Noise schedule steps (match our DDPM baseline)
    schedule: str = "cosine"     # Noise schedule type: "cosine" or "linear"

    # === MCVD Conditioning ===
    prob_mask_cond: float = 0.1  # Zero out history 10% of training batches
    cond_emb: bool = True        # Enable masking embedding in model
    noise_in_cond: bool = True   # Noise conditioning frames to match target noise level

    # === Training ===
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5

    # Data split (same as DDPM baseline for fair comparison)
    train_end: int = 4040
    val_start: int = 4040
    val_end: int = 4540
    test_start: int = 4540

    # === Evaluation ===
    n_eval_samples: int = 100
    eval_horizons: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    ci_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95])

    # === Output ===
    output_dir: str = "models/backfill/mcvd_poc"
    checkpoint_every: int = 10
    log_every: int = 50

    # === Device ===
    device: str = "cuda"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> str:
        return f"{self.output_dir}/best_model.pt"

    @property
    def results_dir(self) -> str:
        return "results/mcvd_poc"


def build_mcvd_config(poc_config: MCVDPOCConfig) -> argparse.Namespace:
    """Build the nested namespace config that MCVD's NCSNpp expects.

    MCVD uses config.data.xxx and config.model.xxx attribute access,
    so we create an argparse.Namespace tree.
    """
    config = argparse.Namespace()

    # Data config
    config.data = argparse.Namespace()
    config.data.image_size = poc_config.padded_h  # 8
    config.data.channels = 1                       # IV surface = 1 channel
    config.data.num_frames = poc_config.future_len  # 30 frames to predict
    config.data.num_frames_cond = poc_config.history_len  # 30 history frames
    config.data.num_frames_future = 0
    config.data.prob_mask_cond = poc_config.prob_mask_cond

    # Model config
    config.model = argparse.Namespace()
    config.model.arch = poc_config.arch
    config.model.version = "DDPM"
    config.model.ngf = poc_config.ngf
    config.model.num_res_blocks = poc_config.num_res_blocks
    config.model.ch_mult = tuple(poc_config.ch_mult)
    config.model.attn_resolutions = tuple(poc_config.attn_resolutions)
    config.model.n_head_channels = poc_config.n_head_channels
    config.model.dropout = poc_config.dropout
    config.model.time_conditional = True
    config.model.cond_emb = poc_config.cond_emb
    config.model.output_all_frames = False
    config.model.noise_in_cond = poc_config.noise_in_cond
    config.model.spade = False
    config.model.gamma = False

    # Noise schedule
    config.model.sigma_dist = poc_config.schedule
    config.model.num_classes = poc_config.n_steps  # get_sigmas() uses this
    config.model.sigma_begin = 0.02
    config.model.sigma_end = 0.0001
    config.model.num_scales = poc_config.n_steps

    # Device (needed by get_sigmas)
    config.device = poc_config.device

    return config


def get_default_config() -> MCVDPOCConfig:
    """Get default MCVD POC configuration."""
    return MCVDPOCConfig()


def get_paper_aligned_config() -> MCVDPOCConfig:
    """Config aligned with MCVD paper (SMMNIST big5) for fair comparison.

    Key changes from default:
    - 5-frame blocks (not 30) matching paper; use AR rollout for 30-day forecasts
    - ngf=64 matching paper; first-conv ratio = 10/64 = 0.16 (paper: 0.16)
    - Linear schedule T=1000 matching paper shipped configs
    - noise_in_cond=False, prob_mask_cond=0.0, cond_emb=False matching specialist
    - LR=2e-4 matching paper Adam optimizer settings
    """
    return MCVDPOCConfig(
        history_len=5,
        future_len=5,
        ngf=64,
        n_head_channels=64,
        n_steps=1000,
        schedule="linear",
        noise_in_cond=False,
        prob_mask_cond=0.0,
        cond_emb=False,
        lr=2e-4,
        weight_decay=0.0,
        epochs=500,
        warmup_epochs=16,
        output_dir="models/backfill/mcvd_paper_aligned",
    )


def get_fast_test_config() -> MCVDPOCConfig:
    """Get configuration for fast testing."""
    return MCVDPOCConfig(
        ngf=16,
        num_res_blocks=1,
        n_steps=20,
        batch_size=32,
        epochs=5,
        n_eval_samples=10,
    )


if __name__ == "__main__":
    config = get_default_config()
    print("MCVD POC Default Configuration:")
    print("-" * 40)
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        print(f"  {field_name}: {value}")

    print("\nMCVD Namespace Config:")
    print("-" * 40)
    mcvd_config = build_mcvd_config(config)
    for key in ['data', 'model']:
        ns = getattr(mcvd_config, key)
        print(f"  {key}:")
        for k, v in vars(ns).items():
            print(f"    {k}: {v}")
