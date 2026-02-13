"""
Configuration for Block-AR DDPM on Volatility Surfaces.

Block-AR Diffusion with MCVD multi-task training and Diffusion Forcing
noise schedules. Targets improved CI coverage over the DDPM POC baseline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BlockARPOCConfig:
    """POC configuration for Block-AR DDPM on volatility surfaces."""

    # === Data Configuration ===
    data_path: str = "data/vol_surface_with_ret.npz"
    history_len: int = 30
    future_len: int = 30
    surface_h: int = 5
    surface_w: int = 5
    block_size: int = 10

    # === Encoder ===
    gru_hidden_dim: int = 64
    bottleneck_dim: int = 64
    cond_aug_sigma: float = 0.0
    encoder_dropout: float = 0.1

    # === Denoiser ===
    bigru_hidden_dim: int = 128
    pos_embed_dim: int = 16
    noise_embed_dim: int = 16
    denoiser_dropout: float = 0.1

    # === Diffusion Process ===
    n_steps: int = 100
    schedule: str = "cosine"

    # === MCVD ===
    p_mask: float = 0.2
    jitter_std: float = 0.15

    # === PYoCo correlated noise ===
    noise_rho: float = 0.5

    # === Sampling ===
    max_residual_timestep: int = 20

    # === Training ===
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # Data split (same as DDPM POC for fair comparison)
    train_end: int = 4040
    val_start: int = 4040
    val_end: int = 4540
    test_start: int = 4540

    # === Evaluation ===
    n_eval_samples: int = 10
    eval_horizons: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    ci_levels: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95])

    # === Output ===
    output_dir: str = "models/backfill/block_ar"
    checkpoint_every: int = 10

    # === Device ===
    device: str = "cuda"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> str:
        return f"{self.output_dir}/best_model.pt"

    @property
    def results_dir(self) -> str:
        return "results/block_ar"


def get_default_config() -> BlockARPOCConfig:
    return BlockARPOCConfig()


def get_fast_test_config() -> BlockARPOCConfig:
    return BlockARPOCConfig(
        n_steps=20,
        batch_size=32,
        epochs=5,
        n_eval_samples=5,
        bigru_hidden_dim=64,
        gru_hidden_dim=32,
    )


if __name__ == "__main__":
    config = get_default_config()
    print("Block-AR POC Default Configuration:")
    print("-" * 40)
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        print(f"  {field_name}: {value}")
