"""
Configuration for Causal 3D VAE training on volatility surfaces.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Causal3DTrainingConfig:
    """Training configuration for Causal 3D VAE."""

    # Data
    data_path: str = "data/vol_surface_with_ret.npz"
    train_split: float = 0.8  # 80% train, 20% test

    # Sequence parameters (120 days total to capture mean-reversion dynamics)
    context_length: int = 60  # Number of frames in context
    prediction_horizon: int = 60  # Number of frames to predict autoregressively

    # Model architecture (scaled down for 5x5 single-channel data)
    latent_channels: int = 4
    block_channels: Tuple[int, ...] = (8, 16, 32)
    layers_per_block: int = 1
    temporal_compression: int = 2  # T -> T/2 in latent space
    norm_num_groups: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    kl_weight: float = 1e-6

    # Scheduled sampling for exposure bias
    # Start with 100% teacher forcing, linearly decay to min_teacher_forcing
    scheduled_sampling: bool = True
    min_teacher_forcing: float = 0.5  # Minimum teacher forcing ratio at end
    scheduled_sampling_warmup: int = 20  # Epochs before starting decay

    # Autoregressive training
    # Train with actual AR chaining some fraction of the time
    ar_training_fraction: float = 0.3  # Fraction of batches trained with AR chaining
    ar_training_start_epoch: int = 50  # Start AR training after this epoch
    ar_training_steps: int = 10  # Number of AR steps per training batch (reduce for memory)

    # Logging and checkpointing
    log_interval: int = 100  # Log every N batches
    save_interval: int = 10  # Save checkpoint every N epochs
    checkpoint_dir: str = "models/backfill/causal_3d/"

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    # Reproducibility
    seed: int = 42


@dataclass
class Causal3DEvalConfig:
    """Evaluation configuration for Causal 3D VAE."""

    # Model
    model_path: str = "models/backfill/causal_3d/best_model.pt"

    # Evaluation parameters
    num_samples: int = 100  # Number of samples per prediction
    context_length: int = 60
    prediction_horizon: int = 60

    # Output
    output_dir: str = "results/causal_3d/"

    # CI coverage levels to evaluate
    ci_levels: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)

    # Device
    device: str = "cuda"
