"""
Configuration for backfilling experiments with limited training data.

This config supports training on 1, 2, or 3 years of recent data before the test set,
then using the trained model to generate 30-day backfill sequences for the 2008-2010
financial crisis period.
"""


class BackfillConfig:
    """Configuration for limited data training and backfilling."""

    # ========================================================================
    # Training Data Parameters (CONFIGURABLE)
    # ========================================================================

    train_period_years = 3  # Options: 1, 2, or 3 years
    train_end_idx = 5000    # Use recent data (before test set starts at 5000)

    # Auto-compute start index (assuming ~250 trading days per year)
    train_start_idx = train_end_idx - (250 * train_period_years)

    # ========================================================================
    # Backfill Period (Historical Period to Generate)
    # ========================================================================

    backfill_start_idx = 2000   # Start of 2008 crisis period
    backfill_end_idx = 2765     # End of 2010 period
    backfill_horizon = 30        # Days to generate per window

    # ========================================================================
    # Model Hyperparameters (Reuse Existing Quantile Architecture)
    # ========================================================================

    context_len = 5       # Initial context window (can increase to 30 later)
    latent_dim = 5        # Latent space dimensionality
    mem_hidden = 100      # LSTM hidden size
    mem_layers = 2        # LSTM layers
    mem_dropout = 0.3     # LSTM dropout rate
    kl_weight = 1e-5      # KL divergence weight

    # Surface encoding layers
    surface_hidden = [5, 5, 5]

    # Quantile regression configuration
    use_quantile_regression = True
    num_quantiles = 3
    quantiles = [0.05, 0.5, 0.95]

    # ========================================================================
    # Training Parameters
    # ========================================================================

    epochs = 400
    batch_size = 64
    learning_rate = 1e-5

    # Scheduled sampling (Phase 2.2 from BACKFILL_MVP_PLAN.md)
    teacher_forcing_epochs = 200  # Phase 1: standard teacher forcing
    # Phase 2: multi-horizon training starts after teacher_forcing_epochs

    # ========================================================================
    # Multi-Horizon Training (Phase 2.1 from BACKFILL_MVP_PLAN.md)
    # ========================================================================

    training_horizons = [1, 7, 14, 30]
    horizon_weights = {1: 1.0, 7: 0.8, 14: 0.6, 30: 0.4}

    # ========================================================================
    # Evaluation Parameters
    # ========================================================================

    eval_horizons = [1, 7, 14, 30]

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @classmethod
    def get_train_indices(cls):
        """Get training data indices."""
        return cls.train_start_idx, cls.train_end_idx

    @classmethod
    def get_backfill_indices(cls):
        """Get backfill period indices."""
        return cls.backfill_start_idx, cls.backfill_end_idx

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("BACKFILL CONFIGURATION")
        print("=" * 80)
        print()
        print("Training Configuration:")
        print(f"  Training period: {cls.train_period_years} years")
        print(f"  Training indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print(f"  Training days: {cls.train_end_idx - cls.train_start_idx}")
        print(f"  Epochs: {cls.epochs}")
        print(f"  Batch size: {cls.batch_size}")
        print(f"  Learning rate: {cls.learning_rate}")
        print()
        print("Scheduled Sampling:")
        print(f"  Teacher forcing epochs: {cls.teacher_forcing_epochs} (Phase 1)")
        print(f"  Multi-horizon epochs: {cls.epochs - cls.teacher_forcing_epochs} (Phase 2)")
        print(f"  Training horizons: {cls.training_horizons}")
        print()
        print("Backfill Configuration:")
        print(f"  Backfill indices: [{cls.backfill_start_idx}, {cls.backfill_end_idx}]")
        print(f"  Backfill days: {cls.backfill_end_idx - cls.backfill_start_idx}")
        print(f"  Backfill horizon: {cls.backfill_horizon} days")
        print()
        print("Model Architecture:")
        print(f"  Context length: {cls.context_len} days")
        print(f"  Latent dim: {cls.latent_dim}")
        print(f"  Memory hidden: {cls.mem_hidden}")
        print(f"  Quantile regression: {cls.use_quantile_regression}")
        print(f"  Quantiles: {cls.quantiles}")
        print("=" * 80)
