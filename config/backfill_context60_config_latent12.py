"""
Configuration for Context=60 2-Phase Training with Latent Bottleneck Fix

Training Schedule:
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1,7,14,30,60,90] - 400 EPOCHS

Key Changes from Original:
- latent_dim: 5 → 12 (2.4× capacity)
- kl_weight: 1e-5 → 5e-5 (5× stronger regularization)
- Phase 2: Extended to 400 epochs (was 150)
- phase2_weights: UNIFORM (all 1.0 for equal horizon exposure)
- quantile_loss_weights: [1.0, 1.0, 1.0] (pinball loss has built-in asymmetry)
- Precision: float32 (2× speedup, ~23 hours training)
- Phases 3-4: REMOVED (focus on single-pass generation)

Expected Impact:
- Day-1 spread: 0.0858 → 0.025-0.03 (70% reduction)
- Latent correlation: 0.212 → 0.4-0.5
- PC1: 99.94% → <90%

CRITICAL: Saves checkpoint after EACH phase for comparison.
"""


class BackfillContext60ConfigLatent12:
    """Configuration for Context=60 model with 2-phase training and latent bottleneck fix."""

    # ============================================================================
    # Data Configuration
    # ============================================================================

    train_period_years = 16  # Use full 16-year dataset
    train_start_idx = 1000
    train_end_idx = 5000

    # ============================================================================
    # Model Architecture - LATENT BOTTLENECK FIX
    # ============================================================================

    context_len = 60
    latent_dim = 12         # CHANGED: Was 5 (2.4× capacity)
    mem_hidden = 100
    mem_layers = 2
    mem_dropout = 0.3
    surface_hidden = [5, 5, 5]

    # ============================================================================
    # Training Schedule (2 Phases) - EXTENDED PHASE 2
    # ============================================================================

    total_epochs = 600

    # Phase boundaries
    phase1_end = 200    # Teacher forcing
    phase2_end = 600    # Multi-horizon - CHANGED: Was 350 (400 epochs instead of 150)

    # Sequence length requirements per phase
    # Format: (min_seq_len, max_seq_len)
    phase1_seq_len = (61, 80)       # context=60 + horizon=1 + buffer
    phase2_seq_len = (150, 180)     # context=60 + horizon=90 + buffer

    # ============================================================================
    # Phase 2: Multi-Horizon Training - UNIFORM WEIGHTS
    # ============================================================================

    phase2_horizons = [1, 7, 14, 30, 60, 90]
    phase2_weights = {
        1: 1.0,     # CHANGED: Equal exposure to all horizons
        7: 1.0,     # CHANGED: Was 0.9
        14: 1.0,    # CHANGED: Was 0.8
        30: 1.0,    # CHANGED: Was 0.6
        60: 1.0,    # CHANGED: Was 0.4
        90: 1.0     # CHANGED: Was 0.3
    }

    # ============================================================================
    # Optimization
    # ============================================================================

    learning_rate = 1e-5
    batch_size = 128  # 3070 Ti verified: uses only ~7% GPU memory (576 MB peak)
    valid_batch_size = 256  # Larger batch for validation (no gradients needed)

    # ============================================================================
    # Loss Configuration
    # ============================================================================

    kl_weight = 5e-5  # CHANGED: Was 1e-5 (5× stronger regularization)

    # Quantile regression
    use_quantile_regression = True
    num_quantiles = 3
    quantiles = [0.05, 0.5, 0.95]
    quantile_loss_weights = [1.0, 1.0, 1.0]  # CHANGED: Was [5.0, 1.0, 5.0] (pinball loss has built-in asymmetry)

    # Extra features
    re_feat_weight = 1.0
    ex_loss_on_ret_only = True
    ex_feats_loss_type = "l2"

    # ============================================================================
    # Checkpoint Configuration
    # ============================================================================

    checkpoint_dir = "models/backfill/context60_experiment/checkpoints"
    checkpoint_prefix = "backfill_context60_latent12"

    # Checkpoint naming convention
    @classmethod
    def get_checkpoint_name(cls, epoch):
        """
        Generate checkpoint filename for given epoch.

        Args:
            epoch: Epoch number

        Returns:
            str: Checkpoint filename (e.g., "backfill_context60_latent12_phase1_ep199.pt")
        """
        if epoch == 199 or epoch == cls.phase1_end - 1:
            return f"{cls.checkpoint_prefix}_phase1_ep199.pt"
        elif epoch == 599 or epoch == cls.phase2_end - 1:
            return f"{cls.checkpoint_prefix}_phase2_ep599.pt"
        else:
            return f"{cls.checkpoint_prefix}_ep{epoch}.pt"

    # ============================================================================
    # Phase Information Helper
    # ============================================================================

    @classmethod
    def get_phase_info(cls, epoch):
        """
        Return phase information for given epoch.

        Args:
            epoch: Current epoch number

        Returns:
            dict with keys:
                - phase_num: Phase number (1-2)
                - phase_name: Descriptive name
                - horizon: Horizon(s) for this phase
                - seq_len: Required sequence length range
        """
        if epoch < cls.phase1_end:
            return {
                "phase_num": 1,
                "phase_name": "Teacher Forcing",
                "horizon": 1,
                "seq_len": cls.phase1_seq_len,
            }
        else:
            return {
                "phase_num": 2,
                "phase_name": f"Multi-Horizon {cls.phase2_horizons}",
                "horizon": cls.phase2_horizons,
                "seq_len": cls.phase2_seq_len,
            }

    @classmethod
    def get_phase_end_epoch(cls, phase_num):
        """Get the last epoch of a phase."""
        if phase_num == 1:
            return cls.phase1_end
        elif phase_num == 2:
            return cls.phase2_end
        else:
            raise ValueError(f"Invalid phase number: {phase_num} (only 1-2 supported)")

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("BACKFILL CONTEXT=60 LATENT12 CONFIGURATION")
        print("=" * 80)
        print(f"Training period: {cls.train_period_years} years ({cls.train_end_idx - cls.train_start_idx} days)")
        print(f"Indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print()
        print(f"Context length: {cls.context_len} days")
        print(f"Latent dimension: {cls.latent_dim} (CHANGED from 5)")
        print(f"KL weight: {cls.kl_weight} (CHANGED from 1e-5)")
        print(f"LSTM hidden: {cls.mem_hidden}, layers: {cls.mem_layers}")
        print()
        print(f"Total epochs: {cls.total_epochs}")
        print(f"Batch size: {cls.batch_size}")
        print(f"Learning rate: {cls.learning_rate}")
        print()
        print("Phase Schedule:")
        print(f"  Phase 1 (0-{cls.phase1_end-1}): Teacher Forcing (H=1)")
        print(f"    Sequence length: {cls.phase1_seq_len}")
        print(f"  Phase 2 ({cls.phase1_end}-{cls.phase2_end-1}): Multi-Horizon {cls.phase2_horizons} (400 EPOCHS)")
        print(f"    Sequence length: {cls.phase2_seq_len}")
        print(f"    Weights: UNIFORM (all 1.0)")
        print()
        print("Quantile Loss Weights: [1.0, 1.0, 1.0] (CHANGED from [5.0, 1.0, 5.0])")
        print()
        print("Checkpoints will be saved after each phase:")
        print(f"  {cls.get_checkpoint_name(199)}")
        print(f"  {cls.get_checkpoint_name(599)}")
        print("=" * 80)
