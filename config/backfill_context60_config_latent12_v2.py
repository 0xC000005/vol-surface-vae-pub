"""
Configuration for Context=60 2-Phase Training with Corrected KL Weight (V2)

This is V2 of the latent12 config, correcting the over-regularization issue
discovered in V1 (kl_weight=5e-5 caused posterior collapse with KL=0.854).

Key Changes from Latent12 V1:
- kl_weight: 5e-5 → 1e-5 (CORRECTED - V1 was too strong, caused posterior collapse)
- latent_dim: 12 (unchanged - keep increased capacity)

Training Schedule:
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1,7,14,30,60,90] - 400 EPOCHS

V1 Results (FAILURE):
- KL divergence: 0.854 < 1.0 (severe over-regularization)
- PC1 variance: 99.27% (still over-compressed)
- Correlation: 0.113 (67% WORSE than baseline)
- Day-1 spread: 0.0404 (53% improved but still 63% too wide)
- Effective dimensionality: 8.3% (worse than baseline 20%)

V2 Expected Impact:
- KL divergence: 0.854 → 2-5 (healthy range)
- PC1 variance: 99.27% → 90-95% (better distribution)
- Correlation: 0.113 → 0.35-0.45 (recovery to baseline or better)
- Day-1 spread: 0.0404 → 0.03-0.035 (maintain improvement)
- Effective dimensionality: 8.3% → 15-25% (better utilization)

CRITICAL: This model decouples latent_dim increase from kl_weight increase,
isolating the effect of increased latent capacity without over-regularization.
"""


class BackfillContext60ConfigLatent12V2:
    """Configuration for Context=60 model with 2-phase training and CORRECTED KL weight (V2)."""

    # ============================================================================
    # Data Configuration
    # ============================================================================

    train_period_years = 16  # Use full 16-year dataset
    train_start_idx = 1000
    train_end_idx = 5000

    # ============================================================================
    # Model Architecture - LATENT BOTTLENECK FIX (CORRECTED)
    # ============================================================================

    context_len = 60
    latent_dim = 12         # SAME as V1 (2.4× capacity vs baseline)
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
    phase2_end = 600    # Multi-horizon - 400 epochs

    # Sequence length requirements per phase
    # Format: (min_seq_len, max_seq_len)
    phase1_seq_len = (61, 80)       # context=60 + horizon=1 + buffer
    phase2_seq_len = (150, 180)     # context=60 + horizon=90 + buffer

    # ============================================================================
    # Phase 2: Multi-Horizon Training - UNIFORM WEIGHTS
    # ============================================================================

    phase2_horizons = [1, 7, 14, 30, 60, 90]
    phase2_weights = {
        1: 1.0,     # Equal exposure to all horizons
        7: 1.0,
        14: 1.0,
        30: 1.0,
        60: 1.0,
        90: 1.0
    }

    # ============================================================================
    # Optimization
    # ============================================================================

    learning_rate = 1e-5
    batch_size = 128  # 3070 Ti verified: uses only ~7% GPU memory (576 MB peak)
    valid_batch_size = 256  # Larger batch for validation (no gradients needed)

    # ============================================================================
    # Loss Configuration - CORRECTED KL WEIGHT
    # ============================================================================

    kl_weight = 1e-5  # CORRECTED: Was 5e-5 in V1 (too strong, caused posterior collapse)

    # Quantile regression
    use_quantile_regression = True
    num_quantiles = 3
    quantiles = [0.05, 0.5, 0.95]
    quantile_loss_weights = [1.0, 1.0, 1.0]  # Pinball loss has built-in asymmetry

    # Extra features
    re_feat_weight = 1.0
    ex_loss_on_ret_only = True
    ex_feats_loss_type = "l2"

    # ============================================================================
    # Checkpoint Configuration
    # ============================================================================

    checkpoint_dir = "models/backfill/context60_experiment/checkpoints"
    checkpoint_prefix = "backfill_context60_latent12_v2"

    # Checkpoint naming convention
    @classmethod
    def get_checkpoint_name(cls, epoch):
        """
        Generate checkpoint filename for given epoch.

        Args:
            epoch: Epoch number

        Returns:
            str: Checkpoint filename (e.g., "backfill_context60_latent12_v2_phase1_ep199.pt")
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
        print("BACKFILL CONTEXT=60 LATENT12 V2 CONFIGURATION (CORRECTED)")
        print("=" * 80)
        print(f"Training period: {cls.train_period_years} years ({cls.train_end_idx - cls.train_start_idx} days)")
        print(f"Indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print()
        print(f"Context length: {cls.context_len} days")
        print(f"Latent dimension: {cls.latent_dim} (SAME as V1)")
        print(f"KL weight: {cls.kl_weight} (CORRECTED from 5e-5 in V1)")
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
        print("Quantile Loss Weights: [1.0, 1.0, 1.0] (pinball loss has built-in asymmetry)")
        print()
        print("V2 CORRECTION:")
        print(f"  - kl_weight: 5e-5 (V1) → 1e-5 (V2) - fixes posterior collapse")
        print(f"  - V1 KL divergence: 0.854 (collapsed)")
        print(f"  - V2 Target KL: 2-5 (healthy range)")
        print()
        print("Checkpoints will be saved after each phase:")
        print(f"  {cls.get_checkpoint_name(199)}")
        print(f"  {cls.get_checkpoint_name(599)}")
        print("=" * 80)
