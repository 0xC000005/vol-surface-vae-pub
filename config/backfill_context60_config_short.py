"""
Configuration for Context=60 4-Phase Curriculum Training (SHORT VERSION - 50 epochs per phase)

Training Schedule:
- Phase 1 (epochs 0-199): Teacher forcing (H=1) - ALREADY COMPLETED
- Phase 2 (epochs 200-249): Multi-horizon [1,7,14,30,60,90] - 50 epochs
- Phase 3 (epochs 250-299): AR H=60, offsets=[30,60] - 50 epochs
- Phase 4 (epochs 300-349): AR H=90, offsets=[45,90] - 50 epochs

TOTAL: 150 epochs (Phase 2-4), estimated ~14.5 hours on 3070 Ti (float32)

CRITICAL: Saves checkpoint after EACH phase for comparison.
RESUMES from: backfill_context60_phase1_ep199.pt
"""


class BackfillContext60ConfigShort:
    """Configuration for Context=60 model with SHORT 4-phase curriculum (50 epochs/phase)."""

    # ============================================================================
    # Data Configuration
    # ============================================================================

    train_period_years = 16  # Use full 16-year dataset
    train_start_idx = 1000
    train_end_idx = 5000

    # ============================================================================
    # Model Architecture (IDENTICAL to original config)
    # ============================================================================

    context_len = 60
    latent_dim = 5
    mem_hidden = 100
    mem_layers = 2
    mem_dropout = 0.3
    surface_hidden = [5, 5, 5]

    # ============================================================================
    # Training Schedule (4 Phases - SHORTENED)
    # ============================================================================

    total_epochs = 350  # 200 (Phase 1 done) + 50 + 50 + 50

    # Phase boundaries
    phase1_end = 200    # Teacher forcing (ALREADY COMPLETED)
    phase2_end = 250    # Multi-horizon (50 epochs)
    phase3_end = 300    # AR H=60 (50 epochs)
    phase4_end = 350    # AR H=90 (50 epochs, final)

    # Sequence length requirements per phase
    # Format: (min_seq_len, max_seq_len)
    phase1_seq_len = (61, 80)       # context=60 + horizon=1 + buffer
    phase2_seq_len = (150, 180)     # context=60 + horizon=90 + buffer
    phase3_seq_len = (240, 260)     # context=60 + (3 steps × 60) + buffer
    phase4_seq_len = (330, 350)     # context=60 + (3 steps × 90) + buffer

    # ============================================================================
    # Phase 2: Multi-Horizon Training
    # ============================================================================

    phase2_horizons = [1, 7, 14, 30, 60, 90]
    phase2_weights = {
        1: 1.0,    # Highest priority (short-term)
        7: 0.9,
        14: 0.8,
        30: 0.6,
        60: 0.4,
        90: 0.3    # Lowest priority (long-term)
    }

    # ============================================================================
    # Phase 3: Autoregressive H=60
    # ============================================================================

    phase3_horizon = 60
    phase3_offsets = [30, 60]  # 50% overlap and non-overlapping
    phase3_ar_steps = 3

    # ============================================================================
    # Phase 4: Autoregressive H=90 (Deployment)
    # ============================================================================

    phase4_horizon = 90
    phase4_offsets = [45, 90]  # 50% overlap and deployment target
    phase4_ar_steps = 3

    # ============================================================================
    # Optimization (IDENTICAL to original config)
    # ============================================================================

    learning_rate = 1e-5
    batch_size = 128  # 3070 Ti verified: uses only ~7% GPU memory (576 MB peak)
    valid_batch_size = 256  # Larger batch for validation (no gradients needed)

    # ============================================================================
    # Loss Configuration (IDENTICAL to original config)
    # ============================================================================

    kl_weight = 1e-5

    # Quantile regression
    use_quantile_regression = True
    num_quantiles = 3
    quantiles = [0.05, 0.5, 0.95]
    quantile_loss_weights = [5.0, 1.0, 5.0]  # Emphasize tail quantiles

    # Extra features
    re_feat_weight = 1.0
    ex_loss_on_ret_only = True
    ex_feats_loss_type = "l2"

    # ============================================================================
    # Checkpoint Configuration
    # ============================================================================

    checkpoint_dir = "models_backfill"
    checkpoint_prefix = "backfill_context60_short"  # Different prefix for short version

    # Phase 1 checkpoint to resume from
    phase1_checkpoint_path = "models_backfill/backfill_context60_phase1_ep199.pt"

    # Training log
    log_filename = "context60_training_log_resume.txt"

    # Checkpoint naming convention
    @classmethod
    def get_checkpoint_name(cls, phase_num, epoch):
        """
        Generate checkpoint filename for a phase.

        Args:
            phase_num: Phase number (1, 2, 3, or 4)
            epoch: Epoch number

        Returns:
            str: Checkpoint filename (e.g., "backfill_context60_short_phase2_ep209.pt")
        """
        return f"{cls.checkpoint_prefix}_phase{phase_num}_ep{epoch}.pt"

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
                - phase_num: Phase number (1-4)
                - phase_name: Descriptive name
                - horizon: Horizon(s) for this phase
                - offsets: Offset values (None for phases 1-2)
                - ar_steps: AR steps (None for phases 1-2)
                - seq_len: Required sequence length range
        """
        if epoch < cls.phase1_end:
            return {
                "phase_num": 1,
                "phase_name": "Teacher Forcing",
                "horizon": 1,
                "offsets": None,
                "ar_steps": None,
                "seq_len": cls.phase1_seq_len,
            }
        elif epoch < cls.phase2_end:
            return {
                "phase_num": 2,
                "phase_name": f"Multi-Horizon {cls.phase2_horizons}",
                "horizon": cls.phase2_horizons,
                "offsets": None,
                "ar_steps": None,
                "seq_len": cls.phase2_seq_len,
            }
        elif epoch < cls.phase3_end:
            return {
                "phase_num": 3,
                "phase_name": f"AR H={cls.phase3_horizon}, offsets={cls.phase3_offsets}",
                "horizon": cls.phase3_horizon,
                "offsets": cls.phase3_offsets,
                "ar_steps": cls.phase3_ar_steps,
                "seq_len": cls.phase3_seq_len,
            }
        else:
            return {
                "phase_num": 4,
                "phase_name": f"AR H={cls.phase4_horizon}, offsets={cls.phase4_offsets}",
                "horizon": cls.phase4_horizon,
                "offsets": cls.phase4_offsets,
                "ar_steps": cls.phase4_ar_steps,
                "seq_len": cls.phase4_seq_len,
            }

    @classmethod
    def get_phase_end_epoch(cls, phase_num):
        """Get the last epoch of a phase."""
        if phase_num == 1:
            return cls.phase1_end
        elif phase_num == 2:
            return cls.phase2_end
        elif phase_num == 3:
            return cls.phase3_end
        elif phase_num == 4:
            return cls.phase4_end
        else:
            raise ValueError(f"Invalid phase number: {phase_num}")

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("BACKFILL CONTEXT=60 CONFIGURATION (SHORT VERSION - 50 epochs/phase)")
        print("=" * 80)
        print(f"Training period: {cls.train_period_years} years ({cls.train_end_idx - cls.train_start_idx} days)")
        print(f"Indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print()
        print(f"Context length: {cls.context_len} days")
        print(f"Latent dimension: {cls.latent_dim}")
        print(f"LSTM hidden: {cls.mem_hidden}, layers: {cls.mem_layers}")
        print()
        print(f"Total epochs: {cls.total_epochs} (Phase 1 already done, {cls.total_epochs - cls.phase1_end} remaining)")
        print(f"Batch size: {cls.batch_size}")
        print(f"Learning rate: {cls.learning_rate}")
        print()
        print(f"Resuming from: {cls.phase1_checkpoint_path}")
        print()
        print("Phase Schedule:")
        print(f"  Phase 1 (0-{cls.phase1_end-1}): Teacher Forcing (H=1) - ✅ COMPLETED")
        print(f"    Checkpoint: backfill_context60_phase1_ep199.pt")
        print(f"  Phase 2 ({cls.phase1_end}-{cls.phase2_end-1}): Multi-Horizon {cls.phase2_horizons} - 50 epochs")
        print(f"    Sequence length: {cls.phase2_seq_len}")
        print(f"    Estimated time: ~3.2 hours")
        print(f"  Phase 3 ({cls.phase2_end}-{cls.phase3_end-1}): AR H={cls.phase3_horizon}, offsets={cls.phase3_offsets} - 50 epochs")
        print(f"    Sequence length: {cls.phase3_seq_len}")
        print(f"    Estimated time: ~4.8 hours")
        print(f"  Phase 4 ({cls.phase3_end}-{cls.phase4_end-1}): AR H={cls.phase4_horizon}, offsets={cls.phase4_offsets} - 50 epochs")
        print(f"    Sequence length: {cls.phase4_seq_len}")
        print(f"    Estimated time: ~6.5 hours")
        print()
        print("TOTAL ESTIMATED TIME: ~14.5 hours")
        print()
        print("Checkpoints will be saved after each phase:")
        print(f"  {cls.get_checkpoint_name(2, cls.phase2_end-1)}")
        print(f"  {cls.get_checkpoint_name(3, cls.phase3_end-1)}")
        print(f"  {cls.get_checkpoint_name(4, cls.phase4_end-1)}")
        print("=" * 80)
