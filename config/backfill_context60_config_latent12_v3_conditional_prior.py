"""
Configuration for Context=60 with Conditional Prior Network (V3)

This is V3 of the latent12 config, implementing a learnable conditional prior
network p(z|context) to replace the fixed N(0,1) prior and eliminate VAE prior mismatch.

Key Changes from Latent12 V2:
-----------------------------
- use_conditional_prior: True (NEW - enables conditional prior network)
- latent_dim: 12 (unchanged from V2)
- kl_weight: 1e-5 (unchanged from V2)

How Conditional Prior Works:
----------------------------
**Training:**
- KL loss changes from:  KL(q(z|context, target) || N(0,1))
                  to:    KL(q(z|context, target) || p(z|context))
- Prior network learns to predict appropriate latent distribution for each context
- Matches decoder's training distribution (no more mismatch!)

**Inference:**
- Instead of sampling z ~ N(0,1), we sample z ~ p(z|context)
- Prior adapts to each specific market regime (high vol → different prior than low vol)
- Eliminates systematic negative bias without post-hoc corrections

Expected Benefits:
-----------------
1. **Zero systematic bias**: Predictions centered at ground truth
2. **Context-adaptive uncertainty**: CI width adjusts to regime
3. **Better calibration**: Model learns appropriate uncertainty per context
4. **Eliminates need for**: Fitted GMM post-hoc correction
5. **Theoretically principled**: Proper conditional VAE formulation

Training Schedule:
-----------------
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1,7,14,30,60,90] - 400 EPOCHS

Model Comparison:
----------------
|                    | V1 (Latent12)   | V2 (Fixed KL) | V3 (Conditional Prior) |
|--------------------|-----------------|---------------|------------------------|
| latent_dim         | 12              | 12            | 12                     |
| kl_weight          | 5e-5 (too high) | 1e-5          | 1e-5                   |
| Prior type         | N(0,1)          | N(0,1)        | p(z|context) LEARNED   |
| KL divergence      | 0.854           | 2-5 (target)  | 2-5 (target)           |
| Systematic bias    | Present         | Present       | **Eliminated**         |
| CI calibration     | Poor            | Improved      | **Best (adaptive)**    |
| Regime adaptation  | No              | No            | **Yes**                |
"""


class BackfillContext60ConfigLatent12V3ConditionalPrior:
    """Configuration for Context=60 model with Conditional Prior Network (V3)."""

    # ============================================================================
    # Data Configuration
    # ============================================================================

    train_period_years = 16  # Use full 16-year dataset
    train_start_idx = 1000
    train_end_idx = 5000

    # ============================================================================
    # Model Architecture - WITH CONDITIONAL PRIOR NETWORK
    # ============================================================================

    context_len = 60
    latent_dim = 12         # Same as V1/V2
    mem_hidden = 100
    mem_layers = 2
    mem_dropout = 0.3
    surface_hidden = [5, 5, 5]

    # ============================================================================
    # CONDITIONAL PRIOR CONFIGURATION (NEW!)
    # ============================================================================

    use_conditional_prior = True  # Enable conditional prior p(z|context)
    cache_encoder_multihorizon = True  # Optimize Phase 2: encode once, resample z per horizon (~2.7x faster)

    # Note: Prior network shares architecture with ctx_encoder:
    # - Same surface_hidden layers: [5, 5, 5]
    # - Same mem_hidden: 100
    # - Same mem_layers: 2
    # - Outputs: prior_mean, prior_logvar (instead of fixed N(0,1))

    # ============================================================================
    # Training Schedule (2 Phases)
    # ============================================================================

    total_epochs = 600

    # Phase boundaries
    phase1_end = 200    # Teacher forcing
    phase2_end = 600    # Multi-horizon - 400 epochs

    # Sequence length requirements per phase
    phase1_seq_len = (61, 80)       # context=60 + horizon=1 + buffer
    phase2_seq_len = (150, 180)     # context=60 + horizon=90 + buffer

    # ============================================================================
    # Phase 2: Multi-Horizon Training
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
    batch_size = 256  # RTX 3090 Ti (24GB VRAM) - was 128 for 3070 Ti (8GB)
    valid_batch_size = 512  # Was 256

    # ============================================================================
    # Loss Configuration
    # ============================================================================

    kl_weight = 1e-5  # Same as V2

    # Quantile regression
    use_quantile_regression = True
    num_quantiles = 3
    quantiles = [0.05, 0.5, 0.95]
    quantile_loss_weights = [1.0, 1.0, 1.0]

    # Extra features
    re_feat_weight = 1.0
    ex_loss_on_ret_only = True
    ex_feats_loss_type = "l2"

    # ============================================================================
    # Checkpoint Configuration
    # ============================================================================

    checkpoint_dir = "models/backfill/context60_experiment/checkpoints"
    checkpoint_prefix = "backfill_context60_latent12_v3_conditional_prior"

    @classmethod
    def get_checkpoint_name(cls, epoch):
        """Generate checkpoint filename for given epoch."""
        if epoch == 199 or epoch == cls.phase1_end - 1:
            return f"{cls.checkpoint_prefix}_phase1_ep199.pt"
        elif epoch == 599 or epoch == cls.phase2_end - 1:
            return f"{cls.checkpoint_prefix}_phase2_ep599.pt"
        else:
            return f"{cls.checkpoint_prefix}_ep{epoch}.pt"

    @classmethod
    def get_phase_info(cls, epoch):
        """Return phase information for given epoch."""
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
            raise ValueError(f"Invalid phase number: {phase_num}")

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("BACKFILL CONTEXT=60 LATENT12 V3 - CONDITIONAL PRIOR NETWORK")
        print("=" * 80)
        print(f"Training period: {cls.train_period_years} years ({cls.train_end_idx - cls.train_start_idx} days)")
        print(f"Indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print()
        print(f"Context length: {cls.context_len} days")
        print(f"Latent dimension: {cls.latent_dim}")
        print(f"KL weight: {cls.kl_weight}")
        print(f"LSTM hidden: {cls.mem_hidden}, layers: {cls.mem_layers}")
        print()
        print("=" * 80)
        print("CONDITIONAL PRIOR NETWORK (NEW!)")
        print("=" * 80)
        print(f"Enabled: {cls.use_conditional_prior}")
        print("Architecture: Shares same structure as context encoder")
        print("  - Surface encoding: Conv2D layers [5, 5, 5]")
        print("  - Memory: LSTM (100 hidden, 2 layers)")
        print("  - Output: prior_mean, prior_logvar per context")
        print()
        print("Training modification:")
        print("  Before: KL(q(z|context, target) || N(0,1))")
        print("  Now:    KL(q(z|context, target) || p(z|context))")
        print()
        print("Inference modification:")
        print("  Before: z ~ N(0,1)  [fixed, causes bias]")
        print("  Now:    z ~ p(z|context)  [adaptive to regime]")
        print()
        print("Expected benefits:")
        print("  ✓ Eliminates systematic negative bias")
        print("  ✓ Context-adaptive CI widths (crisis → wider)")
        print("  ✓ Better calibration (learns appropriate uncertainty)")
        print("  ✓ No post-hoc GMM fitting needed")
        print("=" * 80)
        print()
        print(f"Total epochs: {cls.total_epochs}")
        print(f"Batch size: {cls.batch_size}")
        print(f"Learning rate: {cls.learning_rate}")
        print()
        print("Phase Schedule:")
        print(f"  Phase 1 (0-{cls.phase1_end-1}): Teacher Forcing (H=1)")
        print(f"    Sequence length: {cls.phase1_seq_len}")
        print(f"  Phase 2 ({cls.phase1_end}-{cls.phase2_end-1}): Multi-Horizon {cls.phase2_horizons}")
        print(f"    Sequence length: {cls.phase2_seq_len}")
        print(f"    Weights: UNIFORM (all 1.0)")
        print()
        print("Checkpoints will be saved:")
        print(f"  {cls.get_checkpoint_name(199)}")
        print(f"  {cls.get_checkpoint_name(599)}")
        print("=" * 80)
