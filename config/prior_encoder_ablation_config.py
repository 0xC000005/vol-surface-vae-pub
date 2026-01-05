"""
Configuration for Prior Encoder Ablation Experiment

This config supports training three variants:
1. Baseline: Current FullCovariancePrior (context_summary input)
2. Prior Encoder Diagonal: Raw context input, diagonal output
3. Prior Encoder Full Cov: Raw context input, AR(1) covariance

All three variants use the same:
- Training schedule (Phase 1: 0-100, Phase 2: 100-400)
- Architecture (except prior network)
- Hyperparameters (learning rate, batch size, etc.)

This ensures fair comparison isolating the effect of prior network design.
"""

from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov


class PriorEncoderAblationConfig(BackfillContext60ConfigV4FullCov):
    """
    Configuration for Prior Encoder Ablation Experiment.

    Use `variant` parameter to specify which model to train:
    - "baseline": CVAEFullCovPrior (current design)
    - "prior_encoder_diagonal": CVAEWithPriorEncoderDiagonal
    - "prior_encoder_full_cov": CVAEWithPriorEncoderFullCov
    """

    # ============================================================================
    # VARIANT SELECTOR
    # ============================================================================

    variant = "baseline"  # Options: "baseline", "prior_encoder_diagonal", "prior_encoder_full_cov"

    # ============================================================================
    # PRIOR ENCODER SPECIFIC SETTINGS (same Conv2D as context encoder)
    # ============================================================================

    # Surface embedding (Conv2D channels)
    # Context encoder uses: [5, 5, 5] (from BackfillContext60ConfigV4FullCov)
    # Prior encoder uses: SAME size for fair comparison
    prior_surface_hidden = [5, 5, 5]

    # Temporal encoder (LSTM hidden size)
    # Context encoder uses: 100 hidden, 2 layers (from mem_hidden, mem_layers)
    # Prior encoder uses: 64 hidden, 1 layer (~50% capacity)
    prior_mem_hidden = 64
    prior_mem_layers = 1
    prior_dropout = 0.1

    # Position encoding dimension
    prior_pos_dim = 32

    # ============================================================================
    # PATHS (variant-specific)
    # ============================================================================

    checkpoint_dir_template = "models/backfill/prior_encoder_ablation/{variant}/checkpoints"
    checkpoint_prefix_template = "prior_encoder_ablation_{variant}"
    results_dir_template = "results/prior_encoder_ablation/{variant}"

    @classmethod
    def get_checkpoint_dir(cls):
        """Get checkpoint directory for current variant."""
        return cls.checkpoint_dir_template.format(variant=cls.variant)

    @classmethod
    def get_checkpoint_prefix(cls):
        """Get checkpoint prefix for current variant."""
        return cls.checkpoint_prefix_template.format(variant=cls.variant)

    @classmethod
    def get_results_dir(cls):
        """Get results directory for current variant."""
        return cls.results_dir_template.format(variant=cls.variant)

    @classmethod
    def get_checkpoint_name(cls, epoch):
        """Generate checkpoint filename for given epoch."""
        prefix = cls.get_checkpoint_prefix()
        if epoch == 99 or epoch == cls.phase1_end - 1:
            return f"{prefix}_phase1_ep99.pt"
        elif epoch == 399 or epoch == cls.phase2_end - 1:
            return f"{prefix}_phase2_ep399.pt"
        else:
            return f"{prefix}_ep{epoch}.pt"

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print(f"PRIOR ENCODER ABLATION EXPERIMENT - VARIANT: {cls.variant.upper()}")
        print("=" * 80)
        print(f"Training period: {cls.train_period_years} years ({cls.train_end_idx - cls.train_start_idx} days)")
        print(f"Indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print()
        print(f"Context length: {cls.context_len} days (FIXED)")
        print(f"Latent dimension: {cls.latent_dim}")
        print(f"KL weight: {cls.kl_weight}")
        print(f"LSTM hidden: {cls.mem_hidden}, layers: {cls.mem_layers}")
        print()
        print("=" * 80)

        if cls.variant == "baseline":
            print("BASELINE: Full Covariance Prior (Current Design)")
            print("=" * 80)
            print("Prior Network:")
            print(f"  Input: context_summary (B, 12) - last LSTM hidden state")
            print(f"  Architecture: MLP or RNN over 12-dim vector")
            print(f"  Position encoding: {cls.full_cov_pos_dim}-dim")
            print(f"  Hidden layers: {cls.full_cov_hidden_dims}")
            print(f"  Dropout: {cls.full_cov_dropout}")
            print()
            print("Gradient Flow:")
            print("  Context encoder receives gradients from:")
            print("    ✗ Reconstruction loss (via decoder)")
            print("    ✗ KL loss (via prior network)")
            print("  => CONFOUNDED GRADIENTS")

        elif cls.variant == "prior_encoder_diagonal":
            print("VARIANT A: Prior Encoder Diagonal")
            print("=" * 80)
            print("Prior Encoder:")
            print(f"  Input: raw_context (B, {cls.context_len}, 5, 5) - full context surfaces")
            print(f"  Surface embedding: Conv2D {cls.prior_surface_hidden}")
            print(f"  Temporal encoder: LSTM hidden={cls.prior_mem_hidden}, layers={cls.prior_mem_layers}")
            print(f"  Position encoding: {cls.prior_pos_dim}-dim")
            print(f"  Output: mu_p (B, H, {cls.latent_dim}), log_var_p (B, H, {cls.latent_dim})")
            print(f"  Dropout: {cls.prior_dropout}")
            print()
            print("Gradient Flow:")
            print("  Context encoder receives gradients from:")
            print("    ✓ Reconstruction loss (via decoder)")
            print("  Prior encoder receives gradients from:")
            print("    ✓ KL loss ONLY")
            print("  => CLEAN GRADIENTS")

        elif cls.variant == "prior_encoder_full_cov":
            print("VARIANT B: Prior Encoder Full Covariance")
            print("=" * 80)
            print("Prior Encoder:")
            print(f"  Input: raw_context (B, {cls.context_len}, 5, 5) - full context surfaces")
            print(f"  Surface embedding: Conv2D {cls.prior_surface_hidden}")
            print(f"  Temporal encoder: LSTM hidden={cls.prior_mem_hidden}, layers={cls.prior_mem_layers}")
            print(f"  Position encoding: {cls.prior_pos_dim}-dim")
            print(f"  Output: mu_p (B, H, {cls.latent_dim}) + Sigma_p (H, H)")
            print(f"  Dropout: {cls.prior_dropout}")
            print()
            print("AR(1) Covariance:")
            print(f"  Initial φ: {cls.full_cov_init_phi}")
            print(f"  Initial σ²: {cls.full_cov_init_sigma_sq}")
            print("  Learnable params: 2 (φ, σ²)")
            print()
            print("Gradient Flow:")
            print("  Context encoder receives gradients from:")
            print("    ✓ Reconstruction loss (via decoder)")
            print("  Prior encoder receives gradients from:")
            print("    ✓ KL loss ONLY")
            print("  => CLEAN GRADIENTS")

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
        print("Checkpoints will be saved to:")
        print(f"  {cls.get_checkpoint_dir()}/")
        print(f"    - {cls.get_checkpoint_name(99)}")
        print(f"    - {cls.get_checkpoint_name(399)}")
        print()
        print("Results will be saved to:")
        print(f"  {cls.get_results_dir()}/")
        print("=" * 80)
