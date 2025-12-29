"""
Configuration for Context=60 with Full Covariance Prior (V4)

This is V4 of the context60 model, implementing a full covariance AR(1) prior
to address the parameter reuse problem in V3 and achieve smooth temporal predictions.

Key Changes from V3:
--------------------
- use_conditional_prior: False (disabled - replaced by full covariance prior)
- use_full_covariance_prior: True (NEW)
- Only 2 learnable covariance params (φ, σ²) vs 187K in V3
- Position-encoded means: different μ_t for each timestep
- AR(1) covariance: Σ[i,j] = σ² × φ^|i-j|
- Context length FIXED at 60 (no randomization)
- Dropout on prior network for generalization

Full Covariance Prior Architecture:
-----------------------------------
**Prior Mean Network:**
- Input: context summary (12-dim) + sinusoidal position encoding (64-dim)
- MLP: [64, 32] hidden dims with ReLU + Dropout(0.1)
- Output: μ_t for each timestep t=1,...,H

**Covariance Structure:**
- Global AR(1): Σ[i,j] = σ² × φ^|i-j|
- Only 2 learnable scalars: φ ∈ (0, 1), σ² > 0
- Cholesky sampling: z = μ + L @ ε for correlated samples

Expected Benefits:
-----------------
1. **Temporal smoothness**: φ induces correlation between consecutive timesteps
2. **Roughness ratio >40%**: Improves from 9.7% (V3) toward oracle's 75%
3. **Position awareness**: μ_t varies with timestep (near vs far future)
4. **Fewer parameters**: 2 vs 187K covariance params (prevents overfitting)
5. **Better generalization**: Dropout + reduced capacity

Training Schedule:
-----------------
- Phase 1 (epochs 0-200): Teacher forcing (H=1)
- Phase 2 (epochs 201-600): Multi-horizon [1,7,14,30,60,90] - 400 EPOCHS

Model Comparison:
----------------
|                    | V3 (Conditional Prior) | V4 (Full Cov Prior)    |
|--------------------|------------------------|------------------------|
| latent_dim         | 12                     | 12                     |
| kl_weight          | 1e-5                   | 1e-5                   |
| Prior type         | p(z|context) diagonal  | p(z|context) AR(1)     |
| Covariance params  | 187K                   | **2 (φ, σ²)**          |
| Temporal structure | None (IID reuse)       | **AR(1) correlated**   |
| Roughness ratio    | 9.7%                   | **Target: >40%**       |
| Position encoding  | No                     | **Yes (sinusoidal)**   |
| Dropout            | No                     | **Yes (0.1)**          |
"""

from config.backfill_context60_config_latent12_v3_conditional_prior import (
    BackfillContext60ConfigLatent12V3ConditionalPrior
)


class BackfillContext60ConfigV4FullCov(BackfillContext60ConfigLatent12V3ConditionalPrior):
    """Configuration for Context=60 model with Full Covariance Prior (V4)."""

    # ============================================================================
    # CONTEXT LENGTH - FIXED at 60 (CRITICAL!)
    # ============================================================================

    context_len = 60           # FIXED - not variable
    min_context_len = 60       # Same as context_len - no randomization
    max_context_len = 60       # Same as context_len - no randomization

    # ============================================================================
    # DISABLE OLD PRIORS
    # ============================================================================

    use_conditional_prior = False  # Disable V3 diagonal conditional prior
    use_fitted_prior = False       # Disable fitted prior

    # DISABLE QUANTILE REGRESSION (not used in V4)
    use_quantile_regression = False
    num_quantiles = 1
    quantiles = [0.5]
    quantile_loss_weights = [1.0]

    # ============================================================================
    # FULL COVARIANCE PRIOR HYPERPARAMETERS (NEW!)
    # ============================================================================

    # Position encoding
    full_cov_pos_dim = 64          # Dimension of sinusoidal position encoding

    # Prior network capacity control (configurable to prevent over-discrimination)
    full_cov_hidden_dims = [64, 32]  # MLP hidden layers - reduced to [64,32] per capacity sweep

    # Dropout for generalization (NEW!)
    full_cov_dropout = 0.1         # Dropout rate for prior MLP

    # AR(1) covariance initialization
    full_cov_init_phi = 0.5        # Initial φ - Goldilocks zone (NOT 0.7!)
    full_cov_init_sigma_sq = 1.0   # Initial σ²

    # Horizon settings
    max_horizon = 90               # Maximum forecast horizon
    horizon = 30                   # Default training horizon

    # ============================================================================
    # TRAINING SCHEDULE (OPTIMIZED FOR V4)
    # ============================================================================

    # CHANGED: Reduced epochs (V4 has simpler prior - 2 params vs 187K)
    total_epochs = 400             # Was 600
    phase1_end = 100               # Was 200 - Teacher forcing
    phase2_end = 400               # Was 600 - Multi-horizon

    # CHANGED: Removed H=7 and H=14 (33% fewer iterations, focus on key horizons)
    phase2_horizons = [1, 30, 60, 90]  # Was [1, 7, 14, 30, 60, 90]
    phase2_weights = {
        1: 1.0,
        30: 1.0,
        60: 1.0,
        90: 1.0
    }

    # ============================================================================
    # PATHS
    # ============================================================================

    checkpoint_dir = "models/backfill/context60_v4_full_cov/checkpoints"
    checkpoint_prefix = "backfill_context60_latent12_v4_full_cov"
    results_dir = "results/context60_v4_full_cov"

    @classmethod
    def get_checkpoint_name(cls, epoch):
        """Generate checkpoint filename for given epoch (V4 uses epochs 99, 399)."""
        if epoch == 99 or epoch == cls.phase1_end - 1:
            return f"{cls.checkpoint_prefix}_phase1_ep99.pt"
        elif epoch == 399 or epoch == cls.phase2_end - 1:
            return f"{cls.checkpoint_prefix}_phase2_ep399.pt"
        else:
            return f"{cls.checkpoint_prefix}_ep{epoch}.pt"

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("BACKFILL CONTEXT=60 LATENT12 V4 - FULL COVARIANCE PRIOR")
        print("=" * 80)
        print(f"Training period: {cls.train_period_years} years ({cls.train_end_idx - cls.train_start_idx} days)")
        print(f"Indices: [{cls.train_start_idx}, {cls.train_end_idx}]")
        print()
        print(f"Context length: {cls.context_len} days (FIXED - no randomization)")
        print(f"Latent dimension: {cls.latent_dim}")
        print(f"KL weight: {cls.kl_weight}")
        print(f"LSTM hidden: {cls.mem_hidden}, layers: {cls.mem_layers}")
        print()
        print("=" * 80)
        print("FULL COVARIANCE PRIOR (V4 NEW!)")
        print("=" * 80)
        print("Prior Mean Network:")
        print(f"  Position encoding: {cls.full_cov_pos_dim}-dim sinusoidal")
        print(f"  Hidden layers: {cls.full_cov_hidden_dims}")
        print(f"  Dropout: {cls.full_cov_dropout}")
        print(f"  Output: μ_t for each timestep t (position-dependent)")
        print()
        print("Covariance Structure:")
        print(f"  AR(1) covariance: Σ[i,j] = σ² × φ^|i-j|")
        print(f"  Initial φ: {cls.full_cov_init_phi} (target: Goldilocks zone ~0.5)")
        print(f"  Initial σ²: {cls.full_cov_init_sigma_sq}")
        print(f"  Learnable params: 2 (φ, σ²) vs 187K in V3")
        print()
        print("Sampling:")
        print("  Cholesky decomposition: z = μ + L @ ε")
        print("  Result: Temporally correlated samples (not IID)")
        print()
        print("Expected improvements:")
        print("  ✓ Roughness ratio: 9.7% → >40% (target)")
        print("  ✓ Temporal smoothness: φ induces correlation")
        print("  ✓ Position awareness: μ_t varies with timestep")
        print("  ✓ Better generalization: Dropout + reduced capacity")
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
        print(f"  {cls.get_checkpoint_name(99)}")
        print(f"  {cls.get_checkpoint_name(399)}")
        print("=" * 80)
