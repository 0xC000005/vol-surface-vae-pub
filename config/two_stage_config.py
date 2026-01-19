"""
Configuration for Two-Stage CVAE with Tiny Context Bottleneck

Goal: Fix variance collapse by:
1. Tiny context bottleneck (2-3 dims) - forces coarse encoding like k-means
2. True autoencoder training - decoder sees actual ctx_emb for ALL positions
3. Two-stage training - Stage 1 trains autoencoder, Stage 2 trains predictors

Key Architectural Changes:
- ctx_embedding_dim: 3 (separate from latent_dim)
- ctx_encoder: Smaller hidden layers to force coarse representation
- No zero-padding: ctx_emb computed for ALL positions (true autoencoder)
- Full sequence loss: MSE on all positions, not just horizon
"""


class TwoStageConfig:
    """Configuration for Two-Stage CVAE with tiny context bottleneck."""

    # ============================================================================
    # Core Architecture
    # ============================================================================

    feat_dim = (5, 5)           # Volatility surface grid size
    latent_dim = 16             # Expressive z (increased to carry more variance)
    context_len = 30            # Context window
    horizon = 30                # Default prediction horizon
    max_horizon = 30            # Maximum horizon supported

    # ============================================================================
    # Context Encoder - TINY BOTTLENECK (Key Change)
    # ============================================================================
    # Goal: Force coarse representation like k-means clusters
    # Smaller encoder → less precise ctx_emb → z must carry variance
    #
    # NOTE: StudentTMLPDecoder ignores ctx_emb entirely, so ctx_encoder is dead code
    # Set use_ctx_encoder=False to disable it and save computation

    use_ctx_encoder = False             # Disable unused ctx_encoder and prior_net
    ctx_embedding_dim = 3               # Tiny output dim (like k-means clusters)
    ctx_surface_hidden = [2, 4, 2]      # Conv layers (flatten = 2*5*5 = 50)
    ctx_mem_type = "lstm"               # Memory type for ctx encoder
    ctx_mem_hidden = 8                  # Tiny LSTM hidden (limit context capacity)
    ctx_mem_layers = 1                  # Fewer layers (was 2)
    ctx_mem_dropout = 0.2               # Dropout for ctx encoder
    ctx_compress = True                 # Use compression to ctx_embedding_dim

    # ============================================================================
    # Main Encoder (for z) - Expressive Latent
    # ============================================================================
    # z should capture fine-grained variance that ctx_emb can't

    surface_hidden = [2, 4, 2]      # Conv layers (flatten = 2*5*5 = 50)
    mem_type = "lstm"                   # Memory type
    mem_hidden = 8                      # Tiny LSTM hidden (limit context capacity)
    mem_layers = 1                      # Number of LSTM layers
    mem_dropout = 0.2                   # LSTM dropout

    # z_logvar floor to prevent variance collapse
    # Without floor: MSE gradient pushes z_logvar → -∞, making z deterministic
    # With floor: z maintains stochasticity, enabling conditional variance
    z_logvar_floor = -4.0               # exp(-4) ≈ 0.018, min_std ≈ 0.13
                                        # Set to None to disable floor

    # z dropout to prevent z from becoming a lookup key
    # Dropout forces decoder to be robust to missing z information
    # Prevents deterministic (context,target) → z mapping from being memorized
    z_dropout = 0.0                     # Default: no dropout
                                        # Recommended: 0.3-0.5 for regularization

    # ============================================================================
    # Decoder
    # ============================================================================
    # Input size = ctx_embedding_dim + latent_dim = 3 + 16 = 19

    use_dense_surface = False           # Use CNN to respect spatial structure
    decoder_mem_hidden = 8              # LSTM hidden size
    decoder_compress = False            # Whether to compress LSTM hidden before output
    decoder_compress_dim = 4            # Compression dim (only used if decoder_compress=True)
    decoder_mem_layers = 1              # Number of LSTM layers
    decoder_mem_dropout = 0.2           # LSTM dropout
    padding = 1                         # Conv padding
    deconv_output_padding = 0           # Deconv output padding

    # ============================================================================
    # Full Covariance Decoder (for CVAETwoStageFullCovariance)
    # ============================================================================
    # Outputs Cholesky factor L (25x25) instead of diagonal variance
    # Enables correlated sampling: x = μ + L @ ε

    full_covariance = False             # Set True to use full covariance decoder
    cholesky_diag_floor = 1e-3          # Min diagonal value for numerical stability
    cholesky_diag_init = -2.0           # Initial diagonal (softplus(-2) ≈ 0.13)

    # ============================================================================
    # Student-t Decoder (for CVAETwoStageStudentT)
    # ============================================================================
    # Extends Full Covariance with learnable degrees of freedom (nu)
    # Enables fat-tailed sampling via Gamma scale mixture:
    #   x = μ + L @ ε / √u where u_i ~ Gamma(ν_i/2, ν_i/2)
    # For nu=5: kurtosis ≈ 9 (vs Gaussian kurtosis = 3)
    #
    # Per-grid-point nu: 25 learnable parameters (one per grid point)
    # This allows different tail heaviness across the volatility surface
    # (GT kurtosis varies from 3 to 180 across the 5x5 grid)

    student_t = False                   # Set True to use Student-t decoder
    nu_floor = 2.1                      # Min nu (need nu > 2 for finite variance)
    nu_max = 100.0                      # Max nu (prevents collapse to Gaussian)
    nu_init = 5.0                       # Initial nu (kurtosis ≈ 9 for nu=5)

    # Loss weights for Student-t
    mse_weight = 1.0                    # Weight for mean MSE loss
    nll_weight = 1.0                    # Weight for NLL (higher than 0.1 for Full Cov)
    kurtosis_loss_weight = 0.1          # Weight for theoretical kurtosis supervision
                                        # Supervises nu via: excess_kurt = 6/(nu-4)

    # β-NLL for unbiased mean estimation (Seitzer 2022, ICLR)
    # Standard NLL allows model to trade mean accuracy for variance
    # β-NLL weights loss by variance^β to prevent this exploitation
    beta_nll = 0.5                      # 0.0=standard NLL, 0.5=recommended, 1.0=MSE-like

    # ============================================================================
    # Extra Features (Optional)
    # ============================================================================

    ex_feats_dim = 0                    # Set to 3 for returns/skew/slope
    ex_feats_hidden = None              # Hidden layers for ex_feats encoder
    ctx_ex_feats_hidden = None          # Hidden layers for ctx ex_feats encoder

    # ============================================================================
    # Training - Stage 1 (Autoencoder)
    # ============================================================================

    # Loss weights and mode
    kl_weight = 0.001                   # Weak KL for variance preservation
    loss_mode = "horizon"               # "full", "horizon", or "weighted"
                                        # - "full": MSE on all positions
                                        # - "horizon": MSE only on positions >= C (recommended)
                                        # - "weighted": 0.3 * context + 1.0 * horizon
    re_feat_weight = 0.0                # Weight for extra feature reconstruction
    ex_loss_on_ret_only = False         # If True, only compute loss on returns

    # Optimization
    learning_rate = 1e-4
    batch_size = 64
    stage1_epochs = 100                 # Epochs for Stage 1 (autoencoder)

    # ============================================================================
    # Training - Stage 2 (Predictors)
    # ============================================================================

    stage2_epochs = 50                  # Epochs for Stage 2 (predictors)
    stage2_learning_rate = 1e-4         # Learning rate for predictors
    stage2_batch_size = 64              # Batch size for Stage 2

    # ============================================================================
    # Predictor Architecture
    # ============================================================================
    # Single autoregressive LSTM: encodes context, then continues for H steps
    # LatentPredictor: context → (z_mean, z_logvar) for horizon positions
    # ContextPredictor: context → ctx_emb for horizon positions

    predictor_surface_hidden = [2, 4, 2]  # Conv layers (flatten = 2*5*5 = 50)
    predictor_hidden = 50                 # LSTM hidden size (matches embed_dim, no projection needed)
    predictor_layers = 1                  # Number of LSTM layers
    predictor_dropout = 0.2               # LSTM dropout

    # ============================================================================
    # Paths
    # ============================================================================

    checkpoint_dir = "models/backfill/two_stage"
    checkpoint_prefix = "two_stage_ctx3"

    # ============================================================================
    # Device
    # ============================================================================

    device = "cuda"

    # ============================================================================
    # Helper Methods
    # ============================================================================

    @classmethod
    def get_model_config(cls):
        """Return model configuration dictionary for CVAETwoStage."""
        return {
            # Core
            "feat_dim": cls.feat_dim,
            "latent_dim": cls.latent_dim,
            "context_len": cls.context_len,
            "horizon": cls.horizon,
            "max_horizon": cls.max_horizon,

            # Context encoder (tiny bottleneck)
            "use_ctx_encoder": cls.use_ctx_encoder,
            "ctx_embedding_dim": cls.ctx_embedding_dim,
            "ctx_surface_hidden": cls.ctx_surface_hidden,
            "ctx_mem_type": cls.ctx_mem_type,
            "ctx_mem_hidden": cls.ctx_mem_hidden,
            "ctx_mem_layers": cls.ctx_mem_layers,
            "ctx_mem_dropout": cls.ctx_mem_dropout,
            "ctx_compress": cls.ctx_compress,

            # Main encoder
            "surface_hidden": cls.surface_hidden,
            "mem_type": cls.mem_type,
            "mem_hidden": cls.mem_hidden,
            "mem_layers": cls.mem_layers,
            "mem_dropout": cls.mem_dropout,
            "z_logvar_floor": cls.z_logvar_floor,
            "z_dropout": cls.z_dropout,

            # Decoder
            "use_dense_surface": cls.use_dense_surface,
            "decoder_mem_hidden": cls.decoder_mem_hidden,
            "decoder_compress": cls.decoder_compress,
            "decoder_compress_dim": cls.decoder_compress_dim,
            "decoder_mem_layers": cls.decoder_mem_layers,
            "decoder_mem_dropout": cls.decoder_mem_dropout,
            "padding": cls.padding,
            "deconv_output_padding": cls.deconv_output_padding,

            # Full covariance decoder
            "full_covariance": cls.full_covariance,
            "cholesky_diag_floor": cls.cholesky_diag_floor,
            "cholesky_diag_init": cls.cholesky_diag_init,

            # Student-t decoder
            "student_t": cls.student_t,
            "nu_floor": cls.nu_floor,
            "nu_max": cls.nu_max,
            "nu_init": cls.nu_init,
            "mse_weight": cls.mse_weight,
            "nll_weight": cls.nll_weight,
            "kurtosis_loss_weight": cls.kurtosis_loss_weight,
            "beta_nll": cls.beta_nll,

            # Extra features
            "ex_feats_dim": cls.ex_feats_dim,
            "ex_feats_hidden": cls.ex_feats_hidden,
            "ctx_ex_feats_hidden": cls.ctx_ex_feats_hidden,

            # Training
            "kl_weight": cls.kl_weight,
            "loss_mode": cls.loss_mode,
            "re_feat_weight": cls.re_feat_weight,
            "ex_loss_on_ret_only": cls.ex_loss_on_ret_only,

            # Device
            "device": cls.device,
        }

    @classmethod
    def get_predictor_config(cls):
        """Return configuration for predictor networks."""
        return {
            # Architecture targets (must match autoencoder)
            "feat_dim": cls.feat_dim,
            "latent_dim": cls.latent_dim,
            "ctx_embedding_dim": cls.ctx_embedding_dim,
            "context_len": cls.context_len,
            "max_horizon": cls.max_horizon,

            # Predictor architecture (single autoregressive LSTM)
            "surface_hidden": cls.predictor_surface_hidden,
            "hidden_size": cls.predictor_hidden,
            "num_layers": cls.predictor_layers,
            "dropout": cls.predictor_dropout,

            # Training
            "learning_rate": cls.stage2_learning_rate,
            "batch_size": cls.stage2_batch_size,
            "epochs": cls.stage2_epochs,

            # Device
            "device": cls.device,
        }

    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("=" * 80)
        print("TWO-STAGE CVAE WITH TINY CONTEXT BOTTLENECK")
        print("=" * 80)
        print()
        print("GOAL: Fix variance collapse (P1 ~0.77% → target ~75%)")
        print()
        print("KEY CHANGES:")
        print(f"  ctx_embedding_dim: {cls.ctx_embedding_dim} (tiny bottleneck)")
        print(f"  ctx_surface_hidden: {cls.ctx_surface_hidden} (smaller encoder)")
        print(f"  ctx_mem_hidden: {cls.ctx_mem_hidden} (smaller LSTM)")
        print(f"  ctx_mem_layers: {cls.ctx_mem_layers} (fewer layers)")
        print()
        print("ARCHITECTURE:")
        print(f"  Context encoder output: {cls.ctx_embedding_dim} dims (like k-means)")
        print(f"  Main encoder (z): {cls.latent_dim} dims (expressive)")
        print(f"  Decoder input: {cls.ctx_embedding_dim + cls.latent_dim} dims")
        print(f"  LSTM hidden sizes:")
        print(f"    - ctx_encoder:  {cls.ctx_mem_hidden} (tiny)")
        print(f"    - main_encoder: {cls.mem_hidden} (tiny)")
        print(f"    - decoder:      {cls.decoder_mem_hidden} (tiny)")
        print(f"    - predictors:   {cls.predictor_hidden} (matches embed_dim)")
        print()
        print("TRAINING:")
        print(f"  Stage 1 (autoencoder): {cls.stage1_epochs} epochs")
        print(f"    - Loss modes: full, horizon, weighted")
        print(f"    - Weak KL: {cls.kl_weight}")
        print(f"    - LR: {cls.learning_rate}, Batch: {cls.batch_size}")
        print(f"  Stage 2 (predictors): {cls.stage2_epochs} epochs")
        print(f"    - Frozen autoencoder")
        print(f"    - MSE on detached targets")
        print(f"    - LR: {cls.stage2_learning_rate}, Batch: {cls.stage2_batch_size}")
        print()
        print("PREDICTOR ARCHITECTURE (autoregressive LSTM):")
        print(f"  Surface hidden: {cls.predictor_surface_hidden}")
        print(f"  LSTM hidden: {cls.predictor_hidden} (= embed_dim, no feedback projection)")
        print(f"  LSTM layers: {cls.predictor_layers}")
        print()
        print(f"Context length: {cls.context_len} days")
        print(f"Max horizon: {cls.max_horizon} days")
        print("=" * 80)


# For backward compatibility and easy access
TWO_STAGE_CONFIG = TwoStageConfig.get_model_config()
PREDICTOR_CONFIG = TwoStageConfig.get_predictor_config()
