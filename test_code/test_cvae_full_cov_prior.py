"""
Integration tests for CVAEFullCovPrior model.

Tests the complete model including:
1. Initialization and config validation
2. Forward pass
3. Training step
4. Generation with correlated sampling
5. Parameter updates

Run with: python test_code/test_cvae_full_cov_prior.py
"""

import torch
import numpy as np
from vae.cvae_full_cov_prior import CVAEFullCovPrior


def create_minimal_valid_config():
    """Create minimal valid config for testing"""
    return {
        "feat_dim": (5, 5),
        "latent_dim": 12,
        "surface_hidden": [5, 5, 5],
        "ctx_surface_hidden": [5, 5, 5],
        "ex_feats_dim": 3,
        "ex_feats_hidden": None,
        "ctx_ex_feats_hidden": None,
        "context_len": 20,
        "re_feat_weight": 1.0,
        "kl_weight": 1.0,
        "mem_type": "lstm",
        "mem_hidden": 50,
        "mem_layers": 2,
        "mem_dropout": 0.3,
        "interaction_layers": 2,
        "use_dense_surface": False,
        "compress_context": True,
        "ex_loss_on_ret_only": True,
        "ex_feats_loss_type": "l2",
        "device": "cpu",
        "horizon": 5,
        "max_horizon": 90,
        # Full covariance prior config
        "full_cov_pos_dim": 64,
        "full_cov_hidden_dim": 128,
        "full_cov_init_phi": 0.5,
        "full_cov_init_sigma_sq": 1.0,
    }


def create_dummy_input(batch_size=4, seq_len=25):
    """Create dummy input for testing (context=20, horizon=5)"""
    return {
        "surface": torch.randn(batch_size, seq_len, 5, 5),
        "ex_feats": torch.randn(batch_size, seq_len, 3)
    }


def create_context_only(batch_size=10):
    """Create context-only input for generation testing"""
    return {
        "surface": torch.randn(batch_size, 20, 5, 5),
        "ex_feats": torch.randn(batch_size, 20, 3)
    }


class TestCVAEFullCovPriorInit:
    """Tests for model initialization"""

    def test_missing_required_config_raises(self):
        """Missing required config keys raise ValueError"""
        try:
            CVAEFullCovPrior({"feat_dim": (5, 5)})  # Missing latent_dim, context_len
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Missing required config key" in str(e)
            print(f"✓ Correctly raises ValueError: {str(e)[:80]}...")

    def test_valid_config_initializes(self):
        """Valid config initializes model successfully"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        assert hasattr(model, 'full_cov_prior'), "Model missing full_cov_prior!"
        assert model.full_cov_prior is not None, "full_cov_prior is None!"
        print("✓ Model initializes with valid config")

    def test_prior_parameters_initialized(self):
        """Full covariance prior parameters are initialized"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)

        # Check φ and σ² are initialized
        phi = model.full_cov_prior.get_phi().item()
        sigma_sq = model.full_cov_prior.get_sigma_sq().item()

        assert 0 < phi < 1, f"φ={phi} not in (0, 1)"
        assert sigma_sq > 0, f"σ²={sigma_sq} not positive"
        print(f"✓ Prior parameters initialized: φ={phi:.4f}, σ²={sigma_sq:.4f}")


class TestCVAEFullCovPriorForward:
    """Tests for forward pass"""

    def test_forward_output_shapes(self):
        """Forward pass produces correct output shapes"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.eval()

        x = create_dummy_input(batch_size=4, seq_len=25)  # 20 context + 5 horizon

        with torch.no_grad():
            surf_recon, ex_recon, z_mean, z_logvar, z = model.forward(x)

        assert surf_recon.shape == (4, 5, 5, 5), f"Surface shape: {surf_recon.shape}"
        assert ex_recon.shape == (4, 5, 3), f"Ex feats shape: {ex_recon.shape}"
        assert z_mean.shape == (4, 25, 12), f"Z mean shape: {z_mean.shape}"
        assert z_logvar.shape == (4, 25, 12), f"Z logvar shape: {z_logvar.shape}"
        print("✓ Forward pass output shapes correct")

    def test_forward_without_ex_feats(self):
        """Forward pass works without extra features"""
        config = create_minimal_valid_config()
        # Set ex_feats_dim to 0 to disable extra features
        config["ex_feats_dim"] = 0
        config["ex_feats_hidden"] = None
        config["ctx_ex_feats_hidden"] = None
        model = CVAEFullCovPrior(config)
        model.eval()

        x = {"surface": torch.randn(4, 25, 5, 5)}  # No ex_feats

        with torch.no_grad():
            surf_recon, z_mean, z_logvar, z = model.forward(x)

        assert surf_recon.shape == (4, 5, 5, 5)
        print("✓ Forward pass works without ex_feats")


class TestCVAEFullCovPriorGeneration:
    """Tests for generation"""

    def test_generation_shape(self):
        """Generated surfaces have correct shape"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.eval()

        ctx = create_context_only(batch_size=8)

        with torch.no_grad():
            surf, ex = model.get_surface_given_conditions(ctx, horizon=10)

        assert surf.shape == (8, 10, 5, 5), f"Surface shape: {surf.shape}"
        assert ex.shape == (8, 10, 3), f"Ex feats shape: {ex.shape}"
        print("✓ Generation produces correct shapes")

    def test_generation_produces_correlated_samples(self):
        """Generated samples are correlated (not IID)"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.eval()

        ctx = create_context_only(batch_size=100)

        # Generate multiple sequences
        surfaces = []
        with torch.no_grad():
            for _ in range(10):
                surf, _ = model.get_surface_given_conditions(ctx, horizon=30)
                surfaces.append(surf)

        # Stack: (10, 100, 30, 5, 5) -> (1000, 30) for autocorrelation test
        all_surf = torch.stack(surfaces)
        flat = all_surf[:, :, :, 2, 2].reshape(-1, 30).numpy()  # Center grid point

        # Compute lag-1 autocorrelation
        autocorr = np.corrcoef(flat[:, :-1].flatten(), flat[:, 1:].flatten())[0, 1]

        # Get φ from model
        phi = model.full_cov_prior.get_phi().item()

        # Autocorrelation should be positive (not ~0 like IID)
        assert autocorr > 0.2, f"Autocorr {autocorr:.3f} too low - samples may be IID!"

        # Should be somewhat close to φ (within reason, since surfaces go through decoder)
        print(f"✓ Samples show correlation (autocorr={autocorr:.3f}, φ={phi:.3f})")

    def test_prior_mode_validation(self):
        """Only 'full_cov' prior mode is allowed"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.eval()

        ctx = create_context_only(batch_size=2)

        try:
            with torch.no_grad():
                model.get_surface_given_conditions(ctx, prior_mode="standard")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "only supports prior_mode='full_cov'" in str(e)
            print(f"✓ Correctly rejects invalid prior_mode")


class TestCVAEFullCovPriorTraining:
    """Tests for training"""

    def test_train_step_runs(self):
        """Train step completes without error"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = create_dummy_input(batch_size=4, seq_len=25)

        losses = model.train_step(batch, optimizer)

        assert 'loss' in losses, "Missing 'loss' in output"
        assert 'kl_loss' in losses, "Missing 'kl_loss' in output"
        assert 'reconstruction_loss' in losses, "Missing 'reconstruction_loss' in output"
        assert losses['loss'] > 0, "Loss should be positive"
        print(f"✓ Train step runs (loss={losses['loss'].item():.4f})")

    def test_phi_updates_during_training(self):
        """φ parameter updates during training"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Higher LR for test

        phi_before = model.full_cov_prior.get_phi().item()

        # Train for a few steps
        for _ in range(10):
            batch = create_dummy_input(batch_size=8, seq_len=25)
            model.train_step(batch, optimizer)

        phi_after = model.full_cov_prior.get_phi().item()

        assert phi_before != phi_after, f"φ should change during training (before={phi_before:.4f}, after={phi_after:.4f})"
        print(f"✓ φ updates during training ({phi_before:.4f} → {phi_after:.4f})")

    def test_sigma_sq_updates_during_training(self):
        """σ² parameter updates during training"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        sigma_sq_before = model.full_cov_prior.get_sigma_sq().item()

        for _ in range(10):
            batch = create_dummy_input(batch_size=8, seq_len=25)
            model.train_step(batch, optimizer)

        sigma_sq_after = model.full_cov_prior.get_sigma_sq().item()

        assert sigma_sq_before != sigma_sq_after, f"σ² should change during training"
        print(f"✓ σ² updates during training ({sigma_sq_before:.4f} → {sigma_sq_after:.4f})")

    def test_multihorizon_train_step(self):
        """Multi-horizon training step works"""
        config = create_minimal_valid_config()
        config["horizon"] = 30  # Need longer sequences for multi-horizon
        model = CVAEFullCovPrior(config)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = create_dummy_input(batch_size=4, seq_len=60)  # 20 ctx + 40 for multi-horizon

        losses = model.train_step_multihorizon(
            batch, optimizer, horizons=[1, 7, 14, 30]
        )

        assert 'loss' in losses
        assert 'horizon_losses' in losses
        assert losses['loss'] > 0
        assert len(losses['horizon_losses']) == 4
        print(f"✓ Multi-horizon training works (horizons: {list(losses['horizon_losses'].keys())})")


class TestCVAEFullCovPriorUtilities:
    """Tests for utility methods"""

    def test_get_prior_params_summary(self):
        """get_prior_params_summary returns correct info"""
        config = create_minimal_valid_config()
        model = CVAEFullCovPrior(config)

        summary = model.get_prior_params_summary()

        assert 'phi' in summary
        assert 'sigma_sq' in summary
        assert 'num_params' in summary
        assert 'covariance_params' in summary

        assert summary['covariance_params'] == 2, "Should have exactly 2 covariance params (log_phi, log_sigma_sq)"
        print(f"✓ Prior summary: φ={summary['phi']:.4f}, σ²={summary['sigma_sq']:.4f}, total_params={summary['num_params']}")


class TestConfigValidation:
    """Tests for V4 Full Covariance Prior config validation"""

    def test_fixed_context_length_enforced(self):
        """Verify context length is fixed at 60 for V4"""
        from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov as cfg

        assert cfg.context_len == 60, f"Expected context_len=60, got {cfg.context_len}"
        assert cfg.min_context_len == 60, f"Expected min_context_len=60, got {cfg.min_context_len}"
        assert cfg.max_context_len == 60, f"Expected max_context_len=60, got {cfg.max_context_len}"
        assert cfg.min_context_len == cfg.max_context_len, "Context length must be fixed (not variable)"

        print(f"✓ Context length is fixed at 60 (min=max=60)")

    def test_quantile_regression_disabled(self):
        """Verify quantile regression is disabled in V4"""
        from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov as cfg

        assert cfg.use_quantile_regression == False, "Quantile regression should be disabled in V4"
        assert cfg.num_quantiles == 1, f"Expected num_quantiles=1, got {cfg.num_quantiles}"

        print("✓ Quantile regression is disabled (use empirical percentiles instead)")

    def test_v4_priors_enabled(self):
        """Verify V4 full covariance prior is enabled"""
        from config.backfill_context60_config_v4_full_cov import BackfillContext60ConfigV4FullCov as cfg

        # V3 priors should be disabled
        assert cfg.use_conditional_prior == False, "V3 conditional prior should be disabled"
        assert cfg.use_fitted_prior == False, "Fitted prior should be disabled"

        # V4 config should have full cov parameters
        assert hasattr(cfg, 'full_cov_pos_dim'), "Missing full_cov_pos_dim"
        assert hasattr(cfg, 'full_cov_hidden_dims'), "Missing full_cov_hidden_dims"
        assert hasattr(cfg, 'full_cov_dropout'), "Missing full_cov_dropout"
        assert hasattr(cfg, 'full_cov_init_phi'), "Missing full_cov_init_phi"

        print(f"✓ V4 Full Covariance Prior enabled (hidden_dims={cfg.full_cov_hidden_dims}, dropout={cfg.full_cov_dropout})")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 80)
    print("CVAE FULL COVARIANCE PRIOR INTEGRATION TESTS")
    print("=" * 80)
    print()

    test_classes = [
        TestCVAEFullCovPriorInit,
        TestCVAEFullCovPriorForward,
        TestCVAEFullCovPriorGeneration,
        TestCVAEFullCovPriorTraining,
        TestCVAEFullCovPriorUtilities,
        TestConfigValidation,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 80)
        test_instance = test_class()
        methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in methods:
            total_tests += 1
            try:
                getattr(test_instance, method_name)()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

    print()
    print("=" * 80)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    if passed_tests == total_tests:
        print("\n✓ ALL INTEGRATION TESTS PASSED!")
        print("\nReady for full training!")
        return 0
    else:
        print(f"\n✗ {total_tests - passed_tests} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
