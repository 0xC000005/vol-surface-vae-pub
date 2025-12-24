"""
Unit tests for Full Covariance Prior components.

Run with: python -m pytest test_code/test_full_covariance_prior.py -v

Tests all components:
1. SinusoidalPositionEncoding
2. AR(1) covariance matrix
3. Cholesky decomposition
4. FullCovariancePrior sampling
5. Gradient flow
6. KL divergence
"""

import torch
import numpy as np
import pytest

from vae.full_covariance_prior import (
    SinusoidalPositionEncoding,
    PositionEncodedPriorMean,
    FullCovariancePrior,
    build_ar1_covariance,
    build_ar1_cholesky_direct,
    kl_divergence_full_covariance
)


class TestSinusoidalPositionEncoding:
    """Tests for position encoding component"""

    def test_output_shape(self):
        """Position encoding has correct shape (H, d_model)"""
        enc = SinusoidalPositionEncoding(d_model=64)
        out = enc(torch.arange(30))
        assert out.shape == (30, 64), f"Expected (30, 64), got {out.shape}"
        print("✓ Position encoding shape correct")

    def test_different_positions(self):
        """Different positions produce different encodings"""
        enc = SinusoidalPositionEncoding(d_model=64)
        out = enc(torch.arange(10))
        # No two rows should be identical
        for i in range(10):
            for j in range(i+1, 10):
                assert not torch.allclose(out[i], out[j]), \
                    f"Positions {i} and {j} have identical encodings!"
        print("✓ All positions have unique encodings")

    def test_deterministic(self):
        """Same position always gives same encoding"""
        enc = SinusoidalPositionEncoding(d_model=64)
        out1 = enc(torch.tensor([5]))
        out2 = enc(torch.tensor([5]))
        assert torch.allclose(out1, out2), "Position encoding not deterministic!"
        print("✓ Position encoding is deterministic")


class TestPositionEncodedPriorMean:
    """Tests for position-encoded prior mean network"""

    def test_output_shape(self):
        """Output has correct shape (B, H, latent_dim)"""
        net = PositionEncodedPriorMean(context_dim=12, latent_dim=12)
        ctx = torch.randn(8, 12)
        mu = net(ctx, horizon=30)
        assert mu.shape == (8, 30, 12), f"Expected (8, 30, 12), got {mu.shape}"
        print("✓ Prior mean output shape correct")

    def test_different_timesteps(self):
        """Different timesteps produce different means"""
        net = PositionEncodedPriorMean(context_dim=12, latent_dim=12)
        ctx = torch.randn(1, 12)
        mu = net(ctx, horizon=10)

        # Check that means vary across timesteps
        for t in range(9):
            assert not torch.allclose(mu[0, t], mu[0, t+1]), \
                f"Timesteps {t} and {t+1} have identical means!"
        print("✓ Different timesteps produce different means")


class TestAR1Covariance:
    """Tests for AR(1) covariance matrix"""

    def test_symmetric(self):
        """AR(1) covariance is symmetric"""
        Sigma = build_ar1_covariance(phi=0.7, sigma_sq=1.0, horizon=30)
        assert torch.allclose(Sigma, Sigma.T), "Covariance matrix not symmetric!"
        print("✓ AR(1) covariance is symmetric")

    def test_positive_definite(self):
        """AR(1) covariance is positive definite"""
        Sigma = build_ar1_covariance(phi=0.7, sigma_sq=1.0, horizon=30)
        eigenvalues = torch.linalg.eigvalsh(Sigma)
        assert (eigenvalues > 0).all(), \
            f"Covariance has non-positive eigenvalues: min={eigenvalues.min()}"
        print(f"✓ AR(1) covariance is positive definite (min eigenvalue: {eigenvalues.min():.6f})")

    def test_diagonal_equals_sigma_sq(self):
        """Diagonal elements equal σ²"""
        sigma_sq = 2.5
        Sigma = build_ar1_covariance(phi=0.7, sigma_sq=sigma_sq, horizon=30)
        diagonal = torch.diag(Sigma)
        expected = torch.full((30,), sigma_sq)
        assert torch.allclose(diagonal, expected), \
            f"Diagonal mismatch: got {diagonal[0]}, expected {sigma_sq}"
        print(f"✓ All diagonal elements equal σ² = {sigma_sq}")

    def test_off_diagonal_structure(self):
        """Off-diagonal elements follow φ^|i-j| pattern"""
        phi, sigma_sq = 0.6, 1.0
        Sigma = build_ar1_covariance(phi=phi, sigma_sq=sigma_sq, horizon=5)

        # Check Sigma[0,1] = sigma_sq * phi
        expected_lag1 = sigma_sq * phi
        assert torch.allclose(Sigma[0, 1], torch.tensor(expected_lag1)), \
            f"Lag-1 correlation mismatch: got {Sigma[0,1]}, expected {expected_lag1}"

        # Check Sigma[0,2] = sigma_sq * phi^2
        expected_lag2 = sigma_sq * phi**2
        assert torch.allclose(Sigma[0, 2], torch.tensor(expected_lag2)), \
            f"Lag-2 correlation mismatch: got {Sigma[0,2]}, expected {expected_lag2}"

        print(f"✓ AR(1) structure correct: Σ[i,j] = σ² × φ^|i-j|")


class TestCholeskyDecomposition:
    """Tests for Cholesky decomposition"""

    def test_cholesky_reconstructs_covariance(self):
        """L @ L.T = Σ"""
        phi, sigma_sq = 0.7, 1.5
        Sigma = build_ar1_covariance(phi, sigma_sq, horizon=30)
        L = torch.linalg.cholesky(Sigma)
        reconstructed = L @ L.T
        assert torch.allclose(Sigma, reconstructed, atol=1e-6), \
            "Cholesky decomposition does not reconstruct covariance!"
        print("✓ Cholesky decomposition: L @ L.T = Σ")

    def test_direct_cholesky_matches_generic(self):
        """Direct AR(1) Cholesky matches torch.linalg.cholesky"""
        phi, sigma = 0.7, 1.2
        sigma_sq = sigma ** 2
        Sigma = build_ar1_covariance(phi, sigma_sq, horizon=30)
        L_generic = torch.linalg.cholesky(Sigma)
        L_direct = build_ar1_cholesky_direct(phi, sigma, horizon=30)

        assert torch.allclose(L_generic, L_direct, atol=1e-5), \
            "Direct Cholesky doesn't match generic implementation!"
        print("✓ Direct AR(1) Cholesky matches torch.linalg.cholesky")

    def test_direct_cholesky_is_lower_triangular(self):
        """Direct Cholesky produces lower triangular matrix"""
        L = build_ar1_cholesky_direct(phi=0.6, sigma=1.0, horizon=20)
        # Upper triangle (excluding diagonal) should be zero
        upper_triangle = torch.triu(L, diagonal=1)
        assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle)), \
            "Direct Cholesky is not lower triangular!"
        print("✓ Direct Cholesky is lower triangular")


class TestFullCovariancePriorSampling:
    """Tests for FullCovariancePrior sampling"""

    def test_sample_shape_single(self):
        """Single sample has correct shape (B, H, latent_dim)"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12, max_horizon=90)
        ctx_summary = torch.randn(8, 12)  # Batch of 8
        z = prior.sample(ctx_summary, horizon=30)
        assert z.shape == (8, 30, 12), f"Expected (8, 30, 12), got {z.shape}"
        print("✓ Single sample shape correct")

    def test_sample_shape_multiple(self):
        """Multiple samples have correct shape (B, num_samples, H, latent_dim)"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12, max_horizon=90)
        ctx_summary = torch.randn(4, 12)
        z = prior.sample(ctx_summary, horizon=20, num_samples=5)
        assert z.shape == (4, 5, 20, 12), f"Expected (4, 5, 20, 12), got {z.shape}"
        print("✓ Multiple samples shape correct")

    def test_sample_autocorrelation(self):
        """Samples have autocorrelation ≈ φ"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12, init_phi=0.7)
        prior.eval()  # Set to eval mode for deterministic caching
        ctx_summary = torch.randn(1000, 12)

        with torch.no_grad():  # Disable gradients for this test
            z = prior.sample(ctx_summary, horizon=30)

        # Compute lag-1 autocorrelation
        z_flat = z[:, :, 0].detach().numpy()  # (1000, 30) for first latent dim
        autocorr_lag1 = np.corrcoef(z_flat[:, :-1].flatten(), z_flat[:, 1:].flatten())[0, 1]

        phi_actual = prior.get_phi().item()
        error = abs(autocorr_lag1 - phi_actual)
        assert error < 0.1, f"Autocorr {autocorr_lag1:.3f} != φ {phi_actual:.3f} (error: {error:.3f})"
        print(f"✓ Sample autocorrelation {autocorr_lag1:.3f} ≈ φ {phi_actual:.3f}")

    def test_samples_not_constant(self):
        """Samples vary (not all identical)"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        ctx_summary = torch.randn(10, 12)
        z = prior.sample(ctx_summary, horizon=20)

        # Check samples are not all the same
        assert z.std() > 0.01, "Samples are nearly constant!"
        print(f"✓ Samples vary (std: {z.std():.4f})")


class TestEmpiricalQuantiles:
    """Tests for empirical quantile computation"""

    def test_compute_empirical_quantiles_shape(self):
        """Quantiles have correct shape"""
        from vae.full_covariance_prior import compute_empirical_quantiles

        # Create fake samples: (num_samples, B, H, latent_dim)
        samples = torch.randn(1000, 5, 30, 12)
        quantiles_dict = compute_empirical_quantiles(samples)

        # Check default quantiles
        assert 'q05' in quantiles_dict
        assert 'q50' in quantiles_dict
        assert 'q95' in quantiles_dict

        # Check shapes - should remove sample dimension
        for key, val in quantiles_dict.items():
            assert val.shape == (5, 30, 12), f"{key} has wrong shape: {val.shape}"

        print("✓ Quantiles have correct shape (B, H, latent_dim)")

    def test_compute_empirical_quantiles_ordering(self):
        """Quantiles are ordered: q05 < q50 < q95"""
        from vae.full_covariance_prior import compute_empirical_quantiles

        samples = torch.randn(1000, 10, 20, 12)
        quantiles_dict = compute_empirical_quantiles(samples)

        q05 = quantiles_dict['q05']
        q50 = quantiles_dict['q50']
        q95 = quantiles_dict['q95']

        # Check ordering (should be true for most elements)
        ordering_violations = ((q05 > q50) | (q50 > q95)).float().mean()
        assert ordering_violations < 0.01, f"Quantile ordering violated in {ordering_violations*100:.1f}% of elements"

        print(f"✓ Quantiles ordered correctly (q05 < q50 < q95)")

    def test_compute_empirical_quantiles_custom_quantiles(self):
        """Custom quantile values work"""
        from vae.full_covariance_prior import compute_empirical_quantiles

        samples = torch.randn(500, 3, 15, 8)
        quantiles_dict = compute_empirical_quantiles(samples, quantiles=[0.25, 0.75])

        assert 'q25' in quantiles_dict
        assert 'q75' in quantiles_dict
        assert len(quantiles_dict) == 2

        print("✓ Custom quantiles work correctly")

    def test_sample_with_quantiles_method(self):
        """FullCovariancePrior.sample_with_quantiles() works"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        ctx_summary = torch.randn(5, 12)

        quantiles_dict = prior.sample_with_quantiles(ctx_summary, horizon=20, num_samples=500)

        # Check keys
        assert 'q05' in quantiles_dict
        assert 'q50' in quantiles_dict
        assert 'q95' in quantiles_dict

        # Check shapes
        for key, val in quantiles_dict.items():
            assert val.shape == (5, 20, 12), f"{key} has wrong shape: {val.shape}"

        print("✓ sample_with_quantiles() method works correctly")


class TestGradientFlow:
    """Tests for gradient flow through parameters"""

    def test_gradients_reach_phi(self):
        """Gradients flow to log_phi parameter"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        prior.train()  # Enable gradient computation
        ctx_summary = torch.randn(4, 12, requires_grad=True)
        z = prior.sample(ctx_summary, horizon=10)
        loss = z.sum()
        loss.backward()

        assert prior.log_phi.grad is not None, "No gradient for log_phi!"
        assert prior.log_phi.grad.abs() > 0, f"Zero gradient for log_phi!"
        print(f"✓ Gradient reaches log_phi (grad: {prior.log_phi.grad.item():.6f})")

    def test_gradients_reach_sigma(self):
        """Gradients flow to log_sigma_sq parameter"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        prior.train()
        ctx_summary = torch.randn(4, 12, requires_grad=True)
        z = prior.sample(ctx_summary, horizon=10)
        loss = z.sum()
        loss.backward()

        assert prior.log_sigma_sq.grad is not None, "No gradient for log_sigma_sq!"
        assert prior.log_sigma_sq.grad.abs() > 0, f"Zero gradient for log_sigma_sq!"
        print(f"✓ Gradient reaches log_sigma_sq (grad: {prior.log_sigma_sq.grad.item():.6f})")

    def test_mean_network_has_gradients(self):
        """Gradients flow through mean network"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        prior.train()
        ctx_summary = torch.randn(4, 12, requires_grad=True)
        z = prior.sample(ctx_summary, horizon=10)
        loss = z.sum()
        loss.backward()

        # Check at least one parameter has gradient
        has_grad = False
        for param in prior.mean_network.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients in mean network!"
        print("✓ Gradients flow through mean network")


class TestKLDivergence:
    """Tests for KL divergence computation"""

    def test_kl_positive(self):
        """KL divergence is always non-negative"""
        mu_q = torch.randn(8, 30, 12)
        logvar_q = torch.randn(8, 30, 12) * 0.5
        mu_p = torch.randn(8, 30, 12)
        Sigma_p = build_ar1_covariance(0.7, 1.0, 30)

        kl = kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)
        assert kl >= 0, f"KL divergence is negative: {kl}"
        print(f"✓ KL divergence is non-negative ({kl:.6f})")

    def test_kl_zero_when_identical(self):
        """KL ≈ 0 when distributions match"""
        # When posterior is N(0, I) and prior is N(0, I)
        mu = torch.zeros(8, 30, 12)
        logvar_q = torch.zeros(8, 30, 12)  # var = 1
        mu_p = torch.zeros(8, 30, 12)
        Sigma_p = torch.eye(30)  # I

        kl = kl_divergence_full_covariance(mu, logvar_q, mu_p, Sigma_p)
        assert kl < 0.01, f"KL should be ~0 for identical distributions, got {kl}"
        print(f"✓ KL ≈ 0 for identical distributions ({kl:.6f})")

    def test_kl_increases_with_divergence(self):
        """KL increases as distributions diverge"""
        Sigma_p = build_ar1_covariance(0.7, 1.0, 30)

        # Case 1: Similar means
        mu_q1 = torch.zeros(8, 30, 12)
        mu_p1 = torch.zeros(8, 30, 12)
        logvar_q1 = torch.zeros(8, 30, 12)
        kl1 = kl_divergence_full_covariance(mu_q1, logvar_q1, mu_p1, Sigma_p)

        # Case 2: Different means
        mu_q2 = torch.zeros(8, 30, 12)
        mu_p2 = torch.ones(8, 30, 12) * 2.0  # Shifted by 2
        logvar_q2 = torch.zeros(8, 30, 12)
        kl2 = kl_divergence_full_covariance(mu_q2, logvar_q2, mu_p2, Sigma_p)

        assert kl2 > kl1, f"KL should increase with mean difference: {kl1:.4f} vs {kl2:.4f}"
        print(f"✓ KL increases with divergence ({kl1:.4f} → {kl2:.4f})")


class TestFullCovariancePriorIntegration:
    """Integration tests for full prior"""

    def test_get_prior_params_shapes(self):
        """get_prior_params returns correct shapes"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        ctx = torch.randn(4, 12)
        mu_p, Sigma_p = prior.get_prior_params(ctx, horizon=20)

        assert mu_p.shape == (4, 20, 12), f"mu_p shape: {mu_p.shape}"
        assert Sigma_p.shape == (20, 20), f"Sigma_p shape: {Sigma_p.shape}"
        print("✓ get_prior_params returns correct shapes")

    def test_forward_returns_all(self):
        """forward() returns z, mu_p, Sigma_p"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        ctx = torch.randn(4, 12)
        z, mu_p, Sigma_p = prior.forward(ctx, horizon=15)

        assert z.shape == (4, 15, 12), f"z shape: {z.shape}"
        assert mu_p.shape == (4, 15, 12), f"mu_p shape: {mu_p.shape}"
        assert Sigma_p.shape == (15, 15), f"Sigma_p shape: {Sigma_p.shape}"
        print("✓ forward() returns all outputs with correct shapes")

    def test_phi_constrained(self):
        """φ is constrained to (0, 1)"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        phi = prior.get_phi().item()
        assert 0 < phi < 1, f"φ = {phi} not in (0, 1)!"
        print(f"✓ φ constrained to (0, 1): φ = {phi:.4f}")

    def test_sigma_sq_positive(self):
        """σ² is always positive"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        sigma_sq = prior.get_sigma_sq().item()
        assert sigma_sq > 0, f"σ² = {sigma_sq} not positive!"
        print(f"✓ σ² is positive: σ² = {sigma_sq:.4f}")

    def test_cholesky_cache_works(self):
        """Cholesky cache avoids recomputation"""
        prior = FullCovariancePrior(context_dim=12, latent_dim=12)
        ctx = torch.randn(4, 12)

        # First call - should cache
        L1 = prior.get_cholesky(horizon=20, device=ctx.device)
        cache_size_1 = len(prior._cholesky_cache)

        # Second call - should use cache
        L2 = prior.get_cholesky(horizon=20, device=ctx.device)
        cache_size_2 = len(prior._cholesky_cache)

        assert torch.allclose(L1, L2), "Cached Cholesky differs!"
        assert cache_size_1 == cache_size_2, "Cache size changed!"
        print(f"✓ Cholesky cache works (cache size: {cache_size_1})")


def run_all_tests():
    """Run all tests manually (for direct execution)"""
    print("=" * 80)
    print("FULL COVARIANCE PRIOR UNIT TESTS")
    print("=" * 80)
    print()

    test_classes = [
        TestSinusoidalPositionEncoding,
        TestPositionEncodedPriorMean,
        TestAR1Covariance,
        TestCholeskyDecomposition,
        TestFullCovariancePriorSampling,
        TestEmpiricalQuantiles,
        TestGradientFlow,
        TestKLDivergence,
        TestFullCovariancePriorIntegration
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

    print()
    print("=" * 80)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    if passed_tests == total_tests:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total_tests - passed_tests} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
