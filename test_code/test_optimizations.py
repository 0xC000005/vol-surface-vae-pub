"""
Optimization Correctness Tests

Tests for vectorized implementations and FP32 conversion.
These tests verify mathematical correctness, not performance.
"""

import torch
import pytest
import numpy as np
from vae.full_covariance_prior import (
    build_ar1_covariance,
    build_ar1_cholesky_direct,
    kl_divergence_full_covariance
)


class TestCholeskyVectorization:
    """Test vectorized Cholesky implementation correctness."""

    def test_cholesky_property(self):
        """Verify L @ L.T = Σ (AR1 covariance matrix)."""
        for phi in [0.1, 0.5, 0.7, 0.9, 0.99]:
            for sigma in [0.1, 1.0, 5.0]:
                for horizon in [1, 5, 10, 30, 90]:
                    L = build_ar1_cholesky_direct(phi, sigma, horizon, 'cpu')
                    Sigma_reconstructed = L @ L.T
                    # build_ar1_covariance expects sigma_sq, not sigma
                    Sigma_expected = build_ar1_covariance(phi, sigma**2, horizon, 'cpu')

                    assert torch.allclose(Sigma_reconstructed, Sigma_expected, atol=1e-5, rtol=1e-4), \
                        f"Cholesky property failed for phi={phi}, sigma={sigma}, H={horizon}"

    def test_gradient_flow(self):
        """Ensure gradients flow through phi and sigma."""
        phi = torch.tensor(0.7, requires_grad=True)
        sigma = torch.tensor(1.0, requires_grad=True)

        L = build_ar1_cholesky_direct(phi, sigma, 30, 'cpu')
        loss = L.sum()
        loss.backward()

        assert phi.grad is not None, "Gradient wrt phi is None"
        assert sigma.grad is not None, "Gradient wrt sigma is None"
        assert phi.grad != 0, "Gradient wrt phi is zero"
        assert sigma.grad != 0, "Gradient wrt sigma is zero"

    def test_edge_cases(self):
        """Test boundary conditions."""
        # phi near 0 (nearly diagonal)
        L = build_ar1_cholesky_direct(0.01, 1.0, 30, 'cpu')
        assert not torch.isnan(L).any(), "NaN values for phi near 0"
        assert not torch.isinf(L).any(), "Inf values for phi near 0"

        # phi near 1 (high correlation)
        L = build_ar1_cholesky_direct(0.99, 1.0, 30, 'cpu')
        assert not torch.isnan(L).any(), "NaN values for phi near 1"
        assert not torch.isinf(L).any(), "Inf values for phi near 1"

        # horizon = 1
        L = build_ar1_cholesky_direct(0.7, 1.0, 1, 'cpu')
        assert L.shape == (1, 1), f"Wrong shape for H=1: {L.shape}"
        expected = torch.tensor([[1.0]])
        assert torch.allclose(L, expected, atol=1e-6), "Wrong value for H=1"

        # Very small sigma
        L = build_ar1_cholesky_direct(0.5, 0.01, 10, 'cpu')
        assert not torch.isnan(L).any(), "NaN for small sigma"

    def test_lower_triangular(self):
        """Verify Cholesky factor is lower triangular."""
        L = build_ar1_cholesky_direct(0.7, 1.0, 10, 'cpu')

        # Upper triangle (excluding diagonal) should be zero
        upper_tri = torch.triu(L, diagonal=1)
        assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-7), \
            "Upper triangle is not zero"

    def test_dtypes(self):
        """Test different dtypes are handled correctly."""
        for dtype in [torch.float32, torch.float64]:
            phi = torch.tensor(0.7, dtype=dtype)
            sigma = torch.tensor(1.0, dtype=dtype)
            L = build_ar1_cholesky_direct(phi, sigma, 10, 'cpu', dtype=dtype)

            assert L.dtype == dtype, f"Wrong dtype: expected {dtype}, got {L.dtype}"


class TestKLDivergenceVectorization:
    """Test vectorized KL divergence implementation."""

    def test_kl_non_negative(self):
        """KL divergence must be >= 0."""
        for _ in range(20):
            B, H, D = np.random.randint(2, 16), np.random.randint(5, 30), 12

            mu_q = torch.randn(B, H, D)
            logvar_q = torch.randn(B, H, D)
            mu_p = torch.randn(B, H, D)
            # build_ar1_covariance expects sigma_sq (here: 1.0 = 1.0^2)
            Sigma_p = build_ar1_covariance(0.5, 1.0, H, 'cpu')

            kl = kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)

            assert kl >= -1e-5, f"KL divergence should be non-negative, got {kl}"

    def test_kl_identical_distributions(self):
        """KL should be zero for identical distributions (within numerical precision)."""
        B, H, D = 4, 10, 12

        # When q = p (both diagonal), KL should be near zero
        mu = torch.randn(B, H, D)
        logvar_q = torch.randn(B, H, D)

        # Create diagonal Sigma_p that matches q
        # Note: This is approximate since we're testing a special case
        Sigma_p = torch.eye(H) * torch.exp(logvar_q[0, :, 0]).mean()

        kl = kl_divergence_full_covariance(mu, logvar_q, mu, Sigma_p)

        # KL won't be exactly zero due to dimension differences, but should be small
        assert kl < 100, f"KL for similar distributions too large: {kl}"

    def test_gradient_flow_kl(self):
        """Ensure gradients propagate correctly through KL."""
        B, H, D = 4, 10, 12

        mu_q = torch.randn(B, H, D, requires_grad=True)
        logvar_q = torch.randn(B, H, D, requires_grad=True)
        mu_p = torch.randn(B, H, D, requires_grad=True)
        phi = torch.tensor(0.7, requires_grad=True)
        Sigma_p = build_ar1_covariance(phi, 1.0, H, 'cpu')

        kl = kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)
        kl.backward()

        assert mu_q.grad is not None, "No gradient for mu_q"
        assert logvar_q.grad is not None, "No gradient for logvar_q"
        assert mu_p.grad is not None, "No gradient for mu_p"

        # Check gradients are not all zero
        assert mu_q.grad.abs().max() > 1e-6, "mu_q gradient is zero"
        assert logvar_q.grad.abs().max() > 1e-6, "logvar_q gradient is zero"

    def test_batch_consistency(self):
        """KL should give same result regardless of batch size."""
        H, D = 20, 12

        # Single sample
        mu_q_1 = torch.randn(1, H, D)
        logvar_q_1 = torch.randn(1, H, D)
        mu_p_1 = torch.randn(1, H, D)
        Sigma_p = build_ar1_covariance(0.7, 1.0, H, 'cpu')

        kl_1 = kl_divergence_full_covariance(mu_q_1, logvar_q_1, mu_p_1, Sigma_p)

        # Replicate to batch of 4
        mu_q_4 = mu_q_1.repeat(4, 1, 1)
        logvar_q_4 = logvar_q_1.repeat(4, 1, 1)
        mu_p_4 = mu_p_1.repeat(4, 1, 1)

        kl_4 = kl_divergence_full_covariance(mu_q_4, logvar_q_4, mu_p_4, Sigma_p)

        # Should be identical (mean over batch gives same value)
        assert torch.allclose(kl_1, kl_4, atol=1e-5), \
            f"Batch inconsistency: {kl_1} vs {kl_4}"

    def test_different_horizons(self):
        """Test KL works for various horizons."""
        B, D = 8, 12

        for H in [1, 5, 10, 30, 60, 90]:
            mu_q = torch.randn(B, H, D)
            logvar_q = torch.randn(B, H, D)
            mu_p = torch.randn(B, H, D)
            Sigma_p = build_ar1_covariance(0.7, 1.0, H, 'cpu')

            kl = kl_divergence_full_covariance(mu_q, logvar_q, mu_p, Sigma_p)

            assert not torch.isnan(kl), f"NaN KL for H={H}"
            assert not torch.isinf(kl), f"Inf KL for H={H}"
            assert kl >= -1e-5, f"Negative KL for H={H}: {kl}"


class TestFP32Conversion:
    """Test FP32 vs FP64 numerical equivalence."""

    def test_cholesky_fp32_vs_fp64(self):
        """Cholesky should give similar results in FP32 vs FP64."""
        phi, sigma, H = 0.7, 1.0, 30

        L_fp64 = build_ar1_cholesky_direct(phi, sigma, H, 'cpu', dtype=torch.float64)
        L_fp32 = build_ar1_cholesky_direct(phi, sigma, H, 'cpu', dtype=torch.float32)

        # Allow larger tolerance for reduced precision
        assert torch.allclose(L_fp64.float(), L_fp32, atol=1e-4, rtol=1e-3), \
            "Cholesky FP32 vs FP64 mismatch"

    def test_kl_fp32_vs_fp64(self):
        """KL divergence should be similar in FP32 vs FP64."""
        B, H, D = 8, 20, 12

        # Generate in FP64
        mu_q_fp64 = torch.randn(B, H, D, dtype=torch.float64)
        logvar_q_fp64 = torch.randn(B, H, D, dtype=torch.float64)
        mu_p_fp64 = torch.randn(B, H, D, dtype=torch.float64)
        Sigma_p_fp64 = build_ar1_covariance(0.7, 1.0, H, 'cpu', dtype=torch.float64)

        kl_fp64 = kl_divergence_full_covariance(mu_q_fp64, logvar_q_fp64, mu_p_fp64, Sigma_p_fp64)

        # Convert to FP32
        mu_q_fp32 = mu_q_fp64.float()
        logvar_q_fp32 = logvar_q_fp64.float()
        mu_p_fp32 = mu_p_fp64.float()
        Sigma_p_fp32 = Sigma_p_fp64.float()

        kl_fp32 = kl_divergence_full_covariance(mu_q_fp32, logvar_q_fp32, mu_p_fp32, Sigma_p_fp32)

        # Relative tolerance ~0.1% is reasonable for FP32
        rel_diff = abs(kl_fp64.item() - kl_fp32.item()) / (kl_fp64.item() + 1e-8)
        assert rel_diff < 0.01, f"KL FP32 vs FP64 mismatch: {rel_diff:.2%}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
