"""
Test KL divergence between two diagonal Gaussians.

Tests the function in vae/conditional_prior_network.py:212-233
"""

import pytest
import torch
import numpy as np


# Import the function to test
from vae.conditional_prior_network import kl_divergence_gaussians


class TestKLDivergenceGaussiansCorrectness:
    """Test mathematical correctness of KL divergence formula."""

    def test_kl_is_non_negative(self):
        """KL divergence must always be >= 0."""
        torch.manual_seed(42)

        for _ in range(10):
            B, T, D = np.random.randint(2, 10), np.random.randint(5, 20), np.random.randint(5, 15)

            mu_q = torch.randn(B, T, D)
            logvar_q = torch.randn(B, T, D)
            mu_p = torch.randn(B, T, D)
            logvar_p = torch.randn(B, T, D)

            kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

            assert kl.item() >= -1e-6, f"KL should be non-negative, got {kl.item()}"

        print("\n✓ KL is non-negative for 10 random configurations")

    def test_kl_zero_for_identical_distributions(self):
        """KL(q || q) = 0 for identical distributions."""
        torch.manual_seed(42)

        B, T, D = 5, 10, 8

        mu = torch.randn(B, T, D)
        logvar = torch.randn(B, T, D)

        # KL between identical distributions should be zero
        kl = kl_divergence_gaussians(mu, logvar, mu, logvar)

        assert abs(kl.item()) < 1e-6, f"KL(q||q) should be 0, got {kl.item()}"

        print(f"\n✓ KL(q || q) = {kl.item():.2e} ≈ 0")

    def test_kl_increases_with_divergence(self):
        """KL should increase as distributions become more different."""
        torch.manual_seed(42)

        B, T, D = 3, 5, 4

        mu_q = torch.zeros(B, T, D)
        logvar_q = torch.zeros(B, T, D)

        # Prior with increasing mean distance
        kl_values = []
        for mu_p_val in [0.0, 0.5, 1.0, 2.0, 5.0]:
            mu_p = torch.full((B, T, D), mu_p_val)
            logvar_p = torch.zeros(B, T, D)

            kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)
            kl_values.append(kl.item())

        # KL should be increasing
        for i in range(len(kl_values) - 1):
            assert kl_values[i] < kl_values[i+1], f"KL should increase with divergence"

        print(f"\n✓ KL increases with mean separation:")
        for mu_val, kl_val in zip([0.0, 0.5, 1.0, 2.0, 5.0], kl_values):
            print(f"  μ_p={mu_val:.1f}: KL={kl_val:.4f}")

    def test_kl_symmetric_in_variance_ratio(self):
        """KL should behave correctly with variance differences."""
        torch.manual_seed(42)

        B, T, D = 2, 3, 5

        mu_q = torch.zeros(B, T, D)
        mu_p = torch.zeros(B, T, D)

        # Test with different variance ratios
        logvar_q = torch.zeros(B, T, D)  # var_q = 1

        kl_values = {}
        for logvar_p_val in [-2, -1, 0, 1, 2]:  # var_p in [0.135, 0.368, 1, 2.718, 7.389]
            logvar_p = torch.full((B, T, D), float(logvar_p_val))
            kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)
            kl_values[logvar_p_val] = kl.item()

        # Minimum KL should be at logvar_p = logvar_q = 0
        assert kl_values[0] < kl_values[-2], "KL should increase when var_p deviates from var_q"
        assert kl_values[0] < kl_values[2], "KL should increase when var_p deviates from var_q"

        print(f"\n✓ KL with different prior variances (μ_q=μ_p=0, var_q=1):")
        for logvar_p_val in [-2, -1, 0, 1, 2]:
            var_p = np.exp(logvar_p_val)
            print(f"  var_p={var_p:.3f}: KL={kl_values[logvar_p_val]:.4f}")


class TestKLDivergenceAnalyticalVsMonteCarlo:
    """Compare closed-form KL with Monte Carlo estimate."""

    def test_analytical_matches_monte_carlo(self):
        """Verify closed-form KL matches Monte Carlo estimate from samples."""
        torch.manual_seed(42)

        B, T, D = 1, 1, 10  # Use 1x1 for simplicity

        # Setup distributions
        mu_q = torch.ones(B, T, D)
        logvar_q = torch.log(torch.tensor(0.5)).expand(B, T, D)
        mu_p = torch.zeros(B, T, D)
        logvar_p = torch.zeros(B, T, D)

        # Analytical KL
        kl_analytical = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

        # Monte Carlo KL: E_q[log q(z) - log p(z)]
        num_samples = 50000
        sigma_q = torch.exp(0.5 * logvar_q)
        sigma_p = torch.exp(0.5 * logvar_p)

        # Sample from q
        eps = torch.randn(num_samples, D)
        z = mu_q.squeeze() + sigma_q.squeeze() * eps  # (num_samples, D)

        # log q(z) = -0.5 * ((z-mu_q)/sigma_q)^2 - 0.5*logvar_q - 0.5*log(2*pi)
        log_q = -0.5 * ((z - mu_q.squeeze()) / sigma_q.squeeze())**2 - 0.5 * logvar_q.squeeze() - 0.5 * np.log(2 * np.pi)

        # log p(z) = -0.5 * ((z-mu_p)/sigma_p)^2 - 0.5*logvar_p - 0.5*log(2*pi)
        log_p = -0.5 * ((z - mu_p.squeeze()) / sigma_p.squeeze())**2 - 0.5 * logvar_p.squeeze() - 0.5 * np.log(2 * np.pi)

        # KL = E[log_q - log_p]
        kl_monte_carlo = (log_q - log_p).sum(dim=1).mean()

        print(f"\n✓ KL divergence comparison:")
        print(f"  Analytical: {kl_analytical.item():.4f}")
        print(f"  Monte Carlo (50k samples): {kl_monte_carlo.item():.4f}")
        print(f"  Difference: {abs(kl_analytical.item() - kl_monte_carlo.item()):.4f}")

        # Should match within sampling error (~1%)
        assert abs(kl_analytical.item() - kl_monte_carlo.item()) / kl_analytical.item() < 0.02, \
            f"Analytical vs MC mismatch: {kl_analytical.item()} vs {kl_monte_carlo.item()}"


class TestKLDivergenceGradientFlow:
    """Test gradient flow through KL computation."""

    def test_gradients_wrt_posterior_parameters(self):
        """Verify gradients flow back to posterior parameters."""
        torch.manual_seed(42)

        B, T, D = 2, 3, 5

        mu_q = torch.randn(B, T, D, requires_grad=True)
        logvar_q = torch.randn(B, T, D, requires_grad=True)
        mu_p = torch.randn(B, T, D)
        logvar_p = torch.randn(B, T, D)

        kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)
        kl.backward()

        assert mu_q.grad is not None, "Gradient should flow to mu_q"
        assert logvar_q.grad is not None, "Gradient should flow to logvar_q"
        assert mu_q.grad.abs().sum() > 0, "Gradients should be non-zero"
        assert logvar_q.grad.abs().sum() > 0, "Gradients should be non-zero"

        print(f"\n✓ Gradients flow to posterior parameters")
        print(f"  ∂KL/∂mu_q: mean={mu_q.grad.abs().mean():.4f}")
        print(f"  ∂KL/∂logvar_q: mean={logvar_q.grad.abs().mean():.4f}")

    def test_gradients_wrt_prior_parameters(self):
        """Verify gradients flow back to prior parameters."""
        torch.manual_seed(42)

        B, T, D = 2, 3, 5

        mu_q = torch.randn(B, T, D)
        logvar_q = torch.randn(B, T, D)
        mu_p = torch.randn(B, T, D, requires_grad=True)
        logvar_p = torch.randn(B, T, D, requires_grad=True)

        kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)
        kl.backward()

        assert mu_p.grad is not None, "Gradient should flow to mu_p"
        assert logvar_p.grad is not None, "Gradient should flow to logvar_p"
        assert mu_p.grad.abs().sum() > 0, "Gradients should be non-zero"
        assert logvar_p.grad.abs().sum() > 0, "Gradients should be non-zero"

        print(f"\n✓ Gradients flow to prior parameters")
        print(f"  ∂KL/∂mu_p: mean={mu_p.grad.abs().mean():.4f}")
        print(f"  ∂KL/∂logvar_p: mean={logvar_p.grad.abs().mean():.4f}")


class TestKLDivergenceEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_variance_posterior(self):
        """Test with very small posterior variance (near posterior collapse)."""
        torch.manual_seed(42)

        B, T, D = 2, 3, 5

        mu_q = torch.zeros(B, T, D)
        logvar_q = torch.full((B, T, D), -10.0)  # var ≈ 4.5e-5
        mu_p = torch.zeros(B, T, D)
        logvar_p = torch.zeros(B, T, D)  # var = 1

        kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

        assert not torch.isnan(kl), "KL should not be NaN"
        assert not torch.isinf(kl), "KL should not be Inf"
        assert kl > 0, "KL should be positive (small var_q => large KL)"

        print(f"\n✓ Small posterior variance (logvar=-10): KL={kl.item():.4f}")

    def test_very_large_variance_posterior(self):
        """Test with very large posterior variance."""
        torch.manual_seed(42)

        B, T, D = 2, 3, 5

        mu_q = torch.zeros(B, T, D)
        logvar_q = torch.full((B, T, D), 10.0)  # var ≈ 22026
        mu_p = torch.zeros(B, T, D)
        logvar_p = torch.zeros(B, T, D)  # var = 1

        kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

        assert not torch.isnan(kl), "KL should not be NaN"
        assert not torch.isinf(kl), "KL should not be Inf"

        print(f"\n✓ Large posterior variance (logvar=10): KL={kl.item():.4f}")

    def test_extreme_mean_separation(self):
        """Test with very large mean separation."""
        torch.manual_seed(42)

        B, T, D = 2, 3, 5

        mu_q = torch.zeros(B, T, D)
        logvar_q = torch.zeros(B, T, D)
        mu_p = torch.full((B, T, D), 100.0)  # Very far from q
        logvar_p = torch.zeros(B, T, D)

        kl = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

        assert not torch.isnan(kl), "KL should not be NaN"
        assert not torch.isinf(kl), "KL should not be Inf"
        assert kl > 100, "KL should be very large for extreme separation"

        print(f"\n✓ Extreme mean separation (Δμ=100): KL={kl.item():.2f}")


class TestKLDivergenceBatchConsistency:
    """Test that batching doesn't affect results."""

    def test_batch_consistency(self):
        """KL should be consistent across batch dimension."""
        torch.manual_seed(42)

        D = 10

        # Single sample
        mu_q_single = torch.randn(1, 1, D)
        logvar_q_single = torch.randn(1, 1, D)
        mu_p_single = torch.randn(1, 1, D)
        logvar_p_single = torch.randn(1, 1, D)

        kl_single = kl_divergence_gaussians(mu_q_single, logvar_q_single, mu_p_single, logvar_p_single)

        # Replicated batch
        B = 10
        mu_q_batch = mu_q_single.expand(B, -1, -1)
        logvar_q_batch = logvar_q_single.expand(B, -1, -1)
        mu_p_batch = mu_p_single.expand(B, -1, -1)
        logvar_p_batch = logvar_p_single.expand(B, -1, -1)

        kl_batch = kl_divergence_gaussians(mu_q_batch, logvar_q_batch, mu_p_batch, logvar_p_batch)

        print(f"\n✓ Batch consistency:")
        print(f"  Single: {kl_single.item():.6f}")
        print(f"  Batch (10x): {kl_batch.item():.6f}")

        assert abs(kl_single.item() - kl_batch.item()) < 1e-6, "KL should be same for single vs batch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
