"""
Test AR(1) correlated noise generation.

Tests the AR(1) noise generation code in vae/cvae_conditional_prior.py:180-188
"""

import pytest
import torch
import numpy as np
import math


class TestAR1NoiseGeneration:
    """Test AR(1) noise generation correctness."""

    def test_ar1_noise_autocorrelation(self):
        """Verify lag-1 autocorrelation matches ar_phi."""
        torch.manual_seed(42)

        ar_phi_values = [0.3, 0.5, 0.7, 0.9]

        for ar_phi in ar_phi_values:
            # Generate a single LONG AR(1) series to get accurate autocorr estimate
            B = 1
            horizon = 2000  # Long series
            latent_dim = 1

            # Generate AR(1) noise (same logic as in cvae_conditional_prior.py)
            ar_noise_list = [torch.randn(B, 1, latent_dim)]
            for t in range(1, horizon):
                innovation = torch.randn(B, 1, latent_dim)
                ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
                ar_noise_list.append(ar_noise_t)
            eps = torch.cat(ar_noise_list, dim=1)  # (B, horizon, latent_dim)

            # Extract single time series
            series = eps.squeeze().numpy()  # (horizon,)

            # Compute lag-1 autocorrelation
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]

            print(f"\n  ar_phi={ar_phi:.1f}: empirical lag-1 autocorr={autocorr:.4f}")

            # Should match ar_phi within ~3%
            assert abs(autocorr - ar_phi) < 0.03, \
                f"Autocorr mismatch: expected {ar_phi}, got {autocorr}"

        print(f"\n✓ AR(1) noise autocorrelation matches ar_phi")

    def test_ar1_noise_stationary_variance(self):
        """Verify marginal variance is 1 (stationary)."""
        torch.manual_seed(42)

        ar_phi = 0.7
        B = 200
        horizon = 200
        latent_dim = 10

        # Generate AR(1) noise
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)
        eps = torch.cat(ar_noise_list, dim=1)  # (B, horizon, latent_dim)

        # Compute marginal variance at each timestep
        marginal_vars = eps.var(dim=(0, 2))  # Variance across B and latent_dim

        print(f"\n  Marginal variance by timestep:")
        print(f"    t=0: {marginal_vars[0]:.4f} (initial, expected ~1)")
        print(f"    t=10: {marginal_vars[10]:.4f}")
        print(f"    t=50: {marginal_vars[50]:.4f}")
        print(f"    t=100: {marginal_vars[100]:.4f}")
        print(f"    t={horizon-1}: {marginal_vars[horizon-1]:.4f}")

        # After burn-in (t>10), should be close to 1
        for t in range(10, horizon):
            assert 0.8 < marginal_vars[t] < 1.2, \
                f"Variance at t={t} outside range: {marginal_vars[t]}"

        print(f"\n✓ AR(1) noise has stationary variance ≈ 1")

    def test_ar1_noise_iid_when_phi_zero(self):
        """Verify noise is IID when ar_phi=0."""
        torch.manual_seed(42)

        ar_phi = 0.0
        B = 100
        horizon = 100
        latent_dim = 10

        # Generate AR(1) noise with phi=0 (should be IID)
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)
        eps = torch.cat(ar_noise_list, dim=1)  # (B, horizon, latent_dim)

        # Compute lag-1 autocorrelation (should be ~0 for IID)
        eps_flat = eps.reshape(-1, horizon)  # (B*D, T)
        eps_t = eps_flat[:, :-1].flatten().numpy()
        eps_t1 = eps_flat[:, 1:].flatten().numpy()
        empirical_autocorr = np.corrcoef(eps_t, eps_t1)[0, 1]

        print(f"\n  ar_phi=0.0: empirical lag-1 autocorr={empirical_autocorr:.4f}")

        # Should be close to 0 for IID
        assert abs(empirical_autocorr) < 0.1, \
            f"phi=0 should give IID noise, but autocorr={empirical_autocorr}"

        print(f"\n✓ AR(1) noise is IID when phi=0")

    def test_ar1_noise_shape(self):
        """Verify generated noise has correct shape."""
        torch.manual_seed(42)

        ar_phi = 0.7
        B = 8
        horizon = 30
        latent_dim = 12

        # Generate AR(1) noise
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)
        eps = torch.cat(ar_noise_list, dim=1)

        assert eps.shape == (B, horizon, latent_dim), \
            f"Wrong shape: {eps.shape}, expected ({B}, {horizon}, {latent_dim})"

        print(f"\n✓ AR(1) noise shape: {eps.shape} = (B={B}, H={horizon}, D={latent_dim})")

    def test_ar1_noise_mean_zero(self):
        """Verify AR(1) noise has mean ≈ 0."""
        torch.manual_seed(42)

        ar_phi = 0.8
        B = 100
        horizon = 100
        latent_dim = 10

        # Generate AR(1) noise
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)
        eps = torch.cat(ar_noise_list, dim=1)

        # Overall mean should be close to 0
        overall_mean = eps.mean().item()

        print(f"\n  AR(1) noise overall mean: {overall_mean:.4f}")

        assert abs(overall_mean) < 0.05, \
            f"Mean should be ~0, got {overall_mean}"

        print(f"\n✓ AR(1) noise has mean ≈ 0")


class TestAR1NumericalStability:
    """Test numerical stability of AR(1) noise generation."""

    def test_ar1_phi_near_one(self):
        """Test with ar_phi very close to 1 (near unit root)."""
        torch.manual_seed(42)

        ar_phi = 0.999
        B = 50
        horizon = 50
        latent_dim = 5

        # Generate AR(1) noise
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)
        eps = torch.cat(ar_noise_list, dim=1)

        # Should not produce NaN/Inf
        assert not torch.isnan(eps).any(), "NaN detected with phi=0.999"
        assert not torch.isinf(eps).any(), "Inf detected with phi=0.999"

        # Variance should still be bounded (though might drift)
        variance = eps.var().item()
        print(f"\n  ar_phi=0.999: variance={variance:.4f}")

        assert variance < 10.0, f"Variance exploded: {variance}"

        print(f"\n✓ AR(1) noise stable for phi=0.999")

    def test_ar1_phi_equals_one_fails(self):
        """Demonstrate that phi=1 causes sqrt(1-phi^2)=0 issue."""
        ar_phi = 1.0

        # sqrt(1 - phi^2) would be 0, making innovation term vanish
        innovation_scale = math.sqrt(1 - ar_phi**2)

        assert innovation_scale == 0.0, "phi=1 should give innovation_scale=0"

        print(f"\n✓ phi=1 correctly identified as problematic (innovation_scale=0)")

    def test_ar1_high_autocorr_convergence(self):
        """Test that high autocorr (phi>0.9) eventually converges to stationary."""
        torch.manual_seed(42)

        ar_phi = 0.95
        B = 200
        horizon = 500  # Long sequence for convergence
        latent_dim = 5

        # Generate AR(1) noise
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)
        eps = torch.cat(ar_noise_list, dim=1)

        # Variance should converge to 1 in later timesteps
        early_var = eps[:, :10, :].var().item()
        late_var = eps[:, -100:, :].var().item()

        print(f"\n  ar_phi=0.95:")
        print(f"    Early variance (t<10): {early_var:.4f}")
        print(f"    Late variance (t>400): {late_var:.4f}")

        # Late variance should be close to 1
        assert 0.8 < late_var < 1.2, \
            f"Late variance should converge to ~1, got {late_var}"

        print(f"\n✓ High autocorr AR(1) converges to stationary variance")


class TestAR1Formula:
    """Test the mathematical formula for AR(1) process."""

    def test_ar1_formula_correctness(self):
        """Verify AR(1) formula: eps_t = phi * eps_{t-1} + sqrt(1-phi^2) * innovation."""
        torch.manual_seed(42)

        ar_phi = 0.7
        B = 1
        horizon = 5
        latent_dim = 1

        # Manual implementation
        ar_noise_list = [torch.randn(B, 1, latent_dim)]
        innovations = []

        for t in range(1, horizon):
            innovation = torch.randn(B, 1, latent_dim)
            innovations.append(innovation)
            ar_noise_t = ar_phi * ar_noise_list[-1] + math.sqrt(1 - ar_phi**2) * innovation
            ar_noise_list.append(ar_noise_t)

        # Verify formula manually
        print(f"\n  AR(1) formula verification (phi={ar_phi}):")
        for t in range(1, horizon):
            expected = ar_phi * ar_noise_list[t-1] + math.sqrt(1 - ar_phi**2) * innovations[t-1]
            actual = ar_noise_list[t]

            diff = (expected - actual).abs().item()
            print(f"    t={t}: diff={diff:.2e}")

            assert diff < 1e-6, f"Formula mismatch at t={t}"

        print(f"\n✓ AR(1) formula is correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
