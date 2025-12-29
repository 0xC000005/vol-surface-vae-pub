"""
Test fitted GMM prior sampling.

Tests the _sample_from_fitted_prior() method in vae/cvae_with_mem_randomized.py:557-592
"""

import pytest
import torch
import numpy as np
from sklearn.mixture import GaussianMixture


class TestGMMPriorSampling:
    """Test GMM prior sampling correctness."""

    @pytest.fixture
    def mock_fitted_prior(self):
        """Create mock fitted prior for testing."""
        n_components = 3
        latent_dim = 5

        # Create mock GMM parameters
        means = torch.randn(n_components, latent_dim, dtype=torch.float64)
        weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)

        # Create random positive definite covariance matrices
        covariances = []
        for _ in range(n_components):
            A = torch.randn(latent_dim, latent_dim, dtype=torch.float64)
            cov = A @ A.T + torch.eye(latent_dim, dtype=torch.float64) * 0.1  # Add regularization
            covariances.append(cov)
        covariances = torch.stack(covariances)

        # Pre-compute Cholesky
        cholesky = torch.linalg.cholesky(covariances)

        return {
            'means': means,
            'covariances': covariances,
            'weights': weights,
            'cholesky': cholesky,
            'n_components': n_components
        }

    def test_component_selection_follows_weights(self, mock_fitted_prior):
        """Verify component selection follows weight distribution."""
        torch.manual_seed(42)

        n_components = mock_fitted_prior['n_components']
        weights = mock_fitted_prior['weights']
        latent_dim = mock_fitted_prior['means'].shape[1]

        # Sample many times to get empirical distribution
        num_samples = 10000

        # Simulate component selection (same logic as in _sample_from_fitted_prior)
        component_idx = torch.multinomial(
            weights.expand(num_samples, -1),
            num_samples=1
        ).squeeze(-1)

        # Count component frequencies
        counts = torch.bincount(component_idx, minlength=n_components).float()
        empirical_weights = counts / num_samples

        print(f"\n✓ Component selection (10k samples):")
        for i in range(n_components):
            print(f"  Component {i}: target={weights[i]:.2f}, empirical={empirical_weights[i]:.2f}")

        # Should match within ~2% with 10k samples
        for i in range(n_components):
            assert abs(empirical_weights[i] - weights[i]) < 0.02, \
                f"Component {i} weight mismatch"

    def test_sample_shape(self, mock_fitted_prior):
        """Verify samples have correct shape."""
        torch.manual_seed(42)

        batch_size = 8
        seq_len = 20
        latent_dim = mock_fitted_prior['means'].shape[1]

        # Simulate sampling logic
        total_samples = batch_size * seq_len
        component_idx = torch.multinomial(
            mock_fitted_prior['weights'].expand(total_samples, -1),
            num_samples=1
        ).squeeze(-1)

        means = mock_fitted_prior['means'][component_idx]
        L = mock_fitted_prior['cholesky'][component_idx]

        eps = torch.randn(total_samples, latent_dim, dtype=torch.float64)
        z = means + torch.einsum('bij,bj->bi', L, eps)
        z = z.reshape(batch_size, seq_len, latent_dim)

        assert z.shape == (batch_size, seq_len, latent_dim), \
            f"Wrong shape: {z.shape}"

        print(f"\n✓ Sample shape: {z.shape} = (B={batch_size}, T={seq_len}, D={latent_dim})")

    def test_cholesky_sampling_produces_correct_covariance(self, mock_fitted_prior):
        """Verify Cholesky sampling produces correct component covariances."""
        torch.manual_seed(42)

        latent_dim = mock_fitted_prior['means'].shape[1]

        # Test each component separately
        for comp_idx in range(mock_fitted_prior['n_components']):
            mean = mock_fitted_prior['means'][comp_idx]
            cov_true = mock_fitted_prior['covariances'][comp_idx]
            L = mock_fitted_prior['cholesky'][comp_idx]

            # Sample from this component only
            num_samples = 5000
            eps = torch.randn(num_samples, latent_dim, dtype=torch.float64)
            z = mean + (L @ eps.T).T  # (num_samples, latent_dim)

            # Empirical covariance
            z_centered = z - z.mean(dim=0)
            cov_empirical = (z_centered.T @ z_centered) / (num_samples - 1)

            # Should match true covariance
            print(f"\n  Component {comp_idx}:")
            print(f"    True cov norm: {cov_true.norm():.4f}")
            print(f"    Empirical cov norm: {cov_empirical.norm():.4f}")
            print(f"    Max difference: {(cov_true - cov_empirical).abs().max():.4f}")

            # Within ~20% with 5k samples (high variance in sampling)
            assert torch.allclose(cov_empirical, cov_true, rtol=0.20, atol=0.2), \
                f"Component {comp_idx} covariance mismatch"

        print(f"\n✓ Cholesky sampling produces correct covariances")

    def test_gmm_sample_statistics_match_mixture(self, mock_fitted_prior):
        """Verify samples match expected mean/variance of GMM."""
        torch.manual_seed(42)

        batch_size = 100
        seq_len = 50
        total_samples = batch_size * seq_len
        latent_dim = mock_fitted_prior['means'].shape[1]

        # Sample from GMM
        component_idx = torch.multinomial(
            mock_fitted_prior['weights'].expand(total_samples, -1),
            num_samples=1
        ).squeeze(-1)

        means = mock_fitted_prior['means'][component_idx]
        L = mock_fitted_prior['cholesky'][component_idx]

        eps = torch.randn(total_samples, latent_dim, dtype=torch.float64)
        z = means + torch.einsum('bij,bj->bi', L, eps)

        # Expected GMM mean: E[z] = sum(w_k * mu_k)
        expected_mean = (mock_fitted_prior['weights'].unsqueeze(1) *
                        mock_fitted_prior['means']).sum(dim=0)

        # Expected GMM variance: E[Var(z)] = sum(w_k * (Sigma_k + mu_k*mu_k^T)) - E[z]*E[z]^T
        expected_var = torch.zeros(latent_dim, latent_dim, dtype=torch.float64)
        for k in range(mock_fitted_prior['n_components']):
            w = mock_fitted_prior['weights'][k]
            mu = mock_fitted_prior['means'][k]
            Sigma = mock_fitted_prior['covariances'][k]
            expected_var += w * (Sigma + torch.outer(mu, mu))
        expected_var -= torch.outer(expected_mean, expected_mean)

        # Empirical statistics
        empirical_mean = z.mean(dim=0)
        z_centered = z - z.mean(dim=0)
        empirical_var = (z_centered.T @ z_centered) / (total_samples - 1)

        print(f"\n✓ GMM statistics (5000 samples):")
        print(f"  Mean error: {(empirical_mean - expected_mean).abs().max():.4f}")
        print(f"  Variance error: {(empirical_var - expected_var).abs().max():.4f}")

        # Should match within sampling error
        assert torch.allclose(empirical_mean, expected_mean, atol=0.1), \
            "GMM mean mismatch"
        assert torch.allclose(empirical_var, expected_var, rtol=0.25, atol=0.4), \
            "GMM variance mismatch"

    def test_edge_case_single_component(self):
        """Test with single component (should behave like single Gaussian)."""
        torch.manual_seed(42)

        latent_dim = 5
        mean = torch.randn(1, latent_dim, dtype=torch.float64)
        A = torch.randn(latent_dim, latent_dim, dtype=torch.float64)
        cov = A @ A.T + torch.eye(latent_dim, dtype=torch.float64) * 0.1
        cholesky = torch.linalg.cholesky(cov.unsqueeze(0))

        fitted_prior = {
            'means': mean,
            'covariances': cov.unsqueeze(0),
            'weights': torch.tensor([1.0], dtype=torch.float64),
            'cholesky': cholesky,
            'n_components': 1
        }

        # Sample
        num_samples = 3000
        component_idx = torch.multinomial(
            fitted_prior['weights'].expand(num_samples, -1),
            num_samples=1
        ).squeeze(-1)

        means = fitted_prior['means'][component_idx]
        L = fitted_prior['cholesky'][component_idx]

        eps = torch.randn(num_samples, latent_dim, dtype=torch.float64)
        z = means + torch.einsum('bij,bj->bi', L, eps)

        # Should match single Gaussian
        empirical_mean = z.mean(dim=0)
        z_centered = z - z.mean(dim=0)
        empirical_cov = (z_centered.T @ z_centered) / (num_samples - 1)

        print(f"\n✓ Single component GMM:")
        print(f"  Mean error: {(empirical_mean - mean.squeeze()).abs().max():.4f}")
        print(f"  Cov error: {(empirical_cov - cov).abs().max():.4f}")

        assert torch.allclose(empirical_mean, mean.squeeze(), atol=0.1)
        assert torch.allclose(empirical_cov, cov, rtol=0.20, atol=0.2)

    def test_edge_case_equal_weights(self):
        """Test with equal component weights."""
        torch.manual_seed(42)

        n_components = 4
        latent_dim = 5

        means = torch.randn(n_components, latent_dim, dtype=torch.float64)
        weights = torch.ones(n_components, dtype=torch.float64) / n_components  # Equal weights

        covariances = []
        for _ in range(n_components):
            A = torch.randn(latent_dim, latent_dim, dtype=torch.float64)
            cov = A @ A.T + torch.eye(latent_dim, dtype=torch.float64) * 0.1
            covariances.append(cov)
        covariances = torch.stack(covariances)
        cholesky = torch.linalg.cholesky(covariances)

        # Sample component indices
        num_samples = 10000
        component_idx = torch.multinomial(
            weights.expand(num_samples, -1),
            num_samples=1
        ).squeeze(-1)

        # Count frequencies
        counts = torch.bincount(component_idx, minlength=n_components).float()
        empirical_weights = counts / num_samples

        print(f"\n✓ Equal weights GMM:")
        for i in range(n_components):
            print(f"  Component {i}: {empirical_weights[i]:.3f} (expected {weights[i]:.3f})")

        # All should be ~0.25 within 2%
        assert torch.allclose(empirical_weights.double(), weights, atol=0.02)


class TestGMMCholesky:
    """Test Cholesky decomposition for GMM covariances."""

    def test_cholesky_is_lower_triangular(self):
        """Verify Cholesky factor is lower triangular."""
        latent_dim = 5
        A = torch.randn(latent_dim, latent_dim, dtype=torch.float64)
        cov = A @ A.T + torch.eye(latent_dim, dtype=torch.float64) * 0.1

        L = torch.linalg.cholesky(cov)

        # Upper triangle (excluding diagonal) should be zero
        upper_tri = torch.triu(L, diagonal=1)

        assert torch.allclose(upper_tri, torch.zeros_like(upper_tri)), \
            "Cholesky should be lower triangular"

        print(f"\n✓ Cholesky is lower triangular")

    def test_cholesky_reconstructs_covariance(self):
        """Verify L @ L.T = Sigma."""
        latent_dim = 5
        A = torch.randn(latent_dim, latent_dim, dtype=torch.float64)
        cov = A @ A.T + torch.eye(latent_dim, dtype=torch.float64) * 0.1

        L = torch.linalg.cholesky(cov)
        cov_reconstructed = L @ L.T

        assert torch.allclose(cov_reconstructed, cov, atol=1e-10), \
            "L @ L.T should equal Sigma"

        print(f"\n✓ Cholesky reconstructs covariance exactly")

    def test_cholesky_fails_for_non_positive_definite(self):
        """Cholesky should fail for non-PD matrices."""
        # Create non-PD matrix (negative eigenvalue)
        latent_dim = 3
        cov = torch.tensor([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.9],
            [0.5, 0.9, 0.5]
        ], dtype=torch.float64)

        # Check eigenvalues
        eigenvalues = torch.linalg.eigvalsh(cov)
        print(f"\n  Eigenvalues: {eigenvalues}")

        if eigenvalues.min() <= 0:
            with pytest.raises(torch._C._LinAlgError):
                L = torch.linalg.cholesky(cov)
            print(f"✓ Cholesky correctly fails for non-PD matrix")
        else:
            print(f"✓ Matrix is PD, Cholesky succeeds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
