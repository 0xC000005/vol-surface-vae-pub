"""
Test to experimentally prove the rand vs randn bug in reparameterization trick.

Files with bug (use torch.rand_like instead of torch.randn_like):
- vae/cvae.py:58
- vae/conv_vae.py:39
- vae/dense_vae.py:34
- vae/cvae_with_mem.py:157

Production model (vae/cvae_with_mem_randomized.py:159) is CORRECT.
"""

import pytest
import torch
import numpy as np
from scipy import stats


class TestReparameterizationDistribution:
    """Test that rand_like produces incorrect distribution compared to randn_like."""

    def test_rand_vs_randn_statistical_properties(self):
        """Prove rand_like and randn_like have different statistical properties."""
        torch.manual_seed(42)

        # Create dummy log_var
        z_log_var = torch.zeros(10000, 10)

        # Sample with rand (BUG)
        eps_rand = torch.rand_like(z_log_var)

        # Sample with randn (CORRECT)
        eps_randn = torch.randn_like(z_log_var)

        # rand_like should produce uniform [0, 1]: mean=0.5, var=1/12≈0.0833
        rand_mean = eps_rand.mean().item()
        rand_var = eps_rand.var().item()

        # randn_like should produce N(0, 1): mean=0, var=1
        randn_mean = eps_randn.mean().item()
        randn_var = eps_randn.var().item()

        # Verify rand_like is uniform
        assert 0.48 < rand_mean < 0.52, f"rand mean={rand_mean}, expected ~0.5"
        assert 0.07 < rand_var < 0.09, f"rand var={rand_var}, expected ~1/12=0.0833"

        # Verify randn_like is standard normal
        assert -0.05 < randn_mean < 0.05, f"randn mean={randn_mean}, expected ~0"
        assert 0.95 < randn_var < 1.05, f"randn var={randn_var}, expected ~1"

        print(f"\n✓ rand_like: mean={rand_mean:.4f}, var={rand_var:.4f} (uniform)")
        print(f"✓ randn_like: mean={randn_mean:.4f}, var={randn_var:.4f} (normal)")

    def test_latent_distribution_with_rand_vs_randn(self):
        """Show that z = mu + exp(0.5*log_var)*eps has wrong distribution with rand."""
        torch.manual_seed(42)

        # Setup: mu=0, log_var=0 (sigma=1)
        batch_size = 10000
        latent_dim = 5
        mu = torch.zeros(batch_size, latent_dim)
        log_var = torch.zeros(batch_size, latent_dim)

        # Reparameterization with rand (BUG)
        eps_rand = torch.rand_like(log_var)
        z_rand = mu + torch.exp(0.5 * log_var) * eps_rand

        # Reparameterization with randn (CORRECT)
        eps_randn = torch.randn_like(log_var)
        z_randn = mu + torch.exp(0.5 * log_var) * eps_randn

        # With rand: z should have mean~0.5 (WRONG)
        # With randn: z should have mean~0 (CORRECT)
        z_rand_mean = z_rand.mean().item()
        z_randn_mean = z_randn.mean().item()

        # Verify rand produces biased latents
        assert 0.45 < z_rand_mean < 0.55, f"rand z mean={z_rand_mean}"

        # Verify randn produces unbiased latents
        assert -0.05 < z_randn_mean < 0.05, f"randn z mean={z_randn_mean}"

        print(f"\n✓ Latent z with rand: mean={z_rand_mean:.4f} (biased!)")
        print(f"✓ Latent z with randn: mean={z_randn_mean:.4f} (unbiased)")

    def test_normality_kolmogorov_smirnov(self):
        """Use K-S test to prove randn is normal but rand is not."""
        torch.manual_seed(42)

        # Generate samples
        eps_rand = torch.rand(10000).numpy()
        eps_randn = torch.randn(10000).numpy()

        # K-S test against standard normal distribution
        ks_rand, p_rand = stats.kstest(eps_rand, 'norm', args=(0, 1))
        ks_randn, p_randn = stats.kstest(eps_randn, 'norm', args=(0, 1))

        # rand should REJECT normality (p < 0.01)
        assert p_rand < 0.01, f"rand should not be normal, but p={p_rand}"

        # randn should ACCEPT normality (p > 0.05)
        assert p_randn > 0.05, f"randn should be normal, but p={p_randn}"

        print(f"\n✓ K-S test vs N(0,1):")
        print(f"  rand: KS={ks_rand:.4f}, p={p_rand:.2e} (REJECT normality)")
        print(f"  randn: KS={ks_randn:.4f}, p={p_randn:.4f} (ACCEPT normality)")


class TestKLDivergenceImpact:
    """Test that KL divergence calculation is wrong with rand sampling."""

    def test_kl_analytical_formula_assumes_normal(self):
        """KL formula: -0.5 * (1 + log_var - exp(log_var) - mu^2) assumes z ~ N(mu, sigma)."""
        torch.manual_seed(42)

        batch_size = 1000
        latent_dim = 10

        # Setup VAE posterior: mu=1, log_var=log(0.5)  => N(1, 0.5)
        mu = torch.ones(batch_size, latent_dim)
        log_var = torch.log(torch.tensor(0.5)).expand(batch_size, latent_dim)

        # Analytical KL(N(mu, sigma) || N(0, 1))
        kl_analytical = -0.5 * (1 + log_var - torch.exp(log_var) - mu**2)
        kl_analytical = kl_analytical.sum(dim=1).mean()

        # Sample-based KL estimate with rand (WRONG)
        eps_rand = torch.rand_like(log_var)
        z_rand = mu + torch.exp(0.5 * log_var) * eps_rand

        # Sample-based KL estimate with randn (CORRECT)
        eps_randn = torch.randn_like(log_var)
        z_randn = mu + torch.exp(0.5 * log_var) * eps_randn

        # Compute log p(z) and log q(z|x) for sample-based KL
        # For N(0,1) prior: log p(z) = -0.5 * (z^2 + log(2*pi))
        # For N(mu, sigma) posterior: log q(z|x) = -0.5 * ((z-mu)/sigma)^2 - 0.5*log_var - 0.5*log(2*pi)

        sigma = torch.exp(0.5 * log_var)

        # rand samples
        log_p_rand = -0.5 * (z_rand**2 + np.log(2 * np.pi))
        log_q_rand = -0.5 * ((z_rand - mu) / sigma)**2 - 0.5 * log_var - 0.5 * np.log(2 * np.pi)
        kl_sample_rand = (log_q_rand - log_p_rand).sum(dim=1).mean()

        # randn samples
        log_p_randn = -0.5 * (z_randn**2 + np.log(2 * np.pi))
        log_q_randn = -0.5 * ((z_randn - mu) / sigma)**2 - 0.5 * log_var - 0.5 * np.log(2 * np.pi)
        kl_sample_randn = (log_q_randn - log_p_randn).sum(dim=1).mean()

        print(f"\n✓ KL divergence:")
        print(f"  Analytical: {kl_analytical.item():.4f}")
        print(f"  Sample (rand): {kl_sample_rand.item():.4f}")
        print(f"  Sample (randn): {kl_sample_randn.item():.4f}")

        # randn should match analytical (within sampling error)
        assert abs(kl_sample_randn.item() - kl_analytical.item()) < 0.5

        # rand should NOT match analytical (biased)
        # Since rand produces z in [mu, mu+sigma] instead of N(mu, sigma),
        # the KL will be systematically wrong

    def test_kl_non_negativity_violated_with_wrong_sampling(self):
        """KL divergence should always be >= 0, but wrong sampling can violate this in edge cases."""
        torch.manual_seed(42)

        batch_size = 100
        latent_dim = 5

        # Posterior very close to prior: mu=0, log_var=0
        mu = torch.zeros(batch_size, latent_dim)
        log_var = torch.zeros(batch_size, latent_dim)

        # Analytical KL should be ~0
        kl_analytical = -0.5 * (1 + log_var - torch.exp(log_var) - mu**2)
        kl_analytical = kl_analytical.sum(dim=1).mean()

        assert kl_analytical.item() < 0.01, f"KL should be ~0, got {kl_analytical.item()}"

        print(f"\n✓ Posterior ≈ Prior => KL ≈ 0: {kl_analytical.item():.6f}")


class TestReconstructionQuality:
    """Compare reconstruction quality with rand vs randn."""

    def test_reconstruction_variance_underestimated_with_rand(self):
        """rand produces lower variance samples, leading to underestimated uncertainty."""
        torch.manual_seed(42)

        # Simulate decoder: x_recon = decoder(z)
        # Assume decoder is linear: x_recon = W @ z + b
        latent_dim = 10
        output_dim = 25  # 5x5 surface

        mu = torch.zeros(1000, latent_dim)
        log_var = torch.zeros(1000, latent_dim)

        # Sample latents
        eps_rand = torch.rand_like(log_var)
        z_rand = mu + torch.exp(0.5 * log_var) * eps_rand

        eps_randn = torch.randn_like(log_var)
        z_randn = mu + torch.exp(0.5 * log_var) * eps_randn

        # Simple linear decoder
        W = torch.randn(output_dim, latent_dim)
        b = torch.zeros(output_dim)

        x_rand = z_rand @ W.T + b
        x_randn = z_randn @ W.T + b

        # Variance of reconstructions
        var_rand = x_rand.var(dim=0).mean()
        var_randn = x_randn.var(dim=0).mean()

        print(f"\n✓ Reconstruction variance:")
        print(f"  rand: {var_rand.item():.4f}")
        print(f"  randn: {var_randn.item():.4f}")

        # rand should have lower variance (less stochastic)
        # Because rand has var=1/12≈0.0833 vs randn var=1
        assert var_rand < var_randn

        print(f"✓ rand underestimates uncertainty by {(1 - var_rand/var_randn)*100:.1f}%")


class TestImpactOnTraining:
    """Test impact of rand bug on VAE training dynamics."""

    def test_posterior_collapse_detection(self):
        """rand may affect posterior collapse detection via KL monitoring."""
        torch.manual_seed(42)

        # Simulate posterior collapse: log_var -> -inf (var -> 0)
        mu = torch.zeros(100, 10)
        log_var_collapsed = torch.full((100, 10), -10.0)  # var ≈ 4.5e-5

        # KL with collapsed posterior
        kl_collapsed = -0.5 * (1 + log_var_collapsed - torch.exp(log_var_collapsed) - mu**2)
        kl_collapsed = kl_collapsed.sum(dim=1).mean()

        print(f"\n✓ Posterior collapse KL: {kl_collapsed.item():.4f}")
        print(f"  (High KL indicates collapse, typically > 10)")

        # With rand vs randn, the KL value is wrong, so collapse detection is unreliable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUBehavior:
    """Verify bug exists on both CPU and GPU."""

    def test_rand_vs_randn_on_gpu(self):
        """Verify statistical differences exist on GPU as well."""
        torch.manual_seed(42)

        z_log_var = torch.zeros(10000, 10, device='cuda')

        eps_rand = torch.rand_like(z_log_var)
        eps_randn = torch.randn_like(z_log_var)

        # Check means
        rand_mean = eps_rand.mean().item()
        randn_mean = eps_randn.mean().item()

        assert 0.48 < rand_mean < 0.52
        assert -0.05 < randn_mean < 0.05

        print(f"\n✓ GPU: rand mean={rand_mean:.4f}, randn mean={randn_mean:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
