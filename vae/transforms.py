"""
Differentiable Transforms for Probabilistic Modeling

This module provides bijective transforms that can be used to add
skewness and other properties to base distributions.

References:
- Jones, M.C. and Pewsey, A. (2009). Sinh-arcsinh distributions.
  Biometrika 96(4):761-780.
- TensorFlow Probability: tfp.bijectors.SinhArcsinh
"""

import torch
import torch.nn as nn
from typing import Tuple


class SinhArcsinhTransform(nn.Module):
    """
    Sinh-arcsinh bijection for adding skewness to symmetric distributions.

    The transformation is:
        Y = sinh((arcsinh(X) + ε) * δ)

    Where:
        X ~ base distribution (e.g., Student-t)
        ε (epsilon) ∈ (-∞, ∞): skewness parameter
            ε > 0 → positive skew (right tail heavier)
            ε < 0 → negative skew (left tail heavier)
        δ (delta) ∈ (0, ∞): tailweight parameter
            δ > 1 → fatter tails
            δ < 1 → thinner tails

    When ε=0 and δ=1, this is the identity transform.

    This transform is a diffeomorphism of the real line, meaning it's
    invertible and both the forward and inverse are differentiable.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward transform: X → Y (add skewness).

        Args:
            x: Input tensor from base distribution
            epsilon: Skewness parameter (same shape as x or broadcastable)
            delta: Tailweight parameter (same shape as x or broadcastable)

        Returns:
            y: Transformed tensor with skewness applied
        """
        return torch.sinh((torch.asinh(x) + epsilon) * delta)

    def inverse(
        self,
        y: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse transform: Y → X (remove skewness).

        Args:
            y: Transformed tensor
            epsilon: Skewness parameter
            delta: Tailweight parameter

        Returns:
            x: Original tensor in base distribution space
        """
        return torch.sinh(torch.asinh(y) / delta - epsilon)

    def log_abs_det_jacobian(
        self,
        y: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Log absolute determinant of Jacobian |dX/dY|.

        For density transformation:
            log p(y) = log p_base(x) + log |dx/dy|

        The Jacobian for sinh-arcsinh is:
            |dx/dy| = cosh(arcsinh(y)/δ - ε) / (δ * sqrt(1 + y²))

        So:
            log |dx/dy| = log(cosh(arcsinh(y)/δ - ε)) - log(δ) - 0.5*log(1 + y²)

        Note: This returns log |dX/dY|, not log |dY/dX|.
        For NLL, you typically want to ADD this term.

        Args:
            y: Transformed tensor
            epsilon: Skewness parameter
            delta: Tailweight parameter

        Returns:
            Log absolute determinant of Jacobian (same shape as y)
        """
        asinh_y = torch.asinh(y)
        inner = asinh_y / delta - epsilon

        # Numerically stable computation
        # log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        # For large |x|, this ≈ |x| - log(2)
        log_cosh = torch.logaddexp(inner, -inner) - torch.log(torch.tensor(2.0, device=y.device))

        log_jacobian = log_cosh - torch.log(delta) - 0.5 * torch.log1p(y ** 2)

        return log_jacobian

    def log_abs_det_jacobian_forward(
        self,
        x: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Log absolute determinant of Jacobian |dY/dX| (forward direction).

        For sampling, you might need this instead of the inverse Jacobian.

        The Jacobian for the forward transform is:
            |dy/dx| = δ * cosh((arcsinh(x) + ε) * δ) / sqrt(1 + x²)

        So:
            log |dy/dx| = log(δ) + log(cosh((arcsinh(x) + ε) * δ)) - 0.5*log(1 + x²)

        Args:
            x: Base distribution tensor
            epsilon: Skewness parameter
            delta: Tailweight parameter

        Returns:
            Log absolute determinant of forward Jacobian
        """
        asinh_x = torch.asinh(x)
        inner = (asinh_x + epsilon) * delta

        log_cosh = torch.logaddexp(inner, -inner) - torch.log(torch.tensor(2.0, device=x.device))

        log_jacobian = torch.log(delta) + log_cosh - 0.5 * torch.log1p(x ** 2)

        return log_jacobian


def skew_student_t_log_prob(
    y: torch.Tensor,
    mu: torch.Tensor,
    scale: torch.Tensor,
    nu: torch.Tensor,
    epsilon: torch.Tensor,
    delta: torch.Tensor
) -> torch.Tensor:
    """
    Log probability of skewed Student-t distribution.

    The skewed Student-t is defined by applying sinh-arcsinh transform
    to a standard Student-t:

        X ~ StudentT(df=ν)
        Y = μ + scale * sinh((arcsinh(X) + ε) * δ)

    Args:
        y: Observed values
        mu: Location parameter
        scale: Scale parameter (positive)
        nu: Degrees of freedom (positive)
        epsilon: Skewness parameter
        delta: Tailweight parameter (positive)

    Returns:
        Log probability (same shape as y)
    """
    transform = SinhArcsinhTransform()

    # Standardize
    y_std = (y - mu) / scale

    # Transform back to symmetric space
    x = transform.inverse(y_std, epsilon, delta)

    # Student-t log prob
    # log p(x; ν) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - ((ν+1)/2)*log(1 + x²/ν)
    log_prob_base = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * torch.pi)
        - ((nu + 1) / 2) * torch.log1p(x ** 2 / nu)
    )

    # Add Jacobian for sinh-arcsinh transform
    log_jacobian = transform.log_abs_det_jacobian(y_std, epsilon, delta)

    # Add Jacobian for scale
    log_scale_jacobian = -torch.log(scale)

    return log_prob_base + log_jacobian + log_scale_jacobian


def sample_skew_student_t(
    mu: torch.Tensor,
    scale: torch.Tensor,
    nu: torch.Tensor,
    epsilon: torch.Tensor,
    delta: torch.Tensor,
    n_samples: int = 1
) -> torch.Tensor:
    """
    Sample from skewed Student-t distribution.

    Args:
        mu: Location parameter (shape: [...])
        scale: Scale parameter (shape: [...])
        nu: Degrees of freedom (shape: [...])
        epsilon: Skewness parameter (shape: [...])
        delta: Tailweight parameter (shape: [...])
        n_samples: Number of samples

    Returns:
        Samples of shape (n_samples, ...)
    """
    transform = SinhArcsinhTransform()

    # Sample from standard Student-t
    # Using the fact that T ~ Normal(0,1) / sqrt(Chi2(nu)/nu)
    shape = (n_samples,) + mu.shape

    normal_samples = torch.randn(shape, device=mu.device, dtype=mu.dtype)
    chi2_samples = torch.distributions.Chi2(nu).sample((n_samples,))

    x = normal_samples / torch.sqrt(chi2_samples / nu)

    # Apply sinh-arcsinh transform
    y_std = transform.forward(x, epsilon, delta)

    # Scale and shift
    y = mu + scale * y_std

    return y


class SinhArcsinhStudentT(nn.Module):
    """
    Skewed Student-t distribution using sinh-arcsinh transformation.

    This is a convenience class that wraps the transform and provides
    a distribution-like interface.
    """

    def __init__(self):
        super().__init__()
        self.transform = SinhArcsinhTransform()

    def log_prob(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        L: torch.Tensor,
        nu: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Log probability with Cholesky-parameterized covariance.

        Args:
            y: Observed values, shape (..., D)
            mu: Mean, shape (..., D)
            L: Lower Cholesky factor, shape (..., D, D)
            nu: Degrees of freedom, shape (..., D) or (...,)
            epsilon: Skewness, shape (..., D)
            delta: Tailweight, shape (..., D)

        Returns:
            Log probability, shape (...)
        """
        # Solve L @ z = (y - mu) for z (whitened residuals)
        residual = y - mu  # (..., D)
        z = torch.linalg.solve_triangular(L, residual.unsqueeze(-1), upper=False).squeeze(-1)

        # Transform back to symmetric space
        x = self.transform.inverse(z, epsilon, delta)

        # Student-t log prob for each dimension
        log_prob_base = (
            torch.lgamma((nu + 1) / 2)
            - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * torch.pi)
            - ((nu + 1) / 2) * torch.log1p(x ** 2 / nu)
        )

        # Sum over dimensions
        log_prob_base = log_prob_base.sum(dim=-1)

        # Jacobian for sinh-arcsinh (sum over dimensions)
        log_jacobian_sa = self.transform.log_abs_det_jacobian(z, epsilon, delta).sum(dim=-1)

        # Jacobian for Cholesky (log det L)
        log_det_L = torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)

        return log_prob_base + log_jacobian_sa - log_det_L

    def sample(
        self,
        mu: torch.Tensor,
        L: torch.Tensor,
        nu: torch.Tensor,
        epsilon: torch.Tensor,
        delta: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample from the distribution.

        Args:
            mu: Mean, shape (B, D)
            L: Lower Cholesky factor, shape (B, D, D)
            nu: Degrees of freedom, shape (B, D)
            epsilon: Skewness, shape (B, D)
            delta: Tailweight, shape (B, D)
            n_samples: Number of samples

        Returns:
            Samples, shape (n_samples, B, D)
        """
        B, D = mu.shape

        # Sample from standard Student-t
        normal_samples = torch.randn(n_samples, B, D, device=mu.device, dtype=mu.dtype)

        # Per-dimension degrees of freedom
        chi2_dist = torch.distributions.Chi2(nu)  # shape (B, D)
        chi2_samples = chi2_dist.sample((n_samples,))  # (n_samples, B, D)

        x = normal_samples / torch.sqrt(chi2_samples / nu)  # (n_samples, B, D)

        # Apply sinh-arcsinh transform
        z_skewed = self.transform.forward(x, epsilon, delta)  # (n_samples, B, D)

        # Apply Cholesky: y = mu + L @ z_skewed
        # L: (B, D, D), z_skewed: (n_samples, B, D)
        y = mu + torch.einsum('bde,nbe->nbd', L, z_skewed)

        return y
