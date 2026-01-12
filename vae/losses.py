"""
Temporal Loss Functions for VAE Training

This module provides loss functions for preserving temporal dynamics in VAE models,
specifically targeting ACF (autocorrelation) preservation.

Key losses:
1. spectral_loss - Match power spectrum (implicitly matches ACF via Wiener-Khinchin)
2. acf_loss - Direct ACF matching
3. ar_consistency_loss - AR(1) consistency for mean-reversion

References:
- Wiener-Khinchin theorem: Power spectrum ↔ ACF are Fourier transform pairs
- Focal Frequency Loss (ICCV 2021)
- GARCH-NN Hybrid approaches
"""

import torch
import torch.nn.functional as F


def spectral_loss(pred: torch.Tensor, target: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Match power spectrum between predictions and targets.

    By the Wiener-Khinchin theorem, matching the power spectral density
    implicitly matches the autocorrelation function.

    Args:
        pred: Predictions of shape (B, T, ...) or (B, T, H, W)
        target: Targets of shape (B, T, ...) or (B, T, H, W)
        dim: Time dimension to apply FFT along (default: 1)

    Returns:
        Scalar loss value
    """
    # FFT along time dimension
    pred_fft = torch.fft.rfft(pred, dim=dim)
    target_fft = torch.fft.rfft(target, dim=dim)

    # Match magnitude (power spectrum)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    return F.mse_loss(pred_mag, target_mag)


def focal_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal frequency loss - emphasizes hard-to-match frequency components.

    Inspired by Focal Frequency Loss (ICCV 2021) which down-weights easy
    frequencies and focuses on hard ones.

    Args:
        pred: Predictions
        target: Targets
        dim: Time dimension
        gamma: Focusing parameter (higher = more focus on hard frequencies)

    Returns:
        Scalar loss value
    """
    pred_fft = torch.fft.rfft(pred, dim=dim)
    target_fft = torch.fft.rfft(target, dim=dim)

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    # Per-frequency error
    freq_error = (pred_mag - target_mag) ** 2

    # Normalize error to [0, 1] for weighting
    error_normalized = freq_error / (freq_error.max() + 1e-8)

    # Focal weight: higher weight on harder (larger error) frequencies
    focal_weight = error_normalized ** gamma

    # Weighted loss
    weighted_loss = (focal_weight * freq_error).mean()

    return weighted_loss


def differentiable_acf(x: torch.Tensor, lag: int = 1, dim: int = 0) -> torch.Tensor:
    """
    Compute autocorrelation at given lag in a differentiable manner.

    ACF(lag) = Cov(x_t, x_{t+lag}) / Var(x)

    Args:
        x: Time series tensor of shape (T, ...) or (B, T, ...)
        lag: Lag for autocorrelation
        dim: Time dimension

    Returns:
        ACF values (scalar or tensor depending on input shape)
    """
    # Center the data
    x_centered = x - x.mean(dim=dim, keepdim=True)

    # Variance
    var = (x_centered ** 2).mean(dim=dim)

    # Covariance at lag
    if dim == 0:
        x_t = x_centered[:-lag]
        x_t_lag = x_centered[lag:]
    elif dim == 1:
        x_t = x_centered[:, :-lag]
        x_t_lag = x_centered[:, lag:]
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    cov = (x_t * x_t_lag).mean(dim=dim)

    return cov / (var + 1e-8)


def acf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lags: list = [1],
    dim: int = 1
) -> torch.Tensor:
    """
    Match ACF between predictions and targets at specified lags.

    Args:
        pred: Predictions of shape (B, T, ...)
        target: Targets of shape (B, T, ...)
        lags: List of lags to match (default: [1])
        dim: Time dimension

    Returns:
        Scalar loss value (average across lags)
    """
    total_loss = 0.0

    for lag in lags:
        pred_acf = differentiable_acf(pred, lag=lag, dim=dim)
        target_acf = differentiable_acf(target, lag=lag, dim=dim)
        total_loss = total_loss + F.mse_loss(pred_acf, target_acf)

    return total_loss / len(lags)


def multi_lag_acf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_lag: int = 5,
    dim: int = 1,
    weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Match ACF at multiple lags with optional weighting.

    Args:
        pred: Predictions
        target: Targets
        max_lag: Maximum lag to consider
        dim: Time dimension
        weights: Optional weights for each lag (default: uniform)

    Returns:
        Scalar loss value
    """
    if weights is None:
        weights = torch.ones(max_lag, device=pred.device) / max_lag

    total_loss = 0.0
    for lag in range(1, max_lag + 1):
        pred_acf = differentiable_acf(pred, lag=lag, dim=dim)
        target_acf = differentiable_acf(target, lag=lag, dim=dim)
        total_loss = total_loss + weights[lag - 1] * F.mse_loss(pred_acf, target_acf)

    return total_loss


def ar_consistency_loss(
    sequence: torch.Tensor,
    phi: torch.Tensor,
    mu: torch.Tensor = None,
    dim: int = 1
) -> torch.Tensor:
    """
    AR(1) consistency loss: Encourages sequence to follow AR(1) dynamics.

    x_t = μ + φ(x_{t-1} - μ) + ε_t

    The residuals ε_t should be uncorrelated if the AR(1) model is correct.

    Args:
        sequence: Time series of shape (B, T, ...)
        phi: AR(1) coefficient (should be in [-1, 1] for stationarity)
        mu: Long-run mean (default: estimated from sequence)
        dim: Time dimension

    Returns:
        Scalar loss measuring deviation from AR(1) structure
    """
    if mu is None:
        mu = sequence.mean(dim=dim, keepdim=True)

    # Compute implied residuals under AR(1)
    x_prev = sequence[:, :-1] if dim == 1 else sequence[:-1]
    x_curr = sequence[:, 1:] if dim == 1 else sequence[1:]

    # Expected x_curr under AR(1)
    x_expected = mu + phi * (x_prev - mu)

    # Residuals
    residuals = x_curr - x_expected

    # Residuals should have zero mean
    mean_loss = residuals.mean() ** 2

    # Residuals should be uncorrelated (ACF at lag 1 should be ~0)
    if residuals.shape[1] > 1:
        residual_acf = differentiable_acf(residuals, lag=1, dim=1)
        acf_loss_val = (residual_acf ** 2).mean()
    else:
        acf_loss_val = torch.tensor(0.0, device=sequence.device)

    return mean_loss + acf_loss_val


def mean_reversion_regularizer(
    phi: torch.Tensor,
    target_phi: float = -0.35,
    strength: float = 1.0
) -> torch.Tensor:
    """
    Regularize AR(1) coefficient toward target mean-reversion strength.

    For vol surfaces, GT phi ≈ -0.35 (moderate mean reversion).

    Args:
        phi: Learned AR(1) coefficient
        target_phi: Target coefficient (default: -0.35 from GT analysis)
        strength: Regularization strength

    Returns:
        Regularization loss
    """
    return strength * F.mse_loss(phi, torch.tensor(target_phi, device=phi.device))


def combined_temporal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    spectral_weight: float = 0.1,
    acf_weight: float = 0.1,
    acf_lags: list = [1, 2, 3]
) -> tuple:
    """
    Combined temporal loss for ACF preservation.

    Args:
        pred: Predictions
        target: Targets
        spectral_weight: Weight for spectral loss
        acf_weight: Weight for ACF loss
        acf_lags: Lags for ACF matching

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    spec_loss = spectral_loss(pred, target)
    acf_loss_val = acf_loss(pred, target, lags=acf_lags)

    total = spectral_weight * spec_loss + acf_weight * acf_loss_val

    return total, {
        "spectral_loss": spec_loss.item(),
        "acf_loss": acf_loss_val.item(),
    }
