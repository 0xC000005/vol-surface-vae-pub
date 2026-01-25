"""
Fréchet Surface Distance (FSD) for evaluating volatility surface generation.

Analogous to FVD (Fréchet Video Distance) but for IV surfaces.

FSD measures whether generated surface sequences are statistically similar
to real sequences - complementing CI coverage which only measures calibration.

Usage:
    from metrics import FrechetSurfaceDistance, compute_fsd

    # Using class interface
    fsd_calc = FrechetSurfaceDistance(model)
    fsd_encoder = fsd_calc.compute(real_surfaces, gen_surfaces, method='encoder')
    fsd_domain = fsd_calc.compute(real_surfaces, gen_surfaces, method='domain')

    # Using functional interface
    features_real = extract_domain_features(real_surfaces)
    features_gen = extract_domain_features(gen_surfaces)
    fsd = compute_fsd(features_real, features_gen)
"""

from typing import Optional, Literal, Union
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg


def extract_encoder_features(
    model: nn.Module,
    surfaces: torch.Tensor,
) -> torch.Tensor:
    """
    Extract features using the trained DDPM encoder.

    Args:
        model: ConditionalDDPM model with history_encoder
        surfaces: (B, T, H, W) volatility surface sequences

    Returns:
        features: (B, condition_dim) learned representations
    """
    with torch.no_grad():
        # The history encoder expects (B, T, H, W) and returns (B, condition_dim)
        features = model.denoiser.history_encoder(surfaces)
    return features


def extract_domain_features(
    surfaces: torch.Tensor,
    flatten: bool = True,
) -> torch.Tensor:
    """
    Extract domain-specific (hand-crafted) features from IV surfaces.

    Features extracted:
    1. Level: Mean IV per day (ATM proxy)
    2. Skew: OTM put - OTM call (per tenor)
    3. Convexity: Butterfly spread (smile curvature)
    4. Term slope: Short tenor - Long tenor
    5. Daily changes: First differences

    Args:
        surfaces: (B, T, H, W) volatility surfaces
                  H=5 (tenors: 1M, 2M, 3M, 6M, 1Y)
                  W=5 (strikes: 0.9, 0.95, 1.0, 1.05, 1.1)
        flatten: If True, flatten to (B, D). If False, return dict.

    Returns:
        features: (B, D) flattened features or dict of feature tensors
    """
    B, T, H, W = surfaces.shape

    features = {}

    # 1. Level: Mean IV per day (proxy for ATM volatility)
    # Shape: (B, T)
    features['level'] = surfaces.mean(dim=(-1, -2))

    # 2. Skew: OTM Put (strike=0.9) - OTM Call (strike=1.1) for each tenor
    # Negative skew = puts more expensive (typical for equities)
    # Shape: (B, T, H)
    features['skew'] = surfaces[:, :, :, 0] - surfaces[:, :, :, 4]

    # 3. Convexity (Butterfly): IV(put) + IV(call) - 2*IV(ATM)
    # Positive = smile, measures curvature
    # Shape: (B, T, H)
    features['convexity'] = (
        surfaces[:, :, :, 0] + surfaces[:, :, :, 4] - 2 * surfaces[:, :, :, 2]
    )

    # 4. Term slope: Short tenor (1M) - Long tenor (1Y)
    # Positive = inverted term structure (short > long)
    # Shape: (B, T, W)
    features['term_slope'] = surfaces[:, :, 0, :] - surfaces[:, :, 4, :]

    # 5. Daily changes: First differences
    # Captures dynamics / volatility of volatility
    # Shape: (B, T-1, H, W)
    if T > 1:
        features['daily_changes'] = surfaces[:, 1:, :, :] - surfaces[:, :-1, :, :]
    else:
        features['daily_changes'] = torch.zeros(B, 0, H, W, device=surfaces.device)

    # 6. ATM term structure: Just the ATM strike across tenors
    # Shape: (B, T, H)
    features['atm_term'] = surfaces[:, :, :, 2]

    if not flatten:
        return features

    # Flatten all features to (B, D)
    flat_features = []

    # Level: (B, T) -> (B, T)
    flat_features.append(features['level'])

    # Skew: (B, T, H) -> (B, T*H)
    flat_features.append(features['skew'].reshape(B, -1))

    # Convexity: (B, T, H) -> (B, T*H)
    flat_features.append(features['convexity'].reshape(B, -1))

    # Term slope: (B, T, W) -> (B, T*W)
    flat_features.append(features['term_slope'].reshape(B, -1))

    # Daily changes: (B, T-1, H, W) -> (B, (T-1)*H*W)
    flat_features.append(features['daily_changes'].reshape(B, -1))

    # ATM term: (B, T, H) -> (B, T*H)
    flat_features.append(features['atm_term'].reshape(B, -1))

    # Concatenate all features
    all_features = torch.cat(flat_features, dim=1)

    return all_features


def compute_fsd(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Surface Distance between real and generated features.

    FSD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r × Σ_g))

    Args:
        real_features: (N_real, D) features from real sequences
        gen_features: (N_gen, D) features from generated sequences
        eps: Small constant for numerical stability

    Returns:
        fsd: Fréchet distance (lower is better)
    """
    # Convert to numpy for scipy operations
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(gen_features, torch.Tensor):
        gen_features = gen_features.cpu().numpy()

    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # Handle 1D case (single feature)
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
        sigma_gen = np.array([[sigma_gen]])

    # Add small epsilon to diagonal for numerical stability
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_gen += eps * np.eye(sigma_gen.shape[0])

    # Mean difference term
    diff = mu_real - mu_gen
    mean_term = np.dot(diff, diff)

    # Covariance term: Tr(Σ_r + Σ_g - 2√(Σ_r × Σ_g))
    # Compute matrix square root of product
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    # Handle numerical issues (imaginary parts from sqrt)
    if np.iscomplexobj(covmean):
        # Take real part if imaginary component is small
        if np.max(np.abs(covmean.imag)) < 1e-3:
            covmean = covmean.real
        else:
            # Fall back to simpler approximation
            covmean = linalg.sqrtm(sigma_real) @ linalg.sqrtm(sigma_gen)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

    cov_term = np.trace(sigma_real + sigma_gen - 2 * covmean)

    # Handle numerical issues (negative trace from sqrt instability)
    if cov_term < 0:
        cov_term = 0

    fsd = mean_term + cov_term

    return float(fsd)


class FrechetSurfaceDistance:
    """
    Class for computing Fréchet Surface Distance with a trained model.

    Supports both encoder-based and domain-specific feature extraction.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        """
        Initialize FSD calculator.

        Args:
            model: ConditionalDDPM model (required for encoder features)
        """
        self.model = model

    def compute(
        self,
        real_surfaces: torch.Tensor,
        gen_surfaces: torch.Tensor,
        method: Literal['encoder', 'domain', 'both'] = 'both',
    ) -> Union[float, dict]:
        """
        Compute FSD between real and generated surface sequences.

        Args:
            real_surfaces: (N_real, T, H, W) real IV surface sequences
            gen_surfaces: (N_gen, T, H, W) generated IV surface sequences
            method: 'encoder' (learned), 'domain' (hand-crafted), or 'both'

        Returns:
            If method='both': dict with 'fsd_encoder' and 'fsd_domain'
            Otherwise: float FSD value
        """
        results = {}

        if method in ['encoder', 'both']:
            if self.model is None:
                raise ValueError("Model required for encoder features")

            real_enc = extract_encoder_features(self.model, real_surfaces)
            gen_enc = extract_encoder_features(self.model, gen_surfaces)
            results['fsd_encoder'] = compute_fsd(real_enc, gen_enc)

        if method in ['domain', 'both']:
            real_dom = extract_domain_features(real_surfaces)
            gen_dom = extract_domain_features(gen_surfaces)
            results['fsd_domain'] = compute_fsd(real_dom, gen_dom)

        if method == 'both':
            return results
        elif method == 'encoder':
            return results['fsd_encoder']
        else:
            return results['fsd_domain']

    def compute_per_sample(
        self,
        real_surfaces: torch.Tensor,
        gen_samples: torch.Tensor,
        method: Literal['encoder', 'domain'] = 'domain',
    ) -> torch.Tensor:
        """
        Compute FSD contribution per generated sample (for analysis).

        Args:
            real_surfaces: (N_real, T, H, W) real sequences
            gen_samples: (B, N_samples, T, H, W) generated samples per condition

        Returns:
            per_sample_dist: (B, N_samples) distance of each sample from real distribution
        """
        B, N_samples, T, H, W = gen_samples.shape

        # Get real features and statistics
        if method == 'encoder':
            real_feat = extract_encoder_features(self.model, real_surfaces)
        else:
            real_feat = extract_domain_features(real_surfaces)

        mu_real = real_feat.mean(dim=0)

        # Compute distance for each sample
        distances = torch.zeros(B, N_samples)

        for b in range(B):
            for s in range(N_samples):
                sample = gen_samples[b, s:s+1]  # (1, T, H, W)

                if method == 'encoder':
                    feat = extract_encoder_features(self.model, sample)
                else:
                    feat = extract_domain_features(sample)

                # Simple L2 distance from real mean (approximation)
                distances[b, s] = ((feat - mu_real) ** 2).sum().sqrt()

        return distances


def compute_fsd_with_ci(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    n_bootstrap: int = 100,
) -> dict:
    """
    Compute FSD with bootstrap confidence intervals.

    Args:
        real_features: (N_real, D) real features
        gen_features: (N_gen, D) generated features
        n_bootstrap: Number of bootstrap samples

    Returns:
        dict with 'fsd', 'ci_lower', 'ci_upper'
    """
    fsd_main = compute_fsd(real_features, gen_features)

    # Bootstrap for confidence intervals
    fsd_bootstrap = []
    n_real = len(real_features)
    n_gen = len(gen_features)

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx_real = np.random.choice(n_real, n_real, replace=True)
        idx_gen = np.random.choice(n_gen, n_gen, replace=True)

        fsd_b = compute_fsd(real_features[idx_real], gen_features[idx_gen])
        fsd_bootstrap.append(fsd_b)

    fsd_bootstrap = np.array(fsd_bootstrap)

    return {
        'fsd': fsd_main,
        'ci_lower': np.percentile(fsd_bootstrap, 2.5),
        'ci_upper': np.percentile(fsd_bootstrap, 97.5),
        'std': np.std(fsd_bootstrap),
    }
