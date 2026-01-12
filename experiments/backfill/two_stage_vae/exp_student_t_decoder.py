"""
Phase 3: Student-t Decoder for Fat Tails

Problem: Model output has near-zero kurtosis vs GT kurtosis of 5-244.
         Gaussian output severely underestimates tail risk (5x underestimate at 3σ).

Solution:
1. Replace Gaussian NLL with Student-t NLL
2. Use fixed ν (degrees of freedom) computed from GT kurtosis
3. Per-grid ν values capture varying tail heaviness across the surface

Expected outcome:
- Kurtosis recovery: 0.6% → >50%
- 3σ event underestimate: 5x → <2x
- Fat tails properly modeled

Formula:
- For Student-t: excess_kurtosis = 6/(ν-4) for ν>4
- Solving: ν = 4 + 6/(kurtosis - 3)
- Clamp ν to [4.1, 30] for numerical stability

Usage:
    python experiments/backfill/two_stage_vae/exp_student_t_decoder.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TwoStageConfig, TWO_STAGE_CONFIG
from vae.cvae_two_stage import (
    TwoStageCtxEncoder,
    TwoStageMainEncoder,
)


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def compute_nu_from_kurtosis(gt_kurtosis: np.ndarray) -> np.ndarray:
    """
    Compute fixed ν for Student-t from GT kurtosis.

    For Student-t: kurtosis = 3 + 6/(ν-4) for ν>4
    Excess kurtosis = 6/(ν-4)
    Solving: ν = 4 + 6/excess_kurtosis

    Args:
        gt_kurtosis: Ground truth kurtosis (Fisher definition, normal=0)
                     Note: Our GT values are excess kurtosis (normal=0)

    Returns:
        nu: Degrees of freedom, clamped to [4.1, 100]
    """
    # GT kurtosis values from analysis are Fisher kurtosis (excess kurtosis, normal=0)
    # If kurtosis is 5.02, that's excess kurtosis of 5.02, total kurtosis of 8.02
    excess_kurtosis = np.maximum(gt_kurtosis, 0.1)  # Ensure positive

    # ν = 4 + 6/excess_kurtosis
    nu = 4.0 + 6.0 / excess_kurtosis

    # Clamp for numerical stability
    # Lower bound: 4.1 (very fat tails)
    # Upper bound: 100 (essentially Gaussian)
    nu = np.clip(nu, 4.1, 100.0)

    return nu


# GT kurtosis values from shape_diagnostics.json (Fisher/excess kurtosis)
GT_KURTOSIS = np.array([
    [5.53, 2.75, 2.21, 15.33, 19.66],    # Row 0 (short-term moneyness)
    [21.96, 3.24, 2.51, 84.76, 18.65],   # Row 1
    [65.47, 9.98, 5.02, 2.82, 12.75],    # Row 2 (ATM row)
    [147.78, 47.95, 25.18, 8.50, 143.95],# Row 3
    [75.01, 73.92, 69.44, 44.23, 244.48] # Row 4 (long-term)
])

# Compute fixed ν values
GT_NU = compute_nu_from_kurtosis(GT_KURTOSIS)
print(f"GT ν values (lower = fatter tails):")
print(GT_NU.round(2))


class StudentTDecoder(nn.Module):
    """
    Decoder with Student-t output distribution for fat tails.

    Key differences from Gaussian:
    - Student-t NLL instead of Gaussian NLL
    - Per-grid fixed ν from GT kurtosis
    - Full covariance (FF^T + D) like Phase 1
    - z-dependent covariance like Phase 2
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        self.rank = config.get("cov_rank", 4)

        # Fixed ν from GT kurtosis (not learned)
        self.register_buffer("nu", torch.tensor(GT_NU, dtype=torch.float32).view(25))

        # Mean decoder: z → mean (direct MLP from Phase 2)
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Z-dependent covariance (from Phase 2)
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Initialize log_diag to reasonable variance
        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with Student-t sampling.

        Student-t sampling: x = μ + σ * t_ν
        where t_ν = ε / sqrt(χ²_ν / ν) with ε ~ N(0,1), χ²_ν ~ χ²(ν)
        """
        B, T, _ = z.shape
        device = z.device

        # Mean prediction (z-dependent)
        z_flat = z.view(B * T, -1)
        mean = self.mean_net(z_flat)
        mean = mean.view(B, T, 5, 5)

        # Z-dependent covariance parameters
        z_pooled = z.mean(dim=1)  # (B, latent_dim)

        factor_flat = self.factor_net(z_pooled)
        factor = factor_flat.view(B, 25, self.rank)

        log_diag = self.log_diag_net(z_pooled)
        log_diag = torch.clamp(log_diag, min=-10, max=2)

        if not sample:
            return mean, mean, factor, log_diag

        # Student-t sampling
        # t_ν = ε * sqrt(ν / χ²_ν) where ε ~ N(0,1), χ²_ν ~ χ²(ν)

        # For correlated part: sample t distribution for each factor
        eps_rank = torch.randn(B, T, self.rank, device=device)

        # Use same chi-squared for entire sample (conservative)
        # Sample chi-squared via: sum of ν independent N(0,1)^2
        # Approximate with Gamma distribution for efficiency
        nu_expanded = self.nu.view(1, 1, 25)  # (1, 1, 25)

        # For simplicity, use single ν per grid point but apply after combining
        # First create Gaussian samples, then scale by Student-t factor
        correlated_gauss = torch.einsum('bir,btr->bti', factor, eps_rank)  # (B, T, 25)

        eps_diag = torch.randn(B, T, 25, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 25)
        independent_gauss = diag_std * eps_diag  # (B, T, 25)

        total_gauss = correlated_gauss + independent_gauss  # (B, T, 25)

        # Convert to Student-t by dividing by sqrt(chi2/nu)
        # chi2 ~ Gamma(nu/2, 0.5), so chi2/nu ~ Gamma(nu/2, nu/2)
        # Use per-grid nu
        chi2_samples = torch.zeros(B, T, 25, device=device)
        for i in range(25):
            nu_i = self.nu[i].item()
            # Gamma(alpha=nu/2, beta=nu/2) has mean=1, so dividing gives scale 1
            alpha = nu_i / 2
            beta = nu_i / 2
            gamma_samples = torch._standard_gamma(torch.full((B, T), alpha, device=device)) / beta
            chi2_samples[:, :, i] = gamma_samples

        # t = gaussian / sqrt(chi2) has Student-t distribution
        student_t_factor = 1.0 / torch.sqrt(chi2_samples + 1e-8)
        total_t = total_gauss * student_t_factor  # (B, T, 25)

        samples = mean + total_t.view(B, T, 5, 5)

        return mean, samples, factor, log_diag

    def compute_student_t_nll(self, pred_mean, target, factor, log_diag):
        """
        Student-t NLL with per-grid ν and full covariance.

        For multivariate Student-t with Σ = FF^T + D:
        log p(x|μ,Σ,ν) = log Γ((ν+d)/2) - log Γ(ν/2)
                         - d/2 log(νπ) - 0.5 log|Σ|
                         - (ν+d)/2 log(1 + (x-μ)^T Σ^-1 (x-μ) / ν)

        Since we have per-grid ν, we use factorized approach:
        Product of univariate Student-t along each dimension
        """
        B, T = pred_mean.shape[:2]
        device = pred_mean.device

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat  # (B, T, 25)

        D = torch.exp(log_diag)  # (B, 25)
        D = torch.clamp(D, min=1e-8)

        # Compute total variance per grid point
        # Σ_ii = D_i + Σ_k F_ik^2
        # factor is (B, 25, rank)
        factor_sq = (factor ** 2).sum(dim=-1)  # (B, 25)
        total_var = D + factor_sq  # (B, 25)

        # Per-grid univariate Student-t NLL
        # NLL_i = (ν_i+1)/2 * log(1 + z_i^2/ν_i) + 0.5*log(σ_i^2) + const
        # where z_i = (x_i - μ_i) / σ_i

        nu = self.nu.view(1, 1, 25)  # (1, 1, 25)
        sigma_sq = total_var.view(B, 1, 25)  # (B, 1, 25)

        z_sq = residual ** 2 / sigma_sq  # (B, T, 25)

        # Student-t NLL (ignoring normalizing constant which doesn't affect training)
        nll = 0.5 * (nu + 1) * torch.log(1 + z_sq / nu) + 0.5 * torch.log(sigma_sq)

        return nll.mean()


class CVAETwoStageStudentT(nn.Module):
    """CVAETwoStage with Student-t decoder for fat tails."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = StudentTDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with Student-t decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        mean, samples, factor, log_diag = self.decoder(ctx_emb, z, sample=True)

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with Student-t decoder."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        samples = []
        for _ in range(n_samples):
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z
            _, sample, _, _ = self.decoder(ctx_emb, z, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)


def evaluate_z_contribution(model, val_loader, config):
    """Evaluate z contribution to mean prediction."""
    device = config["device"]
    model.eval()

    oracle_mses = []
    zero_z_mses = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            ctx_emb = model.ctx_encoder({"surface": batch_data})
            z_mean, _, z = model.main_encoder({"surface": batch_data})

            # Mean with oracle z
            mean_oracle, _, _, _ = model.decoder(ctx_emb, z, sample=False)
            target = batch_data[:, 1:]
            pred_oracle = mean_oracle[:, :-1]
            oracle_mse = ((pred_oracle - target) ** 2).mean().item()
            oracle_mses.append(oracle_mse)

            # Mean with z=0
            z_zero = torch.zeros_like(z)
            mean_zero, _, _, _ = model.decoder(ctx_emb, z_zero, sample=False)
            pred_zero = mean_zero[:, :-1]
            zero_z_mse = ((pred_zero - target) ** 2).mean().item()
            zero_z_mses.append(zero_z_mse)

    oracle_mse = np.mean(oracle_mses)
    zero_z_mse = np.mean(zero_z_mses)
    z_contribution = (zero_z_mse - oracle_mse) / zero_z_mse * 100 if zero_z_mse > 0 else 0

    return {
        "oracle_mse": oracle_mse,
        "zero_z_mse": zero_z_mse,
        "z_contribution": z_contribution,
    }


def create_dataloader(log_returns, context_len, batch_size, shuffle=True):
    """Create dataloader for training."""
    sequences = []
    for i in range(len(log_returns) - context_len):
        seq = log_returns[i:i + context_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_student_t(model, train_loader, val_loader, config, epochs=100):
    """
    Two-phase training with Student-t NLL.

    Phase A: Train mean with MSE (like Phase 2)
    Phase B: Train variance with Student-t NLL (instead of Gaussian NLL)
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    phase_a_epochs = int(epochs * 0.6)
    phase_b_epochs = epochs - phase_a_epochs

    # ========================================
    # PHASE A: Pre-train mean with MSE
    # ========================================
    print(f"\n--- Phase A: Pre-train Mean with MSE ({phase_a_epochs} epochs) ---")

    for param in model.decoder.factor_net.parameters():
        param.requires_grad = False
    for param in model.decoder.log_diag_net.parameters():
        param.requires_grad = False

    optimizer_a = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, patience=10, factor=0.5)

    for epoch in range(phase_a_epochs):
        model.train()
        train_mses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, _, _ = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            mse_loss = ((pred - target) ** 2).mean()
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            loss = mse_loss + kl_weight * kl_loss

            optimizer_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_a.step()

            train_mses.append(mse_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_a_epochs}: Train MSE={np.mean(train_mses):.6f}")

    # Check z contribution after Phase A
    print("\n    Checking z contribution after Phase A...")
    z_results = evaluate_z_contribution(model, val_loader, config)
    print(f"    Z contribution: {z_results['z_contribution']:.1f}%")

    # ========================================
    # PHASE B: Train variance with Student-t NLL
    # ========================================
    print(f"\n--- Phase B: Train Variance with Student-t NLL ({phase_b_epochs} epochs) ---")

    for param in model.ctx_encoder.parameters():
        param.requires_grad = False
    for param in model.main_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.mean_net.parameters():
        param.requires_grad = False

    for param in model.decoder.factor_net.parameters():
        param.requires_grad = True
    for param in model.decoder.log_diag_net.parameters():
        param.requires_grad = True

    optimizer_b = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    for epoch in range(phase_b_epochs):
        model.train()
        train_nlls = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # Student-t NLL instead of Gaussian NLL
            nll_loss = model.decoder.compute_student_t_nll(pred, target, factor, log_diag)

            optimizer_b.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            train_nlls.append(nll_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_b_epochs}: Train NLL={np.mean(train_nlls):.4f}")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    return model


def evaluate_kurtosis(model, val_loader, config, n_samples=500):
    """
    Evaluate kurtosis of generated samples vs GT.

    Generate many samples, compute empirical kurtosis, compare to GT.
    """
    device = config["device"]
    model.eval()

    # Collect samples for each validation batch
    all_samples = []
    all_targets = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            # Generate samples
            samples = model.sample({"surface": batch_data}, n_samples=n_samples)
            # samples: (n_samples, B, T, 5, 5)

            # Get target (next step)
            target = batch_data[:, 1:]  # (B, T, 5, 5)

            # Take only last prediction for each sequence
            samples_last = samples[:, :, -1, :, :]  # (n_samples, B, 5, 5)
            target_last = target[:, -1, :, :]  # (B, 5, 5)

            all_samples.append(samples_last.cpu().numpy())
            all_targets.append(target_last.cpu().numpy())

    # Stack all
    all_samples = np.concatenate(all_samples, axis=1)  # (n_samples, N, 5, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 5, 5)

    # Compute per-grid kurtosis
    oracle_kurtosis = np.zeros((5, 5))
    gt_kurtosis = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            # Oracle kurtosis: across all samples and sequences
            samples_ij = all_samples[:, :, i, j].flatten()
            oracle_kurtosis[i, j] = kurtosis(samples_ij, fisher=True)

            # GT kurtosis: across sequences
            gt_kurtosis[i, j] = kurtosis(all_targets[:, i, j], fisher=True)

    # Kurtosis recovery
    recovery_ratio = np.abs(oracle_kurtosis) / np.abs(gt_kurtosis + 1e-8)
    mean_recovery = recovery_ratio.mean()

    return {
        "gt_kurtosis": gt_kurtosis,
        "oracle_kurtosis": oracle_kurtosis,
        "recovery_ratio": recovery_ratio,
        "mean_recovery": mean_recovery * 100,  # As percentage
        "atm_gt_kurtosis": gt_kurtosis[2, 2],
        "atm_oracle_kurtosis": oracle_kurtosis[2, 2],
        "atm_recovery": recovery_ratio[2, 2] * 100,
    }


def run_experiment(epochs=100):
    """Run Student-t decoder experiment."""
    print("=" * 70)
    print("Phase 3: Student-t Decoder for Fat Tails")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    log_returns, log_surfaces = to_log_returns(data["surface"])

    # Config
    config = TWO_STAGE_CONFIG.copy()
    config["latent_dim"] = 8
    config["decoder_mem_hidden"] = 64
    config["cov_rank"] = 4
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfig:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  cov_rank: {config['cov_rank']}")
    print(f"  device: {config['device']}")
    print(f"\nFixed ν values from GT kurtosis:")
    print(GT_NU.round(2))

    # Create dataloaders
    context_len = config.get("context_len", 20)
    n_train = int(len(log_returns) * 0.8)

    train_loader = create_dataloader(log_returns[:n_train], context_len, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], context_len, batch_size=32, shuffle=False)

    # Create model
    model = CVAETwoStageStudentT(config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Train
    train_student_t(model, train_loader, val_loader, config, epochs=epochs)

    # Evaluate kurtosis
    print("\n--- Evaluating Kurtosis Recovery ---")
    print("Generating samples (may take a minute)...")

    kurtosis_results = evaluate_kurtosis(model, val_loader, config, n_samples=500)

    print(f"\nGT Kurtosis (per grid):")
    print(kurtosis_results["gt_kurtosis"].round(2))
    print(f"\nOracle Kurtosis (per grid):")
    print(kurtosis_results["oracle_kurtosis"].round(2))
    print(f"\nKurtosis Recovery Ratio (per grid, higher is better):")
    print((kurtosis_results["recovery_ratio"] * 100).round(1))

    print(f"\n  ATM GT Kurtosis: {kurtosis_results['atm_gt_kurtosis']:.2f}")
    print(f"  ATM Oracle Kurtosis: {kurtosis_results['atm_oracle_kurtosis']:.2f}")
    print(f"  ATM Kurtosis Recovery: {kurtosis_results['atm_recovery']:.1f}%")
    print(f"  Mean Kurtosis Recovery: {kurtosis_results['mean_recovery']:.1f}%")

    # Save model
    output_dir = Path("models/backfill/two_stage/student_t")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "student_t_best.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "kurtosis_results": kurtosis_results,
        "gt_nu": GT_NU,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Success criteria
    print("\n" + "=" * 70)
    print("PHASE 3 VERIFICATION")
    print("=" * 70)
    print(f"  Mean Kurtosis Recovery: {kurtosis_results['mean_recovery']:.1f}%")
    print(f"  Target: >50%")
    print(f"  Status: {'PASS' if kurtosis_results['mean_recovery'] > 50 else 'FAIL'}")
    print()
    print(f"  ATM Kurtosis Recovery: {kurtosis_results['atm_recovery']:.1f}%")
    print(f"  Target: >50%")
    print(f"  Status: {'PASS' if kurtosis_results['atm_recovery'] > 50 else 'FAIL'}")

    return model, kurtosis_results


if __name__ == "__main__":
    model, kurtosis_results = run_experiment(epochs=150)
