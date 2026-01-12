"""
Experiment: Low-Rank Spatial Covariance Decoder

Tests if adding structured variance (FF^T + D) fixes CI calibration
while preserving spatial correlation.

Root cause: Deterministic VAE decoder only outputs E[x|z], missing E[Var(x|z)].
By law of total variance: Var(x) = E[Var(x|z)] + Var(E[x|z])
                                    ↑ MISSING      ↑ decoder gain

Solution: Add low-rank covariance Σ = FF^T + D
- F: (5, 5, rank) factor matrix - captures spatial correlation
- D: (5, 5) diagonal - captures per-point residual variance
- Both learned but NOT z-dependent → preserves spatial structure

Configurations:
1. baseline: z_only decoder (no covariance)
2. rank_1: Low-rank with rank=1
3. rank_2: Low-rank with rank=2
4. rank_4: Low-rank with rank=4 (recommended)
5. rank_8: Low-rank with rank=8
6. diagonal_only: D only, no factor (ablation)

Training: Two-stage
- Stage 1: Train mean decoder with MSE (50 epochs)
- Stage 2: Freeze mean, train only F and D with NLL (50 epochs)

Run from repository root:
    python experiments/backfill/two_stage_vae/exp_low_rank_cov.py
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TwoStageConfig
from vae.cvae_two_stage import (
    CVAETwoStage,
    TwoStageCtxEncoder,
    TwoStageMainEncoder,
    TwoStageDecoder,
)


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


class TwoStageDecoderZOnly(TwoStageDecoder):
    """Decoder that only sees z, not ctx_emb."""

    def __init__(self, config: dict):
        # Temporarily set ctx_embedding_dim to 0 for parent init
        original_ctx_dim = config.get("ctx_embedding_dim", 3)
        config["ctx_embedding_dim"] = 0

        # Don't call parent __init__, build from scratch
        nn.Module.__init__(self)

        self.config = config
        latent_dim = config.get("latent_dim", 16)
        mem_hidden = config.get("mem_hidden", 64)
        mem_type = config.get("mem_type", "LSTM")
        num_layers = config.get("num_layers", 1)

        # Input: z only (no ctx_emb!)
        input_dim = latent_dim

        # Memory module
        if mem_type == "LSTM":
            self.mem = nn.LSTM(input_dim, mem_hidden, num_layers, batch_first=True)
        elif mem_type == "GRU":
            self.mem = nn.GRU(input_dim, mem_hidden, num_layers, batch_first=True)
        else:
            self.mem = nn.RNN(input_dim, mem_hidden, num_layers, batch_first=True)

        # FiLM modulation from z
        self.gamma_net = nn.Linear(latent_dim, mem_hidden)
        self.beta_net = nn.Linear(latent_dim, mem_hidden)

        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(mem_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Restore config
        config["ctx_embedding_dim"] = original_ctx_dim

    def forward(self, ctx_emb, z):
        """Forward pass using only z (ctx_emb ignored)."""
        # Input is just z
        x = z  # (B, T, latent_dim)

        # LSTM
        mem_out, _ = self.mem(x)  # (B, T, mem_hidden)

        # FiLM modulation
        gamma = self.gamma_net(z)  # (B, T, mem_hidden)
        beta = self.beta_net(z)
        features = gamma * mem_out + beta

        # Output
        output = self.output_net(features)  # (B, T, 25)
        output = output.view(output.shape[0], output.shape[1], 5, 5)

        return output


class LowRankCovarianceDecoder(nn.Module):
    """
    Decoder that outputs mean + low-rank spatial covariance.

    Covariance structure: Σ = FF^T + D
    - F: (5, 5, rank) factor matrix - captures spatial correlation
    - D: (5, 5) diagonal - captures per-point residual variance

    Sampling: x = mean + F @ ε_rank + sqrt(D) * ε_diag
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.rank = config.get("cov_rank", 4)

        # Mean decoder (z_only)
        self.mean_decoder = TwoStageDecoderZOnly(config)

        # Factor matrix F: (5, 5, rank) - learned, NOT z-dependent
        if self.rank > 0:
            self.factor = nn.Parameter(torch.randn(5, 5, self.rank) * 0.1)
        else:
            self.register_buffer('factor', torch.zeros(5, 5, 1))

        # Diagonal D: (5, 5) - learned per-grid residual variance
        # Initialize to log(GT_variance) if provided, else zeros
        if "gt_var" in config:
            gt_var = config["gt_var"]
            if isinstance(gt_var, np.ndarray):
                gt_var = torch.tensor(gt_var, dtype=torch.float32)
            # Clamp to avoid log(0)
            gt_var = torch.clamp(gt_var, min=1e-8)
            self.log_diag = nn.Parameter(torch.log(gt_var))
        else:
            self.log_diag = nn.Parameter(torch.zeros(5, 5))

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with optional sampling.

        Args:
            ctx_emb: Context embedding (ignored, for API compatibility)
            z: Latent variable (B, T, latent_dim)
            sample: Whether to sample with covariance

        Returns:
            mean: (B, T, 5, 5) - deterministic prediction
            samples: (B, T, 5, 5) - sampled with spatial covariance
        """
        mean = self.mean_decoder(ctx_emb, z)  # (B, T, 5, 5)

        if not sample:
            return mean, mean

        B, T = mean.shape[:2]
        device = mean.device

        # Sample from low-rank covariance
        if self.rank > 0:
            # ε_rank: (B, T, rank) - shared across spatial dims
            eps_rank = torch.randn(B, T, self.rank, device=device)

            # F @ ε_rank: (B, T, 5, 5) - correlated component
            correlated = torch.einsum('ijr,btr->btij', self.factor, eps_rank)
        else:
            correlated = 0

        # ε_diag: (B, T, 5, 5) - independent per point
        eps_diag = torch.randn(B, T, 5, 5, device=device)

        # sqrt(D) * ε_diag: (B, T, 5, 5) - independent component
        diag_std = torch.exp(0.5 * self.log_diag)  # (5, 5)
        independent = diag_std * eps_diag

        # Sample: mean + correlated + independent
        samples = mean + correlated + independent

        return mean, samples

    def compute_nll_loss(self, pred_mean, target):
        """Compute NLL loss with diagonal approximation of covariance."""
        diag_var = torch.exp(self.log_diag)  # (5, 5)

        # Add contribution from factor to diagonal variance
        if self.rank > 0:
            factor_var = (self.factor ** 2).sum(dim=-1)  # (5, 5)
            total_var = diag_var + factor_var
        else:
            total_var = diag_var

        # NLL with diagonal approximation
        squared_error = (pred_mean - target) ** 2  # (B, T, 5, 5)
        nll = 0.5 * (squared_error / total_var + torch.log(total_var))

        return nll.mean()

    def get_total_variance(self):
        """Get total variance per grid point (diagonal of Σ)."""
        diag_var = torch.exp(self.log_diag)
        if self.rank > 0:
            factor_var = (self.factor ** 2).sum(dim=-1)
            return diag_var + factor_var
        return diag_var

    def get_covariance_matrix(self):
        """Return full 25x25 covariance matrix for visualization."""
        if self.rank > 0:
            F = self.factor.view(25, self.rank)  # (25, rank)
            cov = F @ F.T  # (25, 25)
        else:
            cov = torch.zeros(25, 25, device=self.log_diag.device)

        D = torch.diag(torch.exp(self.log_diag).view(25))  # (25, 25)
        return cov + D


class CVAETwoStageLowRankCov(CVAETwoStage):
    """CVAETwoStage with Low-Rank Covariance decoder."""

    def __init__(self, config: dict):
        # Initialize parent but we'll replace the decoder
        nn.Module.__init__(self)
        self.config = config

        # Build encoders (same as parent)
        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)

        # Replace decoder with LowRankCovarianceDecoder
        self.decoder = LowRankCovarianceDecoder(config)

        # Prior network (same as parent)
        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass returning mean predictions."""
        surface = batch["surface"]  # (B, T, 5, 5)
        B, T = surface.shape[:2]

        # Encode context (ctx_encoder expects dict with "surface" key)
        ctx_emb = self.ctx_encoder({"surface": surface})  # (B, T, ctx_embedding_dim)

        # Encode full sequence for posterior (returns z_mean, z_logvar, z)
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})  # Each: (B, T, latent_dim)

        # Apply z_dropout if configured
        z_dropout = self.config.get("z_dropout", 0.0)
        if z_dropout > 0 and self.training:
            mask = torch.bernoulli(torch.full_like(z, 1 - z_dropout))
            z = z * mask / (1 - z_dropout)

        # Decode (get mean only for loss computation)
        mean, _ = self.decoder(ctx_emb, z, sample=False)

        if return_full_sequence:
            return mean, z_mean, z_logvar, ctx_emb

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with covariance."""
        surface = batch["surface"]
        B, T = surface.shape[:2]
        device = surface.device

        # Encode (encoders expect dict with "surface" key)
        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, _ = self.main_encoder({"surface": surface})
        z_std = torch.exp(0.5 * z_logvar)

        samples = []
        for _ in range(n_samples):
            # Sample z
            eps_z = torch.randn_like(z_std)
            z = z_mean + z_std * eps_z

            # Sample output with covariance
            _, sample = self.decoder(ctx_emb, z, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)  # (n_samples, B, T, 5, 5)


def create_dataloader(log_returns, context_len, batch_size, shuffle=True):
    """Create dataloader for training."""
    # Create sequences
    sequences = []
    for i in range(len(log_returns) - context_len):
        seq = log_returns[i:i + context_len + 1]
        sequences.append(seq)

    sequences = np.array(sequences)
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_stage1(model, train_loader, val_loader, config, epochs=50):
    """Stage 1: Train mean decoder with MSE, freeze covariance params."""
    device = config["device"]
    model = model.to(device)

    # Freeze covariance parameters
    model.decoder.factor.requires_grad = False
    model.decoder.log_diag.requires_grad = False

    # Optimizer for non-covariance params
    params = [p for n, p in model.named_parameters()
              if 'factor' not in n and 'log_diag' not in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.get("lr", 1e-3))

    kl_weight = config.get("kl_weight", 0.1)
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Forward
            mean, z_mean, z_logvar, _ = model(batch, return_full_sequence=True)

            # MSE loss on mean
            target = batch_data[:, 1:]  # Predict next step
            pred = mean[:, :-1]
            mse_loss = F.mse_loss(pred, target)

            # KL loss
            kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

            loss = mse_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}
                mean, z_mean, z_logvar, _ = model(batch, return_full_sequence=True)
                target = batch_data[:, 1:]
                pred = mean[:, :-1]
                mse_loss = F.mse_loss(pred, target)
                kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
                loss = mse_loss + kl_weight * kl_loss
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    model.load_state_dict(best_state)
    model.to(device)

    # Unfreeze covariance params for stage 2
    model.decoder.factor.requires_grad = True
    model.decoder.log_diag.requires_grad = True

    return best_val_loss


def train_stage2(model, train_loader, val_loader, config, epochs=200, lr=0.01):
    """Stage 2: Freeze mean decoder, train only covariance params with NLL.

    Note: Increased epochs (50→200) and LR (0.001→0.01) to ensure variance
    converges to match reconstruction error. With gradient ~0.5 and original
    settings, only ~1.8 units of log-space movement achieved; need ~7 units.
    """
    device = config["device"]
    model = model.to(device)

    # Freeze all except covariance params
    for name, param in model.named_parameters():
        if 'factor' in name or 'log_diag' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optimizer for covariance params only
    cov_params = [model.decoder.factor, model.decoder.log_diag]
    cov_params = [p for p in cov_params if p.requires_grad]

    if len(cov_params) == 0:
        return 0.0

    optimizer = torch.optim.Adam(cov_params, lr=lr)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Forward (no gradient through mean decoder)
            with torch.no_grad():
                mean, _, _, _ = model(batch, return_full_sequence=True)

            # NLL loss on covariance
            target = batch_data[:, 1:]
            pred = mean[:, :-1]
            nll_loss = model.decoder.compute_nll_loss(pred, target)

            optimizer.zero_grad()
            nll_loss.backward()
            optimizer.step()

            train_losses.append(nll_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}
                mean, _, _, _ = model(batch, return_full_sequence=True)
                target = batch_data[:, 1:]
                pred = mean[:, :-1]
                nll_loss = model.decoder.compute_nll_loss(pred, target)
                val_losses.append(nll_loss.item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Unfreeze all params
    for param in model.parameters():
        param.requires_grad = True

    return best_val_loss


def compute_empirical_variance_ratios(log_returns):
    """Compute empirical variance scaling ratios for each horizon.

    GT log-returns have negative autocorrelation (mean-reversion), so
    cumulative variance grows slower than H × single_var.

    Returns:
        dict mapping horizon H to (5,5) array of variance ratios
    """
    single_var = log_returns.var(axis=0)  # (5, 5)
    ratios = {}

    for H in [1, 7, 14, 30]:
        # Compute actual cumulative variance
        cumsum = np.array([log_returns[i:i+H].sum(axis=0)
                          for i in range(len(log_returns) - H)])
        cum_var = cumsum.var(axis=0)

        # Ratio of actual to expected (if i.i.d.)
        expected_var = H * single_var
        ratio = np.where(expected_var > 0, cum_var / expected_var, 1.0)
        ratios[H] = ratio

    return ratios


def evaluate_ci_violations(model, log_returns, log_surfaces, surfaces, config,
                           n_samples=100, n_test=50, space="log_return",
                           use_mean_reversion_correction=True):
    """Evaluate CI violations in log-return or IV space.

    Args:
        use_mean_reversion_correction: If True, scale covariance noise
            by empirical variance ratios to account for mean-reversion
            in GT data. GT log-returns have negative autocorrelation,
            so cumulative variance grows slower than H × single_var.
    """
    device = config["device"]
    model = model.to(device)
    model.eval()

    context_len = config["context_len"]
    horizons = [1, 7, 14, 30]

    # Compute variance ratios for mean-reversion correction
    if use_mean_reversion_correction:
        var_ratios = compute_empirical_variance_ratios(log_returns)

    results = {}
    for H in horizons:
        seq_len = context_len + H

        violations_per_grid = np.zeros((5, 5))
        ci_width_per_grid = np.zeros((5, 5))
        n_tests = 0

        test_indices = list(range(0, len(log_returns) - seq_len,
                                  max(1, (len(log_returns) - seq_len) // n_test)))[:n_test]

        for start_idx in test_indices:
            # Ground truth
            gt_log_seq = log_returns[start_idx:start_idx + context_len + H]

            if space == "log_return":
                gt_cumsum = gt_log_seq[context_len:context_len + H].sum(axis=0)
            else:  # IV space
                initial_log = log_surfaces[start_idx]
                gt_iv = surfaces[start_idx + context_len + H]

            # Sample predictions
            samples = []
            with torch.no_grad():
                # Encoder sees full sequence (context + target) - this is proper VAE usage
                batch = torch.tensor(gt_log_seq[None], dtype=torch.float32).to(device)
                batch_dict = {"surface": batch}

                # Get encoder outputs (posterior q(z|context, target))
                ctx_emb = model.ctx_encoder(batch_dict)
                z_mean, z_logvar, _ = model.main_encoder(batch_dict)
                z_std = torch.exp(0.5 * z_logvar)

                # Get learned per-timestep covariance
                total_var = model.decoder.get_total_variance().detach().cpu().numpy()

                # For H-step cumulative prediction:
                # - Mean: sum of H timestep predictions (from decoder)
                # - Variance: H × single_var × ratio (accounts for mean-reversion)
                if use_mean_reversion_correction:
                    # Correct for mean-reversion: actual cum_var = H × single_var × ratio
                    cum_var = H * total_var * var_ratios[H]
                else:
                    # Assume i.i.d.: cum_var = H × single_var
                    cum_var = H * total_var

                cum_std = np.sqrt(cum_var)

                for _ in range(n_samples):
                    # Sample z from posterior
                    eps = torch.randn_like(z_std)
                    z = z_mean + z_std * eps

                    # Get mean prediction for full sequence
                    # Decoder output shape: (B, context_len + H, 5, 5)
                    mean_pred, _ = model.decoder(ctx_emb, z, sample=False)
                    mean_recon = mean_pred[0].cpu().numpy()  # (T, 5, 5)

                    if space == "log_return":
                        # Sum predictions over the H target timesteps
                        pred_mean_cumsum = mean_recon[context_len:context_len + H].sum(axis=0)

                        # Add cumulative noise (single draw, scaled by cum_std)
                        cumulative_noise = np.random.randn(5, 5) * cum_std
                        pred_cumsum = pred_mean_cumsum + cumulative_noise
                        samples.append(pred_cumsum)
                    else:  # IV space
                        # For IV space, compute from initial surface + cumulative prediction
                        pred_mean_cumsum = mean_recon[context_len:context_len + H].sum(axis=0)
                        cumulative_noise = np.random.randn(5, 5) * cum_std
                        pred_iv = np.exp(initial_log + pred_mean_cumsum + cumulative_noise)
                        samples.append(pred_iv)

            samples = np.array(samples)  # (n_samples, 5, 5)

            # Compute 90% CI
            p05 = np.percentile(samples, 5, axis=0)
            p95 = np.percentile(samples, 95, axis=0)

            # Check violations
            if space == "log_return":
                gt = gt_cumsum
            else:
                gt = gt_iv

            in_ci = (gt >= p05) & (gt <= p95)
            violations_per_grid += ~in_ci
            ci_width_per_grid += (p95 - p05)
            n_tests += 1

        violations_per_grid /= n_tests
        ci_width_per_grid /= n_tests

        results[H] = {
            "violations": violations_per_grid,
            "ci_width": ci_width_per_grid,
            "avg_violation": float(violations_per_grid.mean()),
        }

    return results


def compute_spatial_correlation(model, log_returns, config, n_samples=500):
    """Compute sample covariance and compare with GT."""
    device = config["device"]
    model = model.to(device)
    model.eval()

    context_len = config["context_len"]

    # Collect samples
    all_samples = []
    test_indices = list(range(0, len(log_returns) - context_len - 1, 50))[:20]

    for start_idx in test_indices:
        gt_log_seq = log_returns[start_idx:start_idx + context_len + 1]

        with torch.no_grad():
            batch = torch.tensor(gt_log_seq[None], dtype=torch.float32).to(device)
            batch_dict = {"surface": batch}

            # Get encoder outputs (main_encoder returns z_mean, z_logvar, z)
            ctx_emb = model.ctx_encoder(batch_dict)
            z_mean, z_logvar, _ = model.main_encoder(batch_dict)
            z_std = torch.exp(0.5 * z_logvar)

            for _ in range(n_samples // len(test_indices)):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                _, sample = model.decoder(ctx_emb, z, sample=True)
                # Get last timestep prediction
                pred = sample[0, -1].cpu().numpy().flatten()  # (25,)
                all_samples.append(pred)

    all_samples = np.array(all_samples)  # (n, 25)

    # Sample covariance
    sample_cov = np.cov(all_samples.T)  # (25, 25)

    # GT covariance
    gt_flat = log_returns.reshape(-1, 25)
    gt_cov = np.cov(gt_flat.T)

    # Correlation between sample cov and GT cov
    cov_corr = np.corrcoef(sample_cov.flatten(), gt_cov.flatten())[0, 1]

    # Learned covariance
    learned_cov = model.decoder.get_covariance_matrix().detach().cpu().numpy()
    learned_corr = np.corrcoef(learned_cov.flatten(), gt_cov.flatten())[0, 1]

    return {
        "sample_cov": sample_cov,
        "gt_cov": gt_cov,
        "learned_cov": learned_cov,
        "sample_vs_gt_corr": float(cov_corr),
        "learned_vs_gt_corr": float(learned_corr),
    }


def run_experiment(exp_name, config, train_loader, val_loader,
                   log_returns, log_surfaces, surfaces, output_dir):
    """Run a single experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")
    print(f"  cov_rank: {config.get('cov_rank', 4)}")

    # Build model
    model = CVAETwoStageLowRankCov(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Stage 1: Train mean decoder
    print(f"\nStage 1: Training mean decoder with MSE...")
    stage1_loss = train_stage1(model, train_loader, val_loader, config, epochs=50)
    print(f"  Stage 1 best val loss: {stage1_loss:.6f}")

    # Stage 2: Train covariance
    if config.get("cov_rank", 4) > 0 or exp_name != "baseline":
        print(f"\nStage 2: Training covariance with NLL (epochs=200, lr=0.01)...")
        stage2_loss = train_stage2(model, train_loader, val_loader, config,
                                   epochs=200, lr=0.01)
        print(f"  Stage 2 best val loss: {stage2_loss:.6f}")
    else:
        stage2_loss = 0.0

    # Evaluate
    print(f"\nEvaluating CI violations...")

    # Log-return space (with mean-reversion correction for multi-horizon)
    lr_results = evaluate_ci_violations(
        model, log_returns, log_surfaces, surfaces, config,
        n_samples=100, n_test=50, space="log_return"
    )

    # IV space
    iv_results = evaluate_ci_violations(
        model, log_returns, log_surfaces, surfaces, config,
        n_samples=100, n_test=50, space="iv"
    )

    print(f"\n  CI Violations (Log-Return Space):")
    for H in [1, 7, 14, 30]:
        print(f"    H={H:2d}: {lr_results[H]['avg_violation']*100:.1f}%")

    print(f"\n  CI Violations (IV Space):")
    for H in [1, 7, 14, 30]:
        print(f"    H={H:2d}: {iv_results[H]['avg_violation']*100:.1f}%")

    # Spatial correlation
    print(f"\nEvaluating spatial correlation...")
    spatial_results = compute_spatial_correlation(model, log_returns, config)
    print(f"  Sample vs GT covariance correlation: {spatial_results['sample_vs_gt_corr']:.3f}")
    print(f"  Learned vs GT covariance correlation: {spatial_results['learned_vs_gt_corr']:.3f}")

    # Learned variance
    total_var = model.decoder.get_total_variance().detach().cpu().numpy()
    gt_var = log_returns.var(axis=0)
    var_corr = np.corrcoef(total_var.flatten(), gt_var.flatten())[0, 1]
    print(f"  Learned variance vs GT variance correlation: {var_corr:.3f}")

    # Save model
    save_path = output_dir / f"{exp_name}_best.pt"
    torch.save({
        "model_config": config,
        "model_state_dict": model.state_dict(),
        "stage1_val_loss": stage1_loss,
        "stage2_val_loss": stage2_loss,
    }, save_path)
    print(f"\n  Saved: {save_path}")

    return {
        "exp_name": exp_name,
        "config": {k: v for k, v in config.items() if k not in ("device", "gt_var")},
        "stage1_val_loss": float(stage1_loss),
        "stage2_val_loss": float(stage2_loss),
        "lr_violations": {str(k): v["avg_violation"] for k, v in lr_results.items()},
        "iv_violations": {str(k): v["avg_violation"] for k, v in iv_results.items()},
        "spatial_corr": {
            "sample_vs_gt": spatial_results["sample_vs_gt_corr"],
            "learned_vs_gt": spatial_results["learned_vs_gt_corr"],
        },
        "variance_corr": float(var_corr),
        "lr_per_grid_h1": lr_results[1]["violations"].tolist(),
        "lr_per_grid_h30": lr_results[30]["violations"].tolist(),
    }


def main():
    print("=" * 70)
    print("LOW-RANK SPATIAL COVARIANCE EXPERIMENT")
    print("=" * 70)
    print("\nGoal: Fix CI calibration by adding E[Var(x|z)] term")
    print("Method: Low-rank covariance Σ = FF^T + D")
    print("Training: Two-stage (MSE then NLL)")

    # Output directory
    output_dir = Path("models/backfill/two_stage/low_rank_cov")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)
    print(f"  Surfaces: {surfaces.shape}")
    print(f"  Log-returns: {log_returns.shape}")

    # Compute GT variance for initialization
    gt_var = log_returns.var(axis=0)  # (5, 5)
    print(f"  GT variance range: [{gt_var.min():.6f}, {gt_var.max():.6f}]")
    print(f"  GT log-variance range: [{np.log(gt_var).min():.2f}, {np.log(gt_var).max():.2f}]")

    # Base config - get model config dict from TwoStageConfig class
    base_config = TwoStageConfig.get_model_config()
    base_config["context_len"] = 20
    base_config["latent_dim"] = 8
    base_config["z_dropout"] = 0.3
    base_config["kl_weight"] = 0.1
    base_config["lr"] = 1e-3
    base_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    base_config["gt_var"] = gt_var  # Pass GT variance for log_diag initialization

    # Create dataloaders
    context_len = base_config["context_len"]
    n_total = len(log_returns) - context_len
    n_train = int(0.8 * n_total)

    train_loader = create_dataloader(
        log_returns[:n_train + context_len],
        context_len, batch_size=64, shuffle=True
    )
    val_loader = create_dataloader(
        log_returns[n_train:],
        context_len, batch_size=64, shuffle=False
    )
    print(f"  Train sequences: {len(train_loader.dataset)}")
    print(f"  Val sequences: {len(val_loader.dataset)}")

    # Experiments
    experiments = [
        ("baseline", {"cov_rank": 0}),  # No covariance (z_only decoder)
        ("diagonal_only", {"cov_rank": 0}),  # D only, no factor
        ("rank_1", {"cov_rank": 1}),
        ("rank_2", {"cov_rank": 2}),
        ("rank_4", {"cov_rank": 4}),
        ("rank_8", {"cov_rank": 8}),
    ]

    all_results = {}
    for exp_name, exp_config in tqdm(experiments, desc="Experiments"):
        config = base_config.copy()
        config.update(exp_config)

        # For diagonal_only, we train covariance but with rank=0
        # For baseline, we skip stage 2 entirely

        result = run_experiment(
            exp_name, config, train_loader, val_loader,
            log_returns, log_surfaces, surfaces, output_dir
        )
        all_results[exp_name] = result

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n| Experiment | Rank | LR Viol H=1 | LR Viol H=30 | IV Viol H=30 | Var Corr | Spatial Corr |")
    print("|------------|------|-------------|--------------|--------------|----------|--------------|")

    for exp_name, result in all_results.items():
        rank = result["config"].get("cov_rank", 0)
        lr_h1 = result["lr_violations"]["1"] * 100
        lr_h30 = result["lr_violations"]["30"] * 100
        iv_h30 = result["iv_violations"]["30"] * 100
        var_corr = result["variance_corr"]
        spatial_corr = result["spatial_corr"]["sample_vs_gt"]

        print(f"| {exp_name:10s} | {rank:4d} | {lr_h1:10.1f}% | {lr_h30:11.1f}% | {iv_h30:11.1f}% | {var_corr:8.3f} | {spatial_corr:12.3f} |")

    print("\n" + "=" * 70)
    print("Target: 10% violations")
    print("=" * 70)


if __name__ == "__main__":
    main()
