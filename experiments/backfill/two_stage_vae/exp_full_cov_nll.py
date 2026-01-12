"""
Phase 1: Full Covariance NLL for Learning Spatial Correlations

Problem: Diagonal NLL has no gradient signal for off-diagonal correlations.
         Factor F only contributes 5.9% of variance, correlations not learned.

Solution: Replace diagonal NLL with full covariance NLL using Woodbury identity.
         Loss = 0.5 * [log|Σ| + (x-μ)^T Σ^-1 (x-μ)]

Expected outcome:
- Factor F learns to capture spatial correlations
- Mean abs correlation: 0.04 → >0.15
- F contribution: 5.9% → >30%

Usage:
    python experiments/backfill/two_stage_vae/exp_full_cov_nll.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.two_stage_config import TwoStageConfig, TWO_STAGE_CONFIG
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
        nn.Module.__init__(self)

        self.config = config
        latent_dim = config.get("latent_dim", 16)
        mem_hidden = config.get("decoder_mem_hidden", 64)
        mem_type = config.get("mem_type", "LSTM")
        num_layers = config.get("decoder_mem_layers", 1)

        # Input: z only (no ctx_emb!)
        input_dim = latent_dim

        # Memory module
        if mem_type.upper() == "LSTM":
            self.mem = nn.LSTM(input_dim, mem_hidden, num_layers, batch_first=True)
        elif mem_type.upper() == "GRU":
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

    def forward(self, ctx_emb, z):
        """Forward pass using only z (ctx_emb ignored)."""
        x = z  # (B, T, latent_dim)
        mem_out, _ = self.mem(x)  # (B, T, mem_hidden)

        # FiLM modulation
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        features = gamma * mem_out + beta

        # Output
        output = self.output_net(features)  # (B, T, 25)
        output = output.view(output.shape[0], output.shape[1], 5, 5)

        return output


class FullCovarianceDecoder(nn.Module):
    """
    Decoder with full covariance NLL for learning spatial correlations.

    Covariance: Σ = FF^T + D
    - F: (5, 5, rank) factor matrix - captures spatial correlation
    - D: (5, 5) diagonal variance

    Key difference from LowRankCovarianceDecoder:
    - Uses FULL covariance NLL, not diagonal approximation
    - Gradient flows to off-diagonal terms in FF^T
    """

    def __init__(self, config: dict, gt_var: torch.Tensor = None):
        super().__init__()
        self.config = config
        self.rank = config.get("cov_rank", 4)

        # Mean decoder (z-only)
        self.mean_decoder = TwoStageDecoderZOnly(config)

        # Factor matrix F: (5, 5, rank) - learned
        if self.rank > 0:
            self.factor = nn.Parameter(torch.randn(5, 5, self.rank) * 0.1)
        else:
            self.register_buffer('factor', torch.zeros(5, 5, 1))

        # Diagonal D: (5, 5) - initialize from GT variance if provided
        if gt_var is not None:
            gt_var = torch.clamp(gt_var, min=1e-8)
            self.log_diag = nn.Parameter(torch.log(gt_var))
        else:
            self.log_diag = nn.Parameter(torch.zeros(5, 5))

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with optional sampling.

        Returns:
            mean: (B, T, 5, 5)
            samples: (B, T, 5, 5)
        """
        mean = self.mean_decoder(ctx_emb, z)  # (B, T, 5, 5)

        if not sample:
            return mean, mean

        B, T = mean.shape[:2]
        device = mean.device

        # Sample from low-rank covariance: x = μ + F·ε_rank + √D·ε_diag
        if self.rank > 0:
            eps_rank = torch.randn(B, T, self.rank, device=device)
            correlated = torch.einsum('ijr,btr->btij', self.factor, eps_rank)
        else:
            correlated = 0

        eps_diag = torch.randn(B, T, 5, 5, device=device)
        diag_std = torch.exp(0.5 * self.log_diag)
        independent = diag_std * eps_diag

        samples = mean + correlated + independent

        return mean, samples

    def compute_full_cov_nll(self, pred_mean, target):
        """
        Full covariance NLL using Woodbury identity.

        Loss = 0.5 * [log|Σ| + (x-μ)^T Σ^-1 (x-μ)]

        Woodbury: (D + FF^T)^-1 = D^-1 - D^-1 F (I + F^T D^-1 F)^-1 F^T D^-1
        """
        B, T = pred_mean.shape[:2]
        device = pred_mean.device

        # Flatten spatial dims
        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat  # (B, T, 25)

        # Covariance components
        D = torch.exp(self.log_diag).view(25)  # (25,)
        D = torch.clamp(D, min=1e-8)  # Numerical stability

        if self.rank > 0:
            F = self.factor.view(25, self.rank)  # (25, rank)
            rank = self.rank

            # Woodbury identity for Σ^-1
            D_inv = 1.0 / D  # (25,)

            # M = I + F^T D^-1 F
            FtDinvF = (F.T * D_inv) @ F  # (rank, rank)
            M = torch.eye(rank, device=device) + FtDinvF
            M_inv = torch.linalg.inv(M)

            # Quadratic form: (x-μ)^T Σ^-1 (x-μ)
            # = (x-μ)^T D^-1 (x-μ) - (x-μ)^T D^-1 F M^-1 F^T D^-1 (x-μ)
            Dinv_r = D_inv * residual  # (B, T, 25)
            term1 = (residual * Dinv_r).sum(dim=-1)  # (B, T)

            # F^T D^-1 (x-μ)
            Ft_Dinv_r = (residual * D_inv) @ F  # (B, T, rank)
            term2 = torch.einsum('btr,rs,bts->bt', Ft_Dinv_r, M_inv, Ft_Dinv_r)

            quad_form = term1 - term2  # (B, T)

            # log|Σ| = log|D| + log|M| (matrix determinant lemma)
            log_det_D = torch.log(D).sum()
            log_det_M = torch.linalg.slogdet(M)[1]
            log_det_Sigma = log_det_D + log_det_M

        else:
            # No factor, just diagonal
            D_inv = 1.0 / D
            Dinv_r = D_inv * residual
            quad_form = (residual * Dinv_r).sum(dim=-1)
            log_det_Sigma = torch.log(D).sum()

        # NLL = 0.5 * [log|Σ| + quadratic_form] + const
        nll = 0.5 * (log_det_Sigma + quad_form)

        return nll.mean()

    def get_total_variance(self):
        """Get total variance per grid point (diagonal of Σ)."""
        diag_var = torch.exp(self.log_diag)
        if self.rank > 0:
            factor_var = (self.factor ** 2).sum(dim=-1)
            return diag_var + factor_var
        return diag_var

    def get_covariance_matrix(self):
        """Return full 25x25 covariance matrix."""
        if self.rank > 0:
            F = self.factor.view(25, self.rank)
            cov = F @ F.T
        else:
            cov = torch.zeros(25, 25, device=self.log_diag.device)

        D = torch.diag(torch.exp(self.log_diag).view(25))
        return cov + D

    def get_correlation_matrix(self):
        """Return full 25x25 correlation matrix."""
        cov = self.get_covariance_matrix()
        std = torch.sqrt(torch.diag(cov))
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
        return corr


class CVAETwoStageFullCov(CVAETwoStage):
    """CVAETwoStage with Full Covariance decoder."""

    def __init__(self, config: dict, gt_var: torch.Tensor = None):
        nn.Module.__init__(self)
        self.config = config

        # Build encoders
        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)

        # Full covariance decoder
        self.decoder = FullCovarianceDecoder(config, gt_var)

        # Prior network
        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass returning mean predictions."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        # z dropout
        z_dropout = self.config.get("z_dropout", 0.0)
        if z_dropout > 0 and self.training:
            mask = torch.bernoulli(torch.full_like(z, 1 - z_dropout))
            z = z * mask / (1 - z_dropout)

        mean, _ = self.decoder(ctx_emb, z, sample=False)

        if return_full_sequence:
            return mean, z_mean, z_logvar, ctx_emb

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with full covariance."""
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
            _, sample = self.decoder(ctx_emb, z, sample=True)
            samples.append(sample)

        return torch.stack(samples, dim=0)


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


def train_stage1(model, train_loader, val_loader, config, epochs=50):
    """Stage 1: Train mean decoder with MSE, freeze covariance params."""
    device = config["device"]
    model = model.to(device)

    # Freeze covariance parameters
    model.decoder.factor.requires_grad = False
    model.decoder.log_diag.requires_grad = False

    params = [p for n, p in model.named_parameters()
              if 'factor' not in n and 'log_diag' not in n and p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=1e-3)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, _ = model(batch, return_full_sequence=True)

            # MSE loss on predictions
            target = batch_data[:, 1:]
            pred = mean[:, :-1]
            mse_loss = ((pred - target) ** 2).mean()

            # KL loss
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            loss = mse_loss + config.get("kl_weight", 0.001) * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(mse_loss.item())

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
                mse_loss = ((pred - target) ** 2).mean()
                val_losses.append(mse_loss.item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Stage 1 Epoch {epoch+1}/{epochs}: "
                  f"Train MSE={np.mean(train_losses):.6f}, Val MSE={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Unfreeze covariance params for stage 2
    model.decoder.factor.requires_grad = True
    model.decoder.log_diag.requires_grad = True

    return best_val_loss


def train_stage2_full_cov(model, train_loader, val_loader, config, epochs=200, lr=0.01):
    """
    Stage 2: Train covariance params with FULL covariance NLL.

    Key difference: Uses full Σ^-1, not diagonal approximation.
    This provides gradient signal for off-diagonal correlations in F.
    """
    device = config["device"]
    model = model.to(device)

    # Freeze all except covariance params
    for name, param in model.named_parameters():
        if 'factor' in name or 'log_diag' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

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

            with torch.no_grad():
                mean, _, _, _ = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # FULL COVARIANCE NLL (not diagonal!)
            nll_loss = model.decoder.compute_full_cov_nll(pred, target)

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
                nll_loss = model.decoder.compute_full_cov_nll(pred, target)
                val_losses.append(nll_loss.item())

        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            # Print correlation stats
            with torch.no_grad():
                corr = model.decoder.get_correlation_matrix().cpu().numpy()
                mask = np.triu(np.ones((25, 25), dtype=bool), k=1)
                mean_abs_corr = np.abs(corr[mask]).mean()

            print(f"  Stage 2 Epoch {epoch+1}/{epochs}: "
                  f"Train NLL={np.mean(train_losses):.4f}, Val NLL={val_loss:.4f}, "
                  f"Mean Abs Corr={mean_abs_corr:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    return best_val_loss


def evaluate_correlation(model, gt_log_returns):
    """Evaluate learned correlation vs GT correlation."""
    # GT correlation
    gt_flat = gt_log_returns.reshape(-1, 25)
    gt_corr = np.corrcoef(gt_flat.T)
    mask = np.triu(np.ones((25, 25), dtype=bool), k=1)
    gt_mean_abs_corr = np.abs(gt_corr[mask]).mean()

    # Learned correlation
    with torch.no_grad():
        learned_corr = model.decoder.get_correlation_matrix().cpu().numpy()
        learned_mean_abs_corr = np.abs(learned_corr[mask]).mean()

    # Factor contribution
    with torch.no_grad():
        total_var = model.decoder.get_total_variance().cpu().numpy()
        diag_var = np.exp(model.decoder.log_diag.cpu().numpy())
        if model.decoder.rank > 0:
            factor_var = (model.decoder.factor.cpu().numpy() ** 2).sum(axis=-1)
            factor_contribution = (factor_var / total_var).mean()
        else:
            factor_contribution = 0.0

    # Correlation pattern match
    corr_pattern_match = np.corrcoef(gt_corr.flatten(), learned_corr.flatten())[0, 1]

    return {
        "gt_mean_abs_corr": gt_mean_abs_corr,
        "learned_mean_abs_corr": learned_mean_abs_corr,
        "correlation_preserved": learned_mean_abs_corr / gt_mean_abs_corr if gt_mean_abs_corr > 0 else 0,
        "factor_contribution": factor_contribution,
        "corr_pattern_match": corr_pattern_match,
    }


def run_experiment(rank=4, epochs_stage1=50, epochs_stage2=200):
    """Run full covariance NLL experiment."""
    print("=" * 70)
    print("Phase 1: Full Covariance NLL for Learning Spatial Correlations")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    log_returns, log_surfaces = to_log_returns(data["surface"])

    # GT variance for initialization
    gt_var = torch.tensor(log_returns.var(axis=0), dtype=torch.float32)

    # Config
    config = TWO_STAGE_CONFIG.copy()
    config["latent_dim"] = 8
    config["decoder_mem_hidden"] = 64
    config["cov_rank"] = rank
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfig:")
    print(f"  latent_dim: {config['latent_dim']}")
    print(f"  cov_rank: {rank}")
    print(f"  device: {config['device']}")

    # Create dataloaders
    context_len = config.get("context_len", 20)
    n_train = int(len(log_returns) * 0.8)

    train_loader = create_dataloader(log_returns[:n_train], context_len, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], context_len, batch_size=32, shuffle=False)

    # Create model
    model = CVAETwoStageFullCov(config, gt_var)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Stage 1: Train mean decoder
    print(f"\n--- Stage 1: Train Mean Decoder ({epochs_stage1} epochs) ---")
    train_stage1(model, train_loader, val_loader, config, epochs=epochs_stage1)

    # Evaluate before stage 2
    print("\nBefore Stage 2 (Full Cov NLL):")
    results_before = evaluate_correlation(model, log_returns)
    print(f"  GT mean abs corr: {results_before['gt_mean_abs_corr']:.4f}")
    print(f"  Learned mean abs corr: {results_before['learned_mean_abs_corr']:.4f}")
    print(f"  Factor contribution: {results_before['factor_contribution']*100:.1f}%")

    # Stage 2: Train covariance with FULL NLL
    print(f"\n--- Stage 2: Train Covariance with Full NLL ({epochs_stage2} epochs) ---")
    train_stage2_full_cov(model, train_loader, val_loader, config, epochs=epochs_stage2)

    # Evaluate after stage 2
    print("\nAfter Stage 2 (Full Cov NLL):")
    results_after = evaluate_correlation(model, log_returns)
    print(f"  GT mean abs corr: {results_after['gt_mean_abs_corr']:.4f}")
    print(f"  Learned mean abs corr: {results_after['learned_mean_abs_corr']:.4f}")
    print(f"  Correlation preserved: {results_after['correlation_preserved']*100:.1f}%")
    print(f"  Factor contribution: {results_after['factor_contribution']*100:.1f}%")
    print(f"  Correlation pattern match: {results_after['corr_pattern_match']:.4f}")

    # Save model
    output_dir = Path("models/backfill/two_stage/full_cov_nll")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"rank_{rank}_best.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "results": results_after,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Success criteria
    print("\n" + "=" * 70)
    print("PHASE 1 VERIFICATION")
    print("=" * 70)
    print(f"  Mean abs correlation: {results_before['learned_mean_abs_corr']:.4f} → {results_after['learned_mean_abs_corr']:.4f}")
    print(f"  Target: >0.15")
    print(f"  Status: {'PASS' if results_after['learned_mean_abs_corr'] > 0.15 else 'FAIL'}")
    print()
    print(f"  Factor contribution: {results_before['factor_contribution']*100:.1f}% → {results_after['factor_contribution']*100:.1f}%")
    print(f"  Target: >30%")
    print(f"  Status: {'PASS' if results_after['factor_contribution'] > 0.30 else 'FAIL'}")

    return model, results_after


if __name__ == "__main__":
    # Test different ranks
    for rank in [4, 8]:
        print(f"\n{'#' * 70}")
        print(f"# Testing rank={rank}")
        print(f"{'#' * 70}")
        model, results = run_experiment(rank=rank, epochs_stage1=50, epochs_stage2=200)
