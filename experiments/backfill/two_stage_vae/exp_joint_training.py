"""
Phase 2: Joint Training with Z-Dependent Covariance

Problem: Decoder ignores z - mean prediction is same regardless of z value.
         Oracle MSE = Zero-z MSE (0% z contribution)

Solution:
1. Make covariance z-dependent: Σ(z) = F(z)F(z)^T + D(z)
2. Train mean + variance JOINTLY with full NLL (not 2-stage)
3. Decoder MUST use z to predict good variance → forces z usage for mean too

Expected outcome:
- z contribution to mean: 0% → >20%
- Variance changes with different z values
- Model captures regime-dependent uncertainty

Usage:
    python experiments/backfill/two_stage_vae/exp_joint_training.py
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
)


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


class ZDependentCovarianceDecoder(nn.Module):
    """
    Decoder where BOTH mean and covariance depend on z.

    Key differences from Phase 1:
    - F(z) and D(z) are computed from z, not fixed parameters
    - Joint training makes decoder MUST use z for good NLL
    - DIRECT z→mean mapping (no LSTM bypass) to force z usage
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        latent_dim = config.get("latent_dim", 8)
        mem_hidden = config.get("decoder_mem_hidden", 64)
        self.rank = config.get("cov_rank", 4)

        # DIRECT mean decoder: z → MLP → mean (no LSTM that can bypass z!)
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
        )

        # Z-DEPENDENT covariance networks
        # These make Σ depend on z, forcing the decoder to use z
        self.factor_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25 * self.rank),  # F(z): (B, 25*rank)
        )

        self.log_diag_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 25),  # log_diag(z): (B, 25)
        )

        # Initialize log_diag output bias to reasonable variance
        # GT variance ranges from 0.001 to 0.7, so log ranges from -7 to -0.35
        nn.init.constant_(self.log_diag_net[-1].bias, -3.0)  # ~0.05 variance

    def forward(self, ctx_emb, z, sample=True):
        """
        Forward pass with z-dependent covariance.

        Args:
            ctx_emb: Context embedding (ignored for z-only decoder)
            z: Latent variable (B, T, latent_dim)
            sample: Whether to sample with covariance

        Returns:
            mean: (B, T, 5, 5)
            samples: (B, T, 5, 5)
            factor: (B, 25, rank) - z-dependent factor
            log_diag: (B, 25) - z-dependent log-diagonal
        """
        B, T, _ = z.shape
        device = z.device

        # DIRECT mean prediction: z → MLP → mean (z MUST be used!)
        z_flat = z.view(B * T, -1)  # (B*T, latent_dim)
        mean = self.mean_net(z_flat)  # (B*T, 25)
        mean = mean.view(B, T, 5, 5)

        # Z-dependent covariance (pool z over time for covariance params)
        z_pooled = z.mean(dim=1)  # (B, latent_dim)

        factor_flat = self.factor_net(z_pooled)  # (B, 25*rank)
        factor = factor_flat.view(B, 25, self.rank)  # (B, 25, rank)

        log_diag = self.log_diag_net(z_pooled)  # (B, 25)
        log_diag = torch.clamp(log_diag, min=-10, max=2)  # Numerical stability

        if not sample:
            return mean, mean, factor, log_diag

        # Sample: x = μ + F·ε_rank + √D·ε_diag
        # F is now (B, 25, rank), need batch-aware sampling
        eps_rank = torch.randn(B, T, self.rank, device=device)

        # Batched einsum: factor[b,i,r] * eps_rank[b,t,r] -> correlated[b,t,i]
        correlated = torch.einsum('bir,btr->bti', factor, eps_rank)  # (B, T, 25)
        correlated = correlated.view(B, T, 5, 5)

        eps_diag = torch.randn(B, T, 5, 5, device=device)
        diag_std = torch.exp(0.5 * log_diag).view(B, 1, 5, 5)  # (B, 1, 5, 5)
        independent = diag_std * eps_diag

        samples = mean + correlated + independent

        return mean, samples, factor, log_diag

    def compute_full_cov_nll(self, pred_mean, target, factor, log_diag):
        """
        Full covariance NLL with batch-dependent covariance.

        Args:
            pred_mean: (B, T, 5, 5)
            target: (B, T, 5, 5)
            factor: (B, 25, rank) - z-dependent
            log_diag: (B, 25) - z-dependent
        """
        B, T = pred_mean.shape[:2]
        device = pred_mean.device

        mean_flat = pred_mean.view(B, T, 25)
        target_flat = target.view(B, T, 25)
        residual = target_flat - mean_flat  # (B, T, 25)

        # Covariance is now batch-dependent
        D = torch.exp(log_diag)  # (B, 25)
        D = torch.clamp(D, min=1e-8)

        total_nll = 0.0

        for b in range(B):
            D_b = D[b]  # (25,)
            F_b = factor[b]  # (25, rank)
            r_b = residual[b]  # (T, 25)

            D_inv = 1.0 / D_b

            # Woodbury
            FtDinvF = (F_b.T * D_inv) @ F_b  # (rank, rank)
            M = torch.eye(self.rank, device=device) + FtDinvF
            M_inv = torch.linalg.inv(M)

            # Quadratic form for all T timesteps
            Dinv_r = D_inv * r_b  # (T, 25)
            term1 = (r_b * Dinv_r).sum(dim=-1)  # (T,)

            Ft_Dinv_r = (r_b * D_inv) @ F_b  # (T, rank)
            term2 = torch.einsum('tr,rs,ts->t', Ft_Dinv_r, M_inv, Ft_Dinv_r)

            quad_form = term1 - term2  # (T,)

            # Log determinant
            log_det_D = torch.log(D_b).sum()
            log_det_M = torch.linalg.slogdet(M)[1]
            log_det_Sigma = log_det_D + log_det_M

            nll_b = 0.5 * (log_det_Sigma + quad_form.mean())
            total_nll += nll_b

        return total_nll / B


class CVAETwoStageJoint(CVAETwoStage):
    """CVAETwoStage with z-dependent covariance and joint training."""

    def __init__(self, config: dict):
        nn.Module.__init__(self)
        self.config = config

        self.ctx_encoder = TwoStageCtxEncoder(config)
        self.main_encoder = TwoStageMainEncoder(config)
        self.decoder = ZDependentCovarianceDecoder(config)

        latent_dim = config.get("latent_dim", 16)
        ctx_embedding_dim = config.get("ctx_embedding_dim", 3)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2),
        )

    def forward(self, batch, return_full_sequence=False):
        """Forward pass with z-dependent covariance."""
        surface = batch["surface"]
        B, T = surface.shape[:2]

        ctx_emb = self.ctx_encoder({"surface": surface})
        z_mean, z_logvar, z = self.main_encoder({"surface": surface})

        z_dropout = self.config.get("z_dropout", 0.0)
        if z_dropout > 0 and self.training:
            mask = torch.bernoulli(torch.full_like(z, 1 - z_dropout))
            z = z * mask / (1 - z_dropout)

        mean, samples, factor, log_diag = self.decoder(ctx_emb, z, sample=True)

        if return_full_sequence:
            return mean, z_mean, z_logvar, factor, log_diag

        return mean

    def sample(self, batch, n_samples=100):
        """Generate samples with z-dependent covariance."""
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


def train_two_phase(model, train_loader, val_loader, config, epochs=100):
    """
    TWO-PHASE training to force z usage for mean:

    Phase A (epochs 0-59): Train mean decoder with MSE only (NO variance)
        - No variance escape hatch
        - z MUST be used to predict accurate mean
        - Encoder + mean_net get optimized

    Phase B (epochs 60-99): Freeze mean, train variance with NLL
        - Mean decoder frozen
        - Only factor_net and log_diag_net get gradients
        - Learn z-dependent variance

    This prevents the "variance absorbs everything" problem.
    """
    device = config["device"]
    model = model.to(device)

    kl_weight = config.get("kl_weight", 0.001)

    phase_a_epochs = int(epochs * 0.6)  # 60% for mean pre-training
    phase_b_epochs = epochs - phase_a_epochs  # 40% for variance training

    best_val_loss = float('inf')
    best_state = None

    # ==================================================
    # PHASE A: Pre-train mean with MSE only (NO VARIANCE)
    # ==================================================
    print(f"\n--- Phase A: Pre-train Mean with MSE ({phase_a_epochs} epochs) ---")
    print("    (No variance - z MUST be used for accurate mean)")

    # Only train encoder and mean_net (freeze variance networks)
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
        train_kls = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # MSE ONLY - no variance to escape to!
            mse_loss = ((pred - target) ** 2).mean()

            # KL loss (keeps encoder meaningful)
            kl_loss = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).mean()

            loss = mse_loss + kl_weight * kl_loss

            optimizer_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_a.step()

            train_mses.append(mse_loss.item())
            train_kls.append(kl_loss.item())

        # Validation
        model.eval()
        val_mses = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}
                mean, z_mean, z_logvar, _, _ = model(batch, return_full_sequence=True)
                target = batch_data[:, 1:]
                pred = mean[:, :-1]
                val_mses.append(((pred - target) ** 2).mean().item())

        val_mse = np.mean(val_mses)
        scheduler_a.step(val_mse)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_a_epochs}: "
                  f"Train MSE={np.mean(train_mses):.6f}, KL={np.mean(train_kls):.4f}, "
                  f"Val MSE={val_mse:.6f}")

    # Check z contribution after Phase A
    print("\n    Checking z contribution after Phase A...")
    z_results_a = evaluate_z_contribution(model, val_loader, config)
    print(f"    Z contribution after Phase A: {z_results_a['z_contribution']:.1f}%")

    # ==================================================
    # PHASE B: Freeze mean, train variance with NLL
    # ==================================================
    print(f"\n--- Phase B: Train Variance with NLL ({phase_b_epochs} epochs) ---")
    print("    (Mean frozen - only variance networks train)")

    # Freeze mean decoder, unfreeze variance networks
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
    scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, patience=10, factor=0.5)

    for epoch in range(phase_b_epochs):
        model.train()
        train_nlls = []

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)

            target = batch_data[:, 1:]
            pred = mean[:, :-1]

            # NLL only - mean is frozen, just learning variance
            nll_loss = model.decoder.compute_full_cov_nll(pred, target, factor, log_diag)

            optimizer_b.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_b.step()

            train_nlls.append(nll_loss.item())

        # Validation
        model.eval()
        val_nlls = []
        with torch.no_grad():
            for (batch_data,) in val_loader:
                batch_data = batch_data.to(device)
                batch = {"surface": batch_data}
                mean, z_mean, z_logvar, factor, log_diag = model(batch, return_full_sequence=True)
                target = batch_data[:, 1:]
                pred = mean[:, :-1]
                val_nlls.append(model.decoder.compute_full_cov_nll(pred, target, factor, log_diag).item())

        val_nll = np.mean(val_nlls)
        scheduler_b.step(val_nll)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{phase_b_epochs}: "
                  f"Train NLL={np.mean(train_nlls):.4f}, Val NLL={val_nll:.4f}")

    # Unfreeze everything for final state
    for param in model.parameters():
        param.requires_grad = True

    return best_val_loss


def evaluate_z_contribution(model, val_loader, config):
    """
    Evaluate z contribution to mean prediction.

    Compare: MSE(decode(z_oracle)) vs MSE(decode(z=0))
    """
    device = config["device"]
    model.eval()

    oracle_mses = []
    zero_z_mses = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)
            batch = {"surface": batch_data}

            # Oracle z (from encoder)
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


def evaluate_variance_z_dependence(model, val_loader, config):
    """
    Evaluate whether variance changes with z.

    Compare variance for different z values.
    """
    device = config["device"]
    model.eval()

    variances_oracle = []
    variances_zero = []
    variances_random = []

    with torch.no_grad():
        for (batch_data,) in val_loader:
            batch_data = batch_data.to(device)

            ctx_emb = model.ctx_encoder({"surface": batch_data})
            z_mean, z_logvar, z = model.main_encoder({"surface": batch_data})

            # Variance with oracle z
            _, _, factor, log_diag = model.decoder(ctx_emb, z, sample=False)
            var_oracle = torch.exp(log_diag).mean().item()
            variances_oracle.append(var_oracle)

            # Variance with z=0
            z_zero = torch.zeros_like(z)
            _, _, factor_zero, log_diag_zero = model.decoder(ctx_emb, z_zero, sample=False)
            var_zero = torch.exp(log_diag_zero).mean().item()
            variances_zero.append(var_zero)

            # Variance with random z
            z_random = torch.randn_like(z) * 2
            _, _, factor_random, log_diag_random = model.decoder(ctx_emb, z_random, sample=False)
            var_random = torch.exp(log_diag_random).mean().item()
            variances_random.append(var_random)

    return {
        "mean_var_oracle": np.mean(variances_oracle),
        "mean_var_zero": np.mean(variances_zero),
        "mean_var_random": np.mean(variances_random),
        "var_range": np.std(variances_oracle) / np.mean(variances_oracle),
        "variance_varies": np.std(variances_oracle) > 0.01,
    }


def run_experiment(epochs=100):
    """Run joint training experiment."""
    print("=" * 70)
    print("Phase 2: Joint Training with Z-Dependent Covariance")
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

    # Create dataloaders
    context_len = config.get("context_len", 20)
    n_train = int(len(log_returns) * 0.8)

    train_loader = create_dataloader(log_returns[:n_train], context_len, batch_size=32)
    val_loader = create_dataloader(log_returns[n_train:], context_len, batch_size=32, shuffle=False)

    # Create model
    model = CVAETwoStageJoint(config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Two-phase training: MSE first (force z usage), then NLL for variance
    print(f"\n--- Two-Phase Training ({epochs} epochs) ---")
    train_two_phase(model, train_loader, val_loader, config, epochs=epochs)

    # Evaluate z contribution
    print("\n--- Evaluating Z Contribution ---")
    z_results = evaluate_z_contribution(model, val_loader, config)
    print(f"  Oracle MSE: {z_results['oracle_mse']:.6f}")
    print(f"  Zero-z MSE: {z_results['zero_z_mse']:.6f}")
    print(f"  Z contribution: {z_results['z_contribution']:.1f}%")

    # Evaluate variance z-dependence
    print("\n--- Evaluating Variance Z-Dependence ---")
    var_results = evaluate_variance_z_dependence(model, val_loader, config)
    print(f"  Mean variance (oracle z): {var_results['mean_var_oracle']:.6f}")
    print(f"  Mean variance (z=0): {var_results['mean_var_zero']:.6f}")
    print(f"  Mean variance (random z): {var_results['mean_var_random']:.6f}")
    print(f"  Variance varies with z: {var_results['variance_varies']}")

    # Save model
    output_dir = Path("models/backfill/two_stage/joint_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "joint_best.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "z_results": z_results,
        "var_results": var_results,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Success criteria
    print("\n" + "=" * 70)
    print("PHASE 2 VERIFICATION")
    print("=" * 70)
    print(f"  Z contribution to mean: {z_results['z_contribution']:.1f}%")
    print(f"  Target: >20%")
    print(f"  Status: {'PASS' if z_results['z_contribution'] > 20 else 'FAIL'}")
    print()
    print(f"  Variance varies with z: {var_results['variance_varies']}")
    print(f"  Target: True")
    print(f"  Status: {'PASS' if var_results['variance_varies'] else 'FAIL'}")

    return model, z_results, var_results


if __name__ == "__main__":
    model, z_results, var_results = run_experiment(epochs=100)
