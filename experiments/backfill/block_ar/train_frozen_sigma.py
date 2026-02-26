"""
Phase-2 training: Frozen encoder + learned sigma head for condition-dependent uncertainty.

Approach: Load a pretrained vol-scaled Block-AR model, freeze ALL weights, then train
a small MLP sigma head on the frozen encoder features. The sigma head learns to predict
per-sample log_sigma from the condition vector, replacing the hand-coded vol_scale.

Two loss options:
1. MSE against empirical variance targets (NsDiff-style, simplest)
2. beta-NLL on log-ratio residuals (Seitzer et al. ICLR 2022)

At inference, sigma_head(condition) replaces vol_scale in the denormalization:
  sample_abs = exp(diffusion_sample * sigma_learned) * baseline

References:
- NsDiff (ICML 2025): Non-stationary Diffusion for Probabilistic Time Series Forecasting
- Seitzer et al. (ICLR 2022): On the Pitfalls of Heteroscedastic Uncertainty Estimation
- Stirn et al. (AISTATS 2023): Faithful Heteroscedastic Regression with Neural Networks
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


class FrozenSigmaHead(nn.Module):
    """MLP sigma head trained on frozen encoder features.

    Predicts per-sample log_sigma from condition vector.
    Architecture: 3-layer MLP with SiLU activation and softplus output.

    Following NsDiff (ICML 2025): small MLP on frozen features, softplus for positivity.
    """

    def __init__(self, cond_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize so sigma ≈ global_mean_vol (0.0187) initially
        # log(0.0187) ≈ -3.98, but we'll use softplus so init near 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -3.0)  # softplus(-3) ≈ 0.05

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Predict per-sample sigma (positive).

        Args:
            condition: (B, cond_dim) frozen encoder output

        Returns:
            sigma: (B, 1) positive scalar per sample
        """
        raw = self.net(condition)  # (B, 1)
        # Softplus for guaranteed positivity + lower bound
        sigma = F.softplus(raw) + 1e-4  # (B, 1)
        return sigma


def compute_empirical_targets(
    history: torch.Tensor,
    future: torch.Tensor,
    global_mean_vol: float = 0.0187,
) -> tuple:
    """Compute empirical variance targets for sigma head training.

    For each sample, compute:
    1. baseline = last frame of history (denormalized)
    2. log_ratio = log(future / baseline) per frame
    3. target_sigma = std(log_ratio) across frames and cells

    This is what sigma should predict: the std of log-ratios given the condition.

    Args:
        history: (B, T_hist, 5, 5) in [-1, 1]
        future: (B, T_fut, 5, 5) in [-1, 1]
        global_mean_vol: normalization constant

    Returns:
        target_sigma: (B, 1) empirical std of log-ratios
        vol_scale: (B, 1) hand-coded vol_scale for comparison
    """
    eps_iv = 1e-4
    baseline = denormalize_iv(history[:, -1]).clamp(min=0.01).unsqueeze(1)  # (B, 1, 5, 5)
    future_abs = denormalize_iv(future).clamp(min=eps_iv, max=1.0 - eps_iv)

    log_ratio = torch.log(future_abs / baseline)  # (B, T_fut, 5, 5)

    # Target sigma: std of log-ratios per sample (across all frames and cells)
    target_sigma = log_ratio.reshape(log_ratio.shape[0], -1).std(dim=1, keepdim=True)  # (B, 1)

    # Hand-coded vol_scale for comparison
    past_abs = denormalize_iv(history)
    mean_iv = past_abs.mean(dim=(-1, -2))  # (B, T_hist)
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vol = daily_chg.std(dim=1, keepdim=True)  # (B, 1)
    vol_scale = (vol / global_mean_vol).clamp(0.5, 2.0)

    return target_sigma, vol_scale


def train_sigma_head_mse(
    model: ConditionalBlockARDDPM,
    sigma_head: FrozenSigmaHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    global_mean_vol: float = 0.0187,
) -> dict:
    """Train sigma head with MSE against empirical variance targets.

    NsDiff-style: simplest approach, most direct signal.
    """
    sigma_head.train()
    model.eval()

    total_loss = 0.0
    total_corr = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training sigma head", leave=False):
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        # Get frozen encoder features
        with torch.no_grad():
            condition = model.encoder(history, mask=None)
            if model.config.forward_only:
                condition = condition + model.encoder.null_embedding.expand(history.shape[0], -1)

        # Compute empirical targets
        target_sigma, vol_scale = compute_empirical_targets(
            history, future, global_mean_vol
        )

        # Predict sigma
        pred_sigma = sigma_head(condition)  # (B, 1)

        # MSE loss on log-sigma (more numerically stable than raw sigma)
        loss = F.mse_loss(torch.log(pred_sigma), torch.log(target_sigma))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track correlation between predicted and target
        with torch.no_grad():
            corr = torch.corrcoef(
                torch.stack([pred_sigma.squeeze(), target_sigma.squeeze()])
            )[0, 1].item()

        total_loss += loss.item()
        total_corr += corr if not np.isnan(corr) else 0
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "correlation": total_corr / n_batches,
    }


def train_sigma_head_beta_nll(
    model: ConditionalBlockARDDPM,
    sigma_head: FrozenSigmaHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    beta: float = 0.5,
    global_mean_vol: float = 0.0187,
) -> dict:
    """Train sigma head with beta-NLL loss (Seitzer et al. ICLR 2022).

    NLL: 0.5 * log(sigma^2) + 0.5 * residual^2 / sigma^2
    beta-NLL: multiply by (sigma^2.detach())^beta to prevent self-amplifying collapse.

    The residual here is the difference between the log-ratio and its mean (zero),
    so the sigma should capture the spread of log-ratios.
    """
    sigma_head.train()
    model.eval()

    total_loss = 0.0
    total_corr = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training sigma head (beta-NLL)", leave=False):
        history = batch["history"].to(device)
        future = batch["future"].to(device)

        # Get frozen encoder features
        with torch.no_grad():
            condition = model.encoder(history, mask=None)
            if model.config.forward_only:
                condition = condition + model.encoder.null_embedding.expand(history.shape[0], -1)

        # Compute log-ratios (the "residuals" we want to model variance of)
        eps_iv = 1e-4
        baseline = denormalize_iv(history[:, -1]).clamp(min=0.01).unsqueeze(1)
        future_abs = denormalize_iv(future).clamp(min=eps_iv, max=1.0 - eps_iv)
        log_ratio = torch.log(future_abs / baseline)  # (B, T_fut, 5, 5)

        # Predict sigma from frozen features
        pred_sigma = sigma_head(condition)  # (B, 1)
        pred_var = pred_sigma.pow(2)  # (B, 1)

        # NLL: averaged over all frames and cells per sample
        # residual = log_ratio (mean is ~0 by construction)
        residual_sq = log_ratio.reshape(log_ratio.shape[0], -1).pow(2).mean(dim=1, keepdim=True)  # (B, 1)

        nll = 0.5 * torch.log(pred_var) + 0.5 * residual_sq / pred_var  # (B, 1)

        # beta-NLL weighting
        if beta > 0:
            nll = nll * pred_var.detach().pow(beta)

        loss = nll.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track correlation
        with torch.no_grad():
            target_sigma = log_ratio.reshape(log_ratio.shape[0], -1).std(dim=1)
            corr = torch.corrcoef(
                torch.stack([pred_sigma.squeeze(), target_sigma])
            )[0, 1].item()

        total_loss += loss.item()
        total_corr += corr if not np.isnan(corr) else 0
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "correlation": total_corr / n_batches,
    }


def validate_sigma(
    model: ConditionalBlockARDDPM,
    sigma_head: FrozenSigmaHead,
    dataloader: DataLoader,
    device: str,
    global_mean_vol: float = 0.0187,
) -> dict:
    """Validate sigma head predictions against empirical targets."""
    sigma_head.eval()
    model.eval()

    all_pred_sigma = []
    all_target_sigma = []
    all_vol_scale = []

    with torch.no_grad():
        for batch in dataloader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            condition = model.encoder(history, mask=None)
            if model.config.forward_only:
                condition = condition + model.encoder.null_embedding.expand(history.shape[0], -1)

            target_sigma, vol_scale = compute_empirical_targets(
                history, future, global_mean_vol
            )

            pred_sigma = sigma_head(condition)

            all_pred_sigma.append(pred_sigma.cpu())
            all_target_sigma.append(target_sigma.cpu())
            all_vol_scale.append(vol_scale.cpu())

    pred = torch.cat(all_pred_sigma).squeeze()
    target = torch.cat(all_target_sigma).squeeze()
    vol_s = torch.cat(all_vol_scale).squeeze()

    # Correlation with target
    corr_pred = torch.corrcoef(torch.stack([pred, target]))[0, 1].item()
    corr_volscale = torch.corrcoef(torch.stack([vol_s, target]))[0, 1].item()

    # MSE comparison
    mse_pred = F.mse_loss(torch.log(pred), torch.log(target)).item()
    mse_volscale = F.mse_loss(torch.log(vol_s * global_mean_vol), torch.log(target)).item()

    # CoV (coefficient of variation) of predictions
    cov_pred = (pred.std() / pred.mean()).item()
    cov_target = (target.std() / target.mean()).item()

    return {
        "corr_pred_vs_target": corr_pred,
        "corr_volscale_vs_target": corr_volscale,
        "mse_pred": mse_pred,
        "mse_volscale": mse_volscale,
        "cov_pred": cov_pred,
        "cov_target": cov_target,
        "pred_mean": pred.mean().item(),
        "pred_std": pred.std().item(),
        "target_mean": target.mean().item(),
        "target_std": target.std().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train frozen-encoder sigma head")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for sigma head checkpoint")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "beta_nll"],
                        help="Loss function: 'mse' (NsDiff-style) or 'beta_nll' (Seitzer)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="beta-NLL weight (only for --loss beta_nll)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Sigma head hidden dimension")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for sigma head")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for sigma head")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load pretrained model
    print(f"Loading pretrained model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    if dataclasses.is_dataclass(config_dict):
        config_dict = dataclasses.asdict(config_dict)

    model_config = BlockARConfig(**{
        k: v for k, v in config_dict.items()
        if k in {f.name for f in dataclasses.fields(BlockARConfig)}
    })
    model = ConditionalBlockARDDPM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # Freeze ALL model parameters
    for param in model.parameters():
        param.requires_grad = False

    n_frozen = sum(p.numel() for p in model.parameters())
    print(f"Frozen model parameters: {n_frozen:,}")
    print(f"Model config: bottleneck_dim={model_config.bottleneck_dim}, "
          f"ratio_target_mode={model_config.ratio_target_mode}")

    # Create sigma head
    sigma_head = FrozenSigmaHead(
        cond_dim=model_config.bottleneck_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)

    n_sigma_params = sum(p.numel() for p in sigma_head.parameters())
    print(f"Sigma head parameters: {n_sigma_params:,}")

    # Load data
    print("\nLoading data...")
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]

    train_dataset = VolSurfaceDataset(
        surfaces, model_config.history_len, model_config.future_len,
        start_idx=0, end_idx=4040,
    )
    val_dataset = VolSurfaceDataset(
        surfaces, model_config.history_len, model_config.future_len,
        start_idx=4040, end_idx=4540,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        sigma_head.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 10,
    )

    # Training
    print(f"\nTraining sigma head ({args.loss} loss, {args.epochs} epochs)...")
    print("=" * 60)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_corr = -1.0

    for epoch in range(1, args.epochs + 1):
        if args.loss == "mse":
            train_metrics = train_sigma_head_mse(
                model, sigma_head, train_loader, optimizer, args.device,
                global_mean_vol=model_config.global_mean_vol,
            )
        else:
            train_metrics = train_sigma_head_beta_nll(
                model, sigma_head, train_loader, optimizer, args.device,
                beta=args.beta,
                global_mean_vol=model_config.global_mean_vol,
            )

        lr_scheduler.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = validate_sigma(
                model, sigma_head, val_loader, args.device,
                global_mean_vol=model_config.global_mean_vol,
            )
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {train_metrics['loss']:.6f} | "
                f"Train corr: {train_metrics['correlation']:.3f} | "
                f"Val corr: {val_metrics['corr_pred_vs_target']:.3f} | "
                f"Vol_scale corr: {val_metrics['corr_volscale_vs_target']:.3f} | "
                f"Pred CoV: {val_metrics['cov_pred']:.3f} | "
                f"Target CoV: {val_metrics['cov_target']:.3f}"
            )

            if val_metrics["corr_pred_vs_target"] > best_corr:
                best_corr = val_metrics["corr_pred_vs_target"]
                torch.save({
                    "sigma_head_state_dict": sigma_head.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "config": {
                        "hidden_dim": args.hidden_dim,
                        "loss": args.loss,
                        "beta": args.beta if args.loss == "beta_nll" else None,
                        "base_checkpoint": args.checkpoint,
                        "cond_dim": model_config.bottleneck_dim,
                    },
                }, f"{args.output_dir}/best_sigma_head.pt")
                print(f"  -> New best correlation: {best_corr:.3f}")
        else:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {train_metrics['loss']:.6f} | "
                f"Train corr: {train_metrics['correlation']:.3f}"
            )

    # Final validation
    print("\n" + "=" * 60)
    print("Final Validation")
    print("=" * 60)
    final_val = validate_sigma(
        model, sigma_head, val_loader, args.device,
        global_mean_vol=model_config.global_mean_vol,
    )
    for k, v in final_val.items():
        print(f"  {k}: {v:.4f}")

    # Save final
    torch.save({
        "sigma_head_state_dict": sigma_head.state_dict(),
        "epoch": args.epochs,
        "val_metrics": final_val,
        "config": {
            "hidden_dim": args.hidden_dim,
            "loss": args.loss,
            "beta": args.beta if args.loss == "beta_nll" else None,
            "base_checkpoint": args.checkpoint,
            "cond_dim": model_config.bottleneck_dim,
        },
    }, f"{args.output_dir}/final_sigma_head.pt")

    print(f"\nSigma head saved to {args.output_dir}")
    print(f"Best validation correlation: {best_corr:.3f}")


if __name__ == "__main__":
    main()
