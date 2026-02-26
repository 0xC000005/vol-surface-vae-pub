#!/usr/bin/env python
"""
Train a per-cell sigma head for learned spatial uncertainty.

Bitter Lesson approach: let a neural network learn per-cell (5x5) uncertainty
from the condition vector, rather than hand-coding vol_scale.

Phase 1 (--mode posthoc): Train on reconstruction errors from frozen model.
  - Generate samples, compute per-cell error, train sigma head with NLL.
  - At inference: scale per-cell sample spread by learned sigma.

Phase 2 (--mode frozen_nll): Train sigma head integrated into ratio-space model.
  - Frozen encoder+denoiser, sigma head gets NLL gradient on actual targets.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/train_percell_sigma.py \
        --model_path models/backfill/block_ar_vol_scaled_30ep/best_model.pt \
        --mode posthoc --epochs 30 \
        --output_dir models/backfill/block_ar_percell_sigma_v1
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
    normalize_iv,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


class PerCellSigmaHead(nn.Module):
    """Predicts (5, 5) log-sigma from condition vector.

    Learns spatial uncertainty pattern: which cells are hard to predict
    and how that pattern changes with market regime.
    """

    def __init__(self, cond_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 25),  # 5x5 cells
        )
        # Initialize near GT spatial pattern (log of per-cell std ratios)
        # This gives the network a head start on the spatial structure
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: (B, cond_dim) from encoder
        Returns:
            log_sigma: (B, 5, 5) per-cell log-sigma
        """
        return self.net(condition).reshape(-1, 5, 5)


def load_model(model_path, device):
    """Load checkpoint and build model."""
    cp = torch.load(model_path, map_location=device, weights_only=False)
    c = cp["config"]
    if dataclasses.is_dataclass(c):
        c = dataclasses.asdict(c)
    config = BlockARConfig(**c)
    model = ConditionalBlockARDDPM(config)
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    return model, config


def get_condition_vector(model, history):
    """Extract condition vector from frozen encoder."""
    with torch.no_grad():
        cond = model.encoder(history, mask=None)
        if model.config.forward_only:
            cond = cond + model.encoder.null_embedding.expand(cond.shape[0], -1)
    return cond


def generate_errors(model, dataloader, device, n_samples=10):
    """Generate samples and compute per-cell reconstruction errors.

    Returns:
        conditions: list of (B, cond_dim) tensors
        errors: list of (B, future_len, 5, 5) absolute error tensors
        futures: list of (B, future_len, 5, 5) ground truth in [0,1]
    """
    all_conditions = []
    all_errors = []
    all_futures = []

    model.eval()
    for batch in tqdm(dataloader, desc="Generating samples"):
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        B = history.shape[0]

        # Get condition vector
        cond = get_condition_vector(model, history)

        # Generate samples
        with torch.no_grad():
            samples = model.sample(history, n_samples=n_samples)
            # samples: (B, n_samples, future_len, 5, 5) in [0, 1]

        # Ground truth in [0, 1]
        future_01 = denormalize_iv(future)  # (B, future_len, 5, 5)

        # Per-cell reconstruction error: std across samples at each cell
        # This is the "uncertainty" the model actually produces
        sample_std = samples.std(dim=1)  # (B, future_len, 5, 5)

        # Also compute mean prediction error for calibration
        sample_mean = samples.mean(dim=1)  # (B, future_len, 5, 5)
        abs_error = (sample_mean - future_01).abs()  # (B, future_len, 5, 5)

        all_conditions.append(cond.cpu())
        all_errors.append(abs_error.cpu())
        all_futures.append(future_01.cpu())

    return all_conditions, all_errors, all_futures


def train_posthoc(model, config, train_loader, val_loader, sigma_head,
                  device, args):
    """Train per-cell sigma head on reconstruction errors from frozen model.

    NLL loss: mean over cells of [log(sigma) + 0.5*(error/sigma)^2]
    Optimal sigma_* = sqrt(E[error^2]) per cell.
    """
    optimizer = torch.optim.Adam(sigma_head.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Training
        sigma_head.train()
        train_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            # Get condition from frozen encoder
            cond = get_condition_vector(model, history)

            # Generate samples from frozen model
            with torch.no_grad():
                samples = model.sample(history, n_samples=args.n_samples)
                # (B, n_samples, 30, 5, 5) in [0,1]

            future_01 = denormalize_iv(future)  # (B, 30, 5, 5)

            # Predict per-cell sigma
            log_sigma = sigma_head(cond)  # (B, 5, 5)
            log_sigma_clamped = log_sigma.clamp(-4, 4)
            sigma = torch.exp(log_sigma_clamped)  # (B, 5, 5)
            sigma_4d = sigma.unsqueeze(1)  # (B, 1, 5, 5) broadcast over time

            # NLL on each sample individually (not on mean error)
            # This trains sigma to match the actual sample distribution
            # loss = mean over samples of [log(sigma) + 0.5*((sample - gt)/sigma)^2]
            error = samples - future_01.unsqueeze(1)  # (B, n_samples, 30, 5, 5)
            sigma_expand = sigma_4d.unsqueeze(1)  # (B, 1, 1, 5, 5)

            nll = log_sigma_clamped.unsqueeze(1) + 0.5 * (error / sigma_expand) ** 2
            # nll: (B, n_samples, 30, 5, 5)
            loss = nll.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sigma_head.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches

        # Validation
        sigma_head.eval()
        val_loss = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                future = batch["future"].to(device)

                cond = get_condition_vector(model, history)
                samples = model.sample(history, n_samples=args.n_samples)
                future_01 = denormalize_iv(future)

                log_sigma = sigma_head(cond)
                log_sigma_clamped = log_sigma.clamp(-4, 4)
                sigma = torch.exp(log_sigma_clamped)
                sigma_4d = sigma.unsqueeze(1)

                error = samples - future_01.unsqueeze(1)
                sigma_expand = sigma_4d.unsqueeze(1)
                nll = log_sigma_clamped.unsqueeze(1) + 0.5 * (error / sigma_expand) ** 2
                val_loss += nll.mean().item()
                val_n += 1

        val_loss /= val_n

        # Analyze learned sigma
        sigma_head.eval()
        with torch.no_grad():
            # Get sigma for a batch to check spatial pattern
            sample_batch = next(iter(val_loader))
            h = sample_batch["history"].to(device)
            c = get_condition_vector(model, h)
            ls = sigma_head(c)  # (B, 5, 5)
            s = torch.exp(ls.clamp(-4, 4))
            mean_sigma = s.mean(dim=0).cpu().numpy()
            sigma_cov = s.std().item() / s.mean().item()

        print(f"Epoch {epoch}: train_nll={train_loss:.4f}  val_nll={val_loss:.4f}  "
              f"sigma_CoV={sigma_cov:.3f}  sigma_range=[{mean_sigma.min():.4f}, {mean_sigma.max():.4f}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in sigma_head.state_dict().items()}
            print(f"  -> New best val_nll: {val_loss:.4f}")

    return best_state, best_val_loss


def train_frozen_nll(model, config, train_loader, val_loader, sigma_head,
                     device, args):
    """Train per-cell sigma head with NLL on actual targets (not reconstruction errors).

    For each training window:
    1. Encode history → condition (frozen)
    2. Compute log(future/baseline) per cell
    3. Predict per-cell sigma from condition
    4. NLL: log(sigma_{r,c}) + 0.5*(log_ratio_{r,c}/sigma_{r,c})^2

    This directly learns the per-cell variance of log-ratios conditioned on history.
    No sample generation needed — pure supervised learning on the training data.
    """
    optimizer = torch.optim.Adam(sigma_head.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        sigma_head.train()
        train_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            history = batch["history"].to(device)
            future = batch["future"].to(device)

            # Get condition from frozen encoder
            cond = get_condition_vector(model, history)

            # Compute log(future/baseline) — the actual ratio target
            future_abs = denormalize_iv(future).clamp(min=1e-4, max=1.0-1e-4)
            baseline = denormalize_iv(history[:, -1]).clamp(min=0.01)
            baseline = baseline.unsqueeze(1)  # (B, 1, 5, 5)
            log_ratio = torch.log(future_abs / baseline)  # (B, 30, 5, 5)

            # Predict per-cell sigma
            log_sigma = sigma_head(cond)  # (B, 5, 5)
            log_sigma_clamped = log_sigma.clamp(-4, 4)
            sigma = torch.exp(log_sigma_clamped)  # (B, 5, 5)
            sigma_4d = sigma.unsqueeze(1)  # (B, 1, 5, 5) broadcast over time

            # Gaussian NLL
            z = log_ratio / sigma_4d
            nll = log_sigma_clamped.unsqueeze(1) + 0.5 * z ** 2  # (B, 30, 5, 5)
            loss = nll.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sigma_head.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches

        # Validation
        sigma_head.eval()
        val_loss = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                future = batch["future"].to(device)

                cond = get_condition_vector(model, history)
                future_abs = denormalize_iv(future).clamp(min=1e-4, max=1.0-1e-4)
                baseline = denormalize_iv(history[:, -1]).clamp(min=0.01)
                baseline = baseline.unsqueeze(1)
                log_ratio = torch.log(future_abs / baseline)

                log_sigma = sigma_head(cond)
                log_sigma_clamped = log_sigma.clamp(-4, 4)
                sigma = torch.exp(log_sigma_clamped)
                sigma_4d = sigma.unsqueeze(1)

                z = log_ratio / sigma_4d
                nll = log_sigma_clamped.unsqueeze(1) + 0.5 * z ** 2
                val_loss += nll.mean().item()
                val_n += 1

        val_loss /= val_n

        # Analyze learned sigma
        sigma_head.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            h = sample_batch["history"].to(device)
            c = get_condition_vector(model, h)
            ls = sigma_head(c)
            s = torch.exp(ls.clamp(-4, 4))
            mean_sigma = s.mean(dim=0).cpu().numpy()
            sigma_cov = s.std().item() / s.mean().item()

            # Check condition-dependence: compute sigma for different subsets
            all_sigmas = []
            for batch in val_loader:
                h_b = batch["history"].to(device)
                c_b = get_condition_vector(model, h_b)
                s_b = torch.exp(sigma_head(c_b).clamp(-4, 4))
                all_sigmas.append(s_b.cpu())
            all_sigmas = torch.cat(all_sigmas, dim=0)  # (N_val, 5, 5)

            # Cross-sample variation of mean sigma (temporal conditioning signal)
            mean_per_sample = all_sigmas.mean(dim=(-1, -2))  # (N_val,)
            temporal_cov = mean_per_sample.std().item() / mean_per_sample.mean().item()

        print(f"Epoch {epoch}: train_nll={train_loss:.4f}  val_nll={val_loss:.4f}  "
              f"sigma_CoV={sigma_cov:.3f}  temporal_CoV={temporal_cov:.3f}  "
              f"range=[{mean_sigma.min():.4f}, {mean_sigma.max():.4f}]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in sigma_head.state_dict().items()}
            print(f"  -> New best val_nll: {val_loss:.4f}")

    return best_state, best_val_loss


def evaluate_sigma_head(model, sigma_head, test_loader, device, n_samples=50):
    """Evaluate learned sigma: spatial pattern, condition-dependence, coverage."""
    model.eval()
    sigma_head.eval()

    all_sigmas = []
    all_cond_vars = []

    # Collect sigma predictions and conditioning variables
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            history = batch["history"].to(device)
            cond = get_condition_vector(model, history)

            log_sigma = sigma_head(cond)
            sigma = torch.exp(log_sigma.clamp(-4, 4))  # (B, 5, 5)
            all_sigmas.append(sigma.cpu())

            # Compute vol_of_vol for condition-dependence analysis
            hist_abs = denormalize_iv(history).cpu().numpy()
            mean_iv = hist_abs.mean(axis=(-1, -2))
            daily_chg = np.diff(mean_iv, axis=1)
            vov = daily_chg.std(axis=1)
            all_cond_vars.append(vov)

    all_sigmas = torch.cat(all_sigmas, dim=0).numpy()  # (N, 5, 5)
    all_vov = np.concatenate(all_cond_vars)  # (N,)

    results = {}

    # 1. Spatial pattern
    mean_sigma = all_sigmas.mean(axis=0)  # (5, 5)
    results["mean_sigma"] = mean_sigma.tolist()
    results["sigma_range"] = [float(mean_sigma.min()), float(mean_sigma.max())]
    results["sigma_ratio"] = float(mean_sigma.max() / mean_sigma.min())

    print("\nMean sigma per cell:")
    for r in range(5):
        row = "  ".join(f"{mean_sigma[r, c]:.4f}" for c in range(5))
        print(f"  [{row}]")
    print(f"  Max/Min ratio: {mean_sigma.max()/mean_sigma.min():.1f}x")

    # 2. Condition dependence — Q5/Q1 of mean sigma
    vov_q20 = np.percentile(all_vov, 20)
    vov_q80 = np.percentile(all_vov, 80)
    calm_mask = all_vov <= vov_q20
    turb_mask = all_vov >= vov_q80

    calm_sigma = all_sigmas[calm_mask].mean(axis=0)
    turb_sigma = all_sigmas[turb_mask].mean(axis=0)

    results["calm_mean_sigma"] = float(calm_sigma.mean())
    results["turb_mean_sigma"] = float(turb_sigma.mean())
    results["q5q1_mean"] = float(turb_sigma.mean() / calm_sigma.mean())

    print(f"\nCondition dependence:")
    print(f"  Calm mean sigma: {calm_sigma.mean():.4f}")
    print(f"  Turb mean sigma: {turb_sigma.mean():.4f}")
    print(f"  Q5/Q1 (mean): {turb_sigma.mean()/calm_sigma.mean():.3f}x")

    # 3. Per-cell Q5/Q1
    print(f"\nPer-cell Q5/Q1 (turb/calm):")
    percell_q5q1 = turb_sigma / calm_sigma
    for r in range(5):
        row = "  ".join(f"{percell_q5q1[r, c]:.3f}" for c in range(5))
        print(f"  [{row}]")
    results["percell_q5q1"] = percell_q5q1.tolist()

    # 4. Spearman correlation per cell
    from scipy.stats import spearmanr
    print(f"\nSpearman(sigma, vov) per cell:")
    spearman_grid = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            rho, _ = spearmanr(all_vov, all_sigmas[:, r, c])
            spearman_grid[r, c] = rho
    for r in range(5):
        row = "  ".join(f"{spearman_grid[r, c]:+.3f}" for c in range(5))
        print(f"  [{row}]")
    results["spearman_grid"] = spearman_grid.tolist()
    results["spearman_mean"] = float(spearman_grid.mean())

    # 5. Temporal × spatial interaction
    # During turbulence, does the PATTERN change (not just scale up)?
    turb_pattern = turb_sigma / turb_sigma.mean()
    calm_pattern = calm_sigma / calm_sigma.mean()
    pattern_change = np.abs(turb_pattern - calm_pattern).mean()
    results["pattern_change"] = float(pattern_change)
    print(f"\nPattern change (turb vs calm): {pattern_change:.4f}")
    print(f"  (0 = just scaling, >0 = spatial pattern reshapes)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to frozen backbone model checkpoint")
    parser.add_argument("--mode", choices=["posthoc", "frozen_nll"], default="frozen_nll",
                        help="Training mode")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Samples per window for posthoc mode")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load frozen backbone
    print(f"Loading model: {args.model_path}")
    model, config = load_model(args.model_path, args.device)
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model frozen ({sum(p.numel() for p in model.parameters())} params)")

    # Create sigma head
    sigma_head = PerCellSigmaHead(
        cond_dim=config.bottleneck_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    n_params = sum(p.numel() for p in sigma_head.parameters())
    print(f"Sigma head: {n_params} params")

    # Load data
    print("Loading data...")
    data = np.load(getattr(config, 'data_path', 'data/vol_surface_with_ret.npz'))
    surfaces = data["surface"]

    # Data splits (same as BlockARPOCConfig defaults)
    train_end = getattr(config, 'train_end', 4040)
    val_start = getattr(config, 'val_start', 4040)
    val_end = getattr(config, 'val_end', 4540)
    test_start = getattr(config, 'test_start', 4540)

    train_dataset = VolSurfaceDataset(
        surfaces[:train_end],
        history_len=config.history_len,
        future_len=config.future_len,
    )
    val_dataset = VolSurfaceDataset(
        surfaces[val_start:val_end],
        history_len=config.history_len,
        future_len=config.future_len,
    )
    test_dataset = VolSurfaceDataset(
        surfaces[test_start:],
        history_len=config.history_len,
        future_len=config.future_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Mode: {args.mode}, Epochs: {args.epochs}, LR: {args.lr}")

    # Train
    if args.mode == "posthoc":
        best_state, best_val = train_posthoc(
            model, config, train_loader, val_loader, sigma_head,
            args.device, args
        )
    else:
        best_state, best_val = train_frozen_nll(
            model, config, train_loader, val_loader, sigma_head,
            args.device, args
        )

    # Load best and evaluate
    sigma_head.load_state_dict(best_state)
    sigma_head.to(args.device)

    print(f"\n{'='*60}")
    print(f"Evaluation on test set")
    print(f"{'='*60}")

    eval_results = evaluate_sigma_head(model, sigma_head, test_loader, args.device)

    # Save
    save_path = f"{args.output_dir}/sigma_head.pt"
    torch.save({
        "sigma_head_state_dict": best_state,
        "model_path": args.model_path,
        "mode": args.mode,
        "hidden_dim": args.hidden_dim,
        "cond_dim": config.bottleneck_dim,
        "best_val_nll": best_val,
        "eval_results": eval_results,
        "args": vars(args),
    }, save_path)
    print(f"\nSaved to {save_path}")

    # Save eval results as JSON
    json_path = f"{args.output_dir}/eval_results.json"
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Eval results: {json_path}")


if __name__ == "__main__":
    main()
