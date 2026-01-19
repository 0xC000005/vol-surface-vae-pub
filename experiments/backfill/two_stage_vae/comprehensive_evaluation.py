"""
Comprehensive Evaluation: All 4 Cumulative-Aware VAE Options

Reports:
1. Per-grid-point CI coverage (5x5 heatmap)
2. Per-grid-point explosion statistics
3. ACF preservation analysis
4. Comparative summary

Usage:
    python experiments/backfill/two_stage_vae/comprehensive_evaluation.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import (
    CVAETwoStageCumulative,
    CVAETwoStageLevel,
    CVAETwoStageMultiTask,
    CVAETwoStageDualPathAR,
    CVAETwoStagePerGridReversion,
    CVAETwoStageSpatialMultiTask,
)


def to_log_returns(surfaces: np.ndarray):
    """Convert surfaces to log-returns."""
    log_surfaces = np.log(surfaces + 1e-8)
    log_returns = np.diff(log_surfaces, axis=0)
    return log_returns, log_surfaces


def compute_acf(series, lag=1):
    """Compute autocorrelation at given lag."""
    n = len(series)
    mean = series.mean()
    var = ((series - mean) ** 2).mean()
    if var < 1e-10:
        return 0.0
    cov = ((series[:-lag] - mean) * (series[lag:] - mean)).mean()
    return cov / var


def chain_option1(model, context, starting_iv, starting_cumul, horizon=30, n_samples=100, device="cuda"):
    """Chain with Option 1: Cumulative log-return conditioning."""
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()
            cumul = starting_cumul.copy()

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                cumul_tensor = torch.tensor(
                    cumul, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, cumul_log_ret=cumul_tensor,
                    prev_x=prev_x_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()
                all_log_returns[s, h] = log_ret

                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                cumul = cumul + log_ret
                current_iv = new_iv
                prev_log_return = log_ret

                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def chain_option2(model, context, starting_iv, horizon=30, n_samples=100, device="cuda"):
    """Chain with Option 2: Current log-IV level conditioning."""
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                current_log_iv = np.log(current_iv + 1e-8)
                log_iv_tensor = torch.tensor(
                    current_log_iv, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, current_log_iv=log_iv_tensor,
                    prev_x=prev_x_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()
                all_log_returns[s, h] = log_ret

                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                current_iv = new_iv
                prev_log_return = log_ret

                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def chain_option3(model, context, starting_iv, horizon=30, n_samples=100, device="cuda"):
    """Chain with Option 3: Sequence-trained model."""
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, prev_x=prev_x_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()
                all_log_returns[s, h] = log_ret

                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                current_iv = new_iv
                prev_log_return = log_ret

                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def chain_option4(model, context, starting_iv, starting_log_iv, horizon=30, n_samples=100, device="cuda"):
    """Chain with Option 4: Multi-task decoder."""
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_log_iv = starting_log_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                prev_log_iv_tensor = torch.tensor(
                    current_log_iv, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                return_mean, level_mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, prev_x=prev_x_tensor,
                    prev_log_iv=prev_log_iv_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()
                all_log_returns[s, h] = log_ret

                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                current_iv = new_iv
                current_log_iv = np.log(new_iv + 1e-8)
                prev_log_return = log_ret

                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def chain_option5(model, context, starting_iv, starting_cumul, horizon=30, n_samples=100, device="cuda"):
    """
    Chain with Option 5: Per-grid-point variance-scaled level reversion.

    Similar to Option 1 but with stronger reversion at OTM corners.
    """
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()
            cumul = starting_cumul.copy()

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                cumul_tensor = torch.tensor(
                    cumul, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, cumul_log_ret=cumul_tensor,
                    prev_x=prev_x_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()
                all_log_returns[s, h] = log_ret

                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                cumul = cumul + log_ret
                current_iv = new_iv
                prev_log_return = log_ret

                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def chain_option6(model, context, starting_iv, starting_log_iv, horizon=30, n_samples=100, device="cuda"):
    """
    Chain with Option 6: Spatial Multi-task decoder (CNN + Multi-task).

    Same interface as Option 4, but uses spatial smoothing to enforce correlation
    between grid points. ATM stability should propagate to OTM corners.
    """
    all_trajectories = np.zeros((n_samples, horizon, 5, 5))
    all_log_returns = np.zeros((n_samples, horizon, 5, 5))

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
        z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
        z_std = torch.exp(0.5 * z_logvar)

        for s in range(n_samples):
            current_iv = starting_iv.copy()
            current_log_iv = starting_log_iv.copy()
            current_context = context.copy()
            prev_log_return = context[-1].copy()

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                prev_log_iv_tensor = torch.tensor(
                    current_log_iv, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                return_mean, level_mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, prev_x=prev_x_tensor,
                    prev_log_iv=prev_log_iv_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()
                all_log_returns[s, h] = log_ret

                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                current_iv = new_iv
                current_log_iv = np.log(new_iv + 1e-8)
                prev_log_return = log_ret

                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def load_model(option, device):
    """Load trained model for given option."""
    model_dir = Path("models/backfill/two_stage/cumulative_options")

    if option == "1":
        path = model_dir / "cumulative_best.pt"
        model_class = CVAETwoStageCumulative
    elif option == "2":
        path = model_dir / "level_best.pt"
        model_class = CVAETwoStageLevel
    elif option == "3":
        path = model_dir / "sequence_loss_best.pt"
        model_class = CVAETwoStageDualPathAR
    elif option == "4":
        path = model_dir / "multitask_best.pt"
        model_class = CVAETwoStageMultiTask
    elif option == "5":
        path = model_dir / "per_grid_reversion_best.pt"
        model_class = CVAETwoStagePerGridReversion
    elif option == "6":
        path = model_dir / "spatial_multitask_best.pt"
        model_class = CVAETwoStageSpatialMultiTask
    else:
        raise ValueError(f"Unknown option: {option}")

    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = model_class(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def compute_grid_statistics(trajectories, log_returns, gt_surfaces, gt_log_returns):
    """Compute per-grid-point statistics."""
    n_samples, horizon, _, _ = trajectories.shape

    # Per-grid-point results
    ci_coverage = np.zeros((5, 5))
    explosion_rate = np.zeros((5, 5))
    max_iv = np.zeros((5, 5))
    acf_preservation = np.zeros((5, 5))

    # Ground truth ACF for reference
    gt_acf_grid = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            gt_acf_grid[i, j] = compute_acf(gt_log_returns[:, i, j])

    for i in range(5):
        for j in range(5):
            # Get values for this grid point
            gt_vals = gt_surfaces[:, i, j]
            traj_vals = trajectories[:, :, i, j]

            # CI coverage
            p5 = np.percentile(traj_vals, 5, axis=0)
            p95 = np.percentile(traj_vals, 95, axis=0)
            in_ci = (gt_vals >= p5) & (gt_vals <= p95)
            ci_coverage[i, j] = in_ci.mean() * 100

            # Explosion rate (any sample exceeds 2.0 or goes below 0.01)
            exploded_samples = np.any((traj_vals > 2.0) | (traj_vals < 0.01), axis=1)
            explosion_rate[i, j] = exploded_samples.mean() * 100

            # Max IV across all samples
            max_iv[i, j] = traj_vals.max()

            # ACF preservation
            model_acfs = [compute_acf(log_returns[s, :, i, j]) for s in range(n_samples)]
            model_acf_mean = np.mean(model_acfs)
            gt_acf = gt_acf_grid[i, j]
            if abs(gt_acf) > 1e-6:
                # Preservation: how much of the ACF magnitude is captured
                # Positive if same sign and similar magnitude
                if gt_acf * model_acf_mean > 0:  # Same sign
                    acf_preservation[i, j] = min(abs(model_acf_mean / gt_acf), 1.5) * 100
                else:  # Wrong sign
                    acf_preservation[i, j] = -abs(model_acf_mean / gt_acf) * 100

    return {
        "ci_coverage": ci_coverage,
        "explosion_rate": explosion_rate,
        "max_iv": max_iv,
        "acf_preservation": acf_preservation,
        "gt_acf_grid": gt_acf_grid,
    }


def print_grid(grid, title, fmt=".1f"):
    """Print a 5x5 grid with nice formatting."""
    print(f"\n{title}")
    print("-" * 50)

    # Column labels (moneyness)
    print("         ", end="")
    for j in range(5):
        print(f"M{j:>6}", end=" ")
    print()

    for i in range(5):
        print(f"T{i:>7}", end=" ")
        for j in range(5):
            val = grid[i, j]
            if np.isinf(val) or np.isnan(val):
                print(f"{'inf':>7}", end=" ")
            else:
                print(f"{val:>7{fmt}}", end=" ")
        print()


def main():
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION: All 6 Cumulative-Aware VAE Options")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup test data - use multiple starting points for robustness
    context_len = 20
    horizon = 30
    n_samples = 100
    train_end = int(len(surfaces) * 0.7)

    # Test on multiple sequences
    n_test_sequences = 5
    test_start_indices = [train_end + 100 + i * 50 for i in range(n_test_sequences)]

    print(f"\nTest setup:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon: {horizon}")
    print(f"  N samples per sequence: {n_samples}")
    print(f"  N test sequences: {n_test_sequences}")

    # Results storage
    all_results = {}

    for opt in ["1", "2", "3", "4", "5", "6"]:
        print(f"\n{'#' * 70}")
        print(f"Testing Option {opt}")
        print("#" * 70)

        model = load_model(opt, device)
        if model is None:
            print(f"Model for option {opt} not found, skipping...")
            continue

        # Aggregate results across test sequences
        all_ci_coverage = []
        all_explosion_rate = []
        all_max_iv = []
        all_acf_preservation = []
        n_exploded_total = 0
        n_total = 0

        for seq_idx, start_idx in enumerate(test_start_indices):
            # Prepare context
            context = log_returns[start_idx:start_idx + context_len]
            starting_iv = surfaces[start_idx + context_len - 1]
            starting_log_iv = log_surfaces[start_idx + context_len - 1]
            starting_cumul = np.sum(context, axis=0)

            gt_surfaces_seq = surfaces[start_idx + context_len:start_idx + context_len + horizon]
            gt_log_returns_seq = log_returns[start_idx + context_len:start_idx + context_len + horizon]

            # Chain based on option
            if opt == "1":
                trajectories, log_rets = chain_option1(
                    model, context, starting_iv, starting_cumul,
                    horizon=horizon, n_samples=n_samples, device=device
                )
            elif opt == "2":
                trajectories, log_rets = chain_option2(
                    model, context, starting_iv,
                    horizon=horizon, n_samples=n_samples, device=device
                )
            elif opt == "3":
                trajectories, log_rets = chain_option3(
                    model, context, starting_iv,
                    horizon=horizon, n_samples=n_samples, device=device
                )
            elif opt == "4":
                trajectories, log_rets = chain_option4(
                    model, context, starting_iv, starting_log_iv,
                    horizon=horizon, n_samples=n_samples, device=device
                )
            elif opt == "5":
                trajectories, log_rets = chain_option5(
                    model, context, starting_iv, starting_cumul,
                    horizon=horizon, n_samples=n_samples, device=device
                )
            elif opt == "6":
                trajectories, log_rets = chain_option6(
                    model, context, starting_iv, starting_log_iv,
                    horizon=horizon, n_samples=n_samples, device=device
                )

            # Compute statistics
            stats = compute_grid_statistics(trajectories, log_rets, gt_surfaces_seq, gt_log_returns_seq)
            all_ci_coverage.append(stats["ci_coverage"])
            all_explosion_rate.append(stats["explosion_rate"])
            all_max_iv.append(stats["max_iv"])
            all_acf_preservation.append(stats["acf_preservation"])

            # Count explosions
            exploded = np.any(trajectories > 2.0) or np.any(trajectories < 0.01)
            if exploded:
                n_exploded_total += 1
            n_total += 1

        # Average across sequences
        avg_ci_coverage = np.mean(all_ci_coverage, axis=0)
        avg_explosion_rate = np.mean(all_explosion_rate, axis=0)
        avg_max_iv = np.max(all_max_iv, axis=0)  # Max across sequences
        avg_acf_preservation = np.mean(all_acf_preservation, axis=0)

        all_results[opt] = {
            "ci_coverage": avg_ci_coverage,
            "explosion_rate": avg_explosion_rate,
            "max_iv": avg_max_iv,
            "acf_preservation": avg_acf_preservation,
            "n_exploded": n_exploded_total,
            "n_total": n_total,
        }

        # Print detailed results
        print(f"\n{'=' * 60}")
        print(f"Option {opt} Results (averaged over {n_total} sequences)")
        print("=" * 60)

        print(f"\nSequence Explosion Rate: {n_exploded_total}/{n_total} ({n_exploded_total/n_total*100:.1f}%)")

        print_grid(avg_ci_coverage, "CI Coverage (%) - Target: 90%")
        print(f"  Mean CI Coverage: {avg_ci_coverage.mean():.1f}%")

        print_grid(avg_explosion_rate, "Explosion Rate (%) - Target: 0%")
        print(f"  Mean Explosion Rate: {avg_explosion_rate.mean():.1f}%")

        print_grid(avg_max_iv, "Max IV - Target: < 2.0", fmt=".3f")
        print(f"  Overall Max IV: {avg_max_iv.max():.4f}")

        print_grid(avg_acf_preservation, "ACF Preservation (%) - Target: > 30%")
        print(f"  Mean ACF Preservation: {avg_acf_preservation.mean():.1f}%")

    # Comparative summary
    print("\n" + "=" * 70)
    print("COMPARATIVE SUMMARY")
    print("=" * 70)

    print(f"\n{'Option':<8} {'Exploded':<12} {'Mean CI%':<12} {'Mean Expl%':<12} {'Max IV':<12} {'ACF%':<10}")
    print("-" * 70)

    for opt, res in all_results.items():
        print(f"{opt:<8} {res['n_exploded']}/{res['n_total']:<9} "
              f"{res['ci_coverage'].mean():<12.1f} "
              f"{res['explosion_rate'].mean():<12.1f} "
              f"{res['max_iv'].max():<12.4f} "
              f"{res['acf_preservation'].mean():<10.1f}")

    # Explanation: Why ACF preservation doesn't prevent explosion
    print("\n" + "=" * 70)
    print("WHY ACF PRESERVATION DOESN'T PREVENT EXPLOSION")
    print("=" * 70)

    explanation = """
KEY INSIGHT: ACF operates in LOG-RETURN space, not LEVEL space.

Mathematical Explanation:
-------------------------
Let r_t = log(IV_t / IV_{t-1}) be the log-return at time t.

AR(1) with negative ACF gives:
    E[r_t | r_{t-1}] = phi * r_{t-1},  where phi ≈ -0.35

This means returns alternate signs (mean-reverting in return space):
    If r_{t-1} = +0.4, then E[r_t] = -0.14

But the CUMULATIVE log-return (which determines IV level) is:
    sum(r_1, ..., r_T) = cumulative drift

Example trajectory with phi = -0.35:
------------------------------------
Day 1: r = +0.40, IV = 0.15 * exp(0.40) = 0.224
Day 2: r = -0.14, IV = 0.224 * exp(-0.14) = 0.195
Day 3: r = +0.30, IV = 0.195 * exp(0.30) = 0.263
Day 4: r = -0.11, IV = 0.263 * exp(-0.11) = 0.236
Day 5: r = +0.35, IV = 0.236 * exp(0.35) = 0.334

Returns alternate (ACF preserved!), but IV drifts: 0.15 -> 0.33 (+120%)

The Root Cause:
---------------
1. Partial reversion (phi = -0.35) only reverses 35% of each shock
2. Remaining 65% of each shock accumulates
3. After T steps, cumulative drift ~ sqrt(T) * sigma * sqrt(1 - |phi|^2)
4. This grows without bound as T increases

What Would Prevent Explosion:
-----------------------------
1. phi = -1.0: Perfect reversal (returns are anti-persistent walk)
2. Level-dependent reversion: Return magnitude depends on how far IV is from mean
3. Hard bounds on cumulative log-return

The bitter lesson: ACF preservation is NECESSARY but NOT SUFFICIENT for stable chaining.
The model needs to understand CUMULATIVE effects, not just local return dynamics.
"""
    print(explanation)


if __name__ == "__main__":
    main()
