"""
Test Cumulative-Aware VAE Chaining (No Clipping!)

This script tests the trained cumulative-aware models to verify they don't explode
during autoregressive chaining WITHOUT any clipping or temperature scaling.

Success criteria:
- No explosion (max IV < 2.0) without clipping
- ACF preservation > 30%
- CI coverage > 80%

Usage:
    python experiments/backfill/two_stage_vae/test_cumulative_chaining.py --option 1
    python experiments/backfill/two_stage_vae/test_cumulative_chaining.py --option all
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from vae.cvae_two_stage import (
    CVAETwoStageCumulative,
    CVAETwoStageLevel,
    CVAETwoStageMultiTask,
    CVAETwoStageDualPathAR,
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


def chain_option1(model, context, starting_iv, starting_cumul, horizon=30, n_samples=50, device="cuda"):
    """
    Chain with Option 1: Cumulative log-return conditioning.

    NO CLIPPING - let the model handle stability through cumul_log_ret.
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
            cumul = starting_cumul.copy()  # Cumulative log-return for all grid points

            for h in range(horizon):
                # Sample z
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                # Prepare inputs
                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                cumul_tensor = torch.tensor(
                    cumul, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                ctx_emb_step = ctx_emb[:, -1:, :]
                z_step = z[:, -1:, :]

                # Decode with cumulative conditioning (NO CLIPPING!)
                mean, sample, _, _ = model.decoder(
                    ctx_emb_step, z_step, cumul_log_ret=cumul_tensor,
                    prev_x=prev_x_tensor, sample=True
                )

                log_ret = sample[0, 0].cpu().numpy()

                # Store
                all_log_returns[s, h] = log_ret

                # Update IV
                new_iv = current_iv * np.exp(log_ret)
                all_trajectories[s, h] = new_iv

                # Update cumulative
                cumul = cumul + log_ret

                # Update for next iteration
                current_iv = new_iv
                prev_log_return = log_ret

                # Update context
                current_context = np.concatenate(
                    [current_context[1:], log_ret[np.newaxis]], axis=0
                )

                # Recompute encodings
                ctx_tensor = torch.tensor(
                    current_context, dtype=torch.float32
                ).unsqueeze(0).to(device)
                ctx_emb = model.ctx_encoder({"surface": ctx_tensor})
                z_mean, z_logvar, _ = model.main_encoder({"surface": ctx_tensor})
                z_std = torch.exp(0.5 * z_logvar)

    return all_trajectories, all_log_returns


def chain_option2(model, context, starting_iv, horizon=30, n_samples=50, device="cuda"):
    """
    Chain with Option 2: Current log-IV level conditioning.

    NO CLIPPING - let the model handle stability through current_log_iv.
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

            for h in range(horizon):
                eps = torch.randn_like(z_std)
                z = z_mean + z_std * eps

                prev_x_tensor = torch.tensor(
                    prev_log_return, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)

                # Current log-IV level
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


def chain_option3(model, context, starting_iv, horizon=30, n_samples=50, device="cuda"):
    """
    Chain with Option 3: Sequence-trained model.

    Uses base AR model trained with sequence loss.
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


def chain_option4(model, context, starting_iv, starting_log_iv, horizon=30, n_samples=50, device="cuda"):
    """
    Chain with Option 4: Multi-task decoder.

    Uses both return prediction and level prediction.
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


def evaluate_chaining(trajectories, log_returns, gt_surfaces, gt_log_returns, option_name):
    """Evaluate chaining results."""
    print(f"\n{'=' * 70}")
    print(f"Evaluation: {option_name}")
    print("=" * 70)

    horizon = trajectories.shape[1]
    n_samples = trajectories.shape[0]

    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  IV range: {trajectories.min():.4f} - {trajectories.max():.4f}")
    print(f"  GT range: {gt_surfaces.min():.4f} - {gt_surfaces.max():.4f}")

    # Check for explosion
    max_iv = trajectories.max()
    exploded = max_iv > 2.0
    print(f"  Explosion (max IV > 2.0): {'YES' if exploded else 'NO'} (max={max_iv:.4f})")

    # ACF analysis
    gt_acf = compute_acf(gt_log_returns[:, 2, 2])
    model_acfs = [compute_acf(log_returns[s, :, 2, 2]) for s in range(n_samples)]
    model_acf_mean = np.mean(model_acfs)
    acf_preservation = abs(model_acf_mean) / abs(gt_acf) * 100 if abs(gt_acf) > 1e-6 else 0

    print(f"\nACF Analysis (ATM):")
    print(f"  GT ACF(1): {gt_acf:.4f}")
    print(f"  Model ACF(1): {model_acf_mean:.4f}")
    print(f"  Preservation: {acf_preservation:.1f}%")

    # CI coverage
    print(f"\nCI Coverage (90% target):")
    grid_points = [((2, 2), "ATM"), ((0, 0), "OTM Put"), ((4, 4), "OTM Call")]

    coverages = []
    for (i, j), name in grid_points:
        gt_vals = gt_surfaces[:, i, j]
        traj_vals = trajectories[:, :, i, j]
        p5 = np.percentile(traj_vals, 5, axis=0)
        p95 = np.percentile(traj_vals, 95, axis=0)

        in_ci = (gt_vals >= p5) & (gt_vals <= p95)
        coverage = in_ci.mean() * 100
        coverages.append(coverage)
        print(f"  {name}: {coverage:.1f}%")

    mean_coverage = np.mean(coverages)
    print(f"  Mean coverage: {mean_coverage:.1f}%")

    # Summary
    print(f"\nSUMMARY:")
    success = not exploded and acf_preservation > 30 and mean_coverage > 80
    print(f"  - No explosion: {'PASS' if not exploded else 'FAIL'}")
    print(f"  - ACF > 30%: {'PASS' if acf_preservation > 30 else 'FAIL'} ({acf_preservation:.1f}%)")
    print(f"  - CI > 80%: {'PASS' if mean_coverage > 80 else 'FAIL'} ({mean_coverage:.1f}%)")
    print(f"  - Overall: {'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")

    return {
        "max_iv": max_iv,
        "exploded": exploded,
        "acf_preservation": acf_preservation,
        "mean_coverage": mean_coverage,
        "success": success,
    }


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
    else:
        raise ValueError(f"Unknown option: {option}")

    if not path.exists():
        print(f"Model not found: {path}")
        print(f"Please train option {option} first using train_cumulative_options.py")
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    config["device"] = device

    model = model_class(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {path}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Test cumulative-aware VAE chaining")
    parser.add_argument("--option", type=str, default="all",
                        help="Which option to test: 1, 2, 3, 4, or all")
    parser.add_argument("--horizon", type=int, default=30, help="Chaining horizon")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples")
    args = parser.parse_args()

    print("=" * 70)
    print("Cumulative-Aware VAE Chaining Test (NO CLIPPING!)")
    print("=" * 70)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    log_returns, log_surfaces = to_log_returns(surfaces)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup test data
    context_len = 20
    train_end = int(len(surfaces) * 0.7)
    start_idx = train_end + 100  # Test set

    context = log_returns[start_idx:start_idx + context_len]
    starting_iv = surfaces[start_idx + context_len - 1]
    starting_log_iv = log_surfaces[start_idx + context_len - 1]
    starting_cumul = np.sum(context, axis=0)  # Cumulative log-return

    gt_surfaces = surfaces[start_idx + context_len:start_idx + context_len + args.horizon]
    gt_log_returns = log_returns[start_idx + context_len:start_idx + context_len + args.horizon]

    print(f"\nTest setup:")
    print(f"  Context length: {context_len}")
    print(f"  Horizon: {args.horizon}")
    print(f"  N samples: {args.n_samples}")
    print(f"  Starting IV (ATM): {starting_iv[2, 2]:.4f}")

    # Test requested options
    options = [args.option] if args.option != "all" else ["1", "2", "3", "4"]
    results = {}

    for opt in options:
        print(f"\n{'#' * 70}")
        print(f"Testing Option {opt}")
        print("#" * 70)

        model = load_model(opt, device)
        if model is None:
            continue

        # Chain based on option
        if opt == "1":
            trajectories, log_rets = chain_option1(
                model, context, starting_iv, starting_cumul,
                horizon=args.horizon, n_samples=args.n_samples, device=device
            )
        elif opt == "2":
            trajectories, log_rets = chain_option2(
                model, context, starting_iv,
                horizon=args.horizon, n_samples=args.n_samples, device=device
            )
        elif opt == "3":
            trajectories, log_rets = chain_option3(
                model, context, starting_iv,
                horizon=args.horizon, n_samples=args.n_samples, device=device
            )
        elif opt == "4":
            trajectories, log_rets = chain_option4(
                model, context, starting_iv, starting_log_iv,
                horizon=args.horizon, n_samples=args.n_samples, device=device
            )

        # Evaluate
        results[opt] = evaluate_chaining(
            trajectories, log_rets, gt_surfaces, gt_log_returns,
            f"Option {opt}"
        )

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Option':<10} {'Max IV':<10} {'Exploded':<10} {'ACF %':<10} {'CI %':<10} {'Success':<10}")
    print("-" * 60)

    for opt, res in results.items():
        print(f"{opt:<10} {res['max_iv']:<10.4f} {'YES' if res['exploded'] else 'NO':<10} "
              f"{res['acf_preservation']:<10.1f} {res['mean_coverage']:<10.1f} "
              f"{'YES' if res['success'] else 'NO':<10}")


if __name__ == "__main__":
    main()
