"""
Kurtosis diagnostic: evaluate kurtosis ratio across epochs and per grid cell.

Experiments 1.1 (kurtosis vs epoch) and 1.2 (per-cell kurtosis).
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.stats import kurtosis

from diffusion.block_ar.block_ar_ddpm import BlockARConfig, ConditionalBlockARDDPM, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset
from torch.utils.data import DataLoader


def generate_samples(model, dataloader, n_samples=10, max_batches=10, device="cuda"):
    """Generate conditional samples and collect ground truth."""
    model.eval()
    all_gt = []
    all_gen = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_gt = batch["future"].to(device)

            future_gt_denorm = denormalize_iv(future_gt)
            samples = model.sample_batched(history, n_samples=n_samples)

            all_gt.append(future_gt_denorm.cpu().numpy())
            all_gen.append(samples.cpu().numpy())

            print(f"  Batch {batch_idx+1}/{max_batches}")

    gt = np.concatenate(all_gt, axis=0)       # (N, T, 5, 5)
    gen = np.concatenate(all_gen, axis=0)      # (N, n_samples, T, 5, 5)
    return gt, gen


def compute_kurtosis_pooled(gt, gen):
    """Compute pooled kurtosis ratio (current metric)."""
    gt_diff = np.diff(gt, axis=1)
    gen_diff = np.diff(gen[:, 0], axis=1)  # sample[0]

    gt_kurt = kurtosis(gt_diff.flatten(), fisher=True)
    gen_kurt = kurtosis(gen_diff.flatten(), fisher=True)

    return gen_kurt / gt_kurt if gt_kurt != 0 else 0.0, gt_kurt, gen_kurt


def compute_kurtosis_per_cell(gt, gen):
    """Compute kurtosis ratio per (tenor, moneyness) cell."""
    gt_diff = np.diff(gt, axis=1)       # (N, T-1, 5, 5)
    gen_diff = np.diff(gen[:, 0], axis=1)  # (N, T-1, 5, 5)

    ratios = np.zeros((5, 5))
    gt_kurts = np.zeros((5, 5))
    gen_kurts = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            gt_k = kurtosis(gt_diff[:, :, i, j].flatten(), fisher=True)
            gen_k = kurtosis(gen_diff[:, :, i, j].flatten(), fisher=True)
            gt_kurts[i, j] = gt_k
            gen_kurts[i, j] = gen_k
            ratios[i, j] = gen_k / gt_k if gt_k != 0 else 0.0

    return ratios, gt_kurts, gen_kurts


def compute_ensemble_kurtosis(gt, gen):
    """Compute kurtosis using all samples (inter-sample variation)."""
    # For each (batch, horizon, cell), compute std across samples
    # Then compute kurtosis of these stds
    sample_stds = gen.std(axis=1)  # (N, T, 5, 5)
    gt_diff = np.diff(gt, axis=1)

    # Kurtosis of inter-sample standard deviations
    ensemble_kurt = kurtosis(sample_stds.flatten(), fisher=True)

    # Also: kurtosis across ALL samples, not just sample[0]
    n_samples = gen.shape[1]
    all_gen_diff = np.diff(gen, axis=2)  # (N, n_samples, T-1, 5, 5)
    all_gen_diff_flat = all_gen_diff.reshape(-1)
    gt_diff_flat = gt_diff.flatten()

    gt_kurt = kurtosis(gt_diff_flat, fisher=True)
    all_samples_kurt = kurtosis(all_gen_diff_flat, fisher=True)
    single_sample_kurt = kurtosis(np.diff(gen[:, 0], axis=1).flatten(), fisher=True)

    return {
        "gt_kurtosis": gt_kurt,
        "single_sample_ratio": single_sample_kurt / gt_kurt,
        "all_samples_ratio": all_samples_kurt / gt_kurt,
        "ensemble_std_kurtosis": ensemble_kurt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="models/backfill/block_ar_dual_path")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_ema", action="store_true", default=True)
    parser.add_argument("--per_cell", action="store_true", help="Run per-cell kurtosis analysis")
    parser.add_argument("--ensemble", action="store_true", help="Run ensemble kurtosis analysis")
    parser.add_argument("--epochs", type=str, default="10,20,30,40,50",
                        help="Comma-separated epochs to evaluate")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data_path)
    surfaces = data["surface"]
    test_dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4540)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    epochs = [int(e) for e in args.epochs.split(",")]

    print("=" * 70)
    print("KURTOSIS DIAGNOSTIC")
    print("=" * 70)

    # === Experiment 1.1: Kurtosis vs Epoch ===
    print("\n--- Experiment 1.1: Kurtosis vs Epoch ---\n")

    results = {}
    for epoch in epochs:
        ckpt_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        if not ckpt_path.exists():
            print(f"  Epoch {epoch}: checkpoint not found, skipping")
            continue

        print(f"Loading epoch {epoch}...")
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=args.device)

        config = checkpoint["config"]
        if isinstance(config, dict):
            config = BlockARConfig(**config)

        model = ConditionalBlockARDDPM(config).to(args.device)

        if args.no_ema:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            ema = checkpoint.get("ema_params", {})
            if ema:
                state = model.state_dict()
                for name in ema:
                    if name in state:
                        state[name] = ema[name]
                model.load_state_dict(state)

        gt, gen = generate_samples(model, test_loader, args.n_samples, args.max_batches, args.device)
        ratio, gt_k, gen_k = compute_kurtosis_pooled(gt, gen)

        results[epoch] = {"ratio": ratio, "gt_kurt": gt_k, "gen_kurt": gen_k}
        print(f"  Epoch {epoch}: kurtosis ratio = {ratio:.4f} (GT={gt_k:.1f}, Gen={gen_k:.1f})")

        # Per-cell and ensemble only for best epoch
        if args.per_cell and epoch == 10:
            print(f"\n--- Experiment 1.2: Per-Cell Kurtosis (epoch {epoch}) ---\n")
            cell_ratios, gt_kurts, gen_kurts = compute_kurtosis_per_cell(gt, gen)

            tenor_labels = ["1mo", "2mo", "4mo", "8mo", "12mo"]
            money_labels = ["deep ITM", "ITM", "ATM", "OTM", "deep OTM"]

            print("  Kurtosis Ratio per Cell (rows=tenor, cols=moneyness):")
            print(f"  {'':>10s}", end="")
            for ml in money_labels:
                print(f"  {ml:>8s}", end="")
            print()
            for i, tl in enumerate(tenor_labels):
                print(f"  {tl:>10s}", end="")
                for j in range(5):
                    r = cell_ratios[i, j]
                    marker = " *" if 0.5 <= r <= 2.0 else "  "
                    print(f"  {r:>6.3f}{marker}", end="")
                print()

            print(f"\n  Cells passing (0.5-2.0): {np.sum((cell_ratios >= 0.5) & (cell_ratios <= 2.0))}/25")
            print(f"  Mean ratio: {cell_ratios.mean():.4f}")
            print(f"  Min ratio: {cell_ratios.min():.4f} at {np.unravel_index(cell_ratios.argmin(), (5,5))}")
            print(f"  Max ratio: {cell_ratios.max():.4f} at {np.unravel_index(cell_ratios.argmax(), (5,5))}")

        if args.ensemble and epoch == 10:
            print(f"\n--- Experiment 1.3: Ensemble Kurtosis (epoch {epoch}) ---\n")
            ens = compute_ensemble_kurtosis(gt, gen)
            print(f"  GT kurtosis: {ens['gt_kurtosis']:.1f}")
            print(f"  Single sample[0] ratio: {ens['single_sample_ratio']:.4f}")
            print(f"  All {args.n_samples} samples ratio: {ens['all_samples_ratio']:.4f}")
            print(f"  Ensemble std kurtosis: {ens['ensemble_std_kurtosis']:.2f}")

        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n--- Summary: Kurtosis vs Epoch ---\n")
    print(f"  {'Epoch':>6s}  {'Ratio':>8s}  {'GT Kurt':>10s}  {'Gen Kurt':>10s}  {'Status':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for epoch in sorted(results.keys()):
        r = results[epoch]
        status = "PASS" if 0.5 <= r["ratio"] <= 2.0 else "FAIL"
        print(f"  {epoch:6d}  {r['ratio']:8.4f}  {r['gt_kurt']:10.1f}  {r['gen_kurt']:10.1f}  {status:>8s}")


if __name__ == "__main__":
    main()
