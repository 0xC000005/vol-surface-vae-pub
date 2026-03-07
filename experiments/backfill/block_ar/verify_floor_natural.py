#!/usr/bin/env python
"""Verify whether floor_clamp=0.01 is doing mechanical work or GRU naturally stays above.

Checks:
1. Distribution of per-path minimum IV (if spike at 0.01, clamp is doing work)
2. How often any cell is within epsilon of the floor (0.01-0.015 range)
3. Delta distribution when IV is near the floor (are deltas trying to go lower?)
4. Compare with floor=0.001 to show the cascade effect

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/verify_floor_natural.py --device cuda
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def load_model(path, device, floor_clamp=0.01, freeze_gru_state=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", ckpt.get("model_config", {}))
    if isinstance(cfg_dict, dict):
        cfg = SinglePassConfig(**{k: v for k, v in cfg_dict.items() if hasattr(SinglePassConfig, k)})
    else:
        cfg = cfg_dict
    cfg.ar_frame_floor_clamp = floor_clamp
    cfg.ar_freeze_gru_state = freeze_gru_state
    model = SinglePassBlockAR(cfg)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def generate_trajectories(model, dataset, device, n_windows=30, n_samples=50, n_frames=252):
    """Generate long trajectories, return (n_total, T, 5, 5) in [0,1] scale."""
    all_trajs = []
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), min(n_windows, len(dataset)), replace=False)
    for i, idx in enumerate(indices):
        item = dataset[idx]
        hist = item["history"]
        hist_t = hist.unsqueeze(0).to(device) if isinstance(hist, torch.Tensor) else torch.from_numpy(hist).unsqueeze(0).to(device)
        with torch.no_grad():
            samples = model.sample(hist_t, n_samples=n_samples, n_frames=n_frames)
        all_trajs.append(samples.cpu().numpy()[0])  # (S, T, 5, 5)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(indices)}")
    return np.concatenate(all_trajs, axis=0)  # (N, T, 5, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_windows", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--n_frames", type=int, default=252)
    parser.add_argument("--output_dir", default="results/block_ar/verify_floor")
    parser.add_argument("--freeze_gru_state", action="store_true",
                        help="Freeze GRU state during generation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=4040, end_idx=len(surfaces))
    model_path = "models/backfill/afcrps_90d/best_model.pt"

    freeze = args.freeze_gru_state
    if freeze:
        print("*** GRU state FROZEN: using initial condition for all frames ***")

    # Generate with floor=0.01
    print("=== Generating with floor=0.01 ===")
    model_01 = load_model(model_path, device, floor_clamp=0.01, freeze_gru_state=freeze)
    trajs_01 = generate_trajectories(model_01, dataset, device, args.n_windows, args.n_samples, args.n_frames)
    del model_01; torch.cuda.empty_cache()

    # Generate with floor=0.001
    print("\n=== Generating with floor=0.001 ===")
    model_001 = load_model(model_path, device, floor_clamp=0.001, freeze_gru_state=freeze)
    trajs_001 = generate_trajectories(model_001, dataset, device, args.n_windows, args.n_samples, args.n_frames)
    del model_001; torch.cuda.empty_cache()

    # Analysis
    print("\n" + "="*70)
    print("VERIFICATION: Is floor=0.01 doing mechanical work?")
    print("="*70)

    # 1. Per-path minimum IV
    min_iv_01 = trajs_01.reshape(trajs_01.shape[0], -1).min(axis=1)   # (N,)
    min_iv_001 = trajs_001.reshape(trajs_001.shape[0], -1).min(axis=1)

    print(f"\n1. Per-path minimum IV over {args.n_frames} days:")
    for label, mins in [("floor=0.01", min_iv_01), ("floor=0.001", min_iv_001)]:
        print(f"  {label}:")
        print(f"    mean={mins.mean():.4f}, median={np.median(mins):.4f}")
        print(f"    p5={np.percentile(mins,5):.4f}, p1={np.percentile(mins,1):.4f}, min={mins.min():.4f}")
        at_floor = (mins <= 0.0101).mean() * 100 if "0.01" in label else (mins <= 0.0011).mean() * 100
        print(f"    paths touching floor: {at_floor:.1f}%")

    # 2. Per-frame: fraction of cells within epsilon of floor
    print(f"\n2. Fraction of cell-frames near floor (within 0.005):")
    for label, trajs, floor in [("floor=0.01", trajs_01, 0.01), ("floor=0.001", trajs_001, 0.001)]:
        near_floor = (trajs < floor + 0.005).mean() * 100
        at_floor = (trajs <= floor + 0.0001).mean() * 100
        print(f"  {label}: near_floor(<{floor+0.005:.3f})={near_floor:.3f}%, at_floor={at_floor:.3f}%")

    # 3. Per-horizon minimum IV percentiles
    print(f"\n3. Minimum IV across cells at key horizons (p5 of paths):")
    for h in [1, 7, 30, 60, 120, 252]:
        if h <= args.n_frames:
            min_at_h_01 = trajs_01[:, :h].reshape(trajs_01.shape[0], -1).min(axis=1)
            min_at_h_001 = trajs_001[:, :h].reshape(trajs_001.shape[0], -1).min(axis=1)
            print(f"  h={h:3d}: floor=0.01 p5={np.percentile(min_at_h_01,5):.4f} "
                  f"min={min_at_h_01.min():.4f}  |  "
                  f"floor=0.001 p5={np.percentile(min_at_h_001,5):.4f} "
                  f"min={min_at_h_001.min():.4f}")

    # 4. Delta distribution when near floor
    print(f"\n4. Delta when cell IV is in lowest decile:")
    for label, trajs, floor in [("floor=0.01", trajs_01, 0.01), ("floor=0.001", trajs_001, 0.001)]:
        deltas = np.diff(trajs, axis=1)  # (N, T-1, 5, 5)
        prev_frames = trajs[:, :-1]       # (N, T-1, 5, 5)
        low_thresh = np.percentile(prev_frames, 10)
        mask = prev_frames < low_thresh
        low_deltas = deltas[mask]
        if len(low_deltas) > 0:
            print(f"  {label} (thresh={low_thresh:.4f}, n={len(low_deltas)}):")
            print(f"    delta mean={low_deltas.mean():.6f}, median={np.median(low_deltas):.6f}")
            print(f"    negative deltas: {(low_deltas < 0).mean()*100:.1f}%")
            print(f"    delta < -0.005: {(low_deltas < -0.005).mean()*100:.1f}%")

    # 5. Per-cell drift analysis
    print(f"\n5. Per-cell cumulative drift over {args.n_frames} days:")
    for label, trajs in [("floor=0.01", trajs_01), ("floor=0.001", trajs_001)]:
        daily_deltas = np.diff(trajs, axis=1)  # (N, T-1, 5, 5)
        mean_daily = daily_deltas.mean(axis=(0, 1))  # (5, 5)
        cum_drift = mean_daily * args.n_frames
        print(f"  {label} -- Mean daily delta per cell (x1e4):")
        for r in range(5):
            print("    " + " ".join(f"{mean_daily[r,c]*1e4:+6.2f}" for c in range(5)))
        print(f"  Cumulative {args.n_frames}d drift range: "
              f"[{cum_drift.min()*100:+.2f}, {cum_drift.max()*100:+.2f}] IV pts")
        print(f"  Drift spread (max-min): {(cum_drift.max()-cum_drift.min())*100:.2f} IV pts")

    # 6. Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 5a. Histogram of per-path minimum IV
    axes[0,0].hist(min_iv_01, bins=100, alpha=0.7, label='floor=0.01', density=True)
    axes[0,0].hist(min_iv_001, bins=100, alpha=0.7, label='floor=0.001', density=True)
    axes[0,0].set_xlabel('Minimum IV across path')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Per-path minimum IV (252 days)')
    axes[0,0].legend()
    axes[0,0].axvline(0.01, color='red', ls='--', label='floor=0.01')

    # 5b. Min IV percentile over horizon
    horizons = list(range(1, args.n_frames+1))
    p5_01 = [np.percentile(trajs_01[:, :h].reshape(trajs_01.shape[0], -1).min(axis=1), 5) for h in horizons]
    p5_001 = [np.percentile(trajs_001[:, :h].reshape(trajs_001.shape[0], -1).min(axis=1), 5) for h in horizons]
    axes[0,1].plot(horizons, p5_01, label='floor=0.01 (p5)')
    axes[0,1].plot(horizons, p5_001, label='floor=0.001 (p5)')
    axes[0,1].axhline(0.01, color='red', ls='--', alpha=0.5)
    axes[0,1].set_xlabel('Horizon')
    axes[0,1].set_ylabel('Min IV (p5 of paths)')
    axes[0,1].set_title('5th percentile of path-minimum IV vs horizon')
    axes[0,1].legend()

    # 5c. Fraction of cells near floor over horizon
    near_01 = [(trajs_01[:, :h] < 0.015).mean() * 100 for h in horizons]
    near_001 = [(trajs_001[:, :h] < 0.006).mean() * 100 for h in horizons]
    axes[1,0].plot(horizons, near_01, label='floor=0.01: IV<0.015')
    axes[1,0].plot(horizons, near_001, label='floor=0.001: IV<0.006')
    axes[1,0].set_xlabel('Horizon')
    axes[1,0].set_ylabel('% cell-frames')
    axes[1,0].set_title('Fraction of cells near floor vs horizon')
    axes[1,0].legend()

    # 5d. Example paths for lowest-IV cell
    cell_means_01 = trajs_01[:, :, :, :].reshape(trajs_01.shape[0], trajs_01.shape[1], 25).min(axis=2)  # (N, T)
    for i in range(min(20, len(cell_means_01))):
        axes[1,1].plot(cell_means_01[i], alpha=0.3, color='blue', linewidth=0.5)
    cell_means_001 = trajs_001[:, :, :, :].reshape(trajs_001.shape[0], trajs_001.shape[1], 25).min(axis=2)
    for i in range(min(20, len(cell_means_001))):
        axes[1,1].plot(cell_means_001[i], alpha=0.3, color='red', linewidth=0.5)
    axes[1,1].axhline(0.01, color='black', ls='--')
    axes[1,1].set_xlabel('Day')
    axes[1,1].set_ylabel('Min cell IV')
    axes[1,1].set_title('Sample paths: min-cell IV (blue=0.01, red=0.001)')

    plt.tight_layout()
    plt.savefig(outdir / "floor_verification.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {outdir / 'floor_verification.png'}")


if __name__ == "__main__":
    main()
