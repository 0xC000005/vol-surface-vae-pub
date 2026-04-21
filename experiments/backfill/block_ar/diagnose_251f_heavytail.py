"""
251f heavy-tail diagnostic — 3-level causal chain for H2.

L1 Mechanism activation:
  - Learned nu (softplus(nu_raw) + 2)
  - Empirical kurtosis of z_0 samples — compare to theoretical Student-t kurtosis

L2 Causal-chain propagation:
  - Per-horizon kurtosis(z_t) for t in {0, 5, 10, 15, 20, 25, 29}
  - CLT smoothing ratio kurt(z_29) / kurt(z_0); gate >= 0.3
  - Surface-level kurtosis kurt(factor=Lambda*z_t) — if Lambda Gaussianizes the
    projected heavy tails, per-cell surface tails stay Gaussian.

L3 Target metric:
  - Cross-reference with eval's time_series.kurtosis_ratio and pathwise_jump_realism.

Gate:
  - L1: nu bounded away from infinity (softplus output finite), kurt(z_0) > 5
  - L2: kurt(z_29)/kurt(z_0) >= 0.3 AND kurt(factor_t) / kurt(z_t) >= 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import load_model
from diffusion.block_ar.single_pass_ar import normalize_iv


def excess_kurtosis(x: torch.Tensor) -> float:
    x32 = x.float().reshape(-1)
    m = x32.mean()
    s = x32.std().clamp(min=1e-12)
    return float(((x32 - m) ** 4).mean().item() / (s ** 4).item() - 3.0)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n_windows", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    model.eval()

    if model.latent_fm is None:
        print("ERROR: model has no LatentFM")
        return 2

    nu_raw = model.latent_fm.nu_raw
    if nu_raw is None:
        print("NOTE: latent_fm.nu_raw is None — model is Gaussian baseline")
        nu = float("inf")
    else:
        nu = float(2.0 + F.softplus(nu_raw).item())
    print(f"Learned nu = {nu:.3f}")
    if nu > 4:
        theo_kurt = 6.0 / (nu - 4.0)
        print(f"  Theoretical Student-t({nu:.2f}) excess kurtosis = 6/(nu-4) = {theo_kurt:.3f}")
    else:
        theo_kurt = float("inf")
        print(f"  Theoretical Student-t kurtosis = inf (nu<=4)")

    # Load val data
    surfaces = np.load(args.data_path)["surface"].astype("float32")
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    hist_list = []
    fut_list = []
    for i in val_indices[: args.n_windows]:
        hist_list.append(surf_tensor[i:i + args.history_len].reshape(args.history_len, -1))
        fut_list.append(surf_tensor[i + args.history_len:i + args.history_len + args.future_len].reshape(args.future_len, -1))
    history = torch.stack(hist_list, dim=0)  # (N, T_hist, D)
    hist_norm = history * 2.0 - 1.0

    # Forward pass — get z_0, z_path, Lambda
    samples, aux = model(hist_norm, n_samples=args.K)
    # z from aux is the posterior sample (after LatentFM.sample)
    z_0 = aux["z"]  # (N, K, L); the posterior z sampled (Gaussian reparam applied after FM)
    z_path = aux.get("z_path")  # (N, K, T, L) if SDE active
    Lambda = aux["Lambda"]  # (N, T, D, L)
    factor = aux["factor"]  # (N, K, T, D)

    # L1: kurtosis at z_0
    # NOTE: aux["z"] is POST-LatentFM sample (reparam + FM), not raw eta.
    # For H2 we care about whether Student-t survives FM -> SDE pipeline.
    kurt_z0 = excess_kurtosis(z_0)
    print(f"\n=== L1 / L2 Kurtosis propagation (excess kurtosis) ===")
    print(f"  z_0 (posterior sample after LatentFM): {kurt_z0:.3f}")

    per_horizon_kurt = {}
    if z_path is not None:
        for t in [0, 5, 10, 15, 20, 25, 29]:
            kurt_zt = excess_kurtosis(z_path[:, :, t, :])
            per_horizon_kurt[t] = kurt_zt
            print(f"  z_{t:2d}: {kurt_zt:.3f}")
        ratio_29_0 = per_horizon_kurt[29] / max(abs(per_horizon_kurt[0]), 1e-12)
        print(f"  kurt ratio z_29 / z_0 = {ratio_29_0:.3f}  (gate >= 0.3)")
    else:
        ratio_29_0 = float("nan")
        print("  z_path not available (SDE inactive)")

    # Surface-level: kurtosis of factor = Lambda @ z
    kurt_factor = excess_kurtosis(factor)
    kurt_samples = excess_kurtosis(samples)
    print(f"\n  factor (Lambda*z at surface level): {kurt_factor:.3f}")
    print(f"  samples (final surface output): {kurt_samples:.3f}")
    gauss_ratio = kurt_factor / max(abs(kurt_z0), 1e-12)
    print(f"  factor/z_0 ratio: {gauss_ratio:.3f}  (gate >= 0.5 so Lambda doesn't Gaussianize)")

    # Per-cell surface kurtosis (daily changes)
    changes = samples[:, :, 1:] - samples[:, :, :-1]  # (N, K, T-1, D)
    kurt_changes = excess_kurtosis(changes)
    print(f"\n  daily-changes kurtosis (surface): {kurt_changes:.3f}")

    # Gates
    l1_pass = abs(kurt_z0) > 5 and not np.isnan(nu) and np.isfinite(nu)
    l2_propagation_pass = ratio_29_0 >= 0.3 if not np.isnan(ratio_29_0) else False
    l2_surface_pass = gauss_ratio >= 0.5 if kurt_z0 > 0.1 else False

    summary = {
        "checkpoint": str(args.checkpoint),
        "nu_learned": nu,
        "theoretical_kurt_for_nu": theo_kurt,
        "L1_kurt_z0": kurt_z0,
        "L2_per_horizon_kurt": per_horizon_kurt,
        "L2_kurt_ratio_z29_z0": ratio_29_0,
        "L2_factor_kurt": kurt_factor,
        "L2_factor_over_z0_ratio": gauss_ratio,
        "surface_sample_kurt": kurt_samples,
        "daily_change_kurt": kurt_changes,
        "gates": {
            "L1_pass": bool(l1_pass),
            "L2_propagation_pass": bool(l2_propagation_pass),
            "L2_surface_pass": bool(l2_surface_pass),
        },
    }

    print()
    print(f"L1 (nu finite + kurt_z0 > 5): {'PASS' if l1_pass else 'FAIL'}")
    print(f"L2 propagation (kurt ratio >= 0.3): {'PASS' if l2_propagation_pass else 'FAIL'}")
    print(f"L2 surface (Lambda doesn't Gaussianize): {'PASS' if l2_surface_pass else 'FAIL'}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heavytail_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'heavytail_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
