"""
251c alpha + OU alignment diagnostic.

Reports:
  1. alpha = softplus(alpha_raw)  — the learned MR rate
  2. Cosine alignment: cos(f_theta(z_t, h), -alpha * z_t), averaged over samples.
     High cosine (≥ 0.3) = drift aligns with OU prior (mean reversion active).
     Low cosine = model "hedges" — satisfies the regularizer superficially without true MR.
  3. alpha_from_fit: empirical -slope(drift / z) via OLS over gathered points.
     Compare to alpha_param — mismatch signals model reroutes the OU pull through
     other directions.

Reuses `gather_sde_trajectories` from diagnose_251a_sde_dynamics.

Gate thresholds (from 251c plan):
  - alpha_param >= 0.005 (non-trivial learned MR rate)
  - cos_alignment >= 0.3 (drift at least partially MR-aligned)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import load_model
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.diagnose_251a_sde_dynamics import (
    gather_sde_trajectories,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    if model.latent_sde is None:
        print("ERROR: model has no LatentSDE")
        return 2

    # Report the scalar alpha if present
    if model.latent_sde.alpha_raw is not None:
        alpha_raw = float(model.latent_sde.alpha_raw.item())
        alpha = float(F.softplus(model.latent_sde.alpha_raw).item())
    else:
        alpha_raw = float("nan")
        alpha = 0.0

    # Load val data for trajectory gathering
    import numpy as np
    surfaces = np.load(args.data_path)["surface"].astype("float32")
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.future_len
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False,
    )

    # Gather trajectories (z_path, drift, diffusion) — reuses 251a gatherer
    traj = gather_sde_trajectories(model, val_loader, device, args.K, max_batches=args.max_batches)
    z_path = traj["z_path"]      # (N, K, T, L)
    drift = traj["drift"]        # (N, K, T-1, L)
    # Align z to drift's time range
    z_before = z_path[:, :, :-1, :]  # (N, K, T-1, L)

    # Cosine alignment: cos(drift, -alpha * z)
    # per-(n,k,t) vector cosine, then averaged.
    target = -alpha * z_before  # (N, K, T-1, L)
    dot = (drift * target).sum(dim=-1)                     # (N, K, T-1)
    d_norm = drift.norm(dim=-1).clamp(min=1e-8)
    t_norm = target.norm(dim=-1).clamp(min=1e-8)
    cos_per = dot / (d_norm * t_norm)                      # (N, K, T-1)
    cos_mean = float(cos_per.mean().item())
    cos_median = float(cos_per.median().item())
    cos_min = float(cos_per.min().item())
    cos_max = float(cos_per.max().item())

    # Empirical alpha from OLS: drift_l = -alpha_fit * z_l per latent dim (flattened)
    # drift ≈ -alpha z  =>  alpha_fit = -sum(drift * z) / sum(z * z)
    z_flat = z_before.reshape(-1)
    d_flat = drift.reshape(-1)
    denom = (z_flat * z_flat).sum().clamp(min=1e-12)
    alpha_fit = -(d_flat * z_flat).sum() / denom
    alpha_fit_val = float(alpha_fit.item())

    # Gates per plan
    alpha_gate = alpha >= 0.005
    cos_gate = cos_mean >= 0.3
    verdict = "PASS" if (alpha_gate and cos_gate) else "FAIL"

    summary = {
        "checkpoint": str(args.checkpoint),
        "alpha_raw": alpha_raw,
        "alpha_param": alpha,
        "alpha_from_fit": alpha_fit_val,
        "alpha_mismatch": abs(alpha - alpha_fit_val),
        "cos_alignment": {
            "mean": cos_mean,
            "median": cos_median,
            "min": cos_min,
            "max": cos_max,
        },
        "gates": {
            "alpha_ge_0p005": alpha_gate,
            "cos_ge_0p3": cos_gate,
            "verdict": verdict,
        },
        "n_windows": int(z_path.shape[0]),
        "K": int(z_path.shape[1]),
        "T_minus_1": int(drift.shape[2]),
    }

    print(f"alpha_raw = {alpha_raw:.4f}  →  alpha = softplus = {alpha:.4f}")
    print(f"alpha_from_fit (OLS: drift = -alpha*z) = {alpha_fit_val:.4f}")
    print(f"alpha mismatch (|param - fit|) = {abs(alpha - alpha_fit_val):.4f}")
    print(f"cos(drift, -alpha*z):  mean={cos_mean:.3f}  median={cos_median:.3f}  range=[{cos_min:.3f}, {cos_max:.3f}]")
    print(f"Gate alpha >= 0.005: {'PASS' if alpha_gate else 'FAIL'}")
    print(f"Gate cos >= 0.3:     {'PASS' if cos_gate else 'FAIL'}")
    print(f"Verdict: {verdict}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "alpha_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'alpha_diagnostic.json'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
