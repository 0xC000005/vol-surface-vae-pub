"""
251a SDE dynamics diagnostic.

Three mechanism checks (rc27 kill criteria + gate matrix):
  1) effective_rank(z_t) per horizon t ∈ [0, T-1]. Kill criterion: rank(z_t at h30) < 2
     ⇒ SDE collapsed to deterministic; eps_floor insufficient.
  2) Drift / diffusion magnitudes per t. PRINCIPLED SUCCESS gate requires
     mean drift magnitude > 0.01 at h30 AND effective_rank ≥ 2.
  3) z_t cross-sample variance as function of t (should grow with t — Brownian-
     like dispersion); flags deterministic collapse independently of rank.

Plus an OU-fit check:
  4) Regress f_θ(z_t, ·) onto -α · z_t via OLS across gathered (t, B, K) points.
     Reports R², mean α. High R² + positive α ⇒ SDE is learning mean-reversion
     (principled explanation for any MR gate flip).

Outputs JSON + markdown to --output_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import NeuralFactorModel, load_model
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def effective_rank(X: torch.Tensor) -> float:
    """Shannon-based effective rank: exp(-Σ p_i log p_i), p_i = s_i² / Σ s_j²."""
    if X.numel() == 0:
        return 0.0
    X32 = X.float()
    try:
        s = torch.linalg.svdvals(X32)
    except RuntimeError:
        return 0.0
    s_sq = s.pow(2)
    p = s_sq / s_sq.sum().clamp(min=1e-12)
    H = -(p * (p.clamp(min=1e-12)).log()).sum()
    return float(torch.exp(H).item())


@torch.no_grad()
def gather_sde_trajectories(
    model: NeuralFactorModel,
    loader: DataLoader,
    device: torch.device,
    K: int,
    max_batches: int = 30,
) -> dict:
    """Collect z_path, drift(z_t, t), diffusion(z_t, t) across validation windows."""
    if model.latent_sde is None:
        raise ValueError("Model has no LatentSDE — this diagnostic requires 251a.")

    z_paths: list[torch.Tensor] = []
    drifts: list[torch.Tensor] = []   # per-step f_theta(z_t, t, h), shape (B, K, T-1, L)
    diffusions: list[torch.Tensor] = []

    model.eval()
    for i, (hist_01, _fut) in enumerate(loader):
        if i >= max_batches:
            break
        hist_01 = hist_01.to(device, non_blocking=True)
        B = hist_01.shape[0]
        hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
        _samples, aux = model(hist_norm, n_samples=K)
        z_path = aux["z_path"]  # (B, K, T, L)
        if z_path is None:
            raise ValueError("aux['z_path'] is None — LatentSDE was not active during forward pass.")
        z_paths.append(z_path.detach().cpu())
        # Step-by-step re-evaluate drift / diffusion at each z_t state.
        h = aux["h"]
        B2, K2, T, L = z_path.shape
        step_drifts = torch.zeros(B2, K2, T - 1, L, device=device)
        step_diffs = torch.zeros(B2, K2, T - 1, L, device=device)
        for t_idx in range(T - 1):
            z_t = z_path[:, :, t_idx, :]
            drift, diffusion = model.latent_sde.step(z_t, t_idx, h)
            step_drifts[:, :, t_idx, :] = drift
            step_diffs[:, :, t_idx, :] = diffusion
        drifts.append(step_drifts.detach().cpu())
        diffusions.append(step_diffs.detach().cpu())
    return {
        "z_path": torch.cat(z_paths, dim=0) if z_paths else torch.empty(0),
        "drift": torch.cat(drifts, dim=0) if drifts else torch.empty(0),
        "diffusion": torch.cat(diffusions, dim=0) if diffusions else torch.empty(0),
    }


def per_step_rank(z_path: torch.Tensor) -> list[float]:
    """effective_rank of z_path reshaped to (B*K, L) at each step t."""
    B, K, T, L = z_path.shape
    out = []
    for t in range(T):
        X = z_path[:, :, t, :].reshape(B * K, L)
        out.append(effective_rank(X))
    return out


def per_step_magnitude(X: torch.Tensor) -> list[float]:
    """Mean L2 norm over (B, K) at each t. X shape (B, K, T, L) or (B, K, T-1, L)."""
    # L2 norm per (B, K, t), then average.
    norms = X.float().pow(2).sum(dim=-1).sqrt()  # (B, K, T)
    return norms.mean(dim=(0, 1)).tolist()


def per_step_variance(z_path: torch.Tensor) -> list[float]:
    """Mean over (B, L) of var over K ensemble members at each t."""
    B, K, T, L = z_path.shape
    var_k = z_path.float().var(dim=1, unbiased=False)  # (B, T, L)
    return var_k.mean(dim=(0, 2)).tolist()


def ou_fit(drift: torch.Tensor, z_path: torch.Tensor) -> dict:
    """Regress drift onto -z with a single scalar α: drift ≈ -α · z_t.

    Uses all gathered (B, K, T-1, L) points. Reports α (OLS on scalar), R².
    """
    # Match shapes: drift is (B, K, T-1, L); z at those steps is z_path[:, :, :T-1, :].
    z = z_path[:, :, :drift.shape[2], :]
    drift_flat = drift.reshape(-1).float()
    z_flat = z.reshape(-1).float()
    # drift = -α · z  ⇒  α = -(z·drift) / (z·z)
    denom = (z_flat * z_flat).sum().clamp(min=1e-12)
    alpha = -(z_flat * drift_flat).sum() / denom
    pred = -alpha * z_flat
    ss_res = ((drift_flat - pred) ** 2).sum()
    ss_tot = ((drift_flat - drift_flat.mean()) ** 2).sum().clamp(min=1e-12)
    r2 = 1.0 - (ss_res / ss_tot)
    return {"alpha": float(alpha.item()), "r2": float(r2.item())}


def main() -> None:
    parser = argparse.ArgumentParser(description="251a SDE dynamics diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, payload = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {payload.get('epoch', -1)})")
    print(f"  cfg: use_latent_sde={model.cfg.use_latent_sde} "
          f"use_latent_fm={model.cfg.use_latent_fm} L={model.cfg.latent_dim}")

    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.future_len,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False,
    )

    gathered = gather_sde_trajectories(
        model, val_loader, device, K=args.K, max_batches=args.max_batches,
    )
    z_path = gathered["z_path"]   # (B, K, T, L)
    drift = gathered["drift"]     # (B, K, T-1, L)
    diffusion = gathered["diffusion"]

    B, K, T, L = z_path.shape
    print(f"Gathered z_path: (B={B}, K={K}, T={T}, L={L})")

    rank_per_t = per_step_rank(z_path)
    drift_mag_per_t = per_step_magnitude(drift)
    diffusion_mag_per_t = per_step_magnitude(diffusion)
    var_per_t = per_step_variance(z_path)
    ou = ou_fit(drift, z_path)

    # Kill / gate summary
    rank_h30 = rank_per_t[min(T - 1, 29)]
    drift_mag_h30 = drift_mag_per_t[min(T - 2, 28)] if drift.numel() > 0 else 0.0
    diffusion_mag_h30 = diffusion_mag_per_t[min(T - 2, 28)] if diffusion.numel() > 0 else 0.0

    gate_rank = rank_h30 >= 2.0
    gate_drift = drift_mag_h30 > 0.01
    kill_collapse = rank_h30 < 2.0 or drift_mag_h30 < 1e-4

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": payload.get("epoch", -1),
        "config": {
            "L": int(L), "T": int(T),
            "latent_sde_hidden": model.cfg.latent_sde_hidden,
            "latent_sde_time_embed": model.cfg.latent_sde_time_embed,
            "latent_sde_eps_floor": model.cfg.latent_sde_eps_floor,
        },
        "rank_per_t": rank_per_t,
        "drift_magnitude_per_t": drift_mag_per_t,
        "diffusion_magnitude_per_t": diffusion_mag_per_t,
        "z_variance_per_t": var_per_t,
        "h30_summary": {
            "rank": rank_h30,
            "drift_mag": drift_mag_h30,
            "diffusion_mag": diffusion_mag_h30,
            "gate_rank_ge_2": bool(gate_rank),
            "gate_drift_gt_0p01": bool(gate_drift),
            "kill_latent_collapse": bool(kill_collapse),
        },
        "ou_fit": ou,
    }

    out_json = Path(args.output_dir) / "sde_dynamics.json"
    out_md = Path(args.output_dir) / "sde_dynamics.md"
    out_json.write_text(json.dumps(report, indent=2))

    md_lines = [
        f"# 251a SDE Dynamics — `{Path(args.checkpoint).name}` (ep{report['checkpoint_epoch']})",
        "",
        f"- **effective_rank(z_t at h30)**: {rank_h30:.3f}  (gate ≥ 2.0: {'PASS' if gate_rank else 'FAIL'})",
        f"- **drift magnitude at h30**: {drift_mag_h30:.4f}  (gate > 0.01: {'PASS' if gate_drift else 'FAIL'})",
        f"- **diffusion magnitude at h30**: {diffusion_mag_h30:.4f}",
        f"- **OU fit**: α = {ou['alpha']:.4f}, R² = {ou['r2']:.4f}",
        f"- **KILL latent-collapse**: {'YES' if kill_collapse else 'NO'}",
        "",
        "## Per-horizon profile",
        "",
        "| t | eff_rank(z_t) | drift mag | diffusion mag | var_K(z_t) |",
        "|---|---|---|---|---|",
    ]
    for t in range(T):
        dm = drift_mag_per_t[t] if t < len(drift_mag_per_t) else float("nan")
        gm = diffusion_mag_per_t[t] if t < len(diffusion_mag_per_t) else float("nan")
        md_lines.append(
            f"| {t} | {rank_per_t[t]:.3f} | {dm:.4f} | {gm:.4f} | {var_per_t[t]:.4f} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"\nJSON -> {out_json}\nMD   -> {out_md}")
    print(f"\nSummary: rank={rank_h30:.2f}  drift_mag={drift_mag_h30:.4f}  "
          f"OU α={ou['alpha']:.3f} R²={ou['r2']:.3f}")
    if kill_collapse:
        print("*** KILL: LATENT COLLAPSE — eps_floor insufficient or SDE didn't learn. ***")


if __name__ == "__main__":
    main()
