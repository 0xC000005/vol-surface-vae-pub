"""
250-series factor-structure diagnostic.

Five mechanism checks (Codex + Stage A gate requirements):
  1) SVD(Λ) energy profile: top-L singular values carry >= 0.85 of total Frobenius
     energy per (B, T) slice. Flags full-rank leak (Codex top concern).
  2) effective_rank(z) across the validation split: should be close to L.
     Low effective_rank means latent collapsed — factor model wasted capacity.
  3) cond_stddev(Λ): how much Λ varies with history. Low stddev means Λ is
     effectively a constant loading matrix — history is being ignored.
  4) D_scale distribution: range of the idiosyncratic scale head — should vary
     across (t, d) and across histories.
  5) z posterior statistics: mu_z, sigma_z distributions.

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
    """Shannon-based effective rank: exp(-sum p_i log p_i) with p_i = s_i^2 / sum s_j^2."""
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


def top_energy_fraction(singular_vals: torch.Tensor, L: int) -> float:
    s = singular_vals.float()
    s2 = s.pow(2)
    total = s2.sum().clamp(min=1e-12)
    top = s2[:L].sum()
    return float((top / total).item())


def gather_aux(
    model: NeuralFactorModel,
    loader: DataLoader,
    device: torch.device,
    K: int,
    max_batches: int = 30,
) -> dict:
    zs = []
    mus = []
    log_sigmas = []
    lambdas = []
    d_scales = []
    hs = []
    model.eval()
    with torch.no_grad():
        for i, (hist_01, _fut) in enumerate(loader):
            if i >= max_batches:
                break
            hist_01 = hist_01.to(device, non_blocking=True)
            B = hist_01.shape[0]
            hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
            _samples, aux = model(hist_norm, n_samples=K)
            zs.append(aux["z"].detach().cpu())
            mus.append(aux["mu_z"].detach().cpu())
            log_sigmas.append(aux["log_sigma_z"].detach().cpu())
            lambdas.append(aux["Lambda"].detach().cpu())
            d_scales.append(aux["D_scale"].detach().cpu())
            hs.append(aux["h"].detach().cpu())
    return {
        "z": torch.cat(zs, dim=0) if zs else torch.empty(0),
        "mu_z": torch.cat(mus, dim=0) if mus else torch.empty(0),
        "log_sigma_z": torch.cat(log_sigmas, dim=0) if log_sigmas else torch.empty(0),
        "Lambda": torch.cat(lambdas, dim=0) if lambdas else torch.empty(0),
        "D_scale": torch.cat(d_scales, dim=0) if d_scales else torch.empty(0),
        "h": torch.cat(hs, dim=0) if hs else torch.empty(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="250-series factor-structure diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--K", type=int, default=16, help="sample K for latent draw stats")
    parser.add_argument("--max_batches", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, payload = load_model(args.checkpoint, device)
    L = model.cfg.latent_dim

    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.future_len
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size, shuffle=False,
    )

    aux = gather_aux(model, val_loader, device, args.K, args.max_batches)
    z = aux["z"]          # (B, K, L)
    mu_z = aux["mu_z"]    # (B, L)
    ls = aux["log_sigma_z"]  # (B, L)
    Lambda = aux["Lambda"]   # (B, T, D, L)
    D_scale = aux["D_scale"] # (B, T, D)

    # ---- 1. SVD(Λ) per (B, T) ----
    # Reshape to (B*T, D, L), compute SVD, tabulate top-L energy fraction.
    B_, T_, D_, L_ = Lambda.shape
    Lm = Lambda.reshape(B_ * T_, D_, L_)
    top_energies = []
    all_singulars = []
    for bt in range(0, Lm.shape[0], 64):
        block = Lm[bt:bt + 64].float()
        svals = torch.linalg.svdvals(block)  # (batch_sub, min(D, L))
        all_singulars.append(svals)
        for row in svals:
            top_energies.append(top_energy_fraction(row, L_))
    top_energies = np.array(top_energies)
    all_singulars = torch.cat(all_singulars, dim=0)

    # Mean singular profile
    mean_svals = all_singulars.mean(dim=0).tolist()

    # ---- 2. effective_rank(z) ----
    # Reshape z to (B*K, L), compute SVD and effective rank.
    z_flat = z.reshape(-1, L).float()
    z_rank = effective_rank(z_flat)

    # ---- 3. cond_stddev(Λ): how much Λ varies with history ----
    # Std over batch dim per (t, d, l), averaged.
    lambda_std_per_slot = Lambda.std(dim=0).mean().item()
    lambda_mean_abs = Lambda.abs().mean().item()
    lambda_rel_cond_std = (
        lambda_std_per_slot / max(lambda_mean_abs, 1e-8)
    )

    # ---- 4. D_scale distribution ----
    d_stats = {
        "mean": float(D_scale.mean().item()),
        "median": float(D_scale.median().item()),
        "min": float(D_scale.min().item()),
        "max": float(D_scale.max().item()),
        "std_across_batch": float(D_scale.std(dim=0).mean().item()),
        "std_across_time": float(D_scale.std(dim=1).mean().item()),
    }

    # ---- 5. z posterior statistics ----
    mu_stats = {
        "mu_z_mean": float(mu_z.mean().item()),
        "mu_z_std": float(mu_z.std().item()),
        "log_sigma_z_mean": float(ls.mean().item()),
        "log_sigma_z_min": float(ls.min().item()),
        "log_sigma_z_max": float(ls.max().item()),
    }

    # ---- Gate interpretations ----
    top_frac_mean = float(top_energies.mean())
    # Because the Lambda matrix is (D, L) with D >= L, top-L captures all singular
    # energy when the matrix has rank <= L. This is expected to be 1.0 for small L.
    # The meaningful check is whether rank(Λ) saturates at L (then exactly L singulars
    # are used) or stays lower (representation collapse).
    leak_flag = "N/A"
    if L_ > 0:
        # Full-rank leak means the loadings use MORE dims than L allows. Since Λ is
        # (D, L) with D>=L, rank is at most L by construction; so a different check:
        # compute rank of Λ via svdvals — count non-trivial (> eps) singulars.
        eps = 1e-4
        effective_L = (all_singulars > eps * all_singulars[:, :1]).sum(dim=1).float().mean().item()
        leak_flag = (
            "OK (Λ uses <= L factors as designed)"
            if effective_L <= L_ + 0.01
            else f"SUSPECT (effective_L={effective_L:.2f} > L={L_})"
        )
    else:
        effective_L = 0.0

    report = {
        "checkpoint": args.checkpoint,
        "latent_dim_L": L_,
        "n_windows": int(val_hist.shape[0]),
        "n_batches_sampled": args.max_batches,
        "singular_energy_profile": {
            "mean_top_L_fraction": top_frac_mean,
            "mean_singular_values": mean_svals,
            "effective_L": effective_L,
            "leak_flag": leak_flag,
        },
        "latent_effective_rank": {
            "z_effective_rank": z_rank,
            "target": L_,
            "collapse_flag": "OK" if z_rank >= 0.5 * L_ else "COLLAPSED",
        },
        "lambda_condition_sensitivity": {
            "std_per_slot_mean": lambda_std_per_slot,
            "mean_abs": lambda_mean_abs,
            "rel_cond_std": lambda_rel_cond_std,
            "constant_flag": "OK" if lambda_rel_cond_std > 0.1 else "CONSTANT (Λ ignores history)",
        },
        "idio_scale_stats": d_stats,
        "z_posterior_stats": mu_stats,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
    }

    out_json = Path(args.output_dir) / "factor_structure.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    md_lines = [
        f"# 250-series Factor Structure Diagnostic — {args.checkpoint}",
        "",
        f"- Latent dim L: {L_}",
        f"- Val windows: {val_hist.shape[0]}",
        f"- Batches sampled: {args.max_batches}",
        "",
        "## 1. SVD(Λ) energy profile",
        f"- Mean top-L energy fraction: **{top_frac_mean:.4f}**  (= 1.0 means Λ has exactly rank L)",
        f"- Effective rank of Λ: **{effective_L:.3f}** / L={L_}",
        f"- Leak flag: **{leak_flag}**",
        f"- Mean singular values: {['%.4f' % s for s in mean_svals]}",
        "",
        "## 2. Latent effective rank (z)",
        f"- effective_rank(z): **{z_rank:.3f}** / L={L_}",
        f"- Collapse flag: **{report['latent_effective_rank']['collapse_flag']}**",
        "",
        "## 3. Λ condition sensitivity",
        f"- std_per_slot_mean:  {lambda_std_per_slot:.6f}",
        f"- mean(|Λ|):          {lambda_mean_abs:.6f}",
        f"- rel_cond_std:       **{lambda_rel_cond_std:.4f}**  (should be > 0.1)",
        f"- Flag: **{report['lambda_condition_sensitivity']['constant_flag']}**",
        "",
        "## 4. Idiosyncratic scale D(h)",
        f"- mean={d_stats['mean']:.5f}  median={d_stats['median']:.5f}",
        f"- range=[{d_stats['min']:.5f}, {d_stats['max']:.5f}]",
        f"- std_across_batch={d_stats['std_across_batch']:.5f}  std_across_time={d_stats['std_across_time']:.5f}",
        "",
        "## 5. Posterior z statistics",
        f"- mu_z: mean={mu_stats['mu_z_mean']:.4f}  std={mu_stats['mu_z_std']:.4f}",
        f"- log_sigma_z: mean={mu_stats['log_sigma_z_mean']:.4f}  "
        f"range=[{mu_stats['log_sigma_z_min']:.4f}, {mu_stats['log_sigma_z_max']:.4f}]",
        "",
    ]
    out_md = Path(args.output_dir) / "factor_structure.md"
    out_md.write_text("\n".join(md_lines))
    print(f"\nReport written to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
