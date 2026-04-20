"""Diagnose WHY the cross-cell correlation is not emerging in 240c_iter.

Observation: samples are full-rank (eff_rank 40.32 ≈ GT 39.69) with calibrated marginals
but PC1 = 2% of variance vs GT's 55%. corr_ratio stuck at 0.002 / GT 0.436.

Test three mechanistic hypotheses:

H1 (loss landscape) — Is the ES+VS objective actually lower at isotropic samples than at
factor-structured samples? If YES, the loss is rewarding the pathology and no amount of
training will fix it. If NO, the model got stuck in a bad optimum — it's an optimization
issue, not a loss issue.

H2 (factor alignment) — Even if samples have the wrong eigenvalue SPECTRUM, are their top
principal directions aligned with GT's? If yes, factor structure exists but is under-
amplified. If no, the model learned an entirely different coordinate system.

H3 (noise amplification) — We feed IID noise z~N(0, I) with 750 dimensions. If we
replace z with STRUCTURED noise (sampled from the GT factor distribution), does the
model's output snap to factor-structured output, or does it remix into isotropic noise?

Plus a simple AdaLN-gate inspection: have the gates moved from zero-init after 10 epochs?
Zero gates ⟹ the DiT is using the conditioning weakly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.backfill.block_ar.train_240c_iter_ddim_crps import (
    JointChunkIterCRPSModel,
    JointChunkConfig,
    load_model,
)
from experiments.backfill.block_ar._rollout_220_utils import build_rollout_windows
from diffusion.block_ar.single_pass_ar import (
    denormalize_iv,
    normalize_iv,
    energy_score,
    variogram_score,
)


def h1_loss_landscape(model, history, future, device, K=48):
    """Compute ES+VS loss for three synthetic sample clouds:
    (a) model's actual output (what we're training toward by gradient)
    (b) GT-like low-rank factor samples (what we want)
    (c) isotropic noise samples matched to GT marginals (the pathology)
    If (a) ≈ (c) < (b), loss is actively rewarding the pathology."""
    B, T, H, W = future.shape
    D = H * W

    # (a) actual model samples
    with torch.no_grad():
        samp_actual = model.sample_batched(history, n_samples=K)  # (B, K, T, 5, 5) in [0,1]
        samp_actual_norm = normalize_iv(samp_actual)  # back to [-1, 1] for loss compatibility

    future_norm = normalize_iv(future)  # GT in [-1, 1]

    # (b) GT-like low-rank samples: take GT marginal mean/std per-cell, build factor-correlated
    #     samples by drawing a LOW-DIM latent and projecting through the GT factor basis.
    #     Fit SVD of (B, T*D) = (all GT flattened) to get top directions.
    gt_flat = future_norm.reshape(B, -1).cpu().numpy()
    gt_mean = gt_flat.mean(axis=0, keepdims=True)
    gt_centered = gt_flat - gt_mean
    U, S, Vt = np.linalg.svd(gt_centered, full_matrices=False)
    # Generate K samples for each of B windows using the GT spectral basis
    samp_b = np.empty((B, K, T * D))
    for b in range(B):
        # center = this window's future
        center = gt_flat[b]
        # Low-rank noise with variance = eigen-squared / N
        coeffs = np.random.randn(K, len(S)) * S / np.sqrt(B)
        perturbation = coeffs @ Vt  # (K, T*D)
        samp_b[b] = center + perturbation
    samp_b = torch.from_numpy(samp_b).float().reshape(B, K, T, H, W).to(device)

    # (c) isotropic noise: per-cell std matched to GT marginal std
    gt_per_cell = future_norm.reshape(-1, T, H, W)  # (B, T, H, W)
    gt_std_per_cell = gt_per_cell.std(dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
    gt_mean_per_cell = future_norm.mean(dim=0, keepdim=True)  # (1, T, H, W)
    noise_iso = torch.randn(B, K, T, H, W, device=device)
    samp_c = gt_mean_per_cell.unsqueeze(1) + gt_std_per_cell * noise_iso

    results = {}
    with torch.no_grad():
        for label, s in [("a_actual", samp_actual_norm), ("b_gt_like", samp_b), ("c_isotropic", samp_c)]:
            es = energy_score(s, future_norm).item()
            vs = variogram_score(s, future_norm).item()
            # Rank/spectrum on first window for interpretability
            sflat = s[0].reshape(K, -1).cpu().numpy()
            sflat = sflat - sflat.mean(axis=0, keepdims=True)
            sigma = np.linalg.svd(sflat, compute_uv=False)
            pc1_var_frac = float((sigma[0] ** 2) / (sigma ** 2).sum())
            results[label] = {
                "es": es,
                "vs": vs,
                "pc1_variance_fraction": pc1_var_frac,
                "top5_singular_values": sigma[:5].tolist(),
            }
    return results


def h2_factor_alignment(model, history, future, device, K=48):
    """Principal-direction alignment between model samples and GT marginal."""
    B, T, H, W = future.shape
    future_norm = normalize_iv(future)
    D = T * H * W

    # GT marginal basis: SVD of centered GT windows
    gt_flat = future_norm.reshape(B, -1).cpu().numpy()
    gt_centered = gt_flat - gt_flat.mean(axis=0, keepdims=True)
    _, sigma_gt, Vt_gt = np.linalg.svd(gt_centered, full_matrices=False)
    pc1_gt_var = float((sigma_gt[0] ** 2) / (sigma_gt ** 2).sum())

    # Model samples
    with torch.no_grad():
        samp = model.sample_batched(history, n_samples=K)
        samp_norm = normalize_iv(samp)  # (B, K, T, 5, 5)

    # Flatten per-window samples, compute eigen basis, measure alignment to GT PC1
    alignments = []
    pc1_vars = []
    for b in range(B):
        flat = samp_norm[b].reshape(K, -1).cpu().numpy()
        flat -= flat.mean(axis=0, keepdims=True)
        _, sigma_s, Vt_s = np.linalg.svd(flat, full_matrices=False)
        pc1_s_var = float((sigma_s[0] ** 2) / (sigma_s ** 2).sum())
        # Alignment = |<pc1_s, pc1_gt>| in [0, 1]
        align = float(abs(np.dot(Vt_s[0], Vt_gt[0])))
        alignments.append(align)
        pc1_vars.append(pc1_s_var)

    return {
        "gt_pc1_variance_fraction": pc1_gt_var,
        "sample_pc1_variance_fraction_mean": float(np.mean(pc1_vars)),
        "sample_pc1_variance_fraction_std": float(np.std(pc1_vars)),
        "pc1_direction_alignment_mean": float(np.mean(alignments)),
        "pc1_direction_alignment_std": float(np.std(alignments)),
        "n_windows_evaluated": B,
    }


def h3_noise_amplification(model, history, device, K=32):
    """Replace z ~ N(0, I) with low-rank structured noise, see whether the model
    preserves or remixes that structure in its output."""
    B = history.shape[0]
    T = model.cfg.future_len
    S = model.cfg.surface_cells

    cond = model.encoder(history)

    with torch.no_grad():
        model._move_scheduler(device)
        # IID baseline — standard inference path (just call sample_batched)
        samp_iid = model.sample_batched(history, n_samples=K)  # (B, K, T, 5, 5) in [0, 1]

        # Structured low-rank noise injection: we HACK the initial x at t=T-1 to have
        # low-rank structure, then let DDIM denoise normally. If the model's subsequent
        # refinement adds IID perturbation, we'll see rank increase. If it preserves
        # structure, we'll see low-rank output.
        total_steps = model.cfg.n_diffusion_steps
        n_inference_steps = 20
        step_indices = torch.linspace(
            total_steps - 1, 0, n_inference_steps + 1, device=device
        ).round().long()
        cond_rep = cond.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        # Rank-3 structured noise: (B*K, T*S) = Z @ V^T where Z is (B*K, 3) and V is (3, T*S)
        rng = torch.Generator(device=device).manual_seed(1337)
        V = torch.randn(3, T * S, generator=rng, device=device)
        V = V / V.norm(dim=-1, keepdim=True)
        Z = torch.randn(B * K, 3, generator=rng, device=device) * (T * S) ** 0.5 / 3
        x = (Z @ V).reshape(B * K, T, S)
        for i in range(n_inference_steps):
            t_cur = step_indices[i].expand(B * K)
            t_prev = step_indices[i + 1].expand(B * K)
            x = model._ddim_step(x, t_cur, t_prev, cond_rep)
        samp_struct = denormalize_iv(x.clamp(-1, 1)).reshape(B, K, T, 5, 5)

    # Measure rank of each
    def eff_rank(s):
        B, K = s.shape[0], s.shape[1]
        ranks = []
        for b in range(B):
            flat = s[b].reshape(K, -1).cpu().numpy()
            flat -= flat.mean(axis=0, keepdims=True)
            sigma = np.linalg.svd(flat, compute_uv=False)
            s2 = sigma ** 2
            if s2.sum() > 0:
                p = s2 / s2.sum()
                ranks.append(float(np.exp(-(p * np.log(p + 1e-12)).sum())))
        return float(np.mean(ranks))

    return {
        "iid_noise_input_rank": float(K),  # K IID draws = rank K
        "structured_noise_input_rank": 3.0,
        "iid_output_effective_rank": eff_rank(samp_iid),
        "structured_output_effective_rank": eff_rank(samp_struct),
        "interpretation": (
            "If structured output rank << iid output rank, the model PRESERVES input "
            "noise structure (the issue is our IID noise prior). If they are similar, "
            "the model actively REMIXES noise into full-rank output (the issue is "
            "architectural — the DiT is doing high-freq decorrelation)."
        ),
    }


def inspect_adaln_gates(model):
    """Check if AdaLN-Zero gates have moved from zero-init. Returns mean abs of the
    output Linear weights of each block's modulation MLP (these control shift/scale/gate)."""
    per_block = []
    for i, block in enumerate(model.denoiser.blocks):
        # adaLN is Sequential(SiLU, Linear(cond_dim, 6*d_model)) — the last Linear is what
        # was zero-initialized. If it moved, mean abs weight > 0.
        lin = block.adaLN[-1]
        w_abs = lin.weight.detach().abs().mean().item()
        b_abs = lin.bias.detach().abs().mean().item()
        per_block.append({"block": i, "weight_mean_abs": w_abs, "bias_mean_abs": b_abs})
    # Final AdaLN (between last block and output)
    final_lin = model.denoiser.final_adaLN[-1]
    per_block.append({
        "block": "final",
        "weight_mean_abs": final_lin.weight.detach().abs().mean().item(),
        "bias_mean_abs": final_lin.bias.detach().abs().mean().item(),
    })
    # Output projection (zero-init too)
    out_lin = model.denoiser.output_proj
    per_block.append({
        "block": "output_proj",
        "weight_mean_abs": out_lin.weight.detach().abs().mean().item(),
        "bias_mean_abs": out_lin.bias.detach().abs().mean().item(),
    })
    return per_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/backfill/240c_iter_ddim_crps_s42/best_model.pt")
    ap.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    ap.add_argument("--test_start", type=int, default=4511)
    ap.add_argument("--val_size", type=int, default=441)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--output_dir", default="results/block_ar/240c_iter_s42/diagnostic")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[diag] loading {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"[diag]   ep={ckpt.get('epoch')}  val_loss={ckpt.get('val_loss'):.4f}")

    batch = build_rollout_windows(
        data_path=args.data_path, history_len=30, future_len=30,
        test_start=args.test_start, val_size=args.val_size,
        max_windows=args.max_windows, device=device, split="val",
    )
    history_norm = batch.history_norm
    future_01 = batch.future_01  # in [0, 1]
    print(f"[diag] n_windows={history_norm.shape[0]}")

    print("\n[H1] loss-landscape probe")
    h1 = h1_loss_landscape(model, history_norm, future_01, device, K=32)
    for k, v in h1.items():
        print(f"  {k}: ES={v['es']:.3f}  VS={v['vs']:.1f}  PC1_var%={v['pc1_variance_fraction']*100:.1f}")

    print("\n[H2] factor-direction alignment")
    h2 = h2_factor_alignment(model, history_norm, future_01, device, K=32)
    print(f"  GT PC1 variance fraction:      {h2['gt_pc1_variance_fraction']*100:.1f}%")
    print(f"  Sample PC1 variance fraction:  {h2['sample_pc1_variance_fraction_mean']*100:.1f}% ± {h2['sample_pc1_variance_fraction_std']*100:.1f}%")
    print(f"  PC1 direction alignment:       {h2['pc1_direction_alignment_mean']:.3f} ± {h2['pc1_direction_alignment_std']:.3f}  (1.0 = perfectly aligned, 0.0 = orthogonal)")

    print("\n[H3] noise-structure amplification")
    h3 = h3_noise_amplification(model, history_norm[:8], device, K=32)
    for k, v in h3.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.3f}")

    print("\n[AdaLN] gate inspection — are the gates zero-init or have they moved?")
    gates = inspect_adaln_gates(model)
    for g in gates:
        print(f"  block {g['block']}: weight |mean|={g['weight_mean_abs']:.3e}  bias |mean|={g['bias_mean_abs']:.3e}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corr_collapse_diagnostic.json").write_text(json.dumps({
        "H1_loss_landscape": h1,
        "H2_factor_alignment": h2,
        "H3_noise_amplification": h3,
        "adaln_gates": gates,
    }, indent=2))
    print(f"\n[diag] wrote {out / 'corr_collapse_diagnostic.json'}")


if __name__ == "__main__":
    main()
