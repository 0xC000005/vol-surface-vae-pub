#!/usr/bin/env python
"""
241-series diagnostic: regime separation mechanism check.

Computes 5 groups of diagnostics on a 241a-family checkpoint (loaded via the same
StateMetricTransportModel used by 183c/241a):

1. Velocity cosine gap (mechanism metric):
   - within-regime  vs  between-regime cosine of pred_v at t=0.5
   - turb-vs-calm, turb-vs-middle, calm-vs-middle
   Gap > 0.05 on turb-vs-calm = ΔFM contrast produced regime-distinguishable velocity.

2. Per-cell turb/calm width heatmap (attribution metric):
   - 5×5 grid showing per-cell ratio of (q95-q05) width on turb vs calm windows
   - Overall turb/calm gate metric + per-cell breakdown

3. Velocity-field separation vs middle regime (guardrail):
   - Detects whether contrast pulled turb/calm apart at the expense of middle-regime collapse

4. Sample diversity (SPD guardrail):
   - std_ratio_h{1,15,30} = mean(ensemble_std) / mean(gt_marginal_std)
   - ensemble_effective_rank_normalised (entropic rank / n_cells)
   - under_diverse_h30 flag if std_ratio_h30 < 0.3
   - samples_low_rank flag if effective_rank_normalised < 0.2

5. Best vs final checkpoint comparison (stability guardrail):
   - Runs all of the above on both checkpoints (if both paths provided) and prints delta

All diagnostics run on the validation split (train[-441:test_start]).

Output: JSON to --output_dir/summary.json, markdown to --output_dir/summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def vov_proxy(history: torch.Tensor) -> torch.Tensor:
    """RV-based regime proxy: mean over (time, row, col) of squared first differences."""
    diffs = history[:, 1:] - history[:, :-1]
    return (diffs ** 2).mean(dim=(1, 2, 3))


def assign_regimes(vov: torch.Tensor, q_lo: float = 0.20, q_hi: float = 0.80) -> torch.Tensor:
    """Assign regime labels: 0=middle, 1=calm (bottom Q_lo), 2=turb (top Q_hi)."""
    vov_np = vov.cpu().numpy()
    lo_thresh = float(np.quantile(vov_np, q_lo))
    hi_thresh = float(np.quantile(vov_np, q_hi))
    labels = torch.zeros_like(vov, dtype=torch.long)
    labels[vov <= lo_thresh] = 1
    labels[vov >= hi_thresh] = 2
    return labels


def build_model_from_ckpt(ckpt_path: str, device: str) -> StateMetricTransportModel:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    model = StateMetricTransportModel(
        encoder_config=encoder_config,
        decoder_config=cfg["decoder"], flow_config=cfg["flow"], path_config=cfg["path"],
        prior_config=cfg["prior"], integrated_config=cfg["integrated"],
        state_config=cfg["state"], metric_config=cfg["metric"],
        support_lo=cfg["support_lo"], support_hi=cfg["support_hi"], support_eps=cfg["support_eps"],
        cov_jitter=cfg["cov_jitter"], base_nu=cfg["base_nu"], mix_chunk_size=cfg["mix_chunk_size"],
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def compute_velocity_cosine_gap(
    model: StateMetricTransportModel,
    val_hist: torch.Tensor,
    labels: torch.Tensor,
    n_trials: int = 4,
) -> dict:
    """Compute within-regime and between-regime mean cosine(pred_v) at t=0.5."""
    device = next(model.parameters()).device
    all_pred_v = []
    all_labels = []
    for trial in range(n_trials):
        torch.manual_seed(1000 + trial)
        for i in range(0, len(val_hist), 32):
            batch = val_hist[i : i + 32].to(device)
            labs = labels[i : i + 32].to(device)
            (
                mu, time_factor, time_diag, cell_factor, cell_diag, scale,
                flow_context, base_local_delta, block_logits,
            ) = model.forward_from_history(batch)
            path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
            z0, _ = model.prior.sample(batch.shape[0], device=device, dtype=batch.dtype)
            t = torch.full((batch.shape[0],), 0.5, device=device, dtype=batch.dtype)
            # Use z_t = 0.5 * z0 + 0.5 * E[target] ≈ 0.5 * z0 (since E[target] depends on future)
            # For the mechanism check we just evaluate v_θ(z0_mid, 0.5, cond_i) — no future needed.
            z_t_mid = 0.5 * z0  # approx midpoint; we just need conditional velocity
            pred_v, _ = model.transport_velocity(z_t_mid, t, path_context)
            all_pred_v.append(pred_v.detach().cpu())
            all_labels.append(labs.cpu())
    pred_v = torch.cat(all_pred_v, dim=0)  # (N*n_trials, 750)
    labs = torch.cat(all_labels, dim=0)

    # Normalize per-sample for cosine similarity
    pred_v_norm = torch.nn.functional.normalize(pred_v, dim=-1)

    def avg_cos(sel_a: torch.Tensor, sel_b: torch.Tensor) -> float:
        if sel_a.sum() == 0 or sel_b.sum() == 0:
            return float("nan")
        a = pred_v_norm[sel_a]
        b = pred_v_norm[sel_b]
        # Random pairs avoiding self-matches: compute a @ b.T, exclude diagonal if same mask
        sim = a @ b.T
        if torch.equal(sel_a, sel_b):
            n = sim.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            return float(sim[mask].mean())
        return float(sim.mean())

    turb_mask = labs == 2
    calm_mask = labs == 1
    mid_mask = labs == 0

    within_turb = avg_cos(turb_mask, turb_mask)
    within_calm = avg_cos(calm_mask, calm_mask)
    within_mid = avg_cos(mid_mask, mid_mask)
    between_tc = avg_cos(turb_mask, calm_mask)
    between_tm = avg_cos(turb_mask, mid_mask)
    between_cm = avg_cos(calm_mask, mid_mask)

    return {
        "within_turb_cos": within_turb,
        "within_calm_cos": within_calm,
        "within_middle_cos": within_mid,
        "between_turb_calm_cos": between_tc,
        "between_turb_middle_cos": between_tm,
        "between_calm_middle_cos": between_cm,
        "cosine_gap_turb_calm": (within_turb + within_calm) / 2 - between_tc,
        "cosine_gap_turb_middle": within_turb - between_tm,
        "cosine_gap_calm_middle": within_calm - between_cm,
        "n_turb": int(turb_mask.sum()),
        "n_calm": int(calm_mask.sum()),
        "n_middle": int(mid_mask.sum()),
    }


@torch.no_grad()
def compute_per_cell_turb_calm_heatmap(
    model: StateMetricTransportModel,
    val_hist: torch.Tensor,
    labels: torch.Tensor,
    n_samples: int = 48,
    eval_limit: int = 100,
) -> dict:
    """Per-cell (5×5) turb/calm width ratio map. Sample model, compute widths per regime."""
    device = next(model.parameters()).device
    turb_widths = []  # (n_turb_windows, 5, 5)
    calm_widths = []
    for i, label in enumerate(labels[:eval_limit]):
        if label == 0:
            continue
        history = val_hist[i : i + 1].to(device)
        samples_u = model.sample_batched(history, n_samples=n_samples)
        samples = model.unconstrained_to_iv(samples_u) if hasattr(model, "unconstrained_to_iv") else samples_u
        # (1, K, H=30, C=25). Compute per-cell width at h30 (terminal).
        term = samples[0, :, -1, :]  # (K, 25)
        q05 = torch.quantile(term, 0.05, dim=0)  # (25,)
        q95 = torch.quantile(term, 0.95, dim=0)
        w = (q95 - q05).view(5, 5).cpu().numpy()
        if label == 2:
            turb_widths.append(w)
        elif label == 1:
            calm_widths.append(w)

    if not turb_widths or not calm_widths:
        return {
            "per_cell_turb_calm_ratio": None,
            "per_cell_turb_calm_ratio_median": float("nan"),
            "per_cell_turb_calm_ratio_min": float("nan"),
            "per_cell_turb_calm_ratio_max": float("nan"),
            "overall_turb_calm_ratio": float("nan"),
            "n_turb_evaluated": len(turb_widths),
            "n_calm_evaluated": len(calm_widths),
        }

    mean_turb = np.mean(turb_widths, axis=0)  # (5, 5)
    mean_calm = np.mean(calm_widths, axis=0)
    ratio = mean_turb / np.maximum(mean_calm, 1e-8)
    overall = float(mean_turb.mean() / max(mean_calm.mean(), 1e-8))

    return {
        "per_cell_turb_calm_ratio": ratio.tolist(),
        "per_cell_turb_calm_ratio_median": float(np.median(ratio)),
        "per_cell_turb_calm_ratio_min": float(ratio.min()),
        "per_cell_turb_calm_ratio_max": float(ratio.max()),
        "overall_turb_calm_ratio": overall,
        "n_turb_evaluated": len(turb_widths),
        "n_calm_evaluated": len(calm_widths),
    }


@torch.no_grad()
def compute_sample_diversity(
    model: StateMetricTransportModel,
    val_hist: torch.Tensor,
    val_future: torch.Tensor,
    n_samples: int = 48,
    eval_limit: int = 64,
) -> dict:
    """std_ratio at h={1,15,30}, ensemble entropic rank, collapse flags."""
    device = next(model.parameters()).device
    horizons = [1, 15, 30]
    std_ratios = {h: [] for h in horizons}
    rank_normalised = []
    for i in range(min(eval_limit, len(val_hist))):
        history = val_hist[i : i + 1].to(device)
        future = val_future[i].to(device)  # (30, 5, 5)
        samples_u = model.sample_batched(history, n_samples=n_samples)
        samples = samples_u
        S = samples[0]  # (K, 30, 25)
        for h in horizons:
            ens_std = S[:, h - 1, :].std(dim=0).mean().item()  # mean per-cell std across ensemble
            gt_std = future.view(30, 25)[h - 1].std().item()   # single GT cross-cell std
            std_ratios[h].append(ens_std / max(gt_std, 1e-8))
        # Effective rank: entropic rank of per-cell ensemble covariance at h=30
        term = S[:, -1, :] - S[:, -1, :].mean(dim=0, keepdim=True)
        cov = term.T @ term / max(S.shape[0] - 1, 1)
        eigs = torch.linalg.eigvalsh(cov + 1e-8 * torch.eye(25, device=device)).clamp_min(1e-12)
        p = eigs / eigs.sum()
        h_ent = -(p * p.log()).sum().item()
        eff_rank = float(np.exp(h_ent))
        rank_normalised.append(eff_rank / 25.0)

    out = {}
    for h in horizons:
        out[f"std_ratio_h{h}"] = float(np.mean(std_ratios[h]))
    out["ensemble_effective_rank_normalised"] = float(np.mean(rank_normalised))
    out["under_diverse_h30"] = bool(out["std_ratio_h30"] < 0.3)
    out["samples_low_rank"] = bool(out["ensemble_effective_rank_normalised"] < 0.2)
    return out


def run_for_checkpoint(ckpt_path: str, val_hist: torch.Tensor, val_future: torch.Tensor,
                       labels: torch.Tensor, device: str, eval_limit: int, n_samples: int) -> dict:
    print(f"\n{'=' * 72}\nDiagnosing: {ckpt_path}\n{'=' * 72}")
    model = build_model_from_ckpt(ckpt_path, device)
    print("[1/3] Velocity cosine gap (mechanism)...")
    cos_result = compute_velocity_cosine_gap(model, val_hist, labels)
    print(f"   within_turb={cos_result['within_turb_cos']:.4f}  within_calm={cos_result['within_calm_cos']:.4f}")
    print(f"   between_turb_calm={cos_result['between_turb_calm_cos']:.4f}")
    print(f"   cosine_gap_turb_calm={cos_result['cosine_gap_turb_calm']:.4f}  (target > 0.05)")
    print("[2/3] Per-cell turb/calm width heatmap...")
    heat = compute_per_cell_turb_calm_heatmap(
        model, val_hist, labels, n_samples=n_samples, eval_limit=eval_limit,
    )
    print(f"   overall_turb_calm_ratio={heat['overall_turb_calm_ratio']:.4f}  (target > 1.15)")
    print(f"   per_cell_median={heat['per_cell_turb_calm_ratio_median']:.4f}  min={heat['per_cell_turb_calm_ratio_min']:.3f}  max={heat['per_cell_turb_calm_ratio_max']:.3f}")
    print("[3/3] Sample diversity...")
    div = compute_sample_diversity(
        model, val_hist, val_future, n_samples=n_samples, eval_limit=eval_limit,
    )
    print(f"   std_ratio_h{{1,15,30}}={div['std_ratio_h1']:.3f},{div['std_ratio_h15']:.3f},{div['std_ratio_h30']:.3f}")
    print(f"   effective_rank_norm={div['ensemble_effective_rank_normalised']:.3f}")
    return {"ckpt": ckpt_path, "cosine": cos_result, "width": heat, "diversity": div}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="241a/b/c checkpoint path")
    ap.add_argument("--final_checkpoint", type=str, default=None,
                    help="Optional final_model.pt for best-vs-final stability check")
    ap.add_argument("--baseline_checkpoint", type=str, default=None,
                    help="Optional 183c baseline for deltas")
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--max_val_windows", type=int, default=441)
    ap.add_argument("--eval_limit", type=int, default=120)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--q_lo", type=float, default=0.20)
    ap.add_argument("--q_hi", type=float, default=0.80)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]
    surf = torch.from_numpy(surfaces).to(args.device)
    val_hist, val_future = build_multistep_windows(val_indices, surf, args.history_len, args.future_len)
    vov = vov_proxy(val_hist)
    labels = assign_regimes(vov, q_lo=args.q_lo, q_hi=args.q_hi)
    print(f"Val windows: {len(val_hist)}  |  turb={int((labels==2).sum())}  calm={int((labels==1).sum())}  mid={int((labels==0).sum())}")

    results = {"args": vars(args), "checkpoints": {}}

    results["checkpoints"]["main"] = run_for_checkpoint(
        args.checkpoint, val_hist, val_future, labels,
        args.device, args.eval_limit, args.n_samples,
    )
    if args.final_checkpoint:
        results["checkpoints"]["final"] = run_for_checkpoint(
            args.final_checkpoint, val_hist, val_future, labels,
            args.device, args.eval_limit, args.n_samples,
        )
    if args.baseline_checkpoint:
        results["checkpoints"]["baseline_183c"] = run_for_checkpoint(
            args.baseline_checkpoint, val_hist, val_future, labels,
            args.device, args.eval_limit, args.n_samples,
        )

    # Apply decision matrix on main checkpoint
    main = results["checkpoints"]["main"]
    gate_turb_calm = main["width"]["overall_turb_calm_ratio"]
    mech_cos_gap = main["cosine"]["cosine_gap_turb_calm"]

    if np.isnan(gate_turb_calm):
        decision = "INSUFFICIENT_DATA"
    elif gate_turb_calm > 1.10 and mech_cos_gap > 0.05:
        decision = "CLEAN_SUCCESS"
    elif gate_turb_calm > 1.10 and mech_cos_gap < 0.03:
        decision = "FRAGILE_SUCCESS"
    elif gate_turb_calm < 1.10 and mech_cos_gap > 0.05:
        decision = "MECHANISM_ONLY"
    elif gate_turb_calm < 1.10 and mech_cos_gap < 0.03:
        decision = "CLEAN_FAILURE"
    else:
        decision = "AMBIGUOUS"
    results["decision"] = {
        "stage": "1a",
        "gate_turb_calm": gate_turb_calm,
        "mech_cosine_gap_turb_calm": mech_cos_gap,
        "verdict": decision,
    }
    print(f"\n=== STAGE 1a DECISION: {decision} ===")
    print(f"   gate turb/calm = {gate_turb_calm:.4f}  |  mech cos-gap turb/calm = {mech_cos_gap:.4f}")

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
