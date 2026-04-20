#!/usr/bin/env python
"""
220b: Multi-horizon path-quality suite for recursive one-day kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    HistoryFutureDictDataset,
    OneDayKernelRolloutWrapper,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    rollout_samples_in_batches,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_ci_coverage_tests,
    run_conditionality_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
    run_surface_validity_tests,
)


def run_sample_diversity_diagnostic(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, Any]:
    """Measure ensemble sample diversity to catch under-convergence (mode collapse).

    Args
    ----
    cond_samples: (N, K, T, 5, 5) generated ensembles
    ground_truth: (N, T, 5, 5) realized futures

    Returns
    -------
    dict with:
        - ensemble_std_h{1,15,30}: mean (over windows and cells) of std across K samples
            at horizon h. If this is << gt_marginal_std_h, the ensemble is under-diverse.
        - gt_marginal_std_h{1,15,30}: std over windows of GT future at horizon h, avg over cells.
        - std_ratio_h{1,15,30}: ensemble_std / gt_marginal_std. For a well-calibrated
            CONDITIONAL generator this should be in (0, 1] with the conditioning narrowing
            the distribution (law of total variance). A value near 0 indicates mode
            collapse; near 1 is consistent with an uninformative conditional; much >1
            signals over-dispersed samples (a different pathology).
        - ensemble_effective_rank_per_window: for each window flatten (K, 750) samples,
            compute SVD, effective rank = (Σσ)² / Σσ². Ceiling = min(K, 750). Low values
            indicate samples span a low-dim subspace (under-diversity). Averaged over
            windows.
        - ensemble_effective_rank_normalised: divided by min(K, 750) — in [0, 1].
        - gt_effective_rank_pooled: analogous rank on pooled (N, 750) GT marginal (not
            directly comparable per-window but a reference for how many independent
            directions the data itself uses).
        - cross_sample_l2_mean: average L2 distance between pairs of samples within a
            window, normalised by per-window GT scale. Cheap redundancy check.
    """
    N, K, T, H, W = cond_samples.shape
    assert ground_truth.shape == (N, T, H, W)
    D = H * W  # 25

    out: dict[str, Any] = {}
    horizons = [1, 15, 30]

    # Per-horizon std ratios
    for h in horizons:
        h_idx = min(h, T) - 1
        samp_h = cond_samples[:, :, h_idx]  # (N, K, 5, 5)
        # Ensemble std at horizon h: std across K for each (window, cell), then mean over cells and windows
        ens_std_h = float(samp_h.std(axis=1).mean())
        # GT marginal std: std across WINDOWS at horizon h, per cell, then mean over cells
        gt_h = ground_truth[:, h_idx]  # (N, 5, 5)
        gt_std_h = float(gt_h.std(axis=0).mean())
        ratio = ens_std_h / gt_std_h if gt_std_h > 0 else float("nan")
        out[f"ensemble_std_h{h}"] = ens_std_h
        out[f"gt_marginal_std_h{h}"] = gt_std_h
        out[f"std_ratio_h{h}"] = ratio

    # Effective rank per window
    #   For each window flatten (K, T*D) and compute SVD → effective rank.
    #   Center each window's samples before SVD so we measure the dispersion, not the mean.
    per_win_ranks = []
    for w in range(N):
        samp = cond_samples[w].reshape(K, -1)  # (K, T*D)
        samp = samp - samp.mean(axis=0, keepdims=True)
        # SVD is expensive for large matrices but K×(T*D) with K=48, T*D=750 is cheap
        try:
            s = np.linalg.svd(samp, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        s2 = s ** 2
        denom = float(s2.sum()) if s2.sum() > 0 else 1e-12
        eff_rank = (float(s.sum()) ** 2) / (denom + 1e-12)
        # Stable normalised form:
        if s2.sum() > 0:
            p = s2 / s2.sum()
            # entropic effective rank: exp(-Σ p log p)
            ent = float(-(p * np.log(p + 1e-12)).sum())
            entropic_rank = float(np.exp(ent))
        else:
            entropic_rank = 0.0
        per_win_ranks.append(entropic_rank)

    if per_win_ranks:
        out["ensemble_effective_rank_per_window"] = float(np.mean(per_win_ranks))
        out["ensemble_effective_rank_normalised"] = float(np.mean(per_win_ranks) / min(K, T * D))
        out["ensemble_effective_rank_per_window_std"] = float(np.std(per_win_ranks))
    else:
        out["ensemble_effective_rank_per_window"] = float("nan")
        out["ensemble_effective_rank_normalised"] = float("nan")

    # GT pooled effective rank (reference)
    gt_flat = ground_truth.reshape(N, -1)
    gt_flat = gt_flat - gt_flat.mean(axis=0, keepdims=True)
    try:
        s_gt = np.linalg.svd(gt_flat, compute_uv=False)
        p_gt = (s_gt ** 2) / (s_gt ** 2).sum()
        out["gt_effective_rank_pooled"] = float(np.exp(-(p_gt * np.log(p_gt + 1e-12)).sum()))
    except np.linalg.LinAlgError:
        out["gt_effective_rank_pooled"] = float("nan")

    # Cross-sample mean L2 (normalised)
    idx_i, idx_j = np.triu_indices(K, k=1)
    diffs = cond_samples[:, idx_i] - cond_samples[:, idx_j]  # (N, n_pairs, T, 5, 5)
    pairwise_l2 = np.sqrt((diffs ** 2).sum(axis=(-3, -2, -1)) + 1e-12)  # (N, n_pairs)
    out["cross_sample_l2_mean"] = float(pairwise_l2.mean())
    gt_scale = float(np.sqrt((ground_truth ** 2).sum(axis=(-3, -2, -1)) + 1e-12).mean())
    out["cross_sample_l2_normalised"] = out["cross_sample_l2_mean"] / gt_scale if gt_scale > 0 else float("nan")

    # Verdict flags (under-convergence focus)
    out["under_diverse_h30"] = bool(out["std_ratio_h30"] < 0.3)  # hard warning
    out["under_diverse_any_horizon"] = bool(min(out[f"std_ratio_h{h}"] for h in horizons) < 0.3)
    out["samples_low_rank"] = bool(out["ensemble_effective_rank_normalised"] < 0.2)
    return out


def suite_summary(results: dict[str, Any]) -> tuple[int, list[str]]:
    names = [
        ("surface", results["surface"]["overall_pass"]),
        ("coverage", results["coverage"]["overall_pass"]),
        ("conditionality", results["conditionality"]["overall_pass"]),
        ("distributional_fidelity", results["distributional_fidelity"]["overall_pass"]),
        ("cross_cell_correlation", results["cross_cell_correlation"]["overall_pass"]),
        ("mean_reversion", results["mean_reversion"]["overall_pass"]),
        ("pathwise_jump_realism", results["pathwise_jump_realism"]["overall_pass"]),
    ]
    failed = [name for name, passed in names if not passed]
    return sum(int(passed) for _name, passed in names), failed


def main() -> None:
    parser = argparse.ArgumentParser(description="220b recursive multi-horizon path suite")
    parser.add_argument("--model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--conditionality_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--conditionality_max_batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--force_native_anchor", action="store_true",
                        help="Force native-path rollout (model.sample_batched) with inference "
                             "anchor(0.50) for any 227a-family checkpoint. Use this to compare "
                             "229a honestly against 232 variants (which always use this regime).")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)

    # 232 variants train without anchor to preserve 229a's innovation-law calibration
    # (see 231 negative finding). They are *evaluated* with inference-time anchor(0.5),
    # which is the production recipe for the 229a family.
    if args.model_type in {'232a', '232b', '232c', '232d'}:
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 232 variant: use_scale_anchor=True, alpha=0.50 at inference")

    # 233a variants always use anchor (set in constructor). Keep it redundantly here
    # so --force_native_anchor still produces consistent settings regardless of how
    # the checkpoint was saved.
    if args.model_type.startswith('233a'):
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 233a {args.model_type}: use_scale_anchor=True, alpha=0.50 at inference")

    # 233a_v1_2 variants: anchor always on, same as v1
    if args.model_type.startswith("233a_v1_2"):
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 233a_v1_2 {args.model_type}: use_scale_anchor=True, alpha=0.50")

    # --force_native_anchor: apply the same regime to 227a/229a baselines for fair comparison
    if args.force_native_anchor and args.model_type == '227a':
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] --force_native_anchor: use_scale_anchor=True alpha=0.50, native path")

    wrapper = OneDayKernelRolloutWrapper(model).eval()

    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )

    # Use model's native sample_batched for models that generate multi-day trajectories
    # (recurrent/adapter 221d+, smooth transport 183c, etc.)
    use_native = hasattr(model, 'sample_batched') and (
        hasattr(model, 'temporal_adapter')
        or args.model_type == '183c'
        or args.model_type in {'231a', '231b', '231c', '232a', '232b', '232c', '232d'}
        or args.model_type.startswith('233a')
        or args.model_type.startswith('240a')
        or args.model_type.startswith('240b')
        or args.force_native_anchor
    )
    if use_native:
        from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
        print("Using native sample_batched (recurrent + adapter)")
        outputs = []
        n_steps = batch.future_01.shape[1]
        for start in range(0, batch.history_01.shape[0], args.batch_size):
            end = min(start + args.batch_size, batch.history_01.shape[0])
            hist_batch = normalize_iv(batch.history_01[start:end])
            with torch.no_grad():
                samp = model.sample_batched(
                    hist_batch, n_samples=args.samples, n_steps=n_steps, chunk_size=args.chunk_size,
                )
            outputs.append(samp.cpu().numpy())
        cond_samples = np.concatenate(outputs, axis=0)
    else:
        cond_samples = rollout_samples_in_batches(
            wrapper=wrapper,
            history_norm=batch.history_norm,
            n_samples=args.samples,
            n_steps=batch.future_01.shape[1],
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    surface = run_surface_validity_tests(cond_samples, ground_truth)
    coverage = run_ci_coverage_tests(cond_samples, ground_truth)

    cond_loader = DataLoader(
        HistoryFutureDictDataset(batch.history_norm.detach().cpu(), batch.future_norm.detach().cpu()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    # For native multi-day models, pass model directly (has sample_batched)
    # For one-day kernels, pass wrapper (wraps sample_next_iv)
    cond_model = model if use_native else wrapper
    conditionality = run_conditionality_tests(
        cond_model,
        cond_loader,
        n_samples=args.conditionality_samples,
        max_batches=args.conditionality_max_batches,
        device=str(device),
    )
    distributional = run_distributional_fidelity_tests(cond_samples, ground_truth, history_01)
    cross_cell = run_cross_cell_correlation_tests(cond_samples, ground_truth)
    mean_reversion = run_mean_reversion_tests(cond_samples, ground_truth, history_01)
    pathwise = run_pathwise_jump_realism_tests(cond_samples, ground_truth)
    sample_diversity = run_sample_diversity_diagnostic(cond_samples, ground_truth)
    print("\nSAMPLE DIVERSITY DIAGNOSTIC:")
    for h in (1, 15, 30):
        print(
            f"  h={h:2d}: ens_std={sample_diversity[f'ensemble_std_h{h}']:.4f} "
            f"gt_std={sample_diversity[f'gt_marginal_std_h{h}']:.4f} "
            f"ratio={sample_diversity[f'std_ratio_h{h}']:.3f}"
        )
    print(
        f"  Ensemble effective rank / window: "
        f"{sample_diversity['ensemble_effective_rank_per_window']:.2f} "
        f"(normalised: {sample_diversity['ensemble_effective_rank_normalised']:.3f})"
    )
    print(
        f"  GT pooled effective rank:         {sample_diversity['gt_effective_rank_pooled']:.2f}"
    )
    if sample_diversity.get("under_diverse_any_horizon"):
        print("  WARNING: under-diverse at >=1 horizon (std_ratio < 0.3)")
    if sample_diversity.get("samples_low_rank"):
        print("  WARNING: ensemble rank < 20% of ceiling (mode collapse?)")

    results = {
        "config": {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "n_windows": int(batch.history_norm.shape[0]),
            "samples": args.samples,
            "conditionality_samples": args.conditionality_samples,
        },
        "surface": surface,
        "coverage": coverage,
        "conditionality": conditionality,
        "distributional_fidelity": distributional,
        "cross_cell_correlation": cross_cell,
        "mean_reversion": mean_reversion,
        "pathwise_jump_realism": pathwise,
        "sample_diversity": sample_diversity,
    }
    n_pass, failed = suite_summary(results)
    results["summary"] = {
        "n_pass": n_pass,
        "n_total": 7,
        "failed_suites": failed,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        f"- suite score: `{n_pass}/7`",
        f"- failed suites: `{', '.join(failed) if failed else 'none'}`",
        "",
        "**Coverage**",
        f"- h1 cov90: `{coverage['per_horizon'].get(1, {}).get(0.9, float('nan')):.3f}`",
        f"- h30 cov90: `{coverage['per_horizon'].get(30, {}).get(0.9, float('nan')):.3f}`",
        f"- worst cell h30: `{coverage['worst_cell_per_horizon'].get(30, float('nan')):.3f}`",
        f"- best cell h30: `{coverage['best_cell_per_horizon'].get(30, float('nan')):.3f}`",
        "",
        "**Conditionality**",
        f"- turb/calm ratio: `{conditionality.get('turb_calm_ratio', float('nan')):.3f}`",
        f"- MAE reduction vs shuffled: `{conditionality.get('mae_reduction_pct', float('nan')):.1f}%`",
        "",
        "**Mean Reversion**",
        f"- aggregate slope ratio: `{mean_reversion.get('mr_gt_ratio', float('nan')):.3f}`",
        f"- active pass count: `{mean_reversion.get('active_pass_count', 0)}/{mean_reversion.get('active_cell_count', 0)}`",
        f"- full-horizon overall: `{mean_reversion.get('full_horizon', {}).get('overall_pass', False)}`",
        "",
        "**Distributional Fidelity**",
        f"- daily-change KS pass cells: `{distributional['ks_test']['n_pass']}/25`",
        f"- level KS pass cells: `{distributional['ks_level_test']['n_pass']}/25`",
        f"- worst floor occupancy: `{distributional['explosion']['worst_cell_floor']:.3%}`",
        "",
        "**Cross-Cell / Pathwise**",
        f"- corr ratio: `{cross_cell['corr_ratio']:.3f}`",
        f"- rank ratio: `{cross_cell['rank_ratio']:.3f}`",
        f"- max-jump KS: `{pathwise['pathwise_max_jump']['ks_stat']:.3f}`",
        "",
        "**Sample Diversity (under-convergence guard)**",
        f"- std_ratio h1 / h15 / h30: "
        f"`{sample_diversity['std_ratio_h1']:.3f} / "
        f"{sample_diversity['std_ratio_h15']:.3f} / "
        f"{sample_diversity['std_ratio_h30']:.3f}` (target > 0.3)",
        f"- ensemble eff rank / window: `{sample_diversity['ensemble_effective_rank_per_window']:.2f}` "
        f"(normalised `{sample_diversity['ensemble_effective_rank_normalised']:.3f}`, target > 0.2)",
        f"- GT pooled eff rank: `{sample_diversity['gt_effective_rank_pooled']:.2f}`",
        f"- under-diverse warning: `{sample_diversity.get('under_diverse_any_horizon', False)}`",
        f"- low-rank warning: `{sample_diversity.get('samples_low_rank', False)}`",
    ]
    write_markdown_summary(args.output_md, "220b Multi-Horizon Path Suite", lines)
    print(json.dumps(make_serializable(results["summary"]), indent=2))


if __name__ == "__main__":
    main()
