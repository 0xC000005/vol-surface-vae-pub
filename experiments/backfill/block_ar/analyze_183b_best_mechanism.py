#!/usr/bin/env python
"""
Focused mechanism review for 183b_best.

Goal:
  Explain why 183b_best preserves the 8/11 frontier but still fails S3/S4/S7,
  and determine the narrowest next principled fix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.analyze_183a_final_mechanism import (
    cell_label,
    classify_cell_failure,
    distribution_compare,
    quantile_profile,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_183a_integrated_width_tail_transport import (
    compute_control_targets,
)
from experiments.backfill.block_ar.train_183b_state_dependent_radial_transport import (
    StateDependentRadialTransportModel,
)


BANDS = ["low", "mid", "high"]
SELECT_HORIZONS = [1, 7, 14, 30]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "state_dependent_radial_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183b"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = StateDependentRadialTransportModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        path_config=raw_config["path"],
        prior_config=raw_config["prior"],
        integrated_config=raw_config["integrated"],
        state_config=raw_config["state"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
        mix_chunk_size=raw_config.get("mix_chunk_size", 27),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def analyze_183b_best(
    model: StateDependentRadialTransportModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    review_183a: dict[str, Any],
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)
    n_windows, future_len, n_cells = future_01.shape

    vov, q20, q80 = regime_masks_from_history(history_norm.cpu())
    calm_mask = vov <= q20
    turb_mask = vov >= q80

    sample_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    sample_std = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90 = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    det_mean = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    target_local_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    raw_local_zero_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    eff_local_zero_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    raw_local_target_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    eff_local_target_all = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    local_gate_all = np.zeros((n_windows, 1), dtype=np.float32)

    raw_band_zero_all = np.zeros((n_windows, 3), dtype=np.float32)
    eff_band_zero_all = np.zeros((n_windows, 3), dtype=np.float32)
    raw_band_target_all = np.zeros((n_windows, 3), dtype=np.float32)
    eff_band_target_all = np.zeros((n_windows, 3), dtype=np.float32)
    band_gate_all = np.zeros((n_windows, 3), dtype=np.float32)

    pooled_abs_delta_gt = []
    pooled_abs_delta_gen = []
    path_max_jump_gt = []
    path_max_jump_gen = []
    path_q99_exceed_count_gt = []
    path_q99_exceed_count_gen = []
    band_energy_gt = {k: [] for k in BANDS}
    band_energy_gen = {k: [] for k in BANDS}

    geometry = model.path_geometry
    band_masks = {
        "low": geometry.low_band_mask().reshape(1, future_len, n_cells).to(device),
        "mid": geometry.mid_band_mask().reshape(1, future_len, n_cells).to(device),
        "high": geometry.high_band_mask().reshape(1, future_len, n_cells).to(device),
    }

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        fut_01_b = future_01[start:end].to(device)
        batch = end - start

        (
            mu_u,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_01_b)
        det_iv = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi)
        det_mean[start:end] = det_iv.detach().cpu().numpy()

        target_u = iv_to_unconstrained(
            fut_01_b,
            lo=model.support_lo,
            hi=model.support_hi,
            eps=model.support_eps,
        )
        target_basis = model.teacher_basis_flat_from_outputs(
            target_u=target_u,
            mu=mu_u,
            time_factor=time_factor,
            time_diag=time_diag,
            cell_factor=cell_factor,
            cell_diag=cell_diag,
            scale=scale,
            base_local_delta=base_local_delta,
            block_logits=block_logits,
        ).view(batch, future_len, n_cells)
        target_basis_flat = target_basis.reshape(batch, -1)
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        target_local_log, _target_band_log = compute_control_targets(model, target_basis)
        target_local_all[start:end] = target_local_log.detach().cpu().numpy()

        t = torch.full((batch,), 0.5, device=device, dtype=target_basis.dtype)
        zero_state = torch.zeros_like(target_basis_flat)
        raw_local_zero, raw_band_zero, eff_local_zero, eff_band_zero, _lg0, _bg0 = model.build_state_dependent_controls(
            zero_state, t, path_context
        )
        raw_local_t, raw_band_t, eff_local_t, eff_band_t, lg_t, bg_t = model.build_state_dependent_controls(
            target_basis_flat, t, path_context
        )

        raw_local_zero_all[start:end] = raw_local_zero.detach().cpu().numpy()
        eff_local_zero_all[start:end] = eff_local_zero.detach().cpu().numpy()
        raw_local_target_all[start:end] = raw_local_t.detach().cpu().numpy()
        eff_local_target_all[start:end] = eff_local_t.detach().cpu().numpy()
        local_gate_all[start:end] = lg_t.detach().cpu().numpy()

        raw_band_zero_all[start:end] = raw_band_zero.detach().cpu().numpy()
        eff_band_zero_all[start:end] = eff_band_zero.detach().cpu().numpy()
        raw_band_target_all[start:end] = raw_band_t.detach().cpu().numpy()
        eff_band_target_all[start:end] = eff_band_t.detach().cpu().numpy()
        band_gate_all[start:end] = bg_t.detach().cpu().numpy()

        samples_u = model.sample_future_u(hist_01_b, n_samples=n_samples)
        samples_01 = unconstrained_to_iv(samples_u, lo=model.support_lo, hi=model.support_hi)
        samples_iv = samples_01.detach().cpu().numpy()
        sample_mean[start:end] = samples_iv.mean(axis=1)
        sample_std[start:end] = samples_iv.std(axis=1)
        lo90[start:end] = np.quantile(samples_iv, 0.05, axis=1)
        hi90[start:end] = np.quantile(samples_iv, 0.95, axis=1)

        gt_path = torch.cat([hist_01_b[:, -1:].reshape(batch, 1, 5, 5), fut_01_b.view(batch, future_len, 5, 5)], dim=1)
        gen_path = torch.cat(
            [
                hist_01_b[:, -1:].reshape(batch, 1, 1, 5, 5).expand(-1, n_samples, -1, -1, -1),
                samples_01.view(batch, n_samples, future_len, 5, 5),
            ],
            dim=2,
        )
        gt_diff = gt_path[:, 1:] - gt_path[:, :-1]
        gen_diff = gen_path[:, :, 1:] - gen_path[:, :, :-1]

        pooled_abs_delta_gt.append(gt_diff.abs().reshape(-1).cpu().numpy())
        pooled_abs_delta_gen.append(gen_diff.abs().reshape(-1).cpu().numpy())
        path_max_jump_gt.append(gt_diff.abs().amax(dim=(1, 2, 3)).cpu().numpy())
        path_max_jump_gen.append(gen_diff.abs().amax(dim=(2, 3, 4)).reshape(-1).cpu().numpy())

        gt_q99 = torch.quantile(gt_diff.abs().reshape(-1), 0.99)
        path_q99_exceed_count_gt.append((gt_diff.abs().reshape(batch, -1) > gt_q99).sum(dim=1).cpu().numpy())
        path_q99_exceed_count_gen.append((gen_diff.abs().reshape(batch * n_samples, -1) > gt_q99).sum(dim=1).cpu().numpy())

        gt_diff_flat = gt_diff.reshape(batch, future_len, n_cells)
        gen_diff_flat = gen_diff.reshape(batch * n_samples, future_len, n_cells)
        gt_coeff = geometry.to_basis(gt_diff_flat)
        gen_coeff = geometry.to_basis(gen_diff_flat)
        gt_total_energy = gt_coeff.reshape(batch, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        gen_total_energy = gen_coeff.reshape(batch * n_samples, -1).pow(2).sum(dim=1).clamp_min(1e-12)
        for band in BANDS:
            mask = band_masks[band].to(dtype=gt_coeff.dtype)
            gt_band_coeff = (gt_coeff * mask).reshape(batch, -1)
            gen_band_coeff = (gen_coeff * mask).reshape(batch * n_samples, -1)
            gt_share = (gt_band_coeff.pow(2).sum(dim=1) / gt_total_energy).cpu().numpy()
            gen_share = (gen_band_coeff.pow(2).sum(dim=1) / gen_total_energy).cpu().numpy()
            band_energy_gt[band].append(gt_share)
            band_energy_gen[band].append(gen_share)

    future_np = future_01.cpu().numpy()
    inside90 = (future_np >= lo90) & (future_np <= hi90)
    det_resid = future_np - det_mean
    z_mean = det_resid / np.maximum(sample_std, 1e-6)

    turb_cov = inside90[turb_mask]
    turb_std = sample_std[turb_mask]
    turb_gt = future_np[turb_mask]
    turb_det = det_mean[turb_mask]
    turb_det_resid = turb_gt - turb_det

    hard_entries = []
    over_entries = []
    for h in (7, 14, 30):
        h_idx = h - 1
        for c in range(n_cells):
            cov = float(turb_cov[:, h_idx, c].mean())
            gen_std = float(turb_std[:, h_idx, c].mean())
            gt_std = float(np.std(turb_det_resid[:, h_idx, c]))
            ratio = gen_std / max(gt_std, 1e-6)
            z = float(z_mean[turb_mask, h_idx, c].mean())
            target_local = float(target_local_all[turb_mask, h_idx, c].mean())
            raw_zero = float(raw_local_zero_all[turb_mask, h_idx, c].mean())
            raw_target = float(raw_local_target_all[turb_mask, h_idx, c].mean())
            eff_zero = float(eff_local_zero_all[turb_mask, h_idx, c].mean())
            eff_target = float(eff_local_target_all[turb_mask, h_idx, c].mean())
            local_gate = float(local_gate_all[turb_mask].mean())
            entry = {
                "regime": "turb",
                "horizon": h,
                "cell": cell_label(c),
                "coverage_90": cov,
                "z_mean": z,
                "gen_std_iv": gen_std,
                "gt_resid_std_iv": gt_std,
                "std_ratio_gen_to_gt": ratio,
                "target_local_log": target_local,
                "pred_raw_local_zero": raw_zero,
                "pred_raw_local_target": raw_target,
                "pred_eff_local_zero": eff_zero,
                "pred_eff_local_target": eff_target,
                "state_uplift_eff_local": eff_target - eff_zero,
                "local_gate_mean": local_gate,
                "classification": classify_cell_failure(cov, z, ratio),
            }
            if cov < 0.70:
                hard_entries.append(entry)
            if cov > 0.95:
                over_entries.append(entry)

    hard_entries.sort(key=lambda x: x["coverage_90"])
    over_entries.sort(key=lambda x: (-x["coverage_90"], -x["std_ratio_gen_to_gt"]))

    review_slices = review_183a["width_allocation_review"]["top_turbulent_hard_slices"][:12]
    slice_progress = []
    for item in review_slices:
        h = int(item["horizon"])
        c = item["cell"][0] * 5 + item["cell"][1]
        h_idx = h - 1
        cov = float(turb_cov[:, h_idx, c].mean())
        gen_std = float(turb_std[:, h_idx, c].mean())
        gt_std = float(np.std(turb_det_resid[:, h_idx, c]))
        ratio = gen_std / max(gt_std, 1e-6)
        slice_progress.append(
            {
                "horizon": h,
                "cell": item["cell"],
                "coverage_183a": float(item["coverage_90"]),
                "coverage_183b": cov,
                "coverage_delta": cov - float(item["coverage_90"]),
                "std_ratio_183a": float(item["std_ratio_gen_to_gt"]),
                "std_ratio_183b": ratio,
                "std_ratio_delta": ratio - float(item["std_ratio_gen_to_gt"]),
                "target_local_log_183b": float(target_local_all[turb_mask, h_idx, c].mean()),
                "pred_eff_local_target_183b": float(eff_local_target_all[turb_mask, h_idx, c].mean()),
                "state_uplift_eff_local_183b": float(
                    (eff_local_target_all[turb_mask, h_idx, c] - eff_local_zero_all[turb_mask, h_idx, c]).mean()
                ),
            }
        )

    target_local_flat = target_local_all[turb_mask].reshape(-1)
    raw_local_target_flat = raw_local_target_all[turb_mask].reshape(-1)
    eff_local_target_flat = eff_local_target_all[turb_mask].reshape(-1)
    raw_local_zero_flat = raw_local_zero_all[turb_mask].reshape(-1)
    eff_local_zero_flat = eff_local_zero_all[turb_mask].reshape(-1)

    hard_mask = np.zeros((future_len, n_cells), dtype=bool)
    for item in hard_entries[:12]:
        hard_mask[int(item["horizon"]) - 1, item["cell"][0] * 5 + item["cell"][1]] = True
    over_mask = np.zeros((future_len, n_cells), dtype=bool)
    for item in over_entries[:8]:
        over_mask[int(item["horizon"]) - 1, item["cell"][0] * 5 + item["cell"][1]] = True

    pooled_gt = np.concatenate(pooled_abs_delta_gt)
    pooled_gen = np.concatenate(pooled_abs_delta_gen)
    gt_path_max = np.concatenate(path_max_jump_gt)
    gen_path_max = np.concatenate(path_max_jump_gen)
    gt_q99_exceed = np.concatenate(path_q99_exceed_count_gt)
    gen_q99_exceed = np.concatenate(path_q99_exceed_count_gen)

    band_summary = {}
    for i, band in enumerate(BANDS):
        band_summary[band] = {
            "energy_ratio_p50": float(
                np.quantile(np.concatenate(band_energy_gen[band]), 0.5)
                / max(np.quantile(np.concatenate(band_energy_gt[band]), 0.5), 1e-12)
            ),
            "raw_zero_mean_turb": float(raw_band_zero_all[turb_mask, i].mean()),
            "raw_target_mean_turb": float(raw_band_target_all[turb_mask, i].mean()),
            "eff_zero_mean_turb": float(eff_band_zero_all[turb_mask, i].mean()),
            "eff_target_mean_turb": float(eff_band_target_all[turb_mask, i].mean()),
            "raw_zero_mean_calm": float(raw_band_zero_all[calm_mask, i].mean()),
            "raw_target_mean_calm": float(raw_band_target_all[calm_mask, i].mean()),
            "eff_zero_mean_calm": float(eff_band_zero_all[calm_mask, i].mean()),
            "eff_target_mean_calm": float(eff_band_target_all[calm_mask, i].mean()),
            "state_uplift_eff_turb": float((eff_band_target_all[turb_mask, i] - eff_band_zero_all[turb_mask, i]).mean()),
            "state_uplift_eff_calm": float((eff_band_target_all[calm_mask, i] - eff_band_zero_all[calm_mask, i]).mean()),
            "band_gate_mean_turb": float(band_gate_all[turb_mask, i].mean()),
            "band_gate_mean_calm": float(band_gate_all[calm_mask, i].mean()),
        }

    return {
        "suite_context": {
            "best_passes": ["surface", "coverage", "block_ar", "cointegration", "distributional", "cross_cell_correlation", "mean_reversion", "pathwise_jump_realism"],
            "best_fails": ["conditionality", "time_series", "regime_coverage"],
        },
        "control_strength_review": {
            "local_strength": float(torch.sigmoid(model.local_strength_logit).item()),
            "band_strength": float(torch.sigmoid(model.band_strength_logit).item()),
            "target_local_abs_mean_turb": float(np.abs(target_local_flat).mean()),
            "pred_raw_local_abs_mean_turb_zero": float(np.abs(raw_local_zero_flat).mean()),
            "pred_raw_local_abs_mean_turb_target": float(np.abs(raw_local_target_flat).mean()),
            "pred_eff_local_abs_mean_turb_zero": float(np.abs(eff_local_zero_flat).mean()),
            "pred_eff_local_abs_mean_turb_target": float(np.abs(eff_local_target_flat).mean()),
            "raw_target_corr_turb_zero_state": _corr(raw_local_zero_flat, target_local_flat),
            "raw_target_corr_turb_target_state": _corr(raw_local_target_flat, target_local_flat),
            "eff_target_corr_turb_zero_state": _corr(eff_local_zero_flat, target_local_flat),
            "eff_target_corr_turb_target_state": _corr(eff_local_target_flat, target_local_flat),
            "mean_eff_local_hard_slices": float(eff_local_target_all[turb_mask][:, hard_mask].mean()) if hard_mask.any() else float("nan"),
            "mean_eff_local_overwide_slices": float(eff_local_target_all[turb_mask][:, over_mask].mean()) if over_mask.any() else float("nan"),
            "mean_target_local_hard_slices": float(target_local_all[turb_mask][:, hard_mask].mean()) if hard_mask.any() else float("nan"),
            "mean_target_local_overwide_slices": float(target_local_all[turb_mask][:, over_mask].mean()) if over_mask.any() else float("nan"),
            "mean_state_uplift_eff_hard_slices": float(
                (eff_local_target_all[turb_mask][:, hard_mask] - eff_local_zero_all[turb_mask][:, hard_mask]).mean()
            ) if hard_mask.any() else float("nan"),
            "mean_state_uplift_eff_overwide_slices": float(
                (eff_local_target_all[turb_mask][:, over_mask] - eff_local_zero_all[turb_mask][:, over_mask]).mean()
            ) if over_mask.any() else float("nan"),
            "local_gate_mean_turb": float(local_gate_all[turb_mask].mean()),
            "local_gate_mean_calm": float(local_gate_all[calm_mask].mean()),
        },
        "width_allocation_review": {
            "top_turbulent_hard_slices": hard_entries[:12],
            "top_overwide_slices": over_entries[:8],
            "progress_vs_183a_hard_slices": slice_progress,
        },
        "tail_shape_review": {
            "pooled_abs_delta": distribution_compare(pooled_gt, pooled_gen),
            "pathwise_max_jump": distribution_compare(gt_path_max, gen_path_max),
            "gt_q99_exceed_count": quantile_profile(gt_q99_exceed),
            "gen_q99_exceed_count": quantile_profile(gen_q99_exceed),
        },
        "band_review": band_summary,
        "diagnosis": {
            "s3_s7": "State dependence is active, but the remaining hard turbulent late-horizon cells are still materially under-allocated. The effective local controls move in the right direction, yet their magnitude remains tiny relative to target local allocation.",
            "s4_vs_s11": "Pathwise jump realism remains acceptable, but pooled tail concentration still fails because residual energy is still distributed across too many moderate moves, especially through the high band, rather than concentrated with the right radial tail shape.",
            "state_control_limit": "The new gates are alive, but global local/band strengths stay very small. That means state-conditioned raw controls do not translate into large enough effective amplitude changes on the exact hard slices.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 183b_best mechanism")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--review_183a", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    model, _checkpoint = load_model(args.checkpoint, args.device)
    _summary = load_json(args.summary)
    review_183a = load_json(args.review_183a)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    out = analyze_183b_best(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        review_183a=review_183a,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(out), indent=2))
    print(f"Wrote mechanism review to {output_path}")


if __name__ == "__main__":
    main()
