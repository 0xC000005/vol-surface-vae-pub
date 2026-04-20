#!/usr/bin/env python
"""
241-series diagnostic: CONTROL-PATH regime discriminability.

Motivation
----------
The 241a velocity-cosine probe reported cosine_gap_turb_calm=0.0026 (target > 0.05),
suggesting the velocity field is NOT regime-discriminable. Yet the gate metric
(turb/calm width ratio) moved +0.037 under contrastive-FM. Hypothesis: the
contrastive objective acts through the STATE-METRIC CONTROL PATH
(metric_local, metric_band, width_allocator outputs), not through the main
velocity field.

This script measures per-regime separation of the control-path outputs of
`build_state_metric_controls(z_t, t, path_context)` for both 183c (baseline)
and 241a (contrastive). It answers:

    "Did 241a's control-path outputs become more regime-discriminable than
     183c's? If yes, by how much?"

Outputs (per checkpoint, per control tensor): per-regime means, discriminability
scalars, and a delta row (241a - 183c) for the headline comparison.
A (5,5) per-cell heatmap of time-averaged metric_local for each regime is also
written.

Usage
-----
PYTHONPATH=. python experiments/backfill/block_ar/diagnose_241_control_path.py \\
    --checkpoint models/backfill/241a_contrastive_fm_s42/best_model.pt \\
    --baseline   models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt \\
    --output_dir results/block_ar/241a/control_path_diag --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


# ---------------------------------------------------------------------------
# Regime labelling (RV-based vov proxy, same as diagnose_241_regime_separation)
# ---------------------------------------------------------------------------
def vov_proxy(history: torch.Tensor) -> torch.Tensor:
    diffs = history[:, 1:] - history[:, :-1]
    return (diffs ** 2).mean(dim=(1, 2, 3))


def assign_regimes(vov: torch.Tensor, q_lo: float = 0.20, q_hi: float = 0.80) -> torch.Tensor:
    vov_np = vov.cpu().numpy()
    lo_thresh = float(np.quantile(vov_np, q_lo))
    hi_thresh = float(np.quantile(vov_np, q_hi))
    labels = torch.zeros_like(vov, dtype=torch.long)
    labels[vov <= lo_thresh] = 1  # calm
    labels[vov >= hi_thresh] = 2  # turb
    return labels


# ---------------------------------------------------------------------------
# Checkpoint loading (tolerates 183c's missing cov_jitter key)
# ---------------------------------------------------------------------------
def build_model_from_ckpt(ckpt_path: str, device: str) -> StateMetricTransportModel:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    cov_jitter = cfg.get("cov_jitter", 1e-5) or 1e-5
    model = StateMetricTransportModel(
        encoder_config=encoder_config,
        decoder_config=cfg["decoder"],
        flow_config=cfg["flow"],
        path_config=cfg["path"],
        prior_config=cfg["prior"],
        integrated_config=cfg["integrated"],
        state_config=cfg["state"],
        metric_config=cfg["metric"],
        support_lo=cfg["support_lo"],
        support_hi=cfg["support_hi"],
        support_eps=cfg["support_eps"],
        cov_jitter=cov_jitter,
        base_nu=cfg["base_nu"],
        mix_chunk_size=cfg["mix_chunk_size"],
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core probe: control-path outputs per window, averaged over noise draws
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_control_outputs(
    model: StateMetricTransportModel,
    val_hist: torch.Tensor,
    n_trials: int = 4,
    t_value: float = 0.5,
    batch_size: int = 32,
) -> dict:
    """
    For each window and each of `n_trials` independent z_t draws, evaluate
    `build_state_metric_controls(z_t, t=0.5, path_context)` and average the
    four control-path outputs over the noise dim.

    Returns:
        {
            "metric_local": (N, n_frames, n_cells) mean over trials,
            "metric_band":  (N, 3),
            "local_gate":   (N, 1),
            "band_gate":    (N, 3),
        }
    """
    device = next(model.parameters()).device
    n_windows = val_hist.shape[0]

    all_metric_local = []
    all_metric_band = []
    all_local_gate = []
    all_band_gate = []

    for i in range(0, n_windows, batch_size):
        batch = val_hist[i : i + batch_size].to(device)
        B = batch.shape[0]

        # history → static path_context (fixed across trials for this batch)
        outputs = model.forward_from_history(batch)
        (mu, time_factor, time_diag, cell_factor, cell_diag, scale,
         flow_context, base_local_delta, block_logits) = outputs
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)

        # Accumulate control outputs across trials, then average over trial axis
        stack_ml = torch.zeros(B, model.decoder.n_frames, model.decoder.n_cells, device=device)
        stack_mb = torch.zeros(B, 3, device=device)
        stack_lg = torch.zeros(B, 1, device=device)
        stack_bg = torch.zeros(B, 3, device=device)

        for trial in range(n_trials):
            torch.manual_seed(1000 + trial)  # paired seeds across checkpoints
            z0, _ = model.prior.sample(B, device=device, dtype=batch.dtype)
            t = torch.full((B,), t_value, device=device, dtype=batch.dtype)
            # We only need the conditional outputs of build_state_metric_controls;
            # use z_t = 0.5 * z0 as a proxy midpoint (same choice as the cosine probe).
            z_t = 0.5 * z0
            raw_local, raw_band, metric_local, metric_band, local_gate, band_gate = \
                model.build_state_metric_controls(z_t, t, path_context)
            stack_ml = stack_ml + metric_local
            stack_mb = stack_mb + metric_band
            stack_lg = stack_lg + local_gate
            stack_bg = stack_bg + band_gate

        stack_ml = stack_ml / n_trials
        stack_mb = stack_mb / n_trials
        stack_lg = stack_lg / n_trials
        stack_bg = stack_bg / n_trials

        all_metric_local.append(stack_ml.detach().cpu())
        all_metric_band.append(stack_mb.detach().cpu())
        all_local_gate.append(stack_lg.detach().cpu())
        all_band_gate.append(stack_bg.detach().cpu())

    return {
        "metric_local": torch.cat(all_metric_local, dim=0),
        "metric_band": torch.cat(all_metric_band, dim=0),
        "local_gate": torch.cat(all_local_gate, dim=0),
        "band_gate": torch.cat(all_band_gate, dim=0),
    }


# ---------------------------------------------------------------------------
# Per-regime statistics and discriminability
# ---------------------------------------------------------------------------
def _safe_rms_gap(diff: np.ndarray) -> float:
    # RMS of the per-dim gap; scale-agnostic across tensor sizes
    return float(np.sqrt(np.mean(np.square(diff))))


def _safe_cohen_d(turb: np.ndarray, calm: np.ndarray) -> float:
    """Cohen's d using RMS gap over pooled RMS std, treating all tensor dims as features."""
    if turb.shape[0] < 2 or calm.shape[0] < 2:
        return float("nan")
    turb_mean = turb.mean(axis=0)
    calm_mean = calm.mean(axis=0)
    diff = turb_mean - calm_mean
    gap = np.sqrt(np.mean(np.square(diff)))
    # Pooled per-dim std (RMS of per-dim stds across both groups)
    turb_std = turb.std(axis=0)
    calm_std = calm.std(axis=0)
    pooled = np.sqrt(0.5 * (np.mean(np.square(turb_std)) + np.mean(np.square(calm_std))))
    if pooled < 1e-10:
        return float("inf") if gap > 1e-10 else 0.0
    return float(gap / pooled)


def regime_summary(tensor: torch.Tensor, labels: torch.Tensor, name: str) -> dict:
    """
    tensor: (N, *feat) on CPU
    labels: (N,) with 0=mid, 1=calm, 2=turb
    """
    arr = tensor.numpy()
    lbl = labels.cpu().numpy()
    turb = arr[lbl == 2]
    calm = arr[lbl == 1]
    mid = arr[lbl == 0]

    turb_mean = turb.mean(axis=0) if turb.size else np.full(arr.shape[1:], np.nan)
    calm_mean = calm.mean(axis=0) if calm.size else np.full(arr.shape[1:], np.nan)
    mid_mean = mid.mean(axis=0) if mid.size else np.full(arr.shape[1:], np.nan)

    diff_tc = turb_mean - calm_mean
    rms_gap_tc = _safe_rms_gap(diff_tc) if turb.size and calm.size else float("nan")
    cohen_d_tc = _safe_cohen_d(turb, calm) if turb.size and calm.size else float("nan")

    # Overall magnitudes for sanity
    turb_abs_mean = float(np.mean(np.abs(turb_mean))) if turb.size else float("nan")
    calm_abs_mean = float(np.mean(np.abs(calm_mean))) if calm.size else float("nan")
    turb_std_global = float(turb.std()) if turb.size else float("nan")
    calm_std_global = float(calm.std()) if calm.size else float("nan")

    return {
        "tensor_name": name,
        "tensor_shape": list(arr.shape[1:]),
        "n_turb": int((lbl == 2).sum()),
        "n_calm": int((lbl == 1).sum()),
        "n_middle": int((lbl == 0).sum()),
        "turb_abs_mean": turb_abs_mean,
        "calm_abs_mean": calm_abs_mean,
        "turb_std_global": turb_std_global,
        "calm_std_global": calm_std_global,
        "rms_gap_turb_calm": rms_gap_tc,
        "cohen_d_turb_calm": cohen_d_tc,
        # Flattened means for completeness; safe for small tensors
        "turb_mean_flat": turb_mean.flatten().tolist(),
        "calm_mean_flat": calm_mean.flatten().tolist(),
        "mid_mean_flat": mid_mean.flatten().tolist(),
    }


def per_cell_turb_calm_maps(metric_local: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    metric_local: (N, 30, 25). Average over the 30 frames → (N, 25) per-window,
    then regime-split → 5x5 maps for turb / calm / (turb-calm).
    """
    time_avg = metric_local.mean(dim=1).numpy()  # (N, 25)
    lbl = labels.cpu().numpy()
    turb = time_avg[lbl == 2]
    calm = time_avg[lbl == 1]
    if turb.size == 0 or calm.size == 0:
        return {
            "turb_map_5x5": None,
            "calm_map_5x5": None,
            "diff_map_5x5": None,
            "max_abs_diff": float("nan"),
            "rms_diff": float("nan"),
        }
    turb_map = turb.mean(axis=0).reshape(5, 5)
    calm_map = calm.mean(axis=0).reshape(5, 5)
    diff_map = turb_map - calm_map
    return {
        "turb_map_5x5": turb_map.tolist(),
        "calm_map_5x5": calm_map.tolist(),
        "diff_map_5x5": diff_map.tolist(),
        "max_abs_diff": float(np.max(np.abs(diff_map))),
        "rms_diff": float(np.sqrt(np.mean(np.square(diff_map)))),
    }


# ---------------------------------------------------------------------------
# Main per-checkpoint routine
# ---------------------------------------------------------------------------
def run_for_checkpoint(
    ckpt_path: str,
    label_tag: str,
    val_hist: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    n_trials: int,
    t_value: float,
) -> dict:
    print(f"\n{'=' * 72}")
    print(f"[{label_tag}] Diagnosing control path: {ckpt_path}")
    print(f"{'=' * 72}")
    model = build_model_from_ckpt(ckpt_path, device)
    print(f"  n_frames={model.decoder.n_frames}  n_cells={model.decoder.n_cells}")
    print(f"  path_context_dim={model.path_config['context_dim']}")

    print(f"[1/2] Collecting control outputs (n_trials={n_trials}, t={t_value}) ...")
    outputs = collect_control_outputs(model, val_hist, n_trials=n_trials, t_value=t_value)

    print("[2/2] Regime-partitioned summaries ...")
    summaries = {}
    for name in ("metric_local", "metric_band", "local_gate", "band_gate"):
        s = regime_summary(outputs[name], labels, name)
        summaries[name] = s
        print(
            f"  {name:<13}  shape={s['tensor_shape']}  "
            f"rms_gap(turb-calm)={s['rms_gap_turb_calm']:.4f}  "
            f"cohen_d={s['cohen_d_turb_calm']:.3f}  "
            f"turb|mean|={s['turb_abs_mean']:.4f}  calm|mean|={s['calm_abs_mean']:.4f}"
        )

    # Per-cell 5x5 map from metric_local (time-averaged then regime-split)
    heatmap = per_cell_turb_calm_maps(outputs["metric_local"], labels)
    if heatmap["diff_map_5x5"] is not None:
        print(
            f"  per-cell metric_local (time-avg): "
            f"max|turb-calm|={heatmap['max_abs_diff']:.4f}  "
            f"rms|turb-calm|={heatmap['rms_diff']:.4f}"
        )

    # Also log signed gate magnitudes (near-zero ⇒ gate dead ⇒ state path dormant)
    lg = outputs["local_gate"].numpy()
    bg = outputs["band_gate"].numpy()
    gate_mag = {
        "local_gate_mean_abs": float(np.mean(np.abs(lg))),
        "local_gate_global_std": float(np.std(lg)),
        "band_gate_mean_abs": float(np.mean(np.abs(bg))),
        "band_gate_global_std": float(np.std(bg)),
    }
    print(
        f"  gate magnitudes: local|mean|={gate_mag['local_gate_mean_abs']:.4f}  "
        f"band|mean|={gate_mag['band_gate_mean_abs']:.4f}"
    )

    return {
        "ckpt_path": ckpt_path,
        "label_tag": label_tag,
        "summaries": summaries,
        "per_cell_heatmap": heatmap,
        "gate_magnitudes": gate_mag,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(results: dict) -> str:
    args = results["args"]
    main = results["checkpoints"]["main"]
    base = results["checkpoints"].get("baseline")

    lines = []
    lines.append("# 241a Control-Path Regime Discriminability Diagnostic\n")
    lines.append(f"- Main checkpoint (241a): `{main['ckpt_path']}`")
    if base is not None:
        lines.append(f"- Baseline (183c):        `{base['ckpt_path']}`")
    lines.append(f"- Val windows: {args['max_val_windows']}  | n_trials: {args['n_trials']}  | t: {args['t_value']}")
    lines.append(f"- Regime split: q_lo={args['q_lo']}, q_hi={args['q_hi']}")
    first = next(iter(main["summaries"].values()))
    lines.append(f"- Counts: turb={first['n_turb']}  calm={first['n_calm']}  mid={first['n_middle']}\n")

    # Headline table: RMS gap (turb-calm) and Cohen's d per tensor, per checkpoint
    lines.append("## Headline: regime gaps on control outputs\n")
    if base is not None:
        lines.append("| tensor | shape | gap_183c | gap_241a | Δ(241a − 183c) | gain ×  | d_183c | d_241a |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name in ("metric_local", "metric_band", "local_gate", "band_gate"):
            s_m = main["summaries"][name]
            s_b = base["summaries"][name]
            g_m = s_m["rms_gap_turb_calm"]
            g_b = s_b["rms_gap_turb_calm"]
            delta = g_m - g_b
            ratio = float("nan") if g_b <= 1e-10 else g_m / g_b
            d_b = s_b["cohen_d_turb_calm"]
            d_m = s_m["cohen_d_turb_calm"]
            lines.append(
                f"| `{name}` | {s_m['tensor_shape']} | {g_b:.4f} | {g_m:.4f} | {delta:+.4f} | "
                f"{ratio:.2f} | {d_b:.3f} | {d_m:.3f} |"
            )
    else:
        lines.append("| tensor | shape | gap_241a | cohen_d_241a |")
        lines.append("|---|---|---|---|")
        for name in ("metric_local", "metric_band", "local_gate", "band_gate"):
            s_m = main["summaries"][name]
            lines.append(
                f"| `{name}` | {s_m['tensor_shape']} | "
                f"{s_m['rms_gap_turb_calm']:.4f} | {s_m['cohen_d_turb_calm']:.3f} |"
            )
    lines.append("")

    # Gate magnitudes
    lines.append("## Gate magnitudes (dormant if ≈ 0)\n")
    if base is not None:
        lines.append("| quantity | 183c | 241a | Δ |")
        lines.append("|---|---|---|---|")
        for key in ("local_gate_mean_abs", "local_gate_global_std", "band_gate_mean_abs", "band_gate_global_std"):
            gm = main["gate_magnitudes"][key]
            gb = base["gate_magnitudes"][key]
            lines.append(f"| {key} | {gb:.4f} | {gm:.4f} | {gm - gb:+.4f} |")
    else:
        lines.append("| quantity | 241a |")
        lines.append("|---|---|")
        for key, v in main["gate_magnitudes"].items():
            lines.append(f"| {key} | {v:.4f} |")
    lines.append("")

    # Per-cell heatmap summary (5x5)
    lines.append("## Per-cell metric_local time-averaged 5x5 maps\n")
    def _render_5x5(title: str, m: list[list[float]] | None) -> None:
        lines.append(f"### {title}")
        if m is None:
            lines.append("_(insufficient data)_\n")
            return
        lines.append("```")
        for row in m:
            lines.append("  ".join(f"{v:+.4f}" for v in row))
        lines.append("```\n")

    _render_5x5("241a turb map (time-avg metric_local)", main["per_cell_heatmap"]["turb_map_5x5"])
    _render_5x5("241a calm map (time-avg metric_local)", main["per_cell_heatmap"]["calm_map_5x5"])
    _render_5x5("241a turb − calm map", main["per_cell_heatmap"]["diff_map_5x5"])
    if base is not None:
        _render_5x5("183c turb map (time-avg metric_local)", base["per_cell_heatmap"]["turb_map_5x5"])
        _render_5x5("183c calm map (time-avg metric_local)", base["per_cell_heatmap"]["calm_map_5x5"])
        _render_5x5("183c turb − calm map", base["per_cell_heatmap"]["diff_map_5x5"])

    # Verdict
    lines.append("## Verdict\n")
    if base is not None:
        rl_m = main["summaries"]["metric_local"]["rms_gap_turb_calm"]
        rl_b = base["summaries"]["metric_local"]["rms_gap_turb_calm"]
        rb_m = main["summaries"]["metric_band"]["rms_gap_turb_calm"]
        rb_b = base["summaries"]["metric_band"]["rms_gap_turb_calm"]

        def _ratio(num: float, den: float) -> float:
            return float("nan") if den <= 1e-10 else num / den

        rl_gain = _ratio(rl_m, rl_b)
        rb_gain = _ratio(rb_m, rb_b)
        verdict = "AMBIGUOUS"
        # Heuristic: require metric_local OR metric_band gap ≥ 1.5× 183c, with both positive
        max_gain = max([x for x in (rl_gain, rb_gain) if np.isfinite(x)], default=float("nan"))
        if np.isfinite(max_gain):
            if max_gain >= 1.5 and rl_m > rl_b and rb_m > rb_b:
                verdict = "CONTROL_PATH_POSITIVE"
            elif max_gain <= 1.10:
                verdict = "CONTROL_PATH_FLAT"
            else:
                verdict = "CONTROL_PATH_WEAK"
        lines.append(f"- metric_local gap gain: **×{rl_gain:.2f}** (183c={rl_b:.4f} → 241a={rl_m:.4f})")
        lines.append(f"- metric_band  gap gain: **×{rb_gain:.2f}** (183c={rb_b:.4f} → 241a={rb_m:.4f})")
        lines.append(f"- **Control-path verdict: {verdict}**")
        results["verdict"] = {
            "metric_local_gap_gain_x": rl_gain,
            "metric_band_gap_gain_x": rb_gain,
            "label": verdict,
        }
    else:
        lines.append("_(no baseline supplied — delta comparison unavailable)_")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Main checkpoint (e.g. 241a best_model.pt)")
    ap.add_argument("--baseline", type=str, default=None,
                    help="Optional baseline checkpoint (e.g. 183c best_model.pt)")
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--max_val_windows", type=int, default=441)
    ap.add_argument("--n_trials", type=int, default=4)
    ap.add_argument("--t_value", type=float, default=0.5)
    ap.add_argument("--q_lo", type=float, default=0.20)
    ap.add_argument("--q_hi", type=float, default=0.80)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Val split (same convention as 241_regime_separation): train[-441:test_start]
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    surf = torch.from_numpy(surfaces).to(args.device)
    val_hist, _val_future = build_multistep_windows(
        val_indices, surf, args.history_len, args.future_len
    )
    vov = vov_proxy(val_hist)
    labels = assign_regimes(vov, q_lo=args.q_lo, q_hi=args.q_hi)

    print(f"Val windows: {len(val_hist)}  |  "
          f"turb={int((labels==2).sum())}  "
          f"calm={int((labels==1).sum())}  "
          f"mid={int((labels==0).sum())}")

    results = {"args": vars(args), "checkpoints": {}}
    results["checkpoints"]["main"] = run_for_checkpoint(
        args.checkpoint, "main", val_hist, labels,
        args.device, args.n_trials, args.t_value,
    )
    if args.baseline:
        results["checkpoints"]["baseline"] = run_for_checkpoint(
            args.baseline, "baseline_183c", val_hist, labels,
            args.device, args.n_trials, args.t_value,
        )

    # Write markdown (mutates `results` to add verdict) then JSON
    md = render_markdown(results)
    md_path = Path(args.output_dir) / "summary.md"
    md_path.write_text(md)
    print(f"\nWrote markdown → {md_path}")

    json_path = Path(args.output_dir) / "summary.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote JSON     → {json_path}")

    if "verdict" in results:
        v = results["verdict"]
        print(f"\n=== CONTROL-PATH VERDICT: {v['label']} ===")
        print(f"   metric_local gap gain: ×{v['metric_local_gap_gain_x']:.2f}")
        print(f"   metric_band  gap gain: ×{v['metric_band_gap_gain_x']:.2f}")


if __name__ == "__main__":
    main()
