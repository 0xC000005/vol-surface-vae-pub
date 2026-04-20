#!/usr/bin/env python
"""
Saturation audit for the 183c state-metric transport architecture.

Question
--------
Are the control-path outputs (raw_local, raw_band, metric_local, metric_band,
local_gate, band_gate) near their clip / sigmoid saturation boundaries, or
does the architecture have headroom?

Measurements (per checkpoint, per t)
------------------------------------
1. raw_local         — distribution + clip-hit rate vs width_clip (0.80); also
                       compared to local_metric_clip (0.60) for reference.
2. raw_band          — per-band (low/mid/high); vs band_clip (0.45); compared
                       to band_metric_clip (0.35) for reference.
3. metric_local      — vs local_metric_clip (from metric_config).
4. metric_band       — per-band; vs band_metric_clip.
5. local_gate        — sigmoid; fraction in bottom-5% (<0.05) and top-5% (>0.95).
6. band_gate         — per-band sigmoid; same.
7. local_metric_budget / band_metric_budget — scalar; reported vs midpoint of
   [min, max] range and distance to each edge.

Verdict per channel
-------------------
SATURATED         : clip-hit ≥ 20%   OR sigmoid tail total ≥ 30%
NEAR_SATURATION   : clip-hit 10-20%  OR sigmoid tail 15-30%
HEADROOM          : clip-hit < 10%   AND sigmoid tail < 15%

The verdict to test is "architecture is output-range saturated." If most
channels report HEADROOM, that null is the informative finding.

Usage
-----
PYTHONPATH=. python experiments/backfill/block_ar/audit_183c_saturation.py \
  --max_val_windows 200 --device cuda
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
from experiments.backfill.block_ar.train_169a_transformed_student_t import iv_to_unconstrained
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


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
# Teacher-forced z_t construction — mirrors training forward pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_teacher_forced_z_t(
    model: StateMetricTransportModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    t_value: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (z_t, t, path_context) where z_t is the same interpolation between
    prior sample z0 and teacher_basis_flat that training uses. This gives a
    faithful snapshot of what the architecture operates on during training.
    """
    device = next(model.parameters()).device
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        flow_context,
        base_local_delta,
        block_logits,
    ) = model.forward_from_history(history_01)
    target_basis = model.teacher_basis_flat_from_outputs(
        target_u=target_u,
        mu=mu,
        time_factor=time_factor,
        time_diag=time_diag,
        cell_factor=cell_factor,
        cell_diag=cell_diag,
        scale=scale,
        base_local_delta=base_local_delta,
        block_logits=block_logits,
    ).view(history_01.shape[0], model.decoder.n_frames, model.decoder.n_cells)
    target_basis_flat = target_basis.reshape(target_basis.shape[0], -1)
    path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)

    torch.manual_seed(seed)
    z0, _ = model.prior.sample(
        target_basis.shape[0], device=device, dtype=target_basis.dtype
    )
    t = torch.full(
        (target_basis.shape[0],),
        t_value,
        device=device,
        dtype=target_basis.dtype,
    )
    z_t = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * target_basis_flat
    return z_t, t, path_context


# ---------------------------------------------------------------------------
# Collection: run build_state_metric_controls on the val split for one ckpt/t
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_controls(
    model: StateMetricTransportModel,
    val_hist: torch.Tensor,
    val_future: torch.Tensor,
    t_value: float,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device

    raw_local_chunks = []
    raw_band_chunks = []
    metric_local_chunks = []
    metric_band_chunks = []
    local_gate_chunks = []
    band_gate_chunks = []

    for i in range(0, val_hist.shape[0], batch_size):
        hist_batch = val_hist[i : i + batch_size].to(device)
        fut_batch = val_future[i : i + batch_size].to(device)
        z_t, t, path_context = build_teacher_forced_z_t(
            model, hist_batch, fut_batch, t_value=t_value, seed=seed + i
        )
        (raw_local, raw_band, metric_local, metric_band,
         local_gate, band_gate) = model.build_state_metric_controls(z_t, t, path_context)

        raw_local_chunks.append(raw_local.detach().cpu())
        raw_band_chunks.append(raw_band.detach().cpu())
        metric_local_chunks.append(metric_local.detach().cpu())
        metric_band_chunks.append(metric_band.detach().cpu())
        local_gate_chunks.append(local_gate.detach().cpu())
        band_gate_chunks.append(band_gate.detach().cpu())

    return {
        "raw_local": torch.cat(raw_local_chunks, dim=0),
        "raw_band": torch.cat(raw_band_chunks, dim=0),
        "metric_local": torch.cat(metric_local_chunks, dim=0),
        "metric_band": torch.cat(metric_band_chunks, dim=0),
        "local_gate": torch.cat(local_gate_chunks, dim=0),
        "band_gate": torch.cat(band_gate_chunks, dim=0),
    }


# ---------------------------------------------------------------------------
# Percentile + clip-hit stats
# ---------------------------------------------------------------------------
def abs_clip_stats(
    tensor: torch.Tensor,
    clamp_value: float,
) -> dict:
    """For a tensor that is symmetrically clamped at ±clamp_value, report
    distribution stats of |x| and clip-hit rates."""
    flat = tensor.reshape(-1).double().numpy()
    abs_flat = np.abs(flat)
    pct = np.percentile(abs_flat, [50, 90, 95, 99])
    # hard hit: |x| >= clamp (numerically reached the boundary)
    hard_hit = float((abs_flat >= clamp_value - 1e-6).mean())
    # near hit: |x| >= 0.95 * clamp
    near_hit = float((abs_flat >= 0.95 * clamp_value).mean())
    # warn hit: |x| >= 0.80 * clamp
    warn_hit = float((abs_flat >= 0.80 * clamp_value).mean())
    return {
        "mean_abs": float(abs_flat.mean()),
        "std_abs": float(abs_flat.std()),
        "std_signed": float(flat.std()),
        "p50_abs": float(pct[0]),
        "p90_abs": float(pct[1]),
        "p95_abs": float(pct[2]),
        "p99_abs": float(pct[3]),
        "max_abs": float(abs_flat.max()),
        "clamp_value": clamp_value,
        "hard_hit_frac": hard_hit,
        "near_hit_frac_95pct": near_hit,
        "warn_hit_frac_80pct": warn_hit,
    }


def per_band_abs_clip_stats(
    tensor: torch.Tensor,
    clamp_value: float,
    band_names: tuple[str, str, str] = ("low", "mid", "high"),
) -> dict:
    # tensor shape: (N, 3)
    assert tensor.ndim == 2 and tensor.shape[1] == 3, tensor.shape
    per_band = {}
    for k, name in enumerate(band_names):
        per_band[name] = abs_clip_stats(tensor[:, k : k + 1], clamp_value=clamp_value)
    per_band["all"] = abs_clip_stats(tensor, clamp_value=clamp_value)
    return per_band


def sigmoid_saturation_stats(tensor: torch.Tensor) -> dict:
    flat = tensor.reshape(-1).double().numpy()
    pct = np.percentile(flat, [5, 25, 50, 75, 95])
    lo_tail = float((flat < 0.05).mean())
    hi_tail = float((flat > 0.95).mean())
    lo20 = float((flat < 0.20).mean())
    hi80 = float((flat > 0.80).mean())
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p05": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
        "frac_lt_0p05": lo_tail,
        "frac_gt_0p95": hi_tail,
        "frac_lt_0p20": lo20,
        "frac_gt_0p80": hi80,
        "tail_total_5pct": lo_tail + hi_tail,
        "tail_total_20pct": lo20 + hi80,
    }


def per_band_sigmoid_stats(tensor: torch.Tensor) -> dict:
    assert tensor.ndim == 2 and tensor.shape[1] == 3, tensor.shape
    out = {}
    for k, name in enumerate(("low", "mid", "high")):
        out[name] = sigmoid_saturation_stats(tensor[:, k : k + 1])
    out["all"] = sigmoid_saturation_stats(tensor)
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def clip_verdict(near_hit_frac: float) -> str:
    # near_hit_frac = fraction with |x| >= 0.95 * clamp
    if near_hit_frac >= 0.20:
        return "SATURATED"
    if near_hit_frac >= 0.10:
        return "NEAR_SATURATION"
    return "HEADROOM"


def sigmoid_verdict(tail_5pct_total: float) -> str:
    # tail_5pct_total = frac<0.05 + frac>0.95
    if tail_5pct_total >= 0.30:
        return "SATURATED"
    if tail_5pct_total >= 0.15:
        return "NEAR_SATURATION"
    return "HEADROOM"


def budget_report(
    value: float,
    low: float,
    high: float,
) -> dict:
    mid = 0.5 * (low + high)
    return {
        "value": value,
        "min": low,
        "max": high,
        "midpoint": mid,
        "distance_to_min": value - low,
        "distance_to_max": high - value,
        "pinned_at_max": (high - value) < 0.05 * (high - low),
        "pinned_at_min": (value - low) < 0.05 * (high - low),
        "above_mid": value > mid,
    }


# ---------------------------------------------------------------------------
# Main per-checkpoint routine
# ---------------------------------------------------------------------------
def audit_checkpoint(
    ckpt_path: str,
    label: str,
    val_hist: torch.Tensor,
    val_future: torch.Tensor,
    t_values: list[float],
    batch_size: int,
    device: str,
    base_seed: int,
) -> dict:
    print(f"\n{'=' * 72}")
    print(f"[{label}] {ckpt_path}")
    print(f"{'=' * 72}")
    model = build_model_from_ckpt(ckpt_path, device)

    integrated = model.integrated_config
    metric = model.metric_config

    width_clip = float(integrated["width_clip"])
    band_clip = float(integrated["band_clip"])
    local_metric_clip = float(metric["local_metric_clip"])
    band_metric_clip = float(metric["band_metric_clip"])
    local_min = float(metric["local_metric_min"])
    local_max = float(metric["local_metric_max"])
    band_min = float(metric["band_metric_min"])
    band_max = float(metric["band_metric_max"])

    local_budget_val = float(model.local_metric_budget().detach().cpu())
    band_budget_val = float(model.band_metric_budget().detach().cpu())

    print(
        f"  clips: width={width_clip}  band={band_clip}  "
        f"metric_local={local_metric_clip}  metric_band={band_metric_clip}"
    )
    print(
        f"  budgets: local={local_budget_val:.4f} in [{local_min},{local_max}]  "
        f"band={band_budget_val:.4f} in [{band_min},{band_max}]"
    )

    per_t = {}
    for t_val in t_values:
        print(f"  Collecting controls at t={t_val} ...")
        out = collect_controls(
            model,
            val_hist,
            val_future,
            t_value=t_val,
            batch_size=batch_size,
            seed=base_seed + int(round(t_val * 1000)),
        )

        stats = {}

        # raw_local — actual clamp is width_clip (0.80); report vs local_metric_clip too
        stats["raw_local"] = abs_clip_stats(out["raw_local"], clamp_value=width_clip)
        stats["raw_local"]["reference_local_metric_clip"] = {
            "value": local_metric_clip,
            "frac_abs_ge_95pct_of_ref": float(
                (out["raw_local"].abs().reshape(-1).numpy() >= 0.95 * local_metric_clip).mean()
            ),
        }
        stats["raw_local"]["verdict"] = clip_verdict(stats["raw_local"]["near_hit_frac_95pct"])

        # raw_band — actual clamp is band_clip (0.45); report per band + vs band_metric_clip
        stats["raw_band"] = per_band_abs_clip_stats(out["raw_band"], clamp_value=band_clip)
        ref_frac_all = float(
            (out["raw_band"].abs().reshape(-1).numpy() >= 0.95 * band_metric_clip).mean()
        )
        stats["raw_band"]["reference_band_metric_clip"] = {
            "value": band_metric_clip,
            "frac_abs_ge_95pct_of_ref_all": ref_frac_all,
        }
        stats["raw_band"]["verdict"] = clip_verdict(stats["raw_band"]["all"]["near_hit_frac_95pct"])

        # metric_local — clamp is local_metric_clip
        stats["metric_local"] = abs_clip_stats(
            out["metric_local"], clamp_value=local_metric_clip
        )
        stats["metric_local"]["verdict"] = clip_verdict(
            stats["metric_local"]["near_hit_frac_95pct"]
        )

        # metric_band — per-band; clamp is band_metric_clip
        stats["metric_band"] = per_band_abs_clip_stats(
            out["metric_band"], clamp_value=band_metric_clip
        )
        stats["metric_band"]["verdict"] = clip_verdict(
            stats["metric_band"]["all"]["near_hit_frac_95pct"]
        )

        # local_gate — sigmoid
        stats["local_gate"] = sigmoid_saturation_stats(out["local_gate"])
        stats["local_gate"]["verdict"] = sigmoid_verdict(stats["local_gate"]["tail_total_5pct"])

        # band_gate — per-band sigmoid
        stats["band_gate"] = per_band_sigmoid_stats(out["band_gate"])
        stats["band_gate"]["verdict"] = sigmoid_verdict(
            stats["band_gate"]["all"]["tail_total_5pct"]
        )

        per_t[f"t={t_val}"] = stats

    # Budget channels (scalar, constant across t)
    budgets = {
        "local_metric_budget": budget_report(local_budget_val, local_min, local_max),
        "band_metric_budget": budget_report(band_budget_val, band_min, band_max),
    }
    # Budget verdict: pinned at max ⇒ SATURATED (headroom exhausted upward),
    # pinned at min ⇒ SATURATED (headroom exhausted downward, dormant).
    for name, rep in budgets.items():
        if rep["pinned_at_max"] or rep["pinned_at_min"]:
            rep["verdict"] = "SATURATED"
        elif (
            rep["distance_to_max"] < 0.15 * (rep["max"] - rep["min"])
            or rep["distance_to_min"] < 0.15 * (rep["max"] - rep["min"])
        ):
            rep["verdict"] = "NEAR_SATURATION"
        else:
            rep["verdict"] = "HEADROOM"

    return {
        "label": label,
        "ckpt_path": ckpt_path,
        "clips": {
            "width_clip": width_clip,
            "band_clip": band_clip,
            "local_metric_clip": local_metric_clip,
            "band_metric_clip": band_metric_clip,
        },
        "budgets": budgets,
        "per_t": per_t,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _fmt_frac(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def render_markdown(results: dict) -> str:
    args = results["args"]
    lines = []
    lines.append("# 183c / 241a Saturation Audit\n")
    lines.append(f"- Val windows: {args['max_val_windows']}  |  t values: {args['t_values']}")
    lines.append(f"- Verdict thresholds: clip-hit (fraction with |x| >= 95% of clamp) —")
    lines.append(f"  SATURATED ≥ 20%, NEAR_SATURATION 10-20%, HEADROOM < 10%")
    lines.append(f"- Sigmoid tails (frac<0.05 + frac>0.95):")
    lines.append(f"  SATURATED ≥ 30%, NEAR_SATURATION 15-30%, HEADROOM < 15%\n")

    for label, ckpt_audit in results["audits"].items():
        lines.append(f"## {label}\n")
        lines.append(f"- checkpoint: `{ckpt_audit['ckpt_path']}`")
        clips = ckpt_audit["clips"]
        lines.append(
            f"- clips: width={clips['width_clip']}, band={clips['band_clip']}, "
            f"metric_local={clips['local_metric_clip']}, "
            f"metric_band={clips['band_metric_clip']}"
        )

        # Budget scalars
        lb = ckpt_audit["budgets"]["local_metric_budget"]
        bb = ckpt_audit["budgets"]["band_metric_budget"]
        lines.append(
            f"- local_metric_budget: {lb['value']:.4f} in "
            f"[{lb['min']:.3f}, {lb['max']:.3f}] (mid={lb['midpoint']:.3f}), "
            f"verdict={lb['verdict']}"
        )
        lines.append(
            f"- band_metric_budget: {bb['value']:.4f} in "
            f"[{bb['min']:.3f}, {bb['max']:.3f}] (mid={bb['midpoint']:.3f}), "
            f"verdict={bb['verdict']}\n"
        )

        # Per-t per-channel summary table
        for t_key, stats in ckpt_audit["per_t"].items():
            lines.append(f"### {t_key}\n")
            lines.append(
                "| channel | clamp | p50|.| | p90|.| | p95|.| | p99|.| | hit≥95% | hit≥80% | verdict |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            rl = stats["raw_local"]
            lines.append(
                f"| raw_local | {rl['clamp_value']} | "
                f"{rl['p50_abs']:.3f} | {rl['p90_abs']:.3f} | {rl['p95_abs']:.3f} | "
                f"{rl['p99_abs']:.3f} | "
                f"{_fmt_frac(rl['near_hit_frac_95pct'])} | "
                f"{_fmt_frac(rl['warn_hit_frac_80pct'])} | {rl['verdict']} |"
            )
            rb_all = stats["raw_band"]["all"]
            lines.append(
                f"| raw_band (all) | {rb_all['clamp_value']} | "
                f"{rb_all['p50_abs']:.3f} | {rb_all['p90_abs']:.3f} | "
                f"{rb_all['p95_abs']:.3f} | {rb_all['p99_abs']:.3f} | "
                f"{_fmt_frac(rb_all['near_hit_frac_95pct'])} | "
                f"{_fmt_frac(rb_all['warn_hit_frac_80pct'])} | "
                f"{stats['raw_band']['verdict']} |"
            )
            for bname in ("low", "mid", "high"):
                rb = stats["raw_band"][bname]
                lines.append(
                    f"| raw_band ({bname}) | {rb['clamp_value']} | "
                    f"{rb['p50_abs']:.3f} | {rb['p90_abs']:.3f} | "
                    f"{rb['p95_abs']:.3f} | {rb['p99_abs']:.3f} | "
                    f"{_fmt_frac(rb['near_hit_frac_95pct'])} | "
                    f"{_fmt_frac(rb['warn_hit_frac_80pct'])} | — |"
                )
            ml = stats["metric_local"]
            lines.append(
                f"| metric_local | {ml['clamp_value']} | "
                f"{ml['p50_abs']:.3f} | {ml['p90_abs']:.3f} | {ml['p95_abs']:.3f} | "
                f"{ml['p99_abs']:.3f} | "
                f"{_fmt_frac(ml['near_hit_frac_95pct'])} | "
                f"{_fmt_frac(ml['warn_hit_frac_80pct'])} | {ml['verdict']} |"
            )
            mb_all = stats["metric_band"]["all"]
            lines.append(
                f"| metric_band (all) | {mb_all['clamp_value']} | "
                f"{mb_all['p50_abs']:.3f} | {mb_all['p90_abs']:.3f} | "
                f"{mb_all['p95_abs']:.3f} | {mb_all['p99_abs']:.3f} | "
                f"{_fmt_frac(mb_all['near_hit_frac_95pct'])} | "
                f"{_fmt_frac(mb_all['warn_hit_frac_80pct'])} | "
                f"{stats['metric_band']['verdict']} |"
            )
            for bname in ("low", "mid", "high"):
                mb = stats["metric_band"][bname]
                lines.append(
                    f"| metric_band ({bname}) | {mb['clamp_value']} | "
                    f"{mb['p50_abs']:.3f} | {mb['p90_abs']:.3f} | "
                    f"{mb['p95_abs']:.3f} | {mb['p99_abs']:.3f} | "
                    f"{_fmt_frac(mb['near_hit_frac_95pct'])} | "
                    f"{_fmt_frac(mb['warn_hit_frac_80pct'])} | — |"
                )
            lines.append("")

            # Sigmoid gate table
            lines.append("| gate | mean | std | p05 | p50 | p95 | frac<0.05 | frac>0.95 | tail_total | verdict |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            lg = stats["local_gate"]
            lines.append(
                f"| local_gate | {lg['mean']:.3f} | {lg['std']:.3f} | "
                f"{lg['p05']:.3f} | {lg['p50']:.3f} | {lg['p95']:.3f} | "
                f"{_fmt_frac(lg['frac_lt_0p05'])} | {_fmt_frac(lg['frac_gt_0p95'])} | "
                f"{_fmt_frac(lg['tail_total_5pct'])} | {lg['verdict']} |"
            )
            bg_all = stats["band_gate"]["all"]
            lines.append(
                f"| band_gate (all) | {bg_all['mean']:.3f} | {bg_all['std']:.3f} | "
                f"{bg_all['p05']:.3f} | {bg_all['p50']:.3f} | {bg_all['p95']:.3f} | "
                f"{_fmt_frac(bg_all['frac_lt_0p05'])} | "
                f"{_fmt_frac(bg_all['frac_gt_0p95'])} | "
                f"{_fmt_frac(bg_all['tail_total_5pct'])} | "
                f"{stats['band_gate']['verdict']} |"
            )
            for bname in ("low", "mid", "high"):
                bg = stats["band_gate"][bname]
                lines.append(
                    f"| band_gate ({bname}) | {bg['mean']:.3f} | {bg['std']:.3f} | "
                    f"{bg['p05']:.3f} | {bg['p50']:.3f} | {bg['p95']:.3f} | "
                    f"{_fmt_frac(bg['frac_lt_0p05'])} | "
                    f"{_fmt_frac(bg['frac_gt_0p95'])} | "
                    f"{_fmt_frac(bg['tail_total_5pct'])} | — |"
                )
            lines.append("")

    # Verdict rollup: per-channel majority across t values
    lines.append("## Channel verdict rollup (worst across t values)\n")
    worst_rank = {"HEADROOM": 0, "NEAR_SATURATION": 1, "SATURATED": 2}
    rank_to_label = {v: k for k, v in worst_rank.items()}
    rollup: dict[str, dict[str, str]] = {}
    for label, ckpt_audit in results["audits"].items():
        row: dict[str, str] = {}
        for ch in ("raw_local", "raw_band", "metric_local", "metric_band",
                   "local_gate", "band_gate"):
            worst = 0
            for _t_key, stats in ckpt_audit["per_t"].items():
                worst = max(worst, worst_rank[stats[ch]["verdict"]])
            row[ch] = rank_to_label[worst]
        row["local_metric_budget"] = ckpt_audit["budgets"]["local_metric_budget"]["verdict"]
        row["band_metric_budget"] = ckpt_audit["budgets"]["band_metric_budget"]["verdict"]
        rollup[label] = row

    channels = ["raw_local", "raw_band", "metric_local", "metric_band",
                "local_gate", "band_gate",
                "local_metric_budget", "band_metric_budget"]
    header = "| channel | " + " | ".join(rollup.keys()) + " |"
    sep = "|---|" + "|".join(["---"] * len(rollup)) + "|"
    lines.append(header)
    lines.append(sep)
    for ch in channels:
        row_vals = [rollup[lbl][ch] for lbl in rollup.keys()]
        lines.append(f"| {ch} | " + " | ".join(row_vals) + " |")
    lines.append("")

    # Overall architecture verdict
    def _arch_verdict(row: dict[str, str]) -> str:
        # if ≥3 channels SATURATED or any core channel (raw_*, gate) SATURATED ⇒ SATURATED
        n_sat = sum(1 for v in row.values() if v == "SATURATED")
        n_near = sum(1 for v in row.values() if v == "NEAR_SATURATION")
        if n_sat >= 3:
            return "OUTPUT_RANGE_SATURATED"
        if n_sat >= 1 or n_near >= 4:
            return "PARTIAL_SATURATION"
        return "HEADROOM_DOMINANT"

    lines.append("## Architecture-level verdict\n")
    for lbl, row in rollup.items():
        lines.append(f"- **{lbl}**: {_arch_verdict(row)} "
                     f"({sum(1 for v in row.values() if v == 'SATURATED')} SATURATED / "
                     f"{sum(1 for v in row.values() if v == 'NEAR_SATURATION')} NEAR / "
                     f"{sum(1 for v in row.values() if v == 'HEADROOM')} HEADROOM "
                     f"of {len(row)} channels)")

    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- The hypothesis under test: **\"the architecture's output range is"
                 " already exhausted (saturated).\"**")
    lines.append("- If most channels show HEADROOM, the architecture is NOT the"
                 " bottleneck — the hypothesis is not supported and the ceiling must"
                 " lie in loss / signal / training, not in the clip/sigmoid geometry.")
    lines.append("- If most channels show SATURATED, clip widening / budget expansion"
                 " is a principled lever.")
    lines.append("")
    results["rollup"] = rollup
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_183c", type=str,
                    default=(
                        "models/backfill/"
                        "state_metric_transport_pathwise_residual_law_mean_reverting"
                        "_covariance_mixture_structured_joint_student_t_183c/best_model.pt"
                    ))
    ap.add_argument("--ckpt_241a", type=str,
                    default="models/backfill/241a_contrastive_fm_s42/best_model.pt")
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--max_val_windows", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--t_values", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_dir", type=str,
                    default="results/block_ar/241a/saturation_audit")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Val split (same convention as 241 diagnostics): train[-441:test_start]
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    surf = torch.from_numpy(surfaces).to(args.device)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf, args.history_len, args.future_len
    )
    print(f"Val windows: {len(val_hist)}  batch_size={args.batch_size}  "
          f"t_values={args.t_values}")

    audits = {}
    audits["183c_baseline"] = audit_checkpoint(
        args.ckpt_183c, "183c_baseline", val_hist, val_future,
        t_values=args.t_values, batch_size=args.batch_size,
        device=args.device, base_seed=args.seed,
    )
    audits["241a_best"] = audit_checkpoint(
        args.ckpt_241a, "241a_best", val_hist, val_future,
        t_values=args.t_values, batch_size=args.batch_size,
        device=args.device, base_seed=args.seed,
    )

    results = {
        "args": vars(args),
        "audits": audits,
    }

    md = render_markdown(results)
    md_path = Path(args.output_dir) / "summary.md"
    md_path.write_text(md)
    print(f"\nWrote markdown → {md_path}")

    json_path = Path(args.output_dir) / "summary.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote JSON     → {json_path}")

    print("\n=== CHANNEL ROLLUP (worst across t) ===")
    for lbl, row in results["rollup"].items():
        print(f"[{lbl}]")
        for ch, verdict in row.items():
            print(f"  {ch:<25}  {verdict}")


if __name__ == "__main__":
    main()
