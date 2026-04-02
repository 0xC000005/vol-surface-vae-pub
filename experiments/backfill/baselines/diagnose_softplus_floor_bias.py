#!/usr/bin/env python
"""Independent diagnostic for softplus-floor bias in Block-AR surface models.

Compares:
  - models/backfill/afcrps_164a_v3_percell_bptt/final_model.pt
  - models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt

Outputs:
  - clear stdout summary with horizon-level numbers
  - full JSON results with per-cell metrics and parameter norms
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_164a_v3_percell_bptt import (
    ARSpatialTransformerModel,
)


HISTORY_LEN = 30
FUTURE_LEN = 30
N_CELLS = 25
GRID_H = 5
GRID_W = 5
HORIZONS = [1, 5, 10, 20, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-ckpt",
        type=Path,
        default=Path("models/backfill/afcrps_164a_v3_percell_bptt/final_model.pt"),
    )
    parser.add_argument(
        "--softplus-ckpt",
        type=Path,
        default=Path("models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/vol_surface_with_ret.npz"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--test-start",
        type=int,
        default=4540,
        help="Requested evaluation start. Training scripts use 4511; this run honors 4540.",
    )
    parser.add_argument("--n-windows", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-probe-windows", type=int, default=64)
    parser.add_argument("--noise-probe-samples", type=int, default=128)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/softplus_floor_bias_diagnostic.json"),
    )
    parser.add_argument("--skip-grids", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[ARSpatialTransformerModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    encoder_cfg = config["encoder"]
    decoder_cfg = config["decoder"]
    encoder = EncoderConfig(
        input_dim=encoder_cfg["input_dim"],
        gru_hidden_dim=encoder_cfg["gru_hidden_dim"],
        bottleneck_dim=encoder_cfg["bottleneck_dim"],
        dropout=encoder_cfg.get("dropout", 0.1),
        cond_aug_sigma=encoder_cfg.get("cond_aug_sigma", 0.0),
    )
    model = ARSpatialTransformerModel(encoder, decoder_cfg)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model, checkpoint


def load_windows(data_path: Path, test_start: int, n_windows: int) -> dict[str, np.ndarray]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    max_start = surfaces.shape[0] - HISTORY_LEN - FUTURE_LEN
    if test_start > max_start:
        raise ValueError(f"test_start={test_start} exceeds max valid start {max_start}")
    end_exclusive = min(test_start + n_windows, max_start + 1)
    starts = np.arange(test_start, end_exclusive, dtype=np.int64)
    hist_offsets = np.arange(HISTORY_LEN, dtype=np.int64)
    fut_offsets = np.arange(FUTURE_LEN, dtype=np.int64)
    histories = surfaces[starts[:, None] + hist_offsets[None, :]]
    futures = surfaces[starts[:, None] + HISTORY_LEN + fut_offsets[None, :]]
    last_frames = histories[:, -1].reshape(len(starts), N_CELLS)
    return {
        "starts": starts,
        "histories": histories,
        "futures": futures.reshape(len(starts), FUTURE_LEN, N_CELLS),
        "last_frames": last_frames,
        "n_total_surfaces": surfaces.shape[0],
    }


def sample_model(
    model: ARSpatialTransformerModel,
    histories: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray | float]:
    set_seed(seed)
    pred_means = []
    pred_stds = []
    total_neg = 0
    total_floor = 0
    total_vals = 0
    neg_counts = np.zeros((FUTURE_LEN, N_CELLS), dtype=np.int64)
    floor_counts = np.zeros((FUTURE_LEN, N_CELLS), dtype=np.int64)

    for start in range(0, len(histories), batch_size):
        end = min(start + batch_size, len(histories))
        hist_batch = torch.from_numpy(histories[start:end]).to(device)
        hist_norm = normalize_iv(hist_batch)

        with torch.inference_mode():
            if hasattr(model, "sample_batched"):
                samples = model.sample_batched(hist_norm, n_samples=n_samples)
            elif hasattr(model, "sample"):
                samples = model.sample(hist_norm, n_samples=n_samples)
            else:
                raise AttributeError("Model has neither sample_batched() nor sample().")

        samples = samples.reshape(end - start, n_samples, FUTURE_LEN, N_CELLS).detach().cpu()
        pred_means.append(samples.mean(dim=1).numpy())
        pred_stds.append(samples.std(dim=1, unbiased=False).numpy())

        neg_mask = samples < 0.0
        floor_mask = samples < 0.01
        total_neg += int(neg_mask.sum().item())
        total_floor += int(floor_mask.sum().item())
        total_vals += samples.numel()
        neg_counts += neg_mask.sum(dim=(0, 1)).numpy()
        floor_counts += floor_mask.sum(dim=(0, 1)).numpy()

    total_per_horizon_cell = len(histories) * n_samples
    return {
        "pred_mean": np.concatenate(pred_means, axis=0),
        "pred_std": np.concatenate(pred_stds, axis=0),
        "neg_rate_overall": total_neg / total_vals,
        "floor_rate_overall": total_floor / total_vals,
        "neg_rate_per_horizon_cell": neg_counts / total_per_horizon_cell,
        "floor_rate_per_horizon_cell": floor_counts / total_per_horizon_cell,
    }


def compute_behavior_metrics(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    gt: np.ndarray,
    last_frames: np.ndarray,
) -> dict[str, Any]:
    bias = pred_mean - gt
    pred_increment = pred_mean - last_frames[:, None, :]
    gt_increment = gt - last_frames[:, None, :]
    return {
        "pred_mean_by_horizon_cell": pred_mean.mean(axis=0),
        "gt_mean_by_horizon_cell": gt.mean(axis=0),
        "bias_by_horizon_cell": bias.mean(axis=0),
        "abs_bias_by_horizon_cell": np.abs(bias).mean(axis=0),
        "spread_by_horizon_cell": pred_std.mean(axis=0),
        "pred_increment_by_horizon_cell": pred_increment.mean(axis=0),
        "gt_increment_by_horizon_cell": gt_increment.mean(axis=0),
        "bias_global_by_horizon": bias.mean(axis=(0, 2)),
        "abs_bias_global_by_horizon": np.abs(bias).mean(axis=(0, 2)),
        "spread_global_by_horizon": pred_std.mean(axis=(0, 2)),
        "pred_mean_global_by_horizon": pred_mean.mean(axis=(0, 2)),
        "gt_mean_global_by_horizon": gt.mean(axis=(0, 2)),
        "frac_negative_bias_cells_by_horizon": (bias.mean(axis=0) < 0.0).mean(axis=1),
    }


def param_norms(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(t.detach().float().norm().item()) for name, t in state_dict.items()}


def group_norm(state_dict: dict[str, torch.Tensor], match_fn) -> float:
    total_sq = 0.0
    for name, tensor in state_dict.items():
        if match_fn(name):
            total_sq += float(torch.sum(tensor.detach().float() ** 2).item())
    return math.sqrt(total_sq)


def compare_parameter_norms(
    baseline_sd: dict[str, torch.Tensor],
    softplus_sd: dict[str, torch.Tensor],
) -> dict[str, Any]:
    baseline_norms = param_norms(baseline_sd)
    softplus_norms = param_norms(softplus_sd)

    def ratio(a: float, b: float) -> float | None:
        if abs(a) < 1e-12:
            return None
        return b / a

    per_param = {}
    for name in baseline_norms:
        b = baseline_norms[name]
        s = softplus_norms[name]
        per_param[name] = {
            "baseline": b,
            "softplus": s,
            "delta": s - b,
            "ratio": ratio(b, s),
        }

    groups = {
        "encoder_total": lambda n: n.startswith("encoder."),
        "decoder_total": lambda n: n.startswith("decoder."),
        "decoder_non_noise": lambda n: n.startswith("decoder.") and not any(
            key in n for key in ("noise_proj", "scale_proj", "bias_proj")
        ),
        "noise_path_total": lambda n: n.startswith("decoder.") and any(
            key in n for key in ("noise_proj", "scale_proj", "bias_proj")
        ),
        "noise_proj_total": lambda n: "decoder.noise_proj" in n,
        "cln_scale_total": lambda n: "scale_proj" in n,
        "cln_bias_total": lambda n: "bias_proj" in n,
        "decoder_output_proj": lambda n: n.startswith("decoder.output_proj"),
        "decoder_input_proj": lambda n: n.startswith("decoder.input_proj"),
        "decoder_cond_proj": lambda n: n.startswith("decoder.cond_proj"),
        "decoder_attn_total": lambda n: ".attn." in n,
        "decoder_ff_total": lambda n: ".ff." in n and "ff_cln" not in n,
        "decoder_ff_cln_total": lambda n: "ff_cln" in n,
    }
    group_results = {}
    for name, fn in groups.items():
        b = group_norm(baseline_sd, fn)
        s = group_norm(softplus_sd, fn)
        group_results[name] = {
            "baseline": b,
            "softplus": s,
            "delta": s - b,
            "ratio": ratio(b, s),
        }

    cln_param_names = [
        name
        for name in baseline_sd
        if ("scale_proj" in name or "bias_proj" in name) and name.startswith("decoder.layers.")
    ]
    cln_details = {
        name: per_param[name]
        for name in sorted(cln_param_names)
    }

    top_changed = sorted(
        (
            {
                "name": name,
                "baseline": vals["baseline"],
                "softplus": vals["softplus"],
                "delta_abs": abs(vals["delta"]),
                "ratio": vals["ratio"],
            }
            for name, vals in per_param.items()
        ),
        key=lambda item: item["delta_abs"],
        reverse=True,
    )[:20]

    return {
        "group_norms": group_results,
        "cln_param_norms": cln_details,
        "all_param_norms": per_param,
        "top_changed_params_by_abs_norm_delta": top_changed,
    }


def probe_noise_pathway(
    model: ARSpatialTransformerModel,
    histories: np.ndarray,
    futures: np.ndarray,
    n_probe_windows: int,
    n_probe_samples: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    n_probe = min(n_probe_windows, len(histories))
    hist = torch.from_numpy(histories[:n_probe]).to(device)
    gt1 = torch.from_numpy(futures[:n_probe, 0]).to(device)
    hist_norm = normalize_iv(hist)

    with torch.inference_mode():
        last_frame = hist[:, -1].reshape(n_probe, N_CELLS)
        hist_flat = hist_norm.reshape(n_probe, HISTORY_LEN, N_CELLS)
        gru_outputs, _ = model.encoder.gru(hist_flat)
        attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
        cond = model.encoder.bottleneck(pooled)

        cond_k = cond.unsqueeze(1).expand(n_probe, n_probe_samples, -1).reshape(
            n_probe * n_probe_samples, -1
        )
        prev_k = last_frame.unsqueeze(1).expand(n_probe, n_probe_samples, N_CELLS).reshape(
            n_probe * n_probe_samples, N_CELLS
        )
        z = torch.randn(n_probe * n_probe_samples, model.decoder.noise_dim, device=device)
        delta = model.decoder(cond_k, prev_k, z)
        frame1 = prev_k + torch.tanh(delta)
        frame1 = frame1.reshape(n_probe, n_probe_samples, N_CELLS)

        z0 = torch.zeros(n_probe, model.decoder.noise_dim, device=device)
        delta0 = model.decoder(cond, last_frame, z0)
        frame1_z0 = last_frame + torch.tanh(delta0)

    sample_mean = frame1.mean(dim=1).cpu().numpy()
    sample_std = frame1.std(dim=1, unbiased=False).cpu().numpy()
    zero_noise = frame1_z0.cpu().numpy()
    gt1_np = gt1.cpu().numpy()

    return {
        "step1_sample_mean_bias_per_cell": (sample_mean - gt1_np).mean(axis=0),
        "step1_sample_std_per_cell": sample_std.mean(axis=0),
        "step1_sample_std_global": float(sample_std.mean()),
        "step1_zero_noise_pred_per_cell": zero_noise.mean(axis=0),
        "step1_zero_noise_bias_per_cell": (zero_noise - gt1_np).mean(axis=0),
    }


def grid_str(values: np.ndarray, fmt: str = "{:8.5f}") -> str:
    arr = np.asarray(values).reshape(GRID_H, GRID_W)
    return "\n".join(" ".join(fmt.format(float(x)) for x in row) for row in arr)


def print_horizon_summary(
    baseline_behavior: dict[str, Any],
    softplus_behavior: dict[str, Any],
    baseline_raw: dict[str, Any],
    softplus_raw: dict[str, Any],
) -> None:
    print("\n=== Horizon Summary (global means across 25 cells) ===")
    print(
        "h  base_bias  soft_bias  delta_bias  base_spread  soft_spread  "
        "spread_ratio  soft_neg_cells  base_neg_rate  soft_neg_rate  base_<0.01  soft_<0.01"
    )
    for h in HORIZONS:
        i = h - 1
        base_bias = baseline_behavior["bias_global_by_horizon"][i]
        soft_bias = softplus_behavior["bias_global_by_horizon"][i]
        base_spread = baseline_behavior["spread_global_by_horizon"][i]
        soft_spread = softplus_behavior["spread_global_by_horizon"][i]
        ratio = soft_spread / base_spread if abs(base_spread) > 1e-12 else float("nan")
        soft_neg_cells = softplus_behavior["frac_negative_bias_cells_by_horizon"][i]
        base_neg_rate = baseline_raw["neg_rate_per_horizon_cell"][i].mean()
        soft_neg_rate = softplus_raw["neg_rate_per_horizon_cell"][i].mean()
        base_floor = baseline_raw["floor_rate_per_horizon_cell"][i].mean()
        soft_floor = softplus_raw["floor_rate_per_horizon_cell"][i].mean()
        print(
            f"d{h:02d} "
            f"{base_bias:10.5f} {soft_bias:10.5f} {soft_bias - base_bias:10.5f} "
            f"{base_spread:11.5f} {soft_spread:11.5f} {ratio:12.3f} "
            f"{soft_neg_cells:14.2%} {base_neg_rate:13.4%} {soft_neg_rate:13.4%} "
            f"{base_floor:11.4%} {soft_floor:11.4%}"
        )


def print_weight_summary(norm_results: dict[str, Any]) -> None:
    print("\n=== Parameter Group Norms ===")
    print("group  baseline  softplus  delta  ratio")
    for name, vals in norm_results["group_norms"].items():
        ratio = vals["ratio"]
        ratio_str = "nan" if ratio is None else f"{ratio:.3f}"
        print(
            f"{name:22s} {vals['baseline']:9.4f} {vals['softplus']:9.4f} "
            f"{vals['delta']:9.4f} {ratio_str:>8s}"
        )

    print("\n=== Top Parameter Norm Changes (abs delta) ===")
    for item in norm_results["top_changed_params_by_abs_norm_delta"]:
        ratio = item["ratio"]
        ratio_str = "nan" if ratio is None else f"{ratio:.3f}"
        print(
            f"{item['name']:55s} "
            f"base={item['baseline']:.4f} soft={item['softplus']:.4f} "
            f"delta={item['delta_abs']:.4f} ratio={ratio_str}"
        )


def print_cln_summary(norm_results: dict[str, Any]) -> None:
    print("\n=== CLN / Noise Pathway Norms ===")
    for name, vals in norm_results["cln_param_norms"].items():
        ratio = vals["ratio"]
        ratio_str = "nan" if ratio is None else f"{ratio:.3f}"
        print(
            f"{name:60s} base={vals['baseline']:.4f} "
            f"soft={vals['softplus']:.4f} delta={vals['delta']:.4f} ratio={ratio_str}"
        )


def print_per_horizon_grids(
    gt_mean: np.ndarray,
    baseline_behavior: dict[str, Any],
    softplus_behavior: dict[str, Any],
) -> None:
    for h in HORIZONS:
        i = h - 1
        print(f"\n=== d{h} per-cell GT mean ===")
        print(grid_str(gt_mean[i]))

        print(f"\n=== d{h} baseline pred mean ===")
        print(grid_str(baseline_behavior["pred_mean_by_horizon_cell"][i]))
        print(f"\n=== d{h} baseline bias ===")
        print(grid_str(baseline_behavior["bias_by_horizon_cell"][i]))
        print(f"\n=== d{h} baseline spread ===")
        print(grid_str(baseline_behavior["spread_by_horizon_cell"][i]))

        print(f"\n=== d{h} softplus pred mean ===")
        print(grid_str(softplus_behavior["pred_mean_by_horizon_cell"][i]))
        print(f"\n=== d{h} softplus bias ===")
        print(grid_str(softplus_behavior["bias_by_horizon_cell"][i]))
        print(f"\n=== d{h} softplus spread ===")
        print(grid_str(softplus_behavior["spread_by_horizon_cell"][i]))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    windows = load_windows(args.data_path, args.test_start, args.n_windows)
    histories = windows["histories"]
    futures = windows["futures"]
    last_frames = windows["last_frames"]

    print("=== Evaluation Setup ===")
    print(f"baseline_ckpt={args.baseline_ckpt}")
    print(f"softplus_ckpt={args.softplus_ckpt}")
    print(f"data_path={args.data_path}")
    print(f"device={device}")
    print(f"requested_test_start={args.test_start}")
    print("training_script_test_start=4511")
    print(f"n_windows={len(histories)} n_samples={args.n_samples} batch_size={args.batch_size}")
    print(f"surface_count={windows['n_total_surfaces']}")

    baseline_model, baseline_ckpt = build_model(args.baseline_ckpt, device)
    softplus_model, softplus_ckpt = build_model(args.softplus_ckpt, device)

    print("\n=== Checkpoint Metadata ===")
    print(
        f"baseline epoch={baseline_ckpt.get('epoch')} val_loss={baseline_ckpt.get('val_loss'):.6f} "
        f"type={baseline_ckpt['config'].get('type')}"
    )
    print(
        f"softplus epoch={softplus_ckpt.get('epoch')} val_loss={softplus_ckpt.get('val_loss'):.6f} "
        f"type={softplus_ckpt['config'].get('type')}"
    )

    baseline_raw = sample_model(
        baseline_model, histories, args.n_samples, args.batch_size, device, args.seed
    )
    softplus_raw = sample_model(
        softplus_model, histories, args.n_samples, args.batch_size, device, args.seed
    )

    baseline_behavior = compute_behavior_metrics(
        baseline_raw["pred_mean"], baseline_raw["pred_std"], futures, last_frames
    )
    softplus_behavior = compute_behavior_metrics(
        softplus_raw["pred_mean"], softplus_raw["pred_std"], futures, last_frames
    )

    baseline_sd = baseline_ckpt["model_state_dict"]
    softplus_sd = softplus_ckpt["model_state_dict"]
    norm_results = compare_parameter_norms(baseline_sd, softplus_sd)

    baseline_noise_probe = probe_noise_pathway(
        baseline_model,
        histories,
        futures,
        args.noise_probe_windows,
        args.noise_probe_samples,
        device,
        args.seed + 100,
    )
    softplus_noise_probe = probe_noise_pathway(
        softplus_model,
        histories,
        futures,
        args.noise_probe_windows,
        args.noise_probe_samples,
        device,
        args.seed + 100,
    )

    print_horizon_summary(baseline_behavior, softplus_behavior, baseline_raw, softplus_raw)
    print_weight_summary(norm_results)
    print_cln_summary(norm_results)

    print("\n=== Step-1 Noise Sensitivity ===")
    print(
        f"baseline step1 noise std global={baseline_noise_probe['step1_sample_std_global']:.6f}"
    )
    print(
        f"softplus step1 noise std global={softplus_noise_probe['step1_sample_std_global']:.6f}"
    )
    print("\nBaseline step1 noise std per cell")
    print(grid_str(baseline_noise_probe["step1_sample_std_per_cell"]))
    print("\nSoftplus step1 noise std per cell")
    print(grid_str(softplus_noise_probe["step1_sample_std_per_cell"]))
    print("\nBaseline step1 zero-noise bias per cell")
    print(grid_str(baseline_noise_probe["step1_zero_noise_bias_per_cell"]))
    print("\nSoftplus step1 zero-noise bias per cell")
    print(grid_str(softplus_noise_probe["step1_zero_noise_bias_per_cell"]))

    if not args.skip_grids:
        print_per_horizon_grids(
            baseline_behavior["gt_mean_by_horizon_cell"],
            baseline_behavior,
            softplus_behavior,
        )

    results = {
        "setup": {
            "baseline_ckpt": args.baseline_ckpt,
            "softplus_ckpt": args.softplus_ckpt,
            "data_path": args.data_path,
            "device": args.device,
            "test_start_requested": args.test_start,
            "training_script_test_start": 4511,
            "n_windows": len(histories),
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "checkpoints": {
            "baseline": {
                "epoch": baseline_ckpt.get("epoch"),
                "val_loss": baseline_ckpt.get("val_loss"),
                "type": baseline_ckpt["config"].get("type"),
            },
            "softplus": {
                "epoch": softplus_ckpt.get("epoch"),
                "val_loss": softplus_ckpt.get("val_loss"),
                "type": softplus_ckpt["config"].get("type"),
            },
        },
        "baseline_behavior": baseline_behavior,
        "softplus_behavior": softplus_behavior,
        "baseline_floor_stats": {
            "neg_rate_overall": baseline_raw["neg_rate_overall"],
            "floor_rate_overall": baseline_raw["floor_rate_overall"],
            "neg_rate_per_horizon_cell": baseline_raw["neg_rate_per_horizon_cell"],
            "floor_rate_per_horizon_cell": baseline_raw["floor_rate_per_horizon_cell"],
        },
        "softplus_floor_stats": {
            "neg_rate_overall": softplus_raw["neg_rate_overall"],
            "floor_rate_overall": softplus_raw["floor_rate_overall"],
            "neg_rate_per_horizon_cell": softplus_raw["neg_rate_per_horizon_cell"],
            "floor_rate_per_horizon_cell": softplus_raw["floor_rate_per_horizon_cell"],
        },
        "parameter_norms": norm_results,
        "noise_probe": {
            "baseline": baseline_noise_probe,
            "softplus": softplus_noise_probe,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_serializable(results), indent=2))
    print(f"\nSaved full JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
