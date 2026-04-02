#!/usr/bin/env python
"""Probe 0: does the AR model actually use the condition vector?

Runs inference under three modes:
  - normal: standard conditioning
  - zero: replace condition vectors with zeros at every AR step
  - permute: use condition state from another window while keeping the same prev_frame

Reports S2/S7-style proxies with the same vol-of-vol regime split as the v2 test suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

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
CI_LO = 0.05
CI_HI = 0.95
REGIME_LO = 0.20
REGIME_HI = 0.80
LAYER2_LO = 0.70
LAYER2_HI = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("models/backfill/afcrps_164a_v3_percell_bptt_softplus/best_model.pt"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/vol_surface_with_ret.npz"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-start", type=int, default=4540)
    parser.add_argument("--n-windows", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/probe0_condition_ablation.json"),
    )
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
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
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
    end_exclusive = min(test_start + n_windows, max_start + 1)
    starts = np.arange(test_start, end_exclusive, dtype=np.int64)
    hist_offsets = np.arange(HISTORY_LEN, dtype=np.int64)
    fut_offsets = np.arange(FUTURE_LEN, dtype=np.int64)
    histories = surfaces[starts[:, None] + hist_offsets[None, :]]
    futures = surfaces[starts[:, None] + HISTORY_LEN + fut_offsets[None, :]]
    return {
        "starts": starts,
        "histories": histories,
        "futures": futures,
    }


def _cond_from_gru(model: ARSpatialTransformerModel, gru_outputs: torch.Tensor) -> torch.Tensor:
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = F.softmax(attn_logits, dim=1)
    h_pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    return model.encoder.bottleneck(h_pooled)


def sample_with_condition_mode(
    model: ARSpatialTransformerModel,
    histories: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    mode: str,
) -> np.ndarray:
    if mode not in {"normal", "zero", "permute"}:
        raise ValueError(f"Unknown mode: {mode}")

    set_seed(seed)
    samples_out: list[np.ndarray] = []
    chunk_size = 10
    permuted_histories = None
    if mode == "permute":
        perm = np.random.RandomState(seed + 999).permutation(len(histories))
        if len(perm) > 1:
            fixed = perm == np.arange(len(perm))
            if fixed.any():
                perm[fixed] = np.roll(perm, 1)[fixed]
        permuted_histories = histories[perm]

    for start in range(0, len(histories), batch_size):
        end = min(start + batch_size, len(histories))
        hist_batch = torch.from_numpy(histories[start:end]).to(device)
        hist_norm = normalize_iv(hist_batch)
        batch = hist_batch.shape[0]

        with torch.inference_mode():
            last_frame = hist_batch[:, -1].reshape(batch, N_CELLS)
            if mode == "permute":
                assert permuted_histories is not None
                cond_hist_batch = torch.from_numpy(permuted_histories[start:end]).to(device)
                cond_hist_norm = normalize_iv(cond_hist_batch)
            else:
                cond_hist_norm = hist_norm

            hist_flat_cond = cond_hist_norm.reshape(batch, HISTORY_LEN, N_CELLS)
            cond_gru_outputs_base, cond_h_last_base = model.encoder.gru(hist_flat_cond)

            batch_samples: list[torch.Tensor] = []
            for k0 in range(0, n_samples, chunk_size):
                k = min(chunk_size, n_samples - k0)
                prev = last_frame.unsqueeze(1).expand(batch, k, -1).reshape(batch * k, N_CELLS)
                gru_outputs = cond_gru_outputs_base.unsqueeze(1).expand(
                    batch, k, -1, -1
                ).reshape(batch * k, -1, model.encoder_config.gru_hidden_dim)
                gru_state = cond_h_last_base.unsqueeze(2).expand(
                    1, batch, k, -1
                ).reshape(1, batch * k, -1).contiguous()

                frames = []
                for _ in range(FUTURE_LEN):
                    z_t = torch.randn(batch * k, model.decoder.noise_dim, device=device)
                    if mode == "zero":
                        cond_t = torch.zeros(
                            batch * k,
                            model.encoder_config.bottleneck_dim,
                            device=device,
                            dtype=prev.dtype,
                        )
                    else:
                        cond_t = _cond_from_gru(model, gru_outputs)
                    delta = model.decoder(cond_t, prev, z_t)
                    frame_t = prev + torch.tanh(delta)
                    frames.append(frame_t)

                    frame_norm = normalize_iv(frame_t).unsqueeze(1)
                    gru_out, gru_state = model.encoder.gru(frame_norm, gru_state)
                    gru_outputs = torch.cat([gru_outputs, gru_out], dim=1)
                    prev = frame_t

                batch_samples.append(torch.stack(frames, dim=1).reshape(batch, k, FUTURE_LEN, GRID_H, GRID_W))

        samples_out.append(torch.cat(batch_samples, dim=1).cpu().numpy())

    return np.concatenate(samples_out, axis=0)


def summarize_mode(samples: np.ndarray, futures: np.ndarray, histories: np.ndarray) -> dict[str, Any]:
    lower = np.quantile(samples, CI_LO, axis=1)
    upper = np.quantile(samples, CI_HI, axis=1)
    median = np.median(samples, axis=1)
    mean_pred = samples.mean(axis=1)
    covered = (futures >= lower) & (futures <= upper)
    ci_width = upper - lower
    abs_err = np.abs(mean_pred - futures)
    mean_err = mean_pred - futures

    mean_iv = histories.mean(axis=(2, 3))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    q20 = float(np.quantile(vol_of_vol, REGIME_LO))
    q80 = float(np.quantile(vol_of_vol, REGIME_HI))
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80

    summary: dict[str, Any] = {
        "overall": {
            "mean_abs_error": float(abs_err.mean()),
            "mean_error": float(mean_err.mean()),
            "coverage_90": float(covered.mean()),
            "ci_width_mean": float(ci_width.mean()),
            "n_windows": int(len(histories)),
            "n_calm": int(calm_mask.sum()),
            "n_turb": int(turb_mask.sum()),
            "vov_q20": q20,
            "vov_q80": q80,
        },
        "path_bias": {},
        "horizons": {},
    }

    median_mean = median.mean(axis=(2, 3))
    gt_mean = futures.mean(axis=(2, 3))
    frac_below = (median_mean < gt_mean).mean(axis=1)
    for regime_name, regime_mask in (("calm", calm_mask), ("turb", turb_mask)):
        if int(regime_mask.sum()) == 0:
            continue
        regime_frac = frac_below[regime_mask]
        summary["path_bias"][regime_name] = {
            "persistent_low_pct": float((regime_frac > 0.80).mean()),
            "persistent_high_pct": float((regime_frac < 0.20).mean()),
            "mean_frac_median_below_gt": float(regime_frac.mean()),
        }

    for h in HORIZONS:
        idx = h - 1
        per_window_width = ci_width[:, idx].mean(axis=(1, 2))
        rho, pval = spearmanr(vol_of_vol, per_window_width)

        horizon_entry: dict[str, Any] = {
            "overall_coverage": float(covered[:, idx].mean()),
            "overall_mean_abs_error": float(abs_err[:, idx].mean()),
            "overall_mean_error": float(mean_err[:, idx].mean()),
            "overall_ci_width": float(ci_width[:, idx].mean()),
            "width_vov_spearman": float(rho),
            "width_vov_pval": float(pval),
        }

        for regime_name, regime_mask in (("calm", calm_mask), ("turb", turb_mask)):
            if int(regime_mask.sum()) == 0:
                continue
            regime_cov = covered[regime_mask, idx]
            regime_ci_width = ci_width[regime_mask, idx]
            regime_median = median[regime_mask, idx]
            regime_gt = futures[regime_mask, idx]
            cell_cov = regime_cov.mean(axis=0)
            cell_pass = (cell_cov >= LAYER2_LO) & (cell_cov <= LAYER2_HI)
            worst_idx = np.unravel_index(cell_cov.argmin(), cell_cov.shape)

            horizon_entry[regime_name] = {
                "coverage": float(regime_cov.mean()),
                "mean_ci_width": float(regime_ci_width.mean()),
                "median_above_gt_pct": float((regime_median > regime_gt).mean()),
                "gt_above_upper_pct": float((regime_gt > upper[regime_mask, idx]).mean()),
                "gt_below_lower_pct": float((regime_gt < lower[regime_mask, idx]).mean()),
                "worst_cell_coverage": float(cell_cov.min()),
                "worst_cell": [int(worst_idx[0]), int(worst_idx[1])],
                "n_cells_in_70_95_band": int(cell_pass.sum()),
            }

        if "calm" in horizon_entry and "turb" in horizon_entry:
            calm_w = horizon_entry["calm"]["mean_ci_width"]
            turb_w = horizon_entry["turb"]["mean_ci_width"]
            horizon_entry["turb_calm_width_ratio"] = turb_w / calm_w if calm_w > 0 else 1.0

        summary["horizons"][str(h)] = horizon_entry

    return summary


def print_summary(mode_results: dict[str, Any]) -> None:
    print("\n=== Probe 0: condition ablation ===")
    print("mode      mae     cov90   width   d1_tc   d10_tc  d30_tc  d30_calm_cov  d30_turb_cov")
    for mode, result in mode_results.items():
        overall = result["overall"]
        d1 = result["horizons"]["1"]
        d10 = result["horizons"]["10"]
        d30 = result["horizons"]["30"]
        print(
            f"{mode:8s} "
            f"{overall['mean_abs_error']:.4f}  "
            f"{overall['coverage_90']:.4f}  "
            f"{overall['ci_width_mean']:.4f}  "
            f"{d1.get('turb_calm_width_ratio', float('nan')):.3f}  "
            f"{d10.get('turb_calm_width_ratio', float('nan')):.3f}  "
            f"{d30.get('turb_calm_width_ratio', float('nan')):.3f}  "
            f"{d30['calm']['coverage']:.3f}  "
            f"{d30['turb']['coverage']:.3f}"
        )

    base = mode_results["normal"]
    print("\n=== Delta vs normal ===")
    print("mode      d_mae   d_cov90  d_width  d_d30_tc  d_d30_calm_cov  d_d30_turb_cov")
    for mode in ("zero", "permute"):
        result = mode_results[mode]
        print(
            f"{mode:8s} "
            f"{result['overall']['mean_abs_error'] - base['overall']['mean_abs_error']:+.4f}  "
            f"{result['overall']['coverage_90'] - base['overall']['coverage_90']:+.4f}  "
            f"{result['overall']['ci_width_mean'] - base['overall']['ci_width_mean']:+.4f}  "
            f"{result['horizons']['30']['turb_calm_width_ratio'] - base['horizons']['30']['turb_calm_width_ratio']:+.3f}  "
            f"{result['horizons']['30']['calm']['coverage'] - base['horizons']['30']['calm']['coverage']:+.3f}  "
            f"{result['horizons']['30']['turb']['coverage'] - base['horizons']['30']['turb']['coverage']:+.3f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"ckpt={args.ckpt}")
    print(f"device={device}")
    print(
        f"test_start={args.test_start} n_windows={args.n_windows} "
        f"n_samples={args.n_samples} batch_size={args.batch_size}"
    )

    model, checkpoint = build_model(args.ckpt, device)
    windows = load_windows(args.data_path, args.test_start, args.n_windows)
    histories = windows["histories"]
    futures = windows["futures"]

    print(
        f"loaded epoch={checkpoint.get('epoch')} "
        f"val_loss={checkpoint.get('val_loss'):.6f} "
        f"type={checkpoint['config'].get('type')}"
    )

    mode_results: dict[str, Any] = {}
    for offset, mode in enumerate(("normal", "zero", "permute")):
        print(f"\nRunning mode={mode} ...")
        samples = sample_with_condition_mode(
            model=model,
            histories=histories,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + offset,
            mode=mode,
        )
        mode_results[mode] = summarize_mode(samples, futures, histories)

    print_summary(mode_results)

    payload = {
        "setup": {
            "ckpt": args.ckpt,
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "type": checkpoint["config"].get("type"),
            "device": str(device),
            "test_start": args.test_start,
            "n_windows": args.n_windows,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "results": mode_results,
    }
    args.output_json.write_text(json.dumps(to_serializable(payload), indent=2))
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
